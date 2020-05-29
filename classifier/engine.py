import numpy as np
import torch
import tqdm

import utils


def loss_fn(start_logits, end_logits, clf_logits,
            start_positions, end_positions, clf_labels):
    m = torch.nn.LogSoftmax(dim=1)
    loss_fct = torch.nn.KLDivLoss(reduction='batchmean')
    start_loss = loss_fct(m(start_logits), start_positions)
    end_loss = loss_fct(m(end_logits), end_positions)

    bce_loss = torch.nn.BCEWithLogitsLoss()
    clf_loss = bce_loss(clf_logits, clf_labels)

    total_loss = (start_loss + end_loss) + clf_loss
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        start_labels = d['start_labels']
        end_labels = d['end_labels']
        clf_labels = d['clf_labels']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        start_labels = start_labels.to(device, dtype=torch.float)
        end_labels = end_labels.to(device, dtype=torch.float)
        clf_labels = clf_labels.to(device, dtype=torch.float)

        model.zero_grad()
        outputs_start, outputs_end, outputs_clf = \
            model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start, outputs_end, outputs_clf,
                       start_labels, end_labels, clf_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


def eval_fn(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            start_labels = d['start_labels']
            end_labels = d['end_labels']
            orig_start = d['orig_start']
            orig_end = d['orig_end']
            orig_selected = d['orig_selected']
            orig_tweet = d['orig_tweet']
            offsets = d['offsets']
            clf_labels = d['clf_labels']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_labels = start_labels.to(device, dtype=torch.float)
            end_labels = end_labels.to(device, dtype=torch.float)
            clf_labels = clf_labels.to(device, dtype=torch.float)

            outputs_start, outputs_end, outputs_clf = \
                model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs_start, outputs_end, outputs_clf,
                           start_labels, end_labels, clf_labels)

            outputs_start = outputs_start.cpu().detach().numpy()
            outputs_end = outputs_end.cpu().detach().numpy()
            outputs_clf = torch.sigmoid(outputs_clf).cpu().detach().numpy()

            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                jaccard_score, _ = \
                    utils.calculate_jaccard(original_tweet=tweet,
                                            target_string=selected_tweet,
                                            start_logits=outputs_start[px, :],
                                            end_logits=outputs_end[px, :],
                                            orig_start=orig_start[px],
                                            orig_end=orig_end[px],
                                            offsets=offsets[px],
                                            clf_score=outputs_clf[px])
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    print(f'Jaccard = {jaccards.avg}')

    return jaccards.avg

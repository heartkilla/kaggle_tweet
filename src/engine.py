import numpy as np
import torch
import tqdm

import utils


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = torch.nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    losses = utils.AverageMeter()
    jaccards = utils.AverageMeter()

    tk0 = tqdm.tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets_start = d['targets_start']
        targets_end = d['targets_end']
        sentiment = d['sentiment']
        orig_selected = d['orig_selected']
        orig_tweet = d['orig_tweet']
        offsets = d['offsets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(ids=ids, mask=mask,
                                           token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        loss.backward()
        optimizer.step()
        scheduler.step()

        outputs_start = torch.softmax(outputs_start,
                                      dim=1).cpu().detach().numpy()
        outputs_end = torch.softmax(outputs_end,
                                    dim=1).cpu().detach().numpy()
        jaccard_scores = []
        for px, tweet in enumerate(orig_tweet):
            selected_tweet = orig_selected[px]
            tweet_sentiment = sentiment[px]
            jaccard_score, _ = \
                utils.calculate_jaccard(original_tweet=tweet,
                                        target_string=selected_tweet,
                                        sentiment_val=tweet_sentiment,
                                        idx_start=np.argmax(
                                            outputs_start[px, :]),
                                        idx_end=np.argmax(outputs_end[px, :]),
                                        offsets=offsets[px])
            jaccard_scores.append(jaccard_score)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


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
            targets_start = d['targets_start']
            targets_end = d['targets_end']
            sentiment = d['sentiment']
            orig_selected = d['orig_selected']
            orig_tweet = d['orig_tweet']
            offsets = d['offsets']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(ids=ids, mask=mask,
                                               token_type_ids=token_type_ids)
            loss = loss_fn(outputs_start, outputs_end,
                           targets_start, targets_end)

            outputs_start = torch.softmax(outputs_start,
                                          dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end,
                                        dim=1).cpu().detach().numpy()

            jaccard_scores = []
            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                jaccard_score, _ = \
                    utils.calculate_jaccard(original_tweet=tweet,
                                            target_string=selected_tweet,
                                            sentiment_val=tweet_sentiment,
                                            idx_start=np.argmax(
                                                outputs_start[px, :]),
                                            idx_end=np.argmax(
                                                outputs_end[px, :]),
                                            offsets=offsets[px])
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

    print(f'Jaccard = {jaccards.avg}')

    return jaccards.avg

import torch
import numpy as np
import pandas as pd
import transformers
import tqdm.autonotebook as tqdm

import utils
import config
import models
import dataset


def eval_fn(data_loader, model, device):
    model.eval()
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

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            start_labels = start_labels.to(device, dtype=torch.float)
            end_labels = end_labels.to(device, dtype=torch.float)

            outputs_start, outputs_end = \
                model(ids=ids, mask=mask, token_type_ids=token_type_ids)

            outputs_start = outputs_start.cpu().detach().numpy()
            outputs_end = outputs_end.cpu().detach().numpy()

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
                                            offsets=offsets[px])
                jaccard_scores.append(jaccard_score)

            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            tk0.set_postfix(jaccard=jaccards.avg)

    return jaccards.avg


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    device = torch.device('cuda')
    model_config = transformers.RobertaConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    model = models.TweetModel(conf=model_config)
    model.to(device)
    model.load_state_dict(torch.load(
        f'{config.TRAINED_MODEL_PATH}/model_{fold}.bin'))
    model.eval()

    valid_dataset = dataset.TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=False)

    jaccard = eval_fn(valid_data_loader, model, device)

    return jaccard


if __name__ == '__main__':
    utils.seed_everything(config.SEED)

    fold_scores = []
    for i in range(config.N_FOLDS):
        fold_score = run(i)
        fold_scores.append(fold_score)

    for i in range(config.N_FOLDS):
        print(f'Fold={i}, Jaccard = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')

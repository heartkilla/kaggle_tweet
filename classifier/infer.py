import pandas as pd
import torch
import transformers
import tqdm

import config
import models
import dataset
import utils


def run():
    df_test = pd.read_csv(config.TEST_FILE)
    df_test.loc[:, 'selected_text'] = df_test.text.values

    device = torch.device('cuda')
    model_config = transformers.RobertaConfig.from_pretrained(
        config.MODEL_CONFIG)
    model_config.output_hidden_states = True

    fold_models = []
    for i in range(config.N_FOLDS):
        model = models.TweetModel(conf=model_config)
        model.to(device)
        model.load_state_dict(torch.load(
            f'{config.TRAINED_MODEL_PATH}/model_{i}.bin'))
        model.eval()
        fold_models.append(model)

    final_output = []

    test_dataset = dataset.TweetDataset(
        tweets=df_test.text.values,
        sentiments=df_test.sentiment.values,
        selected_texts=df_test.selected_text.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4)

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

            outputs_start_folds = []
            outputs_end_folds = []
            outputs_clf_folds = []
            for i in range(config.N_FOLDS):
                outputs_start, outputs_end, outputs_clf = \
                    fold_models[i](ids=ids,
                                   mask=mask,
                                   token_type_ids=token_type_ids)
                outputs_start_folds.append(outputs_start)
                outputs_end_folds.append(outputs_end)
                outputs_clf_folds.append(outputs_clf)

            outputs_start = sum(outputs_start_folds) / config.N_FOLDS
            outputs_end = sum(outputs_end_folds) / config.N_FOLDS
            outputs_clf = sum(outputs_clf_folds) / config.N_FOLDS
            outputs_clf = torch.sigmoid(outputs_clf).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                _, output_sentence = \
                    utils.calculate_jaccard(original_tweet=tweet,
                                            target_string=selected_tweet,
                                            start_logits=outputs_start[px, :],
                                            end_logits=outputs_end[px, :],
                                            orig_start=orig_start[px],
                                            orig_end=orig_end[px],
                                            offsets=offsets[px],
                                            clf_score=outputs_clf[px])
                final_output.append(output_sentence)

    sample = pd.read_csv(config.SUB_FILE)
    sample.loc[:, 'selected_text'] = final_output
    sample.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()

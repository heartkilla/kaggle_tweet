import pandas as pd
import numpy as np
import torch
import transformers
import tqdm

import config
import models
import dataset
import utils


def run():
    df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
    df_test.loc[:, 'selected_text'] = df_test.text.values

    device = torch.device('cuda')
    model_config = transformers.RobertaConfig.from_pretrained(
        config.ROBERTA_PATH)
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
        num_workers=1)

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

            outputs_start_folds = []
            outputs_end_folds = []
            for i in range(config.N_FOLDS):
                outputs_start, outputs_end = \
                    model(ids=ids,
                          mask=mask,
                          token_type_ids=token_type_ids)
                outputs_start_folds.append(outputs_start)
                outputs_end_folds.append(outputs_end)

            outputs_start = sum(outputs_start_folds) / config.N_FOLDS
            outputs_end = sum(outputs_end_folds) / config.N_FOLDS

            outputs_start = torch.softmax(outputs_start,
                                          dim=1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end,
                                        dim=1).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                selected_tweet = orig_selected[px]
                tweet_sentiment = sentiment[px]
                _, output_sentence = \
                    utils.calculate_jaccard(original_tweet=tweet,
                                            target_string=selected_tweet,
                                            sentiment_val=tweet_sentiment,
                                            idx_start=np.argmax(
                                                outputs_start[px, :]),
                                            idx_end=np.argmax(
                                                outputs_end[px, :]),
                                            offsets=offsets[px])
                final_output.append(output_sentence)

    sample = pd.read_csv(
        '/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
    sample.loc[:, 'selected_text'] = final_output
    sample.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    run()

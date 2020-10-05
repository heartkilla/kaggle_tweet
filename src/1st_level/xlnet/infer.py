import pickle

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
    model_config = transformers.XLNetConfig.from_pretrained(
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

    test_dataset = dataset.TweetDataset(
        tweets=df_test.text.values,
        sentiments=df_test.sentiment.values,
        selected_texts=df_test.selected_text.values)

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4)

    char_pred_test_start = []
    char_pred_test_end = []

    with torch.no_grad():
        tk0 = tqdm.tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            orig_tweet = d['orig_tweet']
            offsets = d['offsets']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs_start_folds = []
            outputs_end_folds = []
            for i in range(config.N_FOLDS):
                outputs_start, outputs_end = \
                    fold_models[i](ids=ids,
                                   mask=mask,
                                   token_type_ids=token_type_ids)
                outputs_start_folds.append(outputs_start)
                outputs_end_folds.append(outputs_end)

            outputs_start = sum(outputs_start_folds) / config.N_FOLDS
            outputs_end = sum(outputs_end_folds) / config.N_FOLDS

            outputs_start = torch.softmax(outputs_start, dim=-1).cpu().detach().numpy()
            outputs_end = torch.softmax(outputs_end, dim=-1).cpu().detach().numpy()

            for px, tweet in enumerate(orig_tweet):
                char_pred_test_start.append(
                    utils.token_level_to_char_level(tweet, offsets[px], outputs_start[px]))
                char_pred_test_end.append(
                    utils.token_level_to_char_level(tweet, offsets[px], outputs_end[px]))

    with open('../pickles/xlnet_char_pred_test_start.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_start, handle)
    with open('../pickles/xlnet_char_pred_test_end.pkl', 'wb') as handle:
        pickle.dump(char_pred_test_end, handle)


if __name__ == '__main__':
    run()

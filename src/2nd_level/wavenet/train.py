import numpy as np
import pandas as pd

import config
import utils
import dataset
import engine


def run():
    df_train = pd.read_csv(
        config.DATA_PATH + 'train.csv').dropna().reset_index(drop=True)
    df_train = df_train.sample(
        frac=1, random_state=config.TK_SEED).reset_index(drop=True)
    order_t = list(df_train['textID'].values)

    df_train = pd.read_csv(
        config.DATA_PATH + 'train.csv').dropna()
    df_train = df_train.sample(
        frac=1, random_state=config.HK_SEED).reset_index(drop=True)
    order_hk = list(df_train['textID'].values)

    df_test = pd.read_csv(config.DATA_PATH + 'test.csv').fillna('')
    df_test['selected_text'] = ''
    sub = pd.read_csv(config.DATA_PATH + 'sample_submission.csv')

    orders = {'theo': order_t,
              'hk': order_hk}

    (char_pred_oof_start, char_pred_oof_end,
     char_pred_test_start, char_pred_test_end) = utils.get_char_preds(
        orders, len(df_train), len(df_test))

    tokenizer = config.TOKENIZER
    tokenizer.fit_on_texts(df_train['text'].values)

    len_voc = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(df_train['text'].values)
    X_test = tokenizer.texts_to_sequences(df_test['text'].values)

    preds = {'test_start': np.array(char_pred_test_start),
             'test_end': np.array(char_pred_test_end),
             'oof_start': np.array(char_pred_oof_start),
             'oof_end': np.array(char_pred_oof_end)}

    pred_oof, pred_tests = engine.k_fold(df_train, df_test,
                                         np.array(X_train), np.array(X_test),
                                         preds, len_voc,
                                         k=config.N_FOLDS, model_seed=config.MODEL_SEED,
                                         fold_seed=config.TK_SEED,
                                         verbose=1, save=False, cp=False)

    test_dataset = dataset.TweetCharDataset(df_test, X_test,
                                            preds['test_start'],
                                            preds['test_end'],
                                            max_len=config.MAX_LEN,
                                            train=False,
                                            n_models=len(config.MODELS))
    train_dataset = dataset.TweetCharDataset(df_train, X_train,
                                             preds['test_start'],
                                             preds['test_end'],
                                             max_len=config.MAX_LEN,
                                             train=True,
                                             n_models=len(config.MODELS))

    selected_texts_oof = utils.string_from_preds_char_level(
        train_dataset, pred_oof,
        test=False, remove_neutral=config.REMOVE_NEUTRAL)

    scores = [utils.jaccard(pred, truth) for (pred, truth) in zip(
        selected_texts_oof, df_train['selected_text'])]
    score = np.mean(scores)
    print(f'Local CV score is {score:.4f}')

    selected_texts = utils.string_from_preds_char_level(
        test_dataset, pred_tests,
        test=True, remove_neutral=config.REMOVE_NEUTRAL)

    sub['selected_text'] = selected_texts
    df_test['selected_text'] = selected_texts
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()

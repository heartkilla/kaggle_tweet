import pickle

import numpy as np
import pandas as pd

import config
import engine
import utils
import dataset


def run():
    df_train = pd.read_csv(
        config.DATA_PATH + 'train.csv').dropna().reset_index(drop=True)

    df_test = pd.read_csv(config.DATA_PATH + 'test.csv').fillna('')
    df_test['selected_text'] = ''
    sub = pd.read_csv(config.DATA_PATH + 'sample_submission.csv')

    (char_pred_test_start,
     char_pred_test_end) = utils.get_test_char_preds(len(df_test))

    tokenizer = config.TOKENIZER
    tokenizer.fit_on_texts(df_train['text'].values)

    len_voc = len(tokenizer.word_index) + 1

    X_test = tokenizer.texts_to_sequences(df_test['text'].values)

    preds = {'test_start': np.array(char_pred_test_start),
             'test_end': np.array(char_pred_test_end)}

    pred_tests = engine.infer(df_test,
                              np.array(X_test),
                              preds, len_voc,
                              k=config.N_FOLDS)

    test_dataset = dataset.TweetCharDataset(df_test, X_test,
                                            preds['test_start'],
                                            preds['test_end'],
                                            max_len=config.MAX_LEN,
                                            train=False,
                                            n_models=len(config.MODELS))

    np.save("hk_rnn_preds.npy", np.array(pred_tests))

    selected_texts = utils.string_from_preds_char_level(
        test_dataset, pred_tests,
        test=True, remove_neutral=config.REMOVE_NEUTRAL)

    sub['selected_text'] = selected_texts
    df_test['selected_text'] = selected_texts
    sub.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    run()

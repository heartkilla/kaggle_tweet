import pandas as pd
import transformers
import torch
import torch_xla.core.xla_model as xm
import joblib

import config
import dataset
import models
import train_utils
import engine


def run(fold):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = dataset.TweetDataset(
        tweets=df_train.text.values,
        sentiments=df_train.sentiment.values,
        selected_texts=df_train.selected_text.values)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=1)

    valid_dataset = dataset.TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1)

    device = xm.xla_device(fold + 1)
    model_config = transformers.RobertaConfig.from_pretrained(
        config.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = models.TweetModel(conf=model_config)
    model = model.to(device)

    num_train_steps = int(
        len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]
    optimizer = transformers.AdamW(optimizer_parameters,
                                   lr=config.LEARNING_RATE)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps)

    es = train_utils.EarlyStopping(patience=2, mode='max')

    print(f'Training is starting for fold={fold}')

    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer,
                        device, scheduler=scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        es(jaccard, model,
           model_path=f'{config.MODEL_SAVE_PATH}/model_{fold}.bin')
        if es.early_stop:
            print('EarlyStopping')
            break


if __name__ == '__main__':
    joblib.Parallel(n_jobs=config.N_FOLDS, backend='threading')(
        joblib.delayed(run)(i) for i in range(config.N_FOLDS))

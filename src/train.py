import numpy as np
import pandas as pd
import transformers
import torch
import joblib

import config
import dataset
import models
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
        num_workers=4,
        shuffle=True)

    valid_dataset = dataset.TweetDataset(
        tweets=df_valid.text.values,
        sentiments=df_valid.sentiment.values,
        selected_texts=df_valid.selected_text.values)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
        shuffle=True)

    device = torch.device('cuda')
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
        optimizer=optimizer,
        num_warmup_steps=int(num_train_steps / config.EPOCHS),
        num_training_steps=num_train_steps)

    print(f'Training is starting for fold={fold}')

    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer,
                        device, scheduler=scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        torch.save(model.state_dict(),
                   f'{config.MODEL_SAVE_PATH}/model_{fold}.bin')

    return jaccard


if __name__ == '__main__':
    fold_scores = []
    for fold in range(config.N_FOLDS):
        fold_score = run(fold=fold)
        fold_scores.append(fold_score)

    for i in range(config.N_FOLDS):
        print(f'Fold={i}, Jaccard = {fold_scores[i]}')
    print(f'Mean = {np.mean(fold_scores)}')
    print(f'Std = {np.std(fold_scores)}')

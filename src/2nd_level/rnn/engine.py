import gc
import time
import datetime

import torch
import torchcontrib
import torch.nn.functional as F
import transformers
from sklearn.model_selection import StratifiedKFold

import config
import utils
import dataset
import models


def ce_loss(pred, truth, smoothing=False,
            trg_pad_idx=-1, eps=0.1):
    truth = truth.contiguous().view(-1)

    one_hot = torch.zeros_like(pred).scatter(1, truth.view(-1, 1), 1)

    if smoothing:
        n_class = pred.size(1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)

    loss = -one_hot * F.log_softmax(pred, dim=1)

    if trg_pad_idx >= 0:
        loss = loss.sum(dim=1)
        non_pad_mask = truth.ne(trg_pad_idx)
        loss = loss.masked_select(non_pad_mask)

    return loss.sum()


def loss_fn(start_logits, end_logits, start_positions, end_positions, config):
    bs = start_logits.size(0)

    start_loss = ce_loss(start_logits,
                         start_positions,
                         smoothing=config['smoothing'],
                         eps=config['eps'])

    end_loss = ce_loss(end_logits,
                       end_positions,
                       smoothing=config['smoothing'],
                       eps=config['eps'])

    total_loss = start_loss + end_loss

    return total_loss / bs


def predict(model, dataset, batch_size=32):
    model.eval()
    start_probas = []
    end_probas = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

    with torch.no_grad():
        for data in loader:
            start_logits, end_logits = model(data['ids'].cuda(),
                                             data['sentiment_input'].cuda(),
                                             data['probas_start'].cuda(),
                                             data['probas_end'].cuda())

            start_probs = torch.softmax(
                start_logits, dim=1).cpu().detach().numpy()
            end_probs = torch.softmax(
                end_logits, dim=1).cpu().detach().numpy()

            for s, e in zip(start_probs, end_probs):
                start_probas.append(list(s))
                end_probas.append(list(e))

    return start_probas, end_probas


def fit(model,
        train_dataset, val_dataset,
        loss_config,
        epochs=5,
        batch_size=8,
        acc_steps=1,
        weight_decay=0,
        warmup_prop=0.0,
        lr=5e-4,
        cp=False):

    best_jac = 0

    len_sampler = utils.LenMatchBatchSampler(
        torch.utils.data.RandomSampler(train_dataset),
        batch_size=batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_sampler=len_sampler, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # betas=(0.5, 0.999))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torchcontrib.optim.SWA(optimizer)

    swa_first_epoch = 5

    n_steps = float(epochs * len(train_loader)) / float(acc_steps)
    num_warmup_steps = int(warmup_prop * n_steps)

    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, n_steps)

    total_steps = 0
    for epoch in range(epochs):
        model.train()
        start_time = time.time()

        optimizer.zero_grad()
        avg_loss = 0

        for step, data in enumerate(train_loader):
            total_steps += 1

            start_logits, end_logits = model(data['ids'].cuda(),
                                             data['sentiment_input'].cuda(),
                                             data['probas_start'].cuda(),
                                             data['probas_end'].cuda())

            loss = loss_fn(start_logits,
                           end_logits,
                           data['target_start'].cuda(),
                           data['target_end'].cuda(),
                           config=loss_config)

            avg_loss += loss.item() / len(train_loader)
            loss.backward()

            if (step + 1) % acc_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
        model.eval()
        avg_val_loss = 0.0
        val_jac = 0.0

        if epoch >= swa_first_epoch:
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        with torch.no_grad():
            for data in val_loader:
                start_logits, end_logits = model(
                    data["ids"].cuda(),
                    data['sentiment_input'].cuda(),
                    data['probas_start'].cuda(),
                    data['probas_end'].cuda())

                loss = loss_fn(start_logits.detach(),
                               end_logits.detach(),
                               data["target_start"].cuda().detach(),
                               data["target_end"].cuda().detach(),
                               config=loss_config)

                avg_val_loss += loss.item() / len(val_loader)

                val_jac += utils.jaccard_from_logits_string(
                    data, start_logits, end_logits) / len(val_dataset)

        if epoch >= swa_first_epoch:
            optimizer.swap_swa_sgd()

        if val_jac >= best_jac and cp:
            utils.save_model_weights(model, 'checkpoint.pt', verbose=0)
            best_jac = val_jac

        dt = time.time() - start_time
        lr = scheduler.get_lr()[0]
        print(f'Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={dt:.0f}s \t',
              end='')
        print(f'loss={avg_loss:.3f} \t val_loss={avg_val_loss:.3f} \t val_jaccard={val_jac:.4f}')

    del loss, data, avg_val_loss, avg_loss, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    if epoch >= swa_first_epoch:
        optimizer.swap_swa_sgd()

    return best_jac if cp else val_jac


def k_fold(df_train, df_test,
           X_train, X_test, preds,
           len_voc, k=5,
           fold_seed=42, model_seed=42, verbose=1,
           save=True, cp=False):
    time = str(datetime.datetime.now())[:16]
    score = 0
    splits = list(StratifiedKFold(n_splits=k, random_state=fold_seed).split(
        X=df_train, y=df_train['sentiment']))

    pred_oof = [[[], []] for i in range(len(df_train))]
    pred_tests = []

    test_dataset = dataset.TweetCharDataset(df_test, X_test,
                                            preds['test_start'],
                                            preds['test_end'],
                                            max_len=config.MAX_LEN,
                                            train=False,
                                            n_models=len(config.MODELS))

    for i, (train_idx, val_idx) in enumerate(splits):
        print(f"\n-------------   Fold {i + 1}  -------------")
        utils.seed_everything(model_seed)

        model = models.TweetCharModel(
            len_voc,
            use_msd=config.USE_MSD,
            n_models=len(config.MODELS),
            lstm_dim=config.LSTM_DIM,
            ft_lstm_dim=config.FT_LSTM_DIM,
            char_embed_dim=config.CHAR_EMBED_DIM,
            sent_embed_dim=config.SENT_EMBED_DIM).cuda()
        model.zero_grad()

        train_dataset = dataset.TweetCharDataset(df_train.iloc[train_idx],
                                                 X_train[train_idx],
                                                 preds['oof_start'][train_idx],
                                                 preds['oof_end'][train_idx],
                                                 max_len=config.MAX_LEN,
                                                 n_models=len(config.MODELS))

        val_dataset = dataset.TweetCharDataset(df_train.iloc[val_idx],
                                               X_train[val_idx],
                                               preds['oof_start'][val_idx],
                                               preds['oof_end'][val_idx],
                                               max_len=config.MAX_LEN,
                                               n_models=len(config.MODELS))

        print('\n- Training all layers: ')
        utils.unfreeze(model)
        n_parameters = utils.count_parameters(model)
        print(f'    -> {n_parameters} trainable parameters\n')

        fold_score = fit(model,
                         train_dataset,
                         val_dataset,
                         config.loss_config,
                         epochs=config.EPOCHS,
                         batch_size=config.TRAIN_BATCH_SIZE,
                         lr=config.LR,
                         warmup_prop=config.WAMUP_PROP,
                         cp=cp)

        score += fold_score / k

        print('\n- Predicting ')

        pred_val_start, pred_val_end = predict(
            model, val_dataset, batch_size=config.VALID_BATCH_SIZE)
        for j, idx in enumerate(val_idx):
            pred_oof[idx] = [pred_val_start[j], pred_val_end[j]]

        pred_test = predict(
            model, test_dataset, batch_size=config.VALID_BATCH_SIZE)
        pred_tests.append(pred_test)

        if cp:
            utils.load_model_weights(model, "checkpoint.pt", verbose=0)
        if save:
            utils.save_model_weights(
                model,
                f'{config.selected_model}_{time}_{i + 1}.pt',
                cp_folder=config.CP_PATH)

        del model, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    print(f'\n Local CV jaccard is {score:.4f}')
    return pred_oof, pred_tests

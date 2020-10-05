import os
import random
import pickle

import numpy as np
import torch

import config


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model_weights(model, filename, verbose=1, cp_folder=config.CP_PATH):
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=config.CP_PATH):
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )

    if verbose:
        print(
            f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")

    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def jaccard(str1, str2):
    """Original metric implementation."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_from_logits_string(data, start_logits, end_logits):
    n = start_logits.size(0)
    score = 0

    start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
    end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

    for i in range(n):
        start_idx = np.argmax(start_logits[i])
        end_idx = np.argmax(end_logits[i])
        text = data["text"][i]
        pred = text[start_idx: end_idx]

        score += jaccard(data["selected_text"][i], pred)

    return score


def reorder(order_source, order_target, preds):
    order_source = list(order_source)
    new_preds = []
    for tgt_idx in order_target:
        new_idx = order_source.index(tgt_idx)
        new_preds.append(preds[new_idx])

    return new_preds


def get_char_preds(orders, len_train, len_test):
    char_pred_oof_starts = []
    char_pred_oof_ends = []
    char_pred_test_starts = []
    char_pred_test_ends = []

    n_models = len(config.MODELS)

    for model, author in config.MODELS:
        with open(config.PKL_PATH + model + 'char_pred_oof_start.pkl', "rb") as fp:
            probas = pickle.load(fp)

            if author != 'hk':
                probas = reorder(orders[author], orders['hk'], probas)

            if model in config.ADD_SPACE_TO:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_oof_starts.append(probas)

        with open(config.PKL_PATH + model + 'char_pred_oof_end.pkl', "rb") as fp:
            probas = pickle.load(fp)

            if model in config.ADD_SPACE_TO:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            if author != 'hk':
                probas = reorder(orders[author], orders['hk'], probas)

            char_pred_oof_ends.append(probas)

        with open(config.PKL_PATH + model + 'char_pred_test_start.pkl', "rb") as fp:
            probas = pickle.load(fp)

            if model in config.ADD_SPACE_TO:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_starts.append(probas)

        with open(config.PKL_PATH + model + 'char_pred_test_end.pkl', "rb") as fp:
            probas = pickle.load(fp)

            if model in config.ADD_SPACE_TO:
                probas = [np.concatenate([np.array([0]), p]) for p in probas]

            char_pred_test_ends.append(probas)

    char_pred_oof_start = [np.concatenate(
        [char_pred_oof_starts[m][i][:, np.newaxis]for m in range(n_models)],
        1) for i in range(len_train)]
    char_pred_oof_end = [np.concatenate(
        [char_pred_oof_ends[m][i][:, np.newaxis] for m in range(n_models)],
        1) for i in range(len_train)]
    char_pred_test_start = [np.concatenate(
        [char_pred_test_starts[m][i][:, np.newaxis] for m in range(n_models)],
        1) for i in range(len_test)]
    char_pred_test_end = [np.concatenate(
        [char_pred_test_ends[m][i][:, np.newaxis] for m in range(n_models)],
        1) for i in range(len_test)]

    return (char_pred_oof_start, char_pred_oof_end,
            char_pred_test_start, char_pred_test_end)


def string_from_preds_char_level(dataset, preds, test=False,
                                 remove_neutral=False, uncensored=False,
                                 cleaned=False):
    selected_texts = []
    n_models = len(preds)

    for idx in range(len(dataset)):
        data = dataset[idx]

        if test:
            start_probas = np.mean(
                [preds[i][0][idx] for i in range(n_models)], 0)
            end_probas = np.mean(
                [preds[i][1][idx] for i in range(n_models)], 0)
        else:
            start_probas = preds[idx][0]
            end_probas = preds[idx][1]

        start_idx = np.argmax(start_probas)
        end_idx = np.argmax(end_probas)

        if end_idx < start_idx:
            selected_text = data["text"]
        elif remove_neutral and data["sentiment"] == "neutral":
            selected_text = data["text"]
        else:
            selected_text = data["text"][start_idx: end_idx]

        selected_texts.append(selected_text.strip())

    return selected_texts


class LenMatchBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    Custom PyTorch Sampler that generate batches of similar length.
    Used alongside with trim_tensor, it helps speed up training.
    """

    def __iter__(self):

        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx]['ids'] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:
                buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

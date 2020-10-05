import os
import random

import torch
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def token_level_to_char_level(text, offsets, preds):
    probas_char = np.zeros(len(text))
    for i, offset in enumerate(offsets):
        if offset[0] or offset[1]:
            probas_char[offset[0]:offset[1]] = preds[i]

    return probas_char


def jaccard(str1, str2):
    """Original metric implementation."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_best_start_end_idx(start_logits, end_logits,
                           orig_start, orig_end):
    """Return best start and end indices following BERT paper."""
    best_logit = -np.inf
    best_idxs = None
    start_logits = start_logits[orig_start:orig_end + 1]
    end_logits = end_logits[orig_start:orig_end + 1]
    for start_idx, start_logit in enumerate(start_logits):
        for end_idx, end_logit in enumerate(end_logits[start_idx:]):
            logit_sum = start_logit + end_logit
            if logit_sum > best_logit:
                best_logit = logit_sum
                best_idxs = (orig_start + start_idx,
                             orig_start + start_idx + end_idx)
    return best_idxs


def calculate_jaccard(original_tweet, target_string,
                      start_logits, end_logits,
                      orig_start, orig_end,
                      offsets, 
                      verbose=False):
    """Calculates final Jaccard score using predictions."""
    start_idx, end_idx = get_best_start_end_idx(
        start_logits, end_logits, orig_start, orig_end)

    filtered_output = ''
    for ix in range(start_idx, end_idx + 1):
        filtered_output += original_tweet[offsets[ix][0]:offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += ' '

    # Return orig tweet if it has less then 2 words
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

    if len(filtered_output.split()) == 1:
        filtered_output = filtered_output.replace('!!!!', '!')
        filtered_output = filtered_output.replace('..', '.')
        filtered_output = filtered_output.replace('...', '.')

    filtered_output = filtered_output.replace('ïï', 'ï')
    filtered_output = filtered_output.replace('¿¿', '¿')

    jac = jaccard(target_string.strip(), filtered_output.strip())
    return jac, filtered_output


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

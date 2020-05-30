import numpy as np
import torch

import config


def jaccard_array(a, b):
    """Calculates Jaccard on arrays."""
    a = set(a)
    b = set(b)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def process_data(tweet, selected_text, sentiment,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
    tweet = ' '.join(str(tweet).split())
    selected_text = ' '.join(str(selected_text).split())

    len_sel_text = len(selected_text)

    # Get sel_text start and end idx
    idx_0 = None
    idx_1 = None
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind:ind + len_sel_text] == selected_text:
            idx_0 = ind
            idx_1 = ind + len_sel_text - 1
            break

    # Assign 1 as target for each char in sel_text
    char_targets = [0] * len(tweet)
    if idx_0 is not None and idx_1 is not None:
        for ct in range(idx_0, idx_1 + 1):
            char_targets[ct] = 1

    tokenized_tweet = tokenizer.encode(tweet)
    # Vocab ids
    input_ids_original = tokenized_tweet.ids[1:-1]
    # Start and end char
    tweet_offsets = tokenized_tweet.offsets[1:-1]

    # Get ids within tweet of words that have target char
    target_ids = []
    for i, (offset_0, offset_1) in enumerate(tweet_offsets):
        if sum(char_targets[offset_0:offset_1]) > 0:
            target_ids.append(i)

    targets_start = target_ids[0]
    targets_end = target_ids[-1]

    # Sentiment 'word' id in vocab
    sentiment_id = {'positive': 3893,
                    'negative': 4997,
                    'neutral': 8699}

    # Soft Jaccard labels
    # ----------------------------------
    n = len(input_ids_original)
    sentence = np.arange(n)
    answer = sentence[targets_start:targets_end + 1]

    start_labels = np.zeros(n)
    for i in range(targets_end + 1):
        jac = jaccard_array(answer, sentence[i:targets_end + 1])
        start_labels[i] = jac + jac ** 2
    start_labels = (1 - config.SOFT_ALPHA) * start_labels / start_labels.sum()
    start_labels[targets_start] += config.SOFT_ALPHA

    end_labels = np.zeros(n)
    for i in range(targets_start, n):
        jac = jaccard_array(answer, sentence[targets_start:i + 1])
        end_labels[i] = jac + jac ** 2
    end_labels = (1 - config.SOFT_ALPHA) * end_labels / end_labels.sum()
    end_labels[targets_end] += config.SOFT_ALPHA

    start_labels = [0, 0, 0] + list(start_labels) + [0]
    end_labels = [0, 0, 0] + list(end_labels) + [0]
    # ----------------------------------

    # Input for ELECTRA
    input_ids = [101] + [sentiment_id[sentiment]] + [102] + \
                input_ids_original + [102]
    # No token types in ELECTRA
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_original) + 1)
    # Mask of input without padding
    mask = [1] * len(token_type_ids)
    # Start and end char ids for each word including new tokens
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    # Ids within tweet of words that have target char including new tokens
    targets_start += 3
    targets_end += 3
    orig_start = 3
    orig_end = len(input_ids_original) + 2

    # Input padding: new mask, token type ids, tweet offsets
    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_len)
        start_labels = start_labels + ([0] * padding_len)
        end_labels = end_labels + ([0] * padding_len)

    targets_select = [0] * len(token_type_ids)
    for i in range(len(targets_select)):
        if i in target_ids:
            targets_select[i + 4] = 1

    return {'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'start_labels': start_labels,
            'end_labels': end_labels,
            'orig_start': orig_start,
            'orig_end': orig_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets,
            'targets_select': targets_select}


class TweetDataset:
    def __init__(self, tweets, sentiments, selected_texts):
        self.tweets = tweets
        self.sentiments = sentiments
        self.selected_texts = selected_texts
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        """Returns preprocessed data sample as dict with
        data converted to tensors.
        """
        data = process_data(self.tweets[item],
                            self.selected_texts[item],
                            self.sentiments[item],
                            self.tokenizer,
                            self.max_len)

        return {'ids': torch.tensor(data['ids'], dtype=torch.long),
                'mask': torch.tensor(data['mask'], dtype=torch.long),
                'token_type_ids': torch.tensor(data['token_type_ids'],
                                               dtype=torch.long),
                'start_labels': torch.tensor(data['start_labels'],
                                             dtype=torch.float),
                'end_labels': torch.tensor(data['end_labels'],
                                           dtype=torch.float),
                'orig_start': data['orig_start'],
                'orig_end': data['orig_end'],
                'orig_tweet': data['orig_tweet'],
                'orig_selected': data['orig_selected'],
                'sentiment': data['sentiment'],
                'offsets': torch.tensor(data['offsets'], dtype=torch.long),
                'targets_select': torch.tensor(data['targets_select'],
                                               dtype=torch.float)}

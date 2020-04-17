import torch

import config


def process_data(tweet, selected_text, sentiment,
                 tokenizer, max_len):
    """Preprocesses one data sample and returns a dict
    with targets and other useful info.
    """
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
    # Vocab ids excluding 'CLS' and 'SEP'
    input_ids_original = tokenized_tweet.ids[1:-1]
    # Start and end char ids for each word
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

    # input = 'CLS' + 'sentiment' + 'SEP' + 'tweet' + 'SEP'
    input_ids = [101] + [sentiment_id[sentiment]] + \
                [102] + input_ids_original + [102]
    # Assing token type 1 to 'tweet' and last 'SEP'
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_original) + 1)
    # Mask of input without padding
    mask = [1] * len(token_type_ids)
    # Start and end char ids for each word including new tokens
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    # Ids within tweet of words that have target char including new tokens
    targets_start += 3
    targets_end += 3

    # Input padding: new mask, token type ids, tweet offsets
    padding_len = max_len - len(input_ids)
    if padding_len > 0:
        input_ids = input_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_len)

    return {'ids': input_ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets_start': targets_start,
            'targets_end': targets_end,
            'orig_tweet': tweet,
            'orig_selected': selected_text,
            'sentiment': sentiment,
            'offsets': tweet_offsets}


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
        integers converted to tensors.
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
                'targets_start': torch.tensor(data['targets_start'],
                                              dtype=torch.long),
                'targets_end': torch.tensor(data['targets_end'],
                                            dtype=torch.long),
                'orig_tweet': data['orig_tweet'],
                'orig_selected': data['orig_selected'],
                'sentiment': data['sentiment'],
                'offsets': torch.tensor(data['offsets'], dtype=torch.long)}

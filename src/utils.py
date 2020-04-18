def jaccard(str1, str2):
    """Original metric implementation."""
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard(original_tweet, target_string, sentiment_val,
                      idx_start, idx_end, offsets, verbose=False):
    """Calculates final Jaccard score using predictions."""
    if idx_end < idx_start:
        idx_end = idx_start

    filtered_output = ' '
    for ix in range(idx_start, idx_end + 1):
        filtered_output += original_tweet[offsets[ix][0]:offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            filtered_output += ' '

    # Return orig tweet if it has less then 2 words
    if len(original_tweet.split()) < 2:
        filtered_output = original_tweet

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

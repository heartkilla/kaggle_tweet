import tokenizers


# Paths
TOKENIZER_PATH = '../roberta_tokenizer'
TRAINING_FILE = '../data/train_folds.csv'
TEST_FILE = '../data/test.csv'
SUB_FILE = '../data/sample_submission.csv'
MODEL_SAVE_PATH = './model_save'
TRAINED_MODEL_PATH = './model_save'

# Model config
MODEL_CONFIG = 'distilroberta-base'

# Model params
SEED = 718
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 5e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 96  # actually = 86
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f'{TOKENIZER_PATH}/vocab.json',
    merges_file=f'{TOKENIZER_PATH}/merges.txt',
    lowercase=True,
    add_prefix_space=True)
HIDDEN_SIZE = 768
N_LAST_HIDDEN = 6
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 0.4
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30

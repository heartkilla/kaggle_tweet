import tokenizers


# Paths
TOKENIZER_PATH = '/content/sample_data/electra-tokenizer'
MODEL_PATH = 'pytorch_model.bin'
TRAINING_FILE = '/content/sample_data/twitter-train-folds/train_folds.csv'
TEST_FILE = '/content/test.csv'
SUB_FILE = '/content/sample_submission.csv'
MODEL_SAVE_PATH = '/content/sample_data/model-save'
TRAINED_MODEL_PATH = '/content/electra-discriminator-save/model-save'

# Model config
MODEL_CONFIG = 'google/electra-base-discriminator'

# Model params
SEED = 25
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
MAX_LEN = 128  # actually = 86
TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f'{TOKENIZER_PATH}/vocab.txt', 
        lowercase=True)
HIDDEN_SIZE = 768
N_LAST_HIDDEN = 12
HIGH_DROPOUT = 0.5
SOFT_ALPHA = 0.6
WARMUP_RATIO = 0.25
WEIGHT_DECAY = 0.001
USE_SWA = False
SWA_RATIO = 0.9
SWA_FREQ = 30
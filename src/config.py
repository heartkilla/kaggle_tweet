import tokenizers


# Paths
BERT_PATH = '/kaggle/input/bert-base-uncased'
ROBERTA_PATH = '/kaggle/input/roberta-base'
MODEL_PATH = 'pytorch_model.bin'
TRAINING_FILE = '/kaggle/input/twitter-train-folds/train_folds.csv'
MODEL_SAVE_PATH = '/kaggle/working'
TRAINED_MODEL_PATH = '/kaggle/input/roberta-baseline-train-tpu'

# Model params
N_FOLDS = 5
EPOCHS = 4
LEARNING_RATE = 4e-5
PATIENCE = None
EARLY_STOPPING_DELTA = None
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
MAX_LEN = 96  # actually = 86
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f'{ROBERTA_PATH}/vocab.json',
    merges_file=f'{ROBERTA_PATH}/merges.txt',
    lowercase=True,
    add_prefix_space=True)
N_LAST_HIDDEN = 12
LAST_DROPOUT = 0.5

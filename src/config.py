import tokenizers


# Paths
BERT_PATH = '/kaggle/input/bert-base-uncased'
ROBERTA_PATH = '/kaggle/input/roberta-base'
MODEL_PATH = 'pytorch_model.bin'
TRAINING_FILE = '/kaggle/input/twitter-train-folds/train_folds.csv'
MODEL_SAVE_PATH = '/kaggle/working'
TRAINED_MODEL_PATH = '/kaggle/input/roberta-baseline-train-tpu'

# Model params
N_FOLDS = 8
EPOCHS = 5
LEARNING_RATE = 4e-5
TRAIN_BATCH_SIZE = 50
VALID_BATCH_SIZE = 32
MAX_LEN = 128  # need to check
TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file=f'{ROBERTA_PATH}/vocab.json',
    merges_file=f'{ROBERTA_PATH}/merges.txt',
    lowercase=True,
    add_prefix_space=True)

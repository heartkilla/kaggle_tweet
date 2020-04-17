import tokenizers

MAX_LEN = 128  # need to check
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 16
EPOCHS = 3
BERT_PATH = '../input/bert-base-uncased/'
MODEL_PATH = 'model.bin'
TRAINING_FILE = '../input/tweet-train-folds/train_folds.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    f'{BERT_PATH}/vocab.txt',
    lowercase=True
)
N_FOLDS = 5

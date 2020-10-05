from keras.preprocessing.text import Tokenizer

# Paths
DATA_PATH = '/content/very_final/data/'
PKL_PATH = '/content/very_final/pickles/'
CP_PATH = '/content/very_final/2nd_level_cnn/model_save/'

# Fold seeds
TK_SEED = 2020
HK_SEED = 50898

# 1st level models
MODELS = [('roberta-', 'hk'),
          ('distil_', 'hk'),
          ('large_', 'hk'),
          ('xlnet_', 'hk')]
ADD_SPACE_TO = ['xlnet_']

# Model params
MODEL_SEED = 25
N_FOLDS = 5
EPOCHS = 5
LR = 6e-3
WAMUP_PROP = 0.0
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 512
TOKENIZER = Tokenizer(num_words=None, char_level=True,
                      oov_token='UNK', lower=True)
MAX_LEN = 150
SENT_EMBED_DIM = 16
CHAR_EMBED_DIM = 16
PROBA_CNN_DIM = 16
KERNEL_SIZE = 3
CNN_DIM = 32
USE_BN = True
USE_MSD = True

# Loss
loss_config = {'smoothing': True,
               'eps': 0.2}

# Postprocessing
REMOVE_NEUTRAL = False

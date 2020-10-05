# [Kaggle Tweet Sentiment Extraction Competition](https://www.kaggle.com/c/tweet-sentiment-extraction/leaderboard): 1st place solution (Dark of the Moon team) 

This repository contains the models that I implemented for this competition as a part of our team.

## First level models
### Heartkilla (me)

- Models: RoBERTa-base-squad2, RoBERTa-large-squad2, DistilRoBERTa-base, XLNet-base-cased
- Concat Avg / Max of last n-1 layers (without embedding layer) and feed into Linear head 
- Multi Sample Dropout, AdamW, linear warmup schedule
- I used Colab Pro for training.
- Custom loss: Jaccard-based Soft Labels
Since Cross Entropy doesn’t optimize Jaccard directly, I tried different loss functions to penalize far predictions more than close ones. SoftIOU used in segmentation didn’t help so I came up with a custom loss that modifies usual label smoothing by computing Jaccard on the token level. I then use this new target labels and optimize KL divergence. Alpha here is a parameter to balance between usual CE and Jaccard-based labeling.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F9341bede28263bcf0e9bb259ac790338%2FScreen%20Shot%202020-05-30%20at%2017.31.22.png?generation=1592405028556842&amp;alt=media)
I’ve noticed that probabilities in this case change pretty steeply so I decided to smooth it a bit by adding a square term.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F4d1975e293c33c077fd45e6b81d1aa63%2FScreen%20Shot%202020-05-30%20at%2017.29.17.png?generation=1592405079812280&amp;alt=media)
This worked best for 3 of my models except DistilRoBERTa which used the previous without-square version. Eventually this loss boosted all of my models by around 0.003.
This is a plot of target probabilities for 30 tokens long sentence with start\_idx=5 and end\_idx=25, alpha=0.3.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2Fd746070e62bc05d74f7543785da6df70%2Fplot.jpg?generation=1592405194100691&amp;alt=media)

I claim that since the probabilities from my models are quite decorrelated with regular CE / SmoothedCE ones, they provided necessary diversity and were crucial to each of our 2nd level models.


### Hikkiiii

- max\_len=120, no post-processing
- Append sentiment token to the end of the text
- Models: 5fold-roberta-base-squad2(0.712CV), 5fold-roberta-large-squad2(0.714CV)
- Last 3 hidden states + CNN*1 + linear
- CrossEntropyLoss, AdamW
- epoch=5, lr=3e-5, weight_decay=0.001, no scheduler, warmup=0, bsz=32-per-device
- V100*2, apex(O1) for fast training
- Traverse the top 20 of start\_index and end\_index, ensure start\_index &lt; end\_index

### Theo
I took a bet when I joined @cl2ev1 on the competition, which was that working with Bert models (although they perform worse than Roberta) will help in the long run. It did pay off, as our 2nd level models reached 0.735 public using 2 Bert (base, wwm) and 3 Roberta (base, large, distil). I then trained an Albert-large and a Distilbert for diversity.

- bert-base-uncased (CV 0.710), bert-large-uncased-wwm  (CV 0.710), distilbert (CV 0.705), albert-large-v2  (CV 0.711)
- Squad pretrained weights
- Multi Sample Dropout on the concatenation of the last `n` hidden states
- Simple smoothed categorical cross-entropy on the start and end probabilities
- I use the auxiliary sentiment from the original dataset as an additional input for training. 
``` [CLS] [sentiment] [aux sentiment] [SEP] ...```
During inference, it is set to neutral
- 2 epochs, lr = 7e-5 except for distilbert (3 epochs, lr = 5e-5)
- Sequence bucketing, batch size is the highest power of 2 that could fit on my 2080Ti (128 (distil) / 64 (bert-base) / 32 (albert) / 16 (wwm)) with `max_len = 70`
- Bert models have their learning rate decayed closer to the input, and use a higher learning rate for the head (1e-4)
- Sequence bucketting for faster training 

### Cl_ev
This competition has a lengthy list of things that did not work, here are things that  worked :)


- Models: roberta-base (CV 0.715), Bertweet (thanks to all that shared it - it helped diversity)
- MSD, applying to hidden outputs
- (roberta) pretrained on squad
- (roberta) custom merges.txt  (helps with cases when tokenization would not allow to predict correct start and finish). On it’s own adds about  0.003 - 0.0035 to CV.
- Discriminative learning
- Smoothed CE (in some cases weighted CE performed ok, but was dropped)


## Second level models

###  Architectures
Theo came up with 3 different Char-NN architectures that use character-level probabilities from transformers as input. You can see how we utilize them in [this notebook](https://www.kaggle.com/theoviel/character-level-model-magic).
- RNN
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F4a68e14abf95a157c299f6489c90a6f9%2FML%20Visuals%20by%20dair.ai%20(4).svg?generation=1592405862305380&amp;alt=media)

- CNN
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F440886c9845034b633303bbe4a785cc9%2FML%20Visuals%20by%20dair.ai%20(5).svg?generation=1592405917697990&amp;alt=media)

- WaveNet (yes, we took that one from the Liverpool competition)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F58a112fdd33e549076c61ed78a4d9e93%2FML%20Visuals%20by%20dair.ai%20(6).svg?generation=1592405961585964&amp;alt=media)

### Stacking ensemble
As Theo mentioned [here](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159264), we feed character level probabilities from transformers into Char-NNs.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F6ac3e81a02676611b21b18e0340cb23d%2Fpipe.png?generation=1592409782895291&amp;alt=media)
 
However, we decided not to just do it end-to-end (i.e. training 2nd levels on the training data probas), but to use OOF predictions and perform good old stacking. As our team name suggests (one of the Transformers movies) we built quite an army of transformers. This is the stacking pipeline for our 2 submissions. Note that we used different input combinations to 2nd level models for diversity. Inference is also available in [this](https://www.kaggle.com/aruchomu/no-sampler-ensemble-normal-sub-0-7363) and [this](https://www.kaggle.com/aruchomu/no-sampler-ensemble-normal-sub-0-7365) kernels.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F4a8e506d88ca5783cbb2d1eb1cffcbd7%2FML%20Visuals%20by%20dair.ai%20(7).svg?generation=1592406106435760&amp;alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2000545%2F9d77d0a59ae83dc10cc97d2279bdafb4%2FML%20Visuals%20by%20dair.ai%20(8).svg?generation=1592406151678743&amp;alt=media)

### Pseudo-labeling
We used one of our CV 0.7354 blends to pseudo-label the public test data. We followed the approach from [here](https://www.kaggle.com/c/google-quest-challenge/discussion/129840) and created “leakless” pseudo-labels. We then used a threshold of 0.35 to cut off low-confidence samples. The confidence score was determined like: `(start_probas.max() + end_probas.max()) / 2`. This gave a pretty robust boost of 0.001-0.002 for many models. We’re not sure if it really helps the final score overall since we only did 9 submissions with the full inference.

###  Other details
Adam optimizer, linear decay schedule with no warmup, SmoothedCELoss such as in level 1 models, Multi Sample Dropout. Some of the models also used Stochastic Weighted Average.

## Extra stuff

We did predictions on neutral texts as well, our models were slightly better than doing `selected_text = text`. However, we do `selected_text = text` when `start_idx > end_idx`.

Once [the pattern](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159254) in the labels is detected, it is possible to clean the labels to improve level 1 models performance. Since we found the pattern a bit too late, we decided to stick with the ensembles we already built instead of retraining everything from scratch.
 

Thanks for reading and happy kaggling!

## [Update]
I gave a speech about our solution at the ODS Paris meetup: [YouTube link](https://www.youtube.com/watch?v=S7soN-y5WMg)

The presentation: [SlideShare link](https://www.slideshare.net/ArtsemZhyvalkouski/kaggle-tweet-sentiment-extraction-1st-place-solution)

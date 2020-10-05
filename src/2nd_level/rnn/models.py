import torch


class TweetCharModel(torch.nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 embed_dim=64, lstm_dim=64,
                 char_embed_dim=32, sent_embed_dim=32,
                 ft_lstm_dim=32, n_models=1):
        super().__init__()
        self.use_msd = use_msd

        self.char_embeddings = torch.nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = torch.nn.Embedding(3, sent_embed_dim)

        self.proba_lstm = torch.nn.LSTM(2,
                                        ft_lstm_dim,
                                        batch_first=True, bidirectional=True)

        self.lstm = torch.nn.LSTM(
            char_embed_dim + ft_lstm_dim * 2 + sent_embed_dim,
            lstm_dim,
            batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(lstm_dim * 2,
                                   lstm_dim,
                                   batch_first=True, bidirectional=True)

        self.logits = torch.nn.Sequential(
            torch.nn.Linear(lstm_dim * 4, lstm_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(lstm_dim, 2))

        self.high_dropout = torch.nn.Dropout(p=0.5)

    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()

        probas = torch.cat([start_probas, end_probas], -1)
        probas_fts, _ = self.proba_lstm(probas)

        char_fts = self.char_embeddings(tokens)

        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))

        features = torch.cat([char_fts, sentiment_fts, probas_fts], -1)
        features, _ = self.lstm(features)
        features2, _ = self.lstm2(features)

        features = torch.cat([features, features2], -1)

        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features))
                     for _ in range(5)],
                    dim=0),
                dim=0)
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits

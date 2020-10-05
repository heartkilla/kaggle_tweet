import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, padding="same", use_bn=True):
        super().__init__()
        if padding == "same":
            padding = kernel_size // 2 * dilation
        
        if use_bn:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride, dilation=dilation),
                nn.ReLU(),
            )
                
    def forward(self, x):
        return self.conv(x)


class Wavenet(nn.Module):
    def __init__(self, len_voc, use_msd=True,
                 cnn_dim=64, char_embed_dim=32, sent_embed_dim=32, proba_cnn_dim=32, n_models=1, kernel_size=3, use_bn=False):
        super().__init__()
        self.use_msd = use_msd
        
        self.char_embeddings = nn.Embedding(len_voc, char_embed_dim)
        self.sentiment_embeddings = nn.Embedding(3, sent_embed_dim)
        
        self.probas_cnn = ConvBlock(n_models * 2, proba_cnn_dim, kernel_size=kernel_size, use_bn=use_bn)
         
        self.cnn = nn.Sequential(
            ConvBlock(char_embed_dim + sent_embed_dim + proba_cnn_dim, cnn_dim, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim, cnn_dim * 2, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 2 , cnn_dim * 4, kernel_size=kernel_size, use_bn=use_bn),
            ConvBlock(cnn_dim * 4, cnn_dim * 8, kernel_size=kernel_size, use_bn=use_bn),
        )
        
        self.logits = nn.Sequential(
            nn.Linear(cnn_dim * 8, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 2),
        )
        
        self.high_dropout = nn.Dropout(p=0.5)
        
    def forward(self, tokens, sentiment, start_probas, end_probas):
        bs, T = tokens.size()
        
        probas = torch.cat([start_probas, end_probas], -1).permute(0, 2, 1)
        probas_fts = self.probas_cnn(probas).permute(0, 2, 1)

        char_fts = self.char_embeddings(tokens)
        
        sentiment_fts = self.sentiment_embeddings(sentiment).view(bs, 1, -1)
        sentiment_fts = sentiment_fts.repeat((1, T, 1))
        
        x = torch.cat([char_fts, sentiment_fts, probas_fts], -1).permute(0, 2, 1)

        features = self.cnn(x).permute(0, 2, 1) # [Bs x T x nb_ft]
    
        if self.use_msd and self.training:
            logits = torch.mean(
                torch.stack(
                    [self.logits(self.high_dropout(features)) for _ in range(5)],
                    dim=0,
                    ),
                dim=0,
            )
        else:
            logits = self.logits(features)

        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]

        return start_logits, end_logits

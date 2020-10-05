import torch
import transformers

import config


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.TRAINED_MODEL_PATH + '/model_0.bin',
            config=conf)
        self.high_dropout = torch.nn.Dropout(config.HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(config.HIDDEN_SIZE * 2, 2)

        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # sequence_output of N_LAST_HIDDEN + Embedding states
        # (N_LAST_HIDDEN + 1, batch_size, num_tokens, 768)
        _, _, out = self.roberta(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)

        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

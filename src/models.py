import torch
import transformers

import config


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_PATH,
            config=conf)
        self.dropout0 = torch.nn.Dropout(config.LAST_DROPOUT)
        self.dropout1 = torch.nn.Dropout(config.LAST_DROPOUT)
        self.dropout2 = torch.nn.Dropout(config.LAST_DROPOUT)
        self.dropout3 = torch.nn.Dropout(config.LAST_DROPOUT)
        self.dropout4 = torch.nn.Dropout(config.LAST_DROPOUT)
        self.l0 = torch.nn.Linear(768 * 2, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # sequence_output of N_LAST_HIDDEN states
        # (N_LAST_HIDDEN, batch_size, num_tokens, 768)
        _, _, out = self.roberta(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)

        out = torch.stack(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=0)
        out_mean = torch.mean(out, dim=0)
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)
        out0 = self.dropout0(out)
        out1 = self.dropout1(out)
        out2 = self.dropout2(out)
        out3 = self.dropout3(out)
        out4 = self.dropout4(out)
        out = torch.stack((out0, out1, out2, out3, out4), dim=0)
        out = self.l0(out)
        logits = torch.mean(out, dim=0)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

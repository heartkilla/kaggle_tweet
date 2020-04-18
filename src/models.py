import torch
import transformers

import config


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            config.ROBERTA_PATH,
            config=conf)
        self.dropout = torch.nn.Dropout(0.1)
        self.l0 = torch.nn.Linear(768 * config.N_LAST_HIDDEN, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        # sequence_output of N_LAST_HIDDEN states
        # (N_LAST_HIDDEN, batch_size, num_tokens, 768)
        _, _, out = self.roberta(ids, attention_mask=mask,
                                 token_type_ids=token_type_ids)

        out = torch.cat(
            tuple(out[-i - 1] for i in range(config.N_LAST_HIDDEN)), dim=-1)
        out = self.dropout(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        # (batch_size, num_tokens)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

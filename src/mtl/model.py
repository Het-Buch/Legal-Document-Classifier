import torch
import torch.nn as nn
from transformers import AutoModel


class MTLModel(nn.Module):
    def __init__(self, model_path, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        hidden = self.encoder.config.hidden_size

        self.chunk_head = nn.Linear(hidden, num_labels)
        self.doc_head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]

        chunk_logits = self.chunk_head(cls)
        doc_logits = self.doc_head(cls)

        return chunk_logits, doc_logits

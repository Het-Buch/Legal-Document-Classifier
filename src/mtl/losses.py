import torch.nn as nn


class MTLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, chunk_logits, doc_logits, labels):
        loss_chunk = self.bce(chunk_logits, labels)
        loss_doc = self.bce(doc_logits, labels)
        return loss_chunk + loss_doc

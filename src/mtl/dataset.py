import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.data.dataset_reader import load_split

MAX_LEN = 512
STRIDE = 128


class CUADChunkDataset(Dataset):
    def __init__(self, split, model_path):
        self.texts, self.labels, self.meta = load_split(split)
        self.label_set = self.meta["label_set"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.samples = []

        for doc_id, (text, label_vector) in enumerate(zip(self.texts, self.labels)):
            tokens = self.tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False
            )["input_ids"]

            for start in range(0, len(tokens), MAX_LEN - STRIDE):
                end = start + MAX_LEN
                chunk = tokens[start:end]

                if len(chunk) < 10:
                    continue

                padding_len = MAX_LEN - len(chunk)
                chunk = chunk + [self.tokenizer.pad_token_id] * padding_len
                attention_mask = [1] * (MAX_LEN - padding_len) + [0] * padding_len

                self.samples.append({
                    "input_ids": torch.tensor(chunk),
                    "attention_mask": torch.tensor(attention_mask),
                    "labels": torch.tensor(label_vector, dtype=torch.float),
                    "doc_id": torch.tensor(doc_id)
                })

            

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

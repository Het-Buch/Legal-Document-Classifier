import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from src.data.cuad_loader import load_cuad_dataset
from src.utils.seed import set_seed

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
OUTPUT_DIR = "models/checkpoints/dap"
MAX_LEN = 512


class CUADMLMDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.examples = []

        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = {k: v.squeeze(0) for k, v in self.examples[idx].items()}
        return item


def main():
    set_seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔹 Loading CUAD documents...")
    documents, _, _ = load_cuad_dataset()

    # DAP uses ONLY raw text
    print(f"Total documents for DAP: {len(documents)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    dataset = CUADMLMDataset(documents, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=1000,
        save_total_limit=1,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    print("🚀 Starting Domain-Adaptive Pretraining...")
    trainer.train()

    print("💾 Saving DAP model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "dap_metadata.json"), "w") as f:
        json.dump(
            {
                "base_model": MODEL_NAME,
                "task": "Masked Language Modeling",
                "dataset": "CUAD",
                "epochs": 2,
                "max_length": MAX_LEN
            },
            f,
            indent=2
        )

    print("✅ Stage 5 (DAP) completed successfully.")


if __name__ == "__main__":
    main()

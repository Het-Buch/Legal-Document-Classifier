import torch
from torch.utils.data import DataLoader
from src.mtl.dataset import CUADChunkDataset
from src.mtl.model import MTLModel
from src.mtl.losses import MTLLoss
from src.utils.seed import set_seed
from pathlib import Path
from src.utils.file_io import load_json
import os

MODEL_PATH = "models/checkpoints/dap"
SAVE_PATH = "models/checkpoints/mtl/final_model.pt"
BATCH_SIZE = 2
EPOCHS = 3


def train():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("📦 Building training dataset...")
    train_ds = CUADChunkDataset("train", MODEL_PATH)
    print(f"✅ Total chunks: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    model = MTLModel(MODEL_PATH, len(train_ds.label_set))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = MTLLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 🔑 NO doc_id in forward
            chunk_logits, doc_logits = model(input_ids, attention_mask)

            # 🔑 NO doc_id in loss
            loss = criterion(chunk_logits, doc_logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f}")

    Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    os.makedirs("models/checkpoints/mtl", exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "label_set": train_ds.label_set,
        # "thresholds": load_json("artifacts/thresholds/thresholds_mtl.json")
    }

    torch.save(bundle, "models/checkpoints/mtl/mtl_bundle.pt")
    print("✅ Saved MTL bundle with labels + thresholds")
    print("✅ Stage 6 training complete")


if __name__ == "__main__":
    train()

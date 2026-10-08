from pathlib import Path

import torch
from torch import nn

from src.data import PROJECT_ROOT, create_loaders, load_config
from src.engine import run_epoch
from src.model import TextClassifier


def main() -> None:
    config = load_config()
    artifact_dir = Path(config["artifact_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    checkpoint_path = artifact_dir / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run train.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _, validation_loader, vocab, classes = create_loaders(config)
    if vocab != checkpoint["vocab"] or classes != checkpoint["classes"]:
        raise ValueError("Dataset split or labels differ from the checkpoint; retrain first.")

    model = TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_classes=len(classes),
        dropout=float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = run_epoch(model, validation_loader, nn.CrossEntropyLoss(), device)
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()

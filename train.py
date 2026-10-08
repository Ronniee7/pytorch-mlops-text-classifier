import os
from pathlib import Path

import torch
from torch import nn

from src.data import PROJECT_ROOT, create_loaders, load_config
from src.engine import run_epoch
from src.model import TextClassifier
from src.utils import get_logger, set_seed


def main() -> None:
    config = load_config()
    set_seed(int(config["seed"]))
    artifact_dir = Path(config["artifact_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(config["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    train_loader, validation_loader, vocab, classes = create_loaders(config)
    model = TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_classes=len(classes),
        dropout=float(config["dropout"]),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = artifact_dir / "best_model.pt"
    for epoch in range(int(config["epochs"])):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, criterion, device
        )
        logger.info(
            "Epoch %d/%d | train=%s | validation=%s",
            epoch + 1,
            config["epochs"],
            train_metrics,
            validation_metrics,
        )
        if validation_metrics["f1_macro"] > best_f1:
            best_f1 = validation_metrics["f1_macro"]
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "classes": classes,
                "config": config,
                "metrics": validation_metrics,
            }
            temporary_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, checkpoint_path)
            logger.info("Saved best checkpoint to %s", checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(
                config["early_stopping_patience"]
            ):
                logger.info("Early stopping after epoch %d", epoch + 1)
                break

    logger.info("Best validation macro F1: %.4f", best_f1)


if __name__ == "__main__":
    main()

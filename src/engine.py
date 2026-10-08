import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        raise ValueError("Cannot run an epoch with an empty dataset.")
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for token_ids, lengths, labels in loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(token_ids, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(targets, predictions),
        "f1_macro": f1_score(
            targets, predictions, average="macro", zero_division=0
        ),
    }

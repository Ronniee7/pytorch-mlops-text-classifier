import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD, UNK = "<PAD>", "<UNK>"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.json"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as file:
        return json.load(file)


def build_vocab(
    texts: list[str], max_vocab_size: int, min_frequency: int
) -> dict[str, int]:
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2.")
    if min_frequency < 1:
        raise ValueError("min_token_freq must be at least 1.")

    counter = Counter(token for text in texts for token in tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for token, count in counter.most_common():
        if count < min_frequency or len(vocab) >= max_vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


class TextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        vocab: dict[str, int],
        classes: list[str],
        max_length: int,
    ):
        self.texts = frame["text"].tolist()
        class_to_index = {label: index for index, label in enumerate(classes)}
        self.labels = [class_to_index[label] for label in frame["label"]]
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tokens = tokenize(self.texts[index])[: self.max_length]
        token_ids = [self.vocab.get(token, self.vocab[UNK]) for token in tokens]
        if not token_ids:
            token_ids = [self.vocab[UNK]]
        return torch.tensor(token_ids, dtype=torch.long), self.labels[index]


def _collate_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    token_ids = pad_sequence(sequences, batch_first=True, padding_value=0)
    return token_ids, lengths, torch.tensor(labels, dtype=torch.long)


def create_loaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, dict[str, int], list[str]]:
    data_path = Path(config["data_path"])
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. See README.md for the required CSV format."
        )

    frame = pd.read_csv(data_path)
    required_columns = {"text", "label"}
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset must include columns: {', '.join(sorted(required_columns))}.")

    frame = frame.loc[:, ["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    if len(frame) < 2 or frame["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least two rows and two distinct labels.")

    classes = sorted(frame["label"].unique().tolist())
    validation_fraction = float(config["validation_fraction"])
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    validation_size = max(1, round(len(frame) * validation_fraction))
    if validation_size >= len(frame):
        raise ValueError("Dataset is too small for the configured validation_fraction.")

    labels = frame["label"]
    stratify = labels if labels.value_counts().min() >= 2 else None
    if stratify is not None and (
        validation_size < len(classes) or len(frame) - validation_size < len(classes)
    ):
        stratify = None
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=validation_size,
        random_state=int(config["seed"]),
        stratify=stratify,
    )
    vocab = build_vocab(
        train_frame["text"].tolist(),
        int(config["max_vocab_size"]),
        int(config["min_token_freq"]),
    )
    train_dataset = TextDataset(
        train_frame, vocab, classes, int(config["max_seq_length"])
    )
    validation_dataset = TextDataset(
        validation_frame, vocab, classes, int(config["max_seq_length"])
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=_collate_batch,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        collate_fn=_collate_batch,
    )
    return train_loader, validation_loader, vocab, classes

import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

import server
from src.data import TextDataset, build_vocab
from src.model import TextClassifier


def _client(tmp_path, monkeypatch):
    classes = ["fake", "real"]
    model = TextClassifier(4, 4, 4, len(classes), 0.0)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": {"<PAD>": 0, "<UNK>": 1, "news": 2, "report": 3},
        "classes": classes,
        "config": {
            "embedding_dim": 4,
            "hidden_dim": 4,
            "dropout": 0.0,
            "max_seq_length": 10,
        },
    }
    torch.save(checkpoint, tmp_path / "best_model.pt")
    monkeypatch.setattr(server, "ARTIFACT_DIR", tmp_path)
    return TestClient(server.app)


def test_vocabulary_reserves_padding_and_unknown_tokens():
    vocab = build_vocab(["news news report"], max_vocab_size=4, min_frequency=1)
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert len(vocab) == 4


def test_dataset_encodes_empty_and_unknown_text():
    vocab = {"<PAD>": 0, "<UNK>": 1, "known": 2}
    frame = pd.DataFrame({"text": ["", "unknown"], "label": ["real", "fake"]})
    dataset = TextDataset(frame, vocab, ["fake", "real"], max_length=8)
    assert dataset[0][0].tolist() == [1]
    assert dataset[0][1] == 1
    assert dataset[1][0].tolist() == [1]


def test_model_returns_one_score_per_class():
    model = TextClassifier(8, 4, 5, 2, 0.0)
    output = model(torch.tensor([[2, 3, 0], [4, 0, 0]]), torch.tensor([2, 1]))
    assert output.shape == (2, 2)


def test_api_predicts_from_loaded_model(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/predict", json={"text": "News report"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] in {"fake", "real"}
    assert set(payload["probabilities"]) == {"fake", "real"}
    assert sum(payload["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", "  "])
def test_api_rejects_blank_text(text, tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/predict", json={"text": text})
    assert response.status_code == 422

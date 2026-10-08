import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import tokenize
from src.model import TextClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {path}. Train the model before starting the API."
        )

    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    required = {"config", "vocab", "classes", "model_state_dict"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(sorted(missing))}")
    if not checkpoint["classes"] or len(checkpoint["classes"]) < 2:
        raise ValueError("Checkpoint must contain at least two classes.")
    return checkpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = load_checkpoint(ARTIFACT_DIR / "best_model.pt")
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    classes = checkpoint["classes"]
    model = TextClassifier(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=len(classes),
        dropout=config["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    app.state.model = model
    app.state.vocab = vocab
    app.state.classes = classes
    app.state.max_seq_length = config["max_seq_length"]
    app.state.device = DEVICE
    yield


app = FastAPI(
    title="Fake News Detector",
    description=(
        "Classifies text using a model trained on labeled examples. "
        "Predictions are estimates, not independent fact-checks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be blank.")

    tokens = tokenize(text)[: app.state.max_seq_length]
    vocab = app.state.vocab
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if not token_ids:
        token_ids = [vocab["<UNK>"]]

    inputs = torch.tensor([token_ids], dtype=torch.long, device=app.state.device)
    lengths = torch.tensor([len(token_ids)], dtype=torch.long)
    with torch.inference_mode():
        probabilities = torch.softmax(app.state.model(inputs, lengths), dim=1)[0]

    scores = {
        str(label): float(probability)
        for label, probability in zip(app.state.classes, probabilities.cpu().tolist())
    }
    predicted_index = int(probabilities.argmax().item())
    return PredictionResponse(
        prediction=str(app.state.classes[predicted_index]),
        confidence=scores[str(app.state.classes[predicted_index])],
        probabilities=scores,
    )

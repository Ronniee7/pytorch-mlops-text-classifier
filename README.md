# Fake News Detector

A small, reproducible text-classification project with a PyTorch training
pipeline and a FastAPI prediction service. The model learns patterns in the
provided labels; it does not verify claims against independent sources.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then run:

```powershell
python -m pip install -r requirements.txt
```

## Dataset

Place a UTF-8 CSV at `data/news.csv` with two columns:

| text | label |
| --- | --- |
| Example article text | real |
| Another example | fake |

Include representative examples for every label. Rows with missing or blank
text/labels are ignored. The dataset is split into training and validation
sets using the seed and validation fraction in `configs/config.json`.

## Train and evaluate

```powershell
python train.py
python evaluate.py
```

The best validation macro-F1 checkpoint is saved to `artifacts/best_model.pt`;
training logs are written to `logs/training.log`. The dataset and generated
artifacts are intentionally not part of the source repository.

## Run the API

```powershell
uvicorn server:app --host 0.0.0.0 --port 8000
```

The service requires a trained checkpoint. Check `GET /health`, open
`/docs` for the interactive API, and submit JSON to `POST /predict`:

```json
{"text": "Article text to classify"}
```

The response includes the predicted label, its confidence, and probabilities
for every trained label. Scores reflect the training data and should not be
treated as proof that a story is true or false.

Set `ARTIFACT_DIR` to load a checkpoint from a directory other than
`artifacts/`.

## Tests

```powershell
python -m pytest
```

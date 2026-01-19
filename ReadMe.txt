# ML CI/CD Starter (GitHub Actions Portfolio Project)

## Overview
This repository demonstrates an end-to-end machine learning workflow implemented with GitHub Actions:
- Linting (ruff) and unit testing (pytest)
- Model training and evaluation
- Metric gating (workflow fails if performance is below a threshold)
- Artifact publishing (model + reports)
- Release packaging on semantic version tags

## Pipeline Design
### CI Workflow (Pull Requests + main branch)
Jobs:
1) lint_and_test: static checks and unit tests
2) train: trains a model and uploads artifacts
3) evaluate: downloads artifacts and enforces a metric gate

### Release Workflow (Tags vX.Y.Z)
Re-trains and re-evaluates the model, then packages a deployable bundle as a GitHub Release asset.

## How to Run Locally
```bash
pip install -r requirements.txt
ruff check .
pytest
python src/train.py --config configs/train_config.yml
python src/evaluate.py --config configs/train_config.yml --model-path artifacts/model.joblib

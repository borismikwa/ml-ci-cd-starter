from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import make_regression_data
from src.features import build_preprocess_pipeline
from src.model import build_model
from src.utils import ensure_dir, write_json


@dataclass
class TrainMetrics:
    train_r2: float
    val_r2: float
    n_samples: int
    n_features: int
    seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    seed = int(cfg["data"]["seed"])
    X, y = make_regression_data(
        seed=seed,
        n_samples=int(cfg["data"]["n_samples"]),
        n_features=int(cfg["data"]["n_features"]),
        noise=float(cfg["data"]["noise"]),
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=float(cfg["data"]["test_size"]), random_state=seed
    )

    preprocess = build_preprocess_pipeline()
    model = build_model(
        model_type=str(cfg["model"]["type"]),
        alpha=float(cfg["model"]["alpha"]),
        seed=seed,
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    train_r2 = float(pipe.score(X_train, y_train))
    val_r2 = float(pipe.score(X_val, y_val))

    artifacts_dir = str(cfg["outputs"]["artifacts_dir"])
    reports_dir = str(cfg["outputs"]["reports_dir"])
    ensure_dir(artifacts_dir)
    ensure_dir(reports_dir)

    joblib.dump(pipe, f"{artifacts_dir}/model.joblib")

    metrics = TrainMetrics(
        train_r2=train_r2,
        val_r2=val_r2,
        n_samples=int(cfg["data"]["n_samples"]),
        n_features=int(cfg["data"]["n_features"]),
        seed=seed,
    )
    write_json(f"{reports_dir}/train_metrics.json", metrics)

    print("Training complete.")
    print(f"train_r2={train_r2:.4f} val_r2={val_r2:.4f}")


if __name__ == "__main__":
    main()

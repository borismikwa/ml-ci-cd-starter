from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import yaml

from src.data import make_regression_data
from src.utils import write_json


@dataclass
class EvalReport:
    r2: float
    min_r2: float
    passed: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yml")
    p.add_argument("--model-path", default="artifacts/model.joblib")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model = joblib.load(args.model_path)

    # Fresh evaluation set (seed + 1). Swap to a real holdout dataset later if desired.
    seed = int(cfg["data"]["seed"]) + 1
    X, y = make_regression_data(
        seed=seed,
        n_samples=2000,
        n_features=int(cfg["data"]["n_features"]),
        noise=float(cfg["data"]["noise"]),
    )

    r2 = float(model.score(X, y))
    min_r2 = float(cfg["gates"]["min_r2"])
    passed = r2 >= min_r2

    report = EvalReport(r2=r2, min_r2=min_r2, passed=passed)
    write_json(f"{cfg['outputs']['reports_dir']}/eval_report.json", report)

    print(f"eval_r2={r2:.4f} (min_r2={min_r2:.4f}) passed={passed}")

    # This is the quality gate: failing exits the workflow with non-zero code
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

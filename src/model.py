from __future__ import annotations
from sklearn.linear_model import Ridge


def build_model(model_type: str, alpha: float, seed: int) -> Ridge:
    if model_type != "ridge":
        raise ValueError(f"Unsupported model_type: {model_type}")
    return Ridge(alpha=alpha, random_state=seed)

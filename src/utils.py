from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    payload = asdict(obj) if is_dataclass(obj) else obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

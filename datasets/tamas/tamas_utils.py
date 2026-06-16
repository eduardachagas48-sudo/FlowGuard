from __future__ import annotations

import json
import pandas as pd
from pathlib import Path

def load_processed_table(name: str) -> pd.DataFrame:
    """Load a processed parquet table by name."""

    PROCESSED_DIR = Path("results/tamas/processed")
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run the setup and feature-extraction notebook first."
        )
    return pd.read_parquet(path)

def load_tamas_jsonl(path: str | Path) -> list[dict]:
    """
    Load TAMAS tasks from .jsonl or .json.

    Supported formats:
    - JSONL: one task per line
    - JSON list: [task1, task2, ...]
    - JSON dict with "tasks": [...]
    - Single JSON task dict
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TAMAS file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "tasks" in data:
        return data["tasks"]

    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported TAMAS input format: {type(data)}")

def normalize_tamas_single_task(raw: dict, idx: int = 0, attack_type: str = "benign") -> dict:
    """
    Normalize possible TAMAS JSON variants into a stable internal schema.
    """
    task_id = (
        raw.get("task_id")
        or raw.get("id")
        or raw.get("example_id")
        or f"tamas_{idx:05d}"
    )

    user_query = (
        raw.get("user query")
        or raw.get("user_query")
        or raw.get("query")
        or raw.get("prompt")
        or raw.get("task")
        or ""
    )
    is_attack = attack_type not in {"", "none", "benign", None}

    agents = raw.get("agents", [])
    if agents is None:
        agents = []

    return {
        "task_id": str(task_id),
        "user_query": str(user_query),
        "agents": agents,
        "attack_type": str(attack_type),
        "is_attack": bool(is_attack),
        "expected_label": raw.get("expected_label", raw.get("label")),
        "metadata": raw,
    }


def normalize_tamas_tasks(raw_tasks: list[dict], attack_type: str = "benign") -> list[dict]:
    return [
        normalize_tamas_single_task(raw, idx=i, attack_type=attack_type)
        for i, raw in enumerate(raw_tasks)
    ]


# def get_tamas_task_id(task: dict) -> str:
#     return str(task.get("task_id", ""))


# def get_tamas_user_query(task: dict) -> str:
#     return str(task.get("user_query", ""))


# def get_tamas_agents(task: dict) -> list[dict]:
#     return list(task.get("agents", []))


# def get_tamas_scenario(task: dict) -> str:
#     return str(task.get("scenario", ""))


# def get_tamas_attack_type(task: dict) -> str:
#     return str(task.get("attack_type", "benign"))


# def is_tamas_attack(task: dict) -> bool:
#     return bool(task.get("is_attack", False))


# def infer_scenario_from_path(path: str | Path) -> str | None:
#     name = Path(path).stem.lower()

#     for scenario in ["education", "healthcare", "finance", "legal", "congen", "news"]:
#         if scenario in name:
#             return scenario

#     return None
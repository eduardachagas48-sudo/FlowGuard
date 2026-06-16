import json
from pathlib import Path
from datasets.tamas.tamas_features import build_all_feature_tables

def compute_features(record: dict, scenario: str, attack: str, seed: int):
    FEATURES_DIR = Path("results/tamas/features")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    if record:
        episode_df, agent_df, tool_df, early_df = build_all_feature_tables(record)

        episode_df.to_csv(FEATURES_DIR / f"episode_features_{scenario}_{attack}_seed_{seed}.csv", index=False)
        agent_df.to_csv(FEATURES_DIR / f"agent_features_{scenario}_{attack}_seed_{seed}.csv", index=False)
        tool_df.to_csv(FEATURES_DIR / f"tool_features_{scenario}_{attack}_seed_{seed}.csv", index=False)
        early_df.to_csv(FEATURES_DIR / f"early_trace_features_{scenario}_{attack}_seed_{seed}.csv", index=False)

    print("episode_df:", episode_df.shape)
    print("agent_df:", agent_df.shape)
    print("tool_df:", tool_df.shape)
    print("early_df:", early_df.shape)

    return episode_df, agent_df, tool_df, early_df

def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Erro ao decodificar JSON na linha {line_idx} de {path}: {e}"
                ) from e

    return records

def find_tamas_file(condition: str, scenario: str) -> str:
    """
    Resolve paths for benign and adversarial TAMAS files.

    Expected examples:
        datasets/tamas/benign/education_benign.json
        datasets/tamas/byzantine/education_byzantine.json
        datasets/tamas/DPI/education_DPI.json
    """
    candidates = [
        Path(f"datasets/tamas/{condition}/{scenario}_{condition}.json"),
        Path(f"datasets/tamas/{condition}/{scenario}_{condition}.jsonl"),
        Path(f"datasets/tamas/{condition}/{scenario}.json"),
        Path(f"datasets/tamas/{condition}/{scenario}.jsonl"),
        Path(f"datasets/tamas/{scenario}_{condition}.json"),
        Path(f"datasets/tamas/{scenario}_{condition}.jsonl"),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find TAMAS file for "
        f"condition={condition}, scenario={scenario}. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )
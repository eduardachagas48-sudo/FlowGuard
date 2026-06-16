import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve,
    make_scorer,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("results/tamas")
FEATURES_DIR = RESULTS_DIR / "features"
PLOTS_DIR = RESULTS_DIR / "plots"
BASELINE_CATEGORICAL = ["architecture", "model_name", "scenario"]

OUTPUT_DIR = Path("results/tamas/paper_ready_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARLY_FEATURES_CANDIDATES = [
    "prefix_steps",
    "total_steps",
    "observed_step_ratio",
    "num_observed_steps",
    "num_llm_calls_prefix",

    "prompt_tokens_prefix_sum",
    "response_tokens_prefix_sum",
    "total_tokens_prefix_sum",

    "prompt_tokens_prefix_mean",
    "response_tokens_prefix_mean",
    "total_tokens_prefix_mean",

    "prompt_tokens_prefix_std",
    "response_tokens_prefix_std",
    "total_tokens_prefix_std",

    "latency_prefix_sum",
    "latency_prefix_mean",
    "latency_prefix_std",

    "eval_duration_prefix_sum",
    "eval_duration_prefix_mean",
    "eval_duration_prefix_std",

    "prompt_eval_duration_prefix_sum",
    "prompt_eval_duration_prefix_mean",

    "tokens_per_second_prefix_mean",
    "tokens_per_second_prefix_std",
    "tokens_per_second_prefix_min",
    "tokens_per_second_prefix_max",

    "prompt_eval_count_prefix_sum",
    "eval_count_prefix_sum",

    "token_budget_prefix_sum",
    "token_budget_prefix_mean",

    "num_unique_roles_prefix",
    "num_unique_agents_prefix",

    "num_requested_tools_prefix",
    "num_executed_tools_prefix",

    "final_answer_words_prefix_sum",
    "final_answer_words_prefix_mean",
    "final_answer_words_prefix_std",

    "response_to_prompt_ratio_prefix",
    "tokens_per_latency_prefix",
    "tool_execution_rate_prefix",
]

ALL_TELEMETRY_FEATURES = [
    "num_messages",
    "trace_word_length",
    "avg_message_words",
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "latency_total_s",
    "num_llm_calls",
    "num_tool_calls",
    "unique_tool_calls",
    "tool_call_entropy",
    "first_tool_call_position",
    "injection_marker_count",
    "refusal_marker_count",
    "contradiction_marker_count",
    "avg_pairwise_message_similarity",
    "var_pairwise_message_similarity",
]

OPERATIONAL_ONLY_FEATURES = [
    "num_messages",
    "trace_word_length",
    "avg_message_words",
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "latency_total_s",
    "num_llm_calls",
    "num_tool_calls",
    "unique_tool_calls",
    "tool_call_entropy",
    "first_tool_call_position",
]

SEMANTIC_BEHAVIORAL_FEATURES = [
    "injection_marker_count",
    "refusal_marker_count",
    "contradiction_marker_count",
    "avg_pairwise_message_similarity",
    "var_pairwise_message_similarity",
]

TELEMETRY_NUMERIC = [
    "num_messages",
    "trace_word_length",
    "avg_message_words",
    "prompt_tokens",
    "response_tokens",
    "total_tokens",
    "latency_total_s",
    "num_llm_calls",
    "num_tool_calls",
    "unique_tool_calls",
    "tool_call_entropy",
    "first_tool_call_position",
    "injection_marker_count",
    "refusal_marker_count",
    "contradiction_marker_count",
    "avg_pairwise_message_similarity",
    "var_pairwise_message_similarity",
]

FEATURE_GROUPS = {
    "cost": [
        "prompt_tokens",
        "response_tokens",
        "total_tokens",
        "latency_total_s",
        "num_llm_calls",
    ],
    "tool_use": [
        "num_tool_calls",
        "unique_tool_calls",
        "tool_call_entropy",
        "first_tool_call_position",
        "num_suspicious_tool_calls",
        "used_any_suspicious_tool",
    ],
    "linguistic_markers": [
        "injection_marker_count",
        "refusal_marker_count",
        "contradiction_marker_count",
    ],
    "semantic_coordination": [
        "avg_pairwise_message_similarity",
        "var_pairwise_message_similarity",
        "num_messages",
        "trace_word_length",
        "avg_message_words",
    ],
    "fragmentation": [
        "semantic_fragmentation_proxy",
        "risk_conditioned_fragmentation",
        "fragmentation_excess_over_benign",
    ],
    "all": TELEMETRY_NUMERIC + [
        "num_suspicious_tool_calls",
        "used_any_suspicious_tool",
        "semantic_fragmentation_proxy",
        "risk_conditioned_fragmentation",
        "fragmentation_excess_over_benign",
    ],
}

BASELINE_CATEGORICAL = [
    "architecture",
    "model_name",
    "scenario",
]

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

def mean_std_str(mean, std, decimals=3):
    if pd.isna(std):
        return f"{mean:.{decimals}f} ± NA"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"

def get_classification_metrics(predictions_per_seed_df):

    GROUP_COLS = [
        "target",
        "feature_set",
        "model",
    ]

    summary_per_seed = (
        predictions_per_seed_df
        .groupby(GROUP_COLS)
        .agg(
            n_seeds=("seed", "nunique"),
            n_mean=("n", "mean"),
            positive_rate_mean=("positive_rate", "mean"),
            balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
            balanced_accuracy_std=("balanced_accuracy_mean", "std"),
            f1_mean=("f1_mean", "mean"),
            f1_std=("f1_mean", "std"),
            roc_auc_mean=("roc_auc_mean", "mean"),
            roc_auc_std=("roc_auc_mean", "std"),
        )
        .reset_index()
    )

    paper_table = summary_per_seed.copy()

    paper_table["Balanced Acc."] = paper_table.apply(
        lambda r: mean_std_str(
            r["balanced_accuracy_mean"],
            r["balanced_accuracy_std"],
        ),
        axis=1,
    )

    paper_table["F1"] = paper_table.apply(
        lambda r: mean_std_str(
            r["f1_mean"],
            r["f1_std"],
        ),
        axis=1,
    )

    paper_table["ROC-AUC"] = paper_table.apply(
        lambda r: mean_std_str(
            r["roc_auc_mean"],
            r["roc_auc_std"],
        ),
        axis=1,
    )

    paper_table = paper_table[
        [
            "target",
            "feature_set",
            "model",
            "n_seeds",
            "n_mean",
            "positive_rate_mean",
            "Balanced Acc.",
            "F1",
            "ROC-AUC",
        ]
    ]

    return paper_table

def evaluate_binary_prediction(
    df, 
    target_col, 
    feature_set="telemetry", 
    model_kind="logreg", 
    min_samples=10,
):
    
    df = df[df[target_col].notna()].copy()
    if df.empty or df[target_col].nunique() < 2 or len(df) < min_samples:
        print(f"Skipping {target_col}: insufficient data/classes.")
        return None

    y = df[target_col].astype(int)
    categorical = [c for c in BASELINE_CATEGORICAL if c in df.columns]
    numeric = [] if feature_set == "baseline" else [c for c in TELEMETRY_NUMERIC if c in df.columns]

    transformers = []
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))

    pre = ColumnTransformer(transformers)

    if model_kind == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    min_class = y.value_counts().min()
    if min_class < 2:
        print(f"Skipping {target_col}: min class < 2.")
        return None

    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)
    scoring = {"balanced_accuracy": "balanced_accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    scores = cross_validate(pipe, df, y, cv=cv, scoring=scoring, error_score=np.nan)

    return {
        "target": target_col,
        "feature_set": feature_set,
        "model": model_kind,
        "n": len(df),
        "positive_rate": float(y.mean()),
        "balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
        "f1_mean": float(np.nanmean(scores["test_f1"])),
        "roc_auc_mean": float(np.nanmean(scores["test_roc_auc"])),
    }

def behavioral_state_clustering(df, k=4):

    numeric = [c for c in TELEMETRY_NUMERIC if c in df.columns]
    if df.empty or len(df) < k or not numeric:
        print("Insufficient data for clustering.")
        return df
    
    X = df[numeric].copy()
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    out = df.copy()
    out["behavioral_state"] = labels

    try:
        print("Silhouette:", silhouette_score(X, labels))
    except Exception:
        pass

    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    out["pca_1"], out["pca_2"] = Z[:, 0], Z[:, 1]
    
    return out

def fit_rf_feature_importance(df, target_col):

    if df.empty or target_col not in df or df[target_col].nunique() < 2:
        print("Skipping feature importance:", target_col)
        return pd.DataFrame()

    numeric = [c for c in TELEMETRY_NUMERIC if c in df.columns]
    if not numeric:
        return pd.DataFrame()

    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df[numeric]), columns=numeric)
    # y = df[target_col].astype(int)
    y = df[target_col]

    clf = RandomForestClassifier(n_estimators=500, random_state=42, class_weight="balanced_subsample", min_samples_leaf=2)
    clf.fit(X, y)

    return pd.DataFrame({"feature": numeric, "importance": clf.feature_importances_, "target": target_col}).sort_values("importance", ascending=False)

def evaluate_multiclass_attack_type(
    df,
    feature_set="telemetry",
    model_kind="rf",
    min_samples_per_class=2,
):
    df = df[df["attack_type"].notna()].copy()

    counts = df["attack_type"].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    df = df[df["attack_type"].isin(valid_classes)].copy()

    if df.empty or df["attack_type"].nunique() < 2:
        print("Insufficient classes for multiclass evaluation.")
        return None

    y = df["attack_type"].astype(str)

    categorical = [c for c in BASELINE_CATEGORICAL if c in df.columns]
    numeric = [] if feature_set == "baseline" else [c for c in TELEMETRY_NUMERIC if c in df.columns]

    transformers = []

    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    if numeric:
        transformers.append(
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric)
        )

    pre = ColumnTransformer(transformers)

    if model_kind == "logreg":
        clf = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            # multi_class="auto",
        )
    elif model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        )
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])

    min_class = y.value_counts().min()
    cv = StratifiedKFold(
        n_splits=min(5, min_class),
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }

    scores = cross_validate(
        pipe,
        df,
        y,
        cv=cv,
        scoring=scoring,
        error_score=np.nan,
    )

    return {
        "target": "attack_type",
        "feature_set": feature_set,
        "model": model_kind,
        "n": len(df),
        "num_classes": y.nunique(),
        "balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
        "f1_macro_mean": float(np.nanmean(scores["test_f1_macro"])),
        "f1_weighted_mean": float(np.nanmean(scores["test_f1_weighted"])),
    }

def evaluate_multiclass_feature_group_ablation(
    df: pd.DataFrame,
    target_col: str = "attack_type",
    feature_groups: dict[str, list[str]] = FEATURE_GROUPS,
    model_kind: str = "rf",
    min_samples_per_class: int = 2,
) -> pd.DataFrame:

    rows = []
    df = df[df[target_col].notna()].copy()

    counts = df[target_col].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    df = df[df[target_col].isin(valid_classes)].copy()

    if df.empty or df[target_col].nunique() < 2:
        print("Insufficient classes for multiclass ablation.")
        return pd.DataFrame()

    y = df[target_col].astype(str)
    min_class = y.value_counts().min()

    if min_class < 2:
        print("Each class needs at least 2 examples.")
        return pd.DataFrame()

    cv = StratifiedKFold(
        n_splits=min(5, min_class),
        shuffle=True,
        random_state=42,
    )

    for group_name, features in feature_groups.items():
        numeric_features = [c for c in features if c in df.columns]

        if not numeric_features:
            continue

        X = df[numeric_features].copy()

        pre = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        if model_kind == "logreg":
            clf = LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
            )
        elif model_kind == "rf":
            clf = RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            )
        else:
            raise ValueError(f"Unsupported model_kind: {model_kind}")

        pipe = Pipeline([
            ("pre", pre),
            ("clf", clf),
        ])

        scoring = {
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": make_scorer(f1_score, average="macro"),
            "f1_weighted": make_scorer(f1_score, average="weighted"),
        }

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )

        rows.append({
            "target": target_col,
            "feature_group": group_name,
            "model": model_kind,
            "n": len(df),
            "num_classes": y.nunique(),
            "num_features": len(numeric_features),
            "balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
            "f1_macro_mean": float(np.nanmean(scores["test_f1_macro"])),
            "f1_weighted_mean": float(np.nanmean(scores["test_f1_weighted"])),
        })

    return pd.DataFrame(rows)

# def assign_akc_phase(row: pd.Series) -> str:
#     attack_type = str(row.get("attack_type", "")).lower()

#     if attack_type in ["dpi", "ipi", "impersonation"]:
#         return "semantic_infection"

#     if attack_type in ["byzantine", "contradicting"]:
#         return "cognitive_compromise"

#     if attack_type in ["colluding"]:
#         return "agency_propagation"

#     if bool(row.get("attack_tool_invoked", False)) or bool(row.get("used_any_suspicious_tool", False)):
#         return "systemic_execution"

#     if attack_type == "benign":
#         return "benign"

#     return "unknown"

# def assign_ooda_stage(akc_phase: str) -> str:
#     mapping = {
#         "semantic_infection": "observe",
#         "cognitive_compromise": "orient_decide",
#         "agency_propagation": "act",
#         "systemic_execution": "impact",
#         "benign": "benign",
#     }
#     return mapping.get(akc_phase, "unknown")

# def add_akc_ooda_labels(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     df["akc_phase"] = df.apply(assign_akc_phase, axis=1)
#     df["ooda_stage"] = df["akc_phase"].apply(assign_ooda_stage)
#     return df

# def evaluate_multiclass_target(
#     df: pd.DataFrame,
#     target_col: str,
#     numeric_features: list[str],
#     model_kind: str = "rf",
#     min_samples_per_class: int = 2,
# ) -> dict | None:
#     df = df.copy()
#     df = df[df[target_col].notna()].copy()

#     counts = df[target_col].value_counts()
#     valid = counts[counts >= min_samples_per_class].index
#     df = df[df[target_col].isin(valid)].copy()

#     if df.empty or df[target_col].nunique() < 2:
#         print(f"Insufficient classes for {target_col}.")
#         return None

#     y = df[target_col].astype(str)
#     X = df[[c for c in numeric_features if c in df.columns]].copy()

#     pre = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler()),
#     ])

#     if model_kind == "rf":
#         clf = RandomForestClassifier(
#             n_estimators=500,
#             random_state=42,
#             class_weight="balanced_subsample",
#             min_samples_leaf=2,
#         )
#     else:
#         clf = LogisticRegression(
#             max_iter=3000,
#             class_weight="balanced",
#         )

#     pipe = Pipeline([
#         ("pre", pre),
#         ("clf", clf),
#     ])

#     min_class = y.value_counts().min()

#     cv = StratifiedKFold(
#         n_splits=min(5, min_class),
#         shuffle=True,
#         random_state=42,
#     )

#     scoring = {
#         "balanced_accuracy": "balanced_accuracy",
#         "f1_macro": make_scorer(f1_score, average="macro"),
#         "f1_weighted": make_scorer(f1_score, average="weighted"),
#     }

#     scores = cross_validate(
#         pipe,
#         X,
#         y,
#         cv=cv,
#         scoring=scoring,
#         error_score=np.nan,
#     )

#     return {
#         "target": target_col,
#         "model": model_kind,
#         "n": len(df),
#         "num_classes": y.nunique(),
#         "balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
#         "f1_macro_mean": float(np.nanmean(scores["test_f1_macro"])),
#         "f1_weighted_mean": float(np.nanmean(scores["test_f1_weighted"])),
#     }

# def summarize_akc_telemetry(df: pd.DataFrame) -> pd.DataFrame:
#     metrics = [
#         "prompt_tokens",
#         "response_tokens",
#         "total_tokens",
#         "latency_total_s",
#         "num_tool_calls",
#         "tool_call_entropy",
#         "injection_marker_count",
#         "contradiction_marker_count",
#         "avg_pairwise_message_similarity",
#         "var_pairwise_message_similarity",
#     ]

#     metrics = [m for m in metrics if m in df.columns]

#     return (
#         df.groupby("akc_phase")[metrics]
#         .agg(["mean", "std", "median"])
#         .reset_index()
#     )

def semantic_fragmentation_proxy(row):
    similarity = row.get("avg_pairwise_message_similarity", 0)
    variance = row.get("var_pairwise_message_similarity", 0)
    num_messages = row.get("num_messages", 0)
    tool_entropy = row.get("tool_call_entropy", 0)

    return (
        np.log1p(num_messages)
        + variance
        + tool_entropy
        - similarity
    )

def telemetry_guardrail_decision(row):
    if row.get("injection_marker_count", 0) > 0:
        return "trigger_security_review"

    if row.get("contradiction_marker_count", 0) > 2:
        return "trigger_consistency_check"

    if row.get("used_any_suspicious_tool", False):
        return "block_tool"

    if row.get("avg_pairwise_message_similarity", 0) > 0.85 and row.get("attack_type") == "colluding":
        return "require_independent_verifier"

    return "allow"

def cluster_behavior_states(episode_df_lst, k):

    if not episode_df_lst.empty and len(episode_df_lst) >= 4:
        clustered_df = behavioral_state_clustering(episode_df_lst, k=min(k, len(episode_df_lst)))
        clustered_df.to_csv(FEATURES_DIR / "behavioral_states.csv", index=False)

        plt.figure(figsize=(7, 5))
        for attack_type in clustered_df["attack_type"].fillna("unknown").unique():
            sub = clustered_df[clustered_df["attack_type"].fillna("unknown") == attack_type]
            plt.scatter(sub["pca_1"], sub["pca_2"], label=str(attack_type), alpha=0.8)
            
        plt.title("Behavioral state map by attack type")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        path = PLOTS_DIR / "behavioral_state_map_by_attack_type.png"
        plt.savefig(path, dpi=200)
        plt.show()

        display(clustered_df.groupby(["behavioral_state", "attack_type"], dropna=False).size().reset_index(name="n"))
    else:
        print("Not enough data for clustering.")

def add_benign_normalized_fragmentation(df):

    df["tool_complexity_bin"] = pd.qcut(
        df["num_tool_calls"].rank(method="first"),
        q=4,
        labels=False,
        duplicates="drop",
    )

    benign_ref = (
        df[df["attack_type"] == "benign"]
        .groupby("tool_complexity_bin")["semantic_fragmentation_proxy"]
        .mean()
        .to_dict()
    )

    global_benign_mean = df[df["attack_type"] == "benign"]["semantic_fragmentation_proxy"].mean()

    df["benign_fragmentation_reference"] = df["tool_complexity_bin"].map(benign_ref)
    df["benign_fragmentation_reference"] = df["benign_fragmentation_reference"].fillna(global_benign_mean)

    df["fragmentation_excess_over_benign"] = (
        df["semantic_fragmentation_proxy"]
        - df["benign_fragmentation_reference"]
    )

    return df

def adversarial_fragmentation_index(df):

    index_result_lst = []
    for row in df.itertuples():
        functional_complexity = (
            np.log1p(row.num_tool_calls)
            + row.tool_call_entropy
            + np.log1p(row.num_messages)
        )

        risk_signal = (
            row.injection_marker_count
            + row.contradiction_marker_count
            + row.refusal_marker_count
            + row.num_suspicious_tool_calls
            + row.used_any_suspicious_tool
        )

        semantic_drift = (
            row.var_pairwise_message_similarity
            + (1 - row.avg_pairwise_message_similarity)
        )
        index = float(semantic_drift * np.log1p(1 + risk_signal) / (1 + functional_complexity))
        index_result_lst.append(index)

    return index_result_lst

def risk_conditioned_fragmentation(row):
    base_fragmentation = (
        np.log1p(row.get("num_messages", 0))
        + row.get("var_pairwise_message_similarity", 0)
        + row.get("tool_call_entropy", 0)
        - row.get("avg_pairwise_message_similarity", 0)
    )

    risk_signal = (
        row.get("injection_marker_count", 0)
        + row.get("contradiction_marker_count", 0)
        + row.get("num_suspicious_tool_calls", 0)
        + int(bool(row.get("used_any_suspicious_tool", False)))
        + int(bool(row.get("attack_tool_invoked", False)))
    )

    return base_fragmentation * np.log1p(1 + risk_signal)

# def evaluate_scores_by_seed(df, score_cols, target_col="is_attack"):
#     rows = []

#     for seed, g in df.groupby("seed"):
#         y = g[target_col].astype(int)

#         for score_col in score_cols:
#             score = g[score_col].fillna(0)

#             rows.append({
#                 "seed": seed,
#                 "score": score_col,
#                 "roc_auc": roc_auc_score(y, score),
#                 "auprc": average_precision_score(y, score),
#             })

#     return pd.DataFrame(rows)

def validate_dataframe(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    required_metadata: Optional[List[str]] = None,
) -> None:
    
    required_metadata = required_metadata or []
    required_cols = [target_col] + features + required_metadata
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col} contains NaN values.")

    if len(df) == 0:
        raise ValueError("Dataframe is empty.")

    # print("Dataframe validation passed.")
    # print(f"Rows: {len(df)}")
    # print(f"Target: {target_col}")
    # print(df[target_col].value_counts(dropna=False))

def leave_one_seed_out_eval(
    df,
    target_col="is_attack",
    features=None,
    model_kind="rf",
    feature_group=None,
    return_splits=True,
):
    df = df.copy()

    features = features or TELEMETRY_NUMERIC
    features = [c for c in features if c in df.columns]

    split_rows = []

    for heldout_seed in sorted(df["seed"].unique()):
        train_df = df[df["seed"] != heldout_seed].copy()
        test_df = df[df["seed"] == heldout_seed].copy()

        if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
            continue

        X_train = train_df[features]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[features]
        y_test = test_df[target_col].astype(int)

        model = make_model(model_kind)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_score)
        else:
            auc = np.nan

        split_rows.append({
            "heldout_seed": heldout_seed,
            "feature_group": feature_group,
            "model": model_kind,
            "target": target_col,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "n_features": len(features),
            "features": features,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": auc,
        })

    split_df = pd.DataFrame(split_rows)

    if return_splits:
        return split_df

    if split_df.empty:
        return pd.DataFrame(columns=[
            "feature_group",
            "model",
            "target",
            "n_splits",
            "n_features",
            "features",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
        ])

    return pd.DataFrame([{
        "feature_group": feature_group,
        "model": model_kind,
        "target": target_col,
        "n_splits": len(split_df),
        "n_features": len(features),
        "features": features,
        "balanced_accuracy_mean": split_df["balanced_accuracy"].mean(),
        "balanced_accuracy_std": split_df["balanced_accuracy"].std(),
        "f1_mean": split_df["f1"].mean(),
        "f1_std": split_df["f1"].std(),
        "roc_auc_mean": split_df["roc_auc"].mean(),
        "roc_auc_std": split_df["roc_auc"].std(),
    }])

def evaluate_feature_groups_loso(df, target_col="is_attack", model_kind="rf"):
    rows = []

    for group_name, features in FEATURE_GROUPS.items():
        features = [c for c in features if c in df.columns]

        if not features:
            continue

        res = leave_one_seed_out_eval(
            df,
            target_col=target_col,
            features=features,
            model_kind=model_kind,
            feature_group=group_name,
            return_splits=False,
        )

        if not res.empty:
            rows.append(res)

    if not rows:
        return pd.DataFrame()

    return (
        pd.concat(rows, ignore_index=True)
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )

def evaluate_leave_one_seed_out(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    model_kind: str = "rf",
    seed_col: str = "seed",
) -> pd.DataFrame:
    
    validate_dataframe(df, target_col, features, required_metadata=[seed_col])

    rows = []
    for heldout_seed in sorted(df[seed_col].unique()):
        train_df = df[df[seed_col] != heldout_seed].copy()
        test_df = df[df[seed_col] == heldout_seed].copy()

        X_train = train_df[features]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[features]
        y_test = test_df[target_col].astype(int)

        model = make_model(model_kind=model_kind, seed=int(heldout_seed))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if hasattr(model.named_steps["clf"], "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = y_pred

        row = {
            "heldout_seed": heldout_seed,
            "model": model_kind,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "positive_rate_test": y_test.mean(),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score),
            "auprc": average_precision_score(y_test, y_score),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def evaluate_feature_group_ablation(
    df: pd.DataFrame,
    target_col: str,
    feature_groups: Dict[str, List[str]],
    model_kind: str = "rf",
    seed_col: str = "seed",
) -> pd.DataFrame:
    
    rows = []
    for group_name, features in feature_groups.items():
        features = [f for f in features if f in df.columns]

        if not features:
            continue

        res = evaluate_leave_one_seed_out(
            df=df,
            target_col=target_col,
            features=features,
            model_kind=model_kind,
            seed_col=seed_col,
        )

        rows.append({
            "feature_group": group_name,
            "model": model_kind,
            "num_features": len(features),
            "balanced_accuracy_mean": res["balanced_accuracy"].mean(),
            "balanced_accuracy_std": res["balanced_accuracy"].std(),
            "f1_mean": res["f1"].mean(),
            "f1_std": res["f1"].std(),
            "roc_auc_mean": res["roc_auc"].mean(),
            "roc_auc_std": res["roc_auc"].std(),
            "auprc_mean": res["auprc"].mean(),
            "auprc_std": res["auprc"].std(),
        })

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)

# def evaluate_early_detection(
#     early_df: pd.DataFrame,
#     target_col: str = "is_attack",
#     features: Optional[List[str]] = None,
#     trace_fraction_col: str = "trace_fraction",
#     model_kind: str = "rf",
#     seed_col: str = "seed",
# ) -> pd.DataFrame:
#     if features is None:
#         features = [f for f in TELEMETRY_NUMERIC if f in early_df.columns]

#     rows = []

#     for frac in sorted(early_df[trace_fraction_col].dropna().unique()):
#         frac_df = early_df[early_df[trace_fraction_col] == frac].copy()

#         res = evaluate_leave_one_seed_out(
#             df=frac_df,
#             target_col=target_col,
#             features=features,
#             model_kind=model_kind,
#             seed_col=seed_col,
#         )

#         rows.append({
#             "trace_fraction": frac,
#             "model": model_kind,
#             "balanced_accuracy_mean": res["balanced_accuracy"].mean(),
#             "balanced_accuracy_std": res["balanced_accuracy"].std(),
#             "f1_mean": res["f1"].mean(),
#             "f1_std": res["f1"].std(),
#             "roc_auc_mean": res["roc_auc"].mean(),
#             "roc_auc_std": res["roc_auc"].std(),
#             "auprc_mean": res["auprc"].mean(),
#             "auprc_std": res["auprc"].std(),
#         })

#     return pd.DataFrame(rows)

def evaluate_leave_one_attack_out(
    df: pd.DataFrame,
    target_col: str = "is_attack",
    attack_col: str = "attack_type",
    features: Optional[List[str]] = None,
    model_kind: str = "rf",
) -> pd.DataFrame:
    if features is None:
        features = [f for f in TELEMETRY_NUMERIC if f in df.columns]

    rows = []

    attack_types = [
        a for a in sorted(df[attack_col].dropna().unique())
        if a != "benign"
    ]

    for heldout_attack in attack_types:
        train_df = df[
            (df[attack_col] == "benign") |
            ((df[attack_col] != "benign") & (df[attack_col] != heldout_attack))
        ].copy()

        test_df = df[
            (df[attack_col] == "benign") |
            (df[attack_col] == heldout_attack)
        ].copy()

        X_train = train_df[features]
        y_train = train_df[target_col].astype(int)

        X_test = test_df[features]
        y_test = test_df[target_col].astype(int)

        model = make_model(model_kind=model_kind, seed=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        rows.append({
            "heldout_attack": heldout_attack,
            "model": model_kind,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score),
            "auprc": average_precision_score(y_test, y_score),
        })

    return pd.DataFrame(rows)

def compute_permutation_importance(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    model_kind: str = "rf",
    seed: int = 42,
    n_repeats: int = 30,
) -> pd.DataFrame:
    
    validate_dataframe(df, target_col, features)

    X = df[features]
    y = df[target_col].astype(int)

    model = make_model(model_kind=model_kind, seed=seed)
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=-1,
    )

    imp_df = pd.DataFrame({
        "feature": features,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return imp_df

def plot_early_detection_curve(
    early_results: pd.DataFrame,
    metric: str = "roc_auc_mean",
    output_path: Optional[str] = None,
) -> None:
    
    plt.figure(figsize=(7, 5))
    plt.plot(
        early_results["trace_fraction"],
        early_results[metric],
        marker="o",
    )
    plt.xlabel("Trace fraction observed")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Early Detection Performance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_feature_ablation(
    ablation_df: pd.DataFrame,
    metric: str = "roc_auc_mean",
    output_path: Optional[str] = None,
) -> None:
    plot_df = ablation_df.sort_values(metric, ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["feature_group"], plot_df[metric])
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Feature group")
    plt.title("Feature Group Ablation")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def existing_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

def validate_experiment_df(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    required_cols: Optional[List[str]] = None,
) -> None:
    required_cols = required_cols or []
    missing = [c for c in [target_col] + features + required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataframe is empty.")

    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col} contains NaN.")

    print("Validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Target: {target_col}")
    print(df[target_col].value_counts(dropna=False))

def make_model(model_kind: str = "rf", seed: int = 42) -> Pipeline:
    if model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_kind == "logreg":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

def compute_binary_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
    }

def evaluate_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    features: List[str],
    model_kind: str = "rf",
    seed: int = 42,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], Pipeline]:
    
    X_train = train_df[features]
    y_train = train_df[target_col].astype(int)

    X_test = test_df[features]
    y_test = test_df[target_col].astype(int)

    model = make_model(model_kind=model_kind, seed=seed)
    model.fit(X_train, y_train)

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    metrics = compute_binary_metrics(y_test, y_pred, y_score)
    return metrics, model

def run_operational_only_leave_one_seed(
    df: pd.DataFrame,
    target_col: str = "is_attack",
    seed_col: str = "seed",
    model_kinds: List[str] = ["rf", "logreg"],
    output_dir: Path = OUTPUT_DIR,
) -> pd.DataFrame:
    
    feature_sets = {
        "all_telemetry": existing_cols(df, ALL_TELEMETRY_FEATURES),
        "operational_only": existing_cols(df, OPERATIONAL_ONLY_FEATURES),
        "semantic_behavioral": existing_cols(df, SEMANTIC_BEHAVIORAL_FEATURES),
    }

    validate_experiment_df(
        df=df,
        target_col=target_col,
        features=feature_sets["operational_only"],
        required_cols=[seed_col],
    )

    rows = []

    for feature_set_name, features in feature_sets.items():
        if not features:
            continue

        for model_kind in model_kinds:
            for heldout_seed in sorted(df[seed_col].unique()):
                train_df = df[df[seed_col] != heldout_seed].copy()
                test_df = df[df[seed_col] == heldout_seed].copy()

                metrics, _ = evaluate_train_test(
                    train_df=train_df,
                    test_df=test_df,
                    target_col=target_col,
                    features=features,
                    model_kind=model_kind,
                    seed=int(heldout_seed),
                )

                rows.append({
                    "experiment": "operational_only_leave_one_seed",
                    "feature_set": feature_set_name,
                    "model": model_kind,
                    "heldout_seed": heldout_seed,
                    "num_features": len(features),
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    **metrics,
                })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "exp1_operational_only_leave_one_seed_raw.csv", index=False)

    summary_df = summarize_results(
        result_df,
        group_cols=["feature_set", "model"],
    )
    summary_df.to_csv(output_dir / "exp1_operational_only_leave_one_seed_summary.csv", index=False)

    return result_df

def plot_exp1_operational_comparison(
    summary_df: pd.DataFrame,
    metric: str = "roc_auc_mean",
    output_path: Path = OUTPUT_DIR / "exp1_operational_only_comparison.png",
):
    plot_df = summary_df.sort_values(metric)

    labels = plot_df["feature_set"] + " / " + plot_df["model"]

    plt.figure(figsize=(8, 5))
    plt.barh(labels, plot_df[metric])
    plt.xlabel(metric.replace("_", " ").title())
    plt.ylabel("Feature set / model")
    plt.title("Operational-only vs Semantic-behavioral Telemetry")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

# def build_hard_benign_dataset(
#     df: pd.DataFrame,
#     target_col: str = "is_attack",
#     quantile: float = 0.70,
#     complexity_features: Optional[List[str]] = None,
# ) -> pd.DataFrame:
#     if complexity_features is None:
#         complexity_features = [
#             "num_messages",
#             "trace_word_length",
#             "total_tokens",
#             "latency_total_s",
#             "num_tool_calls",
#             "unique_tool_calls",
#             "tool_call_entropy",
#         ]

#     complexity_features = existing_cols(df, complexity_features)

#     if not complexity_features:
#         raise ValueError("No complexity features found in dataframe.")

#     work_df = df.copy()

#     # Normalização simples por ranking percentil para criar um score de complexidade.
#     for col in complexity_features:
#         work_df[f"{col}_rank"] = work_df[col].rank(pct=True)

#     rank_cols = [f"{col}_rank" for col in complexity_features]
#     work_df["complexity_score"] = work_df[rank_cols].mean(axis=1)

#     benign_df = work_df[work_df[target_col].astype(int) == 0].copy()
#     attack_df = work_df[work_df[target_col].astype(int) == 1].copy()

#     threshold = benign_df["complexity_score"].quantile(quantile)
#     hard_benign_df = benign_df[benign_df["complexity_score"] >= threshold].copy()

#     hard_df = pd.concat([hard_benign_df, attack_df], axis=0).sample(frac=1, random_state=42)

#     print("Original dataset:")
#     print(work_df[target_col].value_counts())

#     print("\nHard benign dataset:")
#     print(hard_df[target_col].value_counts())

#     print(f"\nHard benign threshold: {threshold:.4f}")
#     print(f"Hard benign count: {len(hard_benign_df)} / {len(benign_df)}")

#     return hard_df

# def run_hard_benign_leave_one_seed(
#     df: pd.DataFrame,
#     target_col: str = "is_attack",
#     seed_col: str = "seed",
#     feature_sets: Optional[Dict[str, List[str]]] = None,
#     model_kind: str = "rf",
#     quantile: float = 0.70,
#     output_dir: Path = OUTPUT_DIR,
# ) -> pd.DataFrame:
#     if feature_sets is None:
#         feature_sets = {
#             "all_telemetry": existing_cols(df, ALL_TELEMETRY_FEATURES),
#             "operational_only": existing_cols(df, OPERATIONAL_ONLY_FEATURES),
#         }

#     hard_df = build_hard_benign_dataset(
#         df=df,
#         target_col=target_col,
#         quantile=quantile,
#     )

#     rows = []

#     for feature_set_name, features in feature_sets.items():
#         if not features:
#             continue

#         validate_experiment_df(
#             df=hard_df,
#             target_col=target_col,
#             features=features,
#             required_cols=[seed_col],
#         )

#         for heldout_seed in sorted(hard_df[seed_col].unique()):
#             train_df = hard_df[hard_df[seed_col] != heldout_seed].copy()
#             test_df = hard_df[hard_df[seed_col] == heldout_seed].copy()

#             metrics, _ = evaluate_train_test(
#                 train_df=train_df,
#                 test_df=test_df,
#                 target_col=target_col,
#                 features=features,
#                 model_kind=model_kind,
#                 seed=int(heldout_seed),
#             )

#             rows.append({
#                 "experiment": "hard_benign_leave_one_seed",
#                 "feature_set": feature_set_name,
#                 "model": model_kind,
#                 "heldout_seed": heldout_seed,
#                 "hard_benign_quantile": quantile,
#                 "num_features": len(features),
#                 "n_train": len(train_df),
#                 "n_test": len(test_df),
#                 "positive_rate_test": test_df[target_col].astype(int).mean(),
#                 **metrics,
#             })

#     result_df = pd.DataFrame(rows)
#     result_df.to_csv(output_dir / "exp2_hard_benign_raw.csv", index=False)

#     summary_df = summarize_results(
#         result_df,
#         group_cols=["feature_set", "model", "hard_benign_quantile"],
#     )
#     summary_df.to_csv(output_dir / "exp2_hard_benign_summary.csv", index=False)

#     return result_df, hard_df

# def plot_complexity_distribution(
#     df: pd.DataFrame,
#     hard_df: pd.DataFrame,
#     target_col: str = "is_attack",
#     output_path: Path = OUTPUT_DIR / "exp2_complexity_distribution.png",
# ):
#     plt.figure(figsize=(7, 5))

#     original_benign = df[df[target_col].astype(int) == 0]
#     hard_benign = hard_df[hard_df[target_col].astype(int) == 0]
#     attacks = df[df[target_col].astype(int) == 1]

#     plt.hist(original_benign["complexity_score"], bins=20, alpha=0.5, label="Original benign")
#     plt.hist(hard_benign["complexity_score"], bins=20, alpha=0.5, label="Hard benign")
#     plt.hist(attacks["complexity_score"], bins=20, alpha=0.5, label="Attacks")

#     plt.xlabel("Complexity score")
#     plt.ylabel("Count")
#     plt.title("Hard Benign Negative Construction")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

# def find_best_thresholds(
#     y_true: np.ndarray,
#     y_score: np.ndarray,
# ) -> pd.DataFrame:
#     thresholds = np.linspace(0.01, 0.99, 99)
#     rows = []

#     for t in thresholds:
#         y_pred = (y_score >= t).astype(int)

#         rows.append({
#             "threshold": t,
#             "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
#             "f1": f1_score(y_true, y_pred, zero_division=0),
#             "precision": precision_score(y_true, y_pred, zero_division=0),
#             "recall": recall_score(y_true, y_pred, zero_division=0),
#         })

#     return pd.DataFrame(rows)

# def run_ipi_threshold_calibration(
#     df: pd.DataFrame,
#     target_col: str = "is_attack",
#     attack_col: str = "attack_type",
#     heldout_attack: str = "IPI",
#     features: Optional[List[str]] = None,
#     model_kind: str = "rf",
#     output_dir: Path = OUTPUT_DIR,
# ):
#     if features is None:
#         features = existing_cols(df, ALL_TELEMETRY_FEATURES)

#     validate_experiment_df(
#         df=df,
#         target_col=target_col,
#         features=features,
#         required_cols=[attack_col],
#     )

#     train_df = df[
#         (df[attack_col] == "benign") |
#         ((df[attack_col] != "benign") & (df[attack_col] != heldout_attack))
#     ].copy()

#     test_df = df[
#         (df[attack_col] == "benign") |
#         (df[attack_col] == heldout_attack)
#     ].copy()

#     X_train = train_df[features]
#     y_train = train_df[target_col].astype(int)

#     X_test = test_df[features]
#     y_test = test_df[target_col].astype(int)

#     model = make_model(model_kind=model_kind, seed=42)
#     model.fit(X_train, y_train)

#     y_score = model.predict_proba(X_test)[:, 1]

#     threshold_df = find_best_thresholds(y_test.values, y_score)

#     best_f1 = threshold_df.loc[threshold_df["f1"].idxmax()].to_dict()
#     best_bacc = threshold_df.loc[threshold_df["balanced_accuracy"].idxmax()].to_dict()

#     default_pred = (y_score >= 0.5).astype(int)
#     default_metrics = compute_binary_metrics(y_test, default_pred, y_score)

#     result_summary = {
#         "heldout_attack": heldout_attack,
#         "model": model_kind,
#         "n_train": len(train_df),
#         "n_test": len(test_df),
#         "roc_auc": roc_auc_score(y_test, y_score),
#         "auprc": average_precision_score(y_test, y_score),
#         "default_threshold": 0.5,
#         **{f"default_{k}": v for k, v in default_metrics.items()},
#         **{f"best_f1_{k}": v for k, v in best_f1.items()},
#         **{f"best_bacc_{k}": v for k, v in best_bacc.items()},
#     }

#     threshold_df.to_csv(output_dir / f"exp3_threshold_curve_{heldout_attack}.csv", index=False)
#     pd.DataFrame([result_summary]).to_csv(output_dir / f"exp3_threshold_summary_{heldout_attack}.csv", index=False)

#     return threshold_df, result_summary, y_test.values, y_score

# def plot_threshold_calibration(
#     threshold_df: pd.DataFrame,
#     output_path: Path = OUTPUT_DIR / "exp3_ipi_threshold_calibration.png",
# ):
#     plt.figure(figsize=(8, 5))
#     plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
#     plt.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="Balanced Accuracy")
#     plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision")
#     plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall")
#     plt.xlabel("Decision threshold")
#     plt.ylabel("Metric")
#     plt.title("Threshold Calibration for Held-out IPI")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

# def plot_ipi_pr_curve(
#     y_true,
#     y_score,
#     output_path: Path = OUTPUT_DIR / "exp3_ipi_precision_recall_curve.png",
# ):
#     precision, recall, _ = precision_recall_curve(y_true, y_score)

#     plt.figure(figsize=(6, 5))
#     plt.plot(recall, precision)
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curve for Held-out IPI")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

# def run_early_detection_from_dataframe(
#     early_df: pd.DataFrame,
#     target_col: str = "is_attack",
#     seed_col: str = "seed",
#     trace_fraction_col: str = "trace_fraction",
#     feature_sets: Optional[Dict[str, List[str]]] = None,
#     model_kind: str = "rf",
#     output_dir: Path = OUTPUT_DIR,
# ) -> pd.DataFrame:
#     if feature_sets is None:
#         feature_sets = {
#             "all_telemetry": existing_cols(early_df, ALL_TELEMETRY_FEATURES),
#             "operational_only": existing_cols(early_df, OPERATIONAL_ONLY_FEATURES),
#         }

#     rows = []

#     for feature_set_name, features in feature_sets.items():
#         if not features:
#             continue

#         validate_experiment_df(
#             df=early_df,
#             target_col=target_col,
#             features=features,
#             required_cols=[seed_col, trace_fraction_col],
#         )

#         for frac in sorted(early_df[trace_fraction_col].dropna().unique()):
#             frac_df = early_df[early_df[trace_fraction_col] == frac].copy()

#             for heldout_seed in sorted(frac_df[seed_col].unique()):
#                 train_df = frac_df[frac_df[seed_col] != heldout_seed].copy()
#                 test_df = frac_df[frac_df[seed_col] == heldout_seed].copy()

#                 metrics, _ = evaluate_train_test(
#                     train_df=train_df,
#                     test_df=test_df,
#                     target_col=target_col,
#                     features=features,
#                     model_kind=model_kind,
#                     seed=int(heldout_seed),
#                 )

#                 rows.append({
#                     "experiment": "early_detection",
#                     "feature_set": feature_set_name,
#                     "model": model_kind,
#                     "trace_fraction": frac,
#                     "heldout_seed": heldout_seed,
#                     "n_train": len(train_df),
#                     "n_test": len(test_df),
#                     **metrics,
#                 })

#     result_df = pd.DataFrame(rows)
#     result_df.to_csv(output_dir / "exp4_early_detection_raw.csv", index=False)

#     summary_df = summarize_results(
#         result_df,
#         group_cols=["feature_set", "model", "trace_fraction"],
#     )
#     summary_df.to_csv(output_dir / "exp4_early_detection_summary.csv", index=False)

#     return result_df

# def plot_early_detection_curve(
#     early_summary,
#     metric="roc_auc_mean",
#     output_path="results/tamas/early_detection_prefix_curve.png",
# ):
#     std_col = metric.replace("_mean", "_std")

#     plt.figure(figsize=(7, 5))

#     plt.plot(
#         early_summary["trace_fraction"],
#         early_summary[metric],
#         marker="o",
#     )

#     if std_col in early_summary.columns:
#         plt.fill_between(
#             early_summary["trace_fraction"],
#             early_summary[metric] - early_summary[std_col],
#             early_summary[metric] + early_summary[std_col],
#             alpha=0.2,
#         )

#     plt.xlabel("Fraction of agent outputs observed")
#     plt.ylabel(metric.replace("_", " ").title())
#     plt.title("Early Detection from Prefix Runtime Telemetry")
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

def make_early_rf(seed=42):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

def run_early_detection_leave_one_seed(
    early_df,
    target_col="is_attack",
    seed_col="seed",
    fraction_col="trace_fraction",
    features=None,
):
    EARLY_FEATURES = [
        c for c in early_df.columns
        if c.endswith("_prefix")
        or c in [
            "prefix_steps",
            "total_steps",
            "observed_step_ratio",
            "num_observed_steps",
            "num_llm_calls_prefix",
            "response_to_prompt_ratio_prefix",
            "tokens_per_latency_prefix",
            "tool_execution_rate_prefix",
        ]
    ]
    
    if features is None:
        features = EARLY_FEATURES

    rows = []
    for frac in sorted(early_df[fraction_col].dropna().unique()):
        frac_df = early_df[early_df[fraction_col] == frac].copy()

        for heldout_seed in sorted(frac_df[seed_col].dropna().unique()):
            train_df = frac_df[frac_df[seed_col] != heldout_seed].copy()
            test_df = frac_df[frac_df[seed_col] == heldout_seed].copy()

            if train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
                print(f"Skipping frac={frac}, seed={heldout_seed}: only one class.")
                continue

            print(features)
            X_train = train_df[features]
            y_train = train_df[target_col].astype(int)

            X_test = test_df[features]
            y_test = test_df[target_col].astype(int)

            model = make_early_rf(seed=int(heldout_seed))
            model.fit(X_train, y_train)

            y_score = model.predict_proba(X_test)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)

            rows.append(
                {
                    "trace_fraction": frac,
                    "heldout_seed": heldout_seed,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "positive_rate_test": y_test.mean(),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, y_score),
                    "auprc": average_precision_score(y_test, y_score),
                }
            )

    return pd.DataFrame(rows)

def safe_sum(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.sum(values)) if values else 0.0

def safe_mean(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else 0.0

def safe_std(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.std(values)) if len(values) > 1 else 0.0

def count_unique(values):
    values = [v for v in values if v is not None and not pd.isna(v)]
    return len(set(values))

def extract_prefix_features(agent_outputs_prefix):

    metrics_list = []
    roles = []
    agent_names = []
    requested_tools = []
    executed_tools = []
    final_answer_lengths = []

    for step in agent_outputs_prefix:
        metrics = step.get("metrics", {}) or {}
        metrics_list.append(metrics)

        roles.append(step.get("role"))
        agent_names.append(step.get("agent_name"))

        requested_tools.append(step.get("requested_tool"))
        executed_tools.append(step.get("executed_tool"))

        final_answer = step.get("final_answer", "") or ""
        final_answer_lengths.append(len(final_answer.split()))

    prompt_tokens = [m.get("prompt_tokens", 0) for m in metrics_list]
    response_tokens = [m.get("response_tokens", 0) for m in metrics_list]
    total_tokens = [m.get("total_tokens", 0) for m in metrics_list]
    total_duration_s = [m.get("total_duration_s", 0) for m in metrics_list]
    eval_duration_s = [m.get("eval_duration_s", 0) for m in metrics_list]
    prompt_eval_duration_s = [m.get("prompt_eval_duration_s", 0) for m in metrics_list]
    tokens_per_second = [m.get("tokens_per_second", 0) for m in metrics_list]
    prompt_eval_count = [m.get("prompt_eval_count", 0) for m in metrics_list]
    eval_count = [m.get("eval_count", 0) for m in metrics_list]
    token_budget = [m.get("token_budget", 0) for m in metrics_list]

    num_requested_tools = sum(x is not None for x in requested_tools)
    num_executed_tools = sum(x is not None for x in executed_tools)

    features = {
        "num_observed_steps": len(agent_outputs_prefix),
        "num_llm_calls_prefix": len(metrics_list),

        "prompt_tokens_prefix_sum": safe_sum(prompt_tokens),
        "response_tokens_prefix_sum": safe_sum(response_tokens),
        "total_tokens_prefix_sum": safe_sum(total_tokens),

        "prompt_tokens_prefix_mean": safe_mean(prompt_tokens),
        "response_tokens_prefix_mean": safe_mean(response_tokens),
        "total_tokens_prefix_mean": safe_mean(total_tokens),

        "prompt_tokens_prefix_std": safe_std(prompt_tokens),
        "response_tokens_prefix_std": safe_std(response_tokens),
        "total_tokens_prefix_std": safe_std(total_tokens),

        "latency_prefix_sum": safe_sum(total_duration_s),
        "latency_prefix_mean": safe_mean(total_duration_s),
        "latency_prefix_std": safe_std(total_duration_s),

        "eval_duration_prefix_sum": safe_sum(eval_duration_s),
        "eval_duration_prefix_mean": safe_mean(eval_duration_s),
        "eval_duration_prefix_std": safe_std(eval_duration_s),

        "prompt_eval_duration_prefix_sum": safe_sum(prompt_eval_duration_s),
        "prompt_eval_duration_prefix_mean": safe_mean(prompt_eval_duration_s),

        "tokens_per_second_prefix_mean": safe_mean(tokens_per_second),
        "tokens_per_second_prefix_std": safe_std(tokens_per_second),
        "tokens_per_second_prefix_min": float(np.min(tokens_per_second)) if tokens_per_second else 0.0,
        "tokens_per_second_prefix_max": float(np.max(tokens_per_second)) if tokens_per_second else 0.0,

        "prompt_eval_count_prefix_sum": safe_sum(prompt_eval_count),
        "eval_count_prefix_sum": safe_sum(eval_count),

        "token_budget_prefix_sum": safe_sum(token_budget),
        "token_budget_prefix_mean": safe_mean(token_budget),

        "num_unique_roles_prefix": count_unique(roles),
        "num_unique_agents_prefix": count_unique(agent_names),

        "num_requested_tools_prefix": num_requested_tools,
        "num_executed_tools_prefix": num_executed_tools,

        "final_answer_words_prefix_sum": safe_sum(final_answer_lengths),
        "final_answer_words_prefix_mean": safe_mean(final_answer_lengths),
        "final_answer_words_prefix_std": safe_std(final_answer_lengths),
    }

    # Razões úteis
    features["response_to_prompt_ratio_prefix"] = (
        features["response_tokens_prefix_sum"] /
        max(features["prompt_tokens_prefix_sum"], 1.0)
    )

    features["tokens_per_latency_prefix"] = (
        features["total_tokens_prefix_sum"] /
        max(features["latency_prefix_sum"], 1e-6)
    )

    features["tool_execution_rate_prefix"] = (
        features["num_executed_tools_prefix"] /
        max(features["num_observed_steps"], 1)
    )

    return features

def build_early_df_from_agent_outputs(
    df,
    fractions=(0.25, 0.50, 0.75, 1.00),
    id_cols=None,
):
    
    if id_cols is None:
        id_cols = [
            "benchmark",
            "architecture",
            "model_name",
            "seed",
            "task_id",
            "scenario",
            "attack_type",
            "is_attack",
            "expected_label",
        ]

    available_id_cols = [c for c in id_cols if c in df.columns]

    rows = []
    for _, episode in df.iterrows():
        agent_outputs = episode.get("agent_outputs", [])

        if not isinstance(agent_outputs, list) or len(agent_outputs) == 0:
            continue

        n_steps = len(agent_outputs)

        metadata = {}
        for col in available_id_cols:
            value = episode[col]

            # Normalizar seed caso venha como lista: [1]
            if col == "seed" and isinstance(value, list):
                value = value[0] if len(value) > 0 else None

            metadata[col] = value

        for frac in fractions:
            k = max(1, int(np.ceil(n_steps * frac)))
            prefix = agent_outputs[:k]

            features = extract_prefix_features(prefix)

            row = {
                **metadata,
                "trace_fraction": frac,
                "prefix_steps": k,
                "total_steps": n_steps,
                "observed_step_ratio": k / n_steps,
                **features,
            }

            rows.append(row)

    early_df = pd.DataFrame(rows)

    return early_df

def validate_early_detection_ready(early_df, target_col="is_attack"):
    print("Rows:", len(early_df))
    print("Fractions:")
    print(early_df["trace_fraction"].value_counts().sort_index())

    print("\nTarget distribution:")
    print(early_df[target_col].value_counts(dropna=False))

    n_classes = early_df[target_col].nunique()

    if n_classes < 2:
        print(
            "\nEste early_df ainda NÃO permite treinar um detector, "
            "porque possui apenas uma classe."
        )
        print(
            "Você precisa carregar também arquivos de ataques, além dos benignos."
        )
    else:
        print("\nEste early_df permite treinar e avaliar early detection.")

def inspect_tamas_trace_structure(df):
    print("Number of episodes:", len(df))

    if "is_attack" in df.columns:
        print("\nClass distribution:")
        print(df["is_attack"].value_counts(dropna=False))

    if "attack_type" in df.columns:
        print("\nAttack type distribution:")
        print(df["attack_type"].value_counts(dropna=False))

    if "agent_outputs" not in df.columns:
        print("\nNo agent_outputs column found.")
        return

#     lengths = df["agent_outputs"].apply(lambda x: len(x) if isinstance(x, list) else 0)

#     print("\nAgent outputs per episode:")
#     print(lengths.describe())

#     print("\nUnique trace lengths:")
#     print(lengths.value_counts().sort_index())

#     first_trace = df.iloc[0]["agent_outputs"]

#     print("\nFirst episode trace length:", len(first_trace))

#     for i, step in enumerate(first_trace):
#         print("\nStep:", i)
#         print("agent_id:", step.get("agent_id"))
#         print("agent_name:", step.get("agent_name"))
#         print("role:", step.get("role"))

#         metrics = step.get("metrics", {})
#         print("metrics keys:", list(metrics.keys()))

def load_tamas_jsonl(path):

    path = Path(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)

def validate_early_df_for_attack_analysis(
    df,
    target_col="is_attack",
    attack_col="attack_type",
    seed_col="seed",
    fraction_col="trace_fraction",
    features=None,
):
    features = features or []

    required = [target_col, attack_col, seed_col, fraction_col]
    missing_required = [c for c in required if c not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_features = [c for c in features if c not in df.columns]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    print("Validation passed.")
    print("Rows:", len(df))

    print("\nTrace fractions:")
    print(df[fraction_col].value_counts(dropna=False).sort_index())

    print("\nAttack types:")
    print(df[attack_col].value_counts(dropna=False))

    print("\nTarget distribution:")
    print(df[target_col].value_counts(dropna=False))

    print("\nSeeds:")
    print(df[seed_col].value_counts(dropna=False))

def compute_metrics(y_true, y_pred, y_score):
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["auprc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["auprc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1),
        }
    )

    return metrics

def run_early_detection_by_attack_type(
    early_df,
    features,
    attack_col="attack_type",
    seed_col="seed",
    fraction_col="trace_fraction",
    benign_label="benign",
    model_kind="rf",
    threshold=0.5,
):

    early_df["seed"] = [s[0] if isinstance(s, list) else s for s in early_df["seed"]]

    attack_types = [
        a for a in sorted(early_df[attack_col].dropna().unique())
        if a != benign_label
    ]

    fractions = sorted(early_df[fraction_col].dropna().unique())

    rows = []
    for attack_type in attack_types:
        for frac in fractions:
            frac_df = early_df[early_df[fraction_col] == frac].copy()

            # Subset binário: benign vs ataque específico
            subset_df = frac_df[
                (frac_df[attack_col] == benign_label) |
                (frac_df[attack_col] == attack_type)
            ].copy()

            if subset_df.empty:
                continue

            subset_df["attack_specific_label"] = (
                subset_df[attack_col] == attack_type
            ).astype(int)

            for heldout_seed in sorted(subset_df[seed_col].dropna().unique()):
                train_df = subset_df[subset_df[seed_col] != heldout_seed].copy()
                test_df = subset_df[subset_df[seed_col] == heldout_seed].copy()

                if train_df["attack_specific_label"].nunique() < 2:
                    print(
                        f"Skipping attack={attack_type}, frac={frac}, seed={heldout_seed}: "
                        "train has only one class."
                    )
                    continue

                if test_df["attack_specific_label"].nunique() < 2:
                    print(
                        f"Skipping attack={attack_type}, frac={frac}, seed={heldout_seed}: "
                        "test has only one class."
                    )
                    continue

                X_train = train_df[features]
                y_train = train_df["attack_specific_label"].astype(int)

                X_test = test_df[features]
                y_test = test_df["attack_specific_label"].astype(int)

                model = make_model(model_kind=model_kind, seed=int(heldout_seed))
                model.fit(X_train, y_train)

                y_score = model.predict_proba(X_test)[:, 1]
                y_pred = (y_score >= threshold).astype(int)

                metrics = compute_metrics(y_test, y_pred, y_score)

                rows.append(
                    {
                        "attack_type": attack_type,
                        "trace_fraction": frac,
                        "heldout_seed": heldout_seed,
                        "model": model_kind,
                        "threshold": threshold,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "positive_rate_test": y_test.mean(),
                        **metrics,
                    }
                )

    result_df = pd.DataFrame(rows)
    return result_df

def summarize_early_detection_by_attack(results_df):
    metric_cols = [
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "auprc",
        "false_positive_rate",
        "false_negative_rate",
    ]

    summary = (
        results_df
        .groupby(["attack_type", "trace_fraction"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary

# def classify_detection_timing(
#     summary_df,
#     metric="roc_auc_mean",
#     threshold=0.85,
# ):
#     rows = []

#     for attack_type, group in summary_df.groupby("attack_type"):
#         group = group.sort_values("trace_fraction")

#         reached = group[group[metric] >= threshold]

#         if reached.empty:
#             detection_stage = "hard_or_unstable"
#             first_detectable_fraction = np.nan
#             best_fraction = group.loc[group[metric].idxmax(), "trace_fraction"]
#             best_score = group[metric].max()
#         else:
#             first_detectable_fraction = reached["trace_fraction"].min()
#             best_fraction = group.loc[group[metric].idxmax(), "trace_fraction"]
#             best_score = group[metric].max()

#             if first_detectable_fraction <= 0.50:
#                 detection_stage = "early_detectable"
#             elif first_detectable_fraction <= 0.75:
#                 detection_stage = "mid_detectable"
#             else:
#                 detection_stage = "late_detectable"

#         rows.append(
#             {
#                 "attack_type": attack_type,
#                 "metric": metric,
#                 "threshold": threshold,
#                 "detection_stage": detection_stage,
#                 "first_detectable_fraction": first_detectable_fraction,
#                 "best_fraction": best_fraction,
#                 "best_score": best_score,
#             }
#         )

#     return pd.DataFrame(rows)

# def plot_attack_fraction_heatmap(
#     pivot_df,
#     title,
#     output_path,
#     vmin=0.5,
#     vmax=1.0,
# ):
#     data = pivot_df.values

#     plt.figure(figsize=(8, 5))
#     plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)

#     plt.colorbar(label="Score")

#     plt.xticks(
#         ticks=np.arange(len(pivot_df.columns)),
#         labels=[str(c) for c in pivot_df.columns],
#     )

#     plt.yticks(
#         ticks=np.arange(len(pivot_df.index)),
#         labels=pivot_df.index,
#     )

#     plt.xlabel("Trace fraction observed")
#     plt.ylabel("Attack type")
#     plt.title(title)

#     # Escrever valores nas células
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             value = data[i, j]
#             if not np.isnan(value):
#                 plt.text(
#                     j,
#                     i,
#                     f"{value:.2f}",
#                     ha="center",
#                     va="center",
#                 )

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

def plot_metric_curves_by_attack(
    summary_df,
    metric="roc_auc_mean",
    output_path=OUTPUT_DIR / "curves_by_attack_roc_auc.png",
):
    plt.figure(figsize=(8, 5))

    for attack_type in sorted(summary_df["attack_type"].unique()):
        sub = summary_df[summary_df["attack_type"] == attack_type].sort_values(
            "trace_fraction"
        )

        plt.plot(
            sub["trace_fraction"],
            sub[metric],
            marker="o",
            label=attack_type,
        )

    plt.xlabel("Trace fraction observed")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Early Detection Curves by Attack Type")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def compute_temporal_gain(summary_df, metric="roc_auc_mean"):
    rows = []

    for attack_type, group in summary_df.groupby("attack_type"):
        group = group.set_index("trace_fraction").sort_index()

        def get(frac):
            return group.loc[frac, metric] if frac in group.index else np.nan

        score_25 = get(0.25)
        score_50 = get(0.50)
        score_75 = get(0.75)
        score_100 = get(1.00)

        rows.append(
            {
                "attack_type": attack_type,
                "metric": metric,
                "score_25": score_25,
                "score_50": score_50,
                "score_75": score_75,
                "score_100": score_100,
                "gain_25_to_100": score_100 - score_25,
                "gain_50_to_75": score_75 - score_50,
                "gain_75_to_100": score_100 - score_75,
            }
        )

    return pd.DataFrame(rows)

# def plot_temporal_gain(
#     temporal_gain_df,
#     gain_col="gain_25_to_100",
#     output_path=OUTPUT_DIR / "temporal_gain_25_to_100.png",
# ):
#     plot_df = temporal_gain_df.sort_values(gain_col)

#     plt.figure(figsize=(8, 5))
#     plt.barh(plot_df["attack_type"], plot_df[gain_col])

#     plt.xlabel(gain_col.replace("_", " ").title())
#     plt.ylabel("Attack type")
#     plt.title("Temporal Gain in Detection Performance")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches="tight")
#     plt.show()

def normalize_seed_column(df, seed_col="seed"):

    if seed_col in df.columns:
        df[seed_col] = df[seed_col].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x
        )

    return df

def infer_numeric_features(df):
    metadata_cols = {
        "is_attack",
        "attack_type",
        "akc_phase",
        "seed",
        "task_id",
        "episode_id",
        "benchmark",
        "architecture",
        "model_name",
        "scenario",
        "expected_label",
        "source_file",
        "trace_fraction",
    }

    features = [
        col for col in df.columns
        if col not in metadata_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(features) == 0:
        raise ValueError(
            "Nenhuma feature numérica encontrada. "
            "Verifique se early_df_all foi criado corretamente a partir dos agent_outputs."
        )

    return features

def keep_existing_numeric(df, feature_set):
    return [
        f for f in feature_set
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

def summarize_results(results_df, group_cols):
    metric_cols = [
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "auprc",
        "false_positive_rate",
        "false_negative_rate",
    ]

    metric_cols = [c for c in metric_cols if c in results_df.columns]

    if results_df.empty:
        raise ValueError("results_df está vazio. Nenhum experimento foi executado.")

    summary = (
        results_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary

def run_early_detection_feature_ablation(
    early_df,
    feature_sets,
    target_col="is_attack",
    seed_col="seed",
    fraction_col="trace_fraction",
    model_kind="rf",
    threshold=0.5,
):
    early_df = normalize_seed_column(early_df, seed_col=seed_col).copy()
    early_df[target_col] = early_df[target_col].astype(int)

    required_cols = [target_col, seed_col, fraction_col]
    missing_required = [c for c in required_cols if c not in early_df.columns]

    if missing_required:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_required}")

    rows = []

    for feature_set_name, features in feature_sets.items():
        features = [
            f for f in features
            if f in early_df.columns and pd.api.types.is_numeric_dtype(early_df[f])
        ]

        if len(features) == 0:
            print(f"Skipping {feature_set_name}: conjunto de features vazio.")
            continue

        for frac in sorted(early_df[fraction_col].dropna().unique()):
            frac_df = early_df[early_df[fraction_col] == frac].copy()

            for heldout_seed in sorted(frac_df[seed_col].dropna().unique()):
                train_df = frac_df[frac_df[seed_col] != heldout_seed].copy()
                test_df = frac_df[frac_df[seed_col] == heldout_seed].copy()

                if train_df[target_col].nunique() < 2:
                    print(
                        f"Skipping feature_set={feature_set_name}, frac={frac}, "
                        f"seed={heldout_seed}: train tem apenas uma classe."
                    )
                    continue

                if test_df[target_col].nunique() < 2:
                    print(
                        f"Skipping feature_set={feature_set_name}, frac={frac}, "
                        f"seed={heldout_seed}: test tem apenas uma classe."
                    )
                    continue

                X_train = train_df[features]
                y_train = train_df[target_col].astype(int)

                X_test = test_df[features]
                y_test = test_df[target_col].astype(int)

                model = make_model(model_kind=model_kind, seed=int(heldout_seed))
                model.fit(X_train, y_train)

                y_score = model.predict_proba(X_test)[:, 1]
                y_pred = (y_score >= threshold).astype(int)

                metrics = compute_metrics(y_test, y_pred, y_score)

                rows.append(
                    {
                        "experiment": "early_detection_feature_ablation",
                        "feature_set": feature_set_name,
                        "trace_fraction": frac,
                        "heldout_seed": heldout_seed,
                        "model": model_kind,
                        "threshold": threshold,
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "positive_rate_test": y_test.mean(),
                        "num_features": len(features),
                        **metrics,
                    }
                )

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        raise ValueError(
            "Nenhuma avaliação foi executada. "
            "Verifique se cada seed possui exemplos benignos e ataques."
        )

    return result_df

def plot_ablation_curves(
    summary_df,
    metric="roc_auc_mean",
    output_path=OUTPUT_DIR / "early_detection_ablation_curves_roc_auc.png",
):
    plt.figure(figsize=(9, 6))

    for feature_set in sorted(summary_df["feature_set"].unique()):
        sub = summary_df[summary_df["feature_set"] == feature_set].sort_values(
            "trace_fraction"
        )

        plt.plot(
            sub["trace_fraction"],
            sub[metric],
            marker="o",
            label=feature_set,
        )

    plt.xlabel("Trace fraction observed")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Early Detection Feature Ablations")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

# def build_hard_benign_early_df(
#     early_df,
#     target_col="is_attack",
#     fraction_col="trace_fraction",
#     quantile=0.70,
#     complexity_features=None,
# ):
#     work_df = normalize_seed_column(early_df).copy()
#     work_df[target_col] = work_df[target_col].astype(int)

#     if complexity_features is None:
#         complexity_features = [
#             "total_tokens_prefix_sum",
#             "latency_prefix_sum",
#             "num_llm_calls_prefix",
#             "num_observed_steps",
#             "prompt_tokens_prefix_sum",
#             "response_tokens_prefix_sum",
#             "eval_count_prefix_sum",
#             "final_answer_words_prefix_sum",
#         ]

#     complexity_features = [
#         c for c in complexity_features
#         if c in work_df.columns and pd.api.types.is_numeric_dtype(work_df[c])
#     ]

#     if len(complexity_features) == 0:
#         raise ValueError(
#             "Nenhuma feature válida de complexidade encontrada. "
#             "Defina complexity_features manualmente."
#         )

#     print("Complexity features:", complexity_features)

#     frames = []

#     for frac in sorted(work_df[fraction_col].dropna().unique()):
#         frac_df = work_df[work_df[fraction_col] == frac].copy()

#         for col in complexity_features:
#             frac_df[f"{col}_rank"] = frac_df[col].rank(pct=True)

#         rank_cols = [f"{col}_rank" for col in complexity_features]
#         frac_df["complexity_score"] = frac_df[rank_cols].mean(axis=1)

#         benign_df = frac_df[frac_df[target_col] == 0].copy()
#         attack_df = frac_df[frac_df[target_col] == 1].copy()

#         if benign_df.empty:
#             print(f"Skipping frac={frac}: nenhum benigno.")
#             continue

#         threshold = benign_df["complexity_score"].quantile(quantile)

#         hard_benign_df = benign_df[
#             benign_df["complexity_score"] >= threshold
#         ].copy()

#         frac_hard_df = pd.concat(
#             [hard_benign_df, attack_df],
#             ignore_index=True,
#         )

#         frac_hard_df["hard_benign_quantile"] = quantile
#         frac_hard_df["hard_benign_threshold"] = threshold

#         frames.append(frac_hard_df)

#         print(
#             f"frac={frac}: benign={len(benign_df)}, "
#             f"hard_benign={len(hard_benign_df)}, attacks={len(attack_df)}"
#         )

#     if len(frames) == 0:
#         raise ValueError("Nenhum dataframe hard benign foi criado.")

#     hard_df = pd.concat(frames, ignore_index=True)

#     return hard_df

# def run_early_detection_by_attack_type_safe(
#     early_df,
#     features,
#     target_col="is_attack",
#     attack_col="attack_type",
#     seed_col="seed",
#     fraction_col="trace_fraction",
#     benign_label="benign",
#     model_kind="rf",
#     threshold=0.5,
#     experiment_name="early_by_attack",
# ):
#     early_df = normalize_seed_column(early_df, seed_col=seed_col).copy()

#     required_cols = [target_col, attack_col, seed_col, fraction_col]
#     missing_required = [c for c in required_cols if c not in early_df.columns]

#     if missing_required:
#         raise ValueError(f"Colunas obrigatórias ausentes: {missing_required}")

#     early_df[target_col] = early_df[target_col].astype(int)

#     features = [
#         f for f in features
#         if f in early_df.columns and pd.api.types.is_numeric_dtype(early_df[f])
#     ]

#     if len(features) == 0:
#         raise ValueError(
#             "Nenhuma feature numérica válida encontrada. "
#             "Recrie ALL_EARLY_FEATURES com infer_numeric_features(early_df)."
#         )

#     rows = []

#     attack_types = [
#         a for a in sorted(early_df[attack_col].dropna().unique())
#         if a != benign_label
#     ]

#     fractions = sorted(early_df[fraction_col].dropna().unique())

#     for attack_type in attack_types:
#         for frac in fractions:
#             frac_df = early_df[early_df[fraction_col] == frac].copy()

#             subset_df = frac_df[
#                 (frac_df[attack_col] == benign_label)
#                 | (frac_df[attack_col] == attack_type)
#             ].copy()

#             if subset_df.empty:
#                 continue

#             subset_df["attack_specific_label"] = (
#                 subset_df[attack_col] == attack_type
#             ).astype(int)

#             for heldout_seed in sorted(subset_df[seed_col].dropna().unique()):
#                 train_df = subset_df[subset_df[seed_col] != heldout_seed].copy()
#                 test_df = subset_df[subset_df[seed_col] == heldout_seed].copy()

#                 if train_df["attack_specific_label"].nunique() < 2:
#                     print(
#                         f"Skipping attack={attack_type}, frac={frac}, seed={heldout_seed}: "
#                         "train tem apenas uma classe."
#                     )
#                     continue

#                 if test_df["attack_specific_label"].nunique() < 2:
#                     print(
#                         f"Skipping attack={attack_type}, frac={frac}, seed={heldout_seed}: "
#                         "test tem apenas uma classe."
#                     )
#                     continue

#                 X_train = train_df[features]
#                 y_train = train_df["attack_specific_label"].astype(int)

#                 X_test = test_df[features]
#                 y_test = test_df["attack_specific_label"].astype(int)

#                 model = make_model(model_kind=model_kind, seed=int(heldout_seed))
#                 model.fit(X_train, y_train)

#                 y_score = model.predict_proba(X_test)[:, 1]
#                 y_pred = (y_score >= threshold).astype(int)

#                 metrics = compute_metrics(y_test, y_pred, y_score)

#                 rows.append(
#                     {
#                         "experiment": experiment_name,
#                         "attack_type": attack_type,
#                         "trace_fraction": frac,
#                         "heldout_seed": heldout_seed,
#                         "model": model_kind,
#                         "threshold": threshold,
#                         "n_train": len(train_df),
#                         "n_test": len(test_df),
#                         "positive_rate_test": y_test.mean(),
#                         "num_features": len(features),
#                         **metrics,
#                     }
#                 )

#     result_df = pd.DataFrame(rows)

#     if result_df.empty:
#         raise ValueError(
#             "Nenhuma avaliação foi executada. "
#             "Verifique se há benignos e ataques para cada seed/fração."
#         )

#     return result_df

# def run_hard_benign_attack_feature_ablation(
#     hard_early_df,
#     feature_sets,
#     model_kind="rf",
#     threshold=0.5,
# ):
#     rows = []

#     for feature_set_name, features in feature_sets.items():
#         valid_features = [
#             f for f in features
#             if f in hard_early_df.columns and pd.api.types.is_numeric_dtype(hard_early_df[f])
#         ]

#         if len(valid_features) == 0:
#             print(f"Skipping {feature_set_name}: nenhuma feature válida.")
#             continue

#         result_df = run_early_detection_by_attack_type_safe(
#             early_df=hard_early_df,
#             features=valid_features,
#             model_kind=model_kind,
#             threshold=threshold,
#             experiment_name=f"hard_benign_{feature_set_name}",
#         )

#         result_df["feature_set"] = feature_set_name
#         rows.append(result_df)

#     if len(rows) == 0:
#         raise ValueError("Nenhuma ablação hard benign foi executada.")

#     return pd.concat(rows, ignore_index=True)

# def plot_attack_fraction_heatmap_from_summary(
#     summary_df,
#     metric="roc_auc_mean",
#     feature_set=None,
#     output_path=None,
#     title=None,
#     vmin=0.5,
#     vmax=1.0,
# ):
#     plot_df = summary_df.copy()

#     if feature_set is not None and "feature_set" in plot_df.columns:
#         plot_df = plot_df[plot_df["feature_set"] == feature_set].copy()

#     pivot_df = plot_df.pivot(
#         index="attack_type",
#         columns="trace_fraction",
#         values=metric,
#     )

#     data = pivot_df.values

#     plt.figure(figsize=(8, 5))
#     plt.imshow(data, aspect="auto", vmin=vmin, vmax=vmax)
#     plt.colorbar(label=metric.replace("_", " ").title())

#     plt.xticks(
#         ticks=np.arange(len(pivot_df.columns)),
#         labels=[str(c) for c in pivot_df.columns],
#     )

#     plt.yticks(
#         ticks=np.arange(len(pivot_df.index)),
#         labels=pivot_df.index,
#     )

#     plt.xlabel("Trace fraction observed")
#     plt.ylabel("Attack type")

#     if title is None:
#         title = f"{metric} by Attack Type and Trace Fraction"

#     plt.title(title)

#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             value = data[i, j]
#             if not np.isnan(value):
#                 plt.text(j, i, f"{value:.2f}", ha="center", va="center")

#     plt.tight_layout()

#     if output_path is not None:
#         plt.savefig(output_path, dpi=300, bbox_inches="tight")

#     plt.show()

#     return pivot_df

# def compare_feature_sets_at_fraction(
#     summary_df,
#     fraction=0.75,
#     metric="roc_auc_mean",
# ):
#     sub = summary_df[summary_df["trace_fraction"] == fraction].copy()

#     if "attack_type" in sub.columns and "feature_set" in sub.columns:
#         table = sub.pivot(
#             index="attack_type",
#             columns="feature_set",
#             values=metric,
#         )
#     elif "feature_set" in sub.columns:
#         table = sub.set_index("feature_set")[[metric]]
#     else:
#         table = sub

#     return table

# def classify_detection_timing_grouped(
#     summary_df,
#     metric="roc_auc_mean",
#     threshold=0.85,
#     group_cols=None,
# ):
#     if group_cols is None:
#         if "feature_set" in summary_df.columns:
#             group_cols = ["feature_set", "attack_type"]
#         else:
#             group_cols = ["attack_type"]

#     rows = []

#     for group_key, group in summary_df.groupby(group_cols):
#         group = group.sort_values("trace_fraction")

#         reached = group[group[metric] >= threshold]

#         if reached.empty:
#             detection_stage = "hard_or_unstable"
#             first_detectable_fraction = np.nan
#             best_fraction = group.loc[group[metric].idxmax(), "trace_fraction"]
#             best_score = group[metric].max()
#         else:
#             first_detectable_fraction = reached["trace_fraction"].min()
#             best_fraction = group.loc[group[metric].idxmax(), "trace_fraction"]
#             best_score = group[metric].max()

#             if first_detectable_fraction <= 0.50:
#                 detection_stage = "early_detectable"
#             elif first_detectable_fraction <= 0.75:
#                 detection_stage = "mid_detectable"
#             else:
#                 detection_stage = "late_detectable"

#         if not isinstance(group_key, tuple):
#             group_key = (group_key,)

#         row = {
#             col: val for col, val in zip(group_cols, group_key)
#         }

#         row.update(
#             {
#                 "metric": metric,
#                 "threshold": threshold,
#                 "detection_stage": detection_stage,
#                 "first_detectable_fraction": first_detectable_fraction,
#                 "best_fraction": best_fraction,
#                 "best_score": best_score,
#             }
#         )

#         rows.append(row)

#     return pd.DataFrame(rows)

def infer_telemetry_features(df, metadata_cols):
    features = [
        col for col in df.columns
        if col not in metadata_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(features) == 0:
        raise ValueError(
            "Nenhuma feature numérica encontrada. "
            "Verifique se o dataframe contém colunas de telemetria."
        )

    return features

def compute_multiclass_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }

def summarize_multiclass_results(results_df, group_cols):
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    summary = (
        results_df
        .groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary

def prepare_akc_phase_df(
    df,
    phase_col="akc_phase",
    drop_phases=None,
    min_samples_per_phase=None,
):

    if drop_phases is None:
        drop_phases = []

    work_df = df.copy()
    work_df = work_df[~work_df[phase_col].isin(drop_phases)].copy()
    if min_samples_per_phase is not None:
        counts = work_df[phase_col].value_counts()
        keep_phases = counts[counts >= min_samples_per_phase].index.tolist()
        work_df = work_df[work_df[phase_col].isin(keep_phases)].copy()

    return work_df

def run_akc_phase_leave_one_seed(
    df,
    features,
    phase_col="akc_phase",
    seed_col="seed",
    model_kind="rf",
    experiment_name="akc_phase",
):
    df = normalize_seed_column(df, seed_col=seed_col).copy()

    required_cols = [phase_col, seed_col]
    missing_required = [c for c in required_cols if c not in df.columns]

    if missing_required:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing_required}")

    features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    if len(features) == 0:
        raise ValueError("Nenhuma feature numérica válida encontrada.")

    rows = []
    for heldout_seed in sorted(df[seed_col].dropna().unique()):
        train_df = df[df[seed_col] != heldout_seed].copy()
        test_df = df[df[seed_col] == heldout_seed].copy()

        if train_df[phase_col].nunique() < 2:
            print(f"Skipping seed={heldout_seed}: train tem menos de 2 fases.")
            continue

        if test_df[phase_col].nunique() < 2:
            print(f"Skipping seed={heldout_seed}: test tem menos de 2 fases.")
            continue

        X_train = train_df[features]
        y_train = train_df[phase_col].astype(str)

        X_test = test_df[features]
        y_test = test_df[phase_col].astype(str)

        model = make_model(model_kind=model_kind, seed=int(heldout_seed))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_multiclass_metrics(y_test, y_pred)

        rows.append(
            {
                "experiment": experiment_name,
                "heldout_seed": heldout_seed,
                "model": model_kind,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "num_classes_train": y_train.nunique(),
                "num_classes_test": y_test.nunique(),
                "classes_train": ",".join(sorted(y_train.unique())),
                "classes_test": ",".join(sorted(y_test.unique())),
                **metrics,
            }
        )

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        raise ValueError("Nenhuma avaliação AKC foi executada.")

    return result_df

def collect_akc_phase_predictions_leave_one_seed(
    df,
    features,
    phase_col="akc_phase",
    seed_col="seed",
    model_kind="rf",
    experiment_name="akc_phase_predictions",
):
    df = normalize_seed_column(df, seed_col=seed_col).copy()

    features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    prediction_rows = []
    for heldout_seed in sorted(df[seed_col].dropna().unique()):
        train_df = df[df[seed_col] != heldout_seed].copy()
        test_df = df[df[seed_col] == heldout_seed].copy()

        if train_df[phase_col].nunique() < 2 or test_df[phase_col].nunique() < 2:
            continue

        X_train = train_df[features]
        y_train = train_df[phase_col].astype(str)

        X_test = test_df[features]
        y_test = test_df[phase_col].astype(str)

        model = make_model(model_kind=model_kind, seed=int(heldout_seed))
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        for idx, true_label, pred_label in zip(test_df.index, y_test, y_pred):
            row = {
                "experiment": experiment_name,
                "heldout_seed": heldout_seed,
                "index": idx,
                "true_phase": true_label,
                "pred_phase": pred_label,
            }

            for meta_col in ["attack_type", "is_attack", "task_id", "scenario"]:
                if meta_col in test_df.columns:
                    row[meta_col] = test_df.loc[idx, meta_col]

            prediction_rows.append(row)

    pred_df = pd.DataFrame(prediction_rows)

    if pred_df.empty:
        raise ValueError("Nenhuma predição foi coletada.")

    return pred_df

def plot_confusion_matrix_from_predictions(
    pred_df,
    true_col="true_phase",
    pred_col="pred_phase",
    normalize=False,
    output_path=None,
    title="AKC Phase Confusion Matrix",
):
    labels = sorted(
        list(set(pred_df[true_col].unique()) | set(pred_df[pred_col].unique()))
    )

    cm = confusion_matrix(
        pred_df[true_col],
        pred_df[pred_col],
        labels=labels,
    )

    if normalize:
        cm_to_plot = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    else:
        cm_to_plot = cm

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_to_plot, aspect="auto")
    plt.colorbar(label="Normalized count" if normalize else "Count")

    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)

    plt.xlabel("Predicted phase")
    plt.ylabel("True phase")
    plt.title(title)

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            value = cm_to_plot[i, j]
            text = f"{value:.2f}" if normalize else str(int(value))
            plt.text(j, i, text, ha="center", va="center")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return pd.DataFrame(cm_to_plot, index=labels, columns=labels)

def run_akc_phase_leave_one_attack_type_out(
    df,
    features,
    phase_col="akc_phase",
    attack_col="attack_type",
    benign_label="benign",
    model_kind="rf",
    drop_phases=None,
):
    if drop_phases is not None:
        df = df[~df[phase_col].isin(drop_phases)].copy()

    features = [
        f for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    if len(features) == 0:
        raise ValueError("Nenhuma feature numérica válida encontrada.")

    rows = []
    prediction_rows = []

    attack_types = [
        a for a in sorted(df[attack_col].dropna().unique())
        if a != benign_label
    ]

    for heldout_attack in attack_types:
        train_df = df[
            (df[attack_col] == benign_label)
            | ((df[attack_col] != benign_label) & (df[attack_col] != heldout_attack))
        ].copy()

        test_df = df[
            (df[attack_col] == benign_label)
            | (df[attack_col] == heldout_attack)
        ].copy()

        if train_df[phase_col].nunique() < 2:
            print(f"Skipping {heldout_attack}: train tem menos de 2 fases.")
            continue

        if test_df[phase_col].nunique() < 2:
            print(f"Skipping {heldout_attack}: test tem menos de 2 fases.")
            continue

        X_train = train_df[features]
        y_train = train_df[phase_col].astype(str)

        X_test = test_df[features]
        y_test = test_df[phase_col].astype(str)

        model = make_model(model_kind=model_kind, seed=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_multiclass_metrics(y_test, y_pred)

        rows.append(
            {
                "heldout_attack": heldout_attack,
                "true_phase_of_heldout_attack": (
                    test_df[test_df[attack_col] == heldout_attack][phase_col]
                    .mode()
                    .iloc[0]
                ),
                "model": model_kind,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "num_classes_train": y_train.nunique(),
                "num_classes_test": y_test.nunique(),
                "classes_train": ",".join(sorted(y_train.unique())),
                "classes_test": ",".join(sorted(y_test.unique())),
                **metrics,
            }
        )

        for idx, true_label, pred_label in zip(test_df.index, y_test, y_pred):
            pred_row = {
                "heldout_attack": heldout_attack,
                "index": idx,
                "attack_type": test_df.loc[idx, attack_col],
                "true_phase": true_label,
                "pred_phase": pred_label,
            }

            if "seed" in test_df.columns:
                pred_row["seed"] = test_df.loc[idx, "seed"]

            prediction_rows.append(pred_row)

    result_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(prediction_rows)

    if result_df.empty:
        raise ValueError("Nenhum experimento leave-one-attack-type-out foi executado.")

    return result_df, pred_df

def infer_early_akc_features(
    early_df,
    remove_prefix_progress=False,
):
    metadata_cols = {
        "is_attack",
        "attack_type",
        "akc_phase",
        "seed",
        "task_id",
        "episode_id",
        "benchmark",
        "architecture",
        "model_name",
        "scenario",
        "expected_label",
        "source_file",
        "trace_fraction",
    }

    features = [
        c for c in early_df.columns
        if c not in metadata_cols
        and pd.api.types.is_numeric_dtype(early_df[c])
    ]

    if remove_prefix_progress:
        features = [
            f for f in features
            if f not in {
                "prefix_steps",
                "total_steps",
                "observed_step_ratio",
                "num_observed_steps",
            }
        ]

    if len(features) == 0:
        raise ValueError("Nenhuma feature prefix-level válida encontrada.")

    return features

def run_temporal_akc_phase_classification(
    early_df,
    features,
    phase_col="akc_phase",
    seed_col="seed",
    fraction_col="trace_fraction",
    model_kind="rf",
    drop_phases=None,
    experiment_name="temporal_akc",
):
    early_df = normalize_seed_column(early_df, seed_col=seed_col).copy()

    if drop_phases is not None:
        early_df = early_df[~early_df[phase_col].isin(drop_phases)].copy()

    features = [
        f for f in features
        if f in early_df.columns and pd.api.types.is_numeric_dtype(early_df[f])
    ]

    if len(features) == 0:
        raise ValueError("Nenhuma feature válida encontrada.")

    rows = []
    prediction_rows = []
    for frac in sorted(early_df[fraction_col].dropna().unique()):
        frac_df = early_df[early_df[fraction_col] == frac].copy()

        for heldout_seed in sorted(frac_df[seed_col].dropna().unique()):
            train_df = frac_df[frac_df[seed_col] != heldout_seed].copy()
            test_df = frac_df[frac_df[seed_col] == heldout_seed].copy()

            if train_df[phase_col].nunique() < 2:
                print(f"Skipping frac={frac}, seed={heldout_seed}: train tem menos de 2 fases.")
                continue

            if test_df[phase_col].nunique() < 2:
                print(f"Skipping frac={frac}, seed={heldout_seed}: test tem menos de 2 fases.")
                continue

            X_train = train_df[features]
            y_train = train_df[phase_col].astype(str)

            X_test = test_df[features]
            y_test = test_df[phase_col].astype(str)

            model = make_model(model_kind=model_kind, seed=int(heldout_seed))
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics = compute_multiclass_metrics(y_test, y_pred)

            rows.append(
                {
                    "experiment": experiment_name,
                    "trace_fraction": frac,
                    "heldout_seed": heldout_seed,
                    "model": model_kind,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "num_classes_train": y_train.nunique(),
                    "num_classes_test": y_test.nunique(),
                    "num_features": len(features),
                    **metrics,
                }
            )

            for idx, true_label, pred_label in zip(test_df.index, y_test, y_pred):
                pred_row = {
                    "experiment": experiment_name,
                    "trace_fraction": frac,
                    "heldout_seed": heldout_seed,
                    "index": idx,
                    "true_phase": true_label,
                    "pred_phase": pred_label,
                }

                for meta_col in ["attack_type", "is_attack", "task_id"]:
                    if meta_col in test_df.columns:
                        pred_row[meta_col] = test_df.loc[idx, meta_col]

                prediction_rows.append(pred_row)

    result_df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(prediction_rows)

    if result_df.empty:
        raise ValueError("Nenhum experimento temporal AKC foi executado.")

    return result_df, pred_df

def plot_temporal_akc_curves(
    summary_df,
    metric="balanced_accuracy_mean",
    output_path=None,
):
    plt.figure(figsize=(8, 5))

    for experiment in sorted(summary_df["experiment"].unique()):
        sub = summary_df[summary_df["experiment"] == experiment].sort_values(
            "trace_fraction"
        )

        plt.plot(
            sub["trace_fraction"],
            sub[metric],
            marker="o",
            label=experiment,
        )

    plt.xlabel("Trace fraction observed")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title("Temporal AKC Phase Classification")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

def add_akc_phase_from_attack_type(
    df,
    attack_col="attack_type",
    output_col="akc_phase",
):
    if attack_col not in df.columns:
        raise ValueError(
            f"Coluna {attack_col} não encontrada. "
            "Não é possível criar akc_phase sem attack_type."
        )

    akc_mapping = {
        "benign": "benign",

        # Semantic infection: ataques que contaminam entrada, instrução ou identidade sem necessariamente exigir coordenação
        "DPI": "semantic_infection",
        "IPI": "semantic_infection",
        "impersonation": "semantic_infection",

        # Cognitive compromise: ataques que alteram julgamento, consistência ou deliberação
        "byzantine": "cognitive_compromise",
        "contradicting": "cognitive_compromise",

        # Agency propagation: ataques que dependem de propagação/coordenação entre agentes
        "colluding": "agency_propagation",
    }

    df[output_col] = df[attack_col].map(akc_mapping)

    missing = df[df[output_col].isna()][attack_col].unique()

    if len(missing) > 0:
        print("WARNING: Alguns attack_type não foram mapeados:")
        print(missing)
        df[output_col] = df[output_col].fillna("unknown")

    return df

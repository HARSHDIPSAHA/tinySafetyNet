import argparse
import json
import os
from collections import Counter
from typing import Optional
import pandas as pd
import mlflow
import numpy as np


VALID_LABELS = ["happy", "sad", "fear", "neutral", "angry", "disgust", "surprise"]
LABEL_NORMALIZE_MAP = {"surprised": "surprise"}


def read_csv_robust(path: str) -> pd.DataFrame:
    # keep it simple + robust like your validator
    try:
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
    df.columns = df.columns.str.strip()
    return df


def normalize_labels(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace(LABEL_NORMALIZE_MAP)
    )


def label_distribution(df: pd.DataFrame, col: str = "inference_of_emotion") -> dict:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

    labels = normalize_labels(df[col])
    counts = Counter(labels)
    total = sum(counts.values()) or 1

    # include unknowns so you can see schema/data issues as drift too
    dist = {k: v / total for k, v in counts.items()}
    return dist, counts, total


def js_divergence(p: dict, q: dict, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence (base-2), bounded [0, 1].
    """
    keys = sorted(set(p.keys()) | set(q.keys()))
    p_arr = np.array([p.get(k, 0.0) for k in keys], dtype=float) + eps
    q_arr = np.array([q.get(k, 0.0) for k in keys], dtype=float) + eps
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)

    def kl(a, b):
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p_arr, m) + 0.5 * kl(q_arr, m)


def l1_drift(p: dict, q: dict) -> float:
    """
    Total variation * 2 == L1 distance between distributions (bounded [0, 2]).
    """
    keys = set(p.keys()) | set(q.keys())
    return float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys))


def run_drift_check(
    tess_path: str,
    synth_path: str,
    per_label_threshold: float = 0.20,
    js_threshold: float = 0.10,
    min_rows_warn: int = 200,
    out_json: Optional[str] = "drift_report.json",
):
    if not os.path.exists(tess_path):
        raise FileNotFoundError(f"Not found: {tess_path}")
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Not found: {synth_path}")

    tess = read_csv_robust(tess_path)
    synth = read_csv_robust(synth_path)

    p, p_counts, p_n = label_distribution(tess)
    q, q_counts, q_n = label_distribution(synth)

    # per-label absolute share drift (for expected labels)
    per_label_drift = {}
    for lab in VALID_LABELS:
        per_label_drift[lab] = abs(p.get(lab, 0.0) - q.get(lab, 0.0))

    # unknown labels are a *big* signal
    unknown_p = 1.0 - sum(p.get(l, 0.0) for l in VALID_LABELS)
    unknown_q = 1.0 - sum(q.get(l, 0.0) for l in VALID_LABELS)
    unknown_drift = abs(unknown_p - unknown_q)

    # distribution metrics
    js = js_divergence(p, q)
    l1 = l1_drift(p, q)

    alerts = []
    # sample size warnings
    if p_n < min_rows_warn:
        alerts.append(f"WARNING: tess sample size low (n={p_n})")
    if q_n < min_rows_warn:
        alerts.append(f"WARNING: synth sample size low (n={q_n})")

    # thresholds
    high_labels = {k: v for k, v in per_label_drift.items() if v > per_label_threshold}
    if high_labels:
        alerts.append(f"ALERT: per-label drift > {per_label_threshold:.0%} for {list(high_labels.keys())}")

    if unknown_drift > 0.05:
        alerts.append("ALERT: unknown-label share drift is high (check label normalization / new classes)")

    if js > js_threshold:
        alerts.append(f"ALERT: JS divergence {js:.3f} > {js_threshold:.3f}")

    report = {
        "inputs": {"tess_path": tess_path, "synth_path": synth_path},
        "sizes": {"tess_n": p_n, "synth_n": q_n},
        "tess_label_counts": dict(p_counts),
        "synth_label_counts": dict(q_counts),
        "tess_distribution": p,
        "synth_distribution": q,
        "per_label_abs_drift": per_label_drift,
        "unknown_share": {"tess": unknown_p, "synth": unknown_q, "abs_drift": unknown_drift},
        "metrics": {"js_divergence": js, "l1_distance": l1},
        "thresholds": {"per_label": per_label_threshold, "js": js_threshold},
        "alerts": alerts,
    }

    # MLflow logging
    with mlflow.start_run(run_name="TinySafety_Drift_Audit"):
        mlflow.log_metric("js_divergence", js)
        mlflow.log_metric("l1_distance", l1)
        mlflow.log_metric("unknown_label_drift", unknown_drift)
        mlflow.log_metric("tess_n", p_n)
        mlflow.log_metric("synth_n", q_n)

        for lab, d in per_label_drift.items():
            mlflow.log_metric(f"drift_{lab}", d)

        if out_json:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(out_json)

    # Console output (human friendly)
    print("\n=== Drift Summary ===")
    print(f"TESS n={p_n} | SYNTH n={q_n}")
    print(f"JS divergence: {js:.3f}")
    print(f"L1 distance:  {l1:.3f}")
    print(f"Unknown-label drift: {unknown_drift:.3f}")
    print("\nPer-label abs drift:")
    for lab in VALID_LABELS:
        print(f"  {lab:8s} {per_label_drift[lab]:.2%}")

    if alerts:
        print("\n".join(["\n=== Alerts ==="] + alerts))
    else:
        print("\nNo alerts. Drift within thresholds.")

    return report


def main():
    parser = argparse.ArgumentParser(description="TinySafetyNet: label drift detector")
    parser.add_argument("--tess", default="week5/data/tess_emotion_log.csv")
    parser.add_argument("--synth", default="week5/data/synthetic_emotion_inference.csv")
    parser.add_argument("--per_label_threshold", type=float, default=0.20)
    parser.add_argument("--js_threshold", type=float, default=0.10)
    parser.add_argument("--min_rows_warn", type=int, default=200)
    parser.add_argument("--out", default="drift_report.json", help="JSON report output path (set empty to disable)")
    args = parser.parse_args()

    out_json = args.out if args.out and args.out.strip() else None
    run_drift_check(
        tess_path=args.tess,
        synth_path=args.synth,
        per_label_threshold=args.per_label_threshold,
        js_threshold=args.js_threshold,
        min_rows_warn=args.min_rows_warn,
        out_json=out_json,
    )


if __name__ == "__main__":
    main()
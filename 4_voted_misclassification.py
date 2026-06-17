#!/usr/bin/env python
"""
Stage 4: Consensus (voted) misclassification analysis

Builds per-subject consensus labels from the real prediction artifacts created
by:
  - 3_predictions_cnn.py
  - 3_predictions_sfcn.py

A subject is assigned:
  - Voted-TP / Voted-TN / Voted-FP / Voted-FN when at least 80% of
    available predictions agree on the same outcome
  - Mixed otherwise

Output:
  analysis_outputs/voted/
    voted_labels.csv
    all_predictions_long.csv
    voted_subject_summary.csv
    voted_clinical_profiles.csv
    voted_fn_vs_tp_mannwhitney.csv
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


SUBJECT_POOL_CSV = Path(os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv"))
PREDICTION_DIRS = {
    "VoxCNN": Path(os.environ.get("PREDICTIONS_CNN_DIR", "./predictions_cnn")),
    "SFCN": Path(os.environ.get("PREDICTIONS_SFCN_DIR", "./predictions_sfcn")),
}
OUTPUT_DIR = Path(os.environ.get("VOTED_OUTPUT_DIR", "./analysis_outputs/voted"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 5
SUBJ_COL = "SUBJECT"
DEFAULT_AGREEMENT_THRESHOLD = 0.80

CLINICAL_VARS = [
    ("AGE", "Age (years)"),
    ("CDRSB", "CDR-SB"),
    ("TOTSCORE", "ADAS-Cog"),
    ("TOTAL13", "ADAS-Cog 13"),
]

CATEGORY_ORDER = ["Voted-TN", "Voted-FP", "Voted-FN", "Voted-TP"]


def load_subject_pool():
    if not SUBJECT_POOL_CSV.exists():
        raise FileNotFoundError(f"Subject pool not found: {SUBJECT_POOL_CSV}")
    df = pd.read_csv(SUBJECT_POOL_CSV, low_memory=False)
    df = df.drop_duplicates(subset=SUBJ_COL, keep="first").copy()
    return df


def discover_prediction_splits(base_dir):
    split_indices = []
    if not base_dir.exists():
        return split_indices
    for path in sorted(base_dir.glob("split_*")):
        try:
            split_indices.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(split_indices)


def load_prediction_rows():
    rows = []

    for model_name, base_dir in PREDICTION_DIRS.items():
        split_indices = discover_prediction_splits(base_dir)
        if not split_indices:
            print(f"  WARNING: no split_* directories found under {base_dir}")
            continue

        for split_idx in split_indices:
            pred_path = base_dir / f"split_{split_idx}" / "predictions.pkl"
            if not pred_path.exists():
                print(f"  WARNING: missing {pred_path}")
                continue

            with open(pred_path, "rb") as f:
                run_results = pickle.load(f)

            for run_result in run_results:
                run_idx = int(run_result.get("run", -1))
                subjects = run_result.get("subjects", [])
                labels = run_result.get("labels", [])
                preds = run_result.get("preds", [])
                probs = run_result.get("raw_probs", [])

                n = min(len(subjects), len(labels), len(preds), len(probs))
                if n == 0:
                    continue

                for i in range(n):
                    rows.append({
                        "model": model_name,
                        "split": split_idx,
                        "run": run_idx,
                        SUBJ_COL: subjects[i],
                        "true_label": int(labels[i]),
                        "pred_label": int(preds[i]),
                        "p_ad": float(probs[i]),
                        "correct": int(labels[i] == preds[i]),
                    })

    if not rows:
        raise FileNotFoundError(
            "No prediction pickles were found. Run 3_predictions_cnn.py and/or "
            "3_predictions_sfcn.py first."
        )

    return pd.DataFrame(rows)


def _outcome_label(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "Voted-TP"
    if true_label == 0 and pred_label == 0:
        return "Voted-TN"
    if true_label == 1 and pred_label == 0:
        return "Voted-FN"
    if true_label == 0 and pred_label == 1:
        return "Voted-FP"
    return "Mixed"


def assign_voted_category(preds_df, agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD):
    def vote_subject(group):
        true_values = sorted(group["true_label"].dropna().unique())
        outcome_counts = (
            group.apply(lambda row: _outcome_label(row["true_label"], row["pred_label"]), axis=1)
            .value_counts()
        )

        result = {
            "n_predictions": len(group),
            "n_correct": int(group["correct"].sum()),
            "agreement_threshold": agreement_threshold,
            "mean_p_ad": group["p_ad"].mean(),
            "std_p_ad": group["p_ad"].std(ddof=0),
            "models_present": ",".join(sorted(group["model"].astype(str).unique())),
            "splits_present": group["split"].nunique(),
            "runs_present": group[["model", "split", "run"]].drop_duplicates().shape[0],
            "n_voxcnn": int((group["model"] == "VoxCNN").sum()),
            "n_sfcn": int((group["model"] == "SFCN").sum()),
            "has_both_models": bool(set(group["model"].unique()) >= {"VoxCNN", "SFCN"}),
            "true_label": true_values[0] if len(true_values) == 1 else np.nan,
        }

        for category in ["Voted-TN", "Voted-FP", "Voted-FN", "Voted-TP"]:
            result[f"count_{category}"] = int(outcome_counts.get(category, 0))

        if len(true_values) != 1 or outcome_counts.empty:
            result["agreement_fraction"] = np.nan
            result["majority_category"] = "Mixed"
            result["voted_category"] = "Mixed"
            return pd.Series(result)

        category = outcome_counts.idxmax()
        agreement_fraction = outcome_counts.iloc[0] / len(group)
        result["agreement_fraction"] = agreement_fraction
        result["majority_category"] = category
        result["majority_count"] = int(outcome_counts.iloc[0])
        result["minority_count"] = int(len(group) - outcome_counts.iloc[0])
        result["voted_category"] = category
        if agreement_fraction < agreement_threshold:
            result["voted_category"] = "Mixed"
        return pd.Series(result)

    voted = preds_df.groupby(SUBJ_COL, sort=True).apply(vote_subject).reset_index()
    voted["true_group"] = voted["true_label"].map({0: "CN", 1: "AD"})
    return voted


def mannwhitney_rrb(a, b):
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = 1 - (2 * res.statistic) / (n1 * n2)
    return res.statistic, res.pvalue, r


def bonferroni_correct(p_values):
    n = len(p_values)
    if n == 0:
        return []
    return [min(p * n, 1.0) for p in p_values]


def main():
    print("=" * 70)
    print("Stage 4: Voted Misclassification")
    print("=" * 70)

    print("\n[1] Loading subject pool...")
    subject_df = load_subject_pool()
    print(f"    Subjects: {len(subject_df)}")

    print("\n[2] Loading predictions...")
    preds_df = load_prediction_rows()
    print(f"    Prediction rows: {len(preds_df):,}")
    print(f"    Unique subjects: {preds_df[SUBJ_COL].nunique()}")
    preds_df.to_csv(OUTPUT_DIR / "all_predictions_long.csv", index=False)

    print("\n[3] Assigning voted categories...")
    voted_df = assign_voted_category(preds_df, agreement_threshold=DEFAULT_AGREEMENT_THRESHOLD)
    print(f"    Counts (agreement >= {DEFAULT_AGREEMENT_THRESHOLD:.0%}):")
    print(voted_df["voted_category"].value_counts(dropna=False).to_string())

    voted_merged = voted_df.merge(subject_df, on=SUBJ_COL, how="left", suffixes=("", "_meta"))

    voted_df.to_csv(OUTPUT_DIR / "voted_labels.csv", index=False)
    voted_merged.to_csv(OUTPUT_DIR / "voted_subject_summary.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR / 'voted_labels.csv'}")
    print(f"    Saved: {OUTPUT_DIR / 'voted_subject_summary.csv'}")

    print("\n[4] Clinical summary...")
    stats_rows = []
    for col, _ in CLINICAL_VARS:
        if col not in voted_merged.columns:
            continue
        row = {"variable": col}
        values = pd.to_numeric(voted_merged[col], errors="coerce")
        for cat in CATEGORY_ORDER:
            cat_vals = values[voted_merged["voted_category"] == cat].dropna()
            row[cat] = (
                f"{cat_vals.median():.2f} [{cat_vals.quantile(0.25):.2f}–{cat_vals.quantile(0.75):.2f}] (n={len(cat_vals)})"
                if len(cat_vals) > 0 else "NA"
            )
        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / "voted_clinical_profiles.csv", index=False)
    if not stats_df.empty:
        print(stats_df.to_string(index=False))

    print("\n[5] Voted-FN vs Voted-TP tests...")
    test_rows = []
    p_values = []
    for col, _ in CLINICAL_VARS:
        if col not in voted_merged.columns:
            continue
        fn_vals = pd.to_numeric(
            voted_merged.loc[voted_merged["voted_category"] == "Voted-FN", col], errors="coerce"
        ).dropna()
        tp_vals = pd.to_numeric(
            voted_merged.loc[voted_merged["voted_category"] == "Voted-TP", col], errors="coerce"
        ).dropna()
        if len(fn_vals) < 3 or len(tp_vals) < 3:
            continue

        u_stat, p_val, rrb = mannwhitney_rrb(fn_vals.values, tp_vals.values)
        p_values.append(p_val)
        test_rows.append({
            "variable": col,
            "FN_n": len(fn_vals),
            "TP_n": len(tp_vals),
            "FN_median": fn_vals.median(),
            "TP_median": tp_vals.median(),
            "U": u_stat,
            "p_uncorrected": p_val,
            "r_rank_biserial": rrb,
        })

    corrected = bonferroni_correct(p_values)
    for row, p_corr in zip(test_rows, corrected):
        row["p_bonferroni"] = p_corr
        row["significant"] = p_corr < 0.05

    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(OUTPUT_DIR / "voted_fn_vs_tp_mannwhitney.csv", index=False)
    if not test_df.empty:
        print(test_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

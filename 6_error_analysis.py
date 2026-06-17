#!/usr/bin/env python
"""
Stage 6: Biomarker-focused error analysis

Uses the configured prediction outputs, voted labels, and subject pool:
  - prediction directories containing split_*/predictions.pkl
  - voted labels CSV
  - subject pool CSV

Focus:
  1. Biomarker profiles for A beta, pTau217, and APOE
  2. Voted-FN versus Voted-TP comparisons

Output:
  analysis_outputs/error/
    fn_tp_summary_table.csv
    biomarker_group_summary.csv
    biomarker_profiles_table.csv
    biomarker_mannwhitney.csv
    biomarker_apoe_tests.csv
"""

import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


SUBJECT_POOL_CSV = Path(os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv"))
APOERES_CSV = Path(os.environ.get("APOERES_CSV", "./subject_reallocation/apoe_genotype.csv"))
VOTED_LABELS_CSV = Path(os.environ.get("VOTED_LABELS_CSV", "./analysis_outputs/voted/voted_labels.csv"))
PREDICTION_DIRS = {
    "VoxCNN": Path(os.environ.get("PREDICTIONS_CNN_DIR", "./predictions_cnn")),
    "SFCN": Path(os.environ.get("PREDICTIONS_SFCN_DIR", "./predictions_sfcn")),
}
OUTPUT_DIR = Path(os.environ.get("ERROR_OUTPUT_DIR", "./analysis_outputs/error"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUBJECT_POOL_PREFIX = f"{SUBJECT_POOL_CSV.stem}_"

SUBJ_COL = "SUBJECT"
TRUE_COL = "true_label"
PRED_COL = "pred_label"
PROB_COL = "p_ad"

CATEGORY_ORDER = ["Voted-TN", "Voted-FP", "Voted-FN", "Voted-TP"]
PRESENTATION_GROUPS = ["Voted-TN", "Voted-FP", "Voted-FN", "Voted-TP"]


def is_abeta42(name):
    c = name.lower()
    return "42" in c and any(tag in c for tag in ["abeta", "ab", "amyloid"])


def is_abeta40(name):
    c = name.lower()
    return "40" in c and any(tag in c for tag in ["abeta", "ab", "amyloid"])


def is_abeta_ratio(name):
    c = name.lower()
    return (
        (("42" in c and "40" in c) or "ratio" in c or "4240" in c)
        and any(tag in c for tag in ["abeta", "ab", "amyloid"])
    )


def is_ptau217(name):
    c = name.lower()
    return (
        "ptau217" in c
        or "ptau_217" in c
        or "p-tau217" in c
        or "pt217" in c
        or ("t217" in c and "tau" in c)
    ) and "npt217" not in c


def is_apoe(name):
    c = name.lower()
    prefix = SUBJECT_POOL_PREFIX.lower()
    if c.startswith(prefix):
        c = c[len(prefix):]
    # Avoid matching the project/file prefix "...abeta_tau_apoe..." which is
    # present on many non-APOE columns after merging.
    return (
        "_apoe_" in c
        or c.endswith("_apoe")
        or c == "genotype"
        or c.endswith("_genotype")
        or "genotype" in c
    )


def load_subject_pool():
    if not SUBJECT_POOL_CSV.exists():
        raise FileNotFoundError(f"Subject pool not found: {SUBJECT_POOL_CSV}")
    df = pd.read_csv(SUBJECT_POOL_CSV, low_memory=False)
    return df.drop_duplicates(subset=SUBJ_COL, keep="first").copy()


def load_voted_labels():
    if not VOTED_LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Run 4_voted_misclassification.py first. Not found: {VOTED_LABELS_CSV}"
    )
    return pd.read_csv(VOTED_LABELS_CSV)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voted-labels",
        default=str(VOTED_LABELS_CSV),
        help="Path to voted labels CSV.",
    )
    parser.add_argument(
        "--subject-pool",
        default=str(SUBJECT_POOL_CSV),
        help="Path to the subject pool CSV.",
    )
    parser.add_argument(
        "--apoe-csv",
        default=str(APOERES_CSV),
        help="Path to the APOERES genotype CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for Stage 6 outputs.",
    )
    return parser.parse_args()


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
        for split_idx in split_indices:
            pred_path = base_dir / f"split_{split_idx}" / "predictions.pkl"
            if not pred_path.exists():
                continue

            with open(pred_path, "rb") as f:
                run_results = pickle.load(f)

            for run_result in run_results:
                subjects = run_result.get("subjects", [])
                labels = run_result.get("labels", [])
                preds = run_result.get("preds", [])
                probs = run_result.get("raw_probs", [])
                run_idx = int(run_result.get("run", -1))

                n = min(len(subjects), len(labels), len(preds), len(probs))
                for i in range(n):
                    rows.append({
                        "model": model_name,
                        "split": split_idx,
                        "run": run_idx,
                        SUBJ_COL: subjects[i],
                        TRUE_COL: int(labels[i]),
                        PRED_COL: int(preds[i]),
                        PROB_COL: float(probs[i]),
                        "correct": int(labels[i] == preds[i]),
                    })

    if not rows:
        raise FileNotFoundError(
            "No prediction pickles were found. Run the stage-3 prediction scripts first."
        )

    return pd.DataFrame(rows)


def _candidate_score(column_name, measure):
    name = column_name.lower()
    score = 0

    if column_name.startswith(SUBJECT_POOL_PREFIX):
        score -= 5

    if measure == "abeta42":
        if "ab42_f" in name:
            score += 40
        if "ab42_c2n" in name:
            score += 30
        if "ab42" in name:
            score += 10
        if "ab42_ab40" in name or "ratio" in name or "4240" in name:
            score -= 50
        if "pt217" in name:
            score -= 80
    elif measure == "abeta40":
        if "ab40_f" in name:
            score += 40
        if "ab40_c2n" in name:
            score += 30
        if "ab40" in name:
            score += 10
        if "ab42_ab40" in name or "ratio" in name or "4240" in name:
            score -= 50
        if "pt217" in name:
            score -= 80
    elif measure == "abeta_ratio":
        if "ab42_ab40_f" in name:
            score += 50
        if "ab42_ab40_c2n" in name:
            score += 40
        if "ab42_ab40" in name or "ratio" in name or "4240" in name:
            score += 20
        if "pt217" in name:
            score -= 80
    elif measure == "ptau217":
        if name.endswith("_pt217_f") or "_pt217_f" in name:
            score += 50
        if name.endswith("_pt217_c2n") or "_pt217_c2n" in name:
            score += 40
        if "pt217" in name:
            score += 15
        if "ab42" in name:
            score -= 60
    elif measure == "apoe":
        if "apoe" in name:
            score += 40
        if name.endswith("_apoe_c2n") or "_apoe_c2n" in name:
            score += 30
        if "abeta" in name or "pt217" in name:
            score -= 100

    return score


def get_candidate_columns(df, measure):
    candidates = []
    for col in df.columns:
        include = False

        if measure == "abeta42":
            include = is_abeta42(col) and not is_abeta_ratio(col) and not is_ptau217(col)
        elif measure == "abeta40":
            include = is_abeta40(col) and not is_abeta_ratio(col) and not is_ptau217(col)
        elif measure == "abeta_ratio":
            include = is_abeta_ratio(col)
        elif measure == "ptau217":
            include = is_ptau217(col)
        elif measure == "apoe":
            include = is_apoe(col)

        if not include:
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        candidates.append((non_null, _candidate_score(col, measure), col))

    candidates.sort(key=lambda item: (item[1], item[0], item[2]), reverse=True)
    return [col for _, _, col in candidates]


def coalesce_columns(df, columns):
    if not columns:
        return pd.Series([pd.NA] * len(df), index=df.index), pd.Series([pd.NA] * len(df), index=df.index)

    value_series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    source_series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    for col in columns:
        mask = value_series.isna() & df[col].notna()
        if mask.any():
            value_series.loc[mask] = df.loc[mask, col]
            source_series.loc[mask] = col

    return value_series, source_series


def load_apoeres_genotype(subject_df):
    """Return a Series of APOE genotype indexed by SUBJECT, sourced from APOERES.
    Falls back to the C2N plasma column when APOERES has no entry for a subject."""
    if "APOERES_GENOTYPE" in subject_df.columns:
        merged = subject_df[[SUBJ_COL, "APOERES_GENOTYPE"]].copy()
        merged = merged.rename(columns={"APOERES_GENOTYPE": "GENOTYPE"})
        n_covered = merged["GENOTYPE"].notna().sum()
        if n_covered > 0:
            print(f"    [APOE] Subject pool APOERES genotype: {n_covered}/{len(merged)} subjects covered.")
            return merged.set_index(SUBJ_COL)["GENOTYPE"]

    if not APOERES_CSV.exists():
        print(f"    [APOE] APOERES file not found at {APOERES_CSV}, falling back to C2N column.")
        return None
    apoe = pd.read_csv(APOERES_CSV, low_memory=False)
    if "RID" in apoe.columns:
        apoe["SUBJ_ID"] = apoe["RID"].astype(str)
    elif "PTID" in apoe.columns:
        apoe["SUBJ_ID"] = apoe["PTID"].astype(str).str.extract(r"_(\d+)$")[0]
    elif SUBJ_COL in apoe.columns:
        apoe["SUBJ_ID"] = apoe[SUBJ_COL].astype(str).str.extract(r"_(\d+)$")[0]
    else:
        raise ValueError(f"Expected RID, PTID, or {SUBJ_COL} in {APOERES_CSV}")

    subject_keys = subject_df[[SUBJ_COL]].copy()
    if "SUBJ_ID" in subject_df.columns:
        subject_keys["SUBJ_ID"] = subject_df["SUBJ_ID"].astype(str)
    else:
        subject_keys["SUBJ_ID"] = subject_df[SUBJ_COL].astype(str).str.extract(r"_(\d+)$")[0]

    apoe_u = (
        apoe.dropna(subset=["GENOTYPE", "SUBJ_ID"])
        .drop_duplicates(subset="SUBJ_ID", keep="first")[["SUBJ_ID", "GENOTYPE"]]
    )
    # Normalise to E3/E4 style from 3/4 style
    def fmt(g):
        parts = str(g).strip().split("/")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return f"E{parts[0]}/E{parts[1]}"
        return g
    apoe_u["GENOTYPE"] = apoe_u["GENOTYPE"].apply(fmt)
    merged = subject_keys.merge(apoe_u, on="SUBJ_ID", how="left")
    n_covered = merged["GENOTYPE"].notna().sum()
    print(f"    [APOE] APOERES genotype: {n_covered}/{len(merged)} subjects covered.")
    return merged.set_index(SUBJ_COL)["GENOTYPE"]


def build_biomarker_dataframe(subject_df):
    candidate_map = {
        "abeta42": get_candidate_columns(subject_df, "abeta42"),
        "abeta40": get_candidate_columns(subject_df, "abeta40"),
        "abeta_ratio": get_candidate_columns(subject_df, "abeta_ratio"),
        "ptau217": get_candidate_columns(subject_df, "ptau217"),
        "apoe_genotype": get_candidate_columns(subject_df, "apoe"),
    }

    biomarker_df = subject_df[[SUBJ_COL, "GROUP", "SEX", "AGE", "CDRSB", "TOTSCORE", "TOTAL13"]].copy()

    # Inject APOERES genotype as the authoritative APOE source
    apoeres_genotype = load_apoeres_genotype(subject_df)
    if apoeres_genotype is not None:
        biomarker_df = biomarker_df.join(apoeres_genotype, on=SUBJ_COL)
        biomarker_df.rename(columns={"GENOTYPE": "apoe_genotype"}, inplace=True)
        candidate_map["apoe_genotype"] = []  # skip C2N coalesce; column already set

    for feature_name, candidate_columns in candidate_map.items():
        if not candidate_columns:
            # Column already pre-populated (e.g., apoe_genotype from APOERES); skip coalesce.
            biomarker_df[f"{feature_name}_source"] = "APOERES"
            continue

        unified_values, unified_sources = coalesce_columns(subject_df, candidate_columns)
        biomarker_df[feature_name] = unified_values
        biomarker_df[f"{feature_name}_source"] = unified_sources

    for numeric_col in ["abeta42", "abeta40", "abeta_ratio", "ptau217"]:
        if numeric_col in biomarker_df.columns:
            biomarker_df[numeric_col] = pd.to_numeric(biomarker_df[numeric_col], errors="coerce")

    if "apoe_genotype" in biomarker_df.columns:
        biomarker_df["apoe_genotype"] = biomarker_df["apoe_genotype"].astype(str).replace("nan", np.nan)
        biomarker_df["APOE_E4_COUNT"] = biomarker_df["apoe_genotype"].apply(
            lambda value: str(value).count("4") if pd.notna(value) else np.nan
        )
        biomarker_df["APOE4_POSITIVE"] = biomarker_df["APOE_E4_COUNT"].apply(
            lambda value: np.nan if pd.isna(value) else int(value >= 1)
        )
    else:
        biomarker_df["APOE_E4_COUNT"] = np.nan
        biomarker_df["APOE4_POSITIVE"] = np.nan

    return biomarker_df

def add_analysis_groups(merged_df):
    work = merged_df.copy()
    work["analysis_group"] = pd.NA
    work.loc[work["voted_category"].isin(PRESENTATION_GROUPS), "analysis_group"] = work["voted_category"]
    return work


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


def compute_biomarker_profiles(voted_df, biomarker_df):
    merged = voted_df.merge(biomarker_df, on=SUBJ_COL, how="left")

    numeric_features = [col for col in ["abeta42", "abeta40", "abeta_ratio", "ptau217", "APOE4_POSITIVE"] if col in merged.columns]
    stats_rows = []
    for feature in numeric_features:
        row = {"biomarker": feature}
        for cat in CATEGORY_ORDER:
            vals = pd.to_numeric(
                merged.loc[merged["voted_category"] == cat, feature], errors="coerce"
            ).dropna()
            row[f"{cat}_median"] = vals.median() if len(vals) else np.nan
            row[f"{cat}_iqr"] = (
                vals.quantile(0.75) - vals.quantile(0.25) if len(vals) else np.nan
            )
            row[f"{cat}_n"] = len(vals)
        stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)

    mw_rows = []
    p_values = []
    for feature in numeric_features:
        fn_vals = pd.to_numeric(
            merged.loc[merged["voted_category"] == "Voted-FN", feature], errors="coerce"
        ).dropna()
        tp_vals = pd.to_numeric(
            merged.loc[merged["voted_category"] == "Voted-TP", feature], errors="coerce"
        ).dropna()
        if len(fn_vals) < 3 or len(tp_vals) < 3:
            continue

        u_stat, p_val, rrb = mannwhitney_rrb(fn_vals.values, tp_vals.values)
        p_values.append(p_val)
        mw_rows.append({
            "biomarker": feature,
            "FN_n": len(fn_vals),
            "TP_n": len(tp_vals),
            "FN_median": fn_vals.median(),
            "TP_median": tp_vals.median(),
            "U": u_stat,
            "p_uncorrected": p_val,
            "r_rank_biserial": rrb,
        })

    corrected = bonferroni_correct(p_values)
    for row, p_corr in zip(mw_rows, corrected):
        row["p_bonferroni"] = p_corr
        row["significant"] = p_corr < 0.05
    mw_df = pd.DataFrame(mw_rows)

    apoe_rows = []
    if "APOE4_POSITIVE" in merged.columns:
        focus = merged[merged["voted_category"].isin(["Voted-FN", "Voted-TP"])].copy()
        focus = focus[focus["APOE4_POSITIVE"].notna()]
        if not focus.empty:
            contingency = pd.crosstab(focus["voted_category"], focus["APOE4_POSITIVE"])
            contingency = contingency.reindex(
                index=["Voted-FN", "Voted-TP"], columns=[0, 1], fill_value=0
            )
            if contingency.values.sum() > 0:
                odds_ratio, fisher_p = stats.fisher_exact(contingency.values)
                apoe_rows.append({
                    "comparison": "Voted-FN vs Voted-TP",
                    "FN_APOE4_negative": int(contingency.loc["Voted-FN", 0]),
                    "FN_APOE4_positive": int(contingency.loc["Voted-FN", 1]),
                    "TP_APOE4_negative": int(contingency.loc["Voted-TP", 0]),
                    "TP_APOE4_positive": int(contingency.loc["Voted-TP", 1]),
                    "odds_ratio": odds_ratio,
                    "fisher_p": fisher_p,
                })

    return merged, stats_df, mw_df, pd.DataFrame(apoe_rows)


def build_fn_tp_summary(merged_df):
    rows = []

    clinical_features = [
        ("AGE", "Age"),
        ("CDRSB", "CDR-SB"),
        ("TOTSCORE", "ADAS-Cog"),
        ("TOTAL13", "ADAS-Cog 13"),
    ]
    biomarker_features = [
        ("abeta_ratio", "A beta 42/40"),
        ("ptau217", "pTau217"),
        ("APOE4_POSITIVE", "APOE4 positive"),
    ]

    for feature, label in clinical_features + biomarker_features:
        if feature not in merged_df.columns:
            continue
        fn_vals = pd.to_numeric(
            merged_df.loc[merged_df["voted_category"] == "Voted-FN", feature],
            errors="coerce",
        ).dropna()
        tp_vals = pd.to_numeric(
            merged_df.loc[merged_df["voted_category"] == "Voted-TP", feature],
            errors="coerce",
        ).dropna()
        if len(fn_vals) == 0 or len(tp_vals) == 0:
            continue

        row = {
            "feature": feature,
            "label": label,
            "FN_n": len(fn_vals),
            "FN_median": fn_vals.median(),
            "TP_n": len(tp_vals),
            "TP_median": tp_vals.median(),
            "direction_FN_vs_TP": (
                "higher" if fn_vals.median() > tp_vals.median()
                else "lower" if fn_vals.median() < tp_vals.median()
                else "equal"
            ),
        }

        if len(fn_vals) >= 3 and len(tp_vals) >= 3:
            u_stat, p_val, rrb = mannwhitney_rrb(fn_vals.values, tp_vals.values)
            row["U"] = u_stat
            row["p_uncorrected"] = p_val
            row["r_rank_biserial"] = rrb
        else:
            row["U"] = np.nan
            row["p_uncorrected"] = np.nan
            row["r_rank_biserial"] = np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def build_group_summary(merged_df):
    rows = []
    for feature, label in [("abeta_ratio", "A beta 42/40"), ("ptau217", "pTau217"), ("APOE4_POSITIVE", "APOE4 positive")]:
        if feature not in merged_df.columns:
            continue
        for group_name in PRESENTATION_GROUPS:
            vals = pd.to_numeric(
                merged_df.loc[merged_df["analysis_group"] == group_name, feature],
                errors="coerce",
            ).dropna()
            rows.append({
                "feature": feature,
                "label": label,
                "group": group_name,
                "n": len(vals),
                "median": vals.median() if len(vals) else np.nan,
                "iqr": vals.quantile(0.75) - vals.quantile(0.25) if len(vals) else np.nan,
                "mean": vals.mean() if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    global APOERES_CSV, OUTPUT_DIR, SUBJECT_POOL_CSV, SUBJECT_POOL_PREFIX, VOTED_LABELS_CSV
    args = parse_args()
    VOTED_LABELS_CSV = Path(args.voted_labels)
    SUBJECT_POOL_CSV = Path(args.subject_pool)
    APOERES_CSV = Path(args.apoe_csv)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBJECT_POOL_PREFIX = f"{SUBJECT_POOL_CSV.stem}_"

    print("=" * 70)
    print("Stage 6: Error Analysis")
    print("=" * 70)

    print("\n[1] Loading data...")
    subject_df = load_subject_pool()
    voted_df = load_voted_labels()
    preds_df = load_prediction_rows()
    print(f"    Subjects: {len(subject_df)}")
    print(f"    Voted labels: {len(voted_df)}")
    print(f"    Prediction rows: {len(preds_df):,}")
    if "agreement_fraction" in voted_df.columns:
        print(
            f"    Voted label source: {VOTED_LABELS_CSV} | "
            f"mean agreement={voted_df['agreement_fraction'].mean():.3f}"
        )

    print("\n[2] Selecting biomarker columns...")
    biomarker_df = build_biomarker_dataframe(subject_df)
    print("\n[3] Biomarker profile analysis...")
    merged_df, stats_df, mw_df, apoe_df = compute_biomarker_profiles(voted_df, biomarker_df)
    merged_df = add_analysis_groups(merged_df)
    summary_df = build_fn_tp_summary(merged_df)
    group_summary_df = build_group_summary(merged_df)
    summary_df.to_csv(OUTPUT_DIR / "fn_tp_summary_table.csv", index=False)
    group_summary_df.to_csv(OUTPUT_DIR / "biomarker_group_summary.csv", index=False)
    stats_df.to_csv(OUTPUT_DIR / "biomarker_profiles_table.csv", index=False)
    mw_df.to_csv(OUTPUT_DIR / "biomarker_mannwhitney.csv", index=False)
    apoe_df.to_csv(OUTPUT_DIR / "biomarker_apoe_tests.csv", index=False)
    if not summary_df.empty:
        print("\nFN vs TP summary:")
        print(summary_df.to_string(index=False))
    if not group_summary_df.empty:
        print("\nGroup summary:")
        print(group_summary_df.to_string(index=False))
    if not apoe_df.empty:
        print("\nAPOE enrichment:")
        print(apoe_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

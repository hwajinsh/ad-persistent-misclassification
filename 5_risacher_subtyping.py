#!/usr/bin/env python
"""
Stage 5: Risacher-style MRI subtyping

Computes Risacher-style MRI subtypes from:
  - voted labels
  - subject-level cohort metadata
  - a volumetric ROI table

All three input paths are configurable by CLI argument or environment variable.
"""

import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


VOTED_LABELS_CSV = Path(os.environ.get("VOTED_LABELS_CSV", "./analysis_outputs/voted/voted_labels.csv"))
SUBJECT_POOL_CSV = Path(os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv"))
VOLUMES_CSV = Path(os.environ.get("RISACHER_VOLUMES_CSV", "./lab_rotation/volumetric_rois.csv"))
OUTPUT_DIR = Path(os.environ.get("RISACHER_OUTPUT_DIR", "./analysis_outputs/risacher"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJ_COL = "SUBJECT"
LEFT_HIPPO = "ST29SV"
RIGHT_HIPPO = "ST88SV"
ICV_COL = "ST10CV"
CORTICAL_ROIS = ["ST31CV", "ST90CV", "ST40CV", "ST107CV", "ST52CV", "ST119CV", "ST50CV"]

REGRESSION_COEFFS = {
    "HV": {"age": -26.8, "sex_female": 423.0, "field_3T": -58.0, "ICV": 0.0023},
    "CTV": {"age": -2480.0, "sex_female": 12500.0, "field_3T": -3200.0, "ICV": 0.31},
}

def load_subject_pool():
    if not SUBJECT_POOL_CSV.exists():
        raise FileNotFoundError(f"Subject pool not found: {SUBJECT_POOL_CSV}")
    df = pd.read_csv(SUBJECT_POOL_CSV, low_memory=False)
    keep_cols = [c for c in [SUBJ_COL, "SUBJ_ID", "GROUP", "SEX", "AGE", "MRI_FIELD_STR"] if c in df.columns]
    return df[keep_cols].drop_duplicates(subset=SUBJ_COL, keep="first").copy()


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
        "--volumes-csv",
        default=str(VOLUMES_CSV),
        help="Path to the volumetric ROI CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for Risacher outputs.",
    )
    return parser.parse_args()


def load_volumes():
    if not VOLUMES_CSV.exists():
        raise FileNotFoundError(f"Volumes CSV not found: {VOLUMES_CSV}")

    df = pd.read_csv(VOLUMES_CSV, low_memory=False)
    if "PTID" in df.columns:
        df[SUBJ_COL] = df["PTID"].astype(str)
    elif "RID" in df.columns:
        df["SUBJ_ID"] = pd.to_numeric(df["RID"], errors="coerce")
    else:
        raise ValueError("Expected either PTID or RID in volumetric CSV.")

    return df


def select_subject_level_volumes(df):
    """Match descriptive_stats.py by reducing the volumetric table to one scan per subject."""
    df = df.copy()
    if "VISCODE2" in df.columns:
        visit_order = {"bl": 0, "sc": 1}
        df["_vsort"] = df["VISCODE2"].map(visit_order).fillna(99)
        df = df.sort_values("_vsort").drop_duplicates(subset=SUBJ_COL, keep="first")
    else:
        df = df.drop_duplicates(subset=SUBJ_COL, keep="first")
    return df


def sex_to_numeric(series):
    series = series.astype(str).str.upper()
    return series.map({"F": 1.0, "FEMALE": 1.0, "M": 0.0, "MALE": 0.0})


def adjust_volume(df, volume_col, coeffs):
    sex_num = sex_to_numeric(df["SEX"])
    field_3t = (pd.to_numeric(df["MRI_FIELD_STR"], errors="coerce") >= 3.0).astype(float)
    predicted = (
        coeffs["age"] * pd.to_numeric(df["AGE"], errors="coerce")
        + coeffs["sex_female"] * sex_num
        + coeffs["field_3T"] * field_3t
        + coeffs["ICV"] * pd.to_numeric(df["ICV"], errors="coerce")
    )
    return pd.to_numeric(df[volume_col], errors="coerce") - predicted


def assign_risacher_subtypes(df):
    df = df.copy()
    df["adj_HV"] = adjust_volume(df, "HV", REGRESSION_COEFFS["HV"])
    df["adj_CTV"] = adjust_volume(df, "CTV", REGRESSION_COEFFS["CTV"])

    hv_median = df["adj_HV"].median()
    ctv_median = df["adj_CTV"].median()

    def subtype(row):
        low_hv = row["adj_HV"] < hv_median
        low_ctv = row["adj_CTV"] < ctv_median
        if low_hv and low_ctv:
            return "tAD"
        if low_hv and not low_ctv:
            return "LP"
        if not low_hv and low_ctv:
            return "HpSp"
        return "MA"

    df["subtype"] = df.apply(subtype, axis=1)
    return df, hv_median, ctv_median


def cramers_v(contingency_table):
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    min_dim = min(contingency_table.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
    return chi2, p_val, dof, v


def mannwhitney_rrb(a, b):
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = 1 - (2 * res.statistic) / (n1 * n2)
    return res.statistic, res.pvalue, r


def main():
    global OUTPUT_DIR, SUBJECT_POOL_CSV, VOLUMES_CSV, VOTED_LABELS_CSV
    args = parse_args()
    VOTED_LABELS_CSV = Path(args.voted_labels)
    SUBJECT_POOL_CSV = Path(args.subject_pool)
    VOLUMES_CSV = Path(args.volumes_csv)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Stage 5: Risacher Subtyping")
    print("=" * 70)

    print("\n[1] Loading voted labels and subject metadata...")
    voted_df = load_voted_labels()
    subject_df = load_subject_pool()
    if "agreement_fraction" in voted_df.columns:
        print(
            f"    Using voted labels from {VOTED_LABELS_CSV} "
            f"(thresholded categories available)."
        )

    print("\n[2] Loading volumetric data...")
    volumes_df = load_volumes()
    available_roi_cols = [LEFT_HIPPO, RIGHT_HIPPO, ICV_COL] + CORTICAL_ROIS
    missing = [col for col in available_roi_cols if col not in volumes_df.columns]
    if missing:
        raise ValueError(f"Missing ROI columns in {VOLUMES_CSV}: {missing}")

    volumes_df = select_subject_level_volumes(volumes_df)
    if SUBJ_COL in volumes_df.columns:
        merged = volumes_df.merge(subject_df, on=SUBJ_COL, how="inner")
    else:
        merged = volumes_df.merge(subject_df, on="SUBJ_ID", how="inner")

    merged["HV"] = pd.to_numeric(merged[LEFT_HIPPO], errors="coerce") + pd.to_numeric(
        merged[RIGHT_HIPPO], errors="coerce"
    )
    merged["CTV"] = sum(pd.to_numeric(merged[col], errors="coerce") for col in CORTICAL_ROIS)
    merged["ICV"] = pd.to_numeric(merged[ICV_COL], errors="coerce")

    merged = merged.merge(voted_df[[SUBJ_COL, "voted_category"]], on=SUBJ_COL, how="inner")
    merged = merged.dropna(subset=["HV", "CTV", "ICV", "AGE", "SEX", "MRI_FIELD_STR", "GROUP"]).copy()
    merged["true_label"] = (merged["GROUP"] == "AD").astype(int)

    ad_df = merged[merged["true_label"] == 1].copy()
    print(f"    AD subjects with ROI data: {len(ad_df)}")
    if ad_df.empty:
        raise ValueError("No AD subjects with ROI data were found after merging.")

    print("\n[3] Assigning Risacher subtypes...")
    ad_df, hv_median, ctv_median = assign_risacher_subtypes(ad_df)
    ad_df.to_csv(OUTPUT_DIR / "risacher_subtypes.csv", index=False)
    print(ad_df["subtype"].value_counts().to_string())

    print("\n[4] Composition by voted category...")
    focus = ad_df[ad_df["voted_category"].isin(["Voted-TP", "Voted-FN"])].copy()
    composition = focus.groupby(["voted_category", "subtype"]).size().unstack(fill_value=0)
    for subtype in ["tAD", "LP", "HpSp", "MA"]:
        if subtype not in composition.columns:
            composition[subtype] = 0
    composition = composition[["tAD", "LP", "HpSp", "MA"]]
    proportions = composition.div(composition.sum(axis=1), axis=0)
    chi2, p_val, dof, cramers = cramers_v(composition)

    summary_df = proportions.copy()
    summary_df["chi2"] = chi2
    summary_df["p_chi2"] = p_val
    summary_df["cramers_v"] = cramers
    summary_df.to_csv(OUTPUT_DIR / "subtype_composition_table.csv")
    print(composition.to_string())
    print(f"\n    chi2={chi2:.3f}, p={p_val:.4f}, dof={dof}, Cramer's V={cramers:.3f}")

    print("\n[5] HV:CTV ratio comparison...")
    ad_df["hv_ctv_ratio"] = ad_df["adj_HV"] / (ad_df["adj_CTV"] + 1e-8)
    fn_ratio = ad_df.loc[ad_df["voted_category"] == "Voted-FN", "hv_ctv_ratio"].dropna()
    tp_ratio = ad_df.loc[ad_df["voted_category"] == "Voted-TP", "hv_ctv_ratio"].dropna()
    if len(fn_ratio) >= 3 and len(tp_ratio) >= 3:
        u_stat, p_ratio, rrb = mannwhitney_rrb(fn_ratio.values, tp_ratio.values)
        print(
            f"    FN median={fn_ratio.median():.4f}, TP median={tp_ratio.median():.4f}, "
            f"U={u_stat:.1f}, p={p_ratio:.4f}, r={rrb:.3f}"
        )
    else:
        print("    Not enough data for FN vs TP ratio comparison.")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Descriptive Statistics — Pre-classification summary tables for AD vs CN and atrophy subtypes.

Outputs three blocks of tables:

  Block 1: Overall demographics (AD vs CN)
    - N, age (mean ± SD, median, range), sex composition, age × sex cross-tab

  Block 2: Atrophy subtyping of AD subjects
    - Risacher quadrant assignment (tAD / LP / HpSp / MA)
    - Uses ROI volumes from the configured volumetric CSV

  Block 3: Per-subtype demographics
    - N, age (mean ± SD), sex composition, age × sex cross-tab

All tables are saved as CSV files in analysis_outputs/descriptive/.

Sources:
  configured subject pool CSV
  configured volumetric ROI CSV
"""

from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SUBJECT_POOL_CSV = Path(os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv"))
VOLUMES_CSV      = Path(os.environ.get("RISACHER_VOLUMES_CSV", "./lab_rotation/volumetric_rois.csv"))
OUTPUT_DIR       = Path(os.environ.get("DESCRIPTIVE_OUTPUT_DIR", "./analysis_outputs/descriptive"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Risacher ROI constants (identical to 5_risacher_subtyping.py)
# ---------------------------------------------------------------------------
LEFT_HIPPO    = "ST29SV"
RIGHT_HIPPO   = "ST88SV"
ICV_COL       = "ST10CV"
CORTICAL_ROIS = ["ST31CV", "ST90CV", "ST40CV", "ST107CV", "ST52CV", "ST119CV", "ST50CV"]

REGRESSION_COEFFS = {
    "HV":  {"age": -26.8,   "sex_female": 423.0,   "field_3T": -58.0,  "ICV": 0.0023},
    "CTV": {"age": -2480.0, "sex_female": 12500.0,  "field_3T": -3200.0, "ICV": 0.31},
}

SUBTYPE_ORDER = ["tAD", "LP", "HpSp", "MA"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def age_stats(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n":      len(s),
        "mean":   s.mean(),
        "sd":     s.std(),
        "median": s.median(),
        "min":    s.min(),
        "max":    s.max(),
    }


def sex_counts(series):
    s = series.astype(str).str.upper().str.strip()
    m = (s == "M").sum()
    f = (s == "F").sum()
    total = m + f
    return {
        "n_male":     int(m),
        "pct_male":   100 * m / total if total else np.nan,
        "n_female":   int(f),
        "pct_female": 100 * f / total if total else np.nan,
    }


def age_x_sex_table(df, group_label):
    d = df.copy()
    d["sex_clean"]  = d["SEX"].astype(str).str.upper().str.strip()
    d["age_decade"] = (pd.to_numeric(d["AGE"], errors="coerce") // 10 * 10).astype("Int64")
    ct = (
        d.groupby(["age_decade", "sex_clean"])
         .size()
         .reset_index(name="n")
    )
    ct["group"] = group_label
    return ct


def sex_to_numeric(series):
    s = series.astype(str).str.upper().str.strip()
    return s.map({"F": 1.0, "FEMALE": 1.0, "M": 0.0, "MALE": 0.0})


def adjust_volume(df, volume_col, coeffs):
    predicted = (
        coeffs["age"]        * pd.to_numeric(df["AGE"],           errors="coerce")
        + coeffs["sex_female"] * sex_to_numeric(df["SEX"])
        + coeffs["field_3T"]   * (pd.to_numeric(df["MRI_FIELD_STR"], errors="coerce") >= 3.0).astype(float)
        + coeffs["ICV"]        * pd.to_numeric(df["ICV"],            errors="coerce")
    )
    return pd.to_numeric(df[volume_col], errors="coerce") - predicted


def assign_subtypes(ad_df):
    df = ad_df.copy()
    df["adj_HV"]  = adjust_volume(df, "HV",  REGRESSION_COEFFS["HV"])
    df["adj_CTV"] = adjust_volume(df, "CTV", REGRESSION_COEFFS["CTV"])

    hv_med  = df["adj_HV"].median()
    ctv_med = df["adj_CTV"].median()

    def _subtype(row):
        low_hv  = row["adj_HV"]  < hv_med
        low_ctv = row["adj_CTV"] < ctv_med
        if   low_hv and     low_ctv: return "tAD"
        elif low_hv and not low_ctv: return "LP"
        elif not low_hv and low_ctv: return "HpSp"
        else:                        return "MA"

    df["subtype"] = df.apply(_subtype, axis=1)
    return df, hv_med, ctv_med


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Descriptive Statistics — Pre-classification")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load subject pool
    # -----------------------------------------------------------------------
    if not SUBJECT_POOL_CSV.exists():
        raise FileNotFoundError(f"Subject pool not found: {SUBJECT_POOL_CSV}")

    pool = pd.read_csv(SUBJECT_POOL_CSV, low_memory=False)
    pool = pool[pool["GROUP"].isin(["AD", "CN"])].drop_duplicates(
        subset="SUBJECT", keep="first"
    ).copy()
    print(f"\nLoaded {len(pool)} subjects after dedup "
          f"({(pool['GROUP']=='AD').sum()} AD / {(pool['GROUP']=='CN').sum()} CN)\n")

    # -----------------------------------------------------------------------
    # Block 1: Overall demographics
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("BLOCK 1 — Overall demographics (AD vs CN)")
    print("=" * 70)

    demo_rows = []
    age_sex_frames = []

    for grp in ["AD", "CN"]:
        sub = pool[pool["GROUP"] == grp]
        a   = age_stats(sub["AGE"])
        s   = sex_counts(sub["SEX"])

        row = {"group": grp, "n": len(sub)}
        row.update(a)
        row.update(s)
        demo_rows.append(row)

        print(f"\n  {grp} (n={len(sub)})")
        print(f"    Age:  {a['mean']:.1f} ± {a['sd']:.1f}  "
              f"[median {a['median']:.1f}, range {a['min']:.0f}–{a['max']:.0f}]")
        print(f"    Sex:  {s['n_male']} M ({s['pct_male']:.1f}%)  "
              f"/ {s['n_female']} F ({s['pct_female']:.1f}%)")

        age_sex_frames.append(age_x_sex_table(sub, grp))

    demo_df = pd.DataFrame(demo_rows)
    demo_df.to_csv(OUTPUT_DIR / "demographics_overall.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'demographics_overall.csv'}")

    age_sex_df = pd.concat(age_sex_frames, ignore_index=True)
    age_sex_df.to_csv(OUTPUT_DIR / "age_x_sex_by_group.csv", index=False)
    print(f"  Saved: {OUTPUT_DIR / 'age_x_sex_by_group.csv'}")

    print("\n  Age-decade × sex cross-tab:")
    ct_wide = (
        age_sex_df
        .pivot_table(index=["group", "age_decade"], columns="sex_clean",
                     values="n", fill_value=0)
        .reset_index()
    )
    print(ct_wide.to_string(index=False))

    # -----------------------------------------------------------------------
    # Block 2: Atrophy subtyping (AD only)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BLOCK 2 — Atrophy subtyping of AD subjects")
    print("=" * 70)

    if not VOLUMES_CSV.exists():
        print(f"\n  WARNING: {VOLUMES_CSV} not found — skipping atrophy subtyping.\n")
        return

    vol = pd.read_csv(VOLUMES_CSV, low_memory=False)
    vol.columns = vol.columns.str.strip('"')

    if "PTID" in vol.columns:
        vol["SUBJECT"] = vol["PTID"].astype(str).str.strip()
    elif "RID" in vol.columns:
        vol["SUBJECT"] = vol["RID"].astype(str).str.strip()
    else:
        print("  ERROR: No PTID or RID column found in volumes file.")
        return

    if "VISCODE2" in vol.columns:
        visit_order = {"bl": 0, "sc": 1}
        vol["_vsort"] = vol["VISCODE2"].map(visit_order).fillna(99)
        vol = vol.sort_values("_vsort").drop_duplicates(subset="SUBJECT", keep="first")
    else:
        vol = vol.drop_duplicates(subset="SUBJECT", keep="first")

    required = [LEFT_HIPPO, RIGHT_HIPPO, ICV_COL] + CORTICAL_ROIS
    missing  = [c for c in required if c not in vol.columns]
    if missing:
        print(f"  ERROR: Missing ROI columns in volumes file: {missing}")
        return

    vol["HV"]  = (pd.to_numeric(vol[LEFT_HIPPO],  errors="coerce")
                + pd.to_numeric(vol[RIGHT_HIPPO], errors="coerce"))
    vol["CTV"] = sum(pd.to_numeric(vol[c], errors="coerce") for c in CORTICAL_ROIS)
    vol["ICV"] = pd.to_numeric(vol[ICV_COL], errors="coerce")

    ad_pool = pool[pool["GROUP"] == "AD"].copy()
    merged  = ad_pool.merge(
        vol[["SUBJECT", "HV", "CTV", "ICV"]],
        on="SUBJECT", how="inner"
    )
    merged  = merged.dropna(subset=["HV", "CTV", "ICV", "AGE", "SEX", "MRI_FIELD_STR"])

    print(f"\n  AD subjects in pool:            {len(ad_pool)}")
    print(f"  AD subjects with ROI data:      {len(merged)}")
    if merged.empty:
        print("  WARNING: No matching subjects — skipping subtyping.")
        return

    subtyped, hv_med, ctv_med = assign_subtypes(merged)
    subtyped.to_csv(OUTPUT_DIR / "risacher_subtypes_ad.csv", index=False)
    print(f"  HV threshold (median adj_HV):   {hv_med:.1f}")
    print(f"  CTV threshold (median adj_CTV): {ctv_med:.1f}")
    print("\n  Subtype counts:")
    vc = subtyped["subtype"].value_counts()
    for st in SUBTYPE_ORDER:
        n   = vc.get(st, 0)
        pct = 100 * n / len(subtyped) if len(subtyped) else np.nan
        print(f"    {st:6s}: {n:4d}  ({pct:.1f}%)")

    print(f"\n  Saved: {OUTPUT_DIR / 'risacher_subtypes_ad.csv'}")

    # -----------------------------------------------------------------------
    # Block 3: Per-subtype demographics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BLOCK 3 — Per-subtype demographics")
    print("=" * 70)

    subtype_demo_rows = []
    subtype_age_sex_frames = []

    for st in SUBTYPE_ORDER:
        sub = subtyped[subtyped["subtype"] == st]
        if sub.empty:
            continue
        a = age_stats(sub["AGE"])
        s = sex_counts(sub["SEX"])

        row = {"subtype": st, "n": len(sub)}
        row.update(a)
        row.update(s)
        subtype_demo_rows.append(row)

        print(f"\n  {st} (n={len(sub)})")
        print(f"    Age:  {a['mean']:.1f} ± {a['sd']:.1f}  "
              f"[median {a['median']:.1f}, range {a['min']:.0f}–{a['max']:.0f}]")
        print(f"    Sex:  {s['n_male']} M ({s['pct_male']:.1f}%)  "
              f"/ {s['n_female']} F ({s['pct_female']:.1f}%)")

        subtype_age_sex_frames.append(age_x_sex_table(sub, st))

    subtype_demo_df = pd.DataFrame(subtype_demo_rows)
    subtype_demo_df.to_csv(OUTPUT_DIR / "demographics_by_subtype.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'demographics_by_subtype.csv'}")

    if subtype_age_sex_frames:
        subtype_age_sex_df = pd.concat(subtype_age_sex_frames, ignore_index=True)
        subtype_age_sex_df.to_csv(OUTPUT_DIR / "age_x_sex_by_subtype.csv", index=False)
        print(f"  Saved: {OUTPUT_DIR / 'age_x_sex_by_subtype.csv'}")

        print("\n  Age-decade × sex cross-tab by subtype:")
        ct_sub_wide = (
            subtype_age_sex_df
            .pivot_table(index=["group", "age_decade"], columns="sex_clean",
                         values="n", fill_value=0)
            .reset_index()
            .rename(columns={"group": "subtype"})
        )
        print(ct_sub_wide.to_string(index=False))

    # -----------------------------------------------------------------------
    # Statistical comparisons between subtypes
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BLOCK 3b — Statistical comparisons across subtypes")
    print("=" * 70)

    age_groups = [
        subtyped.loc[subtyped["subtype"] == st, "AGE"].dropna().values
        for st in SUBTYPE_ORDER
        if not subtyped[subtyped["subtype"] == st].empty
    ]
    if len(age_groups) >= 2:
        h, p_age = stats.kruskal(*age_groups)
        print(f"\n  Age — Kruskal-Wallis H={h:.3f}, p={p_age:.4f}")

    sex_ct = pd.crosstab(
        subtyped["subtype"],
        subtyped["SEX"].str.upper().str.strip()
    )
    if sex_ct.shape[0] > 1 and sex_ct.shape[1] > 1:
        chi2, p_sex, dof, _ = stats.chi2_contingency(sex_ct)
        print(f"  Sex — chi2={chi2:.3f}, dof={dof}, p={p_sex:.4f}")
        sex_ct.to_csv(OUTPUT_DIR / "subtype_sex_crosstab.csv")
        print(f"  Saved: {OUTPUT_DIR / 'subtype_sex_crosstab.csv'}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

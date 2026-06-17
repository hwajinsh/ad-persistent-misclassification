#!/usr/bin/env python
"""
Stage 7: Later-scan follow-up for consistently misclassified subjects

Workflow:
  1. Read voted labels and select subjects in misclassified categories
     (default: Voted-FN and Voted-FP from the 80% consensus output).
  2. Search the configured longitudinal cohort CSV for all later 3T scans from the same subject.
  3. Re-run only the exact model instances that originally evaluated the
     subject out-of-sample.
  4. Summarize first-scan vs later-scan prediction changes for each follow-up timepoint.

Outputs:
  analysis_outputs/later_scan/
    selected_later_scans.csv
    later_scan_followup_long.csv
    later_scan_followup_scan_summary.csv
    later_scan_followup_subject_summary.csv
    later_scan_transition_counts_occurrences.csv
    later_scan_transition_counts_unique_subjects.csv
    later_scan_timepoint_summary.csv
"""

import argparse
import os
import pickle
import warnings
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


BASELINE_SUBJECT_POOL_CSV = Path(os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv"))
MAIN_LONGITUDINAL_CSV = Path(os.environ.get("LONGITUDINAL_CSV", "./subject_reallocation/ADNI_extended_df_merged.csv"))
VOTED_LABELS_CSV = Path(os.environ.get("VOTED_LABELS_CSV", "./analysis_outputs/voted/voted_labels.csv"))
PREDICTION_DIRS = {
    "VoxCNN": Path(os.environ.get("PREDICTIONS_CNN_DIR", "./predictions_cnn")),
    "SFCN": Path(os.environ.get("PREDICTIONS_SFCN_DIR", "./predictions_sfcn")),
}
MODEL_DIRS = {
    "VoxCNN": Path(os.environ.get("MODELS_CNN_DIR", "./models_cnn")),
    "SFCN": Path(os.environ.get("MODELS_SFCN_DIR", "./models_sfcn")),
}
OUTPUT_DIR = Path(os.environ.get("LATER_SCAN_OUTPUT_DIR", "./analysis_outputs/later_scan"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MRI_PATH = os.environ.get("ADNI_PREPROCESSED_MRI_DIR", "./preprocessed_mri")
SUBJ_COL = "SUBJECT"
TRUE_COL = "true_label"
PRED_COL = "pred_label"
PROB_COL = "p_ad"
SFCN_INPUT_SHAPE = (160, 192, 160)
DEFAULT_CATEGORIES = ["Voted-FN", "Voted-FP"]
DEFAULT_AGREEMENT_THRESHOLD = 0.80
GPU = 0 if torch.cuda.is_available() else None
DEVICE = torch.device(f"cuda:{GPU}" if GPU is not None else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voted-labels",
        default=str(VOTED_LABELS_CSV),
        help="Path to voted labels CSV.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated voted categories to follow up, e.g. Voted-FN,Voted-FP.",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=10,
        help="Minimum number of days after the first scan to consider a follow-up.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=-1,
        help="Maximum number of days after the first scan to consider. Set negative to disable.",
    )
    parser.add_argument(
        "--agreement-threshold",
        type=float,
        default=DEFAULT_AGREEMENT_THRESHOLD,
        help="Consensus threshold used to summarize later-scan voted categories.",
    )
    parser.add_argument(
        "--subject-pool",
        default=str(BASELINE_SUBJECT_POOL_CSV),
        help="Path to the baseline subject pool CSV.",
    )
    parser.add_argument(
        "--longitudinal-csv",
        default=str(MAIN_LONGITUDINAL_CSV),
        help="Path to the longitudinal cohort CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for later-scan outputs.",
    )
    return parser.parse_args()


def standardize_id(df):
    if "RID" in df.columns:
        df["SUBJ_ID"] = df["RID"].astype(str)
    elif "PTID" in df.columns:
        df["SUBJ_ID"] = df["PTID"].astype(str).str.extract(r"_(\d+)$")[0]
    elif SUBJ_COL in df.columns:
        df["SUBJ_ID"] = df[SUBJ_COL].astype(str).str.extract(r"_(\d+)$")[0]
    else:
        df["SUBJ_ID"] = None
    return df


def load_subject_pool():
    if not BASELINE_SUBJECT_POOL_CSV.exists():
        raise FileNotFoundError(f"Subject pool not found: {BASELINE_SUBJECT_POOL_CSV}")
    df = pd.read_csv(BASELINE_SUBJECT_POOL_CSV, low_memory=False)
    df["MRI_DATE_dt"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    return df.drop_duplicates(subset=SUBJ_COL, keep="first").copy()


def load_voted_labels(voted_labels_path):
    path = Path(voted_labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline voted labels not found: {path}")
    return pd.read_csv(path)


def load_longitudinal_subjects():
    if not MAIN_LONGITUDINAL_CSV.exists():
        raise FileNotFoundError(f"Longitudinal source not found: {MAIN_LONGITUDINAL_CSV}")

    df = pd.read_csv(MAIN_LONGITUDINAL_CSV, low_memory=False)
    df = standardize_id(df)
    df = df[~df[SUBJ_COL].astype(str).str.startswith("381_S_10")]
    df = df[df["GROUP"].isin(["AD", "CN"])]
    df = df[df["MRI_FIELD_STR"].astype(str).str.startswith("3")]

    consistent = df.groupby("SUBJ_ID")["GROUP"].nunique()
    consistent_subj = consistent[consistent == 1].index
    df = df[df["SUBJ_ID"].isin(consistent_subj)].copy()
    df["MRI_DATE_dt"] = pd.to_datetime(df["MRI_DATE"], errors="coerce")
    return df


def select_later_scans(subject_pool_df, longitudinal_df, min_days, max_days):
    rows = []
    max_days = None if max_days is not None and max_days < 0 else max_days

    longitudinal_df = longitudinal_df.copy()
    longitudinal_df["MRI_DATE_dt"] = pd.to_datetime(longitudinal_df["MRI_DATE_dt"], errors="coerce")

    for _, base_row in subject_pool_df.iterrows():
        subject = base_row[SUBJ_COL]
        baseline_date = pd.to_datetime(base_row["MRI_DATE_dt"], errors="coerce")
        if pd.isna(baseline_date):
            continue

        later = longitudinal_df[longitudinal_df[SUBJ_COL] == subject].copy()
        later = later[later["MRI_DATE_dt"].notna()]
        later = later[later["MRI_DATE_dt"] > baseline_date]
        if "T1" in later.columns and "T1" in base_row:
            later = later[later["T1"] != base_row["T1"]]
        if later.empty:
            continue

        later["days_from_first"] = (later["MRI_DATE_dt"] - baseline_date).dt.days
        later = later[later["days_from_first"] >= min_days]
        if max_days is not None:
            later = later[later["days_from_first"] <= max_days]
        if later.empty:
            continue

        later = later.sort_values(["days_from_first", "MRI_DATE_dt"])
        for followup_index, (_, chosen) in enumerate(later.iterrows(), start=1):
            rows.append({
                SUBJ_COL: subject,
                "GROUP": base_row["GROUP"],
                "first_T1": base_row["T1"],
                "first_MRI_DATE": base_row["MRI_DATE"],
                "later_T1": chosen["T1"],
                "later_IMAGE_ID": chosen.get("IMAGE_ID", pd.NA),
                "later_MRI_DATE": chosen["MRI_DATE"],
                "later_MRI_VISIT": chosen.get("MRI_VISIT", pd.NA),
                "days_from_first": int(chosen["days_from_first"]),
                "followup_index": followup_index,
            })

    return pd.DataFrame(rows)


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
                run_idx = int(run_result.get("run", -1))
                subjects = run_result.get("subjects", [])
                labels = run_result.get("labels", [])
                preds = run_result.get("preds", [])
                probs = run_result.get("raw_probs", [])

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
                        "first_correct": int(labels[i] == preds[i]),
                    })

    if not rows:
        raise FileNotFoundError("No first-scan prediction pickles were found.")

    return pd.DataFrame(rows)


def outcome_label(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "Voted-TP"
    if true_label == 0 and pred_label == 0:
        return "Voted-TN"
    if true_label == 1 and pred_label == 0:
        return "Voted-FN"
    if true_label == 0 and pred_label == 1:
        return "Voted-FP"
    return "Mixed"


def summarize_subject_votes(df, pred_col, prob_col, agreement_threshold):
    rows = []
    for subject, group in df.groupby(SUBJ_COL, sort=True):
        outcome_counts = group.apply(
            lambda row: outcome_label(int(row[TRUE_COL]), int(row[pred_col])), axis=1
        ).value_counts()
        majority = outcome_counts.idxmax() if not outcome_counts.empty else "Mixed"
        agreement_fraction = outcome_counts.iloc[0] / len(group) if len(group) else np.nan
        voted = majority if agreement_fraction >= agreement_threshold else "Mixed"
        rows.append({
            SUBJ_COL: subject,
            "n_predictions": len(group),
            "agreement_fraction": agreement_fraction,
            "majority_category": majority,
            "voted_category": voted,
            "mean_p_ad": pd.to_numeric(group[prob_col], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def summarize_scan_votes(df, pred_col, prob_col, agreement_threshold):
    rows = []
    group_cols = [SUBJ_COL, "later_T1", "days_from_first", "followup_index"]
    for keys, group in df.groupby(group_cols, sort=True):
        subject, later_t1, days_from_first, followup_index = keys
        outcome_counts = group.apply(
            lambda row: outcome_label(int(row[TRUE_COL]), int(row[pred_col])), axis=1
        ).value_counts()
        majority = outcome_counts.idxmax() if not outcome_counts.empty else "Mixed"
        agreement_fraction = outcome_counts.iloc[0] / len(group) if len(group) else np.nan
        voted = majority if agreement_fraction >= agreement_threshold else "Mixed"
        rows.append({
            SUBJ_COL: subject,
            "later_T1": later_t1,
            "days_from_first": days_from_first,
            "followup_index": followup_index,
            "n_predictions": len(group),
            "agreement_fraction": agreement_fraction,
            "majority_category": majority,
            "voted_category": voted,
            "mean_p_ad": pd.to_numeric(group[prob_col], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


def preprocess_voxcnn(image):
    image = image - image.min()
    max_val = image.max()
    if max_val > 0:
        image = image / max_val
    return torch.from_numpy(image[None, ...]).float()


def center_crop_3d(image, target_shape=SFCN_INPUT_SHAPE):
    starts = [(c - t) // 2 for c, t in zip(image.shape, target_shape)]
    return image[
        starts[0]:starts[0] + target_shape[0],
        starts[1]:starts[1] + target_shape[1],
        starts[2]:starts[2] + target_shape[2],
    ]


def preprocess_sfcn(image):
    image = image - image.min()
    mean_val = image.mean()
    if mean_val > 0:
        image = image / mean_val
    image = center_crop_3d(image)
    return torch.from_numpy(image[None, ...]).float()


class LaterScanDataset(Dataset):
    def __init__(self, rows, preprocess_fn):
        self.rows = rows.reset_index(drop=True)
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        path = Path(BASE_MRI_PATH) / row["later_T1"]
        image = load_nifti(path)
        image = self.preprocess_fn(image)
        label = torch.tensor(int(row[TRUE_COL])).long()
        return {
            "image": image,
            "label": label,
            "subject": row[SUBJ_COL],
        }


class ClassificationModel3D(nn.Module):
    def __init__(self, dropout=0.4, dropout2=0.4):
        super().__init__()
        self.Conv_1 = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2 = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3 = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4 = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.dense_1 = nn.Linear(2304, 128)
        self.dense_2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)

    def forward(self, x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = self.Conv_1_mp(x)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = self.Conv_2_mp(x)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = self.Conv_3_mp(x)
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = self.Conv_4_mp(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.dense_1(x))
        x = self.dropout2(x)
        return self.dense_2(x)


class SFCN(nn.Module):
    def __init__(self, channel_number=(32, 64, 128, 256, 256, 64), output_dim=40, dropout=True):
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_ch = 1 if i == 0 else channel_number[i - 1]
            out_ch = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self._conv_layer(in_ch, out_ch, maxpool=True, kernel_size=3, padding=1),
                )
            else:
                self.feature_extractor.add_module(
                    f"conv_{i}",
                    self._conv_layer(in_ch, out_ch, maxpool=False, kernel_size=1, padding=0),
                )

        self.classifier = nn.Sequential()
        self.classifier.add_module("average_pool", nn.AvgPool3d((5, 6, 5)))
        if dropout:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        self.classifier.add_module(
            f"conv_{n_layer}",
            nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1),
        )

    @staticmethod
    def _conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2),
                nn.ReLU(),
            )
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return [F.log_softmax(x, dim=1)]


class SFCNClassifier(nn.Module):
    def __init__(self, feature_extractor, n_features=64, num_classes=2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def build_sfcn_model():
    sfcn = SFCN(channel_number=[32, 64, 128, 256, 256, 64], output_dim=40)
    return SFCNClassifier(sfcn.feature_extractor, n_features=64, num_classes=2)


def load_model(model_name, checkpoint_path):
    if model_name == "VoxCNN":
        model = ClassificationModel3D()
    else:
        model = build_sfcn_model()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    clean_sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())
    model.load_state_dict(clean_sd)
    model.to(DEVICE).eval()
    return model


def run_group_inference(group_df):
    model_name = group_df["model"].iloc[0]
    split_idx = int(group_df["split"].iloc[0])
    run_idx = int(group_df["run"].iloc[0])
    checkpoint = MODEL_DIRS[model_name] / f"split_{split_idx}" / f"run_{run_idx}_model-best.h5"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    preprocess_fn = preprocess_voxcnn if model_name == "VoxCNN" else preprocess_sfcn
    dataset = LaterScanDataset(group_df, preprocess_fn)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = load_model(model_name, checkpoint)

    later_probs, later_preds = [], []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            logits = model(image)
            probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            later_probs.extend(probs.tolist())
            later_preds.extend(preds.tolist())

    out = group_df.copy().reset_index(drop=True)
    out["later_p_ad"] = later_probs
    out["later_pred_label"] = later_preds
    out["later_correct"] = (out["later_pred_label"] == out[TRUE_COL]).astype(int)
    return out


def main():
    global BASELINE_SUBJECT_POOL_CSV, MAIN_LONGITUDINAL_CSV, OUTPUT_DIR
    args = parse_args()
    BASELINE_SUBJECT_POOL_CSV = Path(args.subject_pool)
    MAIN_LONGITUDINAL_CSV = Path(args.longitudinal_csv)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    print("=" * 70)
    print("Stage 7: Later-scan Follow-up")
    print("=" * 70)

    print("\n[1] Loading cohort, voted labels, and first-scan predictions...")
    subject_pool = load_subject_pool()
    voted = load_voted_labels(args.voted_labels)
    first_preds = load_prediction_rows()
    print(f"    Cohort subjects: {len(subject_pool)}")
    print(f"    Voted labels: {len(voted)}")
    print(f"    First-scan prediction rows: {len(first_preds):,}")

    follow_subjects = voted[voted["voted_category"].isin(categories)].copy()
    if follow_subjects.empty:
        raise ValueError(f"No subjects found in categories: {categories}")
    print(f"    Follow-up categories: {categories}")
    print(f"    Candidate subjects: {len(follow_subjects)}")

    print("\n[2] Selecting later scans...")
    longitudinal_df = load_longitudinal_subjects()
    base_subjects = subject_pool[subject_pool[SUBJ_COL].isin(follow_subjects[SUBJ_COL])].copy()
    later_scans = select_later_scans(
        base_subjects,
        longitudinal_df,
        min_days=args.min_days,
        max_days=args.max_days,
    )
    later_scans.to_csv(OUTPUT_DIR / "selected_later_scans.csv", index=False)
    print(f"    Later scans selected: {len(later_scans)}")
    print(f"    Unique subjects with later scans: {later_scans[SUBJ_COL].nunique()}")
    print(f"    Saved: {OUTPUT_DIR / 'selected_later_scans.csv'}")

    if later_scans.empty:
        print("\nNo eligible later scans found. Stopping.")
        return

    print("\n[3] Pairing later scans with the exact out-of-sample first-scan model instances...")
    follow_subjects = follow_subjects.merge(later_scans, on=SUBJ_COL, how="inner")
    follow_rows = first_preds[first_preds[SUBJ_COL].isin(follow_subjects[SUBJ_COL])].copy()
    follow_rows = follow_rows.merge(
        follow_subjects[[SUBJ_COL, "voted_category", "agreement_fraction", "later_T1", "later_IMAGE_ID",
                         "later_MRI_DATE", "later_MRI_VISIT", "days_from_first", "followup_index"]],
        on=SUBJ_COL,
        how="inner",
    )
    follow_rows.rename(
        columns={
            "voted_category": "first_voted_category",
            "agreement_fraction": "first_agreement_fraction",
        },
        inplace=True,
    )
    follow_subjects = follow_subjects.rename(
        columns={
            "voted_category": "first_voted_category",
            "agreement_fraction": "first_agreement_fraction",
        }
    )
    print(f"    Later-scan evaluation rows: {len(follow_rows)}")

    print("\n[4] Running later-scan inference with the same checkpoints...")
    inferred_groups = []
    for (model_name, split_idx, run_idx), group in follow_rows.groupby(["model", "split", "run"], sort=True):
        print(f"    {model_name} | split {split_idx} | run {run_idx} | subjects {len(group)}")
        inferred_groups.append(run_group_inference(group))

    later_long = pd.concat(inferred_groups, ignore_index=True)
    later_long.to_csv(OUTPUT_DIR / "later_scan_followup_long.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR / 'later_scan_followup_long.csv'}")

    print("\n[5] Summarizing first-scan vs later-scan transitions...")
    first_summary = summarize_subject_votes(
        later_long.rename(columns={PRED_COL: "first_pred_label", PROB_COL: "first_p_ad"}),
        pred_col="first_pred_label",
        prob_col="first_p_ad",
        agreement_threshold=args.agreement_threshold,
    ).rename(
        columns={
            "agreement_fraction": "first_eval_agreement_fraction",
            "majority_category": "first_eval_majority_category",
            "voted_category": "first_eval_voted_category",
            "mean_p_ad": "first_eval_mean_p_ad",
        }
    )
    later_scan_summary = summarize_scan_votes(
        later_long.rename(columns={"later_pred_label": "pred_tmp", "later_p_ad": "prob_tmp"}),
        pred_col="pred_tmp",
        prob_col="prob_tmp",
        agreement_threshold=args.agreement_threshold,
    ).rename(
        columns={
            "agreement_fraction": "later_agreement_fraction",
            "majority_category": "later_majority_category",
            "voted_category": "later_voted_category",
            "mean_p_ad": "later_mean_p_ad",
            "n_predictions": "later_n_predictions",
        }
    )
    later_scan_summary.to_csv(OUTPUT_DIR / "later_scan_followup_scan_summary.csv", index=False)

    subject_summary = (
        follow_subjects[[SUBJ_COL, "first_voted_category", "first_agreement_fraction"]]
        .drop_duplicates(subset=SUBJ_COL)
        .merge(first_summary, on=SUBJ_COL, how="left")
    )
    subject_summary = subject_summary.merge(
        later_scan_summary.groupby(SUBJ_COL, as_index=False).agg(
            n_later_scans=("later_T1", "nunique"),
            first_later_days=("days_from_first", "min"),
            last_later_days=("days_from_first", "max"),
            any_correct_later=("later_voted_category", lambda s: int(any(v in {"Voted-TP", "Voted-TN"} for v in s))),
            ever_tp_later=("later_voted_category", lambda s: int(any(v == "Voted-TP" for v in s))),
            ever_tn_later=("later_voted_category", lambda s: int(any(v == "Voted-TN" for v in s))),
            mean_later_p_ad=("later_mean_p_ad", "mean"),
        ),
        on=SUBJ_COL,
        how="left",
    )
    subject_summary["became_correct_later"] = (
        ((subject_summary["first_voted_category"] == "Voted-FN") & (subject_summary["ever_tp_later"] == 1))
        | ((subject_summary["first_voted_category"] == "Voted-FP") & (subject_summary["ever_tn_later"] == 1))
    ).astype(int)
    subject_summary.to_csv(OUTPUT_DIR / "later_scan_followup_subject_summary.csv", index=False)

    transitions_occurrences = (
        later_scan_summary.assign(
            transition=lambda d: d["later_voted_category"].astype(str)
        )
        .merge(
            subject_summary[[SUBJ_COL, "first_voted_category"]],
            on=SUBJ_COL,
            how="left",
        )
        .assign(transition=lambda d: d["first_voted_category"].astype(str) + " -> " + d["transition"].astype(str))
        ["transition"]
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="n_subjects")
    )
    transitions_occurrences.to_csv(
        OUTPUT_DIR / "later_scan_transition_counts_occurrences.csv", index=False
    )

    transitions_unique = (
        later_scan_summary.assign(
            transition=lambda d: d["later_voted_category"].astype(str)
        )
        .merge(
            subject_summary[[SUBJ_COL, "first_voted_category"]],
            on=SUBJ_COL,
            how="left",
        )
        .assign(transition=lambda d: d["first_voted_category"].astype(str) + " -> " + d["transition"].astype(str))
        [[SUBJ_COL, "transition"]]
        .drop_duplicates()
        ["transition"]
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="n_subjects")
    )
    transitions_unique.to_csv(
        OUTPUT_DIR / "later_scan_transition_counts_unique_subjects.csv", index=False
    )

    timepoint_summary = (
        later_scan_summary.groupby("followup_index", as_index=False)
        .agg(
            n_subjects=(SUBJ_COL, "nunique"),
            median_days_from_first=("days_from_first", "median"),
            mean_days_from_first=("days_from_first", "mean"),
            n_voted_tp=("later_voted_category", lambda s: int((s == "Voted-TP").sum())),
            n_voted_tn=("later_voted_category", lambda s: int((s == "Voted-TN").sum())),
            n_voted_fn=("later_voted_category", lambda s: int((s == "Voted-FN").sum())),
            n_voted_fp=("later_voted_category", lambda s: int((s == "Voted-FP").sum())),
            n_mixed=("later_voted_category", lambda s: int((s == "Mixed").sum())),
            mean_later_p_ad=("later_mean_p_ad", "mean"),
            mean_agreement_fraction=("later_agreement_fraction", "mean"),
        )
    )
    timepoint_summary.to_csv(OUTPUT_DIR / "later_scan_timepoint_summary.csv", index=False)

    print(f"    Saved: {OUTPUT_DIR / 'later_scan_followup_scan_summary.csv'}")
    print(f"    Saved: {OUTPUT_DIR / 'later_scan_followup_subject_summary.csv'}")
    print(f"    Saved: {OUTPUT_DIR / 'later_scan_transition_counts_occurrences.csv'}")
    print(f"    Saved: {OUTPUT_DIR / 'later_scan_transition_counts_unique_subjects.csv'}")
    print(f"    Saved: {OUTPUT_DIR / 'later_scan_timepoint_summary.csv'}")
    if not transitions_occurrences.empty:
        print("\nTransition counts across all later-scan occurrences:")
        print(transitions_occurrences.to_string(index=False))
    if not transitions_unique.empty:
        print("\nTransition counts by unique subjects:")
        print(transitions_unique.to_string(index=False))

    changing_subjects = (
        later_scan_summary.groupby(SUBJ_COL)
        .agg(
            n_later_scans=("later_T1", "nunique"),
            n_unique_later_categories=("later_voted_category", "nunique"),
            first_voted_category=("later_voted_category", "first"),
        )
        .reset_index()
    )
    changing_subjects = changing_subjects[changing_subjects["n_unique_later_categories"] > 1]
    if not changing_subjects.empty:
        print("\nSubjects whose later-scan category changed across timepoints:")
        for subject in changing_subjects[SUBJ_COL]:
            subj_rows = (
                later_scan_summary[later_scan_summary[SUBJ_COL] == subject]
                .sort_values(["days_from_first", "followup_index"])
            )
            pieces = [
                f"{int(row['days_from_first'])}d:{row['later_voted_category']}"
                for _, row in subj_rows.iterrows()
            ]
            print(f"    {subject}: " + " | ".join(pieces))
    else:
        print("\nNo subjects changed later-scan category across available timepoints.")

    print("\nDone.")


if __name__ == "__main__":
    main()

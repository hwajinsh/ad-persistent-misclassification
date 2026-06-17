#!/usr/bin/env python
"""
Stage 1: Create Dataset Splits

Reads the filtered subject pool produced by 0_create_subject_pool.py and
creates repeated stratified subject-level folds.

Design:
  - 5 stratified folds
  - 4 repeats
  - 20 total split instances
  - per repeat, each subject appears in the test set exactly once

Output:
  data_splits/
    split_0/
      train.csv
      val.csv
      test.csv
    split_1/
      ...
    ...
    split_summary.csv
  data_split_subject_assignments.csv
"""

import os
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

SUBJECT_POOL_CSV = os.environ.get("SUBJECT_POOL_CSV", "./subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv")
OUTPUT_BASE = os.environ.get("DATA_SPLITS_DIR", "./data_splits")
SPLIT_MANIFEST_CSV = os.environ.get("SPLIT_MANIFEST_CSV", "./data_split_subject_assignments.csv")
N_SPLITS = 5
N_REPEATS = 4
TOTAL_SPLIT_INSTANCES = N_SPLITS * N_REPEATS
RANDOM_SEED = 427
VAL_FRAC_OF_TOTAL = 0.10

os.makedirs(OUTPUT_BASE, exist_ok=True)
manifest_parent = os.path.dirname(SPLIT_MANIFEST_CSV)
if manifest_parent:
    os.makedirs(manifest_parent, exist_ok=True)

print("=" * 70)
print("Dataset Split Creation — Repeated Stratified Folds")
print("=" * 70)
print(f"Source:       {SUBJECT_POOL_CSV}")
print(f"Folds:        {N_SPLITS}")
print(f"Repeats:      {N_REPEATS}")
print(f"Total splits: {TOTAL_SPLIT_INSTANCES}")
print("Approx ratio: 70 / 10 / 20 (train / val / test)")
print(f"Stratify by:  GROUP (AD / CN)")
print("=" * 70)

df = pd.read_csv(SUBJECT_POOL_CSV, low_memory=False)
print(f"\nLoaded {len(df)} subjects")
print(f"  AD: {(df['GROUP'] == 'AD').sum()}")
print(f"  CN: {(df['GROUP'] == 'CN').sum()}")

# Drop rows where GROUP is missing
df = df[df['GROUP'].isin(['AD', 'CN'])].copy()
df = df.reset_index(drop=True)

if 'SUBJECT' not in df.columns:
    raise KeyError("Expected a SUBJECT column in the subject pool CSV.")


def build_subject_manifest(split_df, split_idx, repeat_idx, fold_idx, partition):
    manifest = {
        'SUBJECT': split_df['SUBJECT'].astype(str).values,
    }
    if 'IMAGE_ID' in split_df.columns:
        manifest['IMAGE_ID'] = split_df['IMAGE_ID'].values
    manifest.update({
        'split': split_idx,
        'repeat': repeat_idx,
        'fold': fold_idx,
        'partition': partition,
    })
    return pd.DataFrame(manifest)


summary_rows = []
subject_manifest_rows = []
rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_SEED,
)

for split_idx, (trainval_idx, test_idx) in enumerate(rskf.split(df, df['GROUP'])):
    repeat_idx = split_idx // N_SPLITS
    fold_idx = split_idx % N_SPLITS
    seed = RANDOM_SEED + split_idx
    print(f"\n--- Split {split_idx}  (repeat={repeat_idx}, fold={fold_idx}, seed={seed}) ---")

    trainval = df.iloc[trainval_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    # Validation split within the trainval portion to retain ~10% of the full dataset
    val_frac_of_trainval = VAL_FRAC_OF_TOTAL / (1.0 - 1.0 / N_SPLITS)
    train, val = train_test_split(
        trainval,
        test_size=val_frac_of_trainval,
        stratify=trainval['GROUP'],
        random_state=seed,
    )

    split_dir = os.path.join(OUTPUT_BASE, f'split_{split_idx}')
    os.makedirs(split_dir, exist_ok=True)

    train.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(split_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

    split_subjects = pd.concat([
        build_subject_manifest(train, split_idx, repeat_idx, fold_idx, 'train'),
        build_subject_manifest(val, split_idx, repeat_idx, fold_idx, 'val'),
        build_subject_manifest(test, split_idx, repeat_idx, fold_idx, 'test'),
    ], ignore_index=True)
    subject_manifest_rows.append(split_subjects)

    train_ad = (train['GROUP'] == 'AD').sum()
    train_cn = (train['GROUP'] == 'CN').sum()
    val_ad   = (val['GROUP']   == 'AD').sum()
    val_cn   = (val['GROUP']   == 'CN').sum()
    test_ad  = (test['GROUP']  == 'AD').sum()
    test_cn  = (test['GROUP']  == 'CN').sum()

    print(f"  train: {len(train):4d}  ({train_ad} AD / {train_cn} CN)")
    print(f"  val:   {len(val):4d}  ({val_ad} AD / {val_cn} CN)")
    print(f"  test:  {len(test):4d}  ({test_ad} AD / {test_cn} CN)")

    summary_rows.append({
        'split':      split_idx,
        'repeat':     repeat_idx,
        'fold':       fold_idx,
        'train_n':    len(train),
        'train_AD':   train_ad,
        'train_CN':   train_cn,
        'val_n':      len(val),
        'val_AD':     val_ad,
        'val_CN':     val_cn,
        'test_n':     len(test),
        'test_AD':    test_ad,
        'test_CN':    test_cn,
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_BASE, 'split_summary.csv'), index=False)
subject_manifest_df = pd.concat(subject_manifest_rows, ignore_index=True)
subject_manifest_df.to_csv(SPLIT_MANIFEST_CSV, index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))
print(f"\nMean train size: {summary_df['train_n'].mean():.1f}")
print(f"Mean val size:   {summary_df['val_n'].mean():.1f}")
print(f"Mean test size:  {summary_df['test_n'].mean():.1f}")
print(f"\nEach subject appears in test exactly {N_REPEATS} times.")
print(f"Saved split subject manifest: {SPLIT_MANIFEST_CSV}")
print(f"\nSaved to: {OUTPUT_BASE}/")
print("Next: python 2_train_cnn.py  (or 2_train_sfcn.py)")

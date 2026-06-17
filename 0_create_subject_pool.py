#!/usr/bin/env python
"""
Stage 0: Create Subject Pool

Filters subjects from ADNI_extended_df_merged.csv in the subject_reallocation
folder, then merges biomarker data from the CSV files in the same folder.

Keeps only:
  - AD or CN diagnoses (no MCI)
  - 3T MRI scans only
  - Consistent diagnosis across all visits
  - First scan per subject
  - APOERES genotype when available
  - Subjects with best available amyloid status
  - Remaining AD subjects must be amyloid-positive
  - Remaining CN subjects must be amyloid-negative

Amyloid source priority:
  1. UPENN 2D-UPLC Mass Spectrometry (CSF), ratio <= 0.133
  2. UPENN Plasma Fujirebio/Quanterix, ratio <= 0.0820
  3. C2N PrecivityAD2 Plasma, APS2 > 47.5

Output: subject_reallocation/firstscan_filtered_abeta_tau_apoe.csv
"""

import os
import pandas as pd

DATA_DIR = './subject_reallocation'
OUTPUT_NAME = 'firstscan_filtered_abeta_tau_apoe.csv'
APOERES_CSV = os.environ.get('APOERES_CSV', os.path.join(DATA_DIR, 'apoe_genotype.csv'))
CSF_RATIO_THRESHOLD = 0.133
UPENN_PLASMA_RATIO_THRESHOLD = 0.0820
C2N_APS2_THRESHOLD = 47.5


def standardize_id(df):
    if 'RID' in df.columns:
        df['SUBJ_ID'] = df['RID'].astype(str)
    elif 'PTID' in df.columns:
        df['SUBJ_ID'] = df['PTID'].astype(str).str.extract(r'_(\d+)$')[0]
    elif 'SUBJECT' in df.columns:
        df['SUBJ_ID'] = df['SUBJECT'].astype(str).str.extract(r'_(\d+)$')[0]
    else:
        df['SUBJ_ID'] = None
    return df


def is_abeta42(name):
    c = name.lower()
    return '42' in c and any(tag in c for tag in ['abeta', 'ab', 'amyloid'])


def is_abeta40(name):
    c = name.lower()
    return '40' in c and any(tag in c for tag in ['abeta', 'ab', 'amyloid'])


def is_abeta_ratio(name):
    c = name.lower()
    return ((('42' in c and '40' in c) or 'ratio' in c or '4240' in c) and
            any(tag in c for tag in ['abeta', 'ab', 'amyloid']))


def is_ptau217(name):
    c = name.lower()
    return ('ptau217' in c or 'ptau_217' in c or 'p-tau217' in c or 'pt217' in c or
            ('t217' in c and 'tau' in c)) and 'npt217' not in c


def is_apoe(name):
    return 'apoe' in name.lower()


def is_c2n_aps2(name):
    c = name.lower()
    return 'aps2' in c and 'c2n' in c


def first_non_null(series):
    for value in series:
        if pd.notna(value):
            return value
    return pd.NA


def to_numeric(series):
    return pd.to_numeric(series, errors='coerce')


def normalize_apoe_genotype(value):
    if pd.isna(value):
        return pd.NA
    value_str = str(value).strip()
    parts = value_str.split('/')
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        return f'E{parts[0]}/E{parts[1]}'
    return value_str


def derive_ratio(df, col_42, col_40):
    if col_42 not in df.columns or col_40 not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype='float64')
    a42 = to_numeric(df[col_42])
    a40 = to_numeric(df[col_40])
    ratio = a42 / a40
    ratio[(a42.isna()) | (a40.isna()) | (a40 == 0)] = pd.NA
    return ratio


def assign_best_available_amyloid_status(df):
    csf_sources = [
        (
            'CSF_UPENNMSMSABETA2CRM',
            derive_ratio(df, 'UPENNMSMSABETA2CRM_07Apr2026_ABETA42',
                         'UPENNMSMSABETA2CRM_07Apr2026_ABETA40'),
            CSF_RATIO_THRESHOLD,
            '<=',
        ),
        (
            'CSF_UPENNMSMSABETA2',
            derive_ratio(df, 'UPENNMSMSABETA2_07Apr2026_ABETA42',
                         'UPENNMSMSABETA2_07Apr2026_ABETA40'),
            CSF_RATIO_THRESHOLD,
            '<=',
        ),
        (
            'CSF_UPENNMSMSABETA',
            derive_ratio(df, 'UPENNMSMSABETA_07Apr2026_ABETA42',
                         'UPENNMSMSABETA_07Apr2026_ABETA40'),
            CSF_RATIO_THRESHOLD,
            '<=',
        ),
    ]
    plasma_sources = [
        (
            'PLASMA_UPENN_FUJIREBIO',
            to_numeric(df.get('UPENN_PLASMA_FUJIREBIO_QUANTERIX_07Apr2026_AB42_AB40_F')),
            UPENN_PLASMA_RATIO_THRESHOLD,
            '<=',
        ),
        (
            'PLASMA_C2N_APS2',
            to_numeric(df.get('C2N_PRECIVITYAD2_PLASMA_07Apr2026_APS2_C2N')),
            C2N_APS2_THRESHOLD,
            '>',
        ),
    ]

    df = df.copy()
    df['csf_abeta_ratio_upennms'] = pd.NA
    df['plasma_abeta_ratio_upenn'] = to_numeric(
        df.get('UPENN_PLASMA_FUJIREBIO_QUANTERIX_07Apr2026_AB42_AB40_F')
    )
    df['c2n_aps2'] = to_numeric(df.get('C2N_PRECIVITYAD2_PLASMA_07Apr2026_APS2_C2N'))
    df['amyloid_source'] = pd.NA
    df['amyloid_value'] = pd.NA
    df['amyloid_threshold'] = pd.NA
    df['amyloid_positive'] = pd.NA

    for source_name, values, threshold, operator in csf_sources:
        mask = df['csf_abeta_ratio_upennms'].isna() & values.notna()
        df.loc[mask, 'csf_abeta_ratio_upennms'] = values[mask]

    ordered_sources = csf_sources + plasma_sources
    for source_name, values, threshold, operator in ordered_sources:
        mask = df['amyloid_source'].isna() & values.notna()
        if operator == '<=':
            positive = values <= threshold
        else:
            positive = values > threshold
        df.loc[mask, 'amyloid_source'] = source_name
        df.loc[mask, 'amyloid_value'] = values[mask]
        df.loc[mask, 'amyloid_threshold'] = threshold
        df.loc[mask, 'amyloid_positive'] = positive[mask]

    return df


def merge_apoeres_genotype(df):
    if not os.path.exists(APOERES_CSV):
        print(f"APOERES file not found, skipping genotype merge: {APOERES_CSV}")
        return df

    header = pd.read_csv(APOERES_CSV, nrows=0)
    columns = set(header.columns)
    if 'GENOTYPE' not in columns:
        print(f"APOERES file has no GENOTYPE column, skipping: {APOERES_CSV}")
        return df

    id_cols = [c for c in ['RID', 'PTID', 'SUBJECT'] if c in columns]
    usecols = id_cols + ['GENOTYPE']
    apoe_df = pd.read_csv(APOERES_CSV, usecols=usecols, low_memory=False)
    apoe_df = standardize_id(apoe_df)
    apoe_df = apoe_df[apoe_df['SUBJ_ID'].isin(df['SUBJ_ID'])].copy()
    apoe_df = (
        apoe_df.dropna(subset=['GENOTYPE'])
        .drop_duplicates(subset='SUBJ_ID', keep='first')[['SUBJ_ID', 'GENOTYPE']]
    )
    apoe_df['GENOTYPE'] = apoe_df['GENOTYPE'].apply(normalize_apoe_genotype)
    apoe_df = apoe_df.rename(columns={'GENOTYPE': 'APOERES_GENOTYPE'})
    merged = df.merge(apoe_df, on='SUBJ_ID', how='left', validate='one_to_one')
    print(f"APOERES genotype merged for {merged['APOERES_GENOTYPE'].notna().sum()} subjects")
    return merged


# 1. Load and filter the starter file
main_df = pd.read_csv(os.path.join(DATA_DIR, 'ADNI_extended_df_merged.csv'), low_memory=False)
main_df = standardize_id(main_df)
main_df = main_df[~main_df['SUBJECT'].astype(str).str.startswith('381_S_10')]
main_df = main_df[main_df['GROUP'].isin(['AD', 'CN'])]
main_df = main_df[main_df['MRI_FIELD_STR'].astype(str).str.startswith('3')]

# keep only subjects whose diagnosis is consistent across visits
consistent = main_df.groupby('SUBJ_ID')['GROUP'].nunique()
consistent_subj = consistent[consistent == 1].index
main_df = main_df[main_df['SUBJ_ID'].isin(consistent_subj)]

# find each subject's first scan using MRI_DATE
main_df['MRI_DATE_dt'] = pd.to_datetime(main_df['MRI_DATE'], errors='coerce')
main_df_first = (main_df
                 .sort_values(['SUBJ_ID', 'MRI_DATE_dt'])
                 .drop_duplicates(subset='SUBJ_ID', keep='first')
                 .reset_index(drop=True))

final_df = main_df_first.copy()

# 2. Merge biomarker data one file at a time
for fname in os.listdir(DATA_DIR):
    if not fname.lower().endswith('.csv'):
        continue
    if fname == OUTPUT_NAME or fname.startswith('firstscan_filtered_'):
        continue
    path = os.path.join(DATA_DIR, fname)
    header = pd.read_csv(path, nrows=0)
    columns = header.columns

    abeta_cols = [c for c in columns if is_abeta42(c) or is_abeta40(c) or is_abeta_ratio(c)]
    tau_cols   = [c for c in columns if is_ptau217(c)]
    apoe_cols  = [c for c in columns if is_apoe(c)]
    aps2_cols  = [c for c in columns if is_c2n_aps2(c)]
    if not (abeta_cols or tau_cols or apoe_cols or aps2_cols):
        continue

    id_cols = [c for c in ['RID', 'PTID', 'SUBJECT'] if c in columns]
    usecols = id_cols + list(set(abeta_cols + tau_cols + apoe_cols + aps2_cols))
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = standardize_id(df)
    df = df[~df['SUBJ_ID'].astype(str).str.startswith('381_S_10')]
    df = df[df['SUBJ_ID'].isin(final_df['SUBJ_ID'])]

    prefix = os.path.splitext(fname)[0]
    if abeta_cols:
        sub = df[['SUBJ_ID'] + abeta_cols].copy()
        sub = sub.groupby('SUBJ_ID', as_index=False).agg({c: first_non_null for c in abeta_cols})
        sub.rename(columns={c: f'{prefix}_{c}' for c in abeta_cols}, inplace=True)
        final_df = final_df.merge(sub, on='SUBJ_ID', how='left', validate='one_to_one')

    if tau_cols:
        sub = df[['SUBJ_ID'] + tau_cols].copy()
        sub = sub.groupby('SUBJ_ID', as_index=False).agg({c: first_non_null for c in tau_cols})
        sub.rename(columns={c: f'{prefix}_{c}' for c in tau_cols}, inplace=True)
        final_df = final_df.merge(sub, on='SUBJ_ID', how='left', validate='one_to_one')

    if apoe_cols:
        sub = df[['SUBJ_ID'] + apoe_cols].copy()
        sub = sub.groupby('SUBJ_ID', as_index=False).agg({c: first_non_null for c in apoe_cols})
        sub.rename(columns={c: f'{prefix}_{c}' for c in apoe_cols}, inplace=True)
        final_df = final_df.merge(sub, on='SUBJ_ID', how='left', validate='one_to_one')

    if aps2_cols:
        sub = df[['SUBJ_ID'] + aps2_cols].copy()
        sub = sub.groupby('SUBJ_ID', as_index=False).agg({c: first_non_null for c in aps2_cols})
        sub.rename(columns={c: f'{prefix}_{c}' for c in aps2_cols}, inplace=True)
        final_df = final_df.merge(sub, on='SUBJ_ID', how='left', validate='one_to_one')

# 3. Build best available amyloid status and keep biologically consistent cases
final_df = merge_apoeres_genotype(final_df)
final_df = assign_best_available_amyloid_status(final_df)
has_amyloid = final_df['amyloid_source'].notna()
is_consistent = (
    ((final_df['GROUP'] == 'AD') & (final_df['amyloid_positive'] == True)) |
    ((final_df['GROUP'] == 'CN') & (final_df['amyloid_positive'] == False))
)
final_df = final_df[has_amyloid & is_consistent].copy()

# 4. Write the result
out_path = os.path.join(DATA_DIR, OUTPUT_NAME)
final_df.to_csv(out_path, index=False)
print('Saved', len(final_df), 'rows to', out_path)
print(f"  AD: {(final_df['GROUP'] == 'AD').sum()}")
print(f"  CN: {(final_df['GROUP'] == 'CN').sum()}")
print('\nAmyloid source counts:')
print(final_df['amyloid_source'].value_counts(dropna=False).to_string())

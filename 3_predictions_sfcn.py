#!/usr/bin/env python
"""
Stage 3b: SFCN Fine-tuning Predictions

Evaluates all fine-tuned SFCN models across every available split instance
and aggregates results.

Metrics per run: balanced accuracy, sensitivity, specificity, AUC.
Final output: mean ± std over all 100 model instances
(20 split instances × 5 runs).

Per-run predictions (subject IDs, labels, raw probabilities, binary
predictions) are saved as pickle files for downstream analysis. 

Output:
  predictions_sfcn/
    split_0/predictions.pkl
    ...
    evaluation_summary.csv
    all_split_results.pkl
"""

import sys
import os

NITORCH_DIR = os.path.expandvars(os.environ.get("NITORCH_DIR", "./external/nitorch"))
if NITORCH_DIR not in sys.path:
    sys.path.append(NITORCH_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pickle
from collections import OrderedDict

from nitorch.transforms import ToTensor
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from nitorch.metrics import sensitivity, specificity

print(torch.__version__)
print(torch.version.cuda)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RUNS_PER_SPLIT = 5
GPU = 0

SFCN_INPUT_SHAPE = (160, 192, 160)

DATA_SPLITS_PATH = './data_splits'
BASE_MRI_PATH = os.environ.get("ADNI_PREPROCESSED_MRI_DIR", "./preprocessed_mri")
MODEL_PATH = './models_sfcn'
OUTPUT_PATH = './predictions_sfcn'

os.makedirs(OUTPUT_PATH, exist_ok=True)


def discover_split_indices():
    split_indices = []
    if not os.path.isdir(DATA_SPLITS_PATH):
        return split_indices
    for name in sorted(os.listdir(DATA_SPLITS_PATH)):
        if name.startswith('split_'):
            try:
                split_indices.append(int(name.split('_')[1]))
            except (IndexError, ValueError):
                continue
    return sorted(split_indices)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class CenterCrop3D:
    def __init__(self, target_shape=(160, 192, 160)):
        self.target_shape = target_shape

    def __call__(self, image):
        starts = [(c - t) // 2 for c, t in zip(image.shape, self.target_shape)]
        return image[
            starts[0]:starts[0] + self.target_shape[0],
            starts[1]:starts[1] + self.target_shape[1],
            starts[2]:starts[2] + self.target_shape[2],
        ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ADNIDatasetLazy(Dataset):
    def __init__(self, file_paths, labels, subjects, base_path, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.subjects = subjects
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_path, self.file_paths[idx])
        try:
            image = nib.load(path).get_fdata().astype(np.float32)
            image = image - image.min()
            mean_val = image.mean()
            if mean_val > 0:
                image = image / mean_val
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = np.zeros((182, 218, 182), dtype=np.float32)

        label = torch.tensor(int(self.labels[idx] >= 0.5)).long()
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}


# ---------------------------------------------------------------------------
# SFCN architecture (must match 2_train_sfcn.py exactly)
# ---------------------------------------------------------------------------
class SFCN(nn.Module):
    def __init__(self, channel_number=(32, 64, 128, 256, 256, 64),
                 output_dim=40, dropout=True):
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            in_ch  = 1 if i == 0 else channel_number[i - 1]
            out_ch = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    f'conv_{i}',
                    self._conv_layer(in_ch, out_ch, maxpool=True,
                                     kernel_size=3, padding=1))
            else:
                self.feature_extractor.add_module(
                    f'conv_{i}',
                    self._conv_layer(in_ch, out_ch, maxpool=False,
                                     kernel_size=1, padding=0))

        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d((5, 6, 5)))
        if dropout:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        self.classifier.add_module(
            f'conv_{n_layer}',
            nn.Conv3d(channel_number[-1], output_dim, padding=0, kernel_size=1))

    @staticmethod
    def _conv_layer(in_channel, out_channel, maxpool=True,
                    kernel_size=3, padding=0):
        if maxpool:
            return nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding,
                          kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2),
                nn.ReLU(),
            )
        return nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding,
                      kernel_size=kernel_size),
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


def build_model(num_classes=2):
    sfcn = SFCN(channel_number=[32, 64, 128, 256, 256, 64], output_dim=40)
    return SFCNClassifier(sfcn.feature_extractor, n_features=64,
                          num_classes=num_classes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_test_csv(csv_path):
    df = pd.read_csv(csv_path)
    paths    = df['T1'].tolist()
    labels   = ((df['GROUP'] == 'AD') * 1).tolist()
    subjects = df['SUBJECT'].tolist()
    return paths, labels, subjects


def run_inference(model, dataset, gpu):
    loader = DataLoader(dataset, batch_size=1, num_workers=1,
                        shuffle=False, pin_memory=False)
    raw_probs, preds, labels_out = [], [], []
    with torch.no_grad():
        for sample in loader:
            img   = sample['image'].cuda(gpu)
            label = sample['label']
            out   = model(img)
            prob  = F.softmax(out, dim=1)[0][1].cpu().item()
            pred  = int(prob >= 0.5)
            raw_probs.append(prob)
            preds.append(pred)
            labels_out.append(label.item())
    return raw_probs, preds, labels_out


def compute_metrics(labels, preds, raw_probs):
    bal_acc = balanced_accuracy_score(labels, preds)
    sens    = sensitivity(labels, preds)
    spec    = specificity(labels, preds)
    auc     = roc_auc_score(labels, raw_probs) if len(set(labels)) > 1 else float('nan')
    return bal_acc, sens, spec, auc


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------
def evaluate_split(split_idx):
    print(f"\n{'='*70}\nSplit {split_idx}\n{'='*70}")

    split_dir       = os.path.join(DATA_SPLITS_PATH, f'split_{split_idx}')
    split_model_dir = os.path.join(MODEL_PATH, f'split_{split_idx}')

    test_paths, test_labels, test_subjects = load_test_csv(
        os.path.join(split_dir, 'test.csv'))
    print(f"  Test: {len(test_paths)} subjects "
          f"({sum(test_labels)} AD / {len(test_labels)-sum(test_labels)} CN)")

    test_transform = transforms.Compose([CenterCrop3D(SFCN_INPUT_SHAPE), ToTensor()])
    ds_test = ADNIDatasetLazy(
        test_paths, test_labels, test_subjects, BASE_MRI_PATH,
        transform=test_transform)

    run_results = []

    for run_idx in range(N_RUNS_PER_SPLIT):
        model_path = os.path.join(split_model_dir, f'run_{run_idx}_model-best.h5')
        if not os.path.exists(model_path):
            print(f"  WARNING: model not found: {model_path}")
            continue

        model = build_model(num_classes=2)
        state_dict = torch.load(model_path, weights_only=True)
        clean_sd = OrderedDict(
            (k[7:] if k.startswith('module.') else k, v)
            for k, v in state_dict.items()
        )
        model.load_state_dict(clean_sd)
        model.cuda(GPU).eval()

        raw_probs, preds, labels_out = run_inference(model, ds_test, GPU)
        bal_acc, sens, spec, auc = compute_metrics(labels_out, preds, raw_probs)

        run_results.append({
            'run':          run_idx,
            'balanced_acc': bal_acc,
            'sensitivity':  sens,
            'specificity':  spec,
            'auc':          auc,
            'raw_probs':    raw_probs,
            'preds':        preds,
            'labels':       labels_out,
            'subjects':     test_subjects,
        })
        print(f"  Run {run_idx}: bal_acc={bal_acc:.4f}, "
              f"sens={sens:.4f}, spec={spec:.4f}, auc={auc:.4f}")

        del model
        torch.cuda.empty_cache()

    if run_results:
        mean_acc = np.mean([r['balanced_acc'] for r in run_results])
        print(f"\n  Split {split_idx} mean bal_acc: {mean_acc:.4f}")

    return run_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    split_indices = discover_split_indices()
    if not split_indices:
        raise FileNotFoundError(f"No split_* directories found under {DATA_SPLITS_PATH}")

    print(f"{'='*70}")
    print("SFCN Fine-tuning Predictions — New Subject Pool")
    print(f"{'='*70}")
    print(f"  Models:  {MODEL_PATH}")
    print(f"  Data:    {DATA_SPLITS_PATH}")
    print(f"  Output:  {OUTPUT_PATH}")
    print(f"  Split instances: {len(split_indices)}  |  Runs/split: {N_RUNS_PER_SPLIT}")
    print(f"{'='*70}\n")

    all_split_results = []

    for split_idx in split_indices:
        run_results = evaluate_split(split_idx)
        all_split_results.append({'split': split_idx, 'runs': run_results})

        split_out = os.path.join(OUTPUT_PATH, f'split_{split_idx}')
        os.makedirs(split_out, exist_ok=True)
        with open(os.path.join(split_out, 'predictions.pkl'), 'wb') as f:
            pickle.dump(run_results, f)

    # Aggregate across all runs
    total_possible_runs = len(split_indices) * N_RUNS_PER_SPLIT
    print(f"\n{'='*70}\nFINAL RESULTS — Up To {total_possible_runs} Runs\n{'='*70}")

    all_bal_accs, all_sens, all_specs, all_aucs = [], [], [], []
    for sr in all_split_results:
        for r in sr['runs']:
            all_bal_accs.append(r['balanced_acc'])
            all_sens.append(r['sensitivity'])
            all_specs.append(r['specificity'])
            if not np.isnan(r['auc']):
                all_aucs.append(r['auc'])

    print(f"Balanced Accuracy: {np.mean(all_bal_accs)*100:.2f}% ± "
          f"{np.std(all_bal_accs)*100:.2f}%")
    print(f"Sensitivity:       {np.mean(all_sens):.4f} ± {np.std(all_sens):.4f}")
    print(f"Specificity:       {np.mean(all_specs):.4f} ± {np.std(all_specs):.4f}")
    print(f"AUC:               {np.mean(all_aucs):.4f} ± {np.std(all_aucs):.4f}")
    print(f"  (based on {len(all_bal_accs)} model evaluations)")

    summary = pd.DataFrame([{
        'metric': 'balanced_accuracy',
        'mean':   np.mean(all_bal_accs),
        'std':    np.std(all_bal_accs),
    }, {
        'metric': 'sensitivity',
        'mean':   np.mean(all_sens),
        'std':    np.std(all_sens),
    }, {
        'metric': 'specificity',
        'mean':   np.mean(all_specs),
        'std':    np.std(all_specs),
    }, {
        'metric': 'auc',
        'mean':   np.mean(all_aucs),
        'std':    np.std(all_aucs),
    }])

    summary.to_csv(os.path.join(OUTPUT_PATH, 'evaluation_summary.csv'), index=False)
    with open(os.path.join(OUTPUT_PATH, 'all_split_results.pkl'), 'wb') as f:
        pickle.dump(all_split_results, f)

    print(f"\nSaved to {OUTPUT_PATH}/")


if __name__ == '__main__':
    main()

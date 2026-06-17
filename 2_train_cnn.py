#!/usr/bin/env python
"""
Stage 2a: Train VoxCNN 

Trains the 5-block 3-D convolutional classifier on each split instance
produced by 1_create_dataset_splits.py.

Current design: 5 folds × 4 repeats = 20 split instances
Total models: 20 split instances × 5 runs = 100

Architecture: Conv3d blocks (8→16→32→64 channels) + two FC layers.
Input: 256³ MRI volume, min-max normalised.
Augmentation (training only): random sagittal flip + sagittal translate.

Output:
  models_cnn/
    split_0/
      run_0_model-best.h5
      run_1_model-best.h5
      ...
    ...
    training_summary.csv        
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
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

from nitorch.transforms import ToTensor, SagittalFlip, SagittalTranslate
from nitorch.callbacks import EarlyStopping, ModelCheckpoint
from nitorch.trainer import Trainer
from sklearn.metrics import balanced_accuracy_score as balanced_accuracy
from nitorch.utils import count_parameters

print(torch.__version__)
print(torch.version.cuda)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RUNS_PER_SPLIT = 5
BATCH_SIZE = 2
NUM_EPOCHS = 200
IGNORE_EPOCHS = 15
GPU = 0

DATA_SPLITS_PATH = './data_splits'
BASE_MRI_PATH = os.environ.get("ADNI_PREPROCESSED_MRI_DIR", "./preprocessed_mri")
MODEL_OUTPUT_PATH = './models_cnn'

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

gpu_ids = list(range(torch.cuda.device_count()))
effective_batch_size = BATCH_SIZE * max(len(gpu_ids), 1)
print(f"Using {len(gpu_ids)} GPU(s), effective batch size: {effective_batch_size}")


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
# Dataset
# ---------------------------------------------------------------------------
class ADNIDatasetLazy(Dataset):
    def __init__(self, file_paths, labels, base_path, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_path, self.file_paths[idx])
        try:
            image = nib.load(path).get_fdata().astype(np.float32)
            image -= image.min()
            image /= image.max()
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = np.zeros((182, 218, 182), dtype=np.float32)

        label = torch.tensor(int(self.labels[idx] >= 0.5)).long()
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ClassificationModel3D(nn.Module):
    def __init__(self, dropout=0.4, dropout2=0.4):
        super().__init__()
        self.Conv_1    = nn.Conv3d(1, 8, 3)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_1_mp = nn.MaxPool3d(2)
        self.Conv_2    = nn.Conv3d(8, 16, 3)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_2_mp = nn.MaxPool3d(3)
        self.Conv_3    = nn.Conv3d(16, 32, 3)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_3_mp = nn.MaxPool3d(2)
        self.Conv_4    = nn.Conv3d(32, 64, 3)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        self.Conv_4_mp = nn.MaxPool3d(3)
        self.dense_1   = nn.Linear(2304, 128)
        self.dense_2   = nn.Linear(128, 2)
        self.relu      = nn.ReLU()
        self.dropout   = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout2)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    paths  = df['T1'].tolist()
    labels = ((df['GROUP'] == 'AD') * 1).tolist()
    print(f"  {os.path.basename(csv_path)}: {len(paths)} scans "
          f"({sum(labels)} AD / {len(labels)-sum(labels)} CN)")
    return paths, labels


def train_split(split_idx, skip_existing=True):
    print(f"\n{'='*70}\nSplit {split_idx}\n{'='*70}")

    split_dir  = os.path.join(DATA_SPLITS_PATH, f'split_{split_idx}')
    model_dir  = os.path.join(MODEL_OUTPUT_PATH, f'split_{split_idx}')
    os.makedirs(model_dir, exist_ok=True)

    train_paths, train_labels = load_csv(os.path.join(split_dir, 'train.csv'))
    val_paths,   val_labels   = load_csv(os.path.join(split_dir, 'val.csv'))

    aug = [SagittalFlip(), SagittalTranslate(dist=(-2, 3))]
    ds_train = ADNIDatasetLazy(train_paths, train_labels, BASE_MRI_PATH,
                               transform=transforms.Compose(aug + [ToTensor()]))
    ds_val   = ADNIDatasetLazy(val_paths,   val_labels,   BASE_MRI_PATH,
                               transform=transforms.Compose([ToTensor()]))

    split_metrics = []

    for run_idx in range(N_RUNS_PER_SPLIT):
        model_path = os.path.join(model_dir, f'run_{run_idx}_model-best.h5')

        if skip_existing and os.path.exists(model_path):
            print(f"\n  Run {run_idx}: model exists, loading for eval...")
            try:
                net = ClassificationModel3D()
                net = nn.DataParallel(net, device_ids=gpu_ids)
                net.load_state_dict(torch.load(model_path, weights_only=True))
                net.cuda(GPU).eval()
                loader = DataLoader(ds_val, batch_size=1, num_workers=1, shuffle=False)
                preds, labs = [], []
                with torch.no_grad():
                    for s in loader:
                        out = net(s['image'].cuda(GPU))
                        preds.append(torch.argmax(F.softmax(out, dim=1)).cpu().item())
                        labs.append(s['label'].item())
                m = balanced_accuracy(labs, preds)
                print(f"  Run {run_idx} val balanced_accuracy: {m:.4f}")
                split_metrics.append(m)
                continue
            except Exception as e:
                print(f"  Load failed ({e}), retraining...")

        net = ClassificationModel3D()
        net = nn.DataParallel(net, device_ids=gpu_ids)
        net.cuda(GPU)
        print(f"\n  Run {run_idx}  params: {count_parameters(net):,}")

        criterion = nn.CrossEntropyLoss().cuda(GPU)
        optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)

        train_loader = DataLoader(ds_train, batch_size=effective_batch_size,
                                  num_workers=2, shuffle=True, pin_memory=False)
        val_loader   = DataLoader(ds_val,   batch_size=1,
                                  num_workers=1, shuffle=False, pin_memory=False)

        check = ModelCheckpoint(path=model_dir, prepend=f'run_{run_idx}_',
                                store_best=True, ignore_before=IGNORE_EPOCHS,
                                retain_metric=balanced_accuracy, mode='max')
        early_stop = EarlyStopping(patience=8, ignore_before=IGNORE_EPOCHS,
                                   retain_metric='loss', mode='min')

        trainer = Trainer(net, criterion, optimizer,
                          metrics=[balanced_accuracy],
                          callbacks=[check, early_stop],
                          device=GPU, prediction_type='classification')

        net, report = trainer.train_model(
            train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            show_train_steps=10,
            show_validation_epochs=1,
        )

        m = report['val_metrics']['balanced_accuracy_score'][-1]
        split_metrics.append(m)
        trainer.visualize_training(report, [balanced_accuracy],
                                   os.path.join(model_dir, f'run_{run_idx}'))
        print(f"  Run {run_idx} val balanced_accuracy: {m:.4f}")

    print(f"\nSplit {split_idx}: {np.mean(split_metrics):.4f} ± {np.std(split_metrics):.4f}")
    return split_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    split_indices = discover_split_indices()
    if not split_indices:
        raise FileNotFoundError(f"No split_* directories found under {DATA_SPLITS_PATH}")

    print(f"\n{'='*70}")
    print("VoxCNN Training — New Subject Pool")
    print(f"{'='*70}")
    print(f"  Split instances: {len(split_indices)}  |  Runs/split: {N_RUNS_PER_SPLIT}  |  "
          f"Total models: {len(split_indices) * N_RUNS_PER_SPLIT}")
    print(f"  Data:   {DATA_SPLITS_PATH}")
    print(f"  Output: {MODEL_OUTPUT_PATH}")
    print(f"{'='*70}\n")

    all_metrics = []
    for split_idx in split_indices:
        metrics = train_split(split_idx, skip_existing=True)
        all_metrics.append({'split': split_idx,
                            'mean_val_bal_acc': np.mean(metrics),
                            'std_val_bal_acc':  np.std(metrics)})

    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for r in all_metrics:
        print(f"  Split {r['split']}: {r['mean_val_bal_acc']:.4f} ± {r['std_val_bal_acc']:.4f}")
    overall = [r['mean_val_bal_acc'] for r in all_metrics]
    print(f"\n  Overall: {np.mean(overall):.4f} ± {np.std(overall):.4f}")

    pd.DataFrame(all_metrics).to_csv(
        os.path.join(MODEL_OUTPUT_PATH, 'training_summary.csv'), index=False)
    print(f"\nSaved summary to {MODEL_OUTPUT_PATH}/training_summary.csv")
    print("Next: python 3_predictions_cnn.py")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        train_split(int(sys.argv[1]), skip_existing=True)
    else:
        main()

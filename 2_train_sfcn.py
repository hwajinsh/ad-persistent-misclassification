#!/usr/bin/env python
"""
Stage 2b: Train SFCN (fine-tuning) 

Fine-tunes a pre-trained SFCN (Peng et al., Medical Image Analysis 2021)
on each split instance produced by 1_create_dataset_splits.py.

The SFCN was pre-trained on brain age prediction using 14,503 UK Biobank
brain MRI scans. All layers are updated (fine-tuning mode).

Pre-trained checkpoint:
  https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

Architecture: 6-block feature extractor [32,64,128,256,256,64] + linear head.
Input: center-cropped to (160, 192, 160) to match UKBiobank pipeline.

Current design: 5 folds × 4 repeats = 20 split instances
Total models: 20 split instances × 5 runs = 100

Output:
  models_sfcn/
    split_0/
      run_0_model-best.h5
      ...
    ...
    training_summary.csv         
"""

import sys
import os
import urllib.request

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
IGNORE_EPOCHS = 1
GPU = 0

SFCN_INPUT_SHAPE = (160, 192, 160)

DATA_SPLITS_PATH = './data_splits'
BASE_MRI_PATH = os.environ.get("ADNI_PREPROCESSED_MRI_DIR", "./preprocessed_mri")
MODEL_OUTPUT_PATH = './models_sfcn'

PRETRAINED_DIR  = './pretrained_checkpoints'
PRETRAINED_CKPT = os.path.join(PRETRAINED_DIR, 'sfcn_ukb_brain_age.p')
PRETRAINED_URL  = ('https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain'
                   '/raw/master/brain_age/run_20190719_00_epoch_best_mae.p')

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

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
# SFCN architecture (matches UKBiobank_deep_pretrain exactly)
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


def load_pretrained_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
    feat_state = {k.replace('feature_extractor.', '', 1): v
                  for k, v in ckpt.items() if k.startswith('feature_extractor.')}
    missing, unexpected = model.feature_extractor.load_state_dict(
        feat_state, strict=True)
    print(f"  Pre-trained keys loaded: {len(feat_state)}")
    if missing:
        print(f"  Missing: {missing}")
    if unexpected:
        print(f"  Unexpected: {unexpected}")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def download_checkpoint():
    if os.path.exists(PRETRAINED_CKPT):
        print(f"Checkpoint found: {PRETRAINED_CKPT}")
        return PRETRAINED_CKPT
    print(f"Downloading SFCN checkpoint from:\n  {PRETRAINED_URL}")
    urllib.request.urlretrieve(PRETRAINED_URL, PRETRAINED_CKPT)
    print(f"Saved to: {PRETRAINED_CKPT}")
    return PRETRAINED_CKPT


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    paths  = df['T1'].values
    labels = (df['GROUP'] == 'AD').astype(int).values
    print(f"  {os.path.basename(csv_path)}: {len(paths)} scans "
          f"({labels.sum()} AD / {len(labels)-labels.sum()} CN)")
    return paths, labels


def train_single_run(run_idx, model_dir, train_loader, val_loader,
                     ckpt_path, train_labels):
    model_path = os.path.join(model_dir, f'run_{run_idx}_model-best.h5')
    torch.cuda.empty_cache()

    model = build_model(num_classes=2)
    model = load_pretrained_weights(model, ckpt_path)
    model = nn.DataParallel(model, device_ids=gpu_ids)
    model.cuda(GPU)

    n_pos = int(train_labels.sum())
    n_neg = len(train_labels) - n_pos
    n_samples = len(train_labels)
    full_w = torch.tensor(
        [n_samples / (2.0 * n_neg), n_samples / (2.0 * n_pos)],
        dtype=torch.float32)
    class_weights = torch.sqrt(full_w)
    print(f"  Class weights (softened sqrt): CN={class_weights[0]:.3f}, "
          f"AD={class_weights[1]:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(GPU)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)

    check = ModelCheckpoint(path=model_dir, prepend=f'run_{run_idx}_',
                            store_best=True, ignore_before=IGNORE_EPOCHS,
                            retain_metric=balanced_accuracy, mode='max')
    early_stop = EarlyStopping(patience=8, ignore_before=IGNORE_EPOCHS,
                               retain_metric='loss', mode='min')

    trainer = Trainer(model, criterion, optimizer,
                      metrics=[balanced_accuracy],
                      callbacks=[check, early_stop],
                      device=GPU, prediction_type='classification')

    try:
        model, report = trainer.train_model(
            train_loader, val_loader,
            num_epochs=NUM_EPOCHS,
            show_train_steps=10,
            show_validation_epochs=1,
        )
        m = report['val_metrics']['balanced_accuracy_score'][-1]
        trainer.visualize_training(report, [balanced_accuracy],
                                   os.path.join(model_dir, f'run_{run_idx}'))
        print(f"  Run {run_idx} val balanced_accuracy: {m:.4f}")
        return m
    finally:
        del model, optimizer, criterion, trainer
        torch.cuda.empty_cache()


def train_split(split_idx, ckpt_path, skip_existing=True):
    print(f"\n{'='*70}\nSplit {split_idx}\n{'='*70}")

    split_dir = os.path.join(DATA_SPLITS_PATH, f'split_{split_idx}')
    model_dir = os.path.join(MODEL_OUTPUT_PATH, f'split_{split_idx}')
    os.makedirs(model_dir, exist_ok=True)

    train_paths, train_labels = load_csv(os.path.join(split_dir, 'train.csv'))
    val_paths,   val_labels   = load_csv(os.path.join(split_dir, 'val.csv'))

    train_transform = transforms.Compose([
        CenterCrop3D(SFCN_INPUT_SHAPE),
        SagittalFlip(),
        SagittalTranslate(dist=(-2, 3)),
        ToTensor(),
    ])
    val_transform = transforms.Compose([
        CenterCrop3D(SFCN_INPUT_SHAPE),
        ToTensor(),
    ])

    ds_train = ADNIDatasetLazy(train_paths, train_labels, BASE_MRI_PATH,
                               transform=train_transform)
    ds_val   = ADNIDatasetLazy(val_paths,   val_labels,   BASE_MRI_PATH,
                               transform=val_transform)

    train_loader = DataLoader(ds_train, batch_size=effective_batch_size,
                              num_workers=1, shuffle=True, pin_memory=False)
    val_loader   = DataLoader(ds_val,   batch_size=1,
                              num_workers=1, shuffle=False, pin_memory=False)

    split_metrics = []
    for run_idx in range(N_RUNS_PER_SPLIT):
        model_path = os.path.join(model_dir, f'run_{run_idx}_model-best.h5')
        if skip_existing and os.path.exists(model_path):
            print(f"\n  Run {run_idx}: model exists, skipping")
            continue
        print(f"\n{'='*50}\nSplit {split_idx} | Run {run_idx}\n{'='*50}")
        try:
            m = train_single_run(run_idx, model_dir, train_loader,
                                 val_loader, ckpt_path, train_labels)
            split_metrics.append(m)
        except Exception as e:
            print(f"ERROR in split {split_idx}, run {run_idx}: {e}")
            import traceback; traceback.print_exc()

    if split_metrics:
        print(f"\nSplit {split_idx}: {np.mean(split_metrics):.4f} ± "
              f"{np.std(split_metrics):.4f}")
    return split_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    split_indices = discover_split_indices()
    if not split_indices:
        raise FileNotFoundError(f"No split_* directories found under {DATA_SPLITS_PATH}")

    print(f"\n{'='*70}")
    print("SFCN Fine-tuning — New Subject Pool")
    print(f"{'='*70}")
    print(f"  Mode:   Fine-tuning (all layers trainable)")
    print(f"  Input:  {SFCN_INPUT_SHAPE} (center-cropped from 182×218×182)")
    print(f"  Split instances: {len(split_indices)}  |  Runs/split: {N_RUNS_PER_SPLIT}  |  "
          f"Total models: {len(split_indices) * N_RUNS_PER_SPLIT}")
    print(f"  Data:   {DATA_SPLITS_PATH}")
    print(f"  Output: {MODEL_OUTPUT_PATH}")
    print(f"{'='*70}\n")

    ckpt_path = download_checkpoint()
    all_metrics = []

    for split_idx in split_indices:
        metrics = train_split(split_idx, ckpt_path, skip_existing=True)
        if metrics:
            all_metrics.append({'split': split_idx,
                                'mean_val_bal_acc': np.mean(metrics),
                                'std_val_bal_acc':  np.std(metrics)})

    if all_metrics:
        print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
        for r in all_metrics:
            print(f"  Split {r['split']}: {r['mean_val_bal_acc']:.4f} ± "
                  f"{r['std_val_bal_acc']:.4f}")
        overall = [r['mean_val_bal_acc'] for r in all_metrics]
        print(f"\n  Overall: {np.mean(overall):.4f} ± {np.std(overall):.4f}")

        pd.DataFrame(all_metrics).to_csv(
            os.path.join(MODEL_OUTPUT_PATH, 'training_summary.csv'), index=False)
        print(f"\nSaved summary to {MODEL_OUTPUT_PATH}/training_summary.csv")

    print("Next: python 3_predictions_sfcn.py")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        ckpt_path = download_checkpoint()
        train_split(int(sys.argv[1]), ckpt_path, skip_existing=True)
    else:
        main()

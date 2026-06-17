# What Do Persistent Misclassifications Tell Us About Alzheimer's Disease Detection using structural MRI?

## Included Scripts

- `0_create_subject_pool.py`
- `1_create_dataset_splits.py`
- `1_descriptive_stats.py`
- `2_train_cnn.py`
- `2_train_sfcn.py`
- `3_predictions_cnn.py`
- `3_predictions_sfcn.py`
- `4_voted_misclassification.py`
- `5_risacher_subtyping.py`
- `6_error_analysis.py`
- `7_later_scan_followup.py`

## Required Inputs

- baseline and longitudinal cohort CSVs in `subject_reallocation/`
- biomarker CSVs in `subject_reallocation/`
- an APOERES genotype CSV supplied via `APOERES_CSV`
- a volumetric ROI CSV supplied via `RISACHER_VOLUMES_CSV`
- preprocessed MRI volumes reachable at `ADNI_PREPROCESSED_MRI_DIR` or placed under `./preprocessed_mri`

## External Dependencies

- Python packages used in the scripts: `numpy`, `pandas`, `scipy`, `scikit-learn`, `nibabel`, `torch`, `torchvision`
- `nitorch` utilities are expected at `NITORCH_DIR` or under `./external/nitorch`
- `2_train_sfcn.py` downloads the published UK Biobank SFCN checkpoint automatically if it is not already present in `pretrained_checkpoints/`

## Code Structure

To reproduce our results, run the scripts in order:

0. `0_create_subject_pool.py`
   Builds the baseline AD/CN cohort, merges available biomarker data, incorporates APOERES genotype when available, and applies the amyloid-consistency filters described in the manuscript.

1. `1_create_dataset_splits.py`
   Creates the repeated stratified train/validation/test splits used throughout the paper.

2. `1_descriptive_stats.py`
   Generates the pre-classification cohort demographics and descriptive Risacher subtype tables.

3. `2_train_cnn.py`
   Trains the VoxCNN models on the repeated stratified splits.

4. `2_train_sfcn.py`
   Fine-tunes the SFCN models on the repeated stratified splits.

5. `3_predictions_cnn.py`
   Saves raw test-set predictions for the trained VoxCNN models.

6. `3_predictions_sfcn.py`
   Saves raw test-set predictions for the trained SFCN models.

7. `4_voted_misclassification.py`
   Aggregates predictions across models, runs, and splits to assign the voted categories used in the manuscript.

8. `5_risacher_subtyping.py`
   Applies the Risacher atrophy subtyping procedure to AD subjects with ROI data and compares subtype composition between voted groups.

9. `6_error_analysis.py`
   Summarizes the clinical, biomarker, and APOE differences between persistently misclassified and correctly classified AD subjects.

10. `7_later_scan_followup.py`
   Re-evaluates later scans for persistently misclassified subjects using the exact out-of-sample model instances that originally tested them.

## Reproducibility Split Export

This repository includes the exact subject assignments from the previous completed run at `data_split_subject_assignments.csv`

This file contains only:

- `SUBJECT`
- `IMAGE_ID`
- `split`
- `repeat`
- `fold`
- `partition`

This is the lightweight split information to share for reproducibility without distributing the full per-subject metadata tables.

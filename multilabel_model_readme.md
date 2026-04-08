# Multilabel Denial Reasons Model README

## Overview

This model predicts one or more denial reason labels for a claim.

- Task: multilabel classification
- Targets: combined values from `target1`, `target2`, `target3`, `target4`
- Label count: `24`
- Endpoint: `POST /predict-denial-reasons/`
- Model file: `models/multilabel_model.pkl`
- Thresholds: `models/multi_label_thresholds.pkl`
- Inference metadata: `models/multilabel_inference_artifacts.pkl`

## Data Preprocessing Steps

The multilabel pipeline uses the following preprocessing flow.

### 1. Load the raw dataset

Source file:

- `ClaimDenialInputMultiLabel.csv`

### 2. Build the multilabel target

The pipeline combines:

- `target1`
- `target2`
- `target3`
- `target4`

into a single list of labels for each row.

Then it:

- removes `NaN` target values
- counts label frequency
- keeps only labels with minimum support
- removes rows that end up with zero valid labels

### 3. Convert date fields

The pipeline converts:

- `ServiceDt`
- `ClaimBillDate`

into datetime and creates:

- `service_month`
- `service_day`

### 4. Remove duplicate rows

Duplicates are dropped using:

- `ClientID`
- `CPTCode`
- `ServiceDt`

### 5. Prevent leakage

The training flow removes:

- `DenialFlag`
- `lastActDt`
- the raw target columns

Also, the leaked `label_count` dependency was removed from inference-time features in the current implementation.

### 6. Train/validation split

The pipeline uses:

- `GroupShuffleSplit`

Grouping column:

- `ClientID`

### 7. Feature engineering

Base numeric features:

- `AmountCharged`
- `CoPay`
- `Deduc`
- `CoIns`
- `CltResp`
- `SameDayCli`
- `DaysBetServiceToBilling`
- `service_month`
- `service_day`

Rate-based features:

- `cpt_denial_rate`
- `payer_denial_rate`

High-cardinality fields frequency encoded:

- `Clinic`
- `ClientID`
- `Payer`
- `Provider`
- `BillingProviderNPI`
- `ClaimFacilityNPI`
- `CPTCode`

Regular categorical fields one-hot encoded:

- `Service`
- `eligStatus`
- `tpcliStrModifier`
- `tpcliStrPOS`
- `f21diag1`

### 8. Align inference columns

The final training feature layout is saved and reused at API inference time.

## Model Training Instructions

You can train the multilabel model in either of these two ways.

### Option 1. Run the notebook

Open and run:

- `train_multilabel_pipeline.ipynb`

### Option 2. Run the Python script

```bash
python train_multilabel_pipeline.py
```

### What the training code does

It:

1. loads the CSV
2. builds the multilabel target
3. creates date and aggregation features
4. splits the data using `GroupShuffleSplit`
5. transforms train and validation features
6. trains a `ClassifierChain`
7. tunes one threshold per label
8. exports model and inference artifacts

Model used:

- `ClassifierChain`
- base estimator: `LightGBM LGBMClassifier`

Exported files:

- `models/multilabel_model.pkl`
- `models/multi_label_thresholds.pkl`
- `models/multilabel_inference_artifacts.pkl`

Current metrics:

- Train Micro F1: `0.9430`
- Train Macro F1: `0.8587`
- Validation Micro F1: `0.8864`
- Validation Macro F1: `0.6389`
- Feature count: `162`
- Label count: `24`

## Threshold Selection Logic

This model uses threshold tuning per label.

### How it works

For each label:

1. predict probabilities on the validation set
2. test thresholds from `0.10` to `0.60`
3. compute label-wise F1 score
4. select the threshold with the best F1
5. save all thresholds to `models/multi_label_thresholds.pkl`

### Why this is useful

Different denial reason labels have different class frequencies, so a single shared threshold would not work well for all labels.

## Sample API Request

Endpoint:

- `POST /predict-denial-reasons/`

```json
{
  "claim_features": {
    "Clinic": "CLN_13838227",
    "Service": "Methadone Maintenance Week",
    "AmountCharged": 297.61,
    "CPTCode": "H0020",
    "ClientID": "CLT_85822412",
    "Payer": "PAY_21461370",
    "Provider": "PRV_75825317",
    "BillingProviderNPI": "NPI_21138757",
    "ClaimFacilityNPI": "FAC_72555030",
    "eligStatus": "Not Verified",
    "CoPay": 0,
    "Deduc": 0,
    "CoIns": 0,
    "CltResp": 0,
    "SameDayCli": 0,
    "DaysBetServiceToBilling": 1,
    "tpcliStrModifier": "missing",
    "tpcliStrPOS": 11,
    "f21diag1": "F1120",
    "ServiceDt": "2025-06-22",
    "ClaimBillDate": "2025-04-06"
  }
}
```

## Sample API Response

```json
{
  "labels": ["23.0"],
  "scores": [0.9982],
  "top_predictions": [
    {
      "label": "23.0",
      "probability": 0.9982,
      "threshold": 0.15,
      "predicted": true
    },
    {
      "label": "8.0",
      "probability": 0.0165,
      "threshold": 0.40,
      "predicted": false
    }
  ],
  "warnings": [
    "The multilabel endpoint depends on the exported preprocessing artifacts. If predictions look unstable, retrain with `train_multilabel_pipeline.py` to regenerate the model and inference metadata together."
  ],
  "expected_feature_count": 162,
  "training_metrics": {
    "train_micro_f1": 0.9429961462674595,
    "train_macro_f1": 0.8586824721967901,
    "val_micro_f1": 0.8864475461503827,
    "val_macro_f1": 0.638918153143298,
    "train_rows": 169792,
    "val_rows": 42294,
    "feature_count": 162,
    "label_count": 24
  }
}
```


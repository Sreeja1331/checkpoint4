# Binary Denial Model README

## Overview

This model predicts whether a healthcare claim is likely to be denied.

- Task: binary classification
- Target: `DenialFlag`
- Endpoint: `POST /predict-denial-risk/`
- Model file: `models/binary_model.pkl`
- Feature list: `models/binary_feature_cols.pkl`

## Data Preprocessing Steps

The binary inference flow in `app.py` performs these steps:

### 1. Normalize input keys

The API accepts raw claim input inside:

```json
{
  "claim_features": { ... }
}
```

### 2. Parse date fields

If present, the API reads:

- `ServiceDt`
- `ClaimBillDate`

and derives:

- `days_to_bill`
- `service_day`

### 3. Prepare model-specific numeric features

The binary model uses:

- `AmountCharged`
- `CoPay`
- `Deduc`
- `CoIns`
- `CltResp`
- `SameDayCli`
- `DaysBetServiceToBilling`
- `days_to_bill`
- `service_day`
- `payer_denial_rate`
- `cpt_denial_rate`

### 4. Handle categorical columns

The original training-time encoders were not fully exported for this model, so the API uses a deterministic fallback transformation for:

- `Clinic`
- `Service`
- `CPTCode`
- `Payer`
- `Provider`
- `eligStatus`
- `tpcliStrModifier`
- `tpcliStrPOS`
- `f21diag1`

### 5. Align to the saved feature list

The transformed input is aligned to the exact feature order stored in:

- `models/binary_feature_cols.pkl`

## Model Training Instructions

The training work for this model lives in:

- `binary_denial.ipynb`

To retrain it:

1. open `binary_denial.ipynb`
2. run the notebook cells in order
3. train the binary LightGBM model
4. export the artifacts

Expected exported files:

- `models/binary_model.pkl`
- `models/binary_feature_cols.pkl`

Model used:

- `LightGBM LGBMClassifier`

## Threshold Selection Logic

The binary notebook evaluates multiple probability cutoffs instead of relying on only one default threshold.

Thresholds checked in training include values such as:

- `0.2`
- `0.3`
- `0.4`
- `0.5`
- `0.6`
- `0.7`

In the FastAPI app, the returned response includes:

- probability for class `1`
- boolean output `denial_risk`

Current API logic:

- `denial_risk = confidence >= 0.5`

If you want exact notebook parity, you can update the API cutoff to the threshold that performed best in the training notebook.

## Sample API Request

Endpoint:

- `POST /predict-denial-risk/`

```json
{
  "claim_features": {
    "Clinic": "CLN_13838227",
    "Service": "Methadone Maintenance Week",
    "AmountCharged": 297.61,
    "CPTCode": "H0020",
    "Payer": "PAY_21461370",
    "Provider": "PRV_75825317",
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
    "ClaimBillDate": "2025-04-06",
    "payer_denial_rate": 0.2,
    "cpt_denial_rate": 0.1
  }
}
```

## Sample API Response

```json
{
  "denial_risk": false,
  "confidence": 0.18,
  "class_probabilities": {
    "0": 0.82,
    "1": 0.18
  },
  "warnings": [
    "Training-time label encoders and aggregate rate mappings were not saved in this repo. Raw categorical inputs are converted with a deterministic fallback encoder, so binary predictions are best-effort."
  ]
}
```


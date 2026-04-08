# MultiFlag Model README

## Overview

This model predicts the claim status class.

- Task: multiclass classification
- Target: `MultiFlag`
- Classes: `F`, `N`, `P`, `Z`
- Endpoint: `POST /predict-multiflag/`
- Model file: `models/multiflag_model.pkl`
- Feature list: `models/feature_cols.pkl`

## Data Preprocessing Steps

The multiflag inference flow in `app.py` performs these steps:

### 1. Normalize input keys

The API accepts aliases such as:

- `Deduct` -> `Deduc`
- `Coins` -> `CoIns`
- `SameDayClm` -> `SameDayCli`
- `DaysBestServiceToBilling` -> `DaysBetServiceToBilling`

### 2. Parse date fields

If present, the API reads:

- `ServiceDt`
- `ClaimBillDate`

and derives:

- `service_month`
- `service_day`
- `billing_delay`

### 3. Create engineered features

The API computes:

- `SameDayCli_bin`
- `AmountCharged_log`
- `charge_to_delay_ratio`
- `total_patient_burden`
- `has_patient_burden`
- `is_high_patient_pay`
- `patient_pay_ratio`
- `burden_vs_charge`
- `likely_zero_pay`

### 4. Handle categorical and high-cardinality columns

Categorical columns kept as categories:

- `eligStatus`
- `tpcliStrPOS`
- `f21diag1`

High-cardinality columns fallback to unseen handling when training frequency maps are unavailable:

- `Clinic`
- `Service`
- `CPTCode`
- `Payer`
- `Provider`
- `BillingProviderNPI`
- `ClaimFacilityNPI`

### 5. Align to the saved feature list

The transformed input is aligned to:

- `models/feature_cols.pkl`

## Model Training Instructions

The training work for this model lives in:

- `multi_flag.ipynb`

To retrain it:

1. open `multi_flag.ipynb`
2. run the notebook cells in order
3. train the multiclass LightGBM model
4. export the model and feature list

Expected exported files:

- `models/multiflag_model.pkl`
- `models/feature_cols.pkl`

Model used:

- `LightGBM LGBMClassifier`

## Threshold Selection Logic

This is a multiclass model, so it does not use per-label threshold tuning like the multilabel model.

Prediction logic:

1. compute probability for each class
2. choose the class with the highest probability
3. return that class as `multiflag`

The response also returns:

- `confidence`
- `class_probabilities`
- `transformed_features`

## Sample API Request

Endpoint:

- `POST /predict-multiflag/`

```json
{
  "claim_features": {
    "Clinic": "CLN_13838227",
    "Service": "Methadone Maintenance Week",
    "CPTCode": "H0020",
    "Payer": "PAY_21461370",
    "Provider": "PRV_75825317",
    "BillingProviderNPI": "NPI_21138757",
    "ClaimFacilityNPI": "FAC_72555030",
    "eligStatus": "Not Verified",
    "CoPay": 0,
    "Deduc": 0,
    "CoIns": 0,
    "CltResp": 0,
    "DaysBetServiceToBilling": 1,
    "tpcliStrPOS": 11,
    "f21diag1": "F1120",
    "SameDayCli": 0,
    "AmountCharged": 297.61,
    "ServiceDt": "2025-06-22",
    "ClaimBillDate": "2025-04-06"
  }
}
```

## Sample API Response

```json
{
  "multiflag": "N",
  "confidence": 0.67,
  "class_probabilities": {
    "F": 0.12,
    "N": 0.67,
    "P": 0.02,
    "Z": 0.19
  },
  "transformed_features": {
    "AmountCharged_log": 5.7,
    "service_month": 6.0,
    "service_day": 6.0
  },
  "warnings": []
}
```


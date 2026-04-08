# Claim Denial Prediction System

This project contains an end-to-end machine learning pipeline for healthcare claim prediction using three different models:

- a binary model to predict whether a claim will be denied
- a multiclass model to predict the claim `MultiFlag` status
- a multilabel model to predict one or more denial reason codes

The repository also includes a FastAPI application so you can test all models through HTTP endpoints.

## Project Structure

```text
checkpoint 4/
|-- app.py
|-- train_multilabel_pipeline.py
|-- ClaimDenialInputMultiLabel.csv
|-- binary_denial.ipynb
|-- multi_flag.ipynb
|-- multi_label.ipynb
|-- binary_model_readme.md
|-- multiflag_model_readme.md
|-- multilabel_model_readme.md
|-- multi_label_readme.md
|-- models/
|   |-- binary_model.pkl
|   |-- binary_feature_cols.pkl
|   |-- multiflag_model.pkl
|   |-- feature_cols.pkl
|   |-- multi_label_feature_cols.pkl
|   |-- multilabel_model.pkl
|   |-- multi_label_thresholds.pkl
|   |-- multilabel_inference_artifacts.pkl
```

## What Each Model Does

### 1. Binary Denial Model

- Predicts whether a claim is likely to be denied
- Target: `DenialFlag`
- Endpoint: `POST /predict-denial-risk/`
- Model file: `models/binary_model.pkl`

Detailed notes: [binary_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/binary_model_readme.md)

### 2. MultiFlag Model

- Predicts claim status class
- Classes: `F`, `N`, `P`, `Z`
- Target: `MultiFlag`
- Endpoint: `POST /predict-multiflag/`
- Model file: `models/multiflag_model.pkl`

Detailed notes: [multiflag_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/multiflag_model_readme.md)

### 3. Multilabel Denial Reasons Model

- Predicts multiple denial reason labels for the same claim
- Targets: combined values from `target1`, `target2`, `target3`, `target4`
- Endpoint: `POST /predict-denial-reasons/`
- Model file: `models/multilabel_model.pkl`

Detailed notes: [multilabel_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/multilabel_model_readme.md)

## README Coverage

Each model README now includes these sections based on the implemented code in this repository:

- data preprocessing steps
- model training instructions
- threshold selection logic
- sample API requests and responses

## Dataset

The main dataset file used in this project is:

- `ClaimDenialInputMultiLabel.csv`

Important columns include:

- claim and service features:
  - `Clinic`
  - `Service`
  - `CPTCode`
  - `Payer`
  - `Provider`
  - `BillingProviderNPI`
  - `ClaimFacilityNPI`
- financial features:
  - `AmountCharged`
  - `CoPay`
  - `Deduc`
  - `CoIns`
  - `CltResp`
- date features:
  - `ServiceDt`
  - `ClaimBillDate`
- targets:
  - `DenialFlag`
  - `MultiFlag`
  - `target1`, `target2`, `target3`, `target4`

## Dependencies

The project depends on the following Python packages:

- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `joblib`
- `lightgbm`
- `scikit-learn`
- `pydantic`

Install them with:

```bash
pip install fastapi uvicorn pandas numpy joblib lightgbm scikit-learn pydantic
```

## Recommended Python Version

Recommended:

- Python `3.10` or newer

This project was developed and tested in a Python 3.10 environment.

## How the Implementation Works

### Binary Model Flow

The binary model:

- loads a saved LightGBM classifier
- accepts raw claim features
- derives date-related fields like `days_to_bill`
- aligns features to the saved feature list
- returns denial probability and class probabilities

Note:

- training-time categorical encoders were not fully exported for this model, so categorical handling in the API is best-effort

### MultiFlag Model Flow

The multiflag model:

- loads a saved LightGBM multiclass classifier
- accepts raw claim fields
- creates engineered features such as:
  - `SameDayCli_bin`
  - `AmountCharged_log`
  - `billing_delay`
  - `charge_to_delay_ratio`
  - `total_patient_burden`
  - `likely_zero_pay`
- returns:
  - predicted class
  - confidence
  - class probabilities
  - transformed features used at inference

The API also warns when important high-cardinality values are unseen and mapped to fallback values.

### Multilabel Model Flow

The multilabel model uses:

- `ClassifierChain`
- LightGBM base estimators
- threshold tuning per label
- saved inference metadata for safe API preprocessing

The retraining/export script:

- file: [train_multilabel_pipeline.py](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/train_multilabel_pipeline.py)

This script:

- rebuilds labels from `target1` to `target4`
- filters low-support labels
- uses `GroupShuffleSplit` with `ClientID`
- frequency-encodes high-cardinality columns
- one-hot encodes selected categorical columns
- tunes per-label thresholds on validation data
- exports:
  - `models/multilabel_model.pkl`
  - `models/multi_label_thresholds.pkl`
  - `models/multilabel_inference_artifacts.pkl`

Important fix:

- the leaked `label_count` input was removed from inference features

## Current Metrics

From the current exported multilabel artifacts:

- Train Micro F1: `0.9430`
- Train Macro F1: `0.8587`
- Validation Micro F1: `0.8864`
- Validation Macro F1: `0.6389`
- Training rows: `169792`
- Validation rows: `42294`
- Feature count: `162`
- Label count: `24`

## How To Run the Project

### 1. Open the project folder

Use your terminal inside:

```bash
cd "c:\Users\SreejaChiluveru\Downloads\checkpoint 4"
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn pandas numpy joblib lightgbm scikit-learn pydantic
```

### 3. Run the FastAPI app

```bash
uvicorn app:app --reload
```

After that, open:

```text
http://127.0.0.1:8000/docs
```

That Swagger UI page lets you test all endpoints interactively.

## FastAPI Endpoints

### `GET /`

Basic health-style root endpoint showing available endpoints.

### `GET /health`

Returns:

- API status
- timestamp
- loaded model status
- multilabel feature count

### `POST /predict-denial-risk/`

Binary prediction endpoint.

Request format:

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

### `POST /predict-multiflag/`

Multiclass prediction endpoint.

Request format:

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

### `POST /predict-denial-reasons/`

Multilabel prediction endpoint.

Request format:

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

## How To Retrain the Multilabel Model

Run:

```bash
python train_multilabel_pipeline.py
```

This will regenerate:

- `models/multilabel_model.pkl`
- `models/multi_label_thresholds.pkl`
- `models/multilabel_inference_artifacts.pkl`

## How To Use the Notebooks

The notebooks are still useful for experimentation:

- `binary_denial.ipynb`
- `multi_flag.ipynb`
- `multi_label.ipynb`

Suggested order:

1. `binary_denial.ipynb`
2. `multi_flag.ipynb`
3. `multi_label.ipynb`

These notebooks are useful for:

- exploratory analysis
- feature engineering experiments
- model tuning
- validation review

## Testing Tips

- Use realistic values from `ClaimDenialInputMultiLabel.csv` whenever possible.
- Synthetic category values may be treated as unseen by the model.
- The multiflag endpoint includes `transformed_features` to help debug inference behavior.
- The multilabel endpoint behaves best when the input looks similar to training data.

## Known Limitations

- The binary API uses best-effort handling for some categorical inputs because the original exported encoder artifacts are not fully available.
- The multiflag model can become less informative when most category values are unseen.
- The multilabel model is improved and retrained, but still works best on realistic in-distribution examples.

## Model-Specific Documentation

For deeper details, see:

- [binary_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/binary_model_readme.md)
- [multiflag_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/multiflag_model_readme.md)
- [multilabel_model_readme.md](/c:/Users/SreejaChiluveru/Downloads/checkpoint%204/multilabel_model_readme.md)

## Summary

This repository now contains:

- trained model artifacts
- a working FastAPI app
- a retraining/export script for the multilabel model
- model-specific READMEs
- a project-level implementation README

You can now:

1. install dependencies
pip install -r requirements.txt
2. run `uvicorn app:app --reload`
3. test the endpoints in `/docs`
4. retrain the multilabel model when needed

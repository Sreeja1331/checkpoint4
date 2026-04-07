#  Claim Denial Prediction System (Binary + MultiFlag + Multi-Label)

---

##  Overview

This project builds an end-to-end machine learning system to predict healthcare claim denials. It includes:

* **Binary Classification** → Will the claim be denied?
* **Multi-Class Classification (MultiFlag)** → Claim status (F, P, Z, N)
* **Multi-Label Classification** → Multiple denial reason codes

The system is designed to handle **real-world challenges** such as:

* Data leakage
* High-cardinality categorical features
* Class imbalance
* Multi-label dependencies

---

##  Objectives

* Predict denial risk accurately
* Identify multiple denial reasons per claim
* Build a leakage-free and production-ready pipeline
* Deploy models using FastAPI

---

## Dataset Description

* ~212,000 claim records
* 25+ features including:

  * Financial: `AmountCharged`, `CoPay`, `Deduc`, `CoIns`
  * Categorical: `CPTCode`, `Payer`, `Provider`
  * Temporal: `ServiceDt`, `ClaimBillDate`
* Targets:

  * Binary: `DenialFlag`
  * MultiFlag: `F`, `P`, `Z`, `N`
  * Multi-label: `target1–target4` (24 unique labels)

---

##  Data Preprocessing

The dataset was cleaned and prepared with the following steps:

* Removed identifier columns:

  * `TPCLIID`, `LIATPCLIid`, `ClaimID`, `ClientID`
  * These do not generalize and can cause overfitting

* Removed leakage columns:

  * `lastActDt` (future information)

* Handled missing values:

  * High-missing columns dropped
  * Remaining categorical missing values filled with `"missing"`

* Converted date columns:

  * Extracted `service_month`, `service_day`

* Removed duplicate rows

---

##  Multi-Label Target Construction

* Combined `target1–target4` into a single list per row
* Removed rows with no labels
* Converted to multi-hot encoding using `MultiLabelBinarizer`
* Final label space: **24 labels**

---

##🔍 Exploratory Data Analysis

* Severe class imbalance observed
* Most claims have only 1 label
* Some labels are very rare

### Label Co-occurrence

A co-occurrence matrix showed that several denial codes appear together frequently.

 This indicated **label dependency**, leading to the use of **Classifier Chains**

---

##  Feature Engineering

### Temporal Features

* `DaysBetServiceToBilling`
* `service_month`
* `service_day`

### Entity-Level Aggregations

* `cpt_denial_rate`: Average label count per CPTCode
* `payer_denial_rate`: Average label count per Payer

### Financial Features

* `AmountCharged`, `CoPay`, `Deduc`, `CoIns`

### Encoding Strategy

* Used **frequency encoding**
* Avoided one-hot encoding due to high cardinality

---

##  Train-Test Split

* Used **GroupShuffleSplit (ClientID)** to prevent leakage
* Split:

  * Train: 80%
  * Validation: 20%

### Validation Check

* Label distribution consistent across splits
* No client overlap → no leakage

---

##  Models Implemented

### 1. Classifier Chain (Primary Model)

* Base model: LightGBM
* Captures label dependencies

### 2. Binary Relevance (Baseline)

* Independent model per label
* Implemented using `MultiOutputClassifier`

---

##  Threshold Tuning

Instead of using a fixed threshold:

* Optimized threshold **per label**
* Range tested: `0.1 → 0.6`
* Selected threshold maximizing F1 score

---

##  Results

| Model            | Micro F1  | Macro F1  |
| ---------------- | --------- | --------- |
| Classifier Chain | **0.895** | **0.656** |
| Binary Relevance | 0.877     | 0.614     |

### Key Insight

Classifier Chains outperform Binary Relevance because:

* Labels are not independent
* Dependencies improve prediction quality

---

##  Overfitting Check

| Metric   | Train | Validation |
| -------- | ----- | ---------- |
| Micro F1 | 0.91  | 0.89       |
| Macro F1 | 0.78  | 0.65       |

 Small gap → good generalization

---

##  FastAPI Deployment

### Endpoints

#### 1. `/predict-denial-risk/`

**Input**

```json
{
  "claim_features": { ... }
}
```

**Output**

```json
{
  "denial_risk": true,
  "confidence": 0.87
}
```

---

#### 2. `/predict-multiflag/`

**Output**

```json
{
  "multiflag": "F",
  "confidence": 0.92,
  "class_probabilities": {
    "F": 0.92,
    "P": 0.05,
    "Z": 0.02,
    "N": 0.01
  }
}
```

---

#### 3. `/predict-denial-reasons/`

**Output**

```json
{
  "labels": ["11", "45"],
  "scores": [0.81, 0.63]
}
```

---

##  Project Structure

```
project/
│
├── models/
│   ├── multilabel_model.pkl
│   ├── binary_model.pkl
│   ├── multiflag_model.pkl
│   ├── thresholds.pkl
│
├── notebooks/
│   ├── binary_denial.ipynb
│   ├── multi_flag.ipynb
│   ├── multi_label.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

##  Installation

Install dependencies:

```bash
pip install fastapi uvicorn scikit-learn lightgbm pandas numpy joblib
```

---

##  Running the Project

### Run Training

Execute notebooks in order:

```bash
binary_denial.ipynb
multi_flag.ipynb
multi_label.ipynb
```

---

### Run API

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

##  Key Learnings

* Group-based splitting is critical to prevent leakage
* Multi-label problems require threshold tuning
* Label dependencies significantly improve performance
* Proper encoding avoids overfitting in high-cardinality data

---

##  Conclusion

This project demonstrates a complete machine learning pipeline from raw data to deployment. The system is robust, scalable, and aligned with real-world healthcare claim processing challenges.

---

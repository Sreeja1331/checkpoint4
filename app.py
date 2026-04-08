from __future__ import annotations

from datetime import datetime
from hashlib import md5
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def load_artifact(name: str) -> Any:
    return joblib.load(MODELS_DIR / name)


app = FastAPI(
    title="Claim Denial Prediction API",
    version="1.0.0",
    description=(
        "FastAPI service for the checkpoint 4 claim denial models. "
        "Binary and multiflag predictions are available with best-effort "
        "fallbacks where training-time encoders were not saved."
    ),
)


binary_model = load_artifact("binary_model.pkl")
multiflag_model = load_artifact("multiflag_model.pkl")
multilabel_model = load_artifact("multilabel_model.pkl")

binary_cols = load_artifact("binary_feature_cols.pkl")
multiflag_cols = load_artifact("feature_cols.pkl")
multi_thresholds = load_artifact("multi_label_thresholds.pkl")
multilabel_artifacts = load_artifact("multilabel_inference_artifacts.pkl")

MULTILABEL_FEATURE_COUNT = getattr(multilabel_model.estimators_[0], "n_features_in_", None)
MULTILABEL_FEATURE_COLUMNS = list(multilabel_artifacts["feature_columns"])
MULTILABEL_LABEL_CLASSES = multilabel_artifacts["label_classes"]

BINARY_HASH_COLS = {
    "Clinic",
    "Service",
    "CPTCode",
    "Payer",
    "Provider",
    "eligStatus",
    "tpcliStrModifier",
    "tpcliStrPOS",
    "f21diag1",
}

MULTIFLAG_FREQ_FALLBACK_COLS = {
    "Clinic",
    "Service",
    "CPTCode",
    "Payer",
    "Provider",
    "BillingProviderNPI",
    "ClaimFacilityNPI",
}

MULTIFLAG_CATEGORY_COLS = {"eligStatus", "tpcliStrPOS", "f21diag1"}

ARTIFACT_WARNINGS = {
    "binary": (
        "Training-time label encoders and aggregate rate mappings were not saved "
        "in this repo. Raw categorical inputs are converted with a deterministic "
        "fallback encoder, so binary predictions are best-effort."
    ),
    "multiflag": (
        "Training-time frequency maps for high-cardinality fields were not saved "
        "in this repo. Raw values for those fields fall back to 0.0, which "
        "matches unseen-category handling but may reduce accuracy."
    ),
    "multilabel": (
        "The multilabel endpoint depends on the exported preprocessing artifacts. "
        "If predictions look unstable, retrain with `train_multilabel_pipeline.py` "
        "to regenerate the model and inference metadata together."
    ),
}

INPUT_ALIASES = {
    "Deduct": "Deduc",
    "Deduction": "Deduc",
    "Coins": "CoIns",
    "Coinsurance": "CoIns",
    "SameDayClm": "SameDayCli",
    "SameDayClaim": "SameDayCli",
    "DaysBestServiceToBilling": "DaysBetServiceToBilling",
    "DaysBetweenServiceToBilling": "DaysBetServiceToBilling",
}

MULTIFLAG_RECOMMENDED_FIELDS = [
    "Clinic",
    "Service",
    "CPTCode",
    "Payer",
    "Provider",
    "eligStatus",
    "tpcliStrPOS",
    "f21diag1",
    "AmountCharged",
    "CoPay",
    "Deduc",
    "CoIns",
    "SameDayCli",
    "DaysBetServiceToBilling",
]


class ClaimRequest(BaseModel):
    claim_features: dict[str, Any] = Field(
        ...,
        description="Raw claim fields used to build model input features.",
    )


def parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value in (None, "", 0):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "nan", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_hash_number(value: Any, modulus: int = 100_000) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    digest = md5(text.encode("utf-8")).hexdigest()
    return float(int(digest[:10], 16) % modulus)


def base_frame(claim_features: dict[str, Any]) -> pd.DataFrame:
    claim_features = normalize_input_keys(claim_features)
    return pd.DataFrame([claim_features]).copy()


def normalize_input_keys(claim_features: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(claim_features)
    for source_key, target_key in INPUT_ALIASES.items():
        if source_key in normalized and target_key not in normalized:
            normalized[target_key] = normalized[source_key]
    return normalized


def add_common_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    service_dt = parse_date(df.at[0, "ServiceDt"]) if "ServiceDt" in df.columns else pd.NaT
    bill_dt = parse_date(df.at[0, "ClaimBillDate"]) if "ClaimBillDate" in df.columns else pd.NaT

    if pd.notna(service_dt):
        df["service_month"] = float(service_dt.month)
        df["service_day"] = float(service_dt.dayofweek)
    else:
        df["service_month"] = as_float(df.get("service_month", pd.Series([0])).iloc[0])
        df["service_day"] = as_float(df.get("service_day", pd.Series([0])).iloc[0])

    if pd.notna(service_dt) and pd.notna(bill_dt):
        delay = float((bill_dt - service_dt).days)
        df["days_to_bill"] = delay
        df["billing_delay"] = abs(delay)
    else:
        df["days_to_bill"] = as_float(df.get("days_to_bill", pd.Series([0])).iloc[0])
        df["billing_delay"] = abs(as_float(df.get("billing_delay", pd.Series([0])).iloc[0]))

    return df


def preprocess_binary(claim_features: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = [ARTIFACT_WARNINGS["binary"]]
    df = add_common_date_features(base_frame(claim_features))

    df["payer_denial_rate"] = as_float(df.get("payer_denial_rate", pd.Series([0])).iloc[0])
    df["cpt_denial_rate"] = as_float(df.get("cpt_denial_rate", pd.Series([0])).iloc[0])

    aligned = pd.DataFrame(0.0, index=[0], columns=binary_cols)

    for col in binary_cols:
        if col not in df.columns:
            continue
        value = df.at[0, col]
        if col in BINARY_HASH_COLS:
            aligned.at[0, col] = stable_hash_number(value)
        else:
            aligned.at[0, col] = as_float(value)

    return aligned.astype(float), warnings


def preprocess_multiflag(claim_features: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = [ARTIFACT_WARNINGS["multiflag"]]
    df = add_common_date_features(base_frame(claim_features))
    normalized_input = normalize_input_keys(claim_features)

    missing_fields = [field for field in MULTIFLAG_RECOMMENDED_FIELDS if field not in normalized_input]
    if missing_fields:
        warnings.append(
            "Missing recommended multiflag fields: " + ", ".join(missing_fields) + ". "
            "When many of these are omitted, predictions often collapse to the same class."
        )

    amount = as_float(df.get("AmountCharged", pd.Series([0])).iloc[0])
    copay = as_float(df.get("CoPay", pd.Series([0])).iloc[0])
    deduc = as_float(df.get("Deduc", pd.Series([0])).iloc[0])
    coins = as_float(df.get("CoIns", pd.Series([0])).iloc[0])
    same_day_cli = as_float(df.get("SameDayCli", pd.Series([0])).iloc[0])

    total_burden = copay + deduc + coins
    amount_log = float(np.log1p(max(amount, 0.0)))
    billing_delay = abs(as_float(df.get("billing_delay", pd.Series([0])).iloc[0]))

    df["SameDayCli_bin"] = 1.0 if same_day_cli > 0 else 0.0
    df["AmountCharged_log"] = amount_log
    df["billing_delay"] = billing_delay
    df["charge_to_delay_ratio"] = amount_log / (billing_delay + 1.0)
    df["total_patient_burden"] = total_burden
    df["has_patient_burden"] = 1.0 if total_burden > 0 else 0.0
    df["is_high_patient_pay"] = 1.0 if total_burden > 0 else 0.0
    df["patient_pay_ratio"] = total_burden / (amount_log + 1.0)
    df["burden_vs_charge"] = total_burden / (amount_log + 1.0)
    df["likely_zero_pay"] = 1.0 if df.at[0, "burden_vs_charge"] > 0.8 else 0.0

    aligned = pd.DataFrame(index=[0], columns=multiflag_cols)

    for col in multiflag_cols:
        if col in MULTIFLAG_CATEGORY_COLS:
            value = df.get(col, pd.Series(["Unknown"])).iloc[0]
            aligned[col] = pd.Series([str(value) if value not in (None, "") else "Unknown"], dtype="category")
            continue

        if col in MULTIFLAG_FREQ_FALLBACK_COLS:
            value = df.get(col, pd.Series([0])).iloc[0]
            aligned.at[0, col] = as_float(value, default=0.0) if isinstance(value, (int, float, np.number)) else 0.0
            continue

        aligned.at[0, col] = as_float(df.get(col, pd.Series([0])).iloc[0])

    numeric_cols = [col for col in multiflag_cols if col not in MULTIFLAG_CATEGORY_COLS]
    aligned[numeric_cols] = aligned[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    zeroed_high_card_cols = []
    for col in MULTIFLAG_FREQ_FALLBACK_COLS:
        if col in aligned.columns and float(aligned.at[0, col]) == 0.0:
            zeroed_high_card_cols.append(col)

    if zeroed_high_card_cols:
        warnings.append(
            "These high-cardinality fields were treated as unseen and mapped to 0.0: "
            + ", ".join(zeroed_high_card_cols)
            + ". Predictions may skew toward the model's common classes."
        )

    return aligned, warnings


def preprocess_multilabel(claim_features: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = [ARTIFACT_WARNINGS["multilabel"]]
    df = add_common_date_features(base_frame(claim_features))

    if "ServiceDt" in claim_features:
        service_dt = parse_date(claim_features.get("ServiceDt"))
        if pd.notna(service_dt):
            df["service_day"] = float(service_dt.day)

    for col in multilabel_artifacts["cat_cols"]:
        df[col] = str(df.get(col, pd.Series(["missing"])).iloc[0] or "missing")

    cpt_code = claim_features.get("CPTCode")
    payer = claim_features.get("Payer")
    df["cpt_denial_rate"] = as_float(multilabel_artifacts["cpt_denial_rate"].get(cpt_code, 0.0))
    df["payer_denial_rate"] = as_float(multilabel_artifacts["payer_denial_rate"].get(payer, 0.0))

    for col in multilabel_artifacts["high_card_cols"]:
        raw_value = claim_features.get(col)
        freq_map = multilabel_artifacts["freq_maps"].get(col, {})
        df[f"{col}_freq"] = as_float(freq_map.get(raw_value, 0.0))

    keep_cols = list(multilabel_artifacts["base_feature_cols"]) + [
        f"{col}_freq" for col in multilabel_artifacts["high_card_cols"]
    ] + list(multilabel_artifacts["cat_cols"])

    for col in keep_cols:
        if col not in df.columns:
            df[col] = 0.0 if col not in multilabel_artifacts["cat_cols"] else "missing"

    df = df[keep_cols].copy()
    df = pd.get_dummies(df, columns=multilabel_artifacts["cat_cols"], drop_first=True)
    df = df.reindex(columns=MULTILABEL_FEATURE_COLUMNS, fill_value=0.0)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return df, warnings


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Claim Denial Prediction API is running.",
        "available_endpoints": [
            "/health",
            "/predict-denial-risk/",
            "/predict-multiflag/",
            "/predict-denial-reasons/",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "models_loaded": {
            "binary": True,
            "multiflag": True,
            "multilabel": True,
        },
        "multilabel_expected_features": MULTILABEL_FEATURE_COUNT,
    }


@app.post("/predict-denial-risk/")
def predict_denial_risk(req: ClaimRequest) -> dict[str, Any]:
    features, warnings = preprocess_binary(req.claim_features)
    probs = binary_model.predict_proba(features)[0]
    denial_index = int(np.where(binary_model.classes_ == 1)[0][0])
    confidence = float(probs[denial_index])

    return {
        "denial_risk": bool(confidence >= 0.5),
        "confidence": confidence,
        "class_probabilities": {
            str(label): float(prob) for label, prob in zip(binary_model.classes_, probs)
        },
        "warnings": warnings,
    }


@app.post("/predict-multiflag/")
def predict_multiflag(req: ClaimRequest) -> dict[str, Any]:
    features, warnings = preprocess_multiflag(req.claim_features)
    probs = multiflag_model.predict_proba(features)[0]
    classes = multiflag_model.classes_
    pred_idx = int(np.argmax(probs))
    transformed_features = {}

    for col in features.columns:
        value = features.iloc[0][col]
        if pd.isna(value):
            transformed_features[col] = None
        elif isinstance(value, (np.integer, np.floating)):
            transformed_features[col] = float(value)
        else:
            transformed_features[col] = str(value)

    return {
        "multiflag": str(classes[pred_idx]),
        "confidence": float(probs[pred_idx]),
        "class_probabilities": {
            str(label): float(prob) for label, prob in zip(classes, probs)
        },
        "transformed_features": transformed_features,
        "warnings": warnings,
    }


@app.post("/predict-denial-reasons/")
def predict_denial_reasons(req: ClaimRequest) -> dict[str, Any]:
    features, warnings = preprocess_multilabel(req.claim_features)
    probs = multilabel_model.predict_proba(features)

    if isinstance(probs, list):
        probs = np.column_stack([arr[:, 1] for arr in probs])
    else:
        probs = np.asarray(probs)
        if probs.ndim == 3:
            probs = probs[:, :, 1]
        elif probs.ndim == 1:
            probs = probs.reshape(1, -1)

    labels = []
    scores = []

    for i, probability in enumerate(probs[0]):
        if probability > multi_thresholds[i]:
            labels.append(str(MULTILABEL_LABEL_CLASSES[i]))
            scores.append(float(probability))

    ranked = sorted(
        [
            {
                "label": str(MULTILABEL_LABEL_CLASSES[i]),
                "probability": float(probability),
                "threshold": float(multi_thresholds[i]),
                "predicted": bool(probability > multi_thresholds[i]),
            }
            for i, probability in enumerate(probs[0])
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )

    return {
        "labels": labels,
        "scores": scores,
        "top_predictions": ranked[:10],
        "warnings": warnings,
        "expected_feature_count": MULTILABEL_FEATURE_COUNT,
        "training_metrics": multilabel_artifacts.get("metrics", {}),
    }

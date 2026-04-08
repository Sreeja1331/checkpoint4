from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ClaimDenialInputMultiLabel.csv"
MODELS_DIR = BASE_DIR / "models"

TARGET_COLS = ["target1", "target2", "target3", "target4"]
HIGH_CARD_COLS = [
    "Clinic",
    "ClientID",
    "Payer",
    "Provider",
    "BillingProviderNPI",
    "ClaimFacilityNPI",
    "CPTCode",
]
CAT_COLS = ["Service", "eligStatus", "tpcliStrModifier", "tpcliStrPOS", "f21diag1"]
BASE_FEATURE_COLS = [
    "AmountCharged",
    "CoPay",
    "Deduc",
    "CoIns",
    "CltResp",
    "SameDayCli",
    "DaysBetServiceToBilling",
    "service_month",
    "service_day",
    "cpt_denial_rate",
    "payer_denial_rate",
]
MIN_LABEL_SUPPORT = 30


def build_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
    df = df.copy()
    df["labels"] = df[TARGET_COLS].values.tolist()
    df["labels"] = df["labels"].apply(lambda values: [value for value in values if pd.notna(value)])

    all_labels = pd.Series([label for labels in df["labels"] for label in labels])
    valid_labels = all_labels.value_counts()
    valid_labels = valid_labels[valid_labels >= MIN_LABEL_SUPPORT].index.tolist()

    df["labels"] = df["labels"].apply(lambda values: [value for value in values if value in valid_labels])
    df = df[df["labels"].map(len) > 0].copy()

    return df, valid_labels


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ServiceDt"] = pd.to_datetime(df["ServiceDt"], errors="coerce")
    df["ClaimBillDate"] = pd.to_datetime(df["ClaimBillDate"], errors="coerce")
    df["service_month"] = df["ServiceDt"].dt.month.fillna(0).astype(int)
    df["service_day"] = df["ServiceDt"].dt.day.fillna(0).astype(int)
    return df


def prepare_split_features(
    frame: pd.DataFrame,
    *,
    cpt_rate_map: dict,
    payer_rate_map: dict,
    freq_maps: dict[str, dict],
    dummy_feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = frame.copy()

    df["cpt_denial_rate"] = df["CPTCode"].map(cpt_rate_map).fillna(0.0)
    df["payer_denial_rate"] = df["Payer"].map(payer_rate_map).fillna(0.0)

    for col in HIGH_CARD_COLS:
        mapped = df[col].map(freq_maps[col]).fillna(0.0)
        df[f"{col}_freq"] = mapped.astype(float)

    for col in CAT_COLS:
        df[col] = df[col].fillna("missing").astype(str)

    keep_cols = BASE_FEATURE_COLS + [f"{col}_freq" for col in HIGH_CARD_COLS] + CAT_COLS
    df = df[keep_cols].copy()
    df = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)

    if dummy_feature_columns is None:
        dummy_feature_columns = df.columns.tolist()
    else:
        df = df.reindex(columns=dummy_feature_columns, fill_value=0.0)

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df, dummy_feature_columns


def predict_chain_probabilities(model: ClassifierChain, X: pd.DataFrame) -> np.ndarray:
    probs = model.predict_proba(X)
    if isinstance(probs, list):
        return np.column_stack([arr[:, 1] for arr in probs])
    probs = np.asarray(probs)
    if probs.ndim == 3:
        return probs[:, :, 1]
    return probs


def tune_thresholds(y_true: np.ndarray, probas: np.ndarray) -> list[float]:
    thresholds: list[float] = []
    for i in range(y_true.shape[1]):
        best_threshold = 0.3
        best_f1 = -1.0
        for threshold in np.arange(0.1, 0.61, 0.05):
            pred = (probas[:, i] > threshold).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df, valid_labels = build_labels(df)
    df = add_date_features(df)
    df = df.drop_duplicates(subset=["ClientID", "CPTCode", "ServiceDt"]).copy()

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])

    feature_df = df.drop(columns=TARGET_COLS + ["labels", "DenialFlag", "lastActDt"], errors="ignore").copy()
    groups = feature_df["ClientID"].fillna("missing").astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(feature_df, Y, groups=groups))

    X_train_raw = feature_df.iloc[train_idx].copy()
    X_val_raw = feature_df.iloc[val_idx].copy()
    Y_train = Y[train_idx]
    Y_val = Y[val_idx]

    train_label_count = Y_train.sum(axis=1)
    train_frame_for_rates = X_train_raw.copy()
    train_frame_for_rates["label_count"] = train_label_count

    cpt_rate_map = train_frame_for_rates.groupby("CPTCode")["label_count"].mean().to_dict()
    payer_rate_map = train_frame_for_rates.groupby("Payer")["label_count"].mean().to_dict()
    freq_maps = {
        col: X_train_raw[col].value_counts().to_dict()
        for col in HIGH_CARD_COLS
    }

    X_train, feature_columns = prepare_split_features(
        X_train_raw,
        cpt_rate_map=cpt_rate_map,
        payer_rate_map=payer_rate_map,
        freq_maps=freq_maps,
    )
    X_val, _ = prepare_split_features(
        X_val_raw,
        cpt_rate_map=cpt_rate_map,
        payer_rate_map=payer_rate_map,
        freq_maps=freq_maps,
        dummy_feature_columns=feature_columns,
    )

    base_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    order = np.argsort(-Y_train.sum(axis=0))
    model = ClassifierChain(base_model, order=order)
    model.fit(X_train, Y_train)

    val_probas = predict_chain_probabilities(model, X_val)
    thresholds = tune_thresholds(Y_val, val_probas)
    val_pred = np.column_stack(
        [(val_probas[:, i] > thresholds[i]).astype(int) for i in range(len(thresholds))]
    )

    train_probas = predict_chain_probabilities(model, X_train)
    train_pred = np.column_stack(
        [(train_probas[:, i] > thresholds[i]).astype(int) for i in range(len(thresholds))]
    )

    metrics = {
        "train_micro_f1": float(f1_score(Y_train, train_pred, average="micro", zero_division=0)),
        "train_macro_f1": float(f1_score(Y_train, train_pred, average="macro", zero_division=0)),
        "val_micro_f1": float(f1_score(Y_val, val_pred, average="micro", zero_division=0)),
        "val_macro_f1": float(f1_score(Y_val, val_pred, average="macro", zero_division=0)),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "feature_count": int(len(feature_columns)),
        "label_count": int(len(mlb.classes_)),
    }

    inference_artifacts = {
        "feature_columns": feature_columns,
        "label_classes": mlb.classes_.tolist(),
        "high_card_cols": HIGH_CARD_COLS,
        "cat_cols": CAT_COLS,
        "cpt_denial_rate": cpt_rate_map,
        "payer_denial_rate": payer_rate_map,
        "freq_maps": freq_maps,
        "base_feature_cols": BASE_FEATURE_COLS,
        "metrics": metrics,
    }

    joblib.dump(model, MODELS_DIR / "multilabel_model.pkl")
    joblib.dump(thresholds, MODELS_DIR / "multi_label_thresholds.pkl")
    joblib.dump(inference_artifacts, MODELS_DIR / "multilabel_inference_artifacts.pkl")

    print("Saved leakage-free multilabel artifacts.")
    print(metrics)


if __name__ == "__main__":
    main()

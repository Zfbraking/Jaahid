# ============================
# === ML EXTENSION: START  ===
# ============================
"""
Lightweight ML tool for training and prediction on Excel/CSV data.

- Tool name: ml_modeler
- Modes:
    - "train": trains a RandomForest model on a target column
    - "predict": predicts for a new input using a model_id

If you want to remove ML completely:
    1. Delete this file.
    2. Remove ml_modeler registration from mcp_server.py.
    3. Remove /ml route and ml.html usage from app.py/templates.
"""

from typing import Dict, Any
import uuid

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tools.mcp_tooling import MCPTool

# In-memory model registry (model_id -> model state)
ML_MODELS: Dict[str, Dict[str, Any]] = {}


def build_ml_tool() -> MCPTool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        mode = (payload.get("mode") or "").lower()

        if mode == "train":
            return _train_model(payload)
        elif mode == "predict":
            return _predict(payload)
        else:
            return {"error": "Invalid mode. Use 'train' or 'predict'."}

    return MCPTool(
        name="ml_modeler",
        description="Train and use ML models on Excel/CSV data.",
        func=_run,
    )


def _read_tabular(file_path: str) -> pd.DataFrame:
    """Read Excel or CSV based on extension."""
    lower = file_path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        return pd.read_excel(file_path)


def _train_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    file_path = payload.get("file_path")
    target_col = payload.get("target_column")

    if not file_path:
        return {"error": "file_path is required for training."}
    if not target_col:
        return {"error": "target_column is required for training."}

    try:
        df = _read_tabular(file_path)
    except Exception as e:
        return {"error": f"Failed to read file for ML: {e}"}

    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in file."}

    # Drop rows where target is NaN
    df = df.dropna(subset=[target_col])
    if df.empty:
        return {"error": "No rows with non-null target values."}

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop completely empty feature columns
    X = X.dropna(axis=1, how="all")

    # Decide regression vs classification
    is_regression = pd.api.types.is_numeric_dtype(y)

    label_encoder = None
    if not is_regression:
        # Classification: encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    # One-hot encode categorical features
    X_processed = pd.get_dummies(X, drop_first=True)

    if X_processed.empty:
        return {"error": "No usable feature columns after preprocessing."}

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    if is_regression:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    score = float(model.score(X_test, y_test)) if len(X_test) > 0 else None

    model_id = str(uuid.uuid4())
    ML_MODELS[model_id] = {
        "model": model,
        "feature_columns": list(X_processed.columns),
        "is_regression": is_regression,
        "label_encoder": label_encoder,
        "target_column": target_col,
    }

    return {
        "status": "trained",
        "model_id": model_id,
        "is_regression": is_regression,
        "target_column": target_col,
        "score": score,
        "feature_columns": ML_MODELS[model_id]["feature_columns"],
    }


def _predict(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = payload.get("model_id")
    input_data = payload.get("input")

    if not model_id:
        return {"error": "model_id is required for prediction."}
    if input_data is None:
        return {"error": "input is required for prediction."}

    state = ML_MODELS.get(model_id)
    if state is None:
        return {"error": f"No model found for model_id={model_id}. Train a model first."}

    model = state["model"]
    feature_columns = state["feature_columns"]
    is_regression = state["is_regression"]
    label_encoder: LabelEncoder = state["label_encoder"]
    target_col = state["target_column"]

    if not isinstance(input_data, dict):
        return {"error": "input must be a JSON object of feature_name -> value."}

    # Build single-row DataFrame
    raw_df = pd.DataFrame([input_data])

    # Same preprocessing as training
    processed = pd.get_dummies(raw_df, drop_first=True)

    # Align columns
    for col in feature_columns:
        if col not in processed.columns:
            processed[col] = 0
    processed = processed[feature_columns]

    try:
        preds = model.predict(processed)
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    if is_regression:
        pred_value = float(preds[0])
        return {
            "model_id": model_id,
            "target_column": target_col,
            "prediction": pred_value,
            "is_regression": True,
        }
    else:
        class_idx = int(preds[0])
        if label_encoder is not None:
            class_label = label_encoder.inverse_transform([class_idx])[0]
        else:
            class_label = str(class_idx)

        proba = None
        if hasattr(model, "predict_proba"):
            proba_arr = model.predict_proba(processed)[0]
            proba = [float(p) for p in proba_arr]

        return {
            "model_id": model_id,
            "target_column": target_col,
            "predicted_class": class_label,
            "class_index": class_idx,
            "probabilities": proba,
            "is_regression": False,
        }

# ==========================
# === ML EXTENSION: END  ===
# ==========================

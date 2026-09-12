"""Train and evaluate the transit demand forecasting model."""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transit_demand.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "demand_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def train():
    df = pd.read_csv(DATA)

    features = ["route", "hour", "weekday", "is_weekend", "weather"]
    target = "passengers"

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.20, random_state=42
    )

    categorical = ["route", "weather"]
    numeric = ["hour", "weekday", "is_weekend"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(float(mean_squared_error(y_test, pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, pred)), 4),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Model trained.")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    train()

import os
import json
import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

DATA_URL = (
    "https://raw.githubusercontent.com/zygmuntz/steam-games-dataset/master/games.csv"
)
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(pd.io.common.BytesIO(r.content))


def owners_to_midpoint(s: str) -> float:
    # owners often like "20000-50000"
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    # expected columns in many steam datasets:
    # price, owners, positive_ratings, negative_ratings, average_playtime, achievements
    needed = ["price", "owners", "positive_ratings", "negative_ratings", "average_playtime", "achievements"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing columns: {missing}. "
            f"Try a different DATA_URL that contains these fields."
        )

    df = df.copy()
    df["owners_mid"] = df["owners"].apply(owners_to_midpoint)

    # keep only target + 5 features
    keep = ["price", "owners_mid", "positive_ratings", "negative_ratings", "average_playtime", "achievements"]
    df = df[keep]

    # price should be numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # basic cleaning
    df = df.dropna(subset=["price"])
    df = df[df["price"] >= 0]

    return df


def train_and_save(random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_raw = download_csv(DATA_URL)
    df = prepare(df_raw)

    X = df[["owners_mid", "positive_ratings", "negative_ratings", "average_playtime", "achievements"]].values
    y = df["price"].values

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "dataset_url": DATA_URL,
        "rows_used": int(len(df)),
        "features": ["owners_mid", "positive_ratings", "negative_ratings", "average_playtime", "achievements"],
        "model": "Ridge Regression (multiple regression)",
        "rmse": rmse,
        "r2": r2,
        "random_state": random_state,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print("Training done. Saved to:", MODEL_PATH)
    print("Metrics:", m)

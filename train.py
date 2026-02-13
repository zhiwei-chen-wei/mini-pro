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


DATA_URL = "https://raw.githubusercontent.com/zhiwei-chen-wei/game-price-data/main/synthetic_game_data.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")     # ใช้ .pkl ให้ตรง error ที่คุณเห็น
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

FEATURES = [
    "release_year",
    "avg_playtime",
    "metacritic_score",
    "number_of_achievements",
    "multiplayer",
]
TARGET = "price"


def download_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(pd.io.common.BytesIO(r.content))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed = FEATURES + [TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    # coerce to numeric
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep valid rows
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] >= 0]

    # multiplayer should be 0/1
    df["multiplayer"] = df["multiplayer"].clip(0, 1)

    # drop rows with missing features
    df = df.dropna(subset=FEATURES)

    return df[needed]


def compute_year_caps(df: pd.DataFrame, q: float = 0.99) -> dict:
    """
    Auto cap แบบ B: เพดานตามช่วงปี
    - pre2015: < 2015
    - 2015_2019: 2015-2019
    - 2020plus: >= 2020
    ใช้ quantile (เช่น p99) ของราคาในแต่ละ bucket
    """
    buckets = {
        "pre2015": df["release_year"] < 2015,
        "2015_2019": (df["release_year"] >= 2015) & (df["release_year"] <= 2019),
        "2020plus": df["release_year"] >= 2020,
    }

    caps = {}
    for key, mask in buckets.items():
        vals = df.loc[mask, TARGET].dropna().astype(float).values
        caps[key] = float(np.quantile(vals, q)) if len(vals) else None

    return caps


def train_and_save(random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_raw = download_csv(DATA_URL)
    df = prepare(df_raw)

    X = df[FEATURES].values
    y = df[TARGET].values

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

    # ✅ caps แบบ B (p99)
    caps = compute_year_caps(df, q=0.99)

    joblib.dump(model, MODEL_PATH)

    metrics = {
        "dataset_url": DATA_URL,
        "rows_used": int(len(df)),
        "target": TARGET,
        "features": FEATURES,
        "model": "Ridge Regression (multiple regression)",
        "rmse": rmse,
        "r2": r2,
        "random_state": random_state,

        # ✅ สำหรับ auto cap
        "price_floor_usd": 0.0,
        "caps_usd_by_year_bucket_p99": caps,
        "cap_quantile": 0.99,
        "year_buckets": {"pre2015": "<2015", "2015_2019": "2015-2019", "2020plus": ">=2020"},
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    print(train_and_save())

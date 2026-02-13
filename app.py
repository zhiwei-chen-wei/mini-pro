import os
import json
import joblib
import numpy as np
import streamlit as st


MODEL_PATH = "models/model.pkl"
METRICS_PATH = "models/metrics.json"


USD_TO_THB = float(os.getenv("USD_TO_THB", "35.0"))

MIN_PRICE_USD = 0.0
MAX_PRICE_USD = 70.0


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


# ---------------- Page config ----------------
st.set_page_config(page_title="Game Price Predictor", layout="centered")

# ---------------- UI CSS (Dark + White Header Card) ----------------
st.markdown(
    """
<style>
.stApp{
  background: radial-gradient(1200px 600px at 30% 0%, rgba(90,110,255,0.12), transparent 60%),
              radial-gradient(900px 500px at 70% 10%, rgba(120,90,255,0.10), transparent 55%),
              linear-gradient(180deg, #0B0F1A 0%, #0A0D14 70%, #090B10 100%);
  color: #E8ECF3;
}
header, footer {visibility: hidden;}
.block-container { max-width: 720px; }

/* White header card */
.header-card {
  background: #FFFFFF;
  border-radius: 18px;
  padding: 26px 28px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.45);
  border: 1px solid rgba(0,0,0,0.06);
  margin: 18px auto 22px auto;
  max-width: 720px;
}

/* ✅ Make header text black */
.header-card * { color: #0E1117 !important; }
.header-title { font-size: 34px; font-weight: 900; margin: 0; }
.header-subtitle { font-size: 13px; opacity: 0.75; margin-top: 6px; }

/* Inputs: long dark bars */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
}
input, textarea { color: #E8ECF3 !important; }
div[data-baseweb="select"] span { color: #E8ECF3 !important; }
label, .stMarkdown, .stCaption { color: #E8ECF3 !important; }

/* Predict button */
.stButton > button {
  width: 100%;
  background: linear-gradient(90deg, #2E6BFF 0%, #1C51D9 100%) !important;
  color: white !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 12px 18px !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 25px rgba(46,107,255,0.25);
}

/* Result card */
.result {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 18px 18px;
  margin-top: 16px;
}
.result h3 { margin: 0 0 8px 0; color: #E8ECF3 !important; }
.result p { margin: 6px 0; color: #E8ECF3 !important; }
.smallmuted { opacity: 0.75; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown(
    """
<div class="header-card">
  <div class="header-title">🎮 Game Price Predictor</div>
  <div class="header-subtitle">
    Multiple Regression · 5 features · Price output clamped to $0–$70
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Load model/metrics ----------------
metrics = load_metrics()

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found: {MODEL_PATH}")
    st.caption("Run training first to generate models/model.pkl")
    st.stop()

model = load_model()

# ---------------- Inputs (NO slicers) ----------------
release_year = st.number_input("Release year", min_value=2000, max_value=2026, value=2020, step=1)
avg_playtime = st.number_input("Avg playtime (hours)", min_value=0.0, max_value=300.0, value=40.0, step=1.0, format="%.2f")
metacritic_score = st.number_input("Metacritic score", min_value=0.0, max_value=100.0, value=75.0, step=1.0, format="%.2f")
number_of_achievements = st.number_input("Number of achievements", min_value=0, max_value=300, value=70, step=1)
multiplayer = st.selectbox("Multiplayer", ["Yes", "No"], index=0)

multiplayer_val = 1 if multiplayer == "Yes" else 0
show_raw = st.checkbox("Show raw prediction (before clamp)", value=True)

# ---------------- Predict ----------------
if st.button("Predict Game Price"):
    x = np.array([[
        float(release_year),
        float(avg_playtime),
        float(metacritic_score),
        float(number_of_achievements),
        float(multiplayer_val)
    ]], dtype=float)

    raw_usd = float(model.predict(x)[0])

    # ✅ Force output to [0, 70]
    pred_usd = clamp(raw_usd, MIN_PRICE_USD, MAX_PRICE_USD)
    pred_thb = pred_usd * USD_TO_THB

    st.markdown(
        f"""
<div class="result">
  <h3>Prediction</h3>
  <p><b>Price (THB):</b> ฿{pred_thb:,.2f}</p>
  <p><b>Price (USD):</b> {pred_usd:.2f}</p>
  <p class="smallmuted">Range forced: ${MIN_PRICE_USD:.0f}–${MAX_PRICE_USD:.0f} · USD→THB: {USD_TO_THB:.2f}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    if show_raw:
        st.caption(f"Raw prediction (before clamp): {raw_usd:.2f} USD")

# ---------------- Optional footer metrics ----------------
if metrics and ("rmse" in metrics) and ("r2" in metrics):
    st.markdown(f"**RMSE:** {metrics['rmse']:.2f} | **R²:** {metrics['r2']:.3f}")

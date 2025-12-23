import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
#import pickle 
#from sb3_contrib import RecurrentPPO


# -------------------------------
# PROJECT PATH SETUP (CRITICAL)
# -------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "results"
REPORTS_DIR = ROOT_DIR / "reports" / "tables"


# -------------------------------
# NEW PATHS FOR REAL-TIME MODE
# -------------------------------
CLEAN_DIR = ROOT_DIR / "data" / "cleaned"
MODEL_DIR = ROOT_DIR / "data" / "models"



@st.cache_resource
def load_arima_model():
    with open(MODEL_DIR / "arima_non_rolling.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_lstm_bundle():
    model = pickle.load(open(MODEL_DIR / "lstm_non_rolling.pkl", "rb"))
    scaler_X = pickle.load(open(MODEL_DIR / "lstm_scaler_X.pkl", "rb"))
    scaler_y = pickle.load(open(MODEL_DIR / "lstm_scaler_y.pkl", "rb"))
    return model, scaler_X, scaler_y

@st.cache_resource
def load_drl_bundle():
    model = RecurrentPPO.load(MODEL_DIR / "DE_drl_recurrentppo.zip")
    scaler_X = pickle.load(open(MODEL_DIR / "DE_drl_scaler_X.pkl", "rb"))
    scaler_y = pickle.load(open(MODEL_DIR / "DE_drl_scaler_y.pkl", "rb"))
    return model, scaler_X, scaler_y

TARGET_COL = "DE_load_actual_entsoe_transparency"
TARGET_COL2 = "AT_load_actual_entsoe_transparency"
TARGET_COL3 = "BE_load_actual_entsoe_transparency"
TARGET_COL4 = "BG_load_actual_entsoe_transparency"

def prepare_features(df):
    df = df.select_dtypes(include=[np.number])
    y = df[TARGET_COL2].values
    X = df.drop(columns=[TARGET_COL]).values
    return X, y

def get_last_window(df, target_col, window=24):
    df = df.select_dtypes(include=[np.number])
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X[-window:], y[-window:]


# -------------------------------
# BASIC PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart Grid DRL Forecasting Dashboard",
    layout="wide"
)

# -------------------------------
# LOAD PREDICTIONS & METRICS
# -------------------------------
@st.cache_data
def load_predictions():
    try:
        arima_preds = np.load(DATA_DIR / "DE_arima_rolling_preds.npy")
        lstm_preds  = np.load(DATA_DIR / "DE_lstm_rolling_preds.npy")
        drl_preds   = np.load(DATA_DIR / "DE_drl_external_preds.npy")
        actuals     = np.load(DATA_DIR / "DE_drl_external_actuals.npy")
    except FileNotFoundError as e:
        st.error(f"Missing data file: {e.filename}")
        st.stop()

    min_len = min(len(arima_preds), len(lstm_preds), len(drl_preds), len(actuals))
    return (
        arima_preds[:min_len],
        lstm_preds[:min_len],
        drl_preds[:min_len],
        actuals[:min_len],
    )

@st.cache_data
def load_results_tables():
    baseline = rq2 = rq3 = None

    try:
        baseline = pd.read_csv(REPORTS_DIR / "baseline_results_DE.csv")
    except FileNotFoundError:
        pass

    try:
        rq2 = pd.read_csv(REPORTS_DIR / "rq2_external_feature_comparison.csv")
    except FileNotFoundError:
        pass

    try:
        rq3 = pd.read_csv(REPORTS_DIR / "rq3_computational_feasibility.csv")
    except FileNotFoundError:
        pass

    return baseline, rq2, rq3


arima_raw, lstm_raw, drl_raw, actual_raw = load_predictions()
baseline_results, rq2_results, rq3_results = load_results_tables()

# -------------------------------
# SIDEBAR – CONTROLS
# -------------------------------
st.sidebar.header("Dashboard Controls")

model_choice = st.sidebar.selectbox(
    "Select Forecasting Model:",
    ["ARIMA (Rolling)", "LSTM (Rolling)", "DRL PPO"]
)

window = st.sidebar.slider("Plot window (last N points)", 50, 1000, 300)

simulate = st.sidebar.checkbox("Enable real-time simulation mode", value=False)
noise_level = st.sidebar.slider("Simulation noise level", 0.0, 200.0, 50.0)

alert_threshold = st.sidebar.slider(
    "Load spike / error alert threshold (MW)", 0.0, 1000.0, 300.0
)

theme = st.sidebar.radio("Plot theme", ["Dark", "Light"], index=0)
plt.style.use("dark_background" if theme == "Dark" else "default")


st.sidebar.markdown("---")
st.sidebar.subheader("Real-Time Forecasting")

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon (Hours)",
    min_value=1,
    max_value=72,
    value=24,
    step=1
)


available_datasets = sorted([
    f.name for f in CLEAN_DIR.glob("*_cleaned.csv")
])

selected_dataset = st.sidebar.selectbox(
    "Select Cleaned Dataset",
    available_datasets
)

run_realtime = st.sidebar.button("▶ Run Real-Time Forecast")


@st.cache_data
def load_cleaned_dataset(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df



# -------------------------------
# SESSION STATE
# -------------------------------
if "actuals" not in st.session_state:
    st.session_state.actuals = actual_raw.copy()
    st.session_state.arima = arima_raw.copy()
    st.session_state.lstm = lstm_raw.copy()
    st.session_state.drl = drl_raw.copy()

if model_choice == "ARIMA (Rolling)":
    forecast = st.session_state.arima
elif model_choice == "LSTM (Rolling)":
    forecast = st.session_state.lstm
else:
    forecast = st.session_state.drl

actuals = st.session_state.actuals

# -------------------------------
# SIMULATION MODE
# -------------------------------
if simulate:
    new_actual = actuals[-1] + np.random.normal(0, noise_level)
    new_forecast = forecast[-1] + np.random.normal(0, noise_level * 0.5)

    st.session_state.actuals = np.append(actuals, new_actual)

    if model_choice == "ARIMA (Rolling)":
        st.session_state.arima = np.append(forecast, new_forecast)
        forecast = st.session_state.arima
    elif model_choice == "LSTM (Rolling)":
        st.session_state.lstm = np.append(forecast, new_forecast)
        forecast = st.session_state.lstm
    else:
        st.session_state.drl = np.append(forecast, new_forecast)
        forecast = st.session_state.drl

    actuals = st.session_state.actuals

if run_realtime:
    df_live = load_cleaned_dataset(CLEAN_DIR / selected_dataset)
    X, y = prepare_features(df_live)

#    if model_choice == "ARIMA (Rolling)":
#        arima = load_arima_model()
#        preds = arima.forecast(steps=len(y))
#        st.session_state.actuals = y
#        st.session_state.arima = preds

    if model_choice == "ARIMA (Rolling)":
        arima = load_arima_model()

        # Forecast NEXT N steps
        future_preds = arima.forecast(steps=forecast_horizon)

        st.session_state.actuals = np.array([])
        st.session_state.arima = future_preds

#    elif model_choice == "LSTM (Rolling)":
#        model, scaler_X, scaler_y = load_lstm_bundle()
#        X_sc = scaler_X.transform(X)
#        preds_sc = model.predict(X_sc)
#        preds = scaler_y.inverse_transform(
#            preds_sc.reshape(-1,1)
#        ).ravel()

    elif model_choice == "LSTM (Rolling)":
        model, scaler_X, scaler_y = load_lstm_bundle()

        X_last, y_last = get_last_window(
            df_live, TARGET_COL2, window=24
        )

        X_sc = scaler_X.transform(X_last)

        preds = []
        seq = X_sc.copy()

        for _ in range(forecast_horizon):
            X_input = seq[-24:].reshape(1, 24, -1)
            pred_sc = model.predict(X_input, verbose=0)
            pred = scaler_y.inverse_transform(pred_sc)[0, 0]

            preds.append(pred)

            # shift window
            new_row = seq[-1].copy()
            new_row[0] = pred_sc[0, 0]  # inject predicted load
            seq = np.vstack([seq, new_row])

        st.session_state.actuals = np.array([])
        st.session_state.lstm = np.array(preds)
    

            #st.session_state.actuals = y
            #st.session_state.lstm = preds

    else:  # DRL PPO
        model, scaler_X, scaler_y = load_drl_bundle()

        X_last, _ = get_last_window(
            df_live, TARGET_COL2, window=24
        )

        X_sc = scaler_X.transform(X_last)

        obs = X_sc.reshape(1, -1)
        preds = []

        for _ in range(forecast_horizon):
            action, _ = model.predict(obs, deterministic=True)
            pred_sc = action[0]
            pred = scaler_y.inverse_transform([[pred_sc]])[0, 0]

            preds.append(pred)

            # roll window
            X_sc = np.roll(X_sc, -1, axis=0)
            X_sc[-1, 0] = pred_sc
            obs = X_sc.reshape(1, -1)

        st.session_state.actuals = np.array([])
        st.session_state.drl = np.array(preds)



# -------------------------------
# METRICS
# -------------------------------
min_len = min(len(actuals), len(forecast))
actuals = actuals[:min_len]
forecast = forecast[:min_len]

errors = actuals - forecast
mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))

# -------------------------------
# MAIN LAYOUT
# -------------------------------
st.title("⚡ Smart Grid DRL Forecasting Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast View",
    "📉 Error Analysis",
    "🤖 Model Stats & Feasibility",
    "📊 Research Metrics (RQ1–RQ3)"
])

# -------------------------------
# TAB 1
# -------------------------------
with tab1:
    fig, ax = plt.subplots(figsize=(14, 5))

    if len(actuals) > 0:
        ax.plot(actuals, label="Historical Load", linewidth=2)

    ax.plot(
        range(len(actuals), len(actuals) + len(forecast)),
        forecast,
        label=f"{model_choice} – Future Forecast",
        linestyle="--"
    )

    ax.set_xlabel("Time Steps (Hours)")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


with tab1:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals[-window:], label="Actual Load", linewidth=2)
    ax.plot(forecast[-window:], label=model_choice, linestyle="--")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    #latest_error = abs(actuals[-1] - forecast[-1])
    #st.metric("Live MAE", f"{mae:.2f}")
    #st.metric("Live RMSE", f"{rmse:.2f}")

    latest_error = abs(actuals[-1] - forecast[-1])

    col1, col2, col3 = st.columns(3)

    col1.metric("Live MAE", f"{mae:.2f}")
    col2.metric("Live RMSE", f"{rmse:.2f}")
    col3.metric("Latest Error", f"{latest_error:.2f}")


    if latest_error > alert_threshold:
        st.error("⚠ High forecast error detected")
    else:
        st.success("✅ Forecast within acceptable range")



# -------------------------------
# TAB 2
# -------------------------------
with tab2:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.abs(errors[-window:]))
    ax.set_title("Absolute Error")
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=50)
    ax.set_title("Error Distribution")
    st.pyplot(fig)

# -------------------------------
# TAB 3
# -------------------------------
with tab3:
    if rq3_results is not None:
        st.subheader("Computational Feasibility Metrics (RQ3)")

    # ---- TABLE ----
        st.dataframe(rq3_results, use_container_width=True)

    # ---- TRAINING TIME BAR CHART ----
        st.markdown("### ⏱ Training Time Comparison")

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(
            rq3_results["Model"],
            rq3_results["Training Time (s)"],
            color=["steelblue", "orange", "green"]
        )
        ax1.set_ylabel("Training Time (seconds)")
        ax1.set_xlabel("Model")
        ax1.set_title("Model Training Time Comparison")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig1)

    # ---- INFERENCE TIME BAR CHART ----
        st.markdown("### ⚡ Inference Time Comparison")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(
            rq3_results["Model"],
            rq3_results["Inference Time (s)"],
            color=["steelblue", "orange", "green"]
        )
        ax2.set_ylabel("Inference Time (seconds)")
        ax2.set_xlabel("Model")
        ax2.set_title("Model Inference Speed Comparison")
        ax2.grid(axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig2)

    else:
        st.info("RQ3 computational results not found.")



# -------------------------------
# TAB 4
# -------------------------------
with tab4:
    if baseline_results is not None:
        st.subheader("Baseline Results (RQ1)")
        st.dataframe(baseline_results,use_container_width=True)
    if rq2_results is not None:
        st.subheader("External Variable Impact (RQ2)")
        st.dataframe(rq2_results,use_container_width=True)

# -------------------------------
# MANUAL REFRESH
# -------------------------------
if st.sidebar.button("🔄 Refresh Dashboard"):
    st.experimental_rerun()

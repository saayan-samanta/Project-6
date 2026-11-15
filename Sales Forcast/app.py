import os
import joblib
import json
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="STORE | Sales Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# THEME SWITCHER
# ==========================================

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ==========================================
# THEME COLORS
# ==========================================

if st.session_state.dark_mode:

    BG = "#0B1120"
    SIDEBAR = "#111B2E"
    CARD = "#172338"
    CARD_ALT = "#1D2C43"
    TEXT = "#F1F5F9"
    MUTED = "#94A3B8"
    BORDER = "#293B55"
    INPUT_BG = "#172338"

else:

    BG = "#F4F7FB"
    SIDEBAR = "#FFFFFF"
    CARD = "#FFFFFF"
    CARD_ALT = "#EEF4FA"
    TEXT = "#172554"
    MUTED = "#64748B"
    BORDER = "#DCE5F0"
    INPUT_BG = "#FFFFFF"


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    f"""
    <style>

    /* Main application */

    .stApp {{
        background: {BG};
        color: {TEXT};
        transition: background 0.4s ease;
    }}

    [data-testid="stSidebar"] {{
        background: {SIDEBAR};
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    /* General text */

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}

    h1, h2, h3, h4, p, label {{
        color: {TEXT};
    }}

    /* Main container */

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}

    /* Hero section */

    .hero {{
        background: linear-gradient(
            120deg,
            #123B72 0%,
            #176B87 55%,
            #0F9B83 100%
        );

        padding: 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 28px;

        animation: fadeIn 0.8s ease;
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }}

    .hero h1 {{
        color: white !important;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 8px;
    }}

    .hero p {{
        color: #DCEAFE !important;
        font-size: 15px;
    }}

    .hero-label {{
        display: inline-block;
        background: rgba(255,255,255,0.17);
        padding: 7px 13px;
        border-radius: 20px;
        font-size: 12px;
        margin-bottom: 12px;
        letter-spacing: 1px;
    }}

    /* Cards */

    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        padding: 20px;
        border-radius: 16px;
        min-height: 120px;

        transition: transform 0.25s ease,
                    box-shadow 0.25s ease;

        animation: fadeIn 0.8s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 25px rgba(0,0,0,0.12);
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 600;
    }}

    .metric-value {{
        color: {TEXT};
        font-size: 25px;
        font-weight: 800;
        margin-top: 10px;
    }}

    .section-heading {{
        color: {TEXT};
        font-size: 21px;
        font-weight: 750;
        margin-top: 25px;
        margin-bottom: 15px;
    }}

    .info-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 15px;
    }}

    /* Sidebar branding */

    .brand {{
        font-size: 25px;
        font-weight: 850;
        color: #16A6A1;
        padding: 10px 0;
    }}

    .sidebar-caption {{
        color: {MUTED};
        font-size: 12px;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
    }}

    /* Sidebar section headings */

     [data-testid="stSidebar"] h1,
     [data-testid="stSidebar"] h2,
     [data-testid="stSidebar"] h3,
     [data-testid="stSidebar"] h4 {{

     color: #16A6A1 !important;
          font-weight: 750 !important;
     }}

     /* Sidebar Markdown text */

     [data-testid="stSidebar"] .stMarkdown p {{

          color: {TEXT} !important;
     }}

    /* Buttons */

    .stButton > button {{
        background: linear-gradient(110deg, #2563EB, #0D9488);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 700;
        transition: all 0.25s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37,99,235,0.25);
        color: white;
    }}

    /* Inputs */

    [data-baseweb="select"],
    [data-baseweb="input"],
    [data-baseweb="textarea"] {{
        background: {INPUT_BG};
    }}

    /* Upload area */

    [data-testid="stFileUploader"] {{
        background: {CARD};
        border: 1px dashed {BORDER};
        border-radius: 14px;
        padding: 15px;
    }}

    /* Animations */

    @keyframes fadeIn {{
        from {{
            opacity: 0;
            transform: translateY(15px);
        }}

        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes pulse {{
        0% {{ opacity: 0.65; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.65; }}
    }}

    .live-indicator {{
        color: #34D399;
        animation: pulse 2s infinite;
        font-weight: 700;
    }}

    /* Divider */

    hr {{
        border-color: {BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# LOAD TRAINED MODELS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_prediction_models():

    # --------------------------------------
    # Load Random Forest
    # --------------------------------------

    rf_path = BASE_DIR / "models" / (
        "tuned_random_forest_11-11-2026-05-38-09-920150.pkl"
    )

    rf_artifacts = joblib.load(rf_path)

    rf_model = rf_artifacts["model"]
    rf_preprocessor = rf_artifacts["preprocessor"]
    rf_feature_names = rf_artifacts["feature_names"]
    rf_metrics = rf_artifacts["metrics"]

    # --------------------------------------
    # Load Sales LSTM
    # --------------------------------------

    lstm_dir = BASE_DIR / "deeplearning"

    lstm_timestamp = "12-11-2026-13-17-32-27"

    lstm_model_path = lstm_dir / (
        f"recurrent_lstm_{lstm_timestamp}.keras"
    )

    X_scaler_path = lstm_dir / (
        f"X_scaler_{lstm_timestamp}.pkl"
    )

    y_scaler_path = lstm_dir / (
        f"y_scaler_{lstm_timestamp}.pkl"
    )

    config_path = lstm_dir / (
        f"model_config_{lstm_timestamp}.json"
    )

    lstm_model = load_model(lstm_model_path)

    X_scaler = joblib.load(X_scaler_path)
    y_scaler = joblib.load(y_scaler_path)

    with open(config_path, "r") as file:
        lstm_config = json.load(file)

    # --------------------------------------
    # Load Customers LSTM
    # --------------------------------------

    customer_timestamp = "14-11-2026-05-30-19-00"

    customer_model_path = lstm_dir / (
        f"customer_lstm_{customer_timestamp}.keras"
    )

    customer_X_scaler_path = lstm_dir / (
        f"customer_X_scaler_{customer_timestamp}.pkl"
    )

    customer_y_scaler_path = lstm_dir / (
        f"customer_y_scaler_{customer_timestamp}.pkl"
    )

    customer_config_path = lstm_dir / (
        f"customer_lstm_config_{customer_timestamp}.json"
    )

    customer_lstm_model = load_model(customer_model_path)

    customer_X_scaler = joblib.load(
        customer_X_scaler_path
    )

    customer_y_scaler = joblib.load(
        customer_y_scaler_path
    )

    with open(customer_config_path, "r") as file:
        customer_lstm_config = json.load(file)

    # --------------------------------------
    # Return All Model Artifacts
    # --------------------------------------

    return {
        "rf_model": rf_model,
        "rf_preprocessor": rf_preprocessor,
        "rf_feature_names": rf_feature_names,
        "rf_metrics": rf_metrics,

        "lstm_model": lstm_model,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "lstm_config": lstm_config,

        "customer_lstm_model": customer_lstm_model,
        "customer_X_scaler": customer_X_scaler,
        "customer_y_scaler": customer_y_scaler,
        "customer_lstm_config": customer_lstm_config
    }


@st.cache_data
def load_test_data():

    test_path = BASE_DIR / "data" / "test_merged.csv"

    test_d = pd.read_csv(test_path)

    # Convert Date column to datetime
    test_d["Date"] = pd.to_datetime(test_d["Date"])

    test_d["MonthPeriod"] = pd.cut(
        test_d["Date"].dt.day,
        bins=[0, 10, 20, 31],
        labels=["Beginning", "Middle", "End"]
    )

    return test_d

models = load_prediction_models()
test_df = load_test_data()

# ==========================================
# LOAD HISTORICAL TRAINING DATA FOR LSTM
# ==========================================

@st.cache_data
def load_train_data():

    train_path = BASE_DIR / "data" / "train_merged.csv"

    train_df = pd.read_csv(train_path)

    train_df["Date"] = pd.to_datetime(train_df["Date"])

    return train_df

# Load historical training data
train_df = load_train_data()


# ==========================================
# CHECK HISTORICAL SEQUENCES
# ==========================================

SALES_WINDOW_SIZE = models["lstm_config"]["window_size"]

CUSTOMER_WINDOW_SIZE = models[
    "customer_lstm_config"
]["window_size"]

test_store_ids = test_df["Store"].unique()

# ------------------------------------------
# Check Sales historical records
# ------------------------------------------

sales_history_counts = (
    train_df[
        train_df["Store"].isin(test_store_ids)
    ]
    .groupby("Store")["Sales"]
    .count()
)

insufficient_sales_stores = [
    store_id
    for store_id in test_store_ids
    if sales_history_counts.get(store_id, 0) < SALES_WINDOW_SIZE
]

# ------------------------------------------
# Check Customers historical records
# ------------------------------------------

customer_history_counts = (
    train_df[
        train_df["Store"].isin(test_store_ids)
    ]
    .groupby("Store")["Customers"]
    .count()
)

insufficient_customer_stores = [
    store_id
    for store_id in test_store_ids
    if customer_history_counts.get(store_id, 0) < CUSTOMER_WINDOW_SIZE
]

# Get ORIGINAL input feature names from preprocessor
rf_input_features = list(
    models["rf_preprocessor"].feature_names_in_
)

# Check required columns
missing_features = [
    col for col in rf_input_features
    if col not in test_df.columns
]

# Select original features in training order
X_test_rf = test_df[rf_input_features].copy()


# Apply fitted preprocessing pipeline
X_test_rf_processed = models["rf_preprocessor"].transform(
    X_test_rf
)

# ==========================================
# RANDOM FOREST SALES PREDICTION
# ==========================================

# Identify open stores
open_mask = test_df["Open"].eq(1).to_numpy()

# Initialize predictions with zero
# Closed stores will retain zero sales
rf_predictions = np.zeros(len(test_df))

# Predict only for open stores
rf_predictions[open_mask] = models["rf_model"].predict(
    X_test_rf_processed[open_mask]
)

# Ensure predictions are not negative
rf_predictions = np.maximum(rf_predictions, 0)


# Store predictions alongside original test data
rf_prediction_df = test_df.copy()

rf_prediction_df["RF_PredictedSales"] = rf_predictions

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.markdown(
        '<div class="brand">STORE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-caption">SALES INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Preferences")

    dark_mode = st.toggle(
        "🌙 Dark Mode",
        value=st.session_state.dark_mode,
        key="theme_toggle"
    )

    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    # Sidebar heading theme colors
    if dark_mode:
          sidebar_heading_color = "#00D1C1"  # Bright teal for dark mode
    else:
          sidebar_heading_color = "#1F3A5F"  # Dark blue for light mode

    st.divider()

    st.markdown("### 🧭 Navigation")

     # Initialize the selected page only once
    if "selected_page" not in st.session_state:
          st.session_state["selected_page"] = "🏠 Dashboard"

     # Sidebar navigation options
    menu_options = [
          "🏠 Dashboard",
          "🔮 Sales Prediction",
          "📈 Model Comparison",
          "ℹ️ About"
     ]

     # Apply navigation request only when one is pending
    if "navigation_request" in st.session_state:
          requested_page = st.session_state.pop("navigation_request")

          if requested_page in menu_options:
               st.session_state["selected_page"] = requested_page

     # Radio navigation
    page = st.sidebar.radio(
          "Navigation",
          menu_options,
          key="selected_page"
     )

    st.divider()

    st.markdown("### 🧠 Prediction Models")

    st.markdown(
        """
        <div style="padding:10px 0;">
            <span style="color:#60A5FA;">●</span>
            <b>Random Forest</b>
            <br>
            <small style="color:#94A3B8;">Machine Learning</small>
        </div>

        <div style="padding:10px 0;">
            <span style="color:#34D399;">●</span>
            <b>LSTM</b>
            <br>
            <small style="color:#94A3B8;">Deep Learning</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption("STORE Sales Project")
    st.caption("Machine Learning + Deep Learning")


# ==========================================
# MAIN HERO
# ==========================================

st.html("""
<style>
.hero {
    background: linear-gradient(
        120deg,
        #123B72 0%,
        #176B87 55%,
        #0F9B83 100%
    );
    padding: 38px;
    border-radius: 20px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    animation: heroFade 0.9s ease;
}

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 800;
    margin: 15px 0;
}

.hero p {
    color: #DCEAFE;
    font-size: 15px;
    line-height: 1.7;
}

.hero-label {
    display: inline-block;
    background: rgba(255,255,255,0.17);
    padding: 7px 14px;
    border-radius: 20px;
    font-size: 12px;
    letter-spacing: 1px;
}

.live-indicator {
    color: #A7F3D0;
    font-weight: 700;
    font-size: 13px;
}

@keyframes heroFade {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>

<div class="hero">

    <div class="hero-label">
        RETAIL ANALYTICS • AI POWERED
    </div>

    <h1>Sales Intelligence Dashboard</h1>

    <p>
        Predict store performance, compare intelligent models,
        and transform historical data into actionable insights.
    </p>

    <div style="margin-top:20px;">
        <span class="live-indicator">
            ● SYSTEM INTERFACE READY
        </span>
    </div>

</div>
""")

def navigate_to_prediction():
    st.session_state["selected_page"] = "🔮 Sales Prediction"


# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-heading">📊 Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Welcome to the Rossmann sales forecasting platform."
    )

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("🏬", "Total Stores", "1,115"),
        ("🤖", "Prediction Models", "3"),
        ("📅", "Forecast Frequency", "Daily"),
        ("⚡", "Platform Status", "Ready")
    ]

    for col, metric in zip(
        [col1, col2, col3, col4], metrics
    ):

        with col:

            st.html(f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {metric[0]} {metric[1]}
                    </div>

                    <div class="metric-value">
                        {metric[2]}
                    </div>
                </div>
                """)

    st.markdown(
        '<div class="section-heading">🚀 Get Started</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>🔮 Generate Sales Predictions</h3>
            <p>
                Select a store, upload future daily information,
                and prepare forecasts using Random Forest and LSTM.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.button("Open Sales Prediction →"):

        st.session_state["navigation_request"] = (
            "🔮 Sales Prediction"
        )

        st.rerun()


# ==========================================
# SALES PREDICTION PAGE
# ==========================================

elif page == "🔮 Sales Prediction":

    st.markdown(
        '<div class="section-heading">🔮 Generate Predictions</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter the store details and upload a CSV containing "
        "the dates and daily promotional information."
    )

    st.markdown(
        '<div class="section-heading">1. Store Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        store_id = st.number_input(
            "🏬 Store ID",
            min_value=1,
            max_value=1115,
            value=1,
            step=1
        )

    with col2:

        forecast_start = st.date_input(
            "📅 Forecast Start Date",
            value=date(2015, 8, 1)
        )

    
    # ==========================================
    # STEP 1: CSV TEMPLATE AND VALIDATION
    # ==========================================

    st.markdown(
        '<div class="section-heading">2. Upload Daily Information</div>',
        unsafe_allow_html=True
    )

    # Required CSV columns
    csv_columns = [
        "Date",
        "IsHoliday",
        "IsWeekend",
        "IsPromo",
        "StateHoliday",
        "SchoolHoliday",
        "Open"
    ]

    # Download CSV template
    template_df = pd.DataFrame(columns=csv_columns)

    st.download_button(
        label="📥 Download CSV Template",
        data=template_df.to_csv(index=False),
        file_name="sales_prediction_template.csv",
        mime="text/csv"
    )

    # CSV upload
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload daily forecasting information using the seven required columns."
    )

    input_df = None
    csv_valid = False

    if uploaded_file is not None:

        try:

            # Read uploaded CSV
            input_df = pd.read_csv(uploaded_file)

            # Clean column names
            input_df.columns = input_df.columns.str.strip()

            st.success("CSV uploaded successfully!")

            st.markdown("#### Uploaded Data Preview")

            st.dataframe(
                input_df.head(10),
                use_container_width=True
            )

            st.write(
                f"Rows: {input_df.shape[0]} | "
                f"Columns: {input_df.shape[1]}"
            )

            # Check required columns
            missing_columns = [
                col for col in csv_columns
                if col not in input_df.columns
            ]

            if missing_columns:

                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            elif input_df.empty:

                st.error("The uploaded CSV contains no records.")

            else:

                # ----------------------------------
                # DATE VALIDATION
                # ----------------------------------

                input_df["Date"] = pd.to_datetime(
                    input_df["Date"],
                    errors="coerce"
                )

                if input_df["Date"].isnull().any():

                    st.error(
                        "Invalid dates found. Use YYYY-MM-DD format."
                    )

                else:

                    input_df["Date"] = input_df["Date"].dt.normalize()

                    # ----------------------------------
                    # BINARY COLUMN VALIDATION
                    # ----------------------------------

                    binary_columns = [
                        "IsHoliday",
                        "IsWeekend",
                        "IsPromo",
                        "SchoolHoliday",
                        "Open"
                    ]

                    for col in binary_columns:

                        input_df[col] = pd.to_numeric(
                            input_df[col],
                            errors="coerce"
                        )

                    invalid_binary = {}

                    for col in binary_columns:

                        invalid_count = (
                            input_df[col].isnull()
                            | ~input_df[col].isin([0, 1])
                        ).sum()

                        if invalid_count > 0:

                            invalid_binary[col] = int(invalid_count)

                    # ----------------------------------
                    # STATE HOLIDAY VALIDATION
                    # ----------------------------------

                    input_df["StateHoliday"] = (
                        input_df["StateHoliday"]
                        .astype("string")
                        .str.strip()
                        .str.lower()
                    )

                    # Convert string "0" to numeric 0
                    input_df["StateHoliday"] = (
                        input_df["StateHoliday"]
                        .replace({"0": 0})
                    )

                    allowed_holidays = [0, "a", "b", "c"]

                    invalid_state_holiday = (
                        input_df["StateHoliday"].isnull()
                        | ~input_df["StateHoliday"].isin(
                            allowed_holidays
                        )
                    )

                    # ----------------------------------
                    # DATE AND DATA CONSISTENCY CHECKS
                    # ----------------------------------

                    duplicate_dates = (
                        input_df["Date"].duplicated().sum()
                    )

                    selected_start = pd.Timestamp(forecast_start)

                    wrong_start_date = (
                        input_df["Date"].min() != selected_start
                    )

                    # Check consecutive dates
                    sorted_dates = pd.DatetimeIndex(
                        input_df["Date"].sort_values()
                    )

                    expected_dates = pd.date_range(
                        start=sorted_dates.min(),
                        end=sorted_dates.max(),
                        freq="D"
                    )

                    missing_dates = expected_dates.difference(
                        sorted_dates
                    )

                    # Check IsWeekend against actual calendar
                    expected_weekend = (
                        input_df["Date"].dt.dayofweek >= 5
                    ).astype(int)

                    weekend_mismatch = (
                        input_df["IsWeekend"].notna()
                        & input_df["IsWeekend"].ne(expected_weekend)
                    ).sum()

                    # Check combined holiday indicator
                    expected_holiday = (
                        input_df["StateHoliday"].ne(0)
                        | input_df["SchoolHoliday"].eq(1)
                    ).astype(int)

                    holiday_mismatch = (
                        input_df["IsHoliday"].notna()
                        & input_df["IsHoliday"].ne(expected_holiday)
                    ).sum()

                    # ----------------------------------
                    # DISPLAY VALIDATION RESULTS
                    # ----------------------------------

                    if invalid_binary:

                        st.error(
                            f"Invalid binary values: {invalid_binary}. "
                            "Use only 0 or 1."
                        )

                    elif invalid_state_holiday.any():

                        st.error(
                            "StateHoliday must contain only "
                            "0, a, b, or c."
                        )

                    elif duplicate_dates > 0:

                        st.error(
                            f"Found {duplicate_dates} duplicate dates."
                        )

                    elif wrong_start_date:

                        st.error(
                            "The first CSV date must match the "
                            "selected Forecast Start Date. "
                            f"CSV starts on {selected_start.date()} "
                            "as selected in the form."
                        )

                    elif len(missing_dates) > 0:

                        st.error(
                            f"Found {len(missing_dates)} missing "
                            "calendar dates in the uploaded CSV."
                        )

                    else:

                        # Sort input chronologically
                        input_df = input_df.sort_values(
                            "Date"
                        ).reset_index(drop=True)

                        # Convert binary columns to integers
                        for col in binary_columns:
                            input_df[col] = input_df[col].astype(int)

                        csv_valid = True

                        st.success(
                            "CSV validation completed successfully!"
                        )

                        if weekend_mismatch > 0:

                            st.warning(
                                f"{weekend_mismatch} IsWeekend values "
                                "do not match their calendar dates."
                            )

                        if holiday_mismatch > 0:

                            st.warning(
                                f"{holiday_mismatch} IsHoliday values "
                                "do not match the combined StateHoliday "
                                "and SchoolHoliday indicators."
                            )

                        st.write(
                            "Forecast records:",
                            len(input_df)
                        )

                        st.write(
                            "Forecast start date:",
                            input_df["Date"].min().date()
                        )

                        st.write(
                            "Forecast end date:",
                            input_df["Date"].max().date()
                        )

                        st.write(
                            "Open days:",
                            int(input_df["Open"].sum())
                        )

                        st.write(
                            "Closed days:",
                            int((input_df["Open"] == 0).sum())
                        )

        except Exception as e:

            st.error(f"Unable to process CSV: {e}")

    else:

        st.info(
            "Download the CSV template, fill in the daily information, "
            "and upload it here."
        )

    if csv_valid:

        selected_store_data = train_df[
            train_df["Store"] == store_id
        ].sort_values("Date")

        if selected_store_data.empty:
            st.error(
                f"No historical information found for Store {store_id}."
            )

        else:
            store_info = selected_store_data.iloc[-1]

            st.markdown("### Retrieved Store Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Store Type",
                    store_info["StoreType"]
                )

            with col2:
                st.metric(
                    "Assortment",
                    store_info["Assortment"]
                )

            with col3:
                st.metric(
                    "Competition Distance",
                    f'{store_info["CompetitionDistance"]:,.0f}'
                )

            st.success(
                f"Historical information retrieved for Store {store_id}."
            )

            # Prepare one row per uploaded forecast date
            rf_input_df = pd.DataFrame({

                # Store information
                "Store": store_id,
                "StoreType": store_info["StoreType"],
                "Assortment": store_info["Assortment"],

                # Daily information from uploaded CSV
                "Open": input_df["Open"].astype(int),
                "Promo": input_df["IsPromo"].astype(int),
                "StateHoliday": input_df["StateHoliday"],
                "SchoolHoliday": input_df["SchoolHoliday"].astype(int),
                "IsWeekend": input_df["IsWeekend"].astype(int),

                # Calendar features
                "DayOfWeek": input_df["Date"].dt.dayofweek + 1,
                "Year": input_df["Date"].dt.year,
                "Month": input_df["Date"].dt.month,
                "Day": input_df["Date"].dt.day,
                "WeekOfYear": input_df["Date"].dt.isocalendar().week.astype(int),

                # Competition information
                "CompetitionDistance": store_info["CompetitionDistance"],
                "CompetitionOpenSinceMonth": store_info["CompetitionOpenSinceMonth"],
                "CompetitionOpenSinceYear": store_info["CompetitionOpenSinceYear"],
                "CompetitionDateMissing": store_info["CompetitionDateMissing"],

                # Promo2 information
                "Promo2": store_info["Promo2"],
                "Promo2SinceWeek": store_info["Promo2SinceWeek"],
                "Promo2SinceYear": store_info["Promo2SinceYear"],
                "PromoInterval": store_info["PromoInterval"],

                # Month period
                "MonthPeriod": pd.cut(
                    input_df["Date"].dt.day,
                    bins=[0, 10, 20, 31],
                    labels=["Beginning", "Middle", "End"]
                )
            })

            # Match the exact feature order used during model training
            rf_feature_names = list(
                models["rf_preprocessor"].feature_names_in_
            )

            rf_input_df = rf_input_df[rf_feature_names].copy()

        if st.button(
            "🚀 Generate Predictions",
            use_container_width=True
        ):

            
            try:
                # ==========================================
                # 1. RANDOM FOREST PREDICTIONS
                # ==========================================

                X_user_processed = models[
                    "rf_preprocessor"
                ].transform(rf_input_df)

                rf_predictions = np.zeros(len(rf_input_df))

                open_mask = (
                    rf_input_df["Open"].eq(1).to_numpy()
                )

                if open_mask.any():
                    rf_predictions[open_mask] = models[
                        "rf_model"
                    ].predict(
                        X_user_processed[open_mask]
                    )

                rf_predictions = np.maximum(
                    rf_predictions, 0
                )

                # ==========================================
                # 2. LSTM PREDICTIONS
                # ==========================================

                window_size = models[
                    "lstm_config"
                ]["window_size"]

                # Get historical sales for selected store
                history_df = train_df[
                    (train_df["Store"] == store_id) &
                    (train_df["Date"] < input_df["Date"].min())
                ].sort_values("Date")

                # Check sufficient historical records
                if len(history_df) < window_size:
                    raise ValueError(
                        f"Store {store_id} does not have "
                        f"{window_size} historical sales records."
                    )

                # Select the latest 7 sales records
                history_df = history_df.tail(window_size)

                # Verify that historical dates are consecutive
                expected_dates = pd.date_range(
                    end=input_df["Date"].min()
                    - pd.Timedelta(days=1),
                    periods=window_size,
                    freq="D"
                )

                if not history_df["Date"].reset_index(
                    drop=True
                ).equals(pd.Series(expected_dates)):
                    raise ValueError(
                        "The selected store does not have "
                        "seven consecutive historical days "
                        "immediately before the forecast."
                    )

                # Initial sales window
                sales_window = history_df[
                    "Sales"
                ].to_numpy(dtype=float)

                # Initial customer window
                customer_window = history_df[
                    "Customers"
                ].to_numpy(dtype=float)

                lstm_predictions = []

                customer_predictions = []

                # Get the Customers LSTM window size
                customer_window_size = models[
                    "customer_lstm_config"
                ]["window_size"]

                # Predict each forecast date recursively
                for i in range(len(input_df)):

                    if int(input_df.iloc[i]["Open"]) == 0:

                        # Closed store: sales must be zero
                        predicted_sales = 0.0

                    else:

                        # Scale the previous 7 sales values
                        X_sequence = models[
                            "X_scaler"
                        ].transform(
                            sales_window.reshape(-1, 1)
                        ).reshape(1, window_size, 1)

                        # Generate prediction
                        scaled_prediction = models[
                            "lstm_model"
                        ].predict(
                            X_sequence,
                            verbose=0
                        )

                        # Convert prediction to original scale
                        predicted_sales = models[
                            "y_scaler"
                        ].inverse_transform(
                            scaled_prediction.reshape(-1, 1)
                        )[0, 0]

                        predicted_sales = max(
                            float(predicted_sales), 0.0
                        )

                    lstm_predictions.append(
                        predicted_sales
                    )

                    # Update window for next forecast date
                    sales_window = np.append(
                        sales_window[1:],
                        predicted_sales
                    )


                    # ==========================================
                    # CUSTOMERS LSTM PREDICTION
                    # ==========================================

                    # If the store is closed, customer count is zero
                    if int(input_df.iloc[i]["Open"]) == 0:

                        predicted_customers = 0.0

                    else:

                        # Scale previous 7 customer values
                        customer_X_sequence = models[
                            "customer_X_scaler"
                        ].transform(
                            customer_window.reshape(-1, 1)
                        ).reshape(1, customer_window_size, 1)

                        # Generate prediction
                        customer_scaled_prediction = models[
                            "customer_lstm_model"
                        ].predict(
                            customer_X_sequence,
                            verbose=0
                        )

                        # Convert prediction to original scale
                        predicted_customers = models[
                            "customer_y_scaler"
                        ].inverse_transform(
                            customer_scaled_prediction.reshape(-1, 1)
                        )[0, 0]

                        predicted_customers = max(
                            float(predicted_customers), 0.0
                        )

                    # Store prediction
                    customer_predictions.append(
                        predicted_customers
                    )

                    # Update customer window for next forecast date
                    customer_window = np.append(
                        customer_window[1:],
                        predicted_customers
                    )


                # ==========================================
                # 3. COMBINE BOTH MODEL PREDICTIONS
                # ==========================================

                result_df = input_df.copy()

                result_df["RF_PredictedSales"] = (
                    rf_predictions
                )

                result_df["LSTM_PredictedSales"] = (
                    lstm_predictions
                )

                result_df["LSTM_PredictedCustomers"] = (
                    customer_predictions
                )

                # ==========================================
                # 4. DISPLAY RESULTS
                # ==========================================

                st.success(
                    "Random Forest and LSTM predictions "
                    "generated successfully!"
                )

                st.markdown("### Prediction Results")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # Summary metrics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Forecast Days",
                        len(result_df)
                    )

                with col2:
                    st.metric(
                        "RF Total Sales",
                        f"{rf_predictions.sum():,.2f}"
                    )

                with col3:
                    st.metric(
                        "LSTM Total Sales",
                        f"{sum(lstm_predictions):,.2f}"
                    )

                                # ==========================================
                # 5. DAILY SALES COMPARISON CHART
                # ==========================================

                st.markdown("### 📈 Daily Sales Forecast Comparison")

                chart_df = result_df[
                    [
                        "Date",
                        "RF_PredictedSales",
                        "LSTM_PredictedSales"
                    ]
                ].copy()

                chart_df = chart_df.set_index("Date")

                st.line_chart(
                    chart_df,
                    use_container_width=True
                )

                # Customer Predictions Chart

                st.subheader("👥 Date-wise Customer Predictions")

                customer_chart_df = result_df[
                    ["Date", "LSTM_PredictedCustomers"]
                ].copy()

                customer_chart_df["Date"] = pd.to_datetime(
                    customer_chart_df["Date"]
                )

                st.line_chart(
                    customer_chart_df,
                    x="Date",
                    y="LSTM_PredictedCustomers",
                    x_label="Date",
                    y_label="Predicted Customers"
                )

                                # ==========================================
                # 6. DOWNLOAD PREDICTIONS
                # ==========================================

                st.markdown("### 📥 Download Forecast Results")

                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_data,
                    file_name=f"store_{store_id}_sales_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


            except Exception as e:

                st.error(
                    f"Random Forest prediction failed: {e}"
                )



# ==========================================
# MODEL COMPARISON PAGE
# ==========================================

elif page == "📈 Model Comparison":

    st.markdown(
        '<div class="section-heading">📈 Model Performance Comparison</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Compare the performance of the two trained forecasting models."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#60A5FA;">🌳 Random Forest</h3>
                <p>Machine Learning Model</p>
                <h2 style="color:#60A5FA;">R² = 0.8920</h2>
                <p>MAE = 685.02</p>
                <p>RMSE = 965.23</p>
                <p>RMSPE = 14.78%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#34D399;">🧠 Sales LSTM</h3>
                <p>Deep Learning Model</p>
                <h2 style="color:#34D399;">R² = 0.8088</h2>
                <p>MAE = 830.84</p>
                <p>RMSE = 1284.40</p>
                <p>RMSPE = 16.81%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="info-card">
                <h3 style="color:#FBBF24;">👥 Customers LSTM</h3>
                <p>Deep Learning Model</p>
                <h2 style="color:#FBBF24;">R² = 0.9359</h2>
                <p>MAE = 75.61</p>
                <p>RMSE = 110.15</p>
                <p>RMSPE = 15.09%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.info(
        "These are validation metrics from your trained models, "
        "not predictions for a newly uploaded CSV."
    )

# ==========================================
# ABOUT PAGE
# ==========================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="section-heading">ℹ️ About This Project</div>',
        unsafe_allow_html=True
    )

    st.html("""
        <div class="info-card">

            <h2>STORE Sales Forecasting</h2>

            <p>
                An intelligent retail forecasting platform
                developed using machine learning and deep learning.
            </p>

            <hr>

            <h3>Models Used</h3>

            <p>🌳 Random Forest Regressor</p>
            <p>🧠 Long Short-Term Memory Network</p>

            <hr>

            <h3>Technology Stack</h3>

            <p>Python • Streamlit • Pandas • NumPy</p>
            <p>Scikit-learn • TensorFlow • Plotly</p>

        </div>
    """)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    f"""
    <div style="
        text-align:center;
        color:{MUTED};
        padding:15px;
        font-size:12px;
    ">
        STORE SALES INTELLIGENCE PLATFORM
        <br>
        Powered by Random Forest & LSTM
        <br>
        © 2026 | Retail Forecasting Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

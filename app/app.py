from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.pipeline import create_forecasting_features

from xgboost import XGBRegressor
import google.generativeai as genai

# Gemini API Configuration
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Gemini Insight Generation
def generate_gemini_insight(prompt):
    model = genai.GenerativeModel("gemini-3.6-flash")
    response = model.generate_content(prompt)
    return response.text

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FBI Crime Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-weight: 700;
        letter-spacing: -0.35px;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 112px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"] {
        color: inherit !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.80rem;
        font-weight: 650;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 750;
    }

    .eyebrow {
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 1.6px;
        opacity: 0.65;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        font-size: 1rem;
        opacity: 0.72;
        margin-top: -0.45rem;
        margin-bottom: 1rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 20px 0;
    }

    .badge {
        border: 1px solid rgba(128, 128, 128, 0.30);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.72rem;
        font-weight: 650;
    }

    .section-kicker {
        font-size: 0.72rem;
        font-weight: 750;
        letter-spacing: 1.35px;
        text-transform: uppercase;
        opacity: 0.58;
        margin-bottom: -0.55rem;
    }

    .insight-box {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0;
    }

    button[data-baseweb="tab"] {
        font-weight: 650;
        padding: 0.65rem 0.8rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #4f46e5;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 42px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.20);
    }

    :root {
        --pink-primary: #d9468f;
        --pink-deep: #a83272;
        --pink-soft: #fce7f3;
        --pink-border: rgba(217, 70, 143, 0.28);
    }

    .pink-note {
        border-left: 4px solid var(--pink-primary);
        background: rgba(217, 70, 143, 0.07);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 10px 0 18px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS AND DATA LOADING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"




@st.cache_resource
def load_model():
    model_path = MODEL_DIR / "final_xgb_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = XGBRegressor()
    model.load_model(str(model_path))
    return model


@st.cache_data
def load_data():
    historical_path = MODEL_DIR / "historical_monthly_crime_data.pkl"
    metadata_path = MODEL_DIR / "xgb_model_metadata.pkl"

    if not historical_path.exists():
        raise FileNotFoundError(f"Historical data file not found: {historical_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    historical = joblib.load(historical_path)
    metadata = joblib.load(metadata_path)

    if isinstance(historical, pd.DataFrame):
        if historical.shape[1] == 1:
            historical = historical.iloc[:, 0]
        else:
            raise ValueError("Historical data must contain one monthly crime-count series.")

    historical = pd.Series(historical).copy()
    historical.index = pd.to_datetime(historical.index)
    historical = pd.to_numeric(historical, errors="coerce").dropna().sort_index()
    historical = historical[~historical.index.duplicated(keep="last")]

    if len(historical) < 12:
        raise ValueError("At least 12 months of historical data are required.")

    return historical, metadata


@st.cache_data
def load_validation_data():
    validation_path = (
        BASE_DIR
        / "data"
        / "processed"
        / "xgb_walk_forward_validation.csv"
    )

    validation_data = pd.read_csv(validation_path)
    validation_data["Date"] = pd.to_datetime(validation_data["Date"])

    return validation_data


try:
    model = load_model()
    historical_data, metadata = load_data()
    validation_data = load_validation_data()
except Exception as error:
    st.error("The dashboard could not load the model or historical data.")
    st.exception(error)
    st.stop()


VALIDATION_METRICS = {
    "MAE": mean_absolute_error(
        validation_data["Actual"],
        validation_data["Predicted"]
    ),
    "RMSE": mean_squared_error(
        validation_data["Actual"],
        validation_data["Predicted"]
    ) ** 0.5,
    "R2": r2_score(
        validation_data["Actual"],
        validation_data["Predicted"]
    ),
}

# Naive baseline MAE
historical_monthly = historical_data.copy()
historical_monthly.index = historical_monthly.index.to_period("M")

validation_periods = validation_data["Date"].dt.to_period("M")
previous_month_periods = validation_periods - 1

baseline_predictions = previous_month_periods.map(historical_monthly)

baseline_mae = (
    validation_data["Actual"] - baseline_predictions
).abs().mean()

# Prediction Interval Calibration
validation_errors = (
    validation_data["Actual"] - validation_data["Predicted"]
).abs()

ERROR_QUANTILE = validation_errors.quantile(0.90)



# ============================================================
# PROJECT CONSTANTS AND DERIVED DATA
# ============================================================

latest_date = historical_data.index[-1]
latest_count = float(historical_data.iloc[-1])
forecast_month = latest_date + pd.DateOffset(months=1)


MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": ["Gradient Boosting", "Random Forest", "XGBoost"],
        "MAE": [175.34, 173.96, 171.47],
        "RMSE": [230.64, 227.48, 223.73],
        "R² Score": [0.4343, 0.4497, 0.4677],
    }
)

FEATURE_IMPORTANCE = pd.DataFrame(
    {
        "Feature": [
            "Lag_12",
            "Rolling_Mean_12",
            "Rolling_Mean_6",
            "Rolling_Mean_3",
            "Lag_1",
        ],
        "Importance (%)": [47.39, 33.13, 5.19, 5.10, 4.74],
    }
)



# ============================================================
# PROFESSIONAL ACCESSIBLE CHART PALETTE
# ============================================================

NAVY = "#1F4E79"
BLUE = "#648FFF"
ORANGE = "#FE6100"
TEAL = "#009E73"
PURPLE = "#785EF0"
MAGENTA = "#DC267F"
YELLOW = "#FFB000"
CHART_PALETTE = [NAVY, BLUE, ORANGE, TEAL, PURPLE, MAGENTA, YELLOW]


def style_plotly_figure(fig, height=390, show_legend=False):
    """Apply a consistent, professional and readable Plotly theme."""
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#374151"),
        margin=dict(l=12, r=12, t=28, b=12),
        hoverlabel=dict(bgcolor="#F3F4F6", font_color="#111827"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="rgba(156,163,175,0.35)",
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(156,163,175,0.20)",
        zeroline=False,
        tickfont=dict(size=11),
    )
    if not show_legend:
        fig.update_layout(showlegend=False)
    return fig


def make_line_chart(dataframe, x, y, color=None, title=None, height=390):
    """Build a readable line chart with distinct series colors."""
    if color:
        fig = px.line(
            dataframe,
            x=x,
            y=y,
            color=color,
            color_discrete_map={"Historical": NAVY, "Forecast": ORANGE},
            color_discrete_sequence=CHART_PALETTE,
            markers=True,
        )
    else:
        fig = px.line(
            dataframe,
            x=x,
            y=y,
            color_discrete_sequence=[NAVY],
            markers=False,
        )
    if not color and isinstance(y, list):
        for index, trace in enumerate(fig.data):
            trace.update(
                line=dict(color=CHART_PALETTE[index % len(CHART_PALETTE)], width=2.6),
                marker=dict(size=5),
            )
    else:
        fig.update_traces(line=dict(width=2.6), marker=dict(size=5))
    if title:
        fig.update_layout(title=title)
    return style_plotly_figure(fig, height=height, show_legend=bool(color))

def make_forecast_interval_chart(forecast_df, height=410):
    """Display historical data, forecasts, and prediction intervals."""

    fig = go.Figure()

    # Historical line
    history = historical_data.tail(24)

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines+markers",
            name="Historical",
            line=dict(color=NAVY, width=2.6),
            marker=dict(size=5),
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Forecast Month"],
            y=forecast_df["Predicted Crime Count"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color=ORANGE, width=2.6),
            marker=dict(size=6),
        )
    )

    # Prediction interval band
    fig.add_trace(
        go.Scatter(
            x=pd.concat(
                [
                    forecast_df["Forecast Month"],
                    forecast_df["Forecast Month"].iloc[::-1],
                ]
            ),
            y=pd.concat(
                [
                    forecast_df["Upper Bound"],
                    forecast_df["Lower Bound"].iloc[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prediction Interval",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Count",
    )

    return style_plotly_figure(
        fig,
        height=height,
        show_legend=True,
    )

def make_bar_chart(dataframe, x, y, title=None, height=330):
    """Build a bar chart using a perceptually ordered blue scale."""
    fig = px.bar(
        dataframe,
        x=x,
        y=y,
        color=y,
        color_continuous_scale=["#E8F1FA", "#9DC3E6", "#648FFF", "#1F4E79"],
    )
    fig.update_coloraxes(showscale=False)
    if title:
        fig.update_layout(title=title)
    return style_plotly_figure(fig, height=height, show_legend=False)


# ============================================================
# FORECASTING FUNCTIONS
# ============================================================

def generate_recursive_forecast(horizon: int) -> pd.DataFrame:
    """Generate recursive monthly forecasts with prediction intervals."""

    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one month.")

    forecast_history = historical_data.copy().astype(float)
    results = []

    for step in range(1, horizon + 1):
        input_features = create_forecasting_features(forecast_history)

        expected_features = metadata.get("feature_names")
        if expected_features:
            input_features = input_features.reindex(
                columns=expected_features,
                fill_value=0
            )

        prediction = float(model.predict(input_features)[0])
        prediction = max(0, int(round(prediction)))

        lower_bound = max(
            0,
            int(round(prediction - ERROR_QUANTILE))
        )

        upper_bound = int(
            round(prediction + ERROR_QUANTILE)
        )

        prediction_date = latest_date + pd.DateOffset(months=step)

        results.append(
            {
                "Forecast Month": prediction_date,
                "Predicted Crime Count": prediction,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
            }
        )

        forecast_history.loc[prediction_date] = prediction

    return pd.DataFrame(results)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 8px 0 18px 0;">
            <div style="font-size: 11px; letter-spacing: 1.8px; font-weight: 750; opacity: 0.65;">
                DATA ANALYTICS · MACHINE LEARNING
            </div>
            <div style="font-size: 25px; font-weight: 800; margin-top: 7px;">FBI Crime</div>
            <div style="font-size: 25px; font-weight: 800;">Forecasting</div>
            <div style="font-size: 12px; opacity: 0.65; margin-top: 8px;">
                Monthly Crime Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### 📌 Project Details")
    st.markdown(
        """
        **Project**  
        FBI Crime Investigation

        **Contribution**  
        Individual

        **Task**  
        Monthly Crime Forecasting

        **Model**  
        XGBoost Regressor

        **Forecast Type**  
        Recursive Multi-Month Forecasting

        **Historical Coverage**  
        January 1999 – December 2011
        """
    )

    st.divider()
    st.markdown("#### 🛠️ Technology Stack")
    st.markdown("`Python` · `Pandas`  \n`XGBoost` · `Streamlit`  \n`Joblib`")

    st.divider()
    st.markdown("#### ● Model Status")
    st.success("Model loaded")
    st.caption("For analytical and educational purposes. Not validated for operational deployment.")


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown('<div class="eyebrow">DATA ANALYTICS · MACHINE LEARNING</div>', unsafe_allow_html=True)
st.title("FBI Crime Analytics")
st.markdown('<div class="subtitle">Monthly Crime Forecasting Platform</div>', unsafe_allow_html=True)
st.write(
    "Analyze historical crime patterns and generate recursive monthly forecasts "
    "using an XGBoost regression model with time-series features."
)

st.markdown(
    """
    <div class="badge-row">
        <span class="badge">● MODEL ACTIVE</span>
        <span class="badge">XGBOOST REGRESSOR</span>
        <span class="badge">1999–2011 DATA</span>
        <span class="badge">WALK-FORWARD VALIDATION</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.success("Model and historical data loaded successfully.")


tab_overview, tab_forecast, tab_history, tab_performance, tab_methodology = st.tabs(
    ["Overview", "Forecast", "Historical Trends", "Model Performance", "Methodology"]
)


# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:
    st.markdown('<div class="section-kicker">PROJECT SUMMARY</div>', unsafe_allow_html=True)
    st.header("Historical Crime Overview")
    st.write(
        "This project transforms incident-level crime records into monthly crime counts, "
        "engineers lag and rolling-average features, and evaluates forecasting models "
        "using chronological walk-forward validation."
    )

    st.subheader("Project Snapshot")
    snapshot_1, snapshot_2, snapshot_3, snapshot_4 = st.columns(4)
    with snapshot_1:
        st.metric("Latest Month", latest_date.strftime("%b %Y"))
    with snapshot_2:
        st.metric("Latest Crime Count", f"{int(latest_count):,}")
    with snapshot_3:
        st.metric("Historical Months", f"{len(historical_data):,}")
    with snapshot_4:
        st.metric("Validation R²", f"{VALIDATION_METRICS['R2']:.4f}")

    st.subheader("ML Performance Snapshot")
    metric_1, metric_2, metric_3 = st.columns(3)
    with metric_1:
        st.metric("Walk-Forward MAE", f"{VALIDATION_METRICS['MAE']:.2f}")
    with metric_2:
        st.metric("Walk-Forward RMSE", f"{VALIDATION_METRICS['RMSE']:.2f}")
    with metric_3:
        st.metric("Validation Method", "Chronological")

    st.caption(
        "The reported metrics represent one-step-ahead walk-forward validation results. "
        "They should not be interpreted as guaranteed performance for longer recursive horizons."
    )

    st.divider()
    st.markdown('<div class="section-kicker">TREND ANALYSIS</div>', unsafe_allow_html=True)
    st.subheader("Historical Monthly Crime Trend")
    st.caption("Recorded monthly crime counts across the available historical period.")
    overview_chart = historical_data.rename("Monthly Crime Count").to_frame().reset_index()
    overview_chart.columns = ["Date", "Monthly Crime Count"]
    st.plotly_chart(
        make_line_chart(overview_chart, "Date", "Monthly Crime Count", height=390),
        use_container_width=True,
        config={"displaylogo": False},
    )

    overview_forecast = generate_recursive_forecast(3)
    st.subheader("Next Three-Month Forecast Preview")
    st.caption("Preview generated directly from the deployed XGBoost model.")
    st.dataframe(
        overview_forecast.assign(
            **{"Forecast Month": overview_forecast["Forecast Month"].dt.strftime("%B %Y")}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <div class="insight-box">
            <strong>Model insight:</strong> Lag_12 and Rolling_Mean_12 were the most influential
            features in the documented XGBoost feature-importance analysis. They represent
            annual historical context and longer-term crime-volume behaviour.
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_combined_forecast_chart(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Combine recent historical data with forecast values."""

    history_tail = (
        historical_data.tail(24)
        .rename("Crime Count")
        .to_frame()
    )

    history_tail["Series"] = "Historical"

    history_tail = (
        history_tail
        .rename_axis("Date")
        .reset_index()
        .rename(columns={"Crime Count": "Count"})
    )

    future = forecast_df.rename(
        columns={
            "Forecast Month": "Date",
            "Predicted Crime Count": "Count",
        }
    ).copy()

    future["Series"] = "Forecast"

    combined = pd.concat(
        [
            history_tail[["Date", "Count", "Series"]],
            future[["Date", "Count", "Series"]],
        ],
        ignore_index=True,
    )

    return combined.sort_values("Date")

# ============================================================
# TAB 2: FORECAST
# ============================================================

with tab_forecast:

    st.markdown(
        '<div class="section-kicker">PREDICTIVE ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    st.header("Monthly Crime Forecast")

    st.write(
        "Select a forecast horizon to generate recursive monthly predictions. "
        "Each predicted month is appended to the historical series before "
        "the next month is forecast."
    )

    # --------------------------------------------------------
    # Forecast Horizon Selection
    # --------------------------------------------------------

    horizon = st.selectbox(
        "Forecast Horizon",
        options=[1, 3, 6, 12],
        format_func=lambda value: (
            f"{value} month" if value == 1 else f"{value} months"
        ),
        key="forecast_horizon",
    )

    # --------------------------------------------------------
    # Generate Forecast
    # --------------------------------------------------------

    forecast_df = generate_recursive_forecast(horizon)

    forecast_start = forecast_df["Forecast Month"].min()
    forecast_end = forecast_df["Forecast Month"].max()

    st.info(
        f"Forecast period: {forecast_start.strftime('%B %Y')} "
        f"to {forecast_end.strftime('%B %Y')}"
    )

    # --------------------------------------------------------
    # Primary Forecast Metrics
    # --------------------------------------------------------

    forecast_metric_1, forecast_metric_2, forecast_metric_3 = st.columns(3)

    with forecast_metric_1:
        st.metric(
            "Forecast Months",
            horizon,
        )

    with forecast_metric_2:
        st.metric(
            "First Forecast",
            f"{int(forecast_df.iloc[0]['Predicted Crime Count']):,}",
        )

    with forecast_metric_3:
        st.metric(
            "Average Forecast",
            f"{forecast_df['Predicted Crime Count'].mean():,.0f}",
        )

    # --------------------------------------------------------
    # Forecast Summary Cards
    # --------------------------------------------------------

    forecast_values = forecast_df["Predicted Crime Count"]

    highest_forecast = forecast_df.loc[
        forecast_values.idxmax()
    ]

    lowest_forecast = forecast_df.loc[
        forecast_values.idxmin()
    ]

    first_forecast = forecast_values.iloc[0]
    last_forecast = forecast_values.iloc[-1]

    if last_forecast > first_forecast:
        forecast_direction = "Increasing"
    elif last_forecast < first_forecast:
        forecast_direction = "Decreasing"
    else:
        forecast_direction = "Stable"

    st.subheader("Forecast Summary")

    summary_1, summary_2, summary_3 = st.columns(3)

    with summary_1:
        st.metric(
            "Forecast Direction",
            forecast_direction,
        )

    with summary_2:
        st.metric(
            "Highest Predicted Month",
            highest_forecast["Forecast Month"].strftime("%b %Y"),
        )

    with summary_3:
        st.metric(
            "Highest Predicted Count",
            f"{int(highest_forecast['Predicted Crime Count']):,}",
        )

    st.caption(
        "Direction compares the first and last predicted values "
        "within the selected forecast horizon."
    )

    # --------------------------------------------------------
    # Historical Context and Forecast Chart
    # --------------------------------------------------------

    chart_data = build_combined_forecast_chart(forecast_df)

    st.subheader("Historical Context and Forecast")

    st.caption(
        "The chart combines historical observations with forecasts "
        "and empirical prediction intervals calibrated from validation errors. "
        "These intervals are not formal confidence intervals and do not guarantee coverage."
    )

    st.plotly_chart(
    make_forecast_interval_chart(
        forecast_df,
        height=410,
    ),
    use_container_width=True,
    config={"displaylogo": False},
    )

    # --------------------------------------------------------
    # Forecast Results Table
    # --------------------------------------------------------

    st.subheader("Forecast Results")

    display_forecast = forecast_df.copy()

    display_forecast["Forecast Month"] = (
        display_forecast["Forecast Month"]
        .dt.strftime("%B %Y")
    )

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Download Forecast
    # --------------------------------------------------------

    st.download_button(
        "Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name=f"crime_forecast_{horizon}_months.csv",
        mime="text/csv",
    )

        # --------------------------------------------------------
    # Forecast Interpretation
    # --------------------------------------------------------

    st.subheader("Analytical Interpretation")

    average_forecast = forecast_values.mean()
    forecast_range = forecast_values.max() - forecast_values.min()


    # --------------------------------------------------------
    # Gemini AI Insights
    # --------------------------------------------------------

    st.subheader("🤖 Gemini AI Insights")

    st.caption(
        "Generate a structured five-section interpretation of the selected forecast."
    )


    if st.button("✨ Generate AI Insights", key="gemini_insight_button"):

        forecast_summary = forecast_df[
            ["Forecast Month", "Predicted Crime Count"]
        ].copy()

        forecast_summary["Forecast Month"] = (
            forecast_summary["Forecast Month"].dt.strftime("%B %Y")
        )

        # Prepare historical context for Gemini
        historical_context = historical_data.tail(12).to_string(index=False)

        prompt = f"""
You are an experienced public-safety data analyst preparing
a concise, evidence-based briefing for stakeholders who
monitor monthly crime volume and plan operational capacity.

Analyze the provided XGBoost forecast and historical data.

Your insights must be specific, numerically accurate, and
directly relevant to stakeholder decision-support.

Use ONLY the information provided. Do not invent causes,
external factors, or conclusions that the data cannot support.

================ FORECAST DATA ================

Forecast period: {forecast_start.strftime("%B %Y")} to {forecast_end.strftime("%B %Y")}
Monthly forecast values: {forecast_values.tolist()}
Average forecast: {average_forecast:.2f}
Forecast range: {forecast_range:.2f}

================ HISTORICAL DATA ================

Recent historical monthly data:
{historical_context}

Use the historical values to provide context.
Do not claim that the forecast is seasonally comparable
unless the supplied data supports that conclusion.

================ MODEL PERFORMANCE ================

Model: XGBoost Regressor
MAE: 171.47
RMSE: 223.73
R² Score: 0.4677
Naive Baseline MAE: 153.97

================ REQUIRED ANALYSIS ================

1. FORECAST OVERVIEW
- Identify the highest and lowest projected months.
- Explain the overall direction of the forecast.
- Include relevant forecast values and the overall range.
- Focus on expected monthly crime volume.

2. KEY CHANGES
- Identify meaningful month-to-month increases or decreases.
- Quantify important changes using actual forecast values.
- Distinguish substantial changes from minor fluctuations.
- Avoid repetition.

3. HISTORICAL CONTEXT
- Compare the forecast with recent historical data.
- Explain whether projected volumes are higher, lower, or similar.
- Mention the change from the final historical month when useful.
- Do not claim unsupported seasonality or unusual patterns.

4. PLANNING CONSIDERATIONS
- Explain how projected volumes may support workload monitoring.
- Identify months that may warrant additional attention.
- Use cautious language.
- Do not prescribe exact staffing or operational actions.
- Do not imply that the forecast identifies causes, locations,
  or specific crime categories.

5. MODEL LIMITATIONS
- Explain that forecasts are estimates, not guarantees.
- Report the supplied evaluation metrics accurately.
- State that XGBoost MAE (171.47) is higher than the
  Naive Baseline MAE (153.97).
- Explain that forecasts should support, not replace,
  professional judgment and current operational information.

================ OUTPUT REQUIREMENTS ================

- Use exactly five numbered headings.
- Provide 2–3 concise bullet points per section.
- Include specific numerical evidence.
- Avoid generic statements and repetition.
- Do not speculate about causes of crime.
- Keep the complete briefing under 300 words.
"""

        with st.spinner("Generating AI insights..."):

            try:
                insight = generate_gemini_insight(prompt)

                st.success("AI insights generated successfully!")
                st.markdown(insight)

            except Exception as error:
                st.error("Unable to generate Gemini insights.")
                st.exception(error)

        st.markdown(
            f"""
            <div class="insight-box">
                <strong>Forecast Summary:</strong><br>
                The selected horizon contains an average predicted crime count
                of <strong>{average_forecast:,.0f}</strong> per month.
                The difference between the highest and lowest predicted values
                is <strong>{forecast_range:,.0f}</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------
# Ask Gemini - Targeted Analytical Question
# --------------------------------------------------------

st.subheader("💬 Ask Gemini")

st.caption(
    "Ask a specific question about the selected forecast, "
    "historical data, or model performance."
)

question = st.text_area(
    "Your question",
    placeholder="e.g., Which month has the highest predicted crime count?",
    height=90,
    key="gemini_question_input"
)

if st.button("💬 Ask Gemini", key="gemini_question_button"):

    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        question_prompt = f"""
You are a data analyst helping a stakeholder interpret a
monthly crime forecasting dashboard.

Answer the user's specific question directly and concisely.

IMPORTANT RULES:
- Use ONLY the forecast, historical data, and model-performance
  information supplied below.
- Do not use the five-section forecast briefing format.
- Do not create numbered sections unless they are necessary
  to answer the question.
- Include relevant numerical evidence when available.
- Do not invent causes, external factors, crime locations,
  crime categories, or operational recommendations.
- Do not claim that a forecast is a confirmed future observation.
- If the supplied information cannot answer the question,
  clearly say that the available data is insufficient.
- Keep the answer focused on the user's question.

USER QUESTION:
{question}

================ FORECAST DATA ================

Forecast period:
{forecast_start.strftime("%B %Y")} to {forecast_end.strftime("%B %Y")}

Monthly forecast values:
{forecast_values.tolist()}

Average forecast:
{average_forecast:.2f}

Forecast range:
{forecast_range:.2f}

Forecast direction:
{forecast_direction}

Highest predicted month:
{highest_forecast["Forecast Month"].strftime("%B %Y")}

Highest predicted count:
{int(highest_forecast["Predicted Crime Count"]):,}

Lowest predicted month:
{lowest_forecast["Forecast Month"].strftime("%B %Y")}

Lowest predicted count:
{int(lowest_forecast["Predicted Crime Count"]):,}

================ HISTORICAL DATA ================

Recent historical monthly data:
{historical_data}

================ MODEL PERFORMANCE ================

Model:
XGBoost Regressor

MAE:
{VALIDATION_METRICS["MAE"]:.2f}

RMSE:
{VALIDATION_METRICS["RMSE"]:.2f}

R²:
{VALIDATION_METRICS["R2"]:.4f}

Naive Baseline MAE:
{baseline_mae:.2f}
"""

        with st.spinner("Generating answer..."):
            try:
                answer = generate_gemini_insight(question_prompt)

                st.success("Answer generated successfully!")
                st.markdown(answer)

            except Exception as error:
                st.error("Unable to generate Gemini answer.")
                st.exception(error)

    # --------------------------------------------------------
    # Decision Support Section
    # --------------------------------------------------------

    st.subheader("Decision Support Context")

    st.markdown(
        """
        <div class="insight-box">
        <strong>Potential analytical applications:</strong>
        <ul>
        <li>Identify expected changes in monthly crime volume.</li>
        <li>Support exploratory resource-planning discussions.</li>
        <li>Compare projected patterns across different horizons.</li>
        <li>Provide a starting point for further investigation.</li>
        </ul>
        <strong>Important:</strong>
        These forecasts should not independently determine
        public-safety actions, resource allocation, or operational decisions.
        Additional validation, current data, and domain expertise are required.
        </div>
        """,
        unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Forecast Limitations
    # --------------------------------------------------------

    st.warning(
        "These are model-generated forecasts, not confirmed future observations. "
        "Recursive uncertainty may increase as the forecast horizon expands."
    )

# ============================================================
# TAB 3: HISTORICAL TRENDS
# ============================================================

with tab_history:
    st.markdown('<div class="section-kicker">EXPLORATORY ANALYSIS</div>', unsafe_allow_html=True)
    st.header("Historical Trends")
    st.write("Explore annual movement, monthly seasonality, and summary statistics derived from the saved monthly series.")

    history_df = historical_data.rename("Crime Count").to_frame()
    history_df.index.name = "Date"
    history_df["Year"] = history_df.index.year
    history_df["Month"] = history_df.index.month
    history_df["Month Name"] = history_df.index.strftime("%b")

    annual = history_df.groupby("Year", as_index=True)["Crime Count"].sum().rename("Annual Crime Count")
    monthly_seasonality = history_df.groupby("Month", as_index=True)["Crime Count"].mean().rename("Average Monthly Crime Count")
    monthly_seasonality.index = [pd.Timestamp(2000, month, 1).strftime("%b") for month in monthly_seasonality.index]

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.subheader("Annual Crime Volume")
        annual_chart = annual.rename_axis("Year").reset_index()
        st.plotly_chart(
            make_bar_chart(annual_chart, "Year", "Annual Crime Count", height=330),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with chart_col_2:
        st.subheader("Average Crime by Calendar Month")
        seasonality_chart = monthly_seasonality.rename_axis("Month").reset_index()
        st.plotly_chart(
            make_bar_chart(
                seasonality_chart,
                "Month",
                "Average Monthly Crime Count",
                height=330,
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.subheader("Rolling Trend")
    rolling_df = history_df[["Crime Count"]].copy()
    rolling_df["3-Month Rolling Mean"] = rolling_df["Crime Count"].rolling(3).mean()
    rolling_df["12-Month Rolling Mean"] = rolling_df["Crime Count"].rolling(12).mean()
    rolling_chart = rolling_df.reset_index().rename(columns={"index": "Date"})
    st.plotly_chart(
        make_line_chart(
            rolling_chart,
            "Date",
            ["Crime Count", "3-Month Rolling Mean", "12-Month Rolling Mean"],
            height=380,
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )

    summary_1, summary_2, summary_3, summary_4 = st.columns(4)
    with summary_1:
        st.metric("Average Monthly Count", f"{historical_data.mean():,.0f}")
    with summary_2:
        st.metric("Median Monthly Count", f"{historical_data.median():,.0f}")
    with summary_3:
        st.metric("Maximum Monthly Count", f"{historical_data.max():,.0f}")
    with summary_4:
        st.metric("Minimum Monthly Count", f"{historical_data.min():,.0f}")

    st.subheader("Historical Data")
    historical_display = historical_data.rename("Crime Count").to_frame().reset_index()
    historical_display.columns = ["Date", "Crime Count"]
    historical_display["Date"] = historical_display["Date"].dt.strftime("%B %Y")
    st.dataframe(historical_display, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Historical Data CSV",
        data=historical_display.to_csv(index=False).encode("utf-8"),
        file_name="historical_monthly_crime_data.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 4: MODEL PERFORMANCE
# ============================================================

with tab_performance:

    st.markdown(
        '<div class="section-kicker">MODEL EVALUATION</div>',
        unsafe_allow_html=True
    )

    st.header("Model Performance")

    st.write(
        "The evaluation uses chronological walk-forward validation "
        "to assess the XGBoost forecasting model. Lower MAE and RMSE "
        "indicate smaller prediction errors, while higher R² indicates "
        "greater explained variation within this evaluation setup."
    )

    # --------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------

    performance_1, performance_2, performance_3 = st.columns(3)

    with performance_1:
        st.metric(
            "XGBoost MAE",
            f"{VALIDATION_METRICS['MAE']:.2f}"
        )

    with performance_2:
        st.metric(
            "XGBoost RMSE",
            f"{VALIDATION_METRICS['RMSE']:.2f}"
        )

    with performance_3:
        st.metric(
            "XGBoost R²",
            f"{VALIDATION_METRICS['R2']:.4f}"
        )

    st.caption(
        "Metrics are based on one-step-ahead chronological "
        "walk-forward validation."
    )

    st.divider()

    # --------------------------------------------------
    # ACTUAL VS PREDICTED VALIDATION
    # --------------------------------------------------

    validation_data = validation_data.copy()

    validation_data["Date"] = pd.to_datetime(
        validation_data["Date"]
    )

    validation_data = validation_data.sort_values(
        "Date"
    ).reset_index(drop=True)

    st.subheader("Actual vs Predicted Crime Counts")

    fig_validation = go.Figure()

    fig_validation.add_trace(
        go.Scatter(
            x=validation_data["Date"],
            y=validation_data["Actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(
                color=NAVY,
                width=2
            )
        )
    )

    fig_validation.add_trace(
        go.Scatter(
            x=validation_data["Date"],
            y=validation_data["Predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(
                color=ORANGE,
                width=2,
                dash="dash"
            )
        )
    )

    fig_validation.update_layout(
        xaxis_title="Date",
        yaxis_title="Monthly Crime Count",
        hovermode="x unified",
        height=450,
        legend_title="Series"
    )

    st.plotly_chart(
        fig_validation,
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.caption(
        "Actual versus predicted values from one-step-ahead "
        "walk-forward validation."
    )

    st.divider()

    # --------------------------------------------------
    # BASELINE COMPARISON
    # --------------------------------------------------

    st.subheader("XGBoost vs Naive Baseline")

    st.write(
        "The naive baseline uses the previous month's actual "
        "crime count as the prediction."
    )

    xgb_mae = (
        validation_data["Actual"]
        - validation_data["Predicted"]
    ).abs().mean()

    baseline_comparison = pd.DataFrame(
        {
            "Model": [
                "Naive Baseline",
                "XGBoost"
            ],
            "MAE": [
                baseline_mae,
                xgb_mae
            ]
        }
    )

    st.dataframe(
        baseline_comparison.round(2),
        use_container_width=True,
        hide_index=True
    )

    baseline_chart = baseline_comparison.copy()

    st.plotly_chart(
        make_bar_chart(
            baseline_chart,
            "Model",
            "MAE",
            height=350
        ),
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.caption(
        "Lower MAE indicates smaller average absolute "
        "prediction errors."
    )

    st.divider()

    # --------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------

    st.subheader("Feature Importance")

    st.caption(
        "Documented XGBoost feature-importance values "
        "from the project analysis."
    )

    importance_chart = (
        FEATURE_IMPORTANCE
        .set_index("Feature")[["Importance (%)"]]
    )

    importance_plot = (
        importance_chart
        .rename_axis("Feature")
        .reset_index()
    )

    st.plotly_chart(
        make_bar_chart(
            importance_plot,
            "Feature",
            "Importance (%)",
            height=350
        ),
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.dataframe(
        FEATURE_IMPORTANCE,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="insight-box">
            <strong>Interpretation:</strong>
            Lag_12 and Rolling_Mean_12 have the largest documented
            feature-importance values. Feature importance indicates
            contribution to model behaviour; it does not establish
            that a feature causes crime levels to change.
        </div>
        """,
        unsafe_allow_html=True
    )



# ============================================================
# TAB 5: METHODOLOGY
# ============================================================

with tab_methodology:
    st.markdown('<div class="section-kicker">TECHNICAL DOCUMENTATION</div>', unsafe_allow_html=True)
    st.header("Methodology")

    st.subheader("Workflow")
    st.markdown(
        """
        1. Load and inspect incident-level crime records.
        2. Remove exact duplicate records.
        3. Handle missing values and standardize date-related fields.
        4. Aggregate incident records into monthly crime counts.
        5. Create lag features: 1, 2, 3, 6, and 12 months.
        6. Create rolling-average features: 3, 6, and 12 months.
        7. Compare forecasting models using chronological walk-forward validation.
        8. Train the final XGBoost model using the available monthly training history.
        9. Generate recursive forecasts for the selected future horizon.
        10. Use feature importance and SHAP analysis for interpretation.
        """
    )

    st.subheader("Forecasting Features")
    feature_names = metadata.get("feature_names", [])
    st.write("Model feature order:")
    st.code("\n".join(feature_names) if feature_names else "Feature names were not found in metadata.")

    st.subheader("Validation Scope")
    st.markdown(
        """
        - Validation type: chronological walk-forward validation.
        - Forecast target: monthly crime count.
        - Reported results: one-step-ahead validation.
        - Deployment forecast: recursive multi-month prediction.
        - Model: XGBoost Regressor.
        """
    )

    st.subheader("Limitations")
    st.markdown(
        """
        - The saved deployment series contains monthly aggregate crime counts, not the complete incident-level dataset.
        - The displayed feature-importance values come from the documented project analysis.
        - Recursive forecasts may accumulate error across future steps.
        - The model requires further validation before operational public-safety use.
        - Historical relationships should not be interpreted as causal conclusions.
        """
    )

    st.subheader("Deployment Artifacts")
    st.markdown(
        """
        - `final_xgb_model.json`
        - `historical_monthly_crime_data.pkl`
        - `xgb_model_metadata.pkl`
        """
    )

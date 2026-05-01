from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components.charts import forecast_chart
from app.components.chatbot import render_chatbot
from src.data_loader import get_cached_data
from src.model import MODEL_PATH, fit_full_forecaster, load_model, predict_submission

css_path = Path(__file__).resolve().parents[1] / "style" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .forecast-header {
        display: grid;
        grid-template-columns: 180px 1fr;
        gap: 14px;
        align-items: center;
        margin-bottom: 12px;
    }
    .forecast-brand {
        background: #0f4a85;
        color: #ffffff;
        font-weight: 800;
        font-size: 30px;
        line-height: 1;
        padding: 18px 14px;
        border-radius: 6px;
        text-align: center;
        letter-spacing: 1px;
    }
    .forecast-title {
        border-left: 4px solid #0f4a85;
        padding-left: 12px;
    }
    .forecast-main {
        font-size: 28px;
        font-weight: 700;
        color: #1c2f4a;
        line-height: 1.1;
    }
    .forecast-sub {
        font-size: 12px;
        color: #5f6b7a;
        margin-top: 2px;
    }
    .kpi-card {
        border: 1px solid #dbe4ef;
        border-radius: 6px;
        padding: 12px 14px;
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin: 4px 0 12px 0;
    }
    .kpi-label {
        font-size: 12px;
        color: #5f6b7a;
        margin: 0;
    }
    .kpi-value {
        font-size: 24px;
        color: #17324d;
        font-weight: 700;
        margin: 4px 0 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="forecast-header">
      <div class="forecast-brand">K-TEAM</div>
      <div class="forecast-title">
        <div class="forecast-main">Predictive Forecast Dashboard</div>
        <div class="forecast-sub">Demand outlook, risk flags, and trend diagnostics</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

data = get_cached_data()
render_chatbot(data)
sales = data["sales"].sort_values("Date").copy()
sales_test = data.get("sales_test")
sample_sub = data.get("sample_sub")
promotions = data.get("promotions")
web_traffic = data.get("web_traffic")
inventory = data.get("inventory")


def _data_signature(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty"
    date_max = pd.to_datetime(df["Date"]).max() if "Date" in df.columns else "na"
    return f"rows={len(df)}|cols={len(df.columns)}|date_max={date_max}"


@st.cache_resource(show_spinner=False)
def _get_or_train_bundle(sales_sig: str, _promotions_sig: str, _traffic_sig: str, _inventory_sig: str):
    try:
        return load_model()
    except (FileNotFoundError, ValueError, EOFError):
        try:
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()
        except Exception:
            pass
        return fit_full_forecaster(
            sales,
            promotions_df=promotions,
            web_traffic_df=web_traffic,
            inventory_df=inventory,
            save_artifacts=True,
        )


def _sig_promotions(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "none"
    return f"rows={len(df)}|cols={len(df.columns)}"


def _sig_traffic(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "none"
    date_max = pd.to_datetime(df["date"]).max() if "date" in df.columns else "na"
    return f"rows={len(df)}|date_max={date_max}"


def _sig_inventory(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "none"
    snap_max = pd.to_datetime(df["snapshot_date"]).max() if "snapshot_date" in df.columns else "na"
    return f"rows={len(df)}|snap_max={snap_max}"


with st.sidebar.expander("Forecast page settings", expanded=False):
    show_powerbi = st.toggle("Show PowerBI embed", value=False)
    powerbi_url = st.text_input(
        "PowerBI embed URL (optional)",
        value=st.session_state.get("forecast_powerbi_url", ""),
        placeholder="Paste PowerBI 'view?r=...' URL here",
    )
    if powerbi_url.strip():
        st.session_state["forecast_powerbi_url"] = powerbi_url.strip()

if show_powerbi:
    url = st.session_state.get("forecast_powerbi_url", "")
    if not url:
        st.info("Bạn bật PowerBI embed nhưng chưa có URL. Hãy paste link vào sidebar.")
    else:
        st.components.v1.html(
            f"""
            <iframe title="Forecast Dashboard"
            width="100%"
            height="1150"
            src="{url}"
            frameborder="0"
            allowFullScreen="true">
            </iframe>
            """,
            height=1150,
        )
        st.stop()

with st.spinner("Loading forecast model (cached)..."):
    bundle = _get_or_train_bundle(
        _data_signature(sales),
        _sig_promotions(promotions),
        _sig_traffic(web_traffic),
        _sig_inventory(inventory),
    )

if sample_sub is not None and not sample_sub.empty:
    # Keep the official submission row order exactly as provided.
    future_df = sample_sub[["Date"]].copy()
elif sales_test is not None and not sales_test.empty:
    future_df = sales_test[["Date"]].copy()
else:
    future_dates = pd.date_range(sales["Date"].max() + pd.Timedelta(days=1), periods=30, freq="D")
    future_df = pd.DataFrame({"Date": future_dates})


@st.cache_data(show_spinner=False)
def _cached_forecast(sales_sig: str, future_sig: str) -> pd.DataFrame:
    out = predict_submission(
        bundle=bundle,
        sales_df=sales[["Date", "Revenue", "COGS"]],
        sample_submission_df=future_df,
        promotions_df=promotions,
        web_traffic_df=web_traffic,
        inventory_df=inventory,
    )
    return out


future_sig = f"rows={len(future_df)}|date_min={pd.to_datetime(future_df['Date']).min()}|date_max={pd.to_datetime(future_df['Date']).max()}"
forecast = _cached_forecast(_data_signature(sales), future_sig)
forecast["Date"] = pd.to_datetime(forecast["Date"])
forecast["Forecast"] = forecast["Revenue"]

sales["Date"] = pd.to_datetime(sales["Date"])
sales = sales.sort_values("Date")
forecast = forecast.sort_values("Date")

high_threshold = sales["Revenue"].quantile(0.9) if not sales.empty else 0
low_threshold = sales["Revenue"].quantile(0.1) if not sales.empty else 0
peak_forecast = float(forecast["Forecast"].max()) if not forecast.empty else 0
min_forecast = float(forecast["Forecast"].min()) if not forecast.empty else 0
next_day_forecast = float(forecast["Forecast"].iloc[0]) if not forecast.empty else 0
forecast_trend_delta = (
    ((float(forecast["Forecast"].iloc[-1]) - next_day_forecast) / next_day_forecast) * 100
    if (not forecast.empty and next_day_forecast != 0)
    else 0
)


def _kpi_card(label: str, value: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _place_legend_top(fig):
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0,
        ),
        legend_title=None,
    )
    return fig


def _force_y_axis_from_zero(fig):
    fig.update_yaxes(rangemode="tozero")
    return fig


@st.cache_data(show_spinner=False)
def _load_metric_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    val_path = Path("outputs") / "forecast_validation_metrics.csv"
    cmp_path = Path("outputs") / "forecast_model_comparison.csv"
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    cmp_df = pd.read_csv(cmp_path) if cmp_path.exists() else None
    return val_df, cmp_df


k1, k2, k3, k4 = st.columns(4)
with k1:
    _kpi_card("History days", f"{len(sales):,}")
with k2:
    _kpi_card("Forecast horizon", f"{len(forecast):,} days")
with k3:
    _kpi_card("Last actual revenue", f"{sales['Revenue'].iloc[-1]:,.0f}")
with k4:
    _kpi_card("Next-day forecast", f"{next_day_forecast:,.0f}")

val_metrics_df, model_cmp_df = _load_metric_tables()
if val_metrics_df is not None and not val_metrics_df.empty:
    st.subheader("Forecast Quality (R2 / MAE / RMSE)")
    metric_cards = st.columns(6)
    rev_row = val_metrics_df[val_metrics_df["target"] == "Revenue"]
    cogs_row = val_metrics_df[val_metrics_df["target"] == "COGS"]

    if not rev_row.empty:
        r = rev_row.iloc[0]
        with metric_cards[0]:
            _kpi_card("Revenue R2", f"{float(r['r2']):.3f}")
        with metric_cards[1]:
            _kpi_card("Revenue MAE", f"{float(r['mae']):,.0f}")
        with metric_cards[2]:
            _kpi_card("Revenue RMSE", f"{float(r['rmse']):,.0f}")

    if not cogs_row.empty:
        c = cogs_row.iloc[0]
        with metric_cards[3]:
            _kpi_card("COGS R2", f"{float(c['r2']):.3f}")
        with metric_cards[4]:
            _kpi_card("COGS MAE", f"{float(c['mae']):,.0f}")
        with metric_cards[5]:
            _kpi_card("COGS RMSE", f"{float(c['rmse']):,.0f}")

if model_cmp_df is not None and not model_cmp_df.empty:
    chosen_metric = st.selectbox(
        "So sánh model theo metric",
        options=["r2", "rmse", "mae"],
        index=0,
        help="R2 càng cao càng tốt; MAE/RMSE càng thấp càng tốt.",
    )
    cmp_plot_df = model_cmp_df.copy()
    cmp_plot_df[chosen_metric] = cmp_plot_df[chosen_metric].astype(float)
    cmp_fig = px.bar(
        cmp_plot_df,
        x="model",
        y=chosen_metric,
        color="target",
        barmode="group",
        title=f"Model comparison by {chosen_metric.upper()}",
        color_discrete_map={"Revenue": "#2563eb", "COGS": "#dc2626"},
    )
    cmp_fig.update_layout(margin=dict(l=20, r=20, t=60, b=20), height=360)
    if chosen_metric in {"mae", "rmse"}:
        _force_y_axis_from_zero(cmp_fig)
    _place_legend_top(cmp_fig)
    st.plotly_chart(cmp_fig, use_container_width=True)

st.subheader("Doanh số thực tế và dự báo")
main_fig = forecast_chart(sales, forecast)
_force_y_axis_from_zero(main_fig)
_place_legend_top(main_fig)
st.plotly_chart(main_fig, use_container_width=True)

if next_day_forecast >= high_threshold:
    st.warning(
        f"Forecast ngắn hạn đang ở vùng cao: {next_day_forecast:,.0f} (>= ngưỡng cao {high_threshold:,.0f}). Cần theo dõi rủi ro hết hàng."
    )
elif next_day_forecast <= low_threshold:
    st.warning(
        f"Forecast ngắn hạn đang ở vùng thấp: {next_day_forecast:,.0f} (<= ngưỡng thấp {low_threshold:,.0f}). Nên rà soát khuyến mãi/marketing."
    )
else:
    st.success("Forecast ngắn hạn đang ở vùng ổn định so với lịch sử.")

analysis_left, analysis_right = st.columns(2)
with analysis_left:
    combined = pd.concat(
        [
            sales[["Date", "Revenue"]].assign(Type="Actual").rename(columns={"Revenue": "Value"}),
            forecast[["Date", "Forecast"]].assign(Type="Forecast").rename(columns={"Forecast": "Value"}),
        ],
        ignore_index=True,
    )
    trend_fig = px.line(
        combined,
        x="Date",
        y="Value",
        color="Type",
        title="Xu hướng doanh số theo thời gian",
        color_discrete_map={"Actual": "#2563eb", "Forecast": "#dc2626"},
    )
    trend_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
    _force_y_axis_from_zero(trend_fig)
    _place_legend_top(trend_fig)
    trend_fig.update_yaxes(title=None, tickformat=",")
    st.plotly_chart(trend_fig, use_container_width=True)

with analysis_right:
    tail_actual = sales.tail(min(120, len(sales))).copy()
    tail_forecast = forecast.copy()
    if not tail_actual.empty:
        base = float(tail_actual["Revenue"].iloc[0]) if float(tail_actual["Revenue"].iloc[0]) != 0 else 1.0
        norm_actual = tail_actual[["Date", "Revenue"]].copy()
        norm_actual["Index"] = (norm_actual["Revenue"] / base) * 100
        norm_actual["Type"] = "Actual (normalized)"

        norm_forecast = tail_forecast[["Date", "Forecast"]].copy()
        norm_forecast["Index"] = (norm_forecast["Forecast"] / base) * 100
        norm_forecast["Type"] = "Forecast (normalized)"

        normalized = pd.concat(
            [norm_actual[["Date", "Index", "Type"]], norm_forecast[["Date", "Index", "Type"]]],
            ignore_index=True,
        )
        norm_fig = px.line(
            normalized,
            x="Date",
            y="Index",
            color="Type",
            title="Xu hướng chuẩn hóa (mốc cơ sở = 100)",
            color_discrete_map={"Actual (normalized)": "#0f766e", "Forecast (normalized)": "#7c3aed"},
        )
        norm_fig.add_hline(y=100, line_dash="dash", line_color="gray")
        norm_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
        _force_y_axis_from_zero(norm_fig)
        _place_legend_top(norm_fig)
        st.plotly_chart(norm_fig, use_container_width=True)

st.subheader("Chẩn đoán dự báo")
diag1, diag2 = st.columns(2)
with diag1:
    monthly = sales.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.groupby("Month", as_index=False)["Revenue"].sum()
    monthly["RollingMean3M"] = monthly["Revenue"].rolling(3, min_periods=1).mean()
    month_fig = go.Figure()
    month_fig.add_trace(
        go.Bar(x=monthly["Month"], y=monthly["Revenue"], name="Doanh số theo tháng", marker_color="#93c5fd")
    )
    month_fig.add_trace(
        go.Scatter(
            x=monthly["Month"],
            y=monthly["RollingMean3M"],
            mode="lines",
            name="Trung bình trượt 3 tháng",
            line=dict(color="#1d4ed8", width=3),
        )
    )
    month_fig.update_layout(
        title="Doanh số tháng và xu hướng trung bình",
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    _force_y_axis_from_zero(month_fig)
    _place_legend_top(month_fig)
    month_fig.update_yaxes(tickformat=",")
    st.plotly_chart(month_fig, use_container_width=True)

with diag2:
    future_diag = forecast.copy()
    future_diag["Day"] = range(1, len(future_diag) + 1)
    future_diag["Band"] = future_diag["Forecast"].apply(
        lambda x: "High" if x >= high_threshold else ("Low" if x <= low_threshold else "Normal")
    )
    band_fig = px.scatter(
        future_diag,
        x="Day",
        y="Forecast",
        color="Band",
        title="Phân vùng rủi ro dự báo",
        color_discrete_map={"High": "#dc2626", "Normal": "#0f766e", "Low": "#d97706"},
    )
    band_fig.update_traces(
        selector=dict(name="High"),
        name="Cao",
    )
    band_fig.update_traces(
        selector=dict(name="Normal"),
        name="Bình thường",
    )
    band_fig.update_traces(
        selector=dict(name="Low"),
        name="Thấp",
    )
    band_fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20))
    _force_y_axis_from_zero(band_fig)
    _place_legend_top(band_fig)
    band_fig.update_yaxes(tickformat=",")
    st.plotly_chart(band_fig, use_container_width=True)

st.info(
    f"Forecast range: thấp nhất {min_forecast:,.0f}, cao nhất {peak_forecast:,.0f}, xu hướng cuối kỳ {forecast_trend_delta:+.1f}% so với ngày đầu forecast."
)

st.markdown("### Bảng dự báo")
st.dataframe(forecast.head(60), use_container_width=True)

csv = forecast.to_csv(index=False).encode("utf-8")
btn_col, _ = st.columns([1, 6])
with btn_col:
    st.download_button(
        "Tải CSV",
        data=csv,
        file_name="forecast.csv",
        mime="text/csv",
        use_container_width=False,
    )

from pathlib import Path

import pandas as pd
import polars as pl
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu

from app.components.chatbot import render_chatbot
from src.data_loader import get_cached_data
from src.optimizer import OptimizeRequest, OptimizationProblem, Optimizer

try:
    from src.optimizer import SCENARIO_WEIGHTS
except ImportError:
    SCENARIO_WEIGHTS = {
        "Balanced": dict(revenue=0.35, profit=0.35, acquire=0.15, retain=0.15)
    }

css_path = Path(__file__).resolve().parents[1] / "style" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Global font and typography */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    html, body, [class*="css-"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    .dash-card {
        border: 1px solid #dbe4ef;
        border-radius: 5px;
        padding: 16px 18px;
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        margin: 4px 0 16px 0;
    }
    .dash-card h4 {
        margin: 0;
        color: #17324d;
        font-size: 14px;
        font-weight: 600;
    }
    .dash-muted {
        color: #5f6b7a;
        font-size: 11px;
        margin: 6px 0 0 0;
    }

    .report-header {
        display: grid;
        grid-template-columns: 200px 1fr;
        gap: 14px;
        align-items: center;
        margin-bottom: 20px;
    }
    .brand-box {
        background: #0f4a85;
        color: #ffffff;
        font-weight: 800;
        font-size: 34px;
        line-height: 1;
        padding: 20px 16px;
        border-radius: 5px;
        letter-spacing: 1px;
        text-align: center;
    }
    .title-box {
        border-left: 4px solid #0f4a85;
        padding-left: 14px;
    }
    .title-main {
        font-size: 30px;
        font-weight: 700;
        color: #1c2f4a;
        line-height: 1.1;
    }
    .title-sub {
        font-size: 12px;
        color: #8a3b1d;
        margin-top: 2px;
    }

    /* Improve metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 10px;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 6px 4px;
    }
    .metric-label {
        font-size: 12px;
        opacity: 0.8;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 23px;
        font-weight: bold;
        margin: 8px 0;
    }
    .metric-delta {
        color: #10b981;
        font-size: 11px;
    }

    /* Better spacing for columns */
    .stColumn {
        padding: 0 8px;
    }

    /* Improve button styling */
    .stButton button {
        border-radius: 5px;
        font-weight: 500;
        min-height: 42px;
        font-size: 13px;
    }

    /* Form styling */
    .stForm {
        background: rgba(255,255,255,0.8);
        border-radius: 5px;
        padding: 18px;
        margin: 12px 0 16px 0;
        border: 1px solid #e5ecf4;
    }

    /* Soften inputs and spacing */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stNumberInput > div > div > input {
        border-radius: 5px !important;
        font-size: 13px !important;
    }

    .stPlotlyChart {
        border: 1px solid #e8edf3;
        border-radius: 5px;
        padding: 8px;
        background: #ffffff;
    }

    .stSubheader {
        font-size: 1.22rem !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def apply_chart_style(fig, height=520):
    fig.update_layout(
        template="plotly_white",
        height=height,

        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

        margin=dict(l=20, r=20, t=110, b=40),

        font=dict(
            family="Inter, sans-serif",
            size=13,
            color="#1e293b"
        ),

        title=dict(
            x=0,
            xanchor="left",
            font=dict(size=20, color="#0f172a")
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            itemwidth=40
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Inter"
        ),

        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.15)",
            zeroline=False,
            showline=False,
            automargin=True
        ),

        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            automargin=True
        )
    )

    fig.update_traces(
        marker_line_width=0,
        opacity=0.92
    )

    return fig


def _styled_bar(df, x, y, title, color=None, orientation="v", text_auto=".2s", height=480):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        orientation=orientation,
        text_auto=text_auto,
        title=title,
        color_discrete_sequence=[
            "#2563eb",
            "#0f766e",
            "#7c3aed",
            "#ea580c",
            "#dc2626",
        ],
    )
    fig.update_traces(
        # Keep labels inside plotting area to avoid overlapping with surrounding elements.
        textposition="auto",
        cliponaxis=True,
    )
    return apply_chart_style(fig, height)


def _sig_table(df) -> str:
    if df is None:
        return "none"
    try:
        return f"rows={len(df)}|cols={len(df.columns)}"
    except Exception:
        return "unknown"


def custom_metric(label, value, delta=None):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def _fmt_num(value, fmt=":,.0f", na="N/A"):
    if value is None:
        return na
    if isinstance(value, float) and pd.isna(value):
        return na
    return format(value, fmt.lstrip(":"))


def _apply_pandas_filters(
    df: pd.DataFrame,
    selections: dict,
    all_value: str = "Tất cả",
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df
    for col, sel in selections.items():
        if sel is None or sel == all_value:
            continue
        if col in out.columns:
            out = out.loc[out[col] == sel]
    return out


def _filter_form_for_df(
    df: pd.DataFrame,
    *,
    form_key: str,
    fields: list[tuple[str, str]],
    all_value: str = "Tất cả",
):
    """
    fields: list of (column_name, label)
    Returns: (filtered_df, selections_dict, summary_text)
    """
    selections: dict = {}
    if df is None:
        return df, selections, "Không có dữ liệu"

    with st.form(form_key):
        cols = st.columns(min(len(fields), 5) or 1)
        for i, (col, label) in enumerate(fields):
            if col not in df.columns:
                continue
            options = [all_value] + sorted([v for v in df[col].dropna().unique().tolist()])
            with cols[i % len(cols)]:
                selections[col] = st.selectbox(label, options=options, index=0, key=f"{form_key}_{col}")
        submitted = st.form_submit_button("Áp dụng bộ lọc", use_container_width=True)

    filtered = _apply_pandas_filters(df, selections, all_value=all_value)
    picked = [f"{label}: {selections.get(col, all_value)}" for col, label in fields if selections.get(col) not in (None, all_value)]
    summary = " • ".join(picked) if picked else "Phạm vi: Tất cả"
    return filtered, selections, summary


def _filter_by_master_scope(df: pd.DataFrame, master_scope: pd.DataFrame):
    """
    Apply Strategy-like filters to a tab dataframe using the filtered master scope.
    We only filter on columns that exist in BOTH dataframes (and a few known key fields).
    Returns: (filtered_df, applied_columns)
    """
    if df is None or df.empty or master_scope is None or master_scope.empty:
        return df, []

    out = df
    applied: list[str] = []

    # Direct dimension columns (preferred if available)
    dim_cols = ["category", "region", "rfm_segment", "month", "discount"]
    for c in dim_cols:
        if (c in out.columns) and (c in master_scope.columns):
            allowed = set(master_scope[c].dropna().unique().tolist())
            before = len(out)
            out = out.loc[out[c].isin(allowed)]
            if len(out) != before:
                applied.append(c)

    # Key-ish fields (best effort)
    key_cols = [
        "product_id",
        "product_name",
        "promo_id",
        "promo_name",
        "promotion_id",
        "promotion_name",
        "acquisition_channel",
        "customer_segment",
    ]
    for c in key_cols:
        if (c in out.columns) and (c in master_scope.columns):
            allowed = set(master_scope[c].dropna().unique().tolist())
            before = len(out)
            out = out.loc[out[c].isin(allowed)]
            if len(out) != before:
                applied.append(c)

    return out, applied


def _decision_block(insight: str, recommendation: str, impact: str, level: str = "success"):
    content = f"""
Nhận định:
{insight}

Khuyến nghị:
{recommendation}

Tác động kỳ vọng:
{impact}
"""
    if level == "warning":
        st.warning(content)
    elif level == "info":
        st.info(content)
    else:
        st.success(content)


st.markdown(
    """
    <div class="report-header">
      <div class="brand-box">K-TEAM</div>
      <div class="title-box">
        <div class="title-main">Revenue and Business Intelligence Report</div>
        <div class="title-sub">Prescriptive Optimization Dashboard</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


data = get_cached_data()
render_chatbot(data)


@st.cache_resource(show_spinner=False)
def _get_optimizer(products_sig, orders_sig, order_items_sig, promotions_sig, customers_sig, geography_sig, inventory_sig):
    return Optimizer(data)


with st.spinner("Loading optimizer..."):
    optimizer = _get_optimizer(
        _sig_table(data.get("products")),
        _sig_table(data.get("orders")),
        _sig_table(data.get("order_items")),
        _sig_table(data.get("promotions")),
        _sig_table(data.get("customers")),
        _sig_table(data.get("geography")),
        _sig_table(data.get("inventory")),
    )


@st.cache_data(show_spinner=False)
def _cached_prescriptive_outputs(simulation_runs: int):
    reorder = optimizer.inventory.suggest_reorder(top_n=10)
    overstock = optimizer.inventory.flag_overstock(top_n=10)
    promo = optimizer.promotion.rank_promos_by_roi(top_n=10)
    channel = optimizer.customer.optimize_acquisition_channel(top_n=10)
    retention_out = optimizer.customer.optimize_retention_cost(simulation_runs=simulation_runs)
    return reorder, overstock, promo, channel, retention_out


reorder_df, overstock_df, promo_df, channel_df, retention = _cached_prescriptive_outputs(simulation_runs=200)
retention_df = retention.get("simulation_summary")

st.markdown('<div class="metric-section">', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------- Global (Strategy-like) filters shared across tabs --------
@st.cache_resource(show_spinner=False)
def _load_master_table():
    master_path = Path("outputs") / "master_dashboard.parquet"
    if not master_path.exists():
        return None
    return pl.read_parquet(master_path)


master_df = _load_master_table()
global_filter_summary = "Phạm vi: Tất cả"
master_scope_pl = None
master_scope_pd = None
if master_df is not None:
    with st.form("global_strategy_filters"):
        f1, f2, f3, f4, f5 = st.columns(5)

        with f1:
            sel_category = st.selectbox(
                "Ngành hàng",
                options=["Tất cả"] + sorted(master_df["category"].unique().to_list()),
                index=0,
                key="global_sel_category",
            )
        with f2:
            sel_region = st.selectbox(
                "Khu vực",
                options=["Tất cả"] + sorted(master_df["region"].unique().to_list()),
                index=0,
                key="global_sel_region",
            )
        with f3:
            sel_segment = st.selectbox(
                "Phân khúc",
                options=["Tất cả"] + sorted(master_df["rfm_segment"].unique().to_list()),
                index=0,
                key="global_sel_segment",
            )
        with f4:
            sel_month = st.selectbox(
                "Tháng",
                options=["Tất cả"] + sorted(master_df["month"].unique().to_list()),
                index=0,
                key="global_sel_month",
            )
        with f5:
            sel_discount = st.selectbox(
                "Chiết khấu (%)",
                options=["Tất cả"] + sorted(master_df["discount"].unique().to_list()),
                index=0,
                key="global_sel_discount",
            )

        st.form_submit_button("Áp dụng bộ lọc", use_container_width=True)

    conditions = []
    if sel_category != "Tất cả":
        conditions.append(pl.col("category") == sel_category)
    if sel_region != "Tất cả":
        conditions.append(pl.col("region") == sel_region)
    if sel_segment != "Tất cả":
        conditions.append(pl.col("rfm_segment") == sel_segment)
    if sel_month != "Tất cả":
        conditions.append(pl.col("month") == sel_month)
    if sel_discount != "Tất cả":
        conditions.append(pl.col("discount") == sel_discount)

    master_scope_pl = master_df.filter(conditions[0] if len(conditions) == 1 else conditions[0].and_(*conditions[1:])) if conditions else master_df
    master_scope_pd = master_scope_pl.to_pandas() if master_scope_pl is not None else None
    picked = []
    if sel_category != "Tất cả":
        picked.append(f"Ngành hàng: {sel_category}")
    if sel_region != "Tất cả":
        picked.append(f"Khu vực: {sel_region}")
    if sel_segment != "Tất cả":
        picked.append(f"Phân khúc: {sel_segment}")
    if sel_month != "Tất cả":
        picked.append(f"Tháng: {sel_month}")
    if sel_discount != "Tất cả":
        picked.append(f"Chiết khấu: {sel_discount}")
    global_filter_summary = " • ".join(picked) if picked else "Phạm vi: Tất cả"
else:
    st.info("Không tìm thấy bảng master để lọc theo Region/Category/Segment/Month/Discount.")

# -------- Derived (filterable) analysis tables from master_scope --------
promo_scope = pd.DataFrame()
acq_scope = pd.DataFrame()
ret_scope = pd.DataFrame()
promo_peak_roi = 0.0
retention_net_benefit = 0.0

if master_scope_pl is not None:
    try:
        promo_peak_roi = float(master_scope_pl["roi"].max())
    except Exception:
        promo_peak_roi = 0.0

    ms_pd = master_scope_pl.to_pandas()
    if not ms_pd.empty:
        # Promotion-like view: effectiveness by discount level
        promo_scope = (
            ms_pd.groupby("discount", as_index=False)
            .agg(
                avg_roi=("roi", "mean"),
                avg_profit=("profit_value", "mean"),
                total_profit=("profit_value", "sum"),
                rows=("roi", "count"),
            )
            .sort_values("discount")
        )
        promo_scope["discount_pct"] = (promo_scope["discount"] * 100).round(0)
        promo_scope["recommendation"] = promo_scope["avg_roi"].apply(
            lambda x: "MỞ RỘNG" if x >= 1.8 else ("TỐI ƯU" if x >= 1.1 else "DỪNG/GIẢM")
        )

        # Acquisition-like view: where acquisition value is high
        acq_scope = (
            ms_pd.groupby(["region", "rfm_segment"], as_index=False)
            .agg(
                acquire=("acquire_value", "mean"),
                roi=("roi", "mean"),
                rows=("roi", "count"),
            )
            .sort_values(["acquire", "roi"], ascending=False)
        )

        # Retention-like view: retention value by segment & discount
        ret_scope = (
            ms_pd.groupby(["rfm_segment", "discount"], as_index=False)
            .agg(
                retain=("retain_value", "mean"),
                roi=("roi", "mean"),
                rows=("roi", "count"),
            )
            .sort_values(["retain", "roi"], ascending=False)
        )
        retention_net_benefit = float(ms_pd["retain_value"].sum())

# Inventory scope for action center (best-effort: by category from master_scope)
reorder_scope, _ = _filter_by_master_scope(reorder_df, master_scope_pd)
overstock_scope, _ = _filter_by_master_scope(overstock_df, master_scope_pd)
high_reorder = 0
if (reorder_scope is not None) and (not reorder_scope.empty) and ("recommended_order_qty" in reorder_scope.columns):
    high_reorder = int((reorder_scope["recommended_order_qty"] > reorder_scope["recommended_order_qty"].median()).sum())

action_df = pd.DataFrame(
    {
        "Mức độ ưu tiên": ["CAO", "CAO", "TRUNG BÌNH", "TRUNG BÌNH"],
        "Mảng": ["Khuyến mãi", "Giữ chân", "Tồn kho", "Thu hút"],
        "Khuyến nghị": [
            "Tập trung mức chiết khấu có ROI cao, giảm mức chiết khấu có ROI thấp (theo bộ lọc hiện tại).",
            "Ưu tiên phân khúc có giá trị giữ chân cao; chọn mức chiết khấu theo hiệu quả trong phạm vi lọc.",
            f"Đặt thêm hàng cho SKU nhu cầu cao ({high_reorder} sản phẩm) và giảm tồn kho dư (theo ngành hàng đã lọc).",
            "Ưu tiên khu vực/phân khúc có acquire_value & ROI cao trong phạm vi lọc hiện tại.",
        ],
        "Tác động kỳ vọng": [
            f"ROI tối đa trong phạm vi lọc có thể đạt {promo_peak_roi:.2f}",
            f"Tổng giá trị giữ chân (retain_value) trong phạm vi lọc: {retention_net_benefit:,.0f}",
            "Giảm rủi ro hết hàng và giải phóng vốn bị tồn",
            "Tăng hiệu quả thu hút khách hàng theo khu vực/phân khúc ưu tiên",
        ],
    }
)


selected = option_menu(
    menu_title=None,
    options=["Strategy", "Inventory", "Promotion", "Acquisition", "Retention"],
    icons=["cpu", "boxes", "tag", "megaphone", "arrow-repeat"],
    orientation="horizontal",
    styles={
        "container": {"padding": "2px 0 12px 0", "background-color": "transparent"},
        "icon": {"font-size": "14px"},
        "nav-link": {
            "font-size": "12px",
            "font-weight": "700",
            "text-transform": "uppercase",
            "letter-spacing": "0.5px",
            "padding": "10px 14px",
            "border-radius": "10px",
            "border": "1px solid #dbe4ef",
            "background-color": "#f3f8ff",
            "color": "#17324d",
            "margin": "0 8px 0 0",
        },
        "nav-link-selected": {
            "background-color": "#0f766e",
            "color": "white",
            "border": "1px solid #0f766e",
            "box-shadow": "0 6px 14px rgba(15, 118, 110, 0.25)",
        },
    },
)


if selected == "Strategy":
    if master_scope_pl is None:
        st.error("Không tìm thấy bảng master để hiển thị phân tích chiến lược.")
        st.stop()

    filtered_df = master_scope_pl
    st.caption(global_filter_summary)

    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

    with metrics_col1:
        st.markdown(
            custom_metric(
                "Số dòng",
                f"{len(filtered_df):,}",
            ),
            unsafe_allow_html=True
        )
    with metrics_col2:
        avg_revenue = filtered_df["revenue_value"].mean()
        st.markdown(
            custom_metric(
                "Doanh thu TB",
                _fmt_num(avg_revenue, ":,.0f"),
            ),
            unsafe_allow_html=True
        )
    with metrics_col3:
        avg_profit = filtered_df["profit_value"].mean()
        st.markdown(
            custom_metric(
                "Lợi nhuận TB",
                _fmt_num(avg_profit, ":,.0f"),
            ),
            unsafe_allow_html=True
        )
    with metrics_col4:
        avg_roi = filtered_df["roi"].mean()
        st.markdown(
            custom_metric(
                "ROI TB",
                _fmt_num(avg_roi, ":.2f"),
            ),
            unsafe_allow_html=True
        )

    # Decision summary + before/after optimization view
    if avg_roi < 1.2:
        strategy_decision = "Giảm mức chiết khấu ở các tổ hợp hiệu quả thấp."
        roi_uplift = 0.22
        profit_uplift = 0.08
    elif avg_profit > 50000:
        strategy_decision = "Tăng phân bổ tồn kho cho các tổ hợp biên lợi nhuận cao."
        roi_uplift = 0.12
        profit_uplift = 0.10
    else:
        strategy_decision = "Duy trì chiến lược hiện tại và tối ưu theo phân khúc/kênh."
        roi_uplift = 0.10
        profit_uplift = 0.06

    current_profit = float(avg_profit or 0)
    current_roi = float(avg_roi or 0)
    compare_df = pd.DataFrame(
        {
            "Chỉ số": ["Lợi nhuận", "ROI"],
            "Hiện tại": [current_profit, current_roi],
            "Đề xuất": [
                current_profit * (1 + profit_uplift),
                current_roi * (1 + roi_uplift),
            ],
        }
    )

    st.markdown("<div class='chart-spacing'></div>", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)

    with ch1:
        # Revenue by category
        rev_by_cat = (
            filtered_df
            .group_by("category")
            .agg(pl.col("revenue_value").sum())
            .sort("revenue_value", descending=True)
            .to_pandas()
        )

        if not rev_by_cat.empty:
            fig = px.bar(
                rev_by_cat,
                x="revenue_value",
                y="category",
                orientation="h",
                title="Tổng doanh thu theo ngành hàng",
                color_discrete_sequence=["#0f766e"]
            )
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu cho bộ lọc này")

    with ch2:
        # Profit by region
        profit_by_reg = (
            filtered_df
            .group_by("region")
            .agg(pl.col("profit_value").sum())
            .sort("profit_value", descending=True)
            .to_pandas()
        )

        if not profit_by_reg.empty:
            fig = px.bar(
                profit_by_reg,
                x="profit_value",
                y="region",
                orientation="h",
                title="Tổng lợi nhuận theo khu vực",
                color_discrete_sequence=["#2563eb"]
            )
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu cho bộ lọc này")

    ch3, ch4 = st.columns(2)

    with ch3:
        # Discount effectiveness
        discount_eff = (
            filtered_df
            .group_by("discount")
            .agg(
                avg_roi=pl.col("roi").mean(),
                count=pl.col("roi").count()
            )
            .sort("discount")
            .to_pandas()
        )

        if not discount_eff.empty:
            discount_eff["discount_pct"] = (discount_eff["discount"] * 100).round(0)
            fig = px.line(
                discount_eff,
                x="discount_pct",
                y="avg_roi",
                markers=True,
                title="ROI trung bình theo mức chiết khấu",
            )
            fig.update_traces(line=dict(width=3, color="#7c3aed"))
            fig.update_layout(xaxis_title="Chiết khấu (%)", yaxis_title="ROI trung bình")
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu cho bộ lọc này")

    with ch4:
        # Segment performance
        seg_perf = (
            filtered_df
            .group_by("rfm_segment")
            .agg(
                total_revenue=pl.col("revenue_value").sum(),
                total_profit=pl.col("profit_value").sum(),
            )
            .sort("total_revenue", descending=True)
            .head(5)
            .to_pandas()
        )

        if not seg_perf.empty:
            fig = px.bar(
                seg_perf,
                x="total_profit",
                y="rfm_segment",
                orientation="h",
                title="Top phân khúc theo lợi nhuận",
                color_discrete_sequence=["#dc2626"]
            )
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Không có dữ liệu cho bộ lọc này")

    st.warning(f"Chiến lược đề xuất: {strategy_decision}")
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    _decision_block(
        insight=f"ROI lúc dữ liệu hiện tại là {current_roi:.2f}; hiệu quả thay đổi rõ theo mức chiết khấu.",
        recommendation=strategy_decision,
        impact=f"ROI dự kiến tăng +{roi_uplift*100:.0f}%, lợi nhuận dự kiến tăng +{profit_uplift*100:.0f}%.",
        level="warning",
    )

    with st.expander("Xem bảng dữ liệu đã lọc", expanded=False):
        st.dataframe(filtered_df.to_pandas(), use_container_width=True, height=400)
elif selected == "Inventory":
    st.subheader("Hành động tồn kho")
    reorder_g, inv_applied_1 = _filter_by_master_scope(reorder_df, master_scope_pd)
    overstock_g, inv_applied_2 = _filter_by_master_scope(overstock_df, master_scope_pd)
    inv_applied = sorted(set(inv_applied_1 + inv_applied_2))
    st.caption(f"{global_filter_summary} | Áp dụng được: {', '.join(inv_applied) if inv_applied else 'không có cột phù hợp'}")

    inv_filter_source = pd.concat([reorder_g, overstock_g], ignore_index=True, sort=False)
    inv_fields = [
        ("category", "Ngành hàng"),
    ]
    inv_filtered_source, inv_sel, inv_filter_summary = _filter_form_for_df(
        inv_filter_source,
        form_key="inventory_filters",
        fields=inv_fields,
    )
    st.caption(inv_filter_summary)

    reorder_f = _apply_pandas_filters(reorder_g, inv_sel)
    overstock_f = _apply_pandas_filters(overstock_g, inv_sel)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(custom_metric("SKU cần đặt thêm", f"{len(reorder_f):,}"), unsafe_allow_html=True)
    with m2:
        total_reorder = float(reorder_f["recommended_order_qty"].sum()) if (not reorder_f.empty and "recommended_order_qty" in reorder_f.columns) else 0.0
        st.markdown(custom_metric("Tổng SL cần đặt", _fmt_num(total_reorder, ":,.0f")), unsafe_allow_html=True)
    with m3:
        st.markdown(custom_metric("SKU tồn kho dư", f"{len(overstock_f):,}"), unsafe_allow_html=True)
    with m4:
        total_overstock_cost = float(overstock_f["overstock_cost"].sum()) if (not overstock_f.empty and "overstock_cost" in overstock_f.columns) else 0.0
        st.markdown(custom_metric("Tổng chi phí tồn", _fmt_num(total_overstock_cost, ":,.0f")), unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        if reorder_f.empty:
            st.info("Không có sản phẩm cần đặt thêm.")
        else:
            reorder_chart = _styled_bar(
                reorder_f.sort_values("recommended_order_qty", ascending=True),
                x="recommended_order_qty",
                y="product_name",
                color="category",
                orientation="h",
                title="Số lượng cần đặt thêm (đề xuất)",
                text_auto=True,
                height=300,
            )
            reorder_chart.update_layout(xaxis_title="Số lượng (đơn vị)", yaxis_title="")
            st.plotly_chart(reorder_chart, use_container_width=True)

    with right:
        if overstock_f.empty:
            st.info("Không có sản phẩm tồn kho dư.")
        else:
            overstock_chart = _styled_bar(
                overstock_f.sort_values("overstock_cost", ascending=True),
                x="overstock_cost",
                y="product_name",
                color="category",
                orientation="h",
                title="Chi phí tồn kho dư theo sản phẩm",
                text_auto=True,
                height=300,
            )
            overstock_chart.update_layout(xaxis_title="Chi phí tồn kho dư", yaxis_title="")
            st.plotly_chart(overstock_chart, use_container_width=True)

    if not reorder_f.empty:
        reorder_action_df = reorder_f.copy()
        if "recommended_order_qty" in reorder_action_df.columns:
            reorder_action_df["priority"] = (
                reorder_action_df["recommended_order_qty"]
                .apply(lambda x: "HIGH" if x > reorder_action_df["recommended_order_qty"].median() else "MEDIUM")
            )
        reorder_action_df["action"] = "REORDER"
    else:
        reorder_action_df = pd.DataFrame(columns=["product_name", "priority", "action"])

    inv_scope = inv_filter_summary.replace("Phạm vi: ", "")
    if reorder_f.empty and overstock_f.empty:
        inv_insight = f"Không có dữ liệu tồn kho theo phạm vi đang chọn ({inv_scope})."
        inv_reco = "Hãy nới bộ lọc hoặc kiểm tra lại dữ liệu đầu vào."
        inv_impact = "Không phát sinh hành động đặt thêm/giảm tồn trong phạm vi này."
        inv_level = "info"
    else:
        inv_insight = (
            f"Trong phạm vi ({inv_scope}), có {len(reorder_f):,} SKU cần đặt thêm và {len(overstock_f):,} SKU tồn kho dư."
        )
        inv_reco = "Ưu tiên xử lý nhóm HIGH trước, sau đó giảm dần các mặt hàng tồn kho dư."
        inv_impact = "Tăng khả năng đáp ứng đơn hàng và giảm chi phí lưu kho."
        inv_level = "warning" if len(reorder_f) else "info"

    _decision_block(
        insight=inv_insight,
        recommendation=inv_reco,
        impact=inv_impact,
        level=inv_level,
    )

    with st.expander("Bảng tồn kho", expanded=False):
        st.dataframe(reorder_f, use_container_width=True)
        st.dataframe(overstock_f, use_container_width=True)
        if not reorder_action_df.empty:
            st.dataframe(
                reorder_action_df[[c for c in ["product_name", "category", "recommended_order_qty", "priority", "action"] if c in reorder_action_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

elif selected == "Promotion":
    st.subheader("Cơ hội khuyến mãi")
    st.caption(global_filter_summary)
    if promo_scope is None or promo_scope.empty:
        st.info("Không đủ dữ liệu trong phạm vi lọc để phân tích chiết khấu/khuyến mãi.")
    else:
        promo_fields = [
            ("recommendation", "Khuyến nghị"),
        ]
        promo_f, promo_sel, promo_filter_summary = _filter_form_for_df(
            promo_scope,
            form_key="promotion_filters",
            fields=promo_fields,
        )
        st.caption(promo_filter_summary)

        pm1, pm2, pm3, pm4 = st.columns(4)
        with pm1:
            st.markdown(custom_metric("Mức chiết khấu", f"{len(promo_f):,}"), unsafe_allow_html=True)
        with pm2:
            roi_avg = float(promo_f["avg_roi"].mean()) if (not promo_f.empty and "avg_roi" in promo_f.columns) else 0.0
            st.markdown(custom_metric("ROI TB", _fmt_num(roi_avg, ":.2f")), unsafe_allow_html=True)
        with pm3:
            roi_max = float(promo_f["avg_roi"].max()) if (not promo_f.empty and "avg_roi" in promo_f.columns) else 0.0
            st.markdown(custom_metric("ROI TB cao nhất", _fmt_num(roi_max, ":.2f")), unsafe_allow_html=True)
        with pm4:
            profit_sum = float(promo_f["total_profit"].sum()) if (not promo_f.empty and "total_profit" in promo_f.columns) else 0.0
            st.markdown(custom_metric("Tổng lợi nhuận", _fmt_num(profit_sum, ":,.0f")), unsafe_allow_html=True)

        lcol, rcol = st.columns([1.2, 1])

        with lcol:
            fig = px.line(
                promo_f,
                x="discount_pct",
                y="avg_roi",
                markers=True,
                title="ROI trung bình theo mức chiết khấu",
                color="recommendation",
                color_discrete_sequence=["#0f766e", "#d97706", "#dc2626"],
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(xaxis_title="Chiết khấu (%)", yaxis_title="ROI trung bình")
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        with rcol:
            rank_chart = _styled_bar(
                promo_f.sort_values("avg_roi", ascending=True),
                x="avg_roi",
                y="discount_pct",
                color="recommendation",
                orientation="h",
                title="Top mức chiết khấu theo ROI TB",
                text_auto=".2f",
                height=300,
            )
            rank_chart.update_layout(xaxis_title="ROI trung bình", yaxis_title="Chiết khấu (%)")
            st.plotly_chart(rank_chart, use_container_width=True)

        best = promo_f.sort_values("avg_roi", ascending=False).head(1)
        if not best.empty:
            best_d = float(best["discount_pct"].iloc[0])
            best_roi = float(best["avg_roi"].iloc[0])
            _decision_block(
                insight=f"Trong phạm vi lọc hiện tại, ROI trung bình thay đổi theo mức chiết khấu; mức hiệu quả nhất đang quanh {best_d:.0f}%.",
                recommendation="Tăng phân bổ vào nhóm chiết khấu có ROI cao; giảm nhóm chiết khấu ROI thấp.",
                impact=f"ROI TB tốt nhất theo dữ liệu lọc: {best_roi:.2f}.",
                level="info",
            )

        with st.expander("Bảng chiết khấu & ROI", expanded=False):
            st.dataframe(promo_f, use_container_width=True, hide_index=True)

elif selected == "Acquisition":
    st.subheader("Ưu tiên thu hút khách hàng")
    st.caption(global_filter_summary)
    if acq_scope is None or acq_scope.empty:
        st.info("Không đủ dữ liệu trong phạm vi lọc để phân tích thu hút theo khu vực/phân khúc.")
    else:
        acq_fields = [
            ("region", "Khu vực"),
            ("rfm_segment", "Phân khúc"),
        ]
        acq_f, acq_sel, acq_filter_summary = _filter_form_for_df(
            acq_scope,
            form_key="acquisition_filters",
            fields=acq_fields,
        )
        st.caption(acq_filter_summary)

        am1, am2, am3, am4 = st.columns(4)
        with am1:
            st.markdown(custom_metric("Tổ hợp", f"{len(acq_f):,}"), unsafe_allow_html=True)
        with am2:
            acquire_avg = float(acq_f["acquire"].mean()) if (not acq_f.empty and "acquire" in acq_f.columns) else 0.0
            st.markdown(custom_metric("Acquire TB", _fmt_num(acquire_avg, ":,.0f")), unsafe_allow_html=True)
        with am3:
            roi_avg = float(acq_f["roi"].mean()) if (not acq_f.empty and "roi" in acq_f.columns) else 0.0
            st.markdown(custom_metric("ROI TB", _fmt_num(roi_avg, ":.2f")), unsafe_allow_html=True)
        with am4:
            rows_sum = int(acq_f["rows"].sum()) if (not acq_f.empty and "rows" in acq_f.columns) else 0
            st.markdown(custom_metric("Số dòng dữ liệu", f"{rows_sum:,}"), unsafe_allow_html=True)

        lcol, rcol = st.columns(2)
        with lcol:
            top = acq_f.sort_values("acquire", ascending=False).head(12)
            if top.empty:
                st.info("Không có dữ liệu sau khi lọc.")
            else:
                fig = _styled_bar(
                    top.sort_values("acquire", ascending=True),
                    x="acquire",
                    y="rfm_segment",
                    color="region",
                    orientation="h",
                    title="Giá trị thu hút trung bình theo phân khúc",
                    text_auto=".2s",
                    height=300,
                )
                fig.update_layout(xaxis_title="Giá trị thu hút trung bình", yaxis_title="Phân khúc")
                st.plotly_chart(fig, use_container_width=True)

        with rcol:
            fig = px.scatter(
                acq_f,
                x="acquire",
                y="roi",
                size="rows",
                color="region",
                title="Giá trị thu hút và ROI theo khu vực/phân khúc",
                color_discrete_sequence=["#0f766e", "#2563eb", "#d97706", "#dc2626", "#7c3aed"],
            )
            fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
            fig.update_layout(xaxis_title="Giá trị thu hút trung bình", yaxis_title="ROI trung bình")
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        best = acq_f.sort_values(["acquire", "roi"], ascending=False).head(1)
        if not best.empty:
            br = best.iloc[0]
            _decision_block(
                insight=f"Tổ hợp nổi bật trong phạm vi lọc: Khu vực {br['region']} – Phân khúc {br['rfm_segment']} (acquire≈{br['acquire']:,.0f}, ROI≈{br['roi']:.2f}).",
                recommendation="Ưu tiên ngân sách/nguồn lực cho các tổ hợp khu vực–phân khúc có acquire cao và ROI tốt; giảm ưu tiên nhóm còn lại.",
                impact="Tối ưu hiệu quả thu hút theo đúng bối cảnh bộ lọc Strategy.",
                level="info",
            )

        with st.expander("Bảng thu hút theo khu vực/phân khúc", expanded=False):
            st.dataframe(acq_f, use_container_width=True, hide_index=True)

else:
    st.subheader("Mô phỏng giữ chân")
    st.caption(global_filter_summary)
    if ret_scope is None or ret_scope.empty:
        st.info("Không đủ dữ liệu trong phạm vi lọc để phân tích giữ chân theo phân khúc/chiết khấu.")
        scope_txt = global_filter_summary
        optimal_discount = 0
        net_benefit_at_opt = 0
        expected_saved_at_opt = 0
        ret_f = pd.DataFrame()
    else:
        ret_fields = [
            ("rfm_segment", "Phân khúc"),
        ]
        ret_f, ret_sel, ret_filter_summary = _filter_form_for_df(
            ret_scope,
            form_key="retention_filters",
            fields=ret_fields,
        )
        st.caption(ret_filter_summary)
        scope_txt = ret_filter_summary.replace("Phạm vi: ", "")

        # Derive "optimal" discount as the discount with max retain (within filter)
        best = ret_f.sort_values(["retain", "roi"], ascending=False).head(1)
        if not best.empty:
            optimal_discount = float(best["discount"].iloc[0]) * 100
            net_benefit_at_opt = float(best["retain"].iloc[0])
            expected_saved_at_opt = float(best["retain"].iloc[0])
        else:
            optimal_discount = 0
            net_benefit_at_opt = 0
            expected_saved_at_opt = 0

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(custom_metric("Chiết khấu tối ưu (%)", _fmt_num(optimal_discount, ":.0f")), unsafe_allow_html=True)
    with r2:
        st.markdown(custom_metric("Doanh thu dự kiến giữ lại", _fmt_num(expected_saved_at_opt, ":,.0f")), unsafe_allow_html=True)
    with r3:
        st.markdown(custom_metric("Lợi ích ròng", _fmt_num(net_benefit_at_opt, ":,.0f")), unsafe_allow_html=True)

    if ret_f is not None and not ret_f.empty:
        ret_f = ret_f.copy()
        if "discount" in ret_f.columns:
            ret_f["discount_pct"] = (ret_f["discount"] * 100).round(0)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(
                ret_f.sort_values("discount"),
                x="discount_pct",
                y="retain",
                markers=True,
                title="Giá trị giữ chân theo mức chiết khấu",
                color="rfm_segment" if "rfm_segment" in ret_f.columns else None,
                color_discrete_sequence=["#0f766e", "#2563eb", "#d97706", "#dc2626", "#7c3aed"],
            )
            fig.update_traces(line=dict(width=3))
            fig.update_layout(xaxis_title="Chiết khấu (%)", yaxis_title="Giá trị giữ chân (TB)")
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.scatter(
                ret_f,
                x="retain",
                y="roi",
                size="rows",
                color="rfm_segment" if "rfm_segment" in ret_f.columns else None,
                title="Giữ chân vs ROI theo phân khúc/chiết khấu",
                color_discrete_sequence=["#0f766e", "#2563eb", "#d97706", "#dc2626", "#7c3aed"],
            )
            fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
            fig.update_layout(xaxis_title="Giá trị giữ chân (TB)", yaxis_title="ROI TB")
            fig = apply_chart_style(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Bảng giữ chân theo phân khúc/chiết khấu", expanded=False):
            st.dataframe(ret_f, use_container_width=True, hide_index=True)

    _decision_block(
        insight=f"Trong phạm vi ({scope_txt}), giá trị giữ chân thay đổi theo phân khúc và mức chiết khấu.",
        recommendation=f"Ưu tiên mức chiết khấu khoảng {optimal_discount:.0f}% cho nhóm phân khúc đã chọn, và theo dõi định kỳ theo tháng/khu vực trong bộ lọc Strategy.",
        impact=f"Giá trị giữ chân TB tốt nhất (trong phạm vi lọc): {net_benefit_at_opt:,.0f}.",
        level="info",
    )

st.subheader("Trung tâm Hành động & Điều hành")
st.dataframe(action_df, use_container_width=True, hide_index=True)




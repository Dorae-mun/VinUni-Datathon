import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import get_cached_data, get_order_items_enriched


def format_compact_money(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}bn"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:,.0f}"


def format_full_money(value: float) -> str:
    return f"{value:,.2f}"


def metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-title">{title}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_card(title: str, value: float) -> None:
    gauge_value = max(0.0, min(value, 100.0))
    gauge_angle = gauge_value * 1.8
    st.markdown(
        f"""
        <div class="gauge-card">
            <div class="gauge-card-title">{title}</div>
            <div class="gauge-wrap">
                <div class="gauge-arc" style="--gauge-angle: {gauge_angle}deg;">
                    <div class="gauge-hole"></div>
                </div>
                <div class="gauge-value">{gauge_value:.2f}%</div>
                <div class="gauge-min">0.00%</div>
                <div class="gauge-max">100.00%</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: #ffffff !important;
    }
    .main {
        background: #ffffff;
    }
    .block-container {
        padding-top: 0.55rem;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
        max-width: 1340px;
    }
    .report-shell {
        border: 1px dotted #595959;
        padding: 0.7rem 0.9rem 0.9rem;
        background: #fff;
    }
    .top-header {
        border-bottom: 1px solid #d9d9d9;
        background: #ffffff;
        margin-bottom: 6px;
        min-height: 78px;
    }
    .brand-box {
        background: #114f86;
        color: white;
        font-size: 1.4rem;
        font-style: italic;
        font-weight: 800;
        padding: 15px 20px;
        clip-path: polygon(0 0, 100% 0, 88% 100%, 0 100%);
        text-align: left;
        line-height: 1;
        margin-top: 8px;
    }
    .report-title {
        color: #14235c;
        font-size: 1.08rem;
        font-weight: 800;
        margin-top: 0.9rem;
        line-height: 1.2;
    }
    .report-date {
        color: #d97706;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }
    .filter-strip {
        border-left: 2px solid #114f86;
        padding-left: 12px;
    }
    .filter-caption {
        color: #14235c;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .kpi-card {
        background: white;
        border: 1px solid #dedede;
        border-radius: 4px;
        padding: 14px 16px;
        min-height: 138px;
    }
    .kpi-value {
        color: #14235c;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .kpi-title {
        color: #14235c;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .kpi-subtitle {
        color: #666666;
        background: #f3f3f3;
        padding: 10px 12px;
        font-size: 0.9rem;
        border-top: 1px solid #e7e7e7;
    }
    .gauge-card {
        background: white;
        border: 1px solid #dedede;
        border-radius: 4px;
        padding: 8px 10px 6px;
        min-height: 138px;
    }
    .gauge-card-title {
        color: #14235c;
        font-size: 0.95rem;
        font-weight: 700;
        text-align: center;
        margin-top: 0.1rem;
        margin-bottom: 0.25rem;
    }
    .gauge-wrap {
        position: relative;
        height: 92px;
        overflow: hidden;
    }
    .gauge-arc {
        width: 150px;
        height: 150px;
        margin: 2px auto 0;
        border-radius: 50%;
        background: conic-gradient(
            from 180deg,
            #0f4c81 0deg,
            #0f4c81 var(--gauge-angle),
            #efefef var(--gauge-angle),
            #efefef 180deg,
            transparent 180deg,
            transparent 360deg
        );
        position: relative;
    }
    .gauge-hole {
        position: absolute;
        inset: 24px;
        background: white;
        border-radius: 50%;
    }
    .gauge-value {
        position: absolute;
        left: 50%;
        bottom: 14px;
        transform: translateX(-50%);
        color: #6b7280;
        font-size: 0.9rem;
    }
    .gauge-min, .gauge-max {
        position: absolute;
        bottom: 0;
        color: #666;
        font-size: 0.78rem;
    }
    .gauge-min {
        left: 28px;
    }
    .gauge-max {
        right: 24px;
    }
    div[data-baseweb="select"] > div {
        min-height: 34px;
        background: #114f86;
        border-radius: 0;
        border: 1px solid #2a5d90;
    }
    div[data-baseweb="select"] {
        margin-bottom: 0.1rem;
    }
    div[data-baseweb="select"] * {
        color: white !important;
    }
    .stRadio {
        margin-top: 0.35rem;
        margin-bottom: 0.5rem;
    }
    .stRadio > div {
        gap: 10px;
    }
    .stRadio label {
        width: 100%;
    }
    .stRadio [role="radiogroup"] {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
    }
    .stRadio [data-baseweb="radio"] {
        margin: 0;
        border: 1px solid #d5d5d5;
        justify-content: center;
        background: #ffffff;
        min-height: 26px;
        padding: 0.05rem 0.25rem;
    }
    .stRadio [data-baseweb="radio"] > div:first-child {
        display: none;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] {
        background: #636363;
        border-color: #636363;
    }
    .stRadio [data-baseweb="radio"] div[role="presentation"] + div {
        color: #14235c;
        font-size: 0.86rem;
        font-weight: 700;
    }
    .stRadio [data-baseweb="radio"][aria-checked="true"] div[role="presentation"] + div {
        color: white;
    }
    div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
        border: 1px solid #dedede;
        border-radius: 4px;
        padding: 6px 6px 2px;
        background: white;
    }
    div[data-testid="stPlotlyChart"] {
        box-shadow: none;
    }
    .panel-title {
        color: #14235c;
        font-size: 0.98rem;
        font-weight: 700;
        margin: 0.35rem 0 0.45rem 0;
    }
    div[data-testid="column"] {
        gap: 0.55rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="report-shell">', unsafe_allow_html=True)

data = get_cached_data()
items = get_order_items_enriched(data).merge(
    data["customers"][["customer_id", "acquisition_channel"]],
    on="customer_id",
    how="left",
)
returns = data["returns"].copy()

items["order_date"] = pd.to_datetime(items["order_date"])
items["gross_sales"] = items["quantity"] * items["unit_price"]
items["year"] = items["order_date"].dt.year

min_date = items["order_date"].min().date()
max_date = items["order_date"].max().date()

header_col1, header_col2, header_col3 = st.columns([1.45, 3.3, 6.25], vertical_alignment="center")
with header_col1:
    st.markdown('<div class="top-header"><div class="brand-box">KTEAM</div></div>', unsafe_allow_html=True)
with header_col2:
    st.markdown(
        f"""
        <div class="top-header" style="padding: 10px 0 10px 0;">
            <div class="report-title">Revenue & Business Intelligence Report</div>
            <div class="report-date">From {min_date:%d/%m/%Y} - {max_date:%d/%m/%Y}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col3:
    st.markdown('<div class="top-header filter-strip">', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown('<div class="filter-caption">Time Filter</div>', unsafe_allow_html=True)
        time_filter = st.selectbox(
            "Time Filter",
            ["All"] + [str(year) for year in sorted(items["year"].unique())],
            label_visibility="collapsed",
        )
    with f2:
        st.markdown('<div class="filter-caption">Customer Segmentation</div>', unsafe_allow_html=True)
        segment_filter = st.selectbox(
            "Customer Segmentation",
            ["All"] + sorted(items["segment"].dropna().unique().tolist()),
            label_visibility="collapsed",
        )
    with f3:
        st.markdown('<div class="filter-caption">Product Category</div>', unsafe_allow_html=True)
        category_filter = st.selectbox(
            "Product Category",
            ["All"] + sorted(items["category"].dropna().unique().tolist()),
            label_visibility="collapsed",
        )
    with f4:
        st.markdown('<div class="filter-caption">Channel</div>', unsafe_allow_html=True)
        channel_filter = st.selectbox(
            "Channel",
            ["All"] + sorted(items["acquisition_channel"].dropna().unique().tolist()),
            label_visibility="collapsed",
        )
    st.markdown("</div>", unsafe_allow_html=True)

filtered_items = items.copy()
if time_filter != "All":
    filtered_items = filtered_items[filtered_items["year"] == int(time_filter)]
if segment_filter != "All":
    filtered_items = filtered_items[filtered_items["segment"] == segment_filter]
if category_filter != "All":
    filtered_items = filtered_items[filtered_items["category"] == category_filter]
if channel_filter != "All":
    filtered_items = filtered_items[filtered_items["acquisition_channel"] == channel_filter]

if filtered_items.empty:
    st.warning("No data matches the selected filters.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

filtered_items["line_profit"] = filtered_items["line_profit"].fillna(0)
filtered_returns = returns[returns["order_id"].isin(filtered_items["order_id"].unique())].copy()

total_revenue = filtered_items["line_revenue"].sum()
total_net_gross = filtered_items["gross_sales"].sum() - filtered_items["discount_amount"].sum()
gross_margin_pct = filtered_items["line_profit"].sum() / total_revenue * 100 if total_revenue else 0
leak_rate_pct = filtered_items["discount_amount"].sum() / filtered_items["gross_sales"].sum() * 100 if filtered_items["gross_sales"].sum() else 0
avg_order_revenue = total_revenue / filtered_items["order_id"].nunique()

prev_items = items.copy()
if time_filter != "All":
    prev_items = prev_items[prev_items["year"] == int(time_filter) - 1]
if segment_filter != "All":
    prev_items = prev_items[prev_items["segment"] == segment_filter]
if category_filter != "All":
    prev_items = prev_items[prev_items["category"] == category_filter]
if channel_filter != "All":
    prev_items = prev_items[prev_items["acquisition_channel"] == channel_filter]

prev_revenue = prev_items["line_revenue"].sum() if not prev_items.empty else 0
prev_net_gross = (
    prev_items["gross_sales"].sum() - prev_items["discount_amount"].sum()
    if not prev_items.empty
    else 0
)
prev_aov = (
    prev_items["line_revenue"].sum() / prev_items["order_id"].nunique()
    if not prev_items.empty and prev_items["order_id"].nunique() > 0
    else 0
)

yearly = (
    filtered_items.groupby("year", as_index=False)
    .agg(
        total_revenue=("line_revenue", "sum"),
        total_cogs=("line_cogs", "sum"),
        total_discount=("discount_amount", "sum"),
        total_profit=("line_profit", "sum"),
        orders=("order_id", "nunique"),
    )
    .sort_values("year")
)
yearly["gross_margin_pct"] = yearly["total_profit"] / yearly["total_revenue"] * 100
yearly["gross_margin_ly"] = yearly["gross_margin_pct"].shift(1)
yearly["yoy_growth"] = yearly["total_revenue"].pct_change() * 100
yearly["net_gross"] = yearly["total_revenue"] - yearly["total_discount"]

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    gauge_card("Gross Margin %", gross_margin_pct)
with k2:
    gauge_card("Leak Rate %", leak_rate_pct)
with k3:
    metric_card("Total Revenue", format_compact_money(total_revenue), f"Total Revenue LY {format_compact_money(prev_revenue)}")
with k4:
    metric_card("Net Gross", format_compact_money(total_net_gross), f"Net Gross LY {format_compact_money(prev_net_gross)}")
with k5:
    metric_card("Avg Order Revenue", format_compact_money(avg_order_revenue), f"AOV LY {format_compact_money(prev_aov)}")

selected_view = st.radio(
    "Section",
    ["Overview", "Product Category", "Geography", "Revenue Leak"],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_view == "Overview":
    left_col, mid_col, right_col = st.columns([4.2, 4.5, 3.0])

    with left_col:
        st.markdown('<div class="panel-title">Total Revenue, Total COGS by Time</div>', unsafe_allow_html=True)
        rev_cogs_fig = go.Figure()
        rev_cogs_fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["total_revenue"],
                mode="lines",
                name="Total Revenue",
                stackgroup="one",
                line=dict(color="#0f4c81", width=2),
            )
        )
        rev_cogs_fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["total_cogs"],
                mode="lines",
                name="Total COGS",
                stackgroup="two",
                line=dict(color="#4da3ff", width=2),
            )
        )
        rev_cogs_fig.update_layout(
            template="plotly_white",
            height=230,
            margin=dict(l=0, r=0, t=6, b=0),
            legend=dict(orientation="h", y=1.12, x=0),
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(rev_cogs_fig, use_container_width=True)

        st.markdown('<div class="panel-title">% YoY Growth Rate by Year</div>', unsafe_allow_html=True)
        yoy_colors = ["#18c83e" if x >= 0 else "#d64550" for x in yearly["yoy_growth"].fillna(0)]
        yoy_fig = go.Figure(
            go.Bar(
                x=yearly["year"],
                y=yearly["yoy_growth"],
                marker_color=yoy_colors,
                text=[f"{x:.0f}%" if pd.notna(x) else "" for x in yearly["yoy_growth"]],
                textposition="outside",
            )
        )
        yoy_fig.update_layout(
            template="plotly_white",
            height=195,
            margin=dict(l=0, r=0, t=6, b=0),
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(yoy_fig, use_container_width=True)

    with mid_col:
        st.markdown('<div class="panel-title">% Gross Margin Same Period Last Year</div>', unsafe_allow_html=True)
        margin_fig = go.Figure()
        margin_fig.add_trace(
            go.Scatter(
                x=yearly["year"],
                y=yearly["gross_margin_pct"],
                mode="lines+markers",
                name="Gross Margin %",
                line=dict(color="#0f4c81", width=3),
                fill="tozeroy",
            )
        )
        if yearly["gross_margin_ly"].notna().any():
            margin_fig.add_trace(
                go.Scatter(
                    x=yearly["year"],
                    y=yearly["gross_margin_ly"],
                    mode="lines+markers",
                    name="Gross Margin % LY",
                    line=dict(color="#4da3ff", width=3),
                    fill="tozeroy",
                )
            )
        if (yearly["year"] == 2017).any():
            margin_fig.add_vline(x=2017, line_dash="dot", line_color="#2f80ed", annotation_text="FDI Peak")
        if (yearly["year"] == 2020).any():
            margin_fig.add_vline(x=2020, line_dash="dot", line_color="#eb5757", annotation_text="Covid 19")
        margin_fig.update_layout(
            template="plotly_white",
            height=445,
            margin=dict(l=0, r=0, t=6, b=0),
            legend=dict(orientation="h", y=1.1, x=0),
            xaxis_title=None,
            yaxis_title=None,
        )
        st.plotly_chart(margin_fig, use_container_width=True)

    with right_col:
        st.markdown('<div class="panel-title">Details Table</div>', unsafe_allow_html=True)
        detail_df = yearly[["year", "total_revenue", "net_gross"]].copy()
        detail_df.columns = ["Year", "Total Revenue", "Net Gross"]
        detail_df["Total Revenue"] = detail_df["Total Revenue"].map(format_full_money)
        detail_df["Net Gross"] = detail_df["Net Gross"].map(format_full_money)
        st.dataframe(detail_df, use_container_width=True, height=445, hide_index=True)

elif selected_view == "Product Category":
    category_df = (
        filtered_items.groupby("category", as_index=False)
        .agg(total_revenue=("line_revenue", "sum"), net_gross=("line_profit", "sum"))
        .sort_values("total_revenue", ascending=False)
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(x=category_df["category"], y=category_df["total_revenue"], name="Revenue", marker_color="#0f4c81"))
    fig.add_trace(go.Bar(x=category_df["category"], y=category_df["net_gross"], name="Net Gross", marker_color="#4da3ff"))
    fig.update_layout(template="plotly_white", barmode="group", height=420, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

elif selected_view == "Geography":
    geo_df = (
        filtered_items.groupby("region", as_index=False)
        .agg(total_revenue=("line_revenue", "sum"), orders=("order_id", "nunique"))
        .sort_values("total_revenue", ascending=False)
    )
    fig = go.Figure(go.Bar(x=geo_df["region"], y=geo_df["total_revenue"], marker_color="#0f4c81"))
    fig.update_layout(template="plotly_white", height=420, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

else:
    leak_df = (
        filtered_items.groupby("year", as_index=False)
        .agg(gross_sales=("gross_sales", "sum"), discount_amount=("discount_amount", "sum"))
        .sort_values("year")
    )
    leak_df["leak_rate"] = leak_df["discount_amount"] / leak_df["gross_sales"] * 100
    fig = go.Figure(go.Bar(x=leak_df["year"], y=leak_df["leak_rate"], marker_color="#d64550"))
    fig.update_layout(template="plotly_white", height=420, xaxis_title=None, yaxis_title="Leak Rate %")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

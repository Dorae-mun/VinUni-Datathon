import plotly.express as px
import plotly.graph_objects as go


def revenue_chart(df):
    fig = px.line(
        df,
        x="Date",
        y="Revenue",
        title="Revenue Over Time",
    )
    fig.update_traces(line=dict(width=3, color="#0f766e"))
    fig.update_layout(template="plotly_white")
    return fig


def category_sales_chart(df):
    value_col = "line_revenue" if "line_revenue" in df.columns else "Revenue"
    fig = px.bar(
        df.groupby("category", dropna=False)[value_col].sum().reset_index(),
        x="category",
        y=value_col,
        title="Revenue by Category",
    )
    fig.update_layout(template="plotly_white")
    return fig


def forecast_chart(actual, forecast):
    fig = go.Figure()
    fig.add_scatter(x=actual["Date"], y=actual["Revenue"], name="Actual")
    fig.add_scatter(x=forecast["Date"], y=forecast["Forecast"], name="Forecast")
    fig.update_layout(template="plotly_white", title="Actual vs Forecast")
    return fig

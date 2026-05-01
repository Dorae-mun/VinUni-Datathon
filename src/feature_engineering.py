from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TET_DATES = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}

LAG_OFFSETS = [1, 7, 14, 28, 30, 56, 91, 365]
ROLLING_WINDOWS = [7, 14, 28, 56, 90]

REVENUE_LAG_FEATURES = [f"revenue_lag_{offset}" for offset in LAG_OFFSETS]
COGS_LAG_FEATURES = [f"cogs_lag_{offset}" for offset in LAG_OFFSETS]
REVENUE_ROLLING_FEATURES = [
    f"revenue_rolling_{stat}_{window}"
    for window in ROLLING_WINDOWS
    for stat in ("mean", "std")
]
COGS_ROLLING_FEATURES = [
    f"cogs_rolling_{stat}_{window}"
    for window in ROLLING_WINDOWS
    for stat in ("mean", "std")
]
PRIOR_FEATURES = [
    "revenue_dow_avg_prior",
    "revenue_month_avg_prior",
    "cogs_dow_avg_prior",
    "cogs_month_avg_prior",
]
PROFILE_FEATURES = [
    "traffic_sessions_profile",
    "traffic_unique_visitors_profile",
    "traffic_page_views_profile",
    "traffic_bounce_rate_profile",
    "traffic_avg_session_duration_profile",
    "traffic_pages_per_session_profile",
    "inventory_stock_on_hand_profile",
    "inventory_units_received_profile",
    "inventory_units_sold_profile",
    "inventory_stockout_days_profile",
    "inventory_days_of_supply_profile",
    "inventory_fill_rate_profile",
    "inventory_sell_through_rate_profile",
    "inventory_stockout_flag_rate_profile",
    "inventory_overstock_flag_rate_profile",
    "inventory_reorder_flag_rate_profile",
]
STATIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_payday_window",
    "is_double_day",
    "days_since_start",
    "days_to_tet",
    "is_pre_tet_rush",
    "is_tet_holiday",
    "is_promo",
    "max_disc",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
] + PROFILE_FEATURES

ALL_FEATURES = (
    STATIC_FEATURES
    + REVENUE_LAG_FEATURES
    + COGS_LAG_FEATURES
    + REVENUE_ROLLING_FEATURES
    + COGS_ROLLING_FEATURES
    + PRIOR_FEATURES
)


@dataclass(frozen=True)
class ForecastFeatureSpec:
    static_features: Sequence[str] = tuple(STATIC_FEATURES)
    all_features: Sequence[str] = tuple(ALL_FEATURES)


def get_feature_spec() -> ForecastFeatureSpec:
    return ForecastFeatureSpec()


def ensure_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out.sort_values(date_col).reset_index(drop=True)


def make_daily_promo_features(
    promotions_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    promo_range = pd.date_range(start=start_date, end=end_date, freq="D")
    daily = pd.DataFrame({"Date": promo_range, "is_promo": 0, "max_disc": 0.0})
    if promotions_df is None or promotions_df.empty:
        return daily

    promos = promotions_df.copy()
    promos["start_date"] = pd.to_datetime(promos["start_date"])
    promos["end_date"] = pd.to_datetime(promos["end_date"])

    for _, row in promos.iterrows():
        mask = daily["Date"].between(row["start_date"], row["end_date"])
        daily.loc[mask, "is_promo"] = 1
        if row.get("promo_type") == "percentage":
            daily.loc[mask, "max_disc"] = np.maximum(
                daily.loc[mask, "max_disc"],
                float(row.get("discount_value", 0.0)),
            )
    return daily


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime(df)
    out["day_of_week"] = out["Date"].dt.dayofweek
    out["day_of_month"] = out["Date"].dt.day
    out["day_of_year"] = out["Date"].dt.dayofyear
    out["week_of_year"] = out["Date"].dt.isocalendar().week.astype(int)
    out["month"] = out["Date"].dt.month
    out["quarter"] = out["Date"].dt.quarter
    out["year"] = out["Date"].dt.year
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["Date"].dt.is_month_end.astype(int)
    out["is_payday_window"] = (
        (out["day_of_month"] >= 25) | (out["day_of_month"] <= 5)
    ).astype(int)
    out["is_double_day"] = (out["month"] == out["day_of_month"]).astype(int)
    out["days_since_start"] = (out["Date"] - out["Date"].min()).dt.days
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["week_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["week_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    return out


def add_tet_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    tet_map = pd.DataFrame(
        {"year": list(TET_DATES.keys()), "tet_date": pd.to_datetime(list(TET_DATES.values()))}
    )
    out = out.merge(tet_map, on="year", how="left")
    out["days_to_tet"] = (out["tet_date"] - out["Date"]).dt.days
    out["is_pre_tet_rush"] = (
        (out["days_to_tet"] > 0) & (out["days_to_tet"] <= 21)
    ).astype(int)
    out["is_tet_holiday"] = (
        (out["days_to_tet"] <= 0) & (out["days_to_tet"] >= -6)
    ).astype(int)
    return out.drop(columns=["tet_date"])


def build_web_traffic_profiles(
    web_traffic_df: pd.DataFrame | None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if web_traffic_df is None or web_traffic_df.empty:
        return pd.DataFrame(
            columns=["month", "day_of_week"] + PROFILE_FEATURES[:6]
        )

    traffic = web_traffic_df.copy()
    traffic["date"] = pd.to_datetime(traffic["date"])
    if end_date is not None:
        traffic = traffic[traffic["date"] <= pd.to_datetime(end_date)].copy()
    traffic["month"] = traffic["date"].dt.month
    traffic["day_of_week"] = traffic["date"].dt.dayofweek
    traffic["pages_per_session"] = traffic["page_views"] / traffic["sessions"].replace(0, np.nan)
    grouped = (
        traffic.groupby(["month", "day_of_week"], as_index=False)
        .agg(
            traffic_sessions_profile=("sessions", "mean"),
            traffic_unique_visitors_profile=("unique_visitors", "mean"),
            traffic_page_views_profile=("page_views", "mean"),
            traffic_bounce_rate_profile=("bounce_rate", "mean"),
            traffic_avg_session_duration_profile=("avg_session_duration_sec", "mean"),
            traffic_pages_per_session_profile=("pages_per_session", "mean"),
        )
    )
    return grouped


def build_inventory_profiles(
    inventory_df: pd.DataFrame | None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if inventory_df is None or inventory_df.empty:
        return pd.DataFrame(
            columns=["month"] + PROFILE_FEATURES[6:]
        )

    inv = inventory_df.copy()
    inv["snapshot_date"] = pd.to_datetime(inv["snapshot_date"])
    if end_date is not None:
        inv = inv[inv["snapshot_date"] <= pd.to_datetime(end_date)].copy()
    inv["month"] = inv["snapshot_date"].dt.month
    grouped = (
        inv.groupby(["year", "month"], as_index=False)
        .agg(
            stock_on_hand=("stock_on_hand", "sum"),
            units_received=("units_received", "sum"),
            units_sold=("units_sold", "sum"),
            stockout_days=("stockout_days", "mean"),
            days_of_supply=("days_of_supply", "mean"),
            fill_rate=("fill_rate", "mean"),
            sell_through_rate=("sell_through_rate", "mean"),
            stockout_flag=("stockout_flag", "mean"),
            overstock_flag=("overstock_flag", "mean"),
            reorder_flag=("reorder_flag", "mean"),
        )
    )
    profile = (
        grouped.groupby("month", as_index=False)
        .agg(
            inventory_stock_on_hand_profile=("stock_on_hand", "mean"),
            inventory_units_received_profile=("units_received", "mean"),
            inventory_units_sold_profile=("units_sold", "mean"),
            inventory_stockout_days_profile=("stockout_days", "mean"),
            inventory_days_of_supply_profile=("days_of_supply", "mean"),
            inventory_fill_rate_profile=("fill_rate", "mean"),
            inventory_sell_through_rate_profile=("sell_through_rate", "mean"),
            inventory_stockout_flag_rate_profile=("stockout_flag", "mean"),
            inventory_overstock_flag_rate_profile=("overstock_flag", "mean"),
            inventory_reorder_flag_rate_profile=("reorder_flag", "mean"),
        )
    )
    return profile


def build_feature_frame(
    sales_df: pd.DataFrame,
    future_dates: Iterable[pd.Timestamp] | None = None,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    profile_end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    sales = ensure_datetime(sales_df)
    base_cols = ["Date", "Revenue", "COGS"]
    frame = sales[base_cols].copy()

    if future_dates is not None:
        future = pd.DataFrame({"Date": pd.to_datetime(list(future_dates))})
        future["Revenue"] = np.nan
        future["COGS"] = np.nan
        frame = pd.concat([frame, future], ignore_index=True)

    frame = frame.sort_values("Date").drop_duplicates("Date", keep="first").reset_index(drop=True)
    frame = add_tet_features(frame)

    profile_end_date = pd.to_datetime(profile_end_date) if profile_end_date is not None else sales["Date"].max()

    promo_features = make_daily_promo_features(
        promotions_df if promotions_df is not None else pd.DataFrame(),
        start_date=frame["Date"].min(),
        end_date=frame["Date"].max(),
    )
    frame = frame.merge(promo_features, on="Date", how="left")
    frame["is_promo"] = frame["is_promo"].fillna(0).astype(int)
    frame["max_disc"] = frame["max_disc"].fillna(0.0)

    traffic_profiles = build_web_traffic_profiles(web_traffic_df, end_date=profile_end_date)
    frame = frame.merge(traffic_profiles, on=["month", "day_of_week"], how="left")

    inventory_profiles = build_inventory_profiles(inventory_df, end_date=profile_end_date)
    frame = frame.merge(inventory_profiles, on="month", how="left")

    for col in PROFILE_FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan
        frame[col] = frame[col].fillna(frame[col].median() if frame[col].notna().any() else 0.0)
    return frame


def add_training_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime(frame)

    for offset in LAG_OFFSETS:
        out[f"revenue_lag_{offset}"] = out["Revenue"].shift(offset)
        out[f"cogs_lag_{offset}"] = out["COGS"].shift(offset)

    for window in ROLLING_WINDOWS:
        out[f"revenue_rolling_mean_{window}"] = out["Revenue"].rolling(window=window).mean()
        out[f"revenue_rolling_std_{window}"] = out["Revenue"].rolling(window=window).std()
        out[f"cogs_rolling_mean_{window}"] = out["COGS"].rolling(window=window).mean()
        out[f"cogs_rolling_std_{window}"] = out["COGS"].rolling(window=window).std()

    out["revenue_dow_avg_prior"] = (
        out.groupby("day_of_week")["Revenue"].transform(lambda s: s.shift(1).expanding().mean())
    )
    out["revenue_month_avg_prior"] = (
        out.groupby("month")["Revenue"].transform(lambda s: s.shift(1).expanding().mean())
    )
    out["cogs_dow_avg_prior"] = (
        out.groupby("day_of_week")["COGS"].transform(lambda s: s.shift(1).expanding().mean())
    )
    out["cogs_month_avg_prior"] = (
        out.groupby("month")["COGS"].transform(lambda s: s.shift(1).expanding().mean())
    )
    return out


def build_supervised_training_frame(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = build_feature_frame(
        sales_df=sales_df,
        future_dates=None,
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )
    frame = add_training_lag_features(frame)
    return frame.dropna().reset_index(drop=True)


def get_training_matrices(
    sales_df: pd.DataFrame,
    promotions_df: pd.DataFrame | None = None,
    web_traffic_df: pd.DataFrame | None = None,
    inventory_df: pd.DataFrame | None = None,
    use_log_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    train_df = build_supervised_training_frame(
        sales_df,
        promotions_df=promotions_df,
        web_traffic_df=web_traffic_df,
        inventory_df=inventory_df,
    )
    if use_log_target:
        y_rev = np.log1p(train_df["Revenue"])
        y_cogs = np.log1p(train_df["COGS"])
    else:
        y_rev = train_df["Revenue"].copy()
        y_cogs = train_df["COGS"].copy()
    return train_df[list(ALL_FEATURES)], y_rev, y_cogs, train_df


def _window_stats(history: pd.Series, end_idx: int, window: int) -> tuple[float, float]:
    start_idx = max(0, end_idx - window)
    window_values = history.iloc[start_idx:end_idx]
    return float(window_values.mean()), float(window_values.std())


def _group_prior(history: pd.DataFrame, end_idx: int, group_col: str, target_col: str) -> float:
    if end_idx <= 0:
        return np.nan
    current_group = history.loc[end_idx, group_col]
    prior = history.iloc[:end_idx]
    subset = prior.loc[prior[group_col] == current_group, target_col]
    if subset.empty:
        return float(prior[target_col].mean()) if not prior.empty else np.nan
    return float(subset.mean())


def build_recursive_feature_row(
    history_df: pd.DataFrame,
    target_date: pd.Timestamp,
    static_feature_source: pd.DataFrame,
) -> pd.DataFrame:
    history = ensure_datetime(history_df)
    target_date = pd.to_datetime(target_date)
    row = static_feature_source.loc[static_feature_source["Date"] == target_date]
    if row.empty:
        raise KeyError(f"Static features not found for {target_date.date()}")
    idx = history.index[history["Date"] == target_date]
    if len(idx) == 0:
        raise KeyError(f"Target date {target_date.date()} not found in history frame")
    idx = int(idx[0])

    def get_past(col: str, offset: int) -> float:
        past_idx = idx - offset
        if past_idx < 0:
            return np.nan
        return float(history.loc[past_idx, col])

    feature_row = row.iloc[0][list(STATIC_FEATURES)].copy()
    for offset in LAG_OFFSETS:
        feature_row[f"revenue_lag_{offset}"] = get_past("Revenue", offset)
        feature_row[f"cogs_lag_{offset}"] = get_past("COGS", offset)

    for window in ROLLING_WINDOWS:
        revenue_mean, revenue_std = _window_stats(history["Revenue"], idx, window)
        cogs_mean, cogs_std = _window_stats(history["COGS"], idx, window)
        feature_row[f"revenue_rolling_mean_{window}"] = revenue_mean
        feature_row[f"revenue_rolling_std_{window}"] = revenue_std
        feature_row[f"cogs_rolling_mean_{window}"] = cogs_mean
        feature_row[f"cogs_rolling_std_{window}"] = cogs_std

    feature_row["revenue_dow_avg_prior"] = _group_prior(history, idx, "day_of_week", "Revenue")
    feature_row["revenue_month_avg_prior"] = _group_prior(history, idx, "month", "Revenue")
    feature_row["cogs_dow_avg_prior"] = _group_prior(history, idx, "day_of_week", "COGS")
    feature_row["cogs_month_avg_prior"] = _group_prior(history, idx, "month", "COGS")

    return pd.DataFrame([feature_row], columns=list(ALL_FEATURES))

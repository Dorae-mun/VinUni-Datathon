from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def recent_trend_scale(
    hist: pd.DataFrame,
    col: str,
    ref_date: pd.Timestamp,
    window_days: int = 90,
    clip_low: float = 0.6,
    clip_high: float = 1.4,
) -> float:
    recent = hist[
        (hist["Date"] < ref_date)
        & (hist["Date"] >= ref_date - pd.Timedelta(days=window_days))
    ]
    past = hist[
        (hist["Date"] < ref_date - pd.DateOffset(years=1))
        & (
            hist["Date"]
            >= ref_date - pd.DateOffset(years=1) - pd.Timedelta(days=window_days)
        )
    ]
    if len(recent) < 30 or len(past) < 30:
        return 1.0
    m_recent = recent[col].median()
    m_past = past[col].median()
    if m_past <= 0:
        return 1.0
    return float(np.clip(m_recent / m_past, clip_low, clip_high))


def add_calendar_only_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["day"] = out["Date"].dt.day
    out["month"] = out["Date"].dt.month
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["quarter"] = out["Date"].dt.quarter
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["Date"].dt.is_month_end.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["doy_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["trend"] = (out["Date"] - pd.Timestamp("2012-01-01")).dt.days
    return out


class BaseForecastModel:
    name: str = "base"

    def fit(self, sales: pd.DataFrame, target: str | None = None) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        raise NotImplementedError


class LastYearTrendModel(BaseForecastModel):
    name = "last_year_mapping"

    def __init__(self, search_delta: int = 3):
        self.search_delta = search_delta
        self.hist_: pd.DataFrame | None = None
        self._idx_: dict[tuple[int, int], tuple[float, float]] = {}

    def fit(self, sales: pd.DataFrame, target: str | None = None) -> "LastYearTrendModel":
        self.hist_ = sales.sort_values("Date").reset_index(drop=True).copy()
        self._idx_ = {}
        for _, row in self.hist_.iterrows():
            key = (row["Date"].year, row["Date"].dayofyear)
            self._idx_[key] = (float(row["Revenue"]), float(row["COGS"]))
        return self

    def _lookup(self, year: int, doy: int, col_idx: int) -> float | None:
        for delta in range(0, self.search_delta + 1):
            offsets = [0] if delta == 0 else [delta, -delta]
            for off in offsets:
                key = (year, doy + off)
                if key in self._idx_:
                    return self._idx_[key][col_idx]
        return None

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        assert self.hist_ is not None
        results: list[dict[str, Any]] = []
        for d in pd.to_datetime(dates):
            year = d.year
            doy = d.dayofyear
            if year == 2023:
                src_years = [(2022, 0.75), (2021, 0.25)]
            elif year >= 2024:
                src_years = [(2022, 0.6), (2021, 0.3), (2020, 0.1)]
            else:
                src_years = [(year - 1, 1.0)]
            scale_rev = recent_trend_scale(self.hist_, "Revenue", d)
            scale_cogs = recent_trend_scale(self.hist_, "COGS", d)
            rev_sum = cog_sum = w_sum = 0.0
            for src_year, weight in src_years:
                rev = self._lookup(src_year, doy, 0)
                cog = self._lookup(src_year, doy, 1)
                if rev is not None and cog is not None:
                    rev_sum += weight * rev
                    cog_sum += weight * cog
                    w_sum += weight
            if w_sum == 0:
                rev_base = float(self.hist_["Revenue"].median())
                cog_base = float(self.hist_["COGS"].median())
            else:
                rev_base = rev_sum / w_sum
                cog_base = cog_sum / w_sum
            results.append(
                {
                    "Date": d,
                    "Revenue": max(0.0, rev_base * scale_rev),
                    "COGS": max(0.0, cog_base * scale_cogs),
                }
            )
        return pd.DataFrame(results)


class DayOfYearProfileModel(BaseForecastModel):
    name = "day_of_year_profile"

    def __init__(self, smooth_window: int = 3):
        self.smooth_window = smooth_window
        self.hist_: pd.DataFrame | None = None
        self.profile_: pd.DataFrame | None = None

    def fit(self, sales: pd.DataFrame, target: str | None = None) -> "DayOfYearProfileModel":
        self.hist_ = sales.sort_values("Date").reset_index(drop=True).copy()
        df = self.hist_.copy()
        df["doy"] = df["Date"].dt.dayofyear
        profile = df.groupby("doy")[["Revenue", "COGS"]].median().reset_index()
        full_doy = pd.DataFrame({"doy": np.arange(1, 367)})
        profile = full_doy.merge(profile, on="doy", how="left")
        profile["Revenue"] = profile["Revenue"].interpolate().bfill().ffill()
        profile["COGS"] = profile["COGS"].interpolate().bfill().ffill()
        if self.smooth_window > 1:
            rev_s = profile["Revenue"].rolling(self.smooth_window, min_periods=1, center=True).mean()
            cog_s = profile["COGS"].rolling(self.smooth_window, min_periods=1, center=True).mean()
            profile["Revenue"] = 0.5 * rev_s + 0.5 * profile["Revenue"]
            profile["COGS"] = 0.5 * cog_s + 0.5 * profile["COGS"]
        self.profile_ = profile.set_index("doy")
        return self

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        assert self.hist_ is not None and self.profile_ is not None
        rows = []
        for d in pd.to_datetime(dates):
            doy = min(int(d.dayofyear), 366)
            rev_base = float(self.profile_.loc[doy, "Revenue"])
            cog_base = float(self.profile_.loc[doy, "COGS"])
            rows.append(
                {
                    "Date": d,
                    "Revenue": max(0.0, rev_base * recent_trend_scale(self.hist_, "Revenue", d)),
                    "COGS": max(0.0, cog_base * recent_trend_scale(self.hist_, "COGS", d)),
                }
            )
        return pd.DataFrame(rows)


class CalendarProfileModel(BaseForecastModel):
    name = "calendar_profile"

    def fit(self, sales: pd.DataFrame, target: str | None = None) -> "CalendarProfileModel":
        df = sales.sort_values("Date").reset_index(drop=True).copy()
        df["month"] = df["Date"].dt.month
        df["dow"] = df["Date"].dt.dayofweek
        df["woy"] = df["Date"].dt.isocalendar().week.astype(int)
        self.md_ = df.groupby(["month", "dow"])[["Revenue", "COGS"]].median().reset_index()
        self.wd_ = df.groupby(["woy", "dow"])[["Revenue", "COGS"]].median().reset_index()
        self.global_rev_ = float(df["Revenue"].median())
        self.global_cog_ = float(df["COGS"].median())
        self.hist_ = df
        return self

    def _lookup(self, table: pd.DataFrame, keys: tuple[int, int], col: str, fallback: float) -> float:
        key_a, key_b = keys
        a, b = table.columns[0], table.columns[1]
        row = table[(table[a] == key_a) & (table[b] == key_b)]
        return float(row[col].iloc[0]) if not row.empty else fallback

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        rows = []
        for d in pd.to_datetime(dates):
            month = int(d.month)
            dow = int(d.dayofweek)
            woy = int(d.isocalendar().week)
            r_md = self._lookup(self.md_, (month, dow), "Revenue", self.global_rev_)
            c_md = self._lookup(self.md_, (month, dow), "COGS", self.global_cog_)
            r_wd = self._lookup(self.wd_, (woy, dow), "Revenue", self.global_rev_)
            c_wd = self._lookup(self.wd_, (woy, dow), "COGS", self.global_cog_)
            rows.append(
                {
                    "Date": d,
                    "Revenue": max(0.0, (0.5 * r_md + 0.5 * r_wd) * recent_trend_scale(self.hist_, "Revenue", d)),
                    "COGS": max(0.0, (0.5 * c_md + 0.5 * c_wd) * recent_trend_scale(self.hist_, "COGS", d)),
                }
            )
        return pd.DataFrame(rows)


class RecentYearWeightedModel(BaseForecastModel):
    name = "recent_year_weighted"

    def __init__(self, year_weights: dict[int, float] | None = None):
        self.year_weights = year_weights or {2022: 0.45, 2021: 0.30, 2020: 0.15, 2019: 0.10}

    def fit(self, sales: pd.DataFrame, target: str | None = None) -> "RecentYearWeightedModel":
        self.hist_ = sales.sort_values("Date").reset_index(drop=True).copy()
        self.hist_["year"] = self.hist_["Date"].dt.year
        self.hist_["doy"] = self.hist_["Date"].dt.dayofyear
        self.idx_ = self.hist_.set_index(["year", "doy"])[["Revenue", "COGS"]]
        return self

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        rows = []
        for d in pd.to_datetime(dates):
            doy = d.dayofyear
            rev_sum = cog_sum = w_sum = 0.0
            for year, weight in self.year_weights.items():
                if (year, doy) in self.idx_.index:
                    row = self.idx_.loc[(year, doy)]
                    rev_sum += weight * float(row["Revenue"])
                    cog_sum += weight * float(row["COGS"])
                    w_sum += weight
            if w_sum == 0:
                rev_base = float(self.hist_["Revenue"].median())
                cog_base = float(self.hist_["COGS"].median())
            else:
                rev_base = rev_sum / w_sum
                cog_base = cog_sum / w_sum
            rows.append(
                {
                    "Date": d,
                    "Revenue": max(0.0, rev_base * recent_trend_scale(self.hist_, "Revenue", d)),
                    "COGS": max(0.0, cog_base * recent_trend_scale(self.hist_, "COGS", d)),
                }
            )
        return pd.DataFrame(rows)


class LGBMCalendarModel(BaseForecastModel):
    name = "lgbm_calendar_only"

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model_: Any | None = None
        self.feat_: list[str] = []
        self._target_: str = "Revenue"

    def _build(self) -> LGBMRegressor:
        return LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=63,
            max_depth=7,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_gain_to_split=0.001,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        sales: pd.DataFrame,
        target: str = "Revenue",
        valid_start: str = "2021-01-01",
        valid_end: str = "2022-12-31",
    ) -> "LGBMCalendarModel":
        self._target_ = target
        df = add_calendar_only_features(sales[["Date", target]].copy())
        self.feat_ = [c for c in df.columns if c not in ["Date", target]]
        train = df[df["Date"] < valid_start]
        valid = df[(df["Date"] >= valid_start) & (df["Date"] <= valid_end)]
        model = self._build()
        model.fit(
            train[self.feat_],
            np.log1p(train[target]),
            eval_set=[(valid[self.feat_], np.log1p(valid[target]))],
            callbacks=[],
        )
        self.model_ = model
        self._full_df_ = df
        return self

    def predict(self, dates: pd.Series) -> pd.DataFrame:
        assert self.model_ is not None
        df = add_calendar_only_features(pd.DataFrame({"Date": pd.to_datetime(dates)}))
        preds = np.expm1(self.model_.predict(df[self.feat_]))
        preds = np.maximum(preds, 0.0)
        return pd.DataFrame({"Date": pd.to_datetime(dates).values, self._target_: preds})

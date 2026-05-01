"""
src/optimizer.py
================
Multi-Objective Optimizer — Datathon 2026 | The Gridbreakers

------------------------------------------------------------------------------
BIẾN ĐẦU VÀO TỔNG QUÁT
------------------------------------------------------------------------------
  Tối ưu Tồn kho & Khuyến mãi (Product-centric):
    x = product       (category, segment, price_tier, size, color)
    y = customer_group(age_group, acquisition_channel, region)
    z = timing        (month, quarter, is_promo, promo_type)
    w = discount_depth(discount_value, stackable, min_order_value)

  Tối ưu Customer Intelligence:
    x = customer_segment  (RFM: Champions, Potential Loyalists, ...)
    y = acquisition_channel (organic_search, social_media, ...)
    z = timing            (month, is_double_day, is_q4_event, ...)
    w = discount_params   (depth, type, min_order_value)
    v = region            (East, Central, West / city tier)

OUTPUTS:
    1. Revenue Optimization       — tối đa doanh thu thuần
    2. Profit Optimization        — tối đa lợi nhuận gộp
    3. Customer Acquisition Opt.  — tối đa khách hàng mới / CAC thấp nhất
    4. Discount Timing Opt.       — lịch giảm giá ROI cao nhất

CUSTOMER INTELLIGENCE:
    - RFM Segmentation (Champions / Loyal / Churn risk / Regular)
    - CLV Estimation   (historical + projected)
    - Churn Probability (heuristic scoring)
    - RCAV: Revenue per Channel per Acquisition Visit

PHƯƠNG PHP:
    - Heuristic / Scoring  → Inventory reorder, overstock
    - Linear Programming   → Discount budget allocation (scipy)
    - Simulation           → Monte Carlo CLV & churn cost
------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# SECTION 0 — Constants & Config
# ------------------------------------------------------------------------------

RFM_SEGMENT_MAP = {}

# Simplified RFM segmentation (2026-04)
# Segment labels produced by RFMEngine._assign_segment():
#   - Champions
#   - Loyal customers
#   - Churn risk
#   - Regular customers
CHURN_RISK_BASE = {
    "Champions": 0.05,
    "Loyal customers": 0.15,
    "Regular customers": 0.35,
    "Churn risk": 0.70,
}

# Retention discount khuyến nghị theo segment
RETENTION_DISCOUNT_PCT = {
    "Champions": 5,
    "Loyal customers": 10,
    "Regular customers": 10,
    "Churn risk": 25,
}
# Vietnamese shopping events (z: timing flags)
VN_EVENTS: Dict[str, Tuple[int,int]] = {
    "tet"              : (1,  2),
    "reunification_day": (4, 30),
    "womens_day"       : (3,  8),
    "national_day"     : (9,  2),
    "double_11"        : (11,11),
    "double_12"        : (12,12),
    "year_end_sale"    : (12,25),
    "mid_year_sale"    : (6, 15),
}

# City tier (v: region)
CITY_TIER_MAP = {
    "tier_1": ["Hà Nội", "Hồ Chí Minh", "à Nẵng"],
    "tier_2": ["Hải Phòng", "Cần Thơ", "Biên Hòa", "Nha Trang"],
}

DEFAULT_WEIGHTS = dict(revenue=0.35, profit=0.35, acquire=0.15, retain=0.15)

# Scenario presets for business users (objective weights are system config)
SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Balanced": dict(DEFAULT_WEIGHTS),
    "Growth": dict(revenue=0.55, profit=0.20, acquire=0.15, retain=0.10),
    "Profitability": dict(revenue=0.20, profit=0.60, acquire=0.05, retain=0.15),
    "Retention": dict(revenue=0.15, profit=0.25, acquire=0.05, retain=0.55),
    "Acquisition": dict(revenue=0.25, profit=0.15, acquire=0.50, retain=0.10),
}


# ------------------------------------------------------------------------------
# SECTION 1 — Input Variable Dataclasses
# ------------------------------------------------------------------------------

@dataclass
class ProductVar:
    """
    x = đặc trưng sản phẩm
    Dùng cho: inventory optimization, promo targeting theo product
    """
    category   : Optional[str] = None   # Streetwear, Activewear, ...
    segment    : Optional[str] = None   # Premium, Standard, Performance
    price_tier : Optional[str] = None   # "low" / "mid" / "high"
    size       : Optional[str] = None   # S, M, L, XL
    color      : Optional[str] = None


@dataclass
class CustomerVar:
    """
    y = đặc trưng nhóm khách hàng
    Dùng cho: customer acquisition, retention targeting
    """
    age_group           : Optional[str] = None   # 18-24, 25-34, 35-44, ...
    acquisition_channel : Optional[str] = None   # organic_search, social_media, ...
    rfm_segment         : Optional[str] = None   # Champions, Churn risk, ...
    gender              : Optional[str] = None


@dataclass
class TimingVar:
    """
    z = đặc trưng thi điểm
    Dùng cho: discount timing, seasonal planning
    """
    month        : Optional[int] = None
    quarter      : Optional[int] = None
    is_promo     : bool          = False
    promo_type   : Optional[str] = None   # percentage / fixed
    is_double_day: bool          = False
    is_q4_event  : bool          = False
    is_tet       : bool          = False
    is_mid_year  : bool          = False


@dataclass
class DiscountVar:
    """
    w = tham số discount
    Dùng cho: discount depth optimization, budget allocation
    """
    discount_value  : float = 0.0        # % hoặc VND tùy promo_type
    promo_type      : str   = "percentage"
    stackable       : bool  = False
    min_order_value : float = 0.0
    budget_cap      : float = np.inf     # tổng ngân sách cho chiến dịch


@dataclass
class RegionVar:
    """
    v = đặc trưng địa lý
    Dùng cho: regional targeting, city-tier strategy
    """
    region    : Optional[str] = None     # East, Central, West
    city      : Optional[str] = None
    city_tier : Optional[str] = None     # tier_1, tier_2, other


@dataclass
class OptimizeRequest:
    """
    Gói toàn bộ biến x, y, z, w, v thành 1 request object.
    Truyn vào bất kỳ hàm optimize_* nào.

    Ví dụ:
        req = OptimizeRequest(
            x=ProductVar(category="Streetwear", segment="Premium"),
            y=CustomerVar(age_group="25-34", rfm_segment="Churn risk"),
            z=TimingVar(month=11, is_double_day=True),
            w=DiscountVar(discount_value=20, promo_type="percentage"),
            v=RegionVar(region="East"),
        )
    """
    x : ProductVar  = field(default_factory=ProductVar)
    y : CustomerVar = field(default_factory=CustomerVar)
    z : TimingVar   = field(default_factory=TimingVar)
    w : DiscountVar = field(default_factory=DiscountVar)
    v : RegionVar   = field(default_factory=RegionVar)
    weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())


# ------------------------------------------------------------------------------
# SECTION 1B — Optimization Orchestration Layer (Decision vars + Constraints)
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizationProblem:
    """
    Prescriptive decision optimization problem.

    User provides:
      - fixed constraints (fixed_request)
      - which fields are decision variables (decision_variables)
      - objective weights (objective_weights)

    Decision variable names use dotted paths:
      - "x.category", "x.segment"
      - "y.rfm_segment", "y.acquisition_channel"
      - "z.month"
      - "w.discount_value", "w.budget_cap"
      - "v.region", "v.city_tier"
    """

    objective_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    fixed_request: OptimizeRequest = field(default_factory=OptimizeRequest)
    decision_variables: List[str] = field(default_factory=list)
    search_space: Dict[str, List[Any]] = field(default_factory=dict)
    method: str = "coordinate_descent"  # "coordinate_descent" | "grid"
    rounds: int = 2
    max_evals: int = 250
    seed: int = 42


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x))
    except Exception:
        return default


def _get_attr_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _set_attr_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def _iter_grid(assignments: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(assignments.keys())
    values = [assignments[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))

# ------------------------------------------------------------------------------
# SECTION 2 — RFM Engine
# ------------------------------------------------------------------------------

class RFMEngine:
    """
    Tính RFM scores và gán segment từ orders.csv + order_items.csv.

    Attributes
    ----------
    rfm : pd.DataFrame
        Bảng RFM với cột: customer_id, recency, frequency, monetary,
        r_score, f_score, m_score, rfm_segment, clv_estimated, churn_prob
    """

    def __init__(
        self,
        orders_df     : pd.DataFrame,
        order_items_df: pd.DataFrame,
        snapshot_date : Optional[pd.Timestamp] = None,
    ):
        self.orders      = orders_df.copy()
        self.items       = order_items_df.copy()
        self.snapshot    = snapshot_date or orders_df["order_date"].max()
        self.rfm         = self._compute_rfm()

    # ── internal ─────────────────────────────────────────────────────────

    def _compute_rfm(self) -> pd.DataFrame:
        """Tính R, F, M cho mỗi khách hàng."""
        # Chỉ tính trên đơn delivered/shipped (không cancelled)
        valid = self.orders[
            ~self.orders["order_status"].isin(["cancelled"])
        ].copy()

        # Monetary: join với order_items để lấy line_revenue
        revenue_per_order = (
            self.items
            .groupby("order_id")["line_revenue"]
            .sum()
            .reset_index()
            .rename(columns={"line_revenue": "order_revenue"})
        )
        valid = valid.merge(revenue_per_order, on="order_id", how="left")
        valid["order_revenue"] = valid["order_revenue"].fillna(0)

        rfm = (
            valid
            .groupby("customer_id")
            .agg(
                last_order_date = ("order_date",    "max"),
                frequency       = ("order_id",      "nunique"),
                monetary        = ("order_revenue", "sum"),
            )
            .reset_index()
        )
        rfm["recency"] = (self.snapshot - rfm["last_order_date"]).dt.days

        # Score 1–5 (qcut: 5=tốt nhất)
        rfm["R_group"] = np.select(
            [rfm["recency"] <= 365, (rfm["recency"] > 365) & (rfm["recency"] <= 1000), rfm["recency"] > 1000],
            ["Recent", "Warm", "Old"],
            default="Warm",
        )

        rfm["F_group"] = np.select(
            [rfm["frequency"] == 1, (rfm["frequency"] >= 2) & (rfm["frequency"] <= 5), rfm["frequency"] >= 6],
            ["One-time", "Occasional", "Loyal"],
            default="One-time",
        )

        rfm["M_group"] = np.select(
            [
                rfm["monetary"] < 90_000,
                (rfm["monetary"] >= 90_000) & (rfm["monetary"] <= 300_000),
                rfm["monetary"] > 300_000,
            ],
            ["Low", "Medium", "High"],
            default="Medium",
        )

        rfm["r_score"] = rfm["R_group"].map({"Recent": 5, "Warm": 3, "Old": 1}).fillna(3).astype(int)
        rfm["f_score"] = rfm["F_group"].map({"One-time": 1, "Occasional": 3, "Loyal": 5}).fillna(1).astype(int)
        rfm["m_score"] = rfm["M_group"].map({"Low": 1, "Medium": 3, "High": 5}).fillna(3).astype(int)

        rfm["rfm_score"] = rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
        rfm["rfm_segment"] = rfm.apply(self._assign_segment, axis=1)
        rfm["clv_estimated"]= self._estimate_clv(rfm)
        rfm["churn_prob"]   = rfm["rfm_segment"].map(CHURN_RISK_BASE).fillna(0.5)
        rfm["retention_discount_pct"] = rfm["rfm_segment"].map(RETENTION_DISCOUNT_PCT).fillna(15)

        return rfm.drop(columns=["last_order_date"])

    @staticmethod
    def _assign_segment(row) -> str:
        """Map (r, f, m) scores → RFM segment label."""
        r = str(row.get("R_group", ""))
        f = str(row.get("F_group", ""))
        m = str(row.get("M_group", ""))

        if r == "Recent" and f == "Loyal" and m == "High":
            return "Champions"

        if f == "Loyal" and m != "Low":
            return "Loyal customers"

        if r == "Old":
            return "Churn risk"

        return "Regular customers"

    @staticmethod
    def _estimate_clv(rfm: pd.DataFrame, avg_lifespan_days: int = 730) -> pd.Series:
        """
        CLV đơn giản = (monetary / frequency) * frequency_per_day * lifespan
        ây là historical CLV, không phải predictive.
        """
        avg_order_value   = rfm["monetary"] / rfm["frequency"].replace(0, 1)
        purchase_freq_est = rfm["frequency"] / rfm["recency"].replace(0, 1) * 30  # orders/month
        return (avg_order_value * purchase_freq_est * (avg_lifespan_days / 30)).round(2)

    # ── public ───────────────────────────────────────────────────────────

    def get_segment_summary(self) -> pd.DataFrame:
        """Thống kê tổng hợp theo RFM segment."""
        return (
            self.rfm
            .groupby("rfm_segment")
            .agg(
                n_customers      = ("customer_id",         "count"),
                avg_recency      = ("recency",             "mean"),
                avg_frequency    = ("frequency",           "mean"),
                avg_monetary     = ("monetary",            "mean"),
                total_clv        = ("clv_estimated",       "sum"),
                avg_churn_prob   = ("churn_prob",          "mean"),
                avg_ret_discount = ("retention_discount_pct","mean"),
            )
            .round(2)
            .reset_index()
            .sort_values("total_clv", ascending=False)
        )

    def filter_by(
        self,
        x: Optional[ProductVar]  = None,
        y: Optional[CustomerVar] = None,
        v: Optional[RegionVar]   = None,
        orders_with_geo: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Lc RFM table theo biến x (segment), y (channel), v (region).
        orders_with_geo: orders đã join geography (cần nếu filter theo region).
        """
        df = self.rfm.copy()

        if y and y.rfm_segment:
            df = df[df["rfm_segment"] == y.rfm_segment]

        if y and y.acquisition_channel and orders_with_geo is not None:
            valid_customers = orders_with_geo[
                orders_with_geo["order_source"] == y.acquisition_channel
            ]["customer_id"].unique()
            df = df[df["customer_id"].isin(valid_customers)]

        if v and v.region and orders_with_geo is not None:
            valid_customers = orders_with_geo[
                orders_with_geo["region"] == v.region
            ]["customer_id"].unique()
            df = df[df["customer_id"].isin(valid_customers)]

        return df


# ------------------------------------------------------------------------------
# SECTION 3 — Inventory Optimizer
# ------------------------------------------------------------------------------

class InventoryOptimizer:
    """
    Tối ưu tồn kho dựa trên inventory.csv + products.csv.

    Biến đầu vào:
        x = ProductVar  (category, segment, price_tier)
        z = TimingVar   (month — xét theo mùa)
        a = lead_time_days       (số ngày nhà cung cấp giao hàng)
        b = service_level        (mức độ dịch vụ: 0.90, 0.95, 0.99)
        c = holding_cost_pct     (% giá trị tồn kho / tháng)
        d = stockout_cost_factor (hệ số chi phí hết hàng so với giá bán)
    """

    def __init__(
        self,
        inventory_df: pd.DataFrame,
        products_df : pd.DataFrame,
        a_lead_time : float = 7.0,    # a: lead time mặc định 7 ngày
        b_service_level: float = 0.95,# b: service level 95%
        c_holding_cost : float = 0.02,# c: 2% giá trị / tháng
        d_stockout_cost: float = 1.5, # d: penalty = 1.5x giá bán
    ):
        self.inv      = inventory_df.copy()
        self.products = products_df.copy()
        self.a        = a_lead_time
        self.b        = b_service_level
        self.c        = c_holding_cost
        self.d        = d_stockout_cost
        self._merged  = self._prepare()

    def _prepare(self) -> pd.DataFrame:
        df = self.inv.merge(
            self.products[["product_id","price","cogs","gross_margin","category","segment"]],
            on="product_id", how="left", suffixes=("","_p")
        )
        # Dùng category / segment từ inventory nếu có, fallback v products
        for col in ["category","segment"]:
            if f"{col}_p" in df.columns:
                df[col] = df[col].fillna(df[f"{col}_p"])
                df.drop(columns=[f"{col}_p"], inplace=True)

        # Tính daily demand từ units_sold / days_in_month
        df["days_in_month"]  = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str)
        ).apply(lambda d: pd.Period(d, "M").days_in_month)
        df["daily_demand"]   = df["units_sold"] / df["days_in_month"].replace(0,1)
        df["demand_std"]     = df.groupby("product_id")["daily_demand"].transform("std").fillna(0)
        return df

    # ── z-score cho service level ─────────────────────────────────────────
    @staticmethod
    def _z_score(service_level: float) -> float:
        from scipy.stats import norm
        return norm.ppf(service_level)

    # ── Core calculations ─────────────────────────────────────────────────

    def calc_safety_stock(
        self,
        x: Optional[ProductVar] = None,
        z: Optional[TimingVar]  = None,
        a: Optional[float]      = None,  # override lead time
        b: Optional[float]      = None,  # override service level
    ) -> pd.DataFrame:
        """
        Safety Stock = z_b × σ_demand × √a
            z_b = z-score của service level b
            σ   = std của daily demand
            a   = lead time days

        Returns DataFrame với: product_id, product_name, category, segment,
                                daily_demand, demand_std, safety_stock,
                                reorder_point, recommended_order_qty
        """
        df  = self._filter(self._merged, x, z)
        z_b = self._z_score(b or self.b)
        lt  = a or self.a

        latest = (
            df.sort_values("snapshot_date")
            .groupby("product_id")
            .last()
            .reset_index()
        )

        latest["safety_stock"]    = (z_b * latest["demand_std"] * np.sqrt(lt)).round(0)
        latest["reorder_point"]   = (latest["daily_demand"] * lt + latest["safety_stock"]).round(0)
        latest["recommended_order_qty"] = np.maximum(
            latest["reorder_point"] - latest["stock_on_hand"], 0
        ).round(0)
        latest["inventory_value"] = latest["stock_on_hand"] * latest["cogs"]
        latest["holding_cost_monthly"] = (latest["inventory_value"] * self.c).round(2)

        cols = ["product_id","product_name","category","segment","price","cogs","gross_margin",
                "stock_on_hand","daily_demand","demand_std","safety_stock",
                "reorder_point","recommended_order_qty",
                "inventory_value","holding_cost_monthly",
                "stockout_flag","overstock_flag","reorder_flag","fill_rate"]
        return latest[[c for c in cols if c in latest.columns]]

    def suggest_reorder(
        self,
        x: Optional[ProductVar] = None,
        z: Optional[TimingVar]  = None,
        a: Optional[float]      = None,
        b: Optional[float]      = None,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Danh sách sản phẩm cần đặt hàng ngay (reorder_flag=1 hoặc sắp hết).
        Ưu tiên theo: urgency_score = stockout_days × gross_margin × price
        """
        df = self.calc_safety_stock(x, z, a, b)
        df = df[df["recommended_order_qty"] > 0].copy()
        df["urgency_score"] = (
            df["stockout_flag"].fillna(0) * 2
            + (df["stock_on_hand"] < df["reorder_point"]).astype(int)
            + df["gross_margin"].fillna(0)
        )
        df["action"] = np.where(
            df["stockout_flag"] == 1,
            "HET HANG - DAT NGAY",
            np.where(
                df["stock_on_hand"] < df["reorder_point"],
                "SAP HET - DAT SOM",
                "CAN NHAC DAT THEM",
            ),
        )
        return (
            df.sort_values("urgency_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def flag_overstock(
        self,
        x       : Optional[ProductVar] = None,
        z       : Optional[TimingVar]  = None,
        c       : Optional[float]      = None,  # override holding cost
        top_n   : int = 20,
    ) -> pd.DataFrame:
        """
        Sản phẩm tồn kho vượt mức — tính chi phí lưu kho lãng phí.
        overstock_cost = (stock_on_hand - reorder_point) × cogs × c
        """
        df = self.calc_safety_stock(x, z)
        holding_pct = c or self.c

        over = df[df["overstock_flag"] == 1].copy()
        over["excess_units"]      = (over["stock_on_hand"] - over["reorder_point"]).clip(lower=0)
        over["overstock_cost"]    = (over["excess_units"] * over["cogs"] * holding_pct).round(2)
        over["recommended_action"] = over.apply(
            lambda r: f"Discount {min(int(r['gross_margin']*50), 30)}% to clear {int(r['excess_units'])} units",
            axis=1
        )
        return (
            over.sort_values("overstock_cost", ascending=False)
            .head(top_n)
            [["product_id","product_name","category","segment",
              "stock_on_hand","reorder_point","excess_units",
              "overstock_cost","recommended_action"]]
            .reset_index(drop=True)
        )

    def optimize_inventory_value(
        self,
        x: Optional[ProductVar] = None,
        z: Optional[TimingVar]  = None,
        budget: float = np.inf,
    ) -> Dict:
        """
        Tối ưu phân bổ ngân sách nhập hàng với LP:
            maximize:  Σ (gross_margin_i × order_qty_i × price_i)
            subject to:
                Σ (cogs_i × order_qty_i) ≤ budget
                order_qty_i ≥ 0
                order_qty_i ≥ recommended_order_qty_i  (đảm bảo safety stock)

        Returns dict với:
            allocated_orders: DataFrame sản phẩm + qty tối ưu
            total_cost, total_revenue_potential, roi
        """
        df = self.suggest_reorder(x, z)
        if df.empty or budget == np.inf:
            return {"allocated_orders": df, "note": "Không có ràng buộc ngân sách"}

        n = len(df)
        # Objective: maximize gross_margin × price × qty → minimize negative
        c_obj = -(df["gross_margin"].fillna(0) * df["price"].fillna(0)).values

        # Constraint: Σ cogs_i × qty_i ≤ budget
        A_ub  = [df["cogs"].fillna(0).values.tolist()]
        b_ub  = [budget]

        # Bounds: qty ≥ recommended_order_qty
        bounds = [(row["recommended_order_qty"], None) for _, row in df.iterrows()]

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            df = df.copy()
            df["optimal_order_qty"]   = np.round(result.x, 0)
            df["optimal_cost"]        = df["optimal_order_qty"] * df["cogs"]
            df["revenue_potential"]   = df["optimal_order_qty"] * df["price"]
            total_cost    = df["optimal_cost"].sum()
            total_rev_pot = df["revenue_potential"].sum()
            return {
                "allocated_orders"    : df,
                "total_cost"          : round(total_cost, 2),
                "total_revenue_potential": round(total_rev_pot, 2),
                "roi"                 : round((total_rev_pot - total_cost) / total_cost, 4) if total_cost else 0,
                "budget_utilization"  : round(total_cost / budget, 4),
                "status"              : "optimal",
            }
        return {"allocated_orders": df, "status": result.message}

    @staticmethod
    def _filter(
        df: pd.DataFrame,
        x : Optional[ProductVar],
        z : Optional[TimingVar],
    ) -> pd.DataFrame:
        if x:
            if x.category: df = df[df["category"] == x.category]
            if x.segment:  df = df[df["segment"]  == x.segment]
            if x.size and "size" in df.columns: df = df[df["size"] == x.size]
        if z and z.month and "month" in df.columns:
            df = df[df["month"] == z.month]
        return df


# ------------------------------------------------------------------------------
# SECTION 4 — Promotion Optimizer
# ------------------------------------------------------------------------------

class PromotionOptimizer:
    """
    Tối ưu chiến lược khuyến mãi.

    Biến đầu vào:
        x = ProductVar   (category, segment, price_tier)
        y = CustomerVar  (age_group, rfm_segment, acquisition_channel)
        z = TimingVar    (month, quarter, is_double_day, is_q4_event)
        w = DiscountVar  (discount_value, promo_type, stackable, budget_cap)
        v = RegionVar    (region, city_tier)
    """

    def __init__(
        self,
        order_items_df: pd.DataFrame,
        promotions_df : pd.DataFrame,
        orders_df     : pd.DataFrame,
        products_df   : pd.DataFrame,
    ):
        self.items   = order_items_df.copy()
        self.promos  = promotions_df.copy()
        self.orders  = orders_df.copy()
        self.products= products_df.copy()
        self._perf   = self._compute_promo_performance()

    # ── internal ─────────────────────────────────────────────────────────

    def _compute_promo_performance(self) -> pd.DataFrame:
        """Tính hiệu quả thực tế của từng chiến dịch từ lịch sử."""
        items = self.items.merge(
            self.products[["product_id","category","segment","price","cogs","gross_margin"]],
            on="product_id", how="left"
        ).merge(
            self.orders[["order_id","order_date","order_source"]],
            on="order_id", how="left"
        )

        def _agg_by_promo(col: str) -> pd.DataFrame:
            return (
                items[items[col].notna()]
                .groupby(col)
                .agg(
                    n_orders        = ("order_id",       "nunique"),
                    n_items         = ("order_id",       "count"),
                    total_qty       = ("quantity",       "sum"),
                    total_discount  = ("discount_amount","sum"),
                    total_revenue   = ("line_revenue",   "sum"),
                    total_cogs      = ("quantity",       lambda q: (q * items.loc[q.index,"cogs"]).sum()),
                )
                .reset_index()
                .rename(columns={col: "promo_id"})
            )

        perf = (
            pd.concat([_agg_by_promo("promo_id"), _agg_by_promo("promo_id_2")],
                      ignore_index=True)
            .groupby("promo_id", as_index=False)
            .sum()
        )

        perf = perf.merge(
            self.promos[["promo_id","promo_name","promo_type","discount_value",
                          "start_date","end_date","applicable_category",
                          "promo_channel","stackable_flag","min_order_value"]],
            on="promo_id", how="left"
        )
        perf["gross_profit"] = perf["total_revenue"] - perf["total_discount"] - perf["total_cogs"]
        perf["roi"]          = perf["total_revenue"] / perf["total_discount"].replace(0, np.nan)
        perf["profit_margin"]= perf["gross_profit"]  / perf["total_revenue"].replace(0, np.nan)
        perf["cost_per_order"]= perf["total_discount"] / perf["n_orders"].replace(0, np.nan)
        perf["duration_days"]= (perf["end_date"] - perf["start_date"]).dt.days.fillna(0)
        return perf.sort_values("roi", ascending=False).reset_index(drop=True)

    def _apply_timing_boost(self, score: pd.Series, z: TimingVar) -> pd.Series:
        """Boost score theo timing (z): double_day, q4, tet."""
        boost = 1.0
        if z.is_double_day : boost += 0.30
        if z.is_q4_event   : boost += 0.20
        if z.is_tet        : boost += 0.25
        if z.is_mid_year   : boost += 0.10
        if z.month in [11, 12, 1]: boost += 0.10  # high season
        return score * boost

    # ── public ───────────────────────────────────────────────────────────

    def rank_promos_by_roi(
        self,
        x: Optional[ProductVar]  = None,
        y: Optional[CustomerVar] = None,
        z: Optional[TimingVar]   = None,
        w: Optional[DiscountVar] = None,
        v: Optional[RegionVar]   = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Xếp hạng chiến dịch theo ROI, có filter theo x, y, z, w, v.
        Score tổng hợp = 0.4×roi_norm + 0.3×profit_norm + 0.3×timing_boost
        """
        df = self._perf.copy()

        # Filter x: category
        if x and x.category:
            df = df[
                df["applicable_category"].isna() |
                (df["applicable_category"] == x.category)
            ]
        # Filter x: segment (price_tier proxy)
        if x and x.price_tier:
            tier_map = {"low": (0, 200000), "mid": (200000, 500000), "high": (500000, np.inf)}
            lo, hi = tier_map.get(x.price_tier, (0, np.inf))

        # Filter w: discount depth
        if w and w.discount_value > 0:
            if w.promo_type == "percentage":
                df = df[df["discount_value"] <= w.discount_value + 5]

        # Filter z: active trong tháng
        if z and z.month:
            df = df[
                (df["start_date"].isna()) |
                (df["start_date"].dt.month <= z.month) &
                (df["end_date"].dt.month   >= z.month)
            ]

        # Normalize scores
        for col in ["roi", "profit_margin"]:
            col_min, col_max = df[col].min(), df[col].max()
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min + 1e-9)

        df["composite_score"] = 0.4 * df["roi_norm"] + 0.3 * df["profit_margin_norm"]
        if z:
            df["composite_score"] = self._apply_timing_boost(df["composite_score"], z)

        df["recommendation"] = df["composite_score"].apply(
            lambda s: "UU TIEN CAO" if s > 0.7 else "NEN CHAY" if s > 0.4 else "CAN NHAC"
        )
        return (
            df.sort_values("composite_score", ascending=False)
            .head(top_n)
            [["promo_id","promo_name","promo_type","discount_value",
              "n_orders","total_revenue","total_discount","gross_profit",
              "roi","profit_margin","cost_per_order","duration_days",
              "composite_score","recommendation"]]
            .reset_index(drop=True)
        )

    def optimize_discount_budget(
        self,
        x: Optional[ProductVar]  = None,
        y: Optional[CustomerVar] = None,
        z: Optional[TimingVar]   = None,
        w: Optional[DiscountVar] = None,
        v: Optional[RegionVar]   = None,
    ) -> Dict:
        """
        Phân bổ ngân sách discount tối ưu bằng LP:
            maximize: Σ (gross_profit_i / total_discount_i × alloc_i)
            subject to:
                Σ alloc_i ≤ w.budget_cap
                alloc_i ≥ 0
                alloc_i ≤ total_discount_i  (không vượt historical spend)

        Returns dict với: allocations DataFrame, total_allocated, expected_profit
        """
        ranked = self.rank_promos_by_roi(x, y, z, w, v)
        if ranked.empty:
            return {"note": "Không có chiến dịch phù hợp"}

        budget = w.budget_cap if w else np.inf
        if budget == np.inf:
            return {
                "allocations"     : ranked,
                "note"            : "Không có ràng buộc ngân sách — chạy tất cả top promos",
                "expected_profit" : ranked["gross_profit"].sum(),
            }

        n      = len(ranked)
        profit_rate = (ranked["gross_profit"] / ranked["total_discount"].replace(0, np.nan)).fillna(0).values
        c_obj  = -profit_rate  # minimize negative profit

        A_ub   = [np.ones(n)]
        b_ub   = [budget]
        bounds = [(0, row["total_discount"]) for _, row in ranked.iterrows()]

        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if result.success:
            ranked = ranked.copy()
            ranked["allocated_budget"] = np.round(result.x, 2)
            ranked["expected_profit"]  = (ranked["allocated_budget"] * profit_rate).round(2)
            return {
                "allocations"        : ranked,
                "total_allocated"    : round(result.x.sum(), 2),
                "expected_profit"    : round(-result.fun, 2),
                "budget_utilization" : round(result.x.sum() / budget, 4),
                "status"             : "optimal",
            }
        return {"note": result.message}

    def suggest_best_promo_next_month(
        self,
        target_month : int,
        x: Optional[ProductVar]  = None,
        y: Optional[CustomerVar] = None,
        w: Optional[DiscountVar] = None,
        v: Optional[RegionVar]   = None,
    ) -> pd.DataFrame:
        """
        Gợi ý top promo nên chạy vào tháng target_month.
        Tự động set timing flags (double_day, q4, tet) theo tháng.
        """
        z = TimingVar(
            month         = target_month,
            quarter       = (target_month - 1) // 3 + 1,
            is_double_day = target_month in [11, 12],
            is_q4_event   = target_month in [10, 11, 12],
            is_tet        = target_month in [1, 2],
            is_mid_year   = target_month in [6, 7],
            is_promo      = True,
        )
        return self.rank_promos_by_roi(x, y, z, w, v, top_n=5)

    def calc_revenue_lift(
        self,
        promo_id: str,
        x: Optional[ProductVar] = None,
        z: Optional[TimingVar]  = None,
    ) -> Dict:
        """
        Ước tính doanh thu tăng thêm nếu chạy promo_id trong kỳ z.
        Revenue lift = historical_avg_revenue_during_promo - baseline_revenue
        """
        row = self._perf[self._perf["promo_id"] == promo_id]
        if row.empty:
            return {"error": f"Không tìm thấy promo_id: {promo_id}"}

        row = row.iloc[0]
        duration = max(row["duration_days"], 1)
        daily_revenue_during = row["total_revenue"] / duration

        # baseline: average daily revenue không promo
        baseline_items = self.items[
            self.items["promo_id"].isna() & self.items["promo_id_2"].isna()
        ]
        baseline_orders = self.orders[
            self.orders["order_id"].isin(baseline_items["order_id"])
        ]
        baseline_daily = (
            baseline_items["line_revenue"].sum() /
            max((baseline_orders["order_date"].max() -
                 baseline_orders["order_date"].min()).days, 1)
        )

        lift_daily  = daily_revenue_during - baseline_daily
        timing_mult = 1.0
        if z:
            if z.is_double_day : timing_mult += 0.25
            if z.is_q4_event   : timing_mult += 0.15
            if z.is_tet        : timing_mult += 0.20

        projected_lift = lift_daily * duration * timing_mult

        return {
            "promo_id"            : promo_id,
            "promo_name"          : row["promo_name"],
            "duration_days"       : int(duration),
            "baseline_daily_rev"  : round(baseline_daily, 2),
            "promo_daily_rev"     : round(daily_revenue_during, 2),
            "lift_per_day"        : round(lift_daily, 2),
            "timing_multiplier"   : timing_mult,
            "projected_total_lift": round(projected_lift, 2),
            "total_discount_cost" : round(row["total_discount"], 2),
            "net_profit_impact"   : round(projected_lift * row["profit_margin"] -
                                          row["total_discount"], 2),
        }


# ------------------------------------------------------------------------------
# SECTION 5 — Customer Intelligence Optimizer
# ------------------------------------------------------------------------------

class CustomerIntelligenceOptimizer:
    """
    Tối ưu chiến lược dựa trên Customer Intelligence.

    Biến đầu vào:
        x = customer_segment  (RFM segment)
        y = acquisition_channel
        z = timing            (month, is_double_day, is_q4_event)
        w = discount_params   (depth, type, min_order_value)
        v = region            (East, Central, West / city tier)

    Outputs:
        1. Profit Optimization by Customer Group
        2. Discount Timing Optimization (Retention + CLV)
        3. Channel Acquisition Optimization (RCAV)
        4. Churn Prediction & Retention Cost Optimization
    """

    def __init__(
        self,
        rfm_engine    : RFMEngine,
        orders_df     : pd.DataFrame,
        order_items_df: pd.DataFrame,
        customers_df  : pd.DataFrame,
        geography_df  : pd.DataFrame,
    ):
        self.rfm       = rfm_engine
        self.orders    = orders_df.copy()
        self.items     = order_items_df.copy()
        self.customers = customers_df.copy()
        self.geo       = geography_df.copy()
        self._enriched = self._enrich_orders()

    def _enrich_orders(self) -> pd.DataFrame:
        """Join orders với customers, geography, RFM."""
        df = (
            self.orders
            .merge(self.customers[["customer_id","age_group","gender",
                                    "acquisition_channel","signup_date"]],
                   on="customer_id", how="left")
            .merge(self.geo[["zip","region","city"]],
                   on="zip", how="left")
            .merge(self.rfm.rfm[["customer_id","rfm_segment","monetary",
                                  "frequency","clv_estimated","churn_prob",
                                  "retention_discount_pct"]],
                   on="customer_id", how="left")
        )
        rev_map = (
            self.items.groupby("order_id")["line_revenue"].sum().reset_index()
        )
        df = df.merge(rev_map, on="order_id", how="left")
        df["city_tier"] = df["city"].apply(
            lambda c: "tier_1" if c in CITY_TIER_MAP["tier_1"]
                      else "tier_2" if c in CITY_TIER_MAP["tier_2"]
                      else "other"
        )
        return df

    # ── Output 1: Profit by Customer Group ───────────────────────────────

    def optimize_profit_by_segment(
        self,
        x: Optional[CustomerVar] = None,  # x = rfm_segment filter
        v: Optional[RegionVar]   = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Tối ưu hóa lợi nhuận theo nhóm khách hàng.
        Xác định segment nào có CLV/CAC ratio tốt nhất → ưu tiên đầu tư.

        Returns: DataFrame với profit_score, clv, churn_risk, priority
        """
        df = self._enriched.copy()
        if x and x.rfm_segment: df = df[df["rfm_segment"] == x.rfm_segment]
        if v and v.region      : df = df[df["region"]      == v.region]

        seg_summary = (
            df.groupby("rfm_segment")
            .agg(
                n_customers    = ("customer_id",        "nunique"),
                total_revenue  = ("line_revenue",       "sum"),
                avg_clv        = ("clv_estimated",      "mean"),
                avg_churn      = ("churn_prob",         "mean"),
                avg_frequency  = ("frequency",          "mean"),
                avg_monetary   = ("monetary",           "mean"),
            )
            .reset_index()
        )

        # Profit score = CLV × (1 - churn) × frequency
        seg_summary["profit_score"] = (
            seg_summary["avg_clv"] *
            (1 - seg_summary["avg_churn"]) *
            seg_summary["avg_frequency"]
        )
        # Normalize
        ps_max = seg_summary["profit_score"].max()
        seg_summary["profit_score_norm"] = (seg_summary["profit_score"] / ps_max).round(4)

        seg_summary["priority"] = seg_summary["profit_score_norm"].apply(
            lambda s: "UU TIEN CAO" if s > 0.7 else "UU TIEN VUA" if s > 0.4 else "UU TIEN THAP"
        )
        seg_summary["investment_recommendation"] = seg_summary.apply(
            lambda r: (
                f"Retention plan: discount {RETENTION_DISCOUNT_PCT.get(r['rfm_segment'],10)}% "
                f"for {int(r['n_customers'])} customers"
            ), axis=1
        )
        return seg_summary.sort_values("profit_score", ascending=False).head(top_n)

    # ── Output 2: Discount Timing (Retention + CLV) ───────────────────────

    def optimize_discount_timing(
        self,
        x: Optional[CustomerVar] = None,
        z: Optional[TimingVar]   = None,
        w: Optional[DiscountVar] = None,
        v: Optional[RegionVar]   = None,
    ) -> pd.DataFrame:
        """
        Tối ưu thi điểm discount kết hợp Retention + CLV.

        Logic:
            - High CLV + High Churn → aggressive discount ngay
            - High CLV + Low  Churn → loyalty reward nh
            - Low  CLV + High Churn → không đáng đầu tư nhiu
            - Low  CLV + Low  Churn → nurture dần

        Returns: Per-segment timing recommendation
        """
        df   = self._enriched.copy()
        rfm_sum = self.rfm.get_segment_summary()

        # Timing multiplier từ z
        timing_bonus = 0.0
        if z:
            if z.is_double_day : timing_bonus += 0.30
            if z.is_q4_event   : timing_bonus += 0.20
            if z.is_tet        : timing_bonus += 0.25
            if z.month in [6,7]: timing_bonus += 0.10

        rfm_sum["clv_tier"] = pd.qcut(rfm_sum["total_clv"], q=2, labels=["low", "high"])

        churn_col = "avg_churn_prob" if "avg_churn_prob" in rfm_sum.columns else "avg_churn"
        # qcut can fail if values are identical; fall back to median split.
        try:
            rfm_sum["churn_tier"] = pd.qcut(rfm_sum[churn_col], q=2, labels=["low", "high"])
        except Exception:
            med = rfm_sum[churn_col].median()
            rfm_sum["churn_tier"] = np.where(rfm_sum[churn_col] <= med, "low", "high")

        def _strategy(row):
            clv, churn = str(row["clv_tier"]), str(row["churn_tier"])
            base_disc  = row["avg_ret_discount"]
            if clv == "high" and churn == "high":
                strat = "Aggressive retention"
                disc = min(base_disc + 10 + timing_bonus * 10, 40)
                timing = "Within this month"
            elif clv == "high" and churn == "low":
                strat = "Loyalty reward"
                disc = base_disc + timing_bonus * 5
                timing = "End of quarter / event"
            elif clv == "low" and churn == "high":
                strat = "Low-cost re-engage"
                disc = base_disc * 0.7
                timing = "Run a small A/B test first"
            else:
                strat = "Nurture"
                disc = base_disc * 0.5 + timing_bonus * 3
                timing = "Off-peak month"
            return pd.Series({
                "strategy": strat,
                "recommended_disc": round(disc, 1),
                "best_timing": timing,
            })

        strategy_cols = rfm_sum.apply(_strategy, axis=1)
        return pd.concat([rfm_sum, strategy_cols], axis=1).sort_values("total_clv", ascending=False)

    # ── Output 3: Channel RCAV ────────────────────────────────────────────

    def optimize_acquisition_channel(
        self,
        y: Optional[CustomerVar] = None,
        z: Optional[TimingVar]   = None,
        v: Optional[RegionVar]   = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        RCAV = Revenue per Channel per Acquisition Visit.
        Xác định kênh acquisition nào sinh ra khách hàng có CLV cao nhất.

        Returns: channel ranking với rcav, avg_clv, n_customers, roi_score
        """
        df = self._enriched.copy()
        if v and v.region     : df = df[df["region"]      == v.region]
        if v and v.city_tier  : df = df[df["city_tier"]   == v.city_tier]
        if z and z.month:
            df = df[df["order_date"].dt.month == z.month]

        channel_stats = (
            df.groupby("acquisition_channel")
            .agg(
                n_customers    = ("customer_id",   "nunique"),
                total_revenue  = ("line_revenue",  "sum"),
                avg_clv        = ("clv_estimated", "mean"),
                avg_churn      = ("churn_prob",    "mean"),
                avg_frequency  = ("frequency",     "mean"),
                n_orders       = ("order_id",      "nunique"),
            )
            .reset_index()
        )

        channel_stats["rcav"] = (
            channel_stats["total_revenue"] / channel_stats["n_customers"].replace(0, np.nan)
        ).round(2)

        # ROI score = RCAV × (1 - avg_churn) × avg_frequency
        channel_stats["roi_score"] = (
            channel_stats["rcav"] *
            (1 - channel_stats["avg_churn"]) *
            channel_stats["avg_frequency"]
        )
        max_score = channel_stats["roi_score"].max()
        channel_stats["roi_score_norm"] = (channel_stats["roi_score"] / max_score).round(4)

        channel_stats["recommendation"] = channel_stats["roi_score_norm"].apply(
            lambda s: "TANG DAU TU" if s > 0.7 else "DUY TRI" if s > 0.4 else "XEM XET LAI"
        )

        if y and y.acquisition_channel:
            channel_stats = channel_stats[
                channel_stats["acquisition_channel"] == y.acquisition_channel
            ]

        return (
            channel_stats
            .sort_values("roi_score", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    # ── Output 4: Churn Prediction & Retention Cost ───────────────────────

    def optimize_retention_cost(
        self,
        x: Optional[CustomerVar] = None,
        w: Optional[DiscountVar] = None,
        v: Optional[RegionVar]   = None,
        simulation_runs: int     = 1000,
        retention_budget: float  = np.inf,
    ) -> Dict:
        """
        Monte Carlo simulation để tối ưu chi phí retention.

        Model:
            - Mỗi khách hàng có churn_prob p_c (từ RFM)
            - Nếu bị churn: lost_value = clv_estimated
            - Nếu áp dụng discount w: p_c giảm theo sigmoid
            - Chi phí retention = discount_value × monetary × n_customers_targeted

        Tối ưu:
            Tìm discount_depth d* sao cho:
                E[Revenue saved] - Cost(d*) được maximize

        Returns: dict với optimal_discount, expected_revenue_saved,
                 net_benefit, simulation_summary
        """
        rfm_df = self.rfm.rfm.copy()
        if x and x.rfm_segment:
            rfm_df = rfm_df[rfm_df["rfm_segment"] == x.rfm_segment]
        if v and v.region:
            enriched = self._enriched[self._enriched["region"] == v.region]
            rfm_df   = rfm_df[rfm_df["customer_id"].isin(enriched["customer_id"])]

        if rfm_df.empty:
            return {"note": "Không có khách hàng phù hợp"}

        # Discount depth range để simulate
        discount_depths = np.arange(0, 0.51, 0.05)  # 0% → 50%
        results = []

        rng = np.random.default_rng(seed=42)

        for d in discount_depths:
            # Churn reduction: sigmoid effect của discount
            # p_churn_after = p_churn × (1 - sigmoid(d × 10 - 2))
            from scipy.special import expit
            churn_reduction = expit(d * 10 - 2)
            p_after = rfm_df["churn_prob"] * (1 - churn_reduction * 0.7)

            # Monte Carlo
            saved_revenues = []
            for _ in range(simulation_runs):
                churned_before = rng.random(len(rfm_df)) < rfm_df["churn_prob"].values
                churned_after  = rng.random(len(rfm_df)) < p_after.values
                saved_mask     = churned_before & ~churned_after
                saved_rev      = rfm_df.loc[saved_mask.astype(bool), "clv_estimated"].sum()
                saved_revenues.append(saved_rev)

            avg_saved   = np.mean(saved_revenues)
            std_saved   = np.std(saved_revenues)
            retain_cost = d * rfm_df["monetary"].mean() * len(rfm_df)
            net_benefit = avg_saved - retain_cost

            results.append({
                "discount_depth"       : round(d * 100, 1),
                "avg_churn_before"     : round(rfm_df["churn_prob"].mean(), 4),
                "avg_churn_after"      : round(p_after.mean(), 4),
                "expected_revenue_saved": round(avg_saved, 2),
                "std_revenue_saved"    : round(std_saved, 2),
                "retention_cost"       : round(retain_cost, 2),
                "net_benefit"          : round(net_benefit, 2),
            })

        sim_df  = pd.DataFrame(results)
        best_row= sim_df.loc[sim_df["net_benefit"].idxmax()]

        # Apply budget constraint nếu có
        if retention_budget < np.inf:
            feasible = sim_df[sim_df["retention_cost"] <= retention_budget]
            if not feasible.empty:
                best_row = feasible.loc[feasible["net_benefit"].idxmax()]

        return {
            "optimal_discount_pct"    : best_row["discount_depth"],
            "expected_revenue_saved"  : best_row["expected_revenue_saved"],
            "retention_cost"          : best_row["retention_cost"],
            "net_benefit"             : best_row["net_benefit"],
            "churn_reduction"         : round(
                best_row["avg_churn_before"] - best_row["avg_churn_after"], 4
            ),
            "n_customers_targeted"    : len(rfm_df),
            "simulation_summary"      : sim_df,
            "note": (
                f"Discount tối ưu {best_row['discount_depth']}% → "
                f"tiết kiệm {best_row['expected_revenue_saved']:,.0f} VND, "
                f"chi phí {best_row['retention_cost']:,.0f} VND"
            ),
        }


# ------------------------------------------------------------------------------
# SECTION 6 — Master Optimizer (facade)
# ------------------------------------------------------------------------------

class Optimizer:
    """
    Facade class — gom tất cả optimizer lại.
    Dùng trong Streamlit pages và Gemini chatbot.

    Khởi tạo:
        from src.optimizer import Optimizer
        opt = Optimizer(data)           # data = load_all()

    Gi:
        # Tồn kho
        opt.inventory.suggest_reorder(x=ProductVar(category="Streetwear"))
        opt.inventory.flag_overstock()
        opt.inventory.optimize_inventory_value(budget=500_000_000)

        # Khuyến mãi
        opt.promotion.rank_promos_by_roi(z=TimingVar(month=11, is_double_day=True))
        opt.promotion.suggest_best_promo_next_month(target_month=12)
        opt.promotion.calc_revenue_lift("PROMO_001")

        # Customer Intelligence
        opt.customer.optimize_profit_by_segment()
        opt.customer.optimize_discount_timing(z=TimingVar(is_q4_event=True))
        opt.customer.optimize_acquisition_channel()
        opt.customer.optimize_retention_cost(simulation_runs=500)

        # Full run — tất cả outputs cùng lúc
        results = opt.run_all(request=OptimizeRequest(...))
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

        self.inventory = InventoryOptimizer(
            inventory_df = data["inventory"],
            products_df  = data["products"],
        )

        self.promotion = PromotionOptimizer(
            order_items_df = data["order_items"],
            promotions_df  = data["promotions"],
            orders_df      = data["orders"],
            products_df    = data["products"],
        )

        self._rfm_engine = RFMEngine(
            orders_df      = data["orders"],
            order_items_df = data["order_items"],
        )

        self.customer = CustomerIntelligenceOptimizer(
            rfm_engine     = self._rfm_engine,
            orders_df      = data["orders"],
            order_items_df = data["order_items"],
            customers_df   = data["customers"],
            geography_df   = data["geography"],
        )

    # ---------------------------------------------------------------------
    # Optimization Orchestration Layer (Decision vars + Constraints)
    # ---------------------------------------------------------------------

    def build_default_search_space(self) -> Dict[str, List[Any]]:
        """
        Build a reasonable (small) discrete search space from loaded data.
        Intended for interactive dashboard optimization, not exhaustive solving.
        """
        products = self.data.get("products")
        customers = self.data.get("customers")
        geography = self.data.get("geography")

        space: Dict[str, List[Any]] = {
            "x.category": sorted(products["category"].dropna().unique().tolist()) if products is not None else [],
            "x.segment": sorted(products["segment"].dropna().unique().tolist()) if products is not None else [],
            "x.size": sorted(products["size"].dropna().unique().tolist()) if products is not None else [],
            "x.color": sorted(products["color"].dropna().unique().tolist()) if products is not None else [],
            "y.age_group": sorted(customers["age_group"].dropna().unique().tolist()) if customers is not None else [],
            "y.gender": sorted(customers["gender"].dropna().unique().tolist()) if customers is not None else [],
            "y.acquisition_channel": sorted(customers["acquisition_channel"].dropna().unique().tolist()) if customers is not None else [],
            "y.rfm_segment": sorted(self._rfm_engine.rfm["rfm_segment"].dropna().unique().tolist())
            if hasattr(self, "_rfm_engine") and getattr(self._rfm_engine, "rfm", None) is not None
            else [],
            "z.month": list(range(1, 13)),
            "w.discount_value": [0, 5, 10, 15, 20, 25, 30],  # %
            "w.promo_type": ["percentage"],
            "w.budget_cap": [np.inf, 100_000_000, 250_000_000, 500_000_000, 1_000_000_000, 2_000_000_000],
            "v.region": sorted(geography["region"].dropna().unique().tolist()) if geography is not None else [],
            "v.city_tier": ["tier_1", "tier_2", "other"],
        }
        # Remove empties to avoid UI confusion
        return {k: v for k, v in space.items() if isinstance(v, list) and len(v) > 0}

    def build_optimization_cube(
        self,
        decision_variables: List[str],
        fixed_request: Optional[OptimizeRequest] = None,
        search_space_overrides: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 200_000,
        simulation_runs: int = 80,
        seed: int = 42,
        cube_path: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Precompute an "optimizer cube" so Streamlit can do:
          filter + reweight -> instant best result

        The cube stores:
          - decision variable values per row
          - raw metrics: revenue_value, profit_value, acquire_value, retain_value
          - normalized metrics (0..1) for fast scoring
        """
        fixed = deepcopy(fixed_request) if fixed_request is not None else OptimizeRequest()

        space = self.build_default_search_space()
        if search_space_overrides:
            space.update(search_space_overrides)

        vars_in_space = [v for v in (decision_variables or []) if v in space]
        if not vars_in_space:
            raise ValueError("decision_variables is empty or not present in search space.")

        sizes = [len(space[v]) for v in vars_in_space]
        total = int(np.prod(sizes)) if sizes else 0
        if total <= 0:
            raise ValueError("Empty search space for selected decision variables.")

        rng = np.random.default_rng(seed=seed)

        # If cube would be too large, sample combinations (still deterministic via seed).
        if total > max_combinations:
            indices = rng.choice(total, size=max_combinations, replace=False)
            indices = np.sort(indices)

            # Map flat indices -> coordinate indices
            radices = []
            acc = 1
            for s in reversed(sizes):
                radices.append(acc)
                acc *= s
            radices = list(reversed(radices))

            combos: List[Dict[str, Any]] = []
            for idx in indices.tolist():
                assign: Dict[str, Any] = {}
                remain = idx
                for var, size, radix in zip(vars_in_space, sizes, radices):
                    pos = (remain // radix) % size
                    assign[var] = space[var][int(pos)]
                combos.append(assign)
        else:
            combos = list(_iter_grid({v: space[v] for v in vars_in_space}))

        rows: List[Dict[str, Any]] = []
        for assign in combos:
            req = deepcopy(fixed)
            for k, val in assign.items():
                _set_attr_path(req, k, val)

            metrics = self._evaluate_request_light(
                req=req,
                objective_weights=DEFAULT_WEIGHTS,
                simulation_runs=simulation_runs,
            )
            rows.append({**assign, **metrics})

        cube = pd.DataFrame(rows)

        # Normalize for fast (vectorized) scoring
        for col in ["revenue_value", "profit_value", "acquire_value", "retain_value"]:
            x = np.log1p(cube[col].clip(lower=0))
            lo = float(x.min())
            hi = float(x.max())
            denom = (hi - lo) if (hi - lo) > 1e-12 else 1.0
            cube[f"{col}_norm"] = ((x - lo) / denom).round(6)

        # Precompute scenario scores so UI only filters + ranks (no reweight sliders)
        for scenario_name, w in SCENARIO_WEIGHTS.items():
            cube[f"score__{scenario_name}"] = (
                w.get("revenue", 0.0) * cube["revenue_value_norm"]
                + w.get("profit", 0.0) * cube["profit_value_norm"]
                + w.get("acquire", 0.0) * cube["acquire_value_norm"]
                + w.get("retain", 0.0) * cube["retain_value_norm"]
            ).round(6)

        # Default score column (Balanced)
        cube["score"] = cube["score__Balanced"]

        cube["combo_id"] = np.arange(1, len(cube) + 1, dtype=int)

        out_path = cube_path or (Path("outputs") / "optimizer_cube.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cube.to_parquet(out_path, index=False)

        return cube

    @staticmethod
    def score_cube(
        cube: pd.DataFrame,
        objective_weights: Dict[str, float],
    ) -> pd.Series:
        weights = dict(DEFAULT_WEIGHTS)
        weights.update(objective_weights or {})
        return (
            weights.get("revenue", 0.0) * cube["revenue_value_norm"]
            + weights.get("profit", 0.0) * cube["profit_value_norm"]
            + weights.get("acquire", 0.0) * cube["acquire_value_norm"]
            + weights.get("retain", 0.0) * cube["retain_value_norm"]
        )

    def _evaluate_request_light(
        self,
        req: OptimizeRequest,
        objective_weights: Dict[str, float],
        simulation_runs: int = 120,
    ) -> Dict[str, Any]:
        """
        Fast-ish evaluation used inside search loops.
        Returns score + key metrics, and keeps heavy tables out by default.
        """
        x, y, z, w, v = req.x, req.y, req.z, req.w, req.v

        inv_lp = self.inventory.optimize_inventory_value(x=x, z=z, budget=w.budget_cap)
        profit_df = self.customer.optimize_profit_by_segment(x=CustomerVar(rfm_segment=y.rfm_segment), v=v, top_n=10)
        acq_df = self.customer.optimize_acquisition_channel(y=CustomerVar(acquisition_channel=y.acquisition_channel), z=z, v=v, top_n=10)
        retention = self.customer.optimize_retention_cost(
            x=CustomerVar(rfm_segment=y.rfm_segment),
            w=DiscountVar(discount_value=w.discount_value, promo_type=w.promo_type),
            v=v,
            simulation_runs=simulation_runs,
        )

        revenue_value = _safe_float(inv_lp.get("total_revenue_potential"), 0.0)
        profit_value = _safe_float(profit_df["profit_score"].iloc[0], 0.0) if isinstance(profit_df, pd.DataFrame) and not profit_df.empty else 0.0
        acquire_value = _safe_float(acq_df["roi_score"].iloc[0], 0.0) if isinstance(acq_df, pd.DataFrame) and not acq_df.empty else 0.0
        retain_value = _safe_float(retention.get("net_benefit"), 0.0)

        weights = dict(DEFAULT_WEIGHTS)
        weights.update(objective_weights or {})
        # Log scaling for numeric stability across different magnitudes.
        score = (
            weights.get("revenue", 0.0) * np.log1p(max(revenue_value, 0.0))
            + weights.get("profit", 0.0) * np.log1p(max(profit_value, 0.0))
            + weights.get("acquire", 0.0) * np.log1p(max(acquire_value, 0.0))
            + weights.get("retain", 0.0) * np.log1p(max(retain_value, 0.0))
        )

        return {
            "score": float(score),
            "revenue_value": float(revenue_value),
            "profit_value": float(profit_value),
            "acquire_value": float(acquire_value),
            "retain_value": float(retain_value),
        }

    def optimize_strategy(self, problem: OptimizationProblem) -> Dict[str, Any]:
        """
        Main entry point for the prescriptive orchestrator.

        Returns:
          - best_request: OptimizeRequest (decision vars filled)
          - best_score + component metrics
          - best_outputs: full optimizer.run_all(best_request)
          - trace: DataFrame of evaluated candidates
        """
        space = self.build_default_search_space()
        space.update(problem.search_space or {})

        decision_vars = [v for v in (problem.decision_variables or []) if v in space]
        if not decision_vars:
            # Nothing to decide -> run fixed request directly
            fixed_req = deepcopy(problem.fixed_request)
            best_metrics = self._evaluate_request_light(fixed_req, problem.objective_weights)
            return {
                "best_request": fixed_req,
                "best_metrics": best_metrics,
                "best_outputs": self.run_all(fixed_req),
                "trace": pd.DataFrame([{"eval": 1, **best_metrics}]),
                "note": "No decision variables selected; used fixed constraints only.",
            }

        rng = np.random.default_rng(seed=problem.seed)
        current = deepcopy(problem.fixed_request)
        trace_rows: List[Dict[str, Any]] = []
        eval_count = 0

        def _eval_and_record(req: OptimizeRequest, meta: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal eval_count
            eval_count += 1
            metrics = self._evaluate_request_light(req, problem.objective_weights)
            row = {"eval": eval_count, **meta, **metrics}
            trace_rows.append(row)
            return metrics

        # Initialize baseline
        best_metrics = _eval_and_record(current, {"candidate_type": "baseline"})
        best_req = deepcopy(current)

        method = (problem.method or "coordinate_descent").strip().lower()
        max_evals = int(problem.max_evals or 250)

        if method == "grid":
            assignments = {k: space[k] for k in decision_vars}
            for assign in _iter_grid(assignments):
                if eval_count >= max_evals:
                    break
                req = deepcopy(problem.fixed_request)
                for k, val in assign.items():
                    _set_attr_path(req, k, val)
                metrics = _eval_and_record(req, {"candidate_type": "grid", **assign})
                if metrics["score"] > best_metrics["score"]:
                    best_metrics = metrics
                    best_req = deepcopy(req)
        else:
            # Coordinate descent: optimize one variable at a time (fast, interactive)
            rounds = max(int(problem.rounds or 1), 1)
            for r in range(rounds):
                improved_any = False
                for var in decision_vars:
                    if eval_count >= max_evals:
                        break

                    candidates = list(space.get(var, []))
                    # If too many candidates, sample to keep runtime bounded.
                    if len(candidates) > 25:
                        candidates = rng.choice(candidates, size=25, replace=False).tolist()

                    local_best_req = deepcopy(current)
                    local_best_metrics = best_metrics
                    for val in candidates:
                        if eval_count >= max_evals:
                            break
                        trial = deepcopy(current)
                        _set_attr_path(trial, var, val)
                        metrics = _eval_and_record(trial, {"candidate_type": "cd", "round": r + 1, "var": var, "val": val})
                        if metrics["score"] > local_best_metrics["score"]:
                            local_best_metrics = metrics
                            local_best_req = deepcopy(trial)

                    # Accept improvement for this coordinate
                    if local_best_metrics["score"] > best_metrics["score"]:
                        improved_any = True
                        best_metrics = local_best_metrics
                        best_req = deepcopy(local_best_req)
                        current = deepcopy(local_best_req)

                if not improved_any or eval_count >= max_evals:
                    break

        # Full outputs for the best strategy (may be heavier)
        best_outputs = self.run_all(best_req)
        trace = pd.DataFrame(trace_rows)

        return {
            "best_request": best_req,
            "best_metrics": best_metrics,
            "best_outputs": best_outputs,
            "trace": trace,
            "decision_variables": decision_vars,
            "method": method,
            "evals_used": eval_count,
        }

    def run_all(self, request: OptimizeRequest) -> Dict:
        """
        Chạy toàn bộ pipeline tối ưu với một OptimizeRequest.
        Trả v dict kết quả cho tất cả 4 outputs.
        """
        x, y, z, w, v = request.x, request.y, request.z, request.w, request.v

        # Convert ProductVar → CustomerVar nếu cần
        y_cust = CustomerVar(rfm_segment=y.rfm_segment if hasattr(y, "rfm_segment") else None,
                             acquisition_channel=y.acquisition_channel)

        return {
            # Output 1: Revenue & Inventory
            "reorder_suggestions"     : self.inventory.suggest_reorder(x=x, z=z),
            "overstock_flags"         : self.inventory.flag_overstock(x=x, z=z),
            "inventory_lp_result"     : self.inventory.optimize_inventory_value(
                                            x=x, z=z, budget=w.budget_cap),
            # Output 2: Profit Optimization
            "profit_by_segment"       : self.customer.optimize_profit_by_segment(v=v),
            # Output 3: Discount Timing
            "discount_timing"         : self.customer.optimize_discount_timing(z=z, w=w),
            "top_promos"              : self.promotion.rank_promos_by_roi(x=x, y=y_cust, z=z, w=w, v=v),
            "promo_budget_allocation" : self.promotion.optimize_discount_budget(x=x, z=z, w=w),
            # Output 4: Customer Acquisition & Retention
            "channel_rcav"            : self.customer.optimize_acquisition_channel(z=z, v=v),
            "retention_simulation"    : self.customer.optimize_retention_cost(
                                            x=y_cust, w=w, v=v, simulation_runs=500),
        }

    def get_gemini_context(self, request: OptimizeRequest) -> str:
        """
        Tạo text summary ngắn gn để inject vào Gemini chatbot prompt.
        Gemini dùng context này để trả li câu hi của user v tối ưu hóa.
        """
        results = self.run_all(request)

        reorder_top3 = results["reorder_suggestions"].head(3)
        top_promo    = results["top_promos"].head(1)
        best_channel = results["channel_rcav"].head(1)
        retention    = results["retention_simulation"]

        lines = [
            "=== KET QUA TOI UU ===",
            f"TOP 3 REORDER: {reorder_top3[['product_name','recommended_order_qty','action']].to_dict('records')}",
            f"TOP PROMO: {top_promo[['promo_name','roi','recommendation']].to_dict('records')}",
            f"BEST ACQUISITION CHANNEL: {best_channel[['acquisition_channel','rcav','recommendation']].to_dict('records')}",
            f"RETENTION: optimal discount {retention.get('optimal_discount_pct','N/A')}%, net benefit {retention.get('net_benefit','N/A'):,.0f} VND",
        ]
        return "\n".join(lines)

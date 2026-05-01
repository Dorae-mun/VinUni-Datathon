"""
src/data_loader.py
==================
Load, validate và merge toàn bộ CSV cho Datathon 2026.

Tính năng chính:
    - Cache bằng @st.cache_data (Streamlit) hoặc functools.lru_cache (notebook)
    - Tự động parse date columns
    - Validate quan hệ FK giữa các bảng
    - Cung cấp các hàm merge sẵn cho từng use-case (EDA, forecast, optimizer)

Cách dùng:
    # Trong Streamlit app
    from src.data_loader import load_all, load_sales_summary
    data = load_all()

    # Trong notebook
    from src.data_loader import load_all
    data = load_all()
    df_orders = data["orders"]
"""

import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_SRC_DIR  = Path(__file__).resolve().parent
ROOT_DIR  = _SRC_DIR.parent
DATA_DIR  = ROOT_DIR / "data"

# ── Date columns cho từng file ───────────────────────────────────────────────
_DATE_COLS: Dict[str, list] = {
    "products"    : [],
    "customers"   : ["signup_date"],
    "promotions"  : ["start_date", "end_date"],
    "geography"   : [],
    "orders"      : ["order_date"],
    "order_items" : [],
    "payments"    : [],
    "shipments"   : ["ship_date", "delivery_date"],
    "returns"     : ["return_date"],
    "reviews"     : ["review_date"],
    "sales"       : ["Date"],
    "sales_test"  : ["Date"],
    "sample_sub"  : ["Date"],
    "inventory"   : ["snapshot_date"],
    "web_traffic" : ["date"],
}

# ── File paths ───────────────────────────────────────────────────────────────
_FILE_MAP: Dict[str, str] = {
    "products"    : "products.csv",
    "customers"   : "customers.csv",
    "promotions"  : "promotions.csv",
    "geography"   : "geography.csv",
    "orders"      : "orders.csv",
    "order_items" : "order_items.csv",
    "payments"    : "payments.csv",
    "shipments"   : "shipments.csv",
    "returns"     : "returns.csv",
    "reviews"     : "reviews.csv",
    "sales"       : "sales.csv",
    "sales_test"  : "sales_test.csv",
    "sample_sub"  : "sample_submission.csv",
    "inventory"   : "inventory.csv",
    "web_traffic" : "web_traffic.csv",
}


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _read_csv(key: str, data_dir: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    """
    Đọc một file CSV, parse date columns, trả về DataFrame.
    Trả về None nếu file không tồn tại (ví dụ sales_test chưa có).
    """
    path = data_dir / _FILE_MAP[key]
    if not path.exists():
        warnings.warn(f"[data_loader] File không tồn tại, bỏ qua: {path}")
        return None

    date_cols = _DATE_COLS.get(key, [])
    df = pd.read_csv(
        path,
        parse_dates=date_cols if date_cols else False,
        low_memory=False,
    )

    # Lowercase tên cột cho nhất quán
    df.columns = df.columns.str.strip()
    return df


def _validate_fk(
    child_df: pd.DataFrame,
    child_col: str,
    parent_df: pd.DataFrame,
    parent_col: str,
    relation_name: str,
) -> None:
    """Kiểm tra FK integrity, warning nếu có orphan records."""
    if child_df is None or parent_df is None:
        return
    orphans = ~child_df[child_col].isin(parent_df[parent_col])
    n_orphans = orphans.sum()
    if n_orphans > 0:
        warnings.warn(
            f"[data_loader] FK warning ({relation_name}): "
            f"{n_orphans} orphan records trong '{child_col}'"
        )


# ════════════════════════════════════════════════════════════════════════════
# Public: load từng bảng riêng lẻ
# ════════════════════════════════════════════════════════════════════════════

def load_products(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("products", data_dir)
    # Tính gross_margin sẵn luôn
    df["gross_margin"] = (df["price"] - df["cogs"]) / df["price"]
    return df


def load_customers(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("customers", data_dir)


def load_promotions(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("promotions", data_dir)


def load_geography(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("geography", data_dir)


def load_orders(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("orders", data_dir)
    # Thêm cột thời gian tiện cho EDA
    df["order_year"]    = df["order_date"].dt.year
    df["order_month"]   = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_dow"]     = df["order_date"].dt.day_name()   # Monday, Tuesday...
    return df


def load_order_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("order_items", data_dir)
    # Revenue và profit cho từng dòng
    df["line_revenue"] = df["quantity"] * df["unit_price"] - df["discount_amount"]
    df["has_promo"]    = df["promo_id"].notna().astype(int)
    return df


def load_payments(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("payments", data_dir)


def load_shipments(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("shipments", data_dir)
    # Tính delivery lead time (ngày)
    df["lead_time_days"] = (df["delivery_date"] - df["ship_date"]).dt.days
    return df


def load_returns(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("returns", data_dir)


def load_reviews(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("reviews", data_dir)


def load_sales(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("sales", data_dir)
    df = df.sort_values("Date").reset_index(drop=True)
    df["gross_profit"] = df["Revenue"] - df["COGS"]
    df["gross_margin"] = df["gross_profit"] / df["Revenue"]
    # Thêm cột thời gian
    df["year"]    = df["Date"].dt.year
    df["month"]   = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week"]    = df["Date"].dt.isocalendar().week.astype(int)
    df["dow"]     = df["Date"].dt.dayofweek          # 0=Mon
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    return df


def load_sales_test(data_dir: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    df = _read_csv("sales_test", data_dir)
    if df is not None:
        df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_sample_submission(data_dir: Path = DATA_DIR) -> Optional[pd.DataFrame]:
    return _read_csv("sample_sub", data_dir)


def load_inventory(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return _read_csv("inventory", data_dir)


def load_web_traffic(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_csv("web_traffic", data_dir)
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# Public: load_all — entry point chính
# ════════════════════════════════════════════════════════════════════════════

def load_all(data_dir: Path = DATA_DIR, validate: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load toàn bộ 15 file CSV, trả về dict[str, DataFrame].

    Parameters
    ----------
    data_dir : Path
        Thư mục chứa CSV. Mặc định = ROOT/data/
    validate : bool
        Nếu True, kiểm tra FK integrity và in warning nếu có lỗi.

    Returns
    -------
    dict với các keys:
        products, customers, promotions, geography,
        orders, order_items, payments, shipments, returns, reviews,
        sales, sales_test, sample_sub, inventory, web_traffic

    Ví dụ
    -----
        data = load_all()
        data["sales"].head()
        data["orders"].query("order_status == 'delivered'")
    """
    print("[data_loader] Loading data...")

    data = {
        # Master
        "products"    : load_products(data_dir),
        "customers"   : load_customers(data_dir),
        "promotions"  : load_promotions(data_dir),
        "geography"   : load_geography(data_dir),
        # Transaction
        "orders"      : load_orders(data_dir),
        "order_items" : load_order_items(data_dir),
        "payments"    : load_payments(data_dir),
        "shipments"   : load_shipments(data_dir),
        "returns"     : load_returns(data_dir),
        "reviews"     : load_reviews(data_dir),
        # Analytical
        "sales"       : load_sales(data_dir),
        "sales_test"  : load_sales_test(data_dir),
        "sample_sub"  : load_sample_submission(data_dir),
        # Operational
        "inventory"   : load_inventory(data_dir),
        "web_traffic" : load_web_traffic(data_dir),
    }

    if validate:
        _run_validation(data)

    # Tóm tắt nhanh
    for key, df in data.items():
        if df is not None:
            print(f"  [ok] {key:<15} {df.shape[0]:>8,} rows x {df.shape[1]:>2} cols")
        else:
            print(f"  [missing] {key:<15} file not found")

    print("[data_loader] Done.\n")
    return data


def _run_validation(data: Dict[str, pd.DataFrame]) -> None:
    """Kiểm tra FK integrity cho các quan hệ chính."""
    checks = [
        # (child_table, child_col, parent_table, parent_col, label)
        ("orders",      "customer_id", "customers", "customer_id", "orders→customers"),
        ("orders",      "zip",         "geography", "zip",         "orders→geography"),
        ("order_items", "order_id",    "orders",    "order_id",    "order_items→orders"),
        ("order_items", "product_id",  "products",  "product_id",  "order_items→products"),
        ("payments",    "order_id",    "orders",    "order_id",    "payments→orders"),
        ("shipments",   "order_id",    "orders",    "order_id",    "shipments→orders"),
        ("returns",     "order_id",    "orders",    "order_id",    "returns→orders"),
        ("returns",     "product_id",  "products",  "product_id",  "returns→products"),
        ("reviews",     "order_id",    "orders",    "order_id",    "reviews→orders"),
        ("reviews",     "product_id",  "products",  "product_id",  "reviews→products"),
        ("inventory",   "product_id",  "products",  "product_id",  "inventory→products"),
    ]
    for child_key, child_col, parent_key, parent_col, label in checks:
        _validate_fk(
            data.get(child_key),
            child_col,
            data.get(parent_key),
            parent_col,
            label,
        )


# ════════════════════════════════════════════════════════════════════════════
# Public: merged DataFrames cho từng use-case
# ════════════════════════════════════════════════════════════════════════════

def get_orders_enriched(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    orders + customers + geography + payments + shipments (1 dòng / đơn hàng).

    Dùng cho: EDA tổng quan, phân tích theo region/channel/device.
    """
    df = (
        data["orders"]
        .merge(data["customers"][["customer_id", "age_group", "gender",
                                   "acquisition_channel", "signup_date"]],
               on="customer_id", how="left")
        .merge(data["geography"][["zip", "city", "region", "district"]],
               on="zip", how="left")
        .merge(data["payments"][["order_id", "payment_value", "installments"]],
               on="order_id", how="left")
        .merge(data["shipments"][["order_id", "ship_date", "delivery_date",
                                   "shipping_fee", "lead_time_days"]],
               on="order_id", how="left")
    )
    return df


def get_order_items_enriched(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    order_items + products + orders (order_date, order_status, region).

    Dùng cho: phân tích doanh thu theo sản phẩm/danh mục/phân khúc.
    """
    df = (
        data["order_items"]
        .merge(data["products"][["product_id", "product_name", "category",
                                  "segment", "size", "color", "price", "cogs",
                                  "gross_margin"]],
               on="product_id", how="left")
        .merge(data["orders"][["order_id", "order_date", "order_status",
                                "order_year", "order_month", "order_quarter",
                                "customer_id", "zip"]],
               on="order_id", how="left")
        .merge(data["geography"][["zip", "region", "city"]],
               on="zip", how="left")
    )
    # Lợi nhuận gộp từng dòng
    df["line_cogs"]   = df["quantity"] * df["cogs"]
    df["line_profit"] = df["line_revenue"] - df["line_cogs"]
    return df


def get_returns_enriched(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    returns + products + orders.

    Dùng cho: phân tích tỷ lệ trả hàng theo danh mục / kích thước / lý do.
    """
    return (
        data["returns"]
        .merge(data["products"][["product_id", "product_name", "category",
                                  "segment", "size", "color"]],
               on="product_id", how="left")
        .merge(data["orders"][["order_id", "order_date", "order_year",
                                "order_month", "customer_id"]],
               on="order_id", how="left")
    )


def get_promo_performance(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Hiệu quả từng chương trình khuyến mãi:
        - Tổng discount_amount đã giảm
        - Tổng revenue sau giảm
        - Số đơn áp dụng
        - ROI = revenue / discount_cost

    Dùng cho: Prescriptive — tối ưu lịch khuyến mãi.
    """
    items = data["order_items"].copy()

    # Gộp theo promo_id (xét cả promo_id và promo_id_2)
    p1 = items[items["promo_id"].notna()].groupby("promo_id").agg(
        n_items        = ("order_id", "count"),
        total_discount = ("discount_amount", "sum"),
        total_revenue  = ("line_revenue", "sum"),
    ).reset_index().rename(columns={"promo_id": "promo_id_key"})

    p2 = items[items["promo_id_2"].notna()].groupby("promo_id_2").agg(
        n_items        = ("order_id", "count"),
        total_discount = ("discount_amount", "sum"),
        total_revenue  = ("line_revenue", "sum"),
    ).reset_index().rename(columns={"promo_id_2": "promo_id_key"})

    perf = (
        pd.concat([p1, p2], ignore_index=True)
        .groupby("promo_id_key", as_index=False)
        .sum()
    )

    perf = perf.merge(
        data["promotions"][["promo_id", "promo_name", "promo_type",
                             "discount_value", "start_date", "end_date",
                             "applicable_category"]],
        left_on="promo_id_key", right_on="promo_id", how="left"
    )
    perf["roi"] = perf["total_revenue"] / perf["total_discount"].replace(0, np.nan)
    return perf.sort_values("roi", ascending=False).reset_index(drop=True)


def get_inventory_enriched(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    inventory + products (đã có sẵn category/segment trong inventory,
    merge thêm price & cogs để tính inventory value).

    Dùng cho: Prescriptive — tối ưu tồn kho.
    """
    return (
        data["inventory"]
        .merge(data["products"][["product_id", "price", "cogs", "gross_margin"]],
               on="product_id", how="left")
    )


def get_web_sales_daily(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Ghép web_traffic (daily) với sales (daily) theo ngày.

    Dùng cho: phân tích leading indicator traffic → revenue.
    """
    traffic = data["web_traffic"].copy()
    sales   = data["sales"][["Date", "Revenue", "COGS", "gross_profit"]].copy()
    return traffic.merge(sales, left_on="date", right_on="Date", how="left")


# ════════════════════════════════════════════════════════════════════════════
# Public: KPI summaries
# ════════════════════════════════════════════════════════════════════════════

def load_sales_summary(sales_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Tính các KPI tổng quan từ sales DataFrame.

    Returns
    -------
    dict với các keys:
        total_revenue, total_cogs, gross_profit, gross_margin_pct,
        date_start, date_end, n_days, avg_daily_revenue,
        best_day, best_day_revenue, worst_day, worst_day_revenue
    """
    if sales_df is None:
        sales_df = load_sales()

    df = sales_df.copy()
    total_rev   = df["Revenue"].sum()
    total_cogs  = df["COGS"].sum()
    gross_profit = total_rev - total_cogs

    best_idx  = df["Revenue"].idxmax()
    worst_idx = df["Revenue"].idxmin()

    return {
        "total_revenue"      : total_rev,
        "total_cogs"         : total_cogs,
        "gross_profit"       : gross_profit,
        "gross_margin_pct"   : gross_profit / total_rev * 100,
        "date_start"         : df["Date"].min(),
        "date_end"           : df["Date"].max(),
        "n_days"             : len(df),
        "avg_daily_revenue"  : df["Revenue"].mean(),
        "best_day"           : df.loc[best_idx, "Date"],
        "best_day_revenue"   : df.loc[best_idx, "Revenue"],
        "worst_day"          : df.loc[worst_idx, "Date"],
        "worst_day_revenue"  : df.loc[worst_idx, "Revenue"],
    }


def get_monthly_sales(sales_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate sales theo tháng.

    Returns DataFrame với: year, month, period (YYYY-MM),
                            Revenue, COGS, gross_profit, gross_margin
    """
    if sales_df is None:
        sales_df = load_sales()

    df = sales_df.copy()
    monthly = (
        df.groupby(["year", "month"], as_index=False)
        .agg(
            Revenue      = ("Revenue", "sum"),
            COGS         = ("COGS", "sum"),
            n_days       = ("Date", "count"),
        )
    )
    monthly["gross_profit"] = monthly["Revenue"] - monthly["COGS"]
    monthly["gross_margin"] = monthly["gross_profit"] / monthly["Revenue"]
    monthly["period"]       = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    )
    return monthly.sort_values("period").reset_index(drop=True)


def get_revenue_by_region(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Tổng revenue theo region (join orders → geography → order_items).

    Dùng cho: Q7 MCQ + bản đồ doanh thu theo vùng.
    """
    enriched = get_order_items_enriched(data)
    return (
        enriched
        .query("order_status != 'cancelled'")
        .groupby("region", as_index=False)
        .agg(total_revenue=("line_revenue", "sum"),
             n_orders=("order_id", "nunique"))
        .sort_values("total_revenue", ascending=False)
    )


# ════════════════════════════════════════════════════════════════════════════
# Streamlit-aware wrapper (cache nếu đang chạy trong Streamlit)
# ════════════════════════════════════════════════════════════════════════════

def get_cached_data() -> Dict[str, pd.DataFrame]:
    """
    Dùng trong Streamlit app — tự động dùng st.cache_data nếu có.

    Ví dụ trong app/main.py:
        from src.data_loader import get_cached_data
        data = get_cached_data()
    """
    try:
        import streamlit as st

        @st.cache_data(show_spinner="Loading data...")
        def _cached():
            return load_all(validate=False)

        return _cached()

    except ImportError:
        # Chạy ngoài Streamlit (notebook, script)
        return load_all(validate=False)


# ════════════════════════════════════════════════════════════════════════════
# Quick test khi chạy trực tiếp: python -m src.data_loader
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = load_all(validate=True)
    summary = load_sales_summary(data["sales"])

    print("=" * 50)
    print("SALES SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<25}: {v}")

    print("\nMONTHLY SALES (5 dòng đầu):")
    print(get_monthly_sales(data["sales"]).head())

    print("\nREVENUE BY REGION:")
    print(get_revenue_by_region(data))

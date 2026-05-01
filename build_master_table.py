#!/usr/bin/env python3
"""
build_master_table.py
=======================
Build master analytics table for prescriptive dashboard.
"""

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_all


def build_master_table() -> pl.DataFrame:
    """Build comprehensive master analytics table from raw data."""

    print("[START] Building Master Analytics Table...")
    print("=" * 70)

    print("[1/5] Loading data...")
    data = load_all()
    print("[OK] Data loaded")

    print("[2/5] Converting to Polars DataFrames...")
    customers_df = pl.from_pandas(data["customers"])
    orders_df = pl.from_pandas(data["orders"])
    order_items_df = pl.from_pandas(data["order_items"])
    products_df = pl.from_pandas(data["products"])
    geography_df = pl.from_pandas(data["geography"])
    reviews_df = pl.from_pandas(data["reviews"])
    print("[OK] DataFrames converted")

    if "rfm_segment" not in customers_df.columns:
        customers_df = customers_df.with_columns(pl.lit("Regular").alias("rfm_segment"))

    dimensions = {
        "category": products_df["category"].unique().to_list(),
        "region": geography_df["region"].unique().to_list(),
        "rfm_segment": ["Champions", "Loyal customers", "Churn risk", "Regular customers"],
        "month": list(range(1, 13)),
        "discount": [0.0, 0.05, 0.10, 0.15, 0.20],
    }

    print("\n[INFO] Dimensions:")
    for dim, values in dimensions.items():
        print(f"   {dim}: {len(values)} values")

    total_combinations = 1
    for values in dimensions.values():
        total_combinations *= len(values)
    print(f"\n[INFO] Total combinations: {total_combinations:,}")

    print("\n[3/5] Building master table from transaction data...")

    # RFM proxy if segment is unavailable: based on order count per customer.
    if customers_df["rfm_segment"].n_unique() <= 1:
        customer_orders = (
            orders_df.group_by("customer_id")
            .agg(pl.n_unique("order_id").alias("n_orders"))
            .with_columns([
                pl.when(pl.col("n_orders") >= 8).then(pl.lit("Champions"))
                .when(pl.col("n_orders") >= 5).then(pl.lit("Loyal customers"))
                .when(pl.col("n_orders") <= 1).then(pl.lit("Churn risk"))
                .otherwise(pl.lit("Regular customers"))
                .alias("rfm_segment")
            ])
        )
    else:
        customer_orders = (
            customers_df
            .select(["customer_id", "rfm_segment"])
            .with_columns(
                pl.col("rfm_segment")
                .replace(
                    {
                        "Loyal": "Loyal customers",
                        "Churn Risk": "Churn risk",
                        "Regular": "Regular customers",
                    }
                )
                .alias("rfm_segment")
            )
        )

    reviews_agg = (
        reviews_df.group_by(["order_id", "product_id"])
        .agg(pl.mean("rating").alias("customer_satisfaction"))
    )

    master_df = (
        order_items_df
        .join(orders_df.select(["order_id", "order_date", "customer_id", "zip", "order_status"]), on="order_id", how="left")
        .join(products_df.select(["product_id", "category", "cogs"]), on="product_id", how="left")
        .join(customer_orders.select(["customer_id", "rfm_segment"]), on="customer_id", how="left")
        .join(geography_df.select(["zip", "region"]), on="zip", how="left")
        .join(reviews_agg, on=["order_id", "product_id"], how="left")
        .filter(pl.col("order_status") != "cancelled")
        .with_columns([
            pl.col("order_date").dt.month().alias("month"),
            pl.when((pl.col("quantity") * pl.col("unit_price")) > 0)
            .then(pl.col("discount_amount") / (pl.col("quantity") * pl.col("unit_price")))
            .otherwise(0.0)
            .clip(0.0, 1.0)
            .alias("discount_raw"),
            pl.col("line_revenue").alias("revenue"),
            (pl.col("quantity") * pl.col("cogs")).alias("line_cogs"),
            (pl.col("line_revenue") - pl.col("quantity") * pl.col("cogs")).alias("profit"),
            pl.col("discount_amount").alias("acquisition_cost"),
            (pl.col("line_revenue") * 0.01).alias("retention_cost"),
            pl.col("customer_satisfaction"),
        ])
        .with_columns([
            pl.when(pl.col("discount_raw") < 0.025).then(0.00)
            .when(pl.col("discount_raw") < 0.075).then(0.05)
            .when(pl.col("discount_raw") < 0.125).then(0.10)
            .when(pl.col("discount_raw") < 0.175).then(0.15)
            .otherwise(0.20)
            .alias("discount")
        ])
    )

    aggregated_df = (
        master_df
        .group_by(["category", "region", "rfm_segment", "month", "discount"])
        .agg([
            pl.sum("revenue").alias("revenue_value"),
            pl.sum("profit").alias("profit_value"),
            pl.sum("acquisition_cost").alias("acquire_value"),
            pl.sum("retention_cost").alias("retain_value"),
            pl.count("order_id").alias("order_count"),
            pl.mean("customer_satisfaction").alias("avg_satisfaction"),
        ])
        .with_columns([
            pl.when(pl.col("acquire_value") > 0)
            .then(pl.col("profit_value") / pl.col("acquire_value"))
            .otherwise(0.0)
            .alias("roi"),
            pl.arange(0, pl.len()).alias("combo_id"),
        ])
    )

    print(f"   Shape: {aggregated_df.shape[0]:,} rows x {aggregated_df.shape[1]} columns")

    print("\n[4/5] Computing normalized metrics...")
    max_revenue = aggregated_df.select(pl.col("revenue_value").max()).item() or 0.0
    max_profit = aggregated_df.select(pl.col("profit_value").max()).item() or 0.0
    max_roi = aggregated_df.select(pl.col("roi").max()).item() or 0.0

    df_with_stats = aggregated_df.with_columns([
        pl.when(pl.lit(max_revenue) > 0).then(pl.col("revenue_value") / max_revenue).otherwise(0.0).alias("revenue_norm"),
        pl.when(pl.lit(max_profit) > 0).then(pl.col("profit_value") / max_profit).otherwise(0.0).alias("profit_norm"),
        pl.when(pl.lit(max_roi) > 0).then(pl.col("roi") / max_roi).otherwise(0.0).alias("roi_norm"),
    ])

    output_path = Path("outputs") / "master_dashboard.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[5/5] Saving to {output_path}...")
    df_with_stats.write_parquet(output_path)

    file_size_mb = output_path.stat().st_size / 1024 / 1024

    print()
    print("=" * 70)
    print("[OK] Master table built successfully!")
    print()
    print("[INFO] Summary:")
    print(f"   Rows: {df_with_stats.shape[0]:,}")
    print(f"   Columns: {df_with_stats.shape[1]}")
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Location: {output_path}")
    print()
    print("[INFO] Available for filtering:")
    print(f"   - category: {df_with_stats['category'].unique().len()} unique values")
    print(f"   - region: {df_with_stats['region'].unique().len()} unique values")
    print(f"   - rfm_segment: {df_with_stats['rfm_segment'].unique().len()} unique values")
    print(f"   - month: {df_with_stats['month'].unique().len()} unique values")
    print(f"   - discount: {df_with_stats['discount'].unique().len()} unique values")
    print()
    print("[READY] Dashboard can be launched")
    print("   Command: streamlit run app/main.py")
    print("=" * 70)

    return df_with_stats


if __name__ == "__main__":
    try:
        _ = build_master_table()
    except Exception as e:
        print()
        print("[ERROR] Error building master table:")
        print(f"   {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

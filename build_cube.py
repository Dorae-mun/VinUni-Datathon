#!/usr/bin/env python3
"""
build_cube.py
==============
Offline cube builder for Datathon 2026 Prescriptive Dashboard

Builds the optimization cube parquet file for instant strategy results.
Run this script to precompute all combinations before starting the dashboard.

Usage:
    python build_cube.py

Output:
    outputs/optimizer_cube.parquet
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import get_cached_data
from optimizer import Optimizer

def main():
    print("🚀 Starting cube build...")
    print("=" * 60)

    # Load data
    print("📊 Loading data...")
    data = get_cached_data()
    print("✅ Data loaded successfully")

    # Initialize optimizer
    print("⚙️  Initializing optimizer...")
    optimizer = Optimizer(data)
    print("✅ Optimizer initialized")

    # Build cube with reduced combinations for faster processing
    print("🔨 Building optimization cube...")
    print("   This may take 3-5 minutes depending on your CPU...")
    print()

    cube_path = Path("outputs") / "optimizer_cube.parquet"

    # Reduced combinations for manageable build time
    max_combinations = 5_000  # Further reduced for speed
    simulation_runs = 40  # Reduced from 60

    cube_dims = [
        "x.category",
        "v.region", 
        "y.rfm_segment",
        "z.month",
        "w.discount_value"
    ]

    try:
        cube_df = optimizer.build_optimization_cube(
            decision_variables=cube_dims,
            fixed_request=None,
            max_combinations=max_combinations,
            simulation_runs=simulation_runs,
            cube_path=cube_path,
        )
        
        print()
        print("=" * 60)
        print("✅ Cube built successfully!")
        print(f"   Location: {cube_path}")
        print(f"   Rows: {len(cube_df):,}")
        print(f"   Columns: {len(cube_df.columns)}")
        print(f"   File size: {cube_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()
        print("📊 Dimensions:")
        for dim in cube_dims:
            print(f"   • {dim}")
        print()
        print("🚀 Dashboard is ready to run!")
        print("   Command: streamlit run app/main.py")
        print("=" * 60)

    except Exception as e:
        print()
        print("❌ Error building cube:")
        print(f"   {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
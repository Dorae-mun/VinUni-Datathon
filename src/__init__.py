"""
src/ — Business Logic Package
==============================
Datathon 2026 — The Gridbreakers
Hosted by VinTelligence | VinUniversity DS&AI Club

Cấu trúc package:
    data_loader.py        — Load & cache toàn bộ CSV, merge các bảng
    feature_engineering.py— Tạo features cho mô hình dự báo
    model.py              — Train, load, predict (LightGBM / Prophet)
    metrics.py            — MAE, RMSE, R² evaluation
    optimizer.py          — Prescriptive: tối ưu tồn kho & khuyến mãi

Cách dùng nhanh:
    from src import load_all, get_sales_summary

    data = load_all()           # dict chứa tất cả DataFrame
    summary = get_sales_summary(data["sales"])
"""

# ── Standard library ────────────────────────────────────────────────────────
import warnings
from pathlib import Path

# ── Suppress noisy warnings ─────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Package metadata ─────────────────────────────────────────────────────────
__version__   = "1.0.0"
__authors__   = ["Your Team Name"]
__email__     = ""
__project__   = "Datathon 2026 — The Gridbreakers"

# ── Root & data paths (tự động resolve dù chạy từ đâu) ──────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # project root
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Tạo thư mục nếu chưa tồn tại
for _dir in [MODEL_DIR, OUTPUT_DIR, OUTPUT_DIR / "figures"]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Data file paths (import ở module khác cho tiện) ──────────────────────────
DATA_FILES = {
    # Master tables
    "products"    : DATA_DIR / "products.csv",
    "customers"   : DATA_DIR / "customers.csv",
    "promotions"  : DATA_DIR / "promotions.csv",
    "geography"   : DATA_DIR / "geography.csv",
    # Transaction tables
    "orders"      : DATA_DIR / "orders.csv",
    "order_items" : DATA_DIR / "order_items.csv",
    "payments"    : DATA_DIR / "payments.csv",
    "shipments"   : DATA_DIR / "shipments.csv",
    "returns"     : DATA_DIR / "returns.csv",
    "reviews"     : DATA_DIR / "reviews.csv",
    # Analytical tables
    "sales"       : DATA_DIR / "sales.csv",
    "sales_test"  : DATA_DIR / "sales_test.csv",
    "sample_sub"  : DATA_DIR / "sample_submission.csv",
    # Operational tables
    "inventory"   : DATA_DIR / "inventory.csv",
    "web_traffic" : DATA_DIR / "web_traffic.csv",
}

# ── Lazy imports (chỉ import khi thực sự cần, tránh lỗi nếu thiếu thư viện) ──
def load_all():
    """
    Shortcut: load toàn bộ dữ liệu.
    Trả về dict[str, pd.DataFrame].

    Ví dụ:
        data = load_all()
        df_orders = data["orders"]
    """
    from src.data_loader import load_all as _load_all
    return _load_all()


def get_sales_summary(sales_df=None):
    """
    Shortcut: tính summary nhanh cho sales DataFrame.
    Nếu không truyền vào, tự load sales.csv.

    Trả về dict chứa các KPI cơ bản:
        total_revenue, total_cogs, gross_profit, date_range, n_days
    """
    from src.data_loader import load_sales_summary
    return load_sales_summary(sales_df)


def get_model():
    """
    Shortcut: load model đã train từ models/lgbm_model.pkl.
    Raise FileNotFoundError nếu chưa train.
    """
    from src.model import load_model
    return load_model()


def get_optimizer():
    """
    Shortcut: khởi tạo Optimizer với dữ liệu inventory & promotions.
    """
    from src.optimizer import Optimizer
    data = load_all()
    return Optimizer(data)


# ── Public API ────────────────────────────────────────────────────────────────
__all__ = [
    # Paths
    "ROOT_DIR",
    "DATA_DIR",
    "MODEL_DIR",
    "OUTPUT_DIR",
    "DATA_FILES",
    # Shortcuts
    "load_all",
    "get_sales_summary",
    "get_model",
    "get_optimizer",
    # Meta
    "__version__",
    "__project__",
]

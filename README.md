# 🏆 VinUni Datathon 2026 — Vòng 1

<div align="center">

**Nhóm:** K-TEAM  
**Nhiệm vụ:** Dự báo doanh thu (`Revenue`) và giá vốn hàng bán (`COGS`) hàng ngày cho doanh nghiệp thời trang thương mại điện tử Việt Nam  
**Phạm vi dự báo:** 01/01/2023 → 01/07/2024 (548 ngày)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM%20%2F%20XGBoost-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## 📋 Tổng quan

Hệ thống phân tích đầu cuối (end-to-end) bao gồm bốn tầng phân tích:

| Tầng | Mô tả |
|------|-------|
| 📊 **Descriptive** | Thống kê mô tả, phân tích xu hướng doanh thu & COGS |
| 🔍 **Diagnostic** | Chẩn đoán nguyên nhân biến động, phân tích mùa vụ |
| 🔮 **Predictive** | Dự báo chuỗi thời gian với mô hình ensemble có trọng số tối ưu |
| 💡 **Prescriptive** | Đề xuất chiến lược tồn kho, khuyến mãi dựa trên dự báo |

Hệ thống được triển khai trên **Streamlit**, tích hợp **RAG Chatbot** hỗ trợ Q&A dựa trên tài liệu nội bộ, và pipeline dự báo với khả năng giải thích (SHAP / Feature Importance).

---

## 🗂️ Cấu trúc thư mục

```text
.
├── app/                              # Giao diện dashboard Streamlit
│   ├── main.py                       # Entry point
│   ├── components/
│   │   ├── charts.py                 # Biểu đồ tuỳ chỉnh
│   │   ├── chatbot.py                # Giao diện RAG Chatbot
│   │   ├── filter.py                 # Bộ lọc dữ liệu
│   │   └── kpi_cards.py              # Thẻ KPI tổng quan
│   ├── pages/
│   │   ├── 1_overview.py             # Trang Descriptive & Diagnostic
│   │   ├── 2_predictive.py           # Trang Predictive (biểu đồ dự báo)
│   │   └── 4_prescriptive.py         # Trang Prescriptive (đề xuất)
│   └── style/custom.css              # CSS tuỳ chỉnh giao diện
│
├── data/                             # Dữ liệu đầu vào
│   ├── sales.csv                     # Dữ liệu doanh thu lịch sử
│   ├── sales_test.csv                # Dữ liệu tập test
│   ├── sample_submission.csv         # Template nộp bài
│   ├── promotions.csv                # Chương trình khuyến mãi
│   ├── web_traffic.csv               # Lưu lượng truy cập web
│   ├── inventory.csv                 # Tồn kho
│   ├── customers.csv                 # Thông tin khách hàng
│   ├── orders.csv / order_items.csv  # Đơn hàng & chi tiết
│   ├── payments.csv                  # Thanh toán
│   ├── products.csv                  # Sản phẩm
│   ├── returns.csv                   # Trả hàng
│   ├── reviews.csv                   # Đánh giá
│   ├── shipments.csv                 # Vận chuyển
│   ├── geography.csv                 # Địa lý
│   └── rag/                          # Vector store cho RAG Chatbot
│       ├── vectorizer.joblib
│       ├── matrix.joblib
│       └── metadata.json
│
├── notebooks/                        # Jupyter Notebooks phân tích
│   ├── 01_mcq.ipynb                  # Câu hỏi trắc nghiệm
│   ├── 02_eda.ipynb                  # Phân tích khám phá dữ liệu
│   ├── 03_forecasting.ipynb          # Pipeline dự báo chính
│   ├── 04_seasonality_decomposition.ipynb  # Phân tích mùa vụ
│   └── rfm.ipynb                     # Phân tích RFM khách hàng
│
├── outputs/                          # Kết quả xuất ra
│   ├── submission.csv                # File nộp bài cuối cùng
│   ├── forecast_validation_metrics.csv
│   ├── forecast_model_comparison.csv
│   ├── forecast_lgbm_walkforward_cv.csv
│   ├── forecast_candidate_metrics.csv
│   ├── forecast_bundle_metadata.json
│   ├── master_dashboard.parquet      # Dữ liệu tổng hợp cho dashboard
│   ├── figures/                      # Biểu đồ SHAP & feature importance
│   │   ├── lgbm_feature_importance_revenue.png
│   │   ├── lgbm_feature_importance_cogs.png
│   │   ├── shap_summary_bar_revenue.png
│   │   ├── shap_summary_bar_cogs.png
│   │   ├── shap_beeswarm_revenue.png
│   │   └── shap_beeswarm_cogs.png
│   ├── charts/                       # Biểu đồ phân tích mùa vụ
│   └── tables/                       # Bảng thống kê mùa vụ
│
├── models/                           # Mô hình đã huấn luyện
│   ├── lgbm_model.pkl
│   ├── shap_explainer.pkl
│   └── submission.csv
│
├── src/                              # Mã nguồn pipeline chính
│   ├── data_loader.py                # Đọc & tiền xử lý dữ liệu
│   ├── feature_engineering.py        # Trích xuất đặc trưng
│   ├── model.py                      # Huấn luyện & dự báo
│   ├── metrics.py                    # Tính toán chỉ số đánh giá
│   ├── optimizer.py                  # Tối ưu trọng số ensemble
│   ├── seasonal_models.py            # Mô hình mùa vụ
│   ├── mlflow_tracking.py            # Tích hợp MLflow
│   └── rag/                          # Module RAG Chatbot
│       ├── ingest.py                 # Nạp & vector hoá tài liệu
│       ├── retriever.py              # Truy xuất ngữ cảnh liên quan
│       └── pipeline.py               # Pipeline Q&A đầu cuối
│
├── mlruns/                           # Lịch sử thí nghiệm MLflow
├── build_master_table.py             # Script tổng hợp dữ liệu master
├── build_cube.py                     # Script xây dựng OLAP cube
├── PBI.pbix                          # File Power BI
├── README.md
├── CHANGES.md
└── requirements.txt
```

---

## 🚀 Hướng dẫn cài đặt & chạy

### Yêu cầu hệ thống

- Python **3.10+**
- pip hoặc conda
- RAM khuyến nghị: 8GB+

---

### Bước 1 — Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

### Bước 2 — Tổng hợp dữ liệu master

Script này đọc toàn bộ file trong `data/`, join các bảng lại và xuất `outputs/master_dashboard.parquet` dùng cho dashboard.

```bash
python build_master_table.py
```

*(Tuỳ chọn)* Xây dựng OLAP cube phục vụ phân tích đa chiều:

```bash
python build_cube.py
```

---

### Bước 3 — Chạy pipeline dự báo

Có hai cách để huấn luyện mô hình và sinh file dự báo:

#### Cách A: Chạy qua Jupyter Notebook *(khuyến nghị để kiểm tra từng bước)*

```bash
jupyter notebook notebooks/03_forecasting.ipynb
```

Chạy tuần tự các cell trong notebook. Kết quả sẽ được xuất vào thư mục `outputs/`.

#### Cách B: Chạy trực tiếp qua Python *(nhanh hơn)*

```python
from src.model import train_and_export, predict_submission

# Huấn luyện toàn bộ pipeline và xuất kết quả
train_and_export()

# Sinh file submission theo thứ tự sample_submission.csv
predict_submission()
```

Hoặc chạy lệnh từ terminal:

```bash
python -c "from src.model import train_and_export; train_and_export()"
```

> **Lưu ý:** Nếu thư mục `outputs/` trống hoặc thiếu file, hãy chạy pipeline dự báo trước khi khởi động dashboard.

---

### Bước 4 — (Tuỳ chọn) Nạp tài liệu vào RAG Chatbot

Vector hoá tài liệu trong `data/raw_docs/` để kích hoạt tính năng Q&A thông minh trên dashboard:

```bash
python src/rag/ingest.py
```

Vector store sẽ được lưu tự động vào `data/rag/`. Chatbot trong dashboard sẽ hoạt động ngay sau bước này.

---

### Bước 5 — (Tuỳ chọn) Xem lịch sử thí nghiệm với MLflow

```bash
mlflow ui
```

Mở trình duyệt tại **http://localhost:5000** để xem toàn bộ lịch sử run, tham số tối ưu và chỉ số đã log trong `mlruns/`.

---

### Bước 6 — Khởi động Dashboard Streamlit

```bash
streamlit run app/main.py
```

Mở trình duyệt và truy cập: **http://localhost:8501**

Dashboard bao gồm 3 trang:
- **Overview** — Phân tích mô tả và chẩn đoán
- **Predictive** — Biểu đồ dự báo và đánh giá mô hình
- **Prescriptive** — Đề xuất hành động tối ưu hoá

---

## 🤖 Mô hình dự báo

### Mô hình nộp bài cuối cùng: `weighted_ensemble`

Pipeline kết hợp nhiều mô hình thành phần (candidate forecasters) với **trọng số được tối ưu hoá riêng** cho từng mục tiêu (`Revenue` và `COGS`).

```
Đầu vào: sales, sample_submission, promotions, web_traffic, inventory
    ↓
[1] Huấn luyện các mô hình mùa vụ (seasonal candidates)
    ↓
[2] Đánh giá & tối ưu trọng số ensemble
    ↓
[3] Xây dựng bundle weighted_ensemble
    ↓
[4] Huấn luyện baseline: LightGBM / XGBoost / ElasticNet (để so sánh)
    ↓
[5] Đánh giá trên tập validation (MAE / RMSE / R²)
    ↓
[6] Xuất submission.csv + SHAP figures + metrics tables
```

---

## 📈 Kết quả đánh giá mô hình

Nguồn: `outputs/forecast_validation_metrics.csv`

| Mục tiêu | R² | MAE | RMSE |
|----------|----|-----|------|
| **Revenue** | **0.858** | 450,689 | 628,267 |
| **COGS** | **0.874** | 368,677 | 514,707 |

So sánh chi tiết các mô hình: `outputs/forecast_model_comparison.csv`

---

## 🔍 Giải thích mô hình (Explainability)

Các biểu đồ giải thích được xuất tự động vào `outputs/figures/`:

- 📊 **LightGBM Feature Importance** — Tầm quan trọng của đặc trưng (Revenue & COGS)
- 🐝 **SHAP Summary Bar Plot** — Đóng góp trung bình của từng đặc trưng
- 🐝 **SHAP Beeswarm Plot** — Phân phối ảnh hưởng của đặc trưng theo từng mẫu

---

## 📂 File nộp bài

File `outputs/submission.csv` được sinh theo đúng thứ tự của `sample_submission.csv`, đảm bảo tuân thủ yêu cầu cuộc thi.

---

## ✅ Tuân thủ quy định cuộc thi

| Yêu cầu | Trạng thái |
|---------|-----------|
| Không sử dụng dữ liệu ngoài | ✅ |
| Validation theo thứ tự thời gian (time-aware) | ✅ |
| Nộp đúng thứ tự `sample_submission.csv` | ✅ |
| Mã nguồn đầy đủ & có thể tái tạo | ✅ |
| Có artifacts giải thích (SHAP + Feature Importance) | ✅ |

---

## ⚙️ Tóm tắt thứ tự chạy đầy đủ

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Tổng hợp dữ liệu master
python build_master_table.py

# 3. (Tuỳ chọn) Build OLAP cube
python build_cube.py

# 4. Huấn luyện mô hình & sinh submission
python -c "from src.model import train_and_export; train_and_export()"

# 5. (Tuỳ chọn) Nạp tài liệu RAG
python src/rag/ingest.py

# 6. (Tuỳ chọn) Xem MLflow UI
mlflow ui

# 7. Khởi động dashboard
streamlit run app/main.py
```

---

## 👥 Nhóm K-TEAM

> VinUni Datathon 2026 — Vòng 1

---

<div align="center">
<sub>Made with ❤️ by K-TEAM · VinUni Datathon 2026</sub>
</div>
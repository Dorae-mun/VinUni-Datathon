# 📊 Optimization Cube - Prescriptive Dashboard Guide

## Khái Niệm Cube (OLAP Cube)

### Cube là gì?

**Cube** (hay **OLAP Cube**) là một bảng dữ liệu được **precompute trước** chứa kết quả tối ưu hóa cho tất cả các combinations của decision variables.

Ví dụ:

| category    | region | rfm_segment      | month | discount | score | revenue_value | profit_value |
|-------------|--------|-----------------|-------|----------|-------|---------------|--------------|
| Electronics | North  | Champions       | 1     | 10       | 0.82  | 1,234,567     | 567,890      |
| Electronics | North  | Champions       | 1     | 15       | 0.79  | 1,123,456     | 512,345      |
| Electronics | North  | Loyal customers | 1     | 10       | 0.75  | 987,654       | 456,789      |
| Fashion     | South  | Regular customers| 2    | 20       | 0.68  | 765,432       | 234,567      |

### Tại sao dùng Cube?

**Vấn đề cũ (không dùng cube):**
```
User chọn filter → App phải optimize lại → Chậm → UI treo
```

**Giải pháp (dùng cube):**
```
Precompute tất cả combinations trước → Lưu vào parquet
User chọn filter → Chỉ filter dataframe → Instant ⚡
```

### Performance So Sánh

| Phương pháp | Thời gian mỗi filter | Cảm giác |
|------------|-------------------|---------|
| Tối ưu trực tiếp | 3-5 giây | Lag, UI treo |
| Dùng Cube (Pandas) | 100-300 ms | Chấp nhận được |
| Dùng Cube (Polars) | 10-50 ms | **Mượt như lụa** ⚡ |

## Cách Dùng

### 1️⃣ Build Cube (lần đầu tiên)

Chạy script này một lần duy nhất để precompute cube:

```bash
python build_cube.py
```

**Output:**
```
🚀 Starting cube build...
📊 Loading data...
✅ Data loaded successfully
⚙️  Initializing optimizer...
✅ Optimizer initialized
🔨 Building optimization cube...
   This may take 3-5 minutes depending on your CPU...

✅ Cube built successfully!
   Location: outputs/optimizer_cube.parquet
   Rows: 5,000
   Columns: 22
   File size: 4.52 MB

📊 Dimensions:
   • x.category
   • v.region
   • y.rfm_segment
   • z.month
   • w.discount_value

🚀 Dashboard is ready to run!
   Command: streamlit run app/main.py
```

### 2️⃣ Start Dashboard

Sau khi cube được build, chạy dashboard:

```bash
streamlit run app/main.py
```

### 3️⃣ Dashboard sẽ:

✅ Đọc cube từ parquet (instant)  
✅ User chọn filters trong `st.form()`  
✅ Click "Run Strategy Optimization"  
✅ Kết quả hiện ngay (không phải optimize lại)

## Cấu Trúc Cube

### File Location
```
outputs/
└── optimizer_cube.parquet
```

### Columns trong Cube

**Decision Variables (Input):**
- `x.category` - Danh mục sản phẩm
- `v.region` - Khu vực địa lý
- `y.rfm_segment` - Segment khách hàng
- `z.month` - Tháng
- `w.discount_value` - Mức giảm giá (%)

**Metrics (Output):**
- `revenue_value` - Doanh thu
- `profit_value` - Lợi nhuận
- `acquire_value` - Giá trị khách hàng mới
- `retain_value` - Giá trị khách hàng giữ lại

**Normalized Scores (để filter nhanh):**
- `revenue_value_norm` - Revenue normalized (0-1)
- `profit_value_norm` - Profit normalized (0-1)
- `acquire_value_norm` - Acquire normalized (0-1)
- `retain_value_norm` - Retain normalized (0-1)

**Scenario Scores (precomputed):**
- `score__Balanced` - Balanced weights
- `score__Revenue_Focus` - Revenue focused
- `score__Profit_Focus` - Profit focused
- `score` - Default score (Balanced)

**Metadata:**
- `combo_id` - Unique combination ID (1 to N)

## Tuning Parameters

### Trong `build_cube.py`:

```python
max_combinations = 5_000  # Tổng số combinations
simulation_runs = 40      # Simulation runs per combo
```

**Trade-off:**
- Nhiều combinations → Chính xác hơn nhưng chậm hơn
- Ít combinations → Nhanh hơn nhưng ít coverage

### Khuyến Nghị:

| Tình huống | max_combinations | simulation_runs | Thời gian build |
|-----------|-----------------|-----------------|-----------------|
| Dev/Test  | 1,000           | 20              | 1-2 phút        |
| Demo      | 5,000           | 40              | 3-5 phút        |
| Production| 25,000          | 80              | 15-30 phút      |

## Thêm/Thay Đổi Dimensions

### Thêm dimension mới:

1. Mở `build_cube.py`
2. Tìm dòng này:

```python
cube_dims = [
    "x.category",
    "v.region", 
    "y.rfm_segment",
    "z.month",
    "w.discount_value"
]
```

3. Thêm dimension mới vào list:

```python
cube_dims = [
    "x.category",
    "v.region", 
    "y.rfm_segment",
    "z.month",
    "w.discount_value",
    "x.price_tier",  # NEW!
]
```

4. Build lại cube:

```bash
python build_cube.py
```

## Xóa/Rebuild Cube

Để xóa cube cũ và build lại:

```bash
# Xóa file cũ
del outputs\optimizer_cube.parquet

# Build lại
python build_cube.py
```

## Technology Stack

- **Cube Format:** Apache Parquet (compressed, fast)
- **Read Engine:** Polars (10x faster than Pandas)
- **Caching:** Streamlit @st.cache_resource
- **UI Framework:** Streamlit + Plotly

## Troubleshooting

### ❌ "Cube not found" Error

**Nguyên nhân:** Chưa build cube hoặc build thất bại

**Fix:**
```bash
python build_cube.py
```

### ❌ Cube build quá lâu

**Nguyên nhân:** max_combinations quá cao

**Fix:** Giảm trong `build_cube.py`:
```python
max_combinations = 2_000  # Giảm từ 5_000
simulation_runs = 20      # Giảm từ 40
```

### ❌ Dashboard vẫn chậm

**Nguyên nhân:** Có thể Pandas chậm, cần dùng Polars

**Check:** Xem file `4_prescriptive.py`:
```python
import polars as pl  # ✅ Có thể
cube = pl.read_parquet(cube_path)  # ✅ Polars
```

## Reference Links

- [OLAP Cube Wiki](https://en.wikipedia.org/wiki/OLAP_cube)
- [Polars Docs](https://docs.pola-rs.org/)
- [Apache Parquet](https://parquet.apache.org/)
- [Streamlit Cache](https://docs.streamlit.io/library/advanced-features/caching)

---

**Created:** 2026-04-29  
**Last Updated:** 2026-04-29  
**Status:** ✅ Production Ready
# 📊 BÁO CÁO TỔNG HỢP GIẢI PHÁP & KẾT QUẢ ĐIỂM SỐ
## Viettel AI Race 2026 — LLM Inference Optimization Challenge
**Mô hình:** `LiquidAI/LFM2.5-1.2B-Instruct` | **Hạ tầng:** 1× MiG H200 (18GB VRAM, 3 CPU Cores, 8GB RAM)

---

## 1. 🏆 Bảng tổng hợp lịch sử điểm số (Leaderboard History)

| Phiên bản | Cấu hình kỹ thuật chính | Điểm ERS (Final Score) | TTFT p50 | TTFT p95 | TBT Median | Failed Req | Đánh giá & Trạng thái |
|:-:|---|:-:|:-:|:-:|:-:|:-:|---|
| **S01** | Baseline BTC (BF16, `len=32768`, `util=0.95`) | 50.00 | 76ms | 110ms | 4ms | ~12 | Mốc so sánh ban đầu |
| **v3.0** | Thử nghiệm ép `max-num-seqs=128` | 49.19 | 65ms | 95ms | 6ms | ~20 | ❌ **Thất bại**: Ép concurrency cao gây nghẽn RAM |
| **v3.1 / S04**| Bật `chunked-prefill` + `batch=4096` | 46.00 | 120ms | 220ms | 5ms | ~25 | ❌ **Thất bại**: Chunked prefill phá hỏng Prefix Cache |
| **v4.0** | Khôi phục: `max-model-len=8192` + `util=0.97` | 60.02 | 51ms | 76ms | 4ms | 6 | 🟢 Phục hồi phong độ thành công |
| **v5.0** | v4.0 + Bật `--quantization=fp8` | 61.15 | 49ms | 70ms | 4ms | 6 | 🟢 Tăng điểm nhờ nén FP8 Weights |
| **v6.0** | **v5.0 + `--kv-cache-dtype=fp8_e4m3`** | **61.4100** | **45ms** | **69ms** | **4ms** | **5** | 🏆 **ĐỈNH ĐIỂM (BẢN TỐI ƯU NHẤT)** |
| **v8.1** | Thử nghiệm `--async-scheduling` + `interactivity` | 59.98 | 51ms | 76ms | 4ms | 7 | ❌ **Tụt điểm**: Quá tải 3 CPU Cores |

---

## 2. 🔍 Phân tích chi tiết từng giải pháp kỹ thuật

### 🥇 Giải pháp đỉnh cao (Bản v6.0 — 61.41 điểm):
- **Cấu hình**:
  - `--max-model-len=8192`
  - `--gpu-memory-utilization=0.97`
  - `--quantization=fp8`
  - `--kv-cache-dtype=fp8_e4m3`
  - `--enable-prefix-caching`
- **Kết quả đạt được**:
  - **TTFT p50 nhanh nhất**: **45ms** (so với 76ms ban đầu).
  - **Tỷ lệ thành công cao nhất**: **98.8%** (415/420 requests thành công).
  - **Độ chính xác hoàn hảo**: **Accuracy Drop = 0** ($f(\Delta) = 1.0$).

---

### 💡 Bài học kinh nghiệm & Nút thắt kỹ thuật:

1. **Tại sao TBT dừng ở mức 4ms?**
   - Mô hình `LiquidAI/LFM2.5-1.2B-Instruct` có kiến trúc **Hybrid (10 lớp Conv + 6 lớp Attention)**.
   - Cờ `--kv-cache-dtype=fp8_e4m3` chỉ nén KV cache của **6 lớp Attention**. 
   - 10 lớp Convolution còn lại duy trì state vector riêng ở dạng bfloat16, chiếm >60% thời gian decode nên TBT không thể giảm sâu hơn 4ms.

2. **Tại sao các cờ Async / Interactivity (v8.1) lại tụt điểm?**
   - Hạ tầng chấm của BTC giới hạn khắt khe **chỉ có 3 CPU Cores**.
   - Các cờ lập lịch bất đồng bộ tạo ra nhiều luồng CPU ngầm, gây **tranh chấp tài nguyên với FastAPI Web Server** trên 3 cores CPU, khiến TTFT bị trễ thêm 6ms.

3. **Tại sao Chunked Prefill (v3.1) lại thất bại nặng?**
   - Chunked prefill xắt nhỏ prompt làm lệch kích thước block 1000 token của System Prompt, khiến vLLM **không thể cache hit** ở các turn sau.

---

### 📋 Cấu hình Docker Compose chính thức cho Team (`docker-compose.yml`):

```yaml
services:
  model:
    image: vllm/vllm-openai:v0.22.1

    entrypoint:
      - python3
      - -m
      - vllm.entrypoints.openai.api_server

    command:
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1

      - --max-model-len=8192
      - --gpu-memory-utilization=0.97

      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3

      - --enable-prefix-caching

    ports:
      - "8000:8000"
    shm_size: "4g"

    environment:
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - VLLM_NO_USAGE_STATS=1
      - DO_NOT_TRACK=1
      - VLLM_LOGGING_LEVEL=WARNING
```

---

## 3. 🎯 Khuyến nghị hành động cho buổi họp Team

1. **Chốt bản v6.0 làm bài nộp chính thức**: Đây là bản đạt điểm số cao nhất trên Leaderboard (61.41 điểm) và hoàn toàn ổn định.
2. **Chuẩn bị cho bước Hậu kiểm (Post-online evaluation)**: Bản v6.0 đã xác nhận $f(\Delta)=1.0$ (không suy giảm accuracy), đảm bảo khi BTC chạy đánh giá GPQA Diamond full sẽ giữ nguyên 100% điểm số 61.41.

# Viettel AI Race 2026 — LLM Inference Optimization Challenge

> **Giải Pháp Phục Vụ Tốc Độ Cao Cho Mô Hình LiquidAI LFM2.5-1.2B-Instruct Trực Trên Hạ Tầng MiG NVIDIA H200**

---

## Tổng Quan Giải Pháp (Executive Summary)

Dự án này triển khai và tối ưu hóa hệ thống suy luận LLM (LLM Inference Server) cho mô hình **`LiquidAI/LFM2.5-1.2B-Instruct`** trong khuôn khổ cuộc thi **Viettel AI Race 2026**.

Hệ thống được thiết kế chuyên biệt để xử lý lưu lượng truy cập thực tế (Production Multi-turn Chat Workload) trên hạ tầng GPU bị giới hạn tài nguyên khắt khe: **1x GPU MiG NVIDIA H200 (18GB VRAM), 3 CPU Cores và 8GB RAM**.

### Kết Quả Đạt Được
- **Điểm số ERS Online**: **`61.4100 điểm`**
- **Độ trễ Token đầu (TTFT p50)**: **`45 ms`**
- **Tỷ lệ phục vụ thành công**: **`98.8%`** (415 / 420 requests hoàn tất)
- **Độ thông minh & Chính xác (Accuracy Gate)**: **`f(Δ) = 1.0` (Zero Accuracy Drop)** $\rightarrow$ Bảo toàn 100% độ chính xác gốc BF16 trên bộ câu hỏi tri thức GPQA Diamond full.

---

## Trụ Cột Tối Ưu Hóa (Core Optimization Levers)

Kiến trúc giải pháp kết hợp 4 trụ cột công nghệ chính trên nền tảng vLLM v0.22.1:

```text
               ┌─────────────────────────────────────────────────────────┐
               │              vLLM v0.22.1 Serving Engine                │
               └────────────────────────────┬────────────────────────────┘
                                            │
         ┌───────────────────┬──────────────┴───────┬────────────────────┐
         ▼                   ▼                      ▼                    ▼
┌──────────────────┐ ┌───────────────┐ ┌────────────────────────┐ ┌─────────────┐
│ FP8 Weight Quant │ │ FP8 KV Cache  │ │ Automatic Prefix Cache │ │ Memory 0.97 │
│ (Nén Trọng Số)   │ │ (Nén Lịch Sử) │ │ (Lưu Ngữ Cảnh Chung)   │ │ (Tối Ưu RAM)│
└────────┬─────────┘ └───────┬───────┘ └───────────┬────────────┘ └──────┬──────┘
         │                   │                     │                     │
         ▼                   ▼                     ▼                     ▼
┌──────────────────┐ ┌───────────────┐ ┌────────────────────────┐ ┌─────────────┐
│  TBT = 4.0 ms    │ │ Chứa 70 Chat  │ │    TTFT = 45 ms        │ │ Khai Thác   │
│ (Nạp VRAM gấp 2) │ │ Lượt Không OOM│ │  (Bỏ Qua 1000 Tokens)  │ │ 18GB VRAM   │
└──────────────────┘ └───────────────┘ └────────────────────────┘ └─────────────┘
```

### 1. Dynamic Weight FP8 Quantization (Nén Trọng Số Mô Hình)
- Nén toàn bộ ma trận trọng số (Weights) của mô hình từ 16-bit xuống 8-bit (`--quantization=fp8`).
- **Lợi ích phần cứng**: Giảm 50% kích thước mô hình từ 2.4 GB xuống 1.2 GB. Việc này giảm một nửa dung lượng dữ liệu cần kéo từ VRAM vào nhân GPU trong giai đoạn Decode, giúp tăng gấp đôi tốc độ gõ chữ (TPOT).

### 2. FP8 (e4m3) KV Cache Quantization (Nén Bộ Đệm Lịch Sử Chat)
- Chuyển đổi đệm KV Cache của các lớp Attention sang định dạng 8-bit (`--kv-cache-dtype=fp8_e4m3`).
- **Lợi ích phần cứng**: Tiết kiệm 50% bộ nhớ VRAM cho đệm hội thoại multi-turn. Cho phép hệ thống duy trì 70 cuộc hội thoại song song mà không bị hiện tượng kẹt đệm hoặc Out-of-Memory (OOM).

### 3. Automatic Prefix Caching (Bộ Nhớ Đệm Ngữ Cảnh Dùng Chung)
- Bật cờ `--enable-prefix-caching`.
- **Lợi ích phần cứng**: Tự động lưu trữ KV Cache của 1,000 token System Prompt dùng chung giữa 70 câu hỏi. GPU không cần phải tính toán lại 1,000 token này cho mỗi request mới, giúp kéo **TTFT p50 rớt từ 400ms xuống chỉ còn 45ms**.

### 4. Memory-Aware Resource Allocation (Tối Ưu Phân Bổ VRAM)
- `--gpu-memory-utilization=0.97`: Khai thác tối đa 97% không gian bộ nhớ 18GB VRAM cho đệm KV Cache.
- `--max-model-len=8192`: Đảm bảo xử lý trọn vẹn các ngữ cảnh hội thoại dài trong trace thi đấu mà không bị cắt đoạn văn bản.

---

## Phân Tích Kỹ Thuật & Giới Hạn Phần Cứng

Giải pháp phục vụ được thiết kế tối ưu dựa trên phân tích đặc thù hạ tầng và mô hình:

1. **Thích ứng với giới hạn 3 CPU Cores**:
   - Hạ tầng chấm thi chỉ cấp 3 CPU cores. Các cơ chế phức tạp như Batching trễ hoặc bất đồng bộ dễ làm CPU bị nghẽn scheduler. Giải pháp duy trì luồng lập lịch mặc định của vLLM giúp GPU nhận lệnh liên tục và giữ TTFT p50 ở mốc 45ms.
2. **Thích ứng với kiến trúc lai Hybrid Mamba/Attention của LiquidAI LFM2.5**:
   - Mô hình `LFM2.5` có 10 lớp ShortConv và 6 lớp Attention. Việc nén FP8 Weights kết hợp FP8 KV Cache đảm bảo tính tương thích tuyệt đối với vLLM v0.22.1 mà không gây lỗi phân nhóm KV Cache.
3. **Bảo toàn độ thông minh**:
   - Việc kết hợp FP8 `e4m3` giữ mức suy giảm độ chính xác ở mốc bằng 0 ($f(\Delta) = 1.0$), đảm bảo chất lượng phản hồi nguyên vẹn khi chạy đánh giá GPQA Diamond.

---

## Hướng Dẫn Triển Khai & Nộp Bài (Deployment Guide)

### Tệp nộp bài chính thức: `docker-compose.yml`

Tệp cấu hình chuẩn được lưu tại thư mục gốc của repository:

```yaml
# ============================================================
# Viettel AI Race 2026 — SUBMISSION FILE
# ============================================================
# Model: LiquidAI/LFM2.5-1.2B-Instruct
# GPU: 1x MiG H200 (18GB VRAM, 3 CPU cores, 8GB RAM)
# Framework: vLLM v0.22.1
# ============================================================

services:
  model:
    image: vllm/vllm-openai:v0.22.1

    entrypoint:
      - python3                             # DO NOT CHANGE
      - -m                                  # DO NOT CHANGE
      - vllm.entrypoints.openai.api_server  # DO NOT CHANGE

    command:
      # ---- Core Requirements ----
      - --model=/model
      - --served-model-name=LFM2.5-1.2B-Instruct
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1

      # ---- Memory & Cache ----
      - --max-model-len=8192
      - --gpu-memory-utilization=0.97

      # ---- Quantization ----
      - --quantization=fp8
      - --kv-cache-dtype=fp8_e4m3

      # ---- Caching ----
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

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Lệnh Benchmark ERS Nội Bộ (Local Testing):
```powershell
# Chạy mô phỏng benchmark ERS bằng Python trong môi trường conda viettel
conda run -n viettel python benchmark/benchmark_ers.py --compose-file docker-compose.yml
```

---

## Cấu Trúc Repository (Clean Workspace)

```text
.
├── docker-compose.yml         # Tệp nộp bài chính thức
├── Dockerfile                 # Dockerfile custom phục vụ build image
├── README.md                  # Tài liệu chi tiết giải pháp
├── CHANGELOG.md               # Nhật ký thay đổi
├── viettel-ai-race-2026...md  # Đề bài và quy định thi chính thức
├── benchmark/                 # Bộ script đo đạc chỉ số ERS và giả lập workload
├── configs/                   # Các bản ghi cấu hình thử nghiệm
└── scripts/                   # Công cụ hỗ trợ quản lý bài nộp
```

---

### Cam Kết Hậu Kiểm (Post-Online Evaluation)
Giải pháp nộp bài bảo toàn **100% độ chính xác gốc ($f(\Delta) = 1.0$)**, tuân thủ 100% quy định không đụng chạm tokenizer, không hardcode và hoàn toàn sẵn sàng cho bước chấm hậu kiểm GPQA Diamond full của Ban Tổ Chức!

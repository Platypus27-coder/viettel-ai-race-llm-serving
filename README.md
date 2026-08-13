# 🏆 Viettel AI Race 2026 — LLM Inference Optimization Challenge

> **Giải Pháp Phục Vụ Tốc Độ Cao Cho Mô Hình LiquidAI LFM2.5-1.2B-Instruct Trực Trên Hạ Tầng MiG NVIDIA H200**

[![Serving Framework](https://img.shields.io/badge/Serving-vLLM%20v0.22.1-blue.svg)](https://github.com/vllm-project/vllm)
[![GPU Infrastructure](https://img.shields.io/badge/GPU-1x%20MiG%20NVIDIA%20H200%20(18GB)-green.svg)](https://www.nvidia.com)
[![ERS Score](https://img.shields.io/badge/ERS%20Score-61.4100-orange.svg)](#-phân-tích-kỹ-thuật--giới-hạn-phần-cứng)
[![Accuracy Gate](https://img.shields.io/badge/Accuracy%20Gate-100%25%20(f(%CE%94)%20%3D%201.0)-brightgreen.svg)](#1-dynamic-weight-fp8-quantization-nén-trọng-số-8-bit)

---

## 📌 Tổng Quan Giải Pháp (Executive Summary)

Dự án này triển khai và tối ưu hóa hệ thống suy luận LLM (LLM Inference Server) cho mô hình **`LiquidAI/LFM2.5-1.2B-Instruct`** trong khuôn khổ cuộc thi **Viettel AI Race 2026**.

Hệ thống được thiết kế chuyên biệt để xử lý lưu lượng truy cập thực tế (Production Multi-turn Chat Workload) trên hạ tầng GPU bị giới hạn tài nguyên khắt khe: **1x GPU MiG NVIDIA H200 (18GB VRAM), 3 CPU Cores và 8GB RAM**.

### 🌟 Thành Tựu Đạt Được (Profile v6.0)
- **Điểm số ERS Online**: **`61.4100 điểm`** (Thuộc top đầu bảng xếp hạng trực tuyến).
- **Độ trễ Token đầu (TTFT p50)**: **`45 ms`** (Thời gian phản hồi ban đầu cực kỳ ấn tượng).
- **Tỷ lệ phục vụ thành công**: **`98.8%`** (415 / 420 requests hoàn tất mượt mà).
- **Độ thông minh & Chính xác (Accuracy Gate)**: **`f(Δ) = 1.0` (Zero Accuracy Drop)** $\rightarrow$ Bảo toàn 100% độ chính xác gốc BF16 trên bộ câu hỏi tri thức GPQA Diamond full.

---

## 🛠️ Trụ Cột Tối Ưu Hóa (Core Optimization Levers)

Kiến trúc giải pháp v6.0 kết hợp 4 trụ cột công nghệ chính trên nền tảng vLLM v0.22.1:

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
│  TBT = 4.0 ms    │ │ Chứa 70 Chat  │ │    TTFT = 45 ms        │ │ Khái Thác   │
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

## 🔬 Phân Tích Kỹ Thuật & Giới Hạn Phần Cứng

### Tại sao v6.0 là Cấu Hình Tối Ưu Tuyệt Đối?

Trong quá trình thực nghiệm, hệ thống đã thử nghiệm các kỹ thuật nâng cao khác nhưng ghi nhận các rào cản phần cứng khắt khe:

1. **Giới hạn 3 CPU Cores (Nghẽn CPU Scheduler)**:
   - Khi thử nghiệm ép batch prefill (`--max-num-batched-tokens=1536`) hoặc bật lập lịch bất đồng bộ (`--async-scheduling`), 3 CPU cores bị quá tải trong việc gửi lệnh CUDA kernel xuống GPU. Điều này làm tăng độ trễ prefill queue và khiến TTFT p50 bị suy giảm từ **45ms lên 55ms** (Điểm ERS giảm từ 61.41 xuống 59.90).
2. **Kiến trúc Lai Hybrid Mamba/Attention của LiquidAI LFM2.5**:
   - Mô hình `LFM2.5` có 10 lớp ShortConv và 6 lớp Attention. 
   - Việc áp dụng Speculative Decoding với Draft Model thứ 2 bị vLLM 0.22.1 từ chối do vi phạm quy tắc nhóm KV Cache duy nhất (`AssertionError: All drafting layers should belong to the same kv cache group`).
3. **Kết luận**: Bản cấu hình **v6.0** đạt mốc **61.41 điểm** chính là đỉnh cao hiệu năng thực tế, vừa khai thác hết tốc độ của GPU H200, vừa hoàn toàn phù hợp với giới hạn 3 CPU cores.

---

## 🚀 Hướng Dẫn Triển Khai & Nộp Bài (Deployment Guide)

### 📄 Tệp nộp bài chính thức: `docker-compose.yml`

Tệp cấu hình chuẩn được lưu tại thư mục gốc của repository:

```yaml
# ============================================================
# Viettel AI Race 2026 — SUBMISSION FILE
# ============================================================
# Model: LiquidAI/LFM2.5-1.2B-Instruct
# GPU: 1x MiG H200 (18GB VRAM, 3 CPU cores, 8GB RAM)
# Framework: vLLM v0.22.1
# PROFILE: v6.0 (BEST PERFORMING SUBMISSION — SCORE 61.41)
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

### 🧪 Lệnh Benchmark ERS Nội Bộ (Local Testing):
```powershell
# Chạy mô phỏng benchmark ERS bằng Python trong môi trường conda viettel
conda run -n viettel python benchmark/benchmark_ers.py --compose-file docker-compose.yml
```

---

## 📊 Bảng Lịch Sử Thử Nghiệm & Kết Quả (Benchmark Matrix)

| Phiên bản Profile | Kỹ thuật áp dụng | ERS Score | TTFT p50 | Success Rate | f(Δ) Accuracy | Ghi chú & Trạng thái |
|---|---|---|---|---|---|---|
| **Baseline BF16** | Mặc định chưa tối ưu | ~38.20 | 180 ms | 95.0% | 1.0 | Bản gốc vLLM |
| **v2.0 FP8 Weight** | FP8 Weight Only | 52.10 | 95 ms | 97.2% | 1.0 | Tăng tốc Decode |
| **v6.0 Incumbent** | **FP8 W + FP8 KV + APC** | **61.4100** | **45 ms** | **98.8%** | **1.0** | **🏆 Đỉnh cao chính thức** |
| **batch1536 Exp** | v6.0 + Max Batched Tokens 1536 | 59.9000 | 55 ms | 98.5% | 1.0 | CPU bị trễ prefill queue |
| **Draft Speculative** | Speculative Draft Model 350M | Failed | N/A | 0.0% | N/A | Lỗi vLLM Hybrid KV Cache |

---

## 📁 Cấu Trúc Repository (Clean Workspace)

```text
.
├── docker-compose.yml         # Tệp nộp bài chính thức (v6.0 — 61.41 điểm)
├── Dockerfile                 # Dockerfile custom phục vụ build image
├── README.md                  # Tài liệu chi tiết dự án
├── CHANGELOG.md               # Nhật ký các thay đổi và kết quả thử nghiệm
├── viettel-ai-race-2026...md  # Đề bài và quy định thi chính thức
├── benchmark/                 # Bộ script đo đạc chỉ số ERS và giả lập workload
├── configs/                   # Các bản ghi cấu hình thử nghiệm
└── scripts/                   # Công cụ hỗ trợ quản lý và chọn bài nộp
```

---

### 🛡️ Cam Kết Hậu Kiểm (Post-Online Evaluation)
Bản nộp **v6.0 (61.4100 điểm)** bảo toàn **100% độ chính xác gốc ($f(\Delta) = 1.0$)**, tuân thủ 100% quy định không đụng chạm tokenizer, không hardcode và hoàn toàn sẵn sàng cho bước chấm hậu kiểm GPQA Diamond full của Ban Tổ Chức!

# PLAN — Viettel AI Race 2026
## LLM Inference Optimization Challenge

> **Trạng thái:** Đây là backlog thử nghiệm mở rộng. Sau kết quả portal 60.02
> và 48.2, ba lượt còn lại đã được chốt trong
> `solution.md`, `README.md` và `configs/README.md`. Không dùng trực tiếp
> các YAML minh họa 32K/64-sequence hoặc quy trình prewarm trong tài liệu này
> để submit; benchmark hiện hành luôn bắt đầu với shared-prefix cache lạnh.

**Model:** `LiquidAI/LFM2.5-1.2B-Instruct`  
**Framework bắt buộc:** vLLM `v0.22.1`  
**Hạ tầng chấm:** 1× MiG H200, 18 GB VRAM, 3 CPU cores, 8 GB RAM  
**Vòng online:** 02/07/2026 – 30/07/2026  
**Mục tiêu:** Tối đa hóa ERS nhưng vẫn vượt Accuracy Gate và hậu kiểm.

---

# 1. Mục tiêu chiến thắng

Không tìm một cấu hình “nghe có vẻ nhanh”. Phải tìm cấu hình có:

1. **Median ERS cao nhất** qua nhiều lần chạy.
2. **Không lỗi, không timeout, không OOM**.
3. **Accuracy drop đủ an toàn**.
4. **p95 TTFT thấp** để có lợi khi tie-break.
5. **Kết quả ổn định** khi BTC re-grade.

Điểm cuối:

\[
Score = 100 \times ERS \times f(\Delta)
\]

Trong đó:

\[
ERS = \frac{1}{N}\sum_{i=1}^{N}
\left(0.5s_{\text{ttft},i}+0.5s_{\text{tpot},i}\right)
\]

\[
s_{\text{ttft}}
=
\left[
\operatorname{clamp}
\left(
\frac{400-\text{TTFT}}{390},
0,
1
\right)
\right]^2
\]

\[
s_{\text{tpot}}
=
\left[
\operatorname{clamp}
\left(
\frac{10-\text{TPOT}}{9},
0,
1
\right)
\right]^2
\]

Accuracy Gate:

\[
\Delta = Accuracy_{\text{baseline}}-Accuracy_{\text{submission}}
\]

- \(\Delta \le 0.10\): không bị phạt.
- \(0.10 < \Delta < 0.16\): bị phạt tuyến tính.
- \(\Delta \ge 0.16\): điểm cuối bằng 0.

---

# 2. Nguyên tắc thực hiện

## 2.1. Chỉ thay đổi một nhóm biến mỗi lần

Không bật đồng thời:

- FP8 weights.
- FP8 KV cache.
- Batch lớn.
- Concurrency lớn.
- Tham số scheduler mới.

Nếu bật tất cả cùng lúc, sẽ không biết kỹ thuật nào giúp tăng điểm hoặc gây lỗi.

## 2.2. Mỗi cấu hình phải chạy lặp lại

Mỗi cấu hình ứng viên:

- Chạy warm-up trước.
- Benchmark ít nhất 3 lần.
- Lấy median ERS.
- Ghi mean/p50/p95/p99 TTFT.
- Ghi mean/p50/p95 TPOT.
- Ghi số request lỗi hoặc timeout.
- Ghi peak VRAM.
- Ghi số lần preemption nếu có.

## 2.3. Không đánh giá bằng throughput đơn thuần

BTC không chấm chỉ số request/second trực tiếp.

Một cấu hình throughput cao nhưng TTFT hoặc TPOT xấu có thể có ERS thấp.

## 2.4. Không tin số liệu dự đoán

Chỉ giữ kết luận có:

- Log benchmark.
- File kết quả.
- Cấu hình chính xác.
- Image digest.
- Kết quả GPQA tương ứng.

---

# 3. Cấu trúc thư mục đề xuất

```text
viettel-ai-race/
├── docker-compose.yml
├── docker/
│   └── Dockerfile
├── configs/
│   ├── baseline.yaml
│   ├── bf16_best.yaml
│   ├── fp8_weight.yaml
│   ├── fp8_kv_e4m3.yaml
│   └── fp8_aggressive.yaml
├── benchmark/
│   ├── benchmark_ers.py
│   ├── calculate_ers.py
│   ├── run_matrix.py
│   ├── validate_endpoint.py
│   └── results/
├── accuracy/
│   ├── bench-gpqa-diamond.sh
│   └── results/
├── scripts/
│   ├── start_server.sh
│   ├── warmup.sh
│   ├── collect_metrics.sh
│   └── build_and_push.sh
├── experiments.csv
├── submissions.csv
└── plan.md
```

---

# 4. Baseline đầu tiên

Mục tiêu của baseline là tạo một hệ thống:

- Chạy chắc chắn.
- Không quantization.
- Không dùng thủ thuật khó kiểm soát.
- Có prefix caching.
- Có chunked prefill.
- Có đủ log để so sánh.

## 4.1. Docker Compose baseline

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

      - --max-model-len=32768
      - --gpu-memory-utilization=0.95

      - --enable-prefix-caching
      - --enable-chunked-prefill

      - --max-num-batched-tokens=4096
      - --max-num-seqs=64

    ports:
      - "8000:8000"

    shm_size: "2g"

    environment:
      - VLLM_NO_USAGE_STATS=1
      - DO_NOT_TRACK=1
      - VLLM_LOGGING_LEVEL=INFO

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## 4.2. Không bật trong baseline

```text
--quantization=fp8
--kv-cache-dtype=fp8_e4m3
--kv-cache-dtype=fp8_e5m2
CPU offloading
NVMe offloading
Speculative decoding
Custom CUDA/Triton
Semantic caching
```

---

# 5. Bước 0 — Kiểm tra khả năng chạy

Trước khi tối ưu, phải xác nhận:

- Container khởi động thành công.
- Model load thành công.
- Endpoint trả đúng OpenAI-compatible response.
- Streaming hoạt động.
- Model name đúng.
- Không gọi mạng ngoài khi chạy.
- Không hết RAM hoặc VRAM.
- Số output token đáp ứng benchmark.
- Image có thể pull công khai.

## Checklist

```text
[ ] docker pull thành công
[ ] container start thành công
[ ] /v1/models trả kết quả
[ ] /v1/chat/completions hoạt động
[ ] stream=true trả từng chunk
[ ] served model name đúng
[ ] không OOM khi warm-up
[ ] không có request trả 0 token
[ ] tokenizer/chat template đúng
```

---

# 6. Bước 1 — Đo baseline BF16

Chạy cấu hình baseline ít nhất 3 lần.

Ghi:

| Run | ERS | TTFT mean | TTFT p95 | TPOT mean | Errors | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| Median | | | | | | |

Baseline này là mốc so sánh cho toàn bộ thử nghiệm sau.

---

# 7. Bước 2 — Tối ưu `max-num-batched-tokens`

Giữ nguyên tất cả tham số khác.

Thử:

```text
2048
4096
8192
16384
```

## Ma trận

| ID | max-num-batched-tokens | max-num-seqs |
|---|---:|---:|
| BT-2048 | 2048 | 64 |
| BT-4096 | 4096 | 64 |
| BT-8192 | 8192 | 64 |
| BT-16384 | 16384 | 64 |

## Quy tắc chọn

Giữ 2 giá trị có:

1. Median ERS cao nhất.
2. Không lỗi.
3. TPOT không bị tăng mạnh.
4. TTFT p95 ổn định.

Không chọn dựa trên một lần chạy tốt bất thường.

---

# 8. Bước 3 — Tối ưu `max-num-seqs`

Sử dụng giá trị `max-num-batched-tokens` tốt nhất ở Bước 2.

Thử:

```text
32
64
96
128
192
```

## Ma trận

| ID | max-num-seqs | Mục đích |
|---|---:|---|
| SEQ-32 | 32 | Bảo vệ latency |
| SEQ-64 | 64 | Cân bằng |
| SEQ-96 | 96 | Concurrency cao |
| SEQ-128 | 128 | Aggressive |
| SEQ-192 | 192 | Stress test |

## Dấu hiệu `max-num-seqs` quá thấp

- Request phải xếp hàng.
- TTFT tăng khi traffic dồn.
- GPU chưa được sử dụng hết.

## Dấu hiệu `max-num-seqs` quá cao

- TPOT tăng.
- Preemption tăng.
- VRAM căng.
- OOM.
- Kết quả giữa các lần chạy dao động mạnh.

---

# 9. Bước 4 — Tối ưu GPU memory utilization

Dùng cấu hình batch và sequence tốt nhất.

Thử:

```text
0.90
0.93
0.95
0.97
```

| ID | gpu-memory-utilization |
|---|---:|
| MEM-90 | 0.90 |
| MEM-93 | 0.93 |
| MEM-95 | 0.95 |
| MEM-97 | 0.97 |

## Quy tắc chọn

Chỉ chọn `0.97` khi:

- Không OOM trong mọi lần chạy.
- Không có preemption bất thường.
- Median ERS cao hơn rõ ràng.
- Peak VRAM vẫn có khoảng an toàn.
- Container ổn định khi restart sạch.

Nếu chênh lệch ERS nhỏ, ưu tiên `0.95`.

---

# 10. Bước 5 — Đánh giá Prefix Caching

So sánh A/B:

```text
CACHE-OFF: không bật prefix caching
CACHE-ON:  bật --enable-prefix-caching
```

Đo riêng:

- Turn 1.
- Turn 2 trở đi.
- Request có shared system prefix.
- Cache hit rate nếu log hỗ trợ.
- TTFT theo từng turn.

## Điều cần xác minh

Prefix caching chỉ có lợi khi:

- Token prefix giống hệt.
- Cache chưa bị eviction.
- Phần prefix đủ dài để bù overhead hashing.
- Request sau thực sự có thể tái sử dụng block.

Không mặc định kết luận rằng mọi turn đều được reuse 100%.

---

# 11. Bước 6 — Đánh giá Chunked Prefill

So sánh:

```text
CP-OFF
CP-ON
```

Với `CP-ON`, thử các token budget tốt nhất ở Bước 2.

## Kỳ vọng

Chunked prefill có thể:

- Giảm việc prompt dài chặn decode.
- Giữ TPOT ổn định hơn.
- Nhưng có thể làm TTFT của prompt dài thay đổi.

Chỉ giữ khi ERS tổng tăng.

---

# 12. Bước 7 — Tối ưu scheduling/runtime

Thử từng tính năng riêng.

## 12.1. Async scheduling

Chỉ bật nếu lệnh sau xác nhận vLLM `v0.22.1` hỗ trợ:

```bash
python3 -m vllm.entrypoints.openai.api_server --help
```

Ứng viên:

```text
--async-scheduling
```

So sánh:

```text
ASYNC-OFF
ASYNC-ON
```

## 12.2. Logging

Sau khi hệ thống ổn định:

```yaml
VLLM_LOGGING_LEVEL=WARNING
```

So sánh với `INFO`.

Không kỳ vọng tăng điểm lớn, nhưng có thể giảm CPU I/O.

## 12.3. PyTorch allocator

Thử riêng:

```yaml
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Chỉ giữ khi:

- Giảm fragmentation.
- Giảm OOM.
- Không gây cảnh báo hoặc lỗi tương thích.
- ERS không giảm.

## 12.4. CUDA Graphs

vLLM thường tự sử dụng CUDA Graphs ở các shape phù hợp.

Không cần thêm cờ mù quáng. Kiểm tra log để xác nhận:

- Graph capture thành công.
- Không fallback quá nhiều.
- Memory overhead chấp nhận được.

---

# 13. Bước 8 — Online FP8 weights

Chỉ bắt đầu sau khi BF16 đã được tối ưu.

So sánh:

| ID | Weights | KV cache |
|---|---|---|
| Q-W-BF16 | BF16 | auto |
| Q-W-FP8 | FP8 online | auto |

Cấu hình thử:

```text
--quantization=fp8
```

## Điều kiện giữ FP8 weights

Chỉ giữ khi:

- Median ERS tăng rõ ràng.
- Không tăng lỗi hoặc variance.
- GPQA Diamond không bị giảm quá mức.
- Startup time vẫn nằm trong giới hạn chấm.
- Không có layer fallback gây bottleneck lớn.

Không giữ chỉ vì VRAM giảm.

---

# 14. Bước 9 — FP8 KV cache

Thử sau khi đã biết cấu hình weights tốt nhất.

| ID | Weights | KV dtype |
|---|---|---|
| KV-AUTO | BF16 hoặc FP8 tốt nhất | auto |
| KV-E4M3 | BF16 hoặc FP8 tốt nhất | fp8_e4m3 |
| KV-E5M2 | BF16 hoặc FP8 tốt nhất | fp8_e5m2 |

Cấu hình:

```text
--kv-cache-dtype=fp8_e4m3
```

hoặc:

```text
--kv-cache-dtype=fp8_e5m2
```

## Thứ tự ưu tiên

1. `auto`.
2. `fp8_e4m3`.
3. `fp8_e5m2`.

## Điều kiện giữ FP8 KV

- Tăng concurrency thực tế.
- Giảm preemption.
- ERS tăng.
- GPQA vẫn an toàn.
- Không xuất hiện lỗi số học hoặc output bất thường.

---

# 15. Bước 10 — Speculative decoding

Ưu tiên thấp.

Model chính chỉ khoảng 1.2B, nên draft model hoặc speculative method có thể tạo overhead lớn.

Chỉ thử khi:

- TPOT vẫn là bottleneck chính.
- Còn đủ thời gian.
- Có draft method tương thích.
- Có đủ VRAM.
- Có benchmark A/B rõ ràng.

Nếu acceptance rate thấp hoặc ERS không tăng, loại ngay.

---

# 16. Bước 11 — Custom CUDA/Triton

Chỉ thực hiện nếu profiler chứng minh một kernel cụ thể chiếm tỷ lệ thời gian lớn.

## Không bắt đầu bằng custom kernel

Không viết custom CUDA chỉ vì đề cho phép.

Quy trình đúng:

```text
Profile
→ xác định kernel nóng
→ tìm kernel có sẵn tốt hơn
→ thử cấu hình backend
→ chỉ sau đó mới viết custom kernel
```

## Các vùng có thể profile

- LIV convolution.
- GQA attention.
- RMSNorm.
- RoPE.
- Linear/MLP.
- Sampling.
- KV cache copy.
- Detokenization/streaming CPU.

## Điều kiện giữ custom kernel

- Kết quả số học đúng.
- Không làm accuracy giảm.
- Tăng ERS ổn định.
- Không crash.
- Không phụ thuộc bất hợp lệ vào trace.
- Có thể build lại trong Docker.
- Chạy đúng trên MiG H200.

---

# 17. Benchmark protocol

## 17.1. Warm-up

Trước khi đo:

```text
1. Khởi động server sạch.
2. Chờ model ready.
3. Gửi một số request warm-up.
4. Xác nhận CUDA Graph/kernel compile hoàn tất.
5. Sau đó mới bắt đầu trace đo.
```

Không trộn warm-up vào kết quả ERS.

## 17.2. Đo streaming

Với mỗi request:

```text
request_start
first_token_time
token_timestamp_2
token_timestamp_3
...
request_end
```

Tính:

```text
TTFT = first_token_time - request_start
```

TPOT cần khớp cách BTC định nghĩa, ưu tiên:

```text
TPOT_mean =
(request_end - first_token_time) / (output_tokens - 1)
```

Nếu BTC cung cấp script chính thức, dùng đúng script BTC.

## 17.3. Lưu kết quả từng request

```csv
experiment_id,run_id,conversation_id,turn_id,input_tokens,output_tokens,arrival,ttft_ms,tpot_ms,request_score,status
```

## 17.4. Tổng hợp

```csv
experiment_id,run_id,ers,ttft_mean,ttft_p50,ttft_p95,ttft_p99,tpot_mean,tpot_p95,error_count,timeout_count,peak_vram_mb,preemption_count
```

---

# 18. Accuracy protocol

Không dùng bộ “GPQA-style” tự tạo để kết luận accuracy.

Phải cố gắng chạy gần nhất với BTC:

- GPQA Diamond full.
- `lm-evaluation-harness`.
- Endpoint OpenAI-compatible.
- Chat template đúng.
- Generation config đúng.
- Filter strict-match.
- Model name đúng.
- Cùng Docker image với submission.

## Bảng kết quả

| Experiment | ERS | GPQA accuracy | Delta | f(Delta) | Estimated Score |
|---|---:|---:|---:|---:|---:|
| | | | | | |

## Mốc an toàn

Nếu baseline chính thức là 0.40:

- Accuracy ≥ 0.30: không bị phạt.
- 0.24 < Accuracy < 0.30: bị phạt.
- Accuracy ≤ 0.24: điểm cuối bằng 0.

Không nên nhắm sát 0.30. Nên giữ thêm safety margin.

---

# 19. Ma trận thí nghiệm tối thiểu

## Phase A — Serving BF16

| ID | Batch tokens | Max seqs | Mem util | Prefix | Chunked |
|---|---:|---:|---:|---|---|
| A01 | 2048 | 64 | 0.95 | On | On |
| A02 | 4096 | 64 | 0.95 | On | On |
| A03 | 8192 | 64 | 0.95 | On | On |
| A04 | 16384 | 64 | 0.95 | On | On |
| A05 | Best | 32 | 0.95 | On | On |
| A06 | Best | 96 | 0.95 | On | On |
| A07 | Best | 128 | 0.95 | On | On |
| A08 | Best | Best | 0.93 | On | On |
| A09 | Best | Best | 0.97 | On | On |
| A10 | Best | Best | Best | Off | On |
| A11 | Best | Best | Best | On | Off |

## Phase B — Quantization

| ID | Weights | KV cache | Base config |
|---|---|---|---|
| B01 | BF16 | auto | Best A |
| B02 | FP8 | auto | Best A |
| B03 | BF16 | E4M3 | Best A |
| B04 | BF16 | E5M2 | Best A |
| B05 | FP8 | E4M3 | Best A |
| B06 | FP8 | E5M2 | Best A |

## Phase C — Runtime

| ID | Async | Allocator | Logging |
|---|---|---|---|
| C01 | Off | Default | INFO |
| C02 | On | Default | INFO |
| C03 | Best | Expandable segments | INFO |
| C04 | Best | Best | WARNING |

---

# 20. Quy tắc loại cấu hình

Loại ngay nếu có một trong các dấu hiệu:

```text
- OOM.
- Request lỗi.
- Timeout.
- Trả 0 token.
- Output token không đúng yêu cầu.
- Median ERS thấp hơn baseline.
- Kết quả dao động quá lớn.
- GPQA giảm nguy hiểm.
- Có hành vi khó giải trình khi hậu kiểm.
- Chỉ nhanh trên một lần chạy.
```

## Ngưỡng ổn định đề xuất

Một cấu hình tốt cần:

```text
max(ERS) - min(ERS) nhỏ
không có lỗi trong 3 lần chạy
p95 TTFT không có spike bất thường
peak VRAM có safety margin
```

---

# 21. Cách chọn 5 submissions cuối

Không chọn 5 cấu hình gần giống nhau.

## Submission 1 — Safety

```text
BF16 weights
KV auto
Prefix caching
Chunked prefill
Batch/seq đã tune
Memory utilization an toàn
```

Mục tiêu: accuracy tốt và hậu kiểm dễ.

## Submission 2 — Best BF16 ERS

```text
BF16
Serving/scheduler tốt nhất
ERS BF16 cao nhất
```

## Submission 3 — FP8 Weights

```text
FP8 weights
KV auto
Serving config tốt nhất
```

## Submission 4 — FP8 KV

```text
Weights an toàn nhất
FP8 E4M3 KV
```

## Submission 5 — Aggressive Pareto

```text
FP8 weights + FP8 KV
hoặc
một scheduler config có ERS cao hơn nhưng accuracy/risk lớn hơn
```

Mỗi submission cần có GPQA và log riêng.

---

# 22. Bảng quản lý submissions

```csv
submission_id,image,image_digest,config_id,ers_online,gpqa_local,delta,estimated_score,ttft_p95,tpot_mean,status,notes
```

Ví dụ:

| Submission | Config | ERS online | GPQA | Estimated Score | Chọn hậu kiểm? |
|---|---|---:|---:|---:|---|
| S01 | BF16-safe | | | | |
| S02 | BF16-best | | | | |
| S03 | FP8-weight | | | | |
| S04 | FP8-KV | | | | |
| S05 | Aggressive | | | | |

---

# 23. Docker và reproducibility

## 23.1. Pin image

Không chỉ dùng tag thay đổi được.

Nên lưu:

```text
repository:tag
sha256 digest
ngày build
git commit
config ID
```

## 23.2. Image public

Trước khi nộp:

```bash
docker pull <image>
docker inspect <image>
docker run ...
```

Kiểm tra từ một máy sạch hoặc tài khoản khác nếu có thể.

## 23.3. Không thay image sau submission

Mỗi submission phải ánh xạ tới đúng image đã benchmark.

---

# 24. Lịch triển khai đề xuất

## 27/07/2026 — Baseline và sweep chính

```text
[ ] Endpoint validation
[ ] BF16 baseline
[ ] Sweep max-num-batched-tokens
[ ] Sweep max-num-seqs
[ ] Sweep memory utilization
[ ] Chốt BF16 best
```

## 28/07/2026 — Quantization và runtime

```text
[ ] FP8 weights A/B
[ ] FP8 KV E4M3 A/B
[ ] FP8 KV E5M2 A/B
[ ] Async scheduling nếu hỗ trợ
[ ] Allocator/logging test
[ ] Chạy GPQA cho top candidates
```

## 29/07/2026 — Stability và submissions

```text
[ ] Chạy mỗi top config ít nhất 3 lần
[ ] Kiểm tra OOM/preemption
[ ] Chạy GPQA full
[ ] Build/push image
[ ] Pin digest
[ ] Nộp các cấu hình Pareto tốt
```

## 30/07/2026 — Chốt vòng online

```text
[ ] Không thử thay đổi lớn vào phút cuối
[ ] Xác minh leaderboard
[ ] Lưu submission IDs
[ ] Lưu image digests
[ ] Lưu docker-compose từng submission
[ ] Chuẩn bị danh sách tối đa 5 bài chọn hậu kiểm
```

---

# 25. Rủi ro chính

| Rủi ro | Tác động | Cách giảm |
|---|---|---|
| FP8 không nhanh hơn BF16 | Mất ERS và accuracy | A/B test |
| FP8 KV làm giảm GPQA | Bị phạt điểm | Thử auto/E4M3/E5M2 |
| Memory util 0.97 gây OOM | Request 0 điểm | Chạy lặp lại, ưu tiên 0.95 |
| Max seqs quá cao | TPOT tăng/preemption | Sweep 32–192 |
| Batch tokens không phù hợp | TTFT hoặc TPOT xấu | Sweep 2048–16384 |
| Prefix cache không hit như kỳ vọng | TTFT không giảm | Đo theo turn/cache hit |
| Benchmark local khác BTC | Leaderboard lệch | Dùng script và trace gần BTC nhất |
| Image tag bị thay đổi | Hậu kiểm thất bại | Pin digest |
| Custom kernel lỗi | Crash/sai output | Chỉ làm sau profiling |
| Một run may mắn | Re-grade tụt điểm | Chọn theo median |

---

# 26. Tiêu chí “on top”

Một cấu hình chỉ được coi là ứng viên top khi:

```text
[ ] Median ERS cao nhất trong ma trận
[ ] Không có lỗi trong ít nhất 3 lần chạy
[ ] p95 TTFT tốt
[ ] TPOT ổn định
[ ] Không preemption bất thường
[ ] Không OOM
[ ] GPQA nằm trong vùng an toàn
[ ] Docker image reproducible
[ ] Hành vi serving hợp lệ
```

---

# 27. Kết luận chiến lược

Giả thuyết mạnh nhất cần kiểm chứng:

```text
BF16 hoặc online FP8 nhẹ
+ KV cache auto hoặc FP8 E4M3 nếu thật sự thiếu KV
+ Prefix caching
+ Chunked prefill
+ Continuous batching mặc định của vLLM
+ max-num-batched-tokens được tune theo trace
+ max-num-seqs được tune theo traffic
+ GPU memory utilization khoảng 0.95
+ runtime ổn định
```

Không nên mặc định rằng cấu hình bật toàn bộ FP8 sẽ thắng.

Khả năng đứng top đến từ:

```text
Benchmark đúng
+ thay đổi có kiểm soát
+ hiểu workload
+ bảo vệ accuracy
+ chọn 5 điểm Pareto thông minh
+ Docker ổn định khi re-grade
```

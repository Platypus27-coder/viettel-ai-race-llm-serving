# Báo cáo giải pháp — Viettel AI Race 2026

## 1. Mục tiêu

Tối đa hóa:

\[
Score = 100 \times ERS \times f(\Delta)
\]

trên `LiquidAI/LFM2.5-1.2B-Instruct`, vLLM 0.22.1 và một MiG H200 18 GB.
Giải pháp không coi ERS online là mục tiêu duy nhất: mọi ứng viên còn phải đạt
accuracy tối thiểu 0.32 trong phép sàng lọc GPQA, không lỗi và tái lập được.

## 2. Workload dùng để tối ưu

| Thuộc tính | Giá trị |
|---|---:|
| Hội thoại | 70 |
| Lượt mỗi hội thoại | 6 |
| Tổng request | 420 |
| Shared prefix | 1.000 token |
| Prefix riêng | 1.000 token |
| User token mới mỗi lượt | 150 |
| Output mỗi lượt | 300 token |
| Arrival | Poisson, seed 42 |

Mỗi hội thoại chỉ có một request đang chạy; turn tiếp theo bắt đầu sau khi turn
trước hoàn tất. Context lớn nhất xấp xỉ 4.700 token kể cả output hiện tại, do đó
`max-model-len=8192` có safety margin nhưng tránh cấp phát theo cửa sổ 32K không
cần thiết.

Benchmark nội bộ dùng tokenizer thật, giữ nguyên output assistant trong history,
kiểm tra đủ 300 output token và tính ERS trên mẫu số cố định 420. Request lỗi,
timeout, thiếu token hoặc turn không được chạy đều đóng góp điểm 0.

## 3. Cấu hình phục hồi sau kết quả portal

```yaml
--gpu-memory-utilization=0.95
--max-model-len=8192
--enable-prefix-caching
--enable-chunked-prefill
--max-num-batched-tokens=4096
--max-num-seqs=64
```

- Prefix caching khai thác shared prefix và growing multi-turn context.
- Chunked prefill cho phép scheduler cân bằng prefill với decode.
- Cấu hình `8192/80` đã đạt 48.2, thấp hơn incumbent 60.02; không tiếp tục dùng.
- `4096/64` là ứng viên phục hồi, ưu tiên TBT hơn trong khi TTFT cũ còn dư địa.
- `shm_size` giữ ở 2 GiB trên host chỉ có 8 GB RAM.
- `OMP_NUM_THREADS=1` và `MKL_NUM_THREADS=1` hạn chế tranh chấp với API server,
  engine core và GPU worker trên ba CPU core.
- CUDA Graph và optimization level của vLLM giữ mặc định; không bật
  `--enforce-eager`.

## 4. Chiến lược ba submission còn lại

1. BF16, batch 4096, 64 sequences: ứng viên phục hồi.
2. BF16, batch 2048, 64 sequences: ưu tiên TBT.
3. Dùng batch thắng với FP8 weights nếu GPQA ít nhất 0.32 và workload đạt
   420/420; nếu không, giữ BF16 và giảm xuống 48 sequences.

Không thử lại batch 8192/80, không dùng batch 16384 và không dùng FP8 KV.
Submission 60.02 được giữ làm incumbent; submission 48.2 không được chọn nếu
không xuất hiện bằng chứng accuracy đặc biệt.

Không dùng kết quả latency T4 để chọn precision cho H200. T4 chỉ xác minh khả
năng khởi động, workload, lỗi và accuracy; ERS từ portal mới là tín hiệu hiệu
năng H200.

Nếu hai ứng viên chênh ERS dưới 0.01, chọn accuracy cao hơn. Nếu accuracy bằng
nhau, chọn p95 TTFT thấp hơn.

## 5. Accuracy Gate

- Quick test chỉ phát hiện lỗi format/API rõ ràng.
- Quyết định quantization phải dựa trên GPQA Diamond full qua `lm-eval`.
- Ngưỡng loại nội bộ: accuracy dưới 0.32.
- Mục tiêu an toàn: accuracy từ 0.35 trở lên.
- FP8 KV chỉ được dùng ở lượt 5 khi đã qua ngưỡng 0.35; cấu hình mặc định không
  dùng FP8 KV để tránh rủi ro scaling/precision không cần thiết.

## 6. Tính tái lập

Mỗi lần chạy lưu:

- Docker Compose độc lập và SHA-256.
- Log khởi động vLLM.
- ERS, TTFT/TPOT p50/p95/p99, success rate và breakdown theo turn.
- Kết quả GPQA và task alias.
- Portal submission ID.

Benchmark không prewarm shared prefix. Khi sweep nhiều request rate trên cùng
server, prefix cache phải được reset; nếu endpoint reset không khả dụng, server
được restart giữa các lần chạy.

## 7. Trạng thái xác minh

- Python source và Docker Compose được kiểm tra cú pháp local.
- Scoring, trace parser, Poisson, SSE fragmentation, growing history, output
  thiếu token và policy chọn submission có unit/mock tests.
- ERS và GPQA trên GPU vẫn là dữ liệu cần thu từ Colab/portal; tài liệu này không
  đưa ra con số hiệu năng dự báo khi chưa đo.

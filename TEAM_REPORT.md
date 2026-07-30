# Team Report — Viettel AI Race 2026

## Trạng thái hiện tại

Mô hình: `LiquidAI/LFM2.5-1.2B-Instruct` · Hạ tầng: 1 MiG H200 (18 GB VRAM,
3 CPU cores, 8 GB RAM) · Runtime: vLLM `0.22.1`.

`docker-compose.yml` ở root là **artifact submit duy nhất** và giữ nguyên v6.0
làm incumbent. Các file trong `configs/` chỉ là baseline hoặc cấu hình thử
nghiệm tái lập được.

| Phiên bản | Điểm ERS | Thay đổi được ghi nhận | TTFT p50 / p95 | TBT median | Lỗi | Kết luận |
|---|---:|---|---:|---:|---:|---|
| S01 | ~50.00 | BTC BF16 baseline | 76 / 110 ms | 4 ms | ~12 | Mốc đầu |
| S02 | 60.02 | BF16 + APC | 51 / 76 ms | 4 ms | 6 | Historical incumbent |
| S03 | 48.20 | chunked + batch 8192 + seqs 80 | — | — | — | Nhiều biến cùng đổi |
| S04 | 46.00 | chunked + batch 4096 + seqs 64 | 120 / 220 ms | 5 ms | ~25 | Nhiều biến cùng đổi |
| v3.0 | 49.19 | len 8192 + util .97 + seqs 128 | 65 / 95 ms | 6 ms | ~20 | Regression nhiều biến |
| v4.0 | 60.02 | len 8192 + util .97 | 51 / 76 ms | 4 ms | 6 | Phục hồi |
| v5.0 | 61.15 | v4 + FP8 weights | 49 / 70 ms | 4 ms | 6 | +1.13 |
| **v6.0** | **61.41** | v5 + FP8 E4M3 KV | **45 / 69 ms** | **4 ms** | **5** | **Incumbent** |
| v8.1 | 59.98 | len 5120 + seqs 48 + async + interactivity + CUDA env | 51 / 76 ms | 4 ms | 7 | Không thể quy lỗi cho một cờ |

Điểm, percentile và lỗi trên là dữ liệu đã được ghi trong team docs; workspace
không có export portal hoặc log H200 gốc. Mọi lượt mới phải lưu submission ID,
compose SHA, image digest, startup log và metrics portal vào manifest.

## v6.0 — cấu hình bảo toàn

```text
--max-model-len=8192
--gpu-memory-utilization=0.97
--quantization=fp8
--kv-cache-dtype=fp8_e4m3
--enable-prefix-caching
```

Không thay `entrypoint`, `/model`, served model name hoặc port. Không dùng
`--calculate-kv-scales`: vLLM tắt nó cho hybrid recurrent model vì calibration
có thể không đáng tin cậy.

## Các kết luận đã được sửa theo vLLM 0.22.1

1. Không khai báo `--enable-chunked-prefill` **không có nghĩa là tắt**. Với
   LFM2 có APC, vLLM dùng Mamba cache mode `align`; mode này yêu cầu chunked
   prefill. Do đó S03/S04 không chứng minh chunked prefill tự nó phá APC — hai
   lượt này còn thay batch và số sequence.
2. Trên MiG 18 GB, nếu không có override và vLLM nhìn thấy đúng dung lượng
   slice, API server dự kiến resolve `max_num_batched_tokens=2048` và
   `max_num_seqs=256`. Đây là suy luận từ source; log startup của từng image là
   bằng chứng quyết định.
3. `--block-size=32`, forced `FLASH_ATTN`, `CUDA_DEVICE_MAX_CONNECTIONS=1` và
   async scheduling chưa có A/B H200 đơn biến. Không dùng chúng trong lượt tối
   ưu chính.

Nguồn: [vLLM defaults](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/engine/arg_utils.py#L2194-L2213),
[hybrid cache mode](https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/model_executor/models/config.py#L304-L352).

## Đường bứt phá đang triển khai

LFM2 có 10 ShortConv và 6 attention blocks. Dynamic `--quantization=fp8` hiện
không truyền `quant_config` vào `ShortConv.in_proj`/`out_proj`; custom image
v0.22.1 sẽ chỉ phủ FP8 hai projection này, không sửa weights, `conv1d` hoặc
`lm_head`. Đây là A/B hẹp và chỉ có upside tăng dần, không phải đường đáng tin
cậy tới 75.

Ứng viên có upside duy nhất hiện tại là speculative decoding tách riêng khỏi
ShortConv: target giữ nguyên v6, draft `LFM2.5-350M` được bake offline, bốn
draft tokens, draft TP=1 và context 8192. Chỉ dùng nếu 420/420, greedy output
khớp parent, GPQA đạt ngưỡng, Mamba cache `align` được xác nhận và metric
`mean_acceptance_length` đạt xấp xỉ 3.5 trở lên. Không bật
`parallel_drafting` vì checkpoint 350M không được huấn luyện cho chế độ đó.

Mục tiêu 75 không chỉ là TTFT: với TTFT xấp xỉ 45 ms và tỷ lệ thành công hiện
tại, TPOT mean cần gần 2.5–2.6 ms. Bỏ năm lỗi chỉ có thể tăng khoảng dưới một
điểm. Nếu speculative không đưa TPOT H200 xuống gần 2.7 ms hoặc thấp hơn, dừng
kỳ vọng 75; khi đó chỉ profiling rồi fuse toàn bộ decode ShortConv/Mamba mới
có upside đủ lớn.

## Quy tắc chọn submission

- Giữ v6 nếu candidate không đạt 420/420, không qua smoke/GPQA, hoặc ERS thấp
  hơn.
- Mỗi lượt portal chỉ thay một nhóm biến từ parent thắng trước đó.
- T4/Colab chỉ xác nhận khả năng khởi động, workload và accuracy sơ bộ; không
  suy luận latency H200 từ T4.
- Candidate speculative phải có artifact so sánh greedy khớp hoàn toàn với
  parent không speculative trước khi được đưa vào manifest/portal.
- Chỉ chọn tối đa năm bản có đủ artifact cho hậu kiểm GPQA.

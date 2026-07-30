# Changelog — Viettel AI Race 2026

## Incumbent được bảo toàn

`v6.0` là root `docker-compose.yml` hiện tại và là bản có điểm đã ghi nhận cao
nhất: **61.41**. Nó dùng FP8 dynamic weights, FP8 E4M3 attention KV và
automatic prefix caching, với `max-model-len=8192` và
`gpu-memory-utilization=0.97`.

## Lịch sử portal đã ghi nhận

| Mốc | ERS | Thay đổi | Nhận xét đáng tin cậy |
|---|---:|---|---|
| S01 | ~50.00 | BTC BF16 baseline | Mốc đầu |
| S02 | 60.02 | BF16 + APC | TBT median 4 ms |
| S03 | 48.20 | chunked + batch 8192 + seqs 80 | Regression bị confound |
| S04 | 46.00 | chunked + batch 4096 + seqs 64 | Regression bị confound |
| v3.0 | 49.19 | len 8192 + util .97 + seqs 128 | Regression bị confound |
| v4.0 | 60.02 | len 8192 + util .97 | Phục hồi |
| v5.0 | 61.15 | + FP8 weights | Tăng 1.13 |
| v6.0 | **61.41** | + FP8 E4M3 KV | Incumbent |
| v7.0 | không nộp | block 32 + forced backend + CUDA env | Không có dữ liệu portal |
| v8.1 | 59.98 | len 5120 + seqs 48 + async + interactivity + CUDA env | Không quy nguyên nhân đơn lẻ |

## 2026-07-29 — Sửa chiến lược sau audit source

- Root v6 không khai báo scheduler knobs. Theo vLLM 0.22.1, MiG 18 GB dự kiến
  chạy API-server defaults batch 2048/seqs 256 nếu startup không override.
- LFM2 + APC dùng hybrid Mamba `align`, yêu cầu chunked prefill. Vì vậy không
  coi các lượt S03/S04 là bằng chứng rằng chunked prefill phá prefix cache.
- Dừng kế hoạch dùng `--calculate-kv-scales`; vLLM vô hiệu nó cho hybrid
  recurrent models.
- Dừng các bundle `async`, `interactivity`, forced attention backend,
  `CUDA_DEVICE_MAX_CONNECTIONS` và block size cho tới khi có A/B đơn biến.

## Thứ tự thử nghiệm mới

1. Preflight speculative draft độc lập: target v6 + draft local
   `LFM2.5-350M`, bốn draft tokens, TP=1, context 8192. Bắt buộc greedy
   equality, 420/420, GPQA và acceptance metrics trước portal.
2. ShortConv FP8 chỉ là A/B độc lập, không kết hợp với draft.
3. Parent thắng + `--max-num-batched-tokens=1536`.
4. Nếu còn lượt: parent thắng + `--max-num-batched-tokens=1024`.

Mọi lượt phải có manifest: compose SHA, image digest, startup-resolved config,
420-request result, GPQA artifact, portal ID và score.

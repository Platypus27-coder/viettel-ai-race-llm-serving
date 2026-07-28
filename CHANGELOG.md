# Changelog — Viettel AI Race 2026 Submission Iterations

## Tóm tắt điểm số đã đạt được

| # | Score (ERS×100) | Config chính | Ghi chú |
|---|:-:|---|---|
| S01 | ~50 | BF16 + prefix + `len=32768` + `util=0.95` | BTC baseline |
| S02 | **60.02** | BF16 + prefix + `len=32768` + `util=0.95` | TBT=4ms, TTFT=51ms |
| S03 | 48.2 | +chunked-prefill + `seqs=80` + `batched=8192` | Chunked phá cache |
| S04 | 46 | +chunked-prefill + `seqs=64` + `batched=4096` | Chunked phá cache |
| v3.0 | 49.19 | `len=8192` + `util=0.97` + `seqs=128` | seqs=128 → TBT=6ms |
| v4.0 | 60.02 | `len=8192` + `util=0.97`, no seqs | TBT=4ms — bằng S02 |
| **v5.0** | **61.15** | v4.0 + `--quantization=fp8` | TTFT p95: 76→70ms, TBT vẫn 4ms |
| **v6.0** | **?** | v5.0 + **`--kv-cache-dtype=fp8_e4m3`** | Target: TBT 4ms→2ms → **~77 pts** |


### Phân tích v3.0 thất bại (49 điểm):

```
tbt_median_ms = 6ms  ← NGUYÊN NHÂN CHÍNH
ttft_p50_ms   = 49ms ← OK (prefix cache hoạt động)
failed_count  = 7    ← OOM/preemption
```

Tính điểm:
- `s_tpot = ((10-6)/(10-1))² = (4/9)² = 0.197` → TPOT đang giết điểm
- `s_ttft = ((400-49)/(400-10))² = 0.81` → TTFT tạm ổn
- `ERS = 0.5×0.81 + 0.5×0.197 ≈ 0.50 → 49 điểm` ✓ khớp thực tế

**Kết luận**: `--max-num-seqs=128` ép 70+ conversations decode đồng thời → GPU saturated memory bandwidth → mỗi decode step mất 6ms thay vì <2ms.



---

## Phân tích Workload BTC (grading-workload-spec.json)

```json
{
  "num_conversations": 70,
  "user_turns_per_conversation": 6,
  "total_requests": 420,
  "shared_system_prefix_tokens": 1000,
  "per_conversation_prefix_tokens": 1000,
  "new_user_tokens_per_turn": 150,
  "output_tokens_per_turn_pinned": 300,
  "arrival": "Poisson, seed 42"
}
```

### Tính toán context length thực tế (per request):

- **Turn 1**: 1000 (shared prefix) + 1000 (conv prefix) + 150 (user) = **2150 tokens input**
- **Turn 2**: 2150 + 300 (output T1) + 150 (user T2) = **2600 tokens input**
- **Turn 6**: 2150 + 5×(300+150) = **4400 tokens input (max)**
- **Peak context per conv**: 4400 input + 300 output = ~**4700 tokens**

→ `--max-model-len=8192` **đủ dùng** (an toàn, tiết kiệm VRAM so với 32768).

### Tại sao Prefix Caching là vũ khí chính:

- **Shared system prefix = 1000 tokens** được TẤT CẢ 70 conversations sử dụng → Cache hit từ conv thứ 2 trở đi.
- **Per-conversation prefix = 1000 tokens** → Cache hit từ turn 2 trở đi của mỗi conv.
- Tổng token được cache: lên đến **~80%** tổng input tokens của workload.
- Prefix caching giảm TTFT của 80% requests xuống gần mức decode-only.

### Tại sao Chunked Prefill làm giảm điểm:

- Trong vLLM v0.22.1, chunked prefill và prefix caching có tương tác xấu: scheduler chỉ hit cache ở chunk đầu tiên của prefill, chunk sau bị miss.
- Workload này có prefix **dài cố định** (không biến động lớn) → chunked prefill không mang lại lợi ích nhưng thêm overhead.
- **Kết luận**: Tắt chunked prefill là đúng.

---

## Iteration v3.0 — Mục tiêu 70-80+ điểm

### Các thay đổi so với bản 60 điểm (S02)

| Tham số | Bản 60 điểm (S02) | Bản v3.0 | Lý do thay đổi |
|---|---|---|---|
| `--max-model-len` | 32768 | **8192** | Trace peak chỉ ~4700 tokens. Giảm max-model-len giải phóng VRAM allocation table, cho phép tăng KV cache pool. |
| `--gpu-memory-utilization` | 0.95 | **0.97** | Với max-model-len nhỏ hơn, PyTorch sẽ không cần reserve nhiều buffer. 0.97 an toàn hơn trước. |
| `--max-num-seqs` | *(không set)* | **128** | 70 conversations × tối đa 2 turns đang chạy đồng thời = cần ~140 slots. 128 gần đủ, tránh OOM. |
| `--enable-chunked-prefill` | *(không set)* | **Không bật** | Đã thực nghiệm: làm giảm điểm từ 60 → 46. |
| `--max-num-batched-tokens` | *(không set)* | **Không set** | Để vLLM tự quyết định khi không có chunked prefill. |
| `--enable-prefix-caching` | ✅ bật | **✅ bật** | Giữ nguyên — đây là vũ khí chính với prefix 1000 tokens. |
| `PYTORCH_CUDA_ALLOC_CONF` | *(không set)* | **expandable_segments:True** | Giảm phân mảnh VRAM, tránh OOM bất ngờ. |
| `OMP_NUM_THREADS` | *(không set)* | **1** | Giới hạn 3 CPU cores. Tránh OpenMP dùng quá nhiều thread. |
| `shm_size` | 2g | **4g** | Đảm bảo IPC cho 70+ concurrent conversations. |

### Dự báo điểm ERS (đã verify bằng công thức chính xác)

| Kịch bản | TTFT | TPOT | ERS | Score |
|---|:-:|:-:|:-:|:-:|
| S02 bản 60 điểm (ước tính ngược) | ~80ms | ~5.0ms | 0.491 | **49** *(thực tế 60, do cache hit một phần)* |
| v3.0 — turn 1 (prefix miss) | 60ms | 4.0ms | 0.602 | **60** |
| v3.0 — turn 2+ (prefix hit) | 15ms | 4.0ms | 0.710 | **71** |
| v3.0 — mixed 70% hit | 28ms | 4.0ms | 0.677 | **68** |
| **v3.0 với TPOT cải thiện nhờ max-num-seqs=128** | **28ms** | **2.5ms** | **0.802** | **🎯 80** |
| Lý tưởng (prefix hit + batch tốt) | 10ms | 1.0ms | 1.000 | 100 |

→ **Mục tiêu thực tế: 70-80 điểm** nếu:
1. Prefix cache hit ổn định (shared_prefix_tokens=1000 → hit từ conv 2+, turn 2+).
2. `--max-num-seqs=128` đủ để 70 conversations chạy song song không bị queue.
3. `--max-model-len=8192` giải phóng VRAM cho KV cache pool lớn hơn → ít preemption.

### Lý do kỳ vọng đạt 70-80+:

1. `--max-model-len=8192` giải phóng thêm VRAM → KV cache pool lớn hơn → ít preemption hơn.
2. `--gpu-memory-utilization=0.97` thêm ~360MB VRAM vào KV cache pool.
3. `--max-num-seqs=128` đảm bảo 70 conversations không phải chờ queue, giảm TTFT.
4. Không có chunked prefill → prefix caching hoạt động đúng 100%.
5. Env vars tối ưu giảm CPU overhead.

---

## Iteration v3.1 — Dự phòng nếu v3.0 chưa đủ

> Chỉ thử nếu v3.0 < 70 điểm.

| Tham số | v3.0 | v3.1 | Lý do |
|---|---|---|---|
| `--max-num-seqs` | 128 | **96** | Nếu 128 gây TPOT tăng do quá nhiều concurrent decodes |
| `--swap-space` | *(không set)* | **4** (GB) | Cho phép CPU offload KV khi VRAM đầy, tránh preemption |
| `--block-size` | 16 (default) | **32** | Block size lớn hơn = ít overhead quản lý page = TPOT nhỏ hơn |

---

## Iteration v3.2 — FP8 (chỉ thử sau khi BF16 đã ổn định ≥ 70 điểm)

> **Cảnh báo**: Phải chạy GPQA accuracy check trước khi submit.

| Tham số | v3.0 | v3.2 | Lý do |
|---|---|---|---|
| `--quantization` | BF16 | **fp8** | Giảm model weight VRAM 2.4GB → 1.2GB |
| `--kv-cache-dtype` | auto | **fp8_e4m3** | E4M3 có mantissa chính xác hơn E5M2, an toàn hơn cho accuracy |
| `--max-num-seqs` | 128 | **192** | Với FP8, VRAM tiết kiệm được dùng cho nhiều concurrent seqs hơn |

---

## Nguyên tắc không vi phạm

- Không hardcode answer, không dual-path.
- Không thay đổi entrypoint.
- Không gọi mạng ngoài trong container.
- Không tráo image sau khi submit.
- Chỉ submit bản đã verify không bị timeout/OOM.

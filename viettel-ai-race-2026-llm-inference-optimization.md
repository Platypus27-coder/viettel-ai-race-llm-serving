# Viettel AI Race 2026 — LLM Inference Optimization Challenge

> **Vòng 1 — Sơ loại:** 02/07/2026 đến 30/07/2026  
> **Đề thi:** LLM Inference Optimization  
> **Cập nhật nhiệm vụ:** 18/07/2026  
> **Serving framework bắt buộc:** vLLM

---

## 1. Tổng quan cuộc thi

Viettel AI Race 2026 — LLM Inference Optimization Challenge mô phỏng một bài toán thực tế trong hạ tầng AI doanh nghiệp:

> Làm thế nào để phục vụ một mô hình ngôn ngữ lớn với độ trễ thấp, thông lượng cao, chất lượng đầu ra ổn định và tài nguyên GPU hữu hạn?

Thí sinh phải triển khai một LLM inference server, đóng gói thành Docker image và tối ưu hệ thống để xử lý workload multi-turn mô phỏng traffic production.

Trong vòng online, hệ thống chỉ chấm hiệu năng phục vụ thông qua **Effective Request Score (ERS)**. Sau khi vòng online kết thúc, mỗi đội được chọn tối đa 5 submissions để Ban Tổ chức hậu kiểm và đánh giá độ chính xác bằng GPQA Diamond full.

Điểm chính thức phụ thuộc đồng thời vào:

- Hiệu năng phục vụ: TTFT và TPOT.
- Khả năng xử lý đầy đủ request, không lỗi và không timeout.
- Mức suy giảm độ chính xác so với baseline BF16.
- Tính hợp lệ của giải pháp theo quy định phòng chống gian lận.

---

## 2. Nhiệm vụ

Triển khai và tối ưu inference server cho mô hình:

```text
LiquidAI/LFM2.5-1.2B-Instruct
```

Nguồn weights:

```text
https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
```

Server phải xử lý workload trace gồm nhiều hội thoại chạy đồng thời, trong đó mỗi hội thoại có nhiều lượt hỏi đáp và lượt tiếp theo chỉ được gửi sau khi lượt trước hoàn tất.

### Mục tiêu vòng online

Tối đa hóa:

```text
ERS — Effective Request Score
```

ERS đánh giá hiệu năng dựa trên:

- **TTFT — Time To First Token:** thời gian từ lúc gửi request đến khi nhận token đầu tiên.
- **TPOT — Time Per Output Token:** thời gian trung bình giữa các token đầu ra liên tiếp.

### Mục tiêu cuối cùng

Tối đa hóa điểm:

\[
Score = 100 \times ERS \times f(\Delta)
\]

Trong đó:

- \(ERS\) là điểm hiệu năng đã được chấm trong vòng online.
- \(\Delta\) là mức suy giảm accuracy so với baseline BF16.
- \(f(\Delta)\) là hệ số phạt accuracy.
- Submission phải vượt qua bước hậu kiểm tính hợp lệ.

---

## 3. Lịch thi

| Giai đoạn | Thời gian |
|---|---|
| Vòng 1 — Sơ loại | 02/07/2026 — 30/07/2026 |
| Chấm online | Thực hiện sau mỗi submission hợp lệ |
| Chọn submissions hậu kiểm | Sau khi vòng online kết thúc |
| Accuracy Gate | Chạy trên tối đa 5 submissions do đội lựa chọn |
| Chốt kết quả | Sau hậu kiểm và đánh giá GPQA Diamond full |

---

## 4. Hạ tầng đánh giá

Toàn bộ benchmark được thực hiện tự động trên hạ tầng của Ban Tổ chức.

Mỗi lượt chấm được cấp:

| Thành phần | Cấu hình |
|---|---|
| GPU | 1 instance MiG NVIDIA H200 |
| VRAM | 18 GB |
| CPU | 3 cores |
| RAM | 8 GB |
| Hệ điều hành host | Ubuntu 24.04 LTS |
| NVIDIA Driver | 590.x |
| CUDA compatibility | CUDA 13.x |
| Tensor Parallel Size | 1 |
| Serving framework | vLLM |
| Model | LFM2.5-1.2B-Instruct |

### Giới hạn quan trọng

- Chỉ có 18 GB VRAM.
- Chỉ có 3 CPU cores.
- Chỉ có 8 GB RAM.
- Không được gọi mạng bên ngoài trong quá trình chấm.
- Docker image phải chứa đầy đủ dependency cần thiết.
- Giải pháp phải chạy trên đúng một MiG instance.
- Serving framework bắt buộc là vLLM.

---

## 5. Workload trace

Workload mô phỏng traffic production với nhiều hội thoại multi-turn chạy đồng thời.

### Các trường trong trace

| Trường | Ý nghĩa |
|---|---|
| `num_conversations` | Số hội thoại độc lập chạy đồng thời |
| `user_turns_per_conversation` | Số lượt hỏi của user trong mỗi hội thoại |
| `total_request` | Tổng số request trong workload |
| `shared_system_prefix_tokens` | Số token của system prefix dùng chung cho các hội thoại |
| `per_conversation_prefix_tokens` | Số token ngữ cảnh riêng của từng hội thoại, được thêm vào turn 1 |
| `new_user_tokens_per_turn` | Số token prompt mới của user tại mỗi turn |
| `output_tokens_per_turn_pinned` | Số token output được ấn định cho mỗi turn |
| `arrival` | Nhịp đến của request |

### Đặc điểm workload

- Các hội thoại chạy đồng thời.
- Mỗi hội thoại có nhiều lượt.
- Turn tiếp theo phụ thuộc vào kết quả của turn trước.
- Turn 1 bao gồm:
  - Shared system prefix.
  - Per-conversation prefix.
  - User tokens của turn hiện tại.
- Các turn sau tái sử dụng context từ lịch sử hội thoại.
- Output token được pin theo trace.
- Arrival pattern mô phỏng traffic thực tế.
- Trace công khai có thể chỉ chứa metadata về độ dài và thời điểm, trong khi prompt thật được giữ lại để chấm.

---

## 6. Effective Request Score — ERS

ERS là điểm trung bình của toàn bộ request:

\[
ERS = \frac{1}{N}\sum_{i=1}^{N} S_{\text{request},i}
\]

Trong đó:

- \(N\) là tổng số request.
- \(S_{\text{request},i}\) nằm trong khoảng từ 0 đến 1.

### Điểm của một request

\[
S_{\text{request}} =
\begin{cases}
0, & \text{nếu lỗi, timeout hoặc trả về 0 token}\\
w \times s_{\text{ttft}} + (1-w) \times s_{\text{tpot}}, & \text{nếu xử lý thành công}
\end{cases}
\]

Hai thành phần TTFT và TPOT có trọng số bằng nhau vì:

\[
w = 0.5
\]

---

## 7. Công thức điểm TTFT

\[
s_{\text{ttft}}
=
\left[
\operatorname{clamp}
\left(
\frac{C_{\text{ttft}}-\text{TTFT}}
{C_{\text{ttft}}-F_{\text{ttft}}},
0,
1
\right)
\right]^\gamma
\]

Với:

| Tham số | Giá trị |
|---|---:|
| \(F_{\text{ttft}}\) | 10 ms |
| \(C_{\text{ttft}}\) | 400 ms |
| \(\gamma\) | 2 |

### Ý nghĩa

- TTFT ≤ 10 ms: nhận điểm TTFT tối đa.
- TTFT ≥ 400 ms: điểm TTFT bằng 0.
- TTFT nằm giữa hai mốc: nội suy liên tục.
- Do \(\gamma = 2\), độ trễ tăng sẽ bị phạt theo hàm bình phương.

---

## 8. Công thức điểm TPOT

\[
s_{\text{tpot}}
=
\left[
\operatorname{clamp}
\left(
\frac{C_{\text{tpot}}-\text{TPOT}_{\text{mean}}}
{C_{\text{tpot}}-F_{\text{tpot}}},
0,
1
\right)
\right]^\gamma
\]

Với:

| Tham số | Giá trị |
|---|---:|
| \(F_{\text{tpot}}\) | 1 ms |
| \(C_{\text{tpot}}\) | 10 ms |
| \(\gamma\) | 2 |

### Ý nghĩa

- TPOT ≤ 1 ms: nhận điểm TPOT tối đa.
- TPOT ≥ 10 ms: điểm TPOT bằng 0.
- TPOT nằm giữa hai mốc: nội suy liên tục.
- TPOT có ảnh hưởng tương đương TTFT trong ERS.

---

## 9. Accuracy Gate

Accuracy Gate không được chạy sau từng lượt nộp online.

Leaderboard online chủ yếu phản ánh ERS và chưa phải kết quả cuối cùng.

Sau khi vòng online kết thúc:

1. Mỗi đội chọn thủ công tối đa 5 submissions.
2. Ban Tổ chức hậu kiểm tính hợp lệ.
3. Ban Tổ chức dựng lại endpoint từ đúng Docker image đã nộp.
4. Chạy GPQA Diamond full bằng `lm_eval` hoặc `bench-gpqa-diamond.sh`.
5. So sánh accuracy submission với baseline BF16.
6. Áp dụng hệ số phạt.
7. Chọn Score tốt nhất trong các submission hợp lệ.

### Baseline accuracy

Accuracy baseline BF16 mặc định:

\[
Accuracy_{\text{baseline}} = 0.4
\]

### Mức suy giảm accuracy

\[
\Delta
=
Accuracy_{\text{baseline}}
-
Accuracy_{\text{submission}}
\]

### Hàm phạt accuracy

\[
f(\Delta)
=
\begin{cases}
1.0, & \Delta \leq 0.10\\
1.0-\frac{\Delta-0.10}{0.06}, & 0.10 < \Delta < 0.16\\
0.0, & \Delta \geq 0.16
\end{cases}
\]

### Diễn giải

| Accuracy drop | Hệ số |
|---|---:|
| \(\Delta \leq 0.10\) | 1.0 |
| \(0.10 < \Delta < 0.16\) | Giảm tuyến tính |
| \(\Delta \geq 0.16\) | 0.0 |

Nếu baseline là 0.4:

- Accuracy submission ≥ 0.30: không bị phạt.
- Accuracy submission nằm giữa 0.24 và 0.30: bị giảm điểm tuyến tính.
- Accuracy submission ≤ 0.24: điểm cuối bằng 0.

---

## 10. Điểm chính thức

Điểm của mỗi submission hợp lệ:

\[
Score = 100 \times ERS \times f(\Delta)
\]

Điểm của đội là Score cao nhất trong các submission:

- Được đội lựa chọn.
- Vượt qua hậu kiểm.
- Được chạy GPQA Diamond full.
- Không vi phạm quy định.

> Lưu ý: Trong một số mô tả có thể xuất hiện chữ `ERC`; theo ngữ cảnh và công thức chính thức, chỉ số đúng là `ERS`.

---

## 11. Không gian tối ưu được phép

Thí sinh chỉ được sử dụng vLLM, nhưng được phép tối ưu nhiều lớp trong serving stack.

### 11.1. Quantization

Các kỹ thuật online quantization có thể được sử dụng nếu không sửa weights trái phép và vẫn bảo đảm Accuracy Gate.

Ví dụ:

- Weight quantization.
- Activation quantization.
- FP8.
- INT8.
- KV cache quantization.
- Mixed precision.

### 11.2. KV cache và memory

Các hướng được phép:

- Paged Attention.
- FP8 KV cache.
- INT8 KV cache.
- Prefix caching.
- Semantic caching.
- Memory-aware scheduling.
- KV cache offloading xuống CPU hoặc NVMe.
- Tối ưu block size.
- Tối ưu tỷ lệ sử dụng GPU memory.
- Tối đa hóa số request chạy đồng thời.

### 11.3. Serving và scheduling

Các hướng được phép:

- Dynamic batching.
- Continuous batching.
- Chunked prefill.
- Request scheduling.
- Memory-aware scheduling.
- Speculative decoding.
- Điều chỉnh số sequence đồng thời.
- Điều chỉnh số token tối đa trong một batch.
- Ưu tiên prefill hoặc decode tùy workload.

### 11.4. System và runtime

Các hướng được phép:

- Custom CUDA kernels.
- Custom Triton kernels.
- Fused kernels.
- FlashAttention.
- FlashInfer.
- CUDA Graphs.
- Memory layout optimization.
- Kernel fusion.
- Giảm CPU overhead.
- Giảm synchronization overhead.

---

## 12. Quy trình nộp bài

### Bước 1 — Develop & Package

Phát triển giải pháp và đóng gói toàn bộ thành Docker image.

Docker image cần chứa:

- vLLM.
- Dependency cần thiết.
- Code tùy chỉnh.
- Custom kernels nếu có.
- Cấu hình runtime.
- Entry point khởi động OpenAI-compatible API server.

### Bước 2 — Push Image

Đẩy Docker image lên Docker Hub cá nhân hoặc tổ chức.

Yêu cầu:

- Repository ở chế độ public.
- Image tag hoặc digest phải rõ ràng.
- Không thay đổi hoặc tráo image sau khi nộp.
- Nên pin image bằng digest để bảo đảm khả năng tái lập.

### Bước 3 — Submit

Nộp file:

```text
docker-compose.yml
```

File phải khai báo:

- Docker image chính xác.
- Entrypoint.
- Command khởi động.
- Port 8000.
- GPU reservation.
- Shared memory.
- Các tham số vLLM.

### Bước 4 — Automated Evaluation

Hệ thống của Ban Tổ chức tự động:

1. Pull Docker image.
2. Tạo container.
3. Cấp MiG H200.
4. Chạy healthcheck.
5. Gửi workload vào endpoint.
6. Đo TTFT và TPOT.
7. Tính ERS.
8. Cập nhật leaderboard.

### Bước 5 — Post-online Evaluation

Sau vòng online:

1. Đội chọn tối đa 5 submissions.
2. Ban Tổ chức hậu kiểm image, cấu hình, log và hành vi serving.
3. Chạy GPQA Diamond full.
4. Tính accuracy drop.
5. Áp dụng hệ số phạt.
6. Chốt Score chính thức.

---

## 13. Docker image baseline

Docker image baseline:

```text
vllm/vllm-openai:v0.22.1
```

Digest được Ban Tổ chức cung cấp:

```text
sha256-55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5
```

---

## 14. Docker Compose mẫu

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
      - --max-model-len=32768
      - --gpu-memory-utilization=0.95
      - --tensor-parallel-size=1
      - --enable-prefix-caching
    ports:
      - "8000:8000"
    shm_size: "2g"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Các dòng không nên thay đổi

Ban Tổ chức yêu cầu giữ cách khởi động bằng:

```yaml
entrypoint:
  - python3
  - -m
  - vllm.entrypoints.openai.api_server
```

Không thay bằng `vllm-server`.

Các giá trị cốt lõi cần giữ:

```yaml
--model=/model
--served-model-name=LFM2.5-1.2B-Instruct
--host=0.0.0.0
--port=8000
```

Các tham số tối ưu như `max-model-len`, `gpu-memory-utilization`, prefix caching, KV cache dtype, batching và scheduling có thể được điều chỉnh nếu phù hợp với luật thi.

---

## 15. Yêu cầu endpoint

Server phải:

- Khởi động thành công trong container.
- Lắng nghe tại `0.0.0.0:8000`.
- Có giao diện OpenAI-compatible.
- Serve đúng model name:

```text
LFM2.5-1.2B-Instruct
```

- Trả output dạng streaming theo yêu cầu benchmark.
- Không lỗi trong quá trình phục vụ.
- Không timeout.
- Không trả 0 token.
- Tạo đủ số token output theo request.
- Không gọi dịch vụ bên ngoài.
- Chạy được trên đúng tài nguyên được cấp.

---

## 16. Quy định phòng chống gian lận

Giải pháp phải thực hiện inference trung thực tại thời điểm request được gửi đến.

### Các hành vi bị cấm

#### Pre-bake hoặc hardcode

- Tính trước câu trả lời.
- Hardcode output theo request.
- Lưu sẵn đáp án.
- Nhận diện trace để trả kết quả đã chuẩn bị.

#### Dual-path

- Dùng một luồng xử lý khi benchmark latency.
- Dùng luồng khác khi chạy accuracy.
- Thay đổi hành vi dựa trên dấu hiệu nhận biết bài chấm.

#### Gaming metrics

- Trả token rỗng.
- Cố tình cắt ngắn output.
- Trả output không hợp lệ để giảm latency.
- Lách cách đo TTFT hoặc TPOT.
- Tạo phản hồi giả không xuất phát từ quá trình inference thật.

#### Can thiệp trái phép

- Gọi mạng bên ngoài.
- Sửa tokenizer trái quy định.
- Sửa weights trái quy định.
- Can thiệp hạ tầng chấm.
- Làm bẩn hoặc chiếm dụng tài nguyên.
- Tráo Docker image sau khi nộp.
- Làm rò rỉ dữ liệu chấm.

---

## 17. Hậu kiểm

Leaderboard online chưa phải kết quả cuối cùng.

Ban Tổ chức có quyền kiểm tra:

- Docker image.
- Image digest.
- Docker Compose.
- Entrypoint.
- Command line.
- Source code.
- Binary.
- Custom kernels.
- Log.
- Cấu hình vLLM.
- Luồng xử lý request.
- Generation behavior.
- Output token.
- Hành vi cache.
- Hành vi quantization.
- Dấu hiệu hardcode hoặc dual-path.

Submission vi phạm có thể:

- Bị void kết quả.
- Bị điều chỉnh thứ hạng.
- Bị thu hồi điểm.
- Bị loại khỏi cuộc thi.

---

## 18. Re-grade

Ban Tổ chức có quyền chấm lại độc lập trên đúng Docker image đã chốt.

Có thể:

- Chạy lại nhiều lần.
- Lấy điểm trung vị.
- Ưu tiên kiểm tra các đội top đầu.
- Rà soát kỹ các đội cạnh tranh sát nhau.
- Hủy kết quả nếu phát hiện vi phạm.

Do đó, giải pháp cần:

- Có tính tái lập.
- Không phụ thuộc trạng thái ngẫu nhiên bất thường.
- Không có lỗi memory không ổn định.
- Không có hành vi chỉ tốt trong một lần chạy.
- Không phụ thuộc vào cache được tạo từ trước.

---

## 19. Tie-break

Nếu các đội có điểm gần nhau trong vùng nhiễu khoảng 1–3 điểm, Ban Tổ chức phân định theo thứ tự:

1. Mức suy giảm accuracy thấp hơn.
2. p95 TTFT thấp hơn.
3. Tốc độ sinh văn bản cao hơn.
4. Thời điểm nộp bài hợp lệ sớm hơn.

Điều này có nghĩa:

- Không nên đánh đổi accuracy quá lớn chỉ để tăng ERS nhỏ.
- Không nên chỉ tối ưu mean TTFT mà bỏ qua p95 TTFT.
- Giải pháp ổn định có lợi thế khi re-grade.
- Submission tốt nên được nộp sớm và pin image rõ ràng.

---

## 20. Khiếu nại

Trước khi chốt bảng xếp hạng chính thức, Ban Tổ chức có thể gửi email thông báo kết quả dự kiến.

Các đội có tối đa:

```text
24 giờ
```

kể từ khi nhận thông báo hoặc kể từ khi kết quả phase được công bố để gửi khiếu nại.

Khi khiếu nại, đội nên chuẩn bị:

- Submission ID.
- Docker image tag hoặc digest.
- Docker Compose đã nộp.
- Mô tả cấu hình.
- Log nội bộ.
- Kết quả benchmark nội bộ.
- Bằng chứng về tính hợp lệ.
- Nội dung cần Ban Tổ chức kiểm tra lại.

---

## 21. Những yêu cầu cốt lõi cần ghi nhớ

### Bắt buộc

- Sử dụng vLLM.
- Serve LFM2.5-1.2B-Instruct.
- Đóng gói bằng Docker.
- Public image trên Docker Hub.
- Nộp Docker Compose hợp lệ.
- Chạy trên 1 MiG H200 18 GB.
- Endpoint tại port 8000.
- OpenAI-compatible API.
- Inference trung thực.
- Không gọi mạng bên ngoài.
- Không sửa tokenizer hoặc weights trái phép.
- Không hardcode hoặc dual-path.
- Bảo đảm accuracy đủ cao sau vòng online.

### Mục tiêu hiệu năng

- TTFT càng gần hoặc thấp hơn 10 ms càng tốt.
- TPOT càng gần hoặc thấp hơn 1 ms càng tốt.
- Tránh TTFT từ 400 ms trở lên.
- Tránh TPOT từ 10 ms trở lên.
- Không để request lỗi, timeout hoặc trả 0 token.
- Tận dụng prefix chung và context multi-turn.
- Tối ưu concurrency trong giới hạn 18 GB VRAM.

### Mục tiêu accuracy

- Giữ accuracy submission ít nhất khoảng 0.30 để tránh bị phạt, nếu baseline chính thức là 0.40.
- Tránh accuracy giảm xuống 0.24 hoặc thấp hơn.
- Mọi cấu hình quantization phải được đánh giá bằng GPQA trước khi chọn submission cuối.

---

## 22. Tóm tắt ngắn

Viettel AI Race 2026 — LLM Inference Optimization Challenge yêu cầu thí sinh tối ưu một server vLLM phục vụ mô hình LFM2.5-1.2B-Instruct trên một MiG H200 18 GB.

Trong vòng online, bài thi chỉ chấm ERS dựa trên TTFT và TPOT. Tuy nhiên, leaderboard online chưa phản ánh điểm chính thức vì sau vòng thi, Ban Tổ chức sẽ chạy GPQA Diamond full trên tối đa 5 submissions được đội lựa chọn.

Một submission chỉ có khả năng đạt điểm cao khi đồng thời:

1. Có TTFT thấp.
2. Có TPOT thấp.
3. Không lỗi hoặc timeout.
4. Duy trì accuracy gần baseline BF16.
5. Vượt qua hậu kiểm chống gian lận.
6. Có khả năng tái lập khi Ban Tổ chức chấm lại.

Công thức cuối cùng:

\[
Score = 100 \times ERS \times f(\Delta)
\]

Do đó, bản chất của cuộc thi không chỉ là làm inference nhanh nhất, mà là tìm ra cấu hình serving nhanh nhất vẫn duy trì được chất lượng đầu ra và đáp ứng đúng tinh thần production.

---

## 23. Nguồn tham khảo chính thức

- Đề bài và quy định Viettel AI Race 2026.
- Bản cập nhật Đề 3 — LLM Inference Optimization.
- Portal vòng sơ loại.
- Model `LiquidAI/LFM2.5-1.2B-Instruct`.
- Docker image baseline `vllm/vllm-openai:v0.22.1`.

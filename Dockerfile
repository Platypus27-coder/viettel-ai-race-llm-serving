# ============================================================
# Viettel AI Race 2026 — LLM Inference Optimization
# Custom Docker Image based on vLLM v0.22.1
# Model: LiquidAI/LFM2.5-1.2B-Instruct
# GPU: 1x MiG H200 18GB VRAM
# ============================================================

FROM vllm/vllm-openai:v0.22.1

# ---- System-level optimizations ----
# Reduce Python startup overhead
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# PyTorch memory allocator optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA optimizations
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# vLLM performance tuning via environment variables
# Use FlashInfer attention backend if available (faster decode)
# Falls back gracefully if not supported for this model
ENV VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Disable usage stats to reduce startup overhead
ENV VLLM_NO_USAGE_STATS=1
ENV DO_NOT_TRACK=1

# Reduce logging overhead during serving
ENV VLLM_LOGGING_LEVEL=WARNING

# ---- Optional: Pre-download model into image ----
# Uncomment these lines if you want to bake model weights into the image
# (faster cold start, but larger image ~2.5GB+)
# RUN pip install huggingface-hub && \
#     huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct \
#       --local-dir /model \
#       --local-dir-use-symlinks False

# ---- Health check ----
HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

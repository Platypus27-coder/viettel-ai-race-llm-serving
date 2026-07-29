# syntax=docker/dockerfile:1
#
# Custom vLLM image for the Viettel AI Race.
#
# The linux/amd64 digest below is the manifest for
# vllm/vllm-openai:v0.22.1. Pinning it makes the vLLM implementation being
# patched deterministic; do not replace it with a floating tag.
FROM vllm/vllm-openai:v0.22.1@sha256:55c9bcee9fc66644b139fddae8a7a03e4c0c8a25ab5c64b0ce614554a8abf5d5

ARG BAKE_DRAFT_MODEL=0
ARG DRAFT_MODEL_ID=LiquidAI/LFM2.5-350M
# Immutable checkpoint revision verified against all required files on
# 2026-07-29. Override only with another immutable commit SHA.
ARG DRAFT_MODEL_REVISION=1575d1b8b67d862834836087765bff2ef4020672

COPY docker/shortconv-fp8/patch_vllm_shortconv_fp8.py /opt/vllm-patches/
COPY docker/shortconv-fp8/bake_draft_model.py /opt/vllm-patches/

# Refuse to patch a different vLLM release and fail the build if the expected
# source anchors drift. The patch only quantizes ShortConv's two GEMM
# projections; conv1d remains untouched.
RUN python3 -B /opt/vllm-patches/patch_vllm_shortconv_fp8.py --apply --verify

# The default path does not download a second model. Supplying
# --build-arg BAKE_DRAFT_MODEL=1 downloads the pinned draft checkpoint into
# /opt/draft/LFM2.5-350M at build time, so serving can remain fully offline.
RUN BAKE_DRAFT_MODEL="${BAKE_DRAFT_MODEL}" \
    DRAFT_MODEL_ID="${DRAFT_MODEL_ID}" \
    DRAFT_MODEL_REVISION="${DRAFT_MODEL_REVISION}" \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/tmp/hf-download \
    python3 -B /opt/vllm-patches/bake_draft_model.py \
    && rm -rf /tmp/hf-download

# The contest mounts the target model at /model. These guards ensure the
# runtime never falls back to downloading model assets or telemetry.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_LOGGING_LEVEL=WARNING

HEALTHCHECK --interval=5s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

EXPOSE 8000

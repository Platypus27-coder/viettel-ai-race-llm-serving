# Configuration Matrix

This directory contains reference configurations for the Viettel AI Race 2026 inference challenge.

## Active Artifacts

- **Root Artifact**: `docker-compose.yml`  
  The active submission file containing the peak-performing **v6.0 profile** (Score: **61.4100**, FP8 Weights + FP8 KV Cache E4M3).
- **Baseline Reference**: `configs/docker-compose.baseline.yml`  
  The official BTC baseline configuration (BF16, prefix caching, 0.95 GPU utilization).

## Cleaned Obsolete Configurations

All outdated/failed candidate configurations (such as the 48.2-point `batch8192/seqs80` run, legacy slot candidates, and experimental flags) have been cleaned up to maintain a lean, reproducible codebase.

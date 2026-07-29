# Viettel AI Race 2026 — LLM Serving

This repository is the reproducible serving workflow for
`LiquidAI/LFM2.5-1.2B-Instruct` on the contest's one 18 GB MiG H200 slice
with vLLM `0.22.1`.

`docker-compose.yml` at the repository root is the **only portal artifact**.
It is intentionally left on the scored v6 incumbent (61.41). Files generated
under `artifacts/` are review/preflight inputs; promote one to the root only
when it is the exact Compose file about to be uploaded.

## Current strategy

The online evidence says v6 is the safest incumbent:

```text
--max-model-len=8192
--gpu-memory-utilization=0.97
--quantization=fp8
--kv-cache-dtype=fp8_e4m3
--enable-prefix-caching
```

The first controlled challenger is a custom image that adds vLLM dynamic FP8
coverage to `ShortConv.in_proj` and `ShortConv.out_proj`. It makes no
scheduler, attention-backend, CUDA-environment, KV-scale, or model-weight
change. The patch is intentionally narrow because LFM2 has ten ShortConv
blocks and only six attention blocks.

Do not infer H200 latency from a Colab T4: T4 is used only for server startup,
the 420-request workload, and accuracy screening. Native FP8 W8A8 throughput
is a Hopper/Ada capability.

## Layout

```text
docker-compose.yml              active v6 portal artifact
Dockerfile                      digest-pinned vLLM 0.22.1 custom image
docker/shortconv-fp8/           fail-closed ShortConv patch and optional draft bake
configs/                        BTC baseline/reference only
benchmark/                      faithful workload, comparison, accuracy, manifest
scripts/                        render, record, build, and local-run helpers
notebooks/colab_benchmark.ipynb reproducible Colab preflight
TEAM_REPORT.md                  portal evidence and decision record
CHANGELOG.md                    controlled experiment history
```

## Before every portal attempt

Run local Python only through the supplied Conda environment:

```powershell
conda run -n viettel python -m pip install -r requirements.txt
conda run -n viettel python -m unittest discover -s tests -v
```

The Windows workstation is for tests and client tools. Build and serve the
image on a Linux GPU runner or through the contest-compatible Docker setup.
The serving process must have the contest-mounted target model at `/model` and
must not need the network.

For Colab, open
[the validation notebook](https://colab.research.google.com/github/Platypus27-coder/viettel-ai-race-llm-serving/blob/main/notebooks/colab_benchmark.ipynb).
It clones this repository and installs the official CUDA 12.9 vLLM wheel. If a
previous runtime imported a CUDA 13 vLLM build, choose **Runtime → Restart
session** before rerunning the setup cell.

The preflight gate is:

- `/health` succeeds after a fresh server start;
- all 420 workload requests succeed with exactly 300 output tokens;
- startup log records the resolved `max_num_batched_tokens`, `max_num_seqs`,
  chunked-prefill state, and Mamba cache mode;
- full GPQA Diamond is retained for v6 and for each quantized candidate; and
- the result directory contains the benchmark JSON, GPQA JSON, startup log,
  resolved configuration, Compose SHA-256, and image digest.

## Controlled candidates

Render candidates from the v6 root rather than maintaining a second active
submission directory. `--custom-image` must be a pushed immutable image
reference, not a tag.

```powershell
# Candidate 1 — ShortConv FP8 only.
conda run -n viettel python scripts/select_submission.py `
  --candidate shortconv-fp8 `
  --custom-image 'DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8@sha256:<64-hex>' `
  --output artifacts/shortconv-fp8.yml

# Candidate 2 — only after the baked-draft image passes preflight.
conda run -n viettel python scripts/select_submission.py `
  --candidate speculative-draft `
  --custom-image 'DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8-draft350@sha256:<64-hex>' `
  --output artifacts/speculative-draft.yml

# Candidate 3 or fallback for candidate 2 — decode-oriented scheduler A/B.
conda run -n viettel python scripts/select_submission.py `
  --candidate batch1536 --output artifacts/batch1536.yml

# Candidate 4, only if another attempt remains.
conda run -n viettel python scripts/select_submission.py `
  --candidate batch1024 --output artifacts/batch1024.yml
```

Validate the rendered file before building/uploading it:

```powershell
docker compose -f artifacts/shortconv-fp8.yml config --quiet
```

To deliberately make a reviewed candidate the root portal artifact, render it
directly to the root with the explicit guard:

```powershell
conda run -n viettel python scripts/select_submission.py `
  --candidate shortconv-fp8 `
  --custom-image 'DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8@sha256:<64-hex>' `
  --output docker-compose.yml --promote
```

The experiment order is ShortConv-FP8, then speculative draft only if it
passes preflight, then the winning parent plus batch 1536, then batch 1024 if
an attempt remains. Do not combine those changes. In particular, leave
`max-num-seqs`, block size, async scheduling, attention backend,
`CUDA_DEVICE_MAX_CONNECTIONS`, and `--calculate-kv-scales` alone.

The speculative candidate requires a second image built with the draft baked
at `/opt/draft/LFM2.5-350M`; it uses exactly four draft tokens, draft TP 1,
and the existing 8192 target context. Its greedy output must be compared with
the non-speculative image before it reaches the portal.

Capture the parent before restarting the server, then compare after the draft
server reaches `/health`:

```powershell
conda run -n viettel python benchmark/compare_greedy.py `
  --base-url http://localhost:8000 `
  --output artifacts/shortconv-fp8/greedy-parent.json

# Restart with the speculative Compose/image, then:
conda run -n viettel python benchmark/compare_greedy.py `
  --base-url http://localhost:8000 `
  --expected artifacts/shortconv-fp8/greedy-parent.json `
  --output artifacts/speculative-draft/greedy-compare.json
```

## Building the custom images

The `Dockerfile` pins the official vLLM `v0.22.1` base by digest and aborts if
the expected source anchors differ. A standard build contains just the
ShortConv FP8 patch:

### Recommended: GitHub Actions / GHCR

No Docker Hub username or local Docker daemon is needed. The tracked workflow
[`publish-shortconv-fp8.yml`](.github/workflows/publish-shortconv-fp8.yml)
runs when the Docker patch changes (or from **Actions → Run workflow**). It
builds a GHCR image, verifies that the digest can be pulled anonymously,
validates Compose, and commits the resulting `shortconv-fp8`
`docker-compose.yml` to `main`.

Wait for all workflow jobs to succeed, then submit the root
`docker-compose.yml` from that resulting commit. If GitHub shows the new GHCR
package as private, make that package public once in its Package settings: the
contest portal has no GitHub credentials with which to pull it.

### Fallback: local Docker / Docker Hub

```powershell
docker build -t DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8 .
```

Or use the helper, which never edits `docker-compose.yml` and prints the
immutable registry reference after a successful push:

```powershell
.\scripts\build_and_push.ps1 -DockerHubUsername DOCKERHUB_USER `
  -Variant shortconv-fp8
```

The speculative variant is opt-in and downloads the immutable draft revision
at build time only. The runtime has `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

```powershell
docker build --build-arg BAKE_DRAFT_MODEL=1 `
  -t DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8-draft350 .
```

```powershell
.\scripts\build_and_push.ps1 -DockerHubUsername DOCKERHUB_USER `
  -Variant shortconv-fp8-draft350m
```

After push, obtain the registry digest and use the full `image@sha256:...`
reference in `select_submission.py`. Never submit a mutable tag.

## Measure and record evidence

Run one workload against a healthy server:

```powershell
conda run -n viettel python benchmark/benchmark_ers.py `
  --trace 019e649f-4e27-74db-82da-920f57b13786/grading-workload-spec.json `
  --tokenizer-path /model --request-rate inf --seed 42 --runs 1 `
  --output artifacts/shortconv-fp8/benchmark.json
```

Run full GPQA only on the GPU runner/Colab:

```powershell
conda run -n viettel python benchmark/test_accuracy.py `
  --base-url http://localhost:8000 --mode gpqa --task gpqa_diamond `
  --output artifacts/shortconv-fp8/gpqa
```

Record the portal result and hashes in the tracked manifest (all artifact paths
are hashed; generated artifact content remains ignored):

```powershell
conda run -n viettel python scripts/record_submission.py `
  --candidate shortconv-fp8 --submission-id '<portal-id>' `
  --compose artifacts/shortconv-fp8.yml `
  --metrics artifacts/shortconv-fp8/benchmark.json `
  --gpqa artifacts/shortconv-fp8/gpqa/results.json `
  --startup-log artifacts/shortconv-fp8/vllm.log `
  --resolved-vllm-config artifacts/shortconv-fp8/runtime.json `
  --healthcheck-passed --preflight-successful-requests 420 `
  --ers <portal-ers> --accuracy <gpqa-accuracy> --f-delta 1.0 --portal-valid
```

`benchmark/submission_results.json` is the tracked decision manifest. It only
recommends a non-v6 candidate when it has a full preflight, passes accuracy,
and strictly improves v6's portal ERS. If ERS values are within 0.01, use
higher accuracy and then lower p95 TTFT as tie-breakers.

For `speculative-draft`, also pass
`--greedy-comparison artifacts/speculative-draft/greedy-compare.json`; the
manifest rejects it unless the recorded comparison says every greedy response
matched the non-speculative parent.

## Constraints

- Keep the required entrypoint, served model name, `/model`, and port 8000.
- Do not pre-bake a prefix cache, hard-code outputs, add an external service,
  or make the serving process download model assets.
- A candidate with an OOM, timeout, zero-token response, or fewer than 420
  successful requests is not eligible for portal selection.
- Preserve v6 when the new candidate does not clearly beat it on the portal.

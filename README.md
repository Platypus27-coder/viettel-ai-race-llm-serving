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

The only current route with enough upside for 75 is a **draft-only
speculative-decoding** image: the pinned `LFM2.5-350M` draft is baked at
`/opt/draft/LFM2.5-350M`, while the target remains exactly v6. Its success is
not assumed: it must prove 420/420, greedy equivalence, GPQA, high acceptance,
and an H200 TPOT near 2.5-2.7 ms. The ShortConv-FP8 patch remains a separate,
small controlled A/B; it is not combined with the draft candidate.

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
[the validation notebook](https://colab.research.google.com/github/Platypus27-coder/viettel-ai-race-llm-serving/blob/candidate/speculative-draft/notebooks/colab_benchmark.ipynb).
It clones this repository and installs the official CUDA 12.9 vLLM wheel. If a
previous runtime imported a CUDA 13 vLLM build, choose **Runtime → Restart
session** before rerunning the setup cell.

Before the setup cell for speculative validation, pin the release identity:

```python
os.environ['VIETTEL_COLAB_PROFILE'] = 'speculative-draft-v6-fp8-smoke'
os.environ['VIETTEL_REPO_REF'] = 'candidate/speculative-draft'
os.environ['VIETTEL_EXPECTED_REPO_SHA'] = '<published-candidate-commit>'
os.environ['VIETTEL_IMAGE_REFERENCE'] = 'yahiisenpai/viettel-ai-vllm@sha256:<64-hex>'
```

The Colab result is explicitly labelled source-equivalent: T4 does not execute
the published image or predict H200 latency.

For a complete greedy gate, first run `v6-fp8-smoke` with
`VIETTEL_RUN_FULL_GPQA=1` and retain its post-workload parent artifact. Restart
the session, then run the speculative profile with the same target revision;
the notebook refuses to continue unless its post-workload comparison matches
that parent. An FP16 speculative smoke is triage only and cannot satisfy the
digest-bound preflight record.

The preflight gate is:

- `/health` succeeds after a fresh server start;
- all 420 workload requests succeed with exactly 300 output tokens;
- startup log records the resolved `max_num_batched_tokens`, `max_num_seqs`,
  chunked-prefill state, and Mamba cache mode;
- for `speculative-draft`, the benchmark records `spec_decode` counter deltas
  and a mean acceptance length; treat less than about 3.5 as a no-go for a
  75-point attempt unless H200 evidence proves otherwise;
- full GPQA Diamond is retained for v6 and for each quantized candidate; and
- the result directory contains the benchmark JSON, GPQA JSON, startup log,
  greedy comparison, resolved configuration, Compose SHA-256, image digest,
  and bound `run_manifest.json`.

## Controlled candidates

Render candidates from the v6 root rather than maintaining a second active
submission directory. `--custom-image` must be a pushed immutable image
reference, not a tag.

```powershell
# Primary high-upside candidate — v6 plus only the baked LFM2.5-350M draft.
conda run -n viettel python scripts/select_submission.py `
  --candidate speculative-draft `
  --custom-image 'yahiisenpai/viettel-ai-vllm:speculative-draft@sha256:<64-hex>' `
  --output artifacts/speculative-draft.yml

# Diagnostic low-upside A/B — v6 plus only ShortConv FP8 wiring.
conda run -n viettel python scripts/select_submission.py `
  --candidate shortconv-fp8 `
  --custom-image 'yahiisenpai/viettel-ai-vllm:shortconv-fp8@sha256:<64-hex>' `
  --output artifacts/shortconv-fp8.yml

# Scheduler child of the exact speculative parent. It inherits its digest and
# draft configuration, and changes only the decode-oriented batch budget.
conda run -n viettel python scripts/select_submission.py `
  --candidate speculative-draft-batch1536 `
  --source artifacts/speculative-draft.yml `
  --output artifacts/speculative-draft-batch1536.yml

# The next controlled child, only after retaining the winning parent evidence.
conda run -n viettel python scripts/select_submission.py `
  --candidate speculative-draft-batch1024 `
  --source artifacts/speculative-draft.yml `
  --output artifacts/speculative-draft-batch1024.yml
```

Validate the rendered file before building/uploading it:

```powershell
docker compose -f artifacts/speculative-draft.yml config --quiet
```

To deliberately make a reviewed candidate the root portal artifact, render it
directly to the root with the explicit guard:

```powershell
conda run -n viettel python scripts/select_submission.py `
  --candidate speculative-draft `
  --custom-image 'yahiisenpai/viettel-ai-vllm:speculative-draft@sha256:<64-hex>' `
  --output docker-compose.yml --promote
```

The experiment order is speculative draft after preflight, then ShortConv-FP8
only as an isolated diagnostic if a portal A/B is still useful, then the
winning parent plus batch 1536, then batch 1024 if an attempt remains. Do not
combine those changes. In particular, leave
`max-num-seqs`, block size, async scheduling, attention backend,
`CUDA_DEVICE_MAX_CONNECTIONS`, and `--calculate-kv-scales` alone.

The speculative candidate requires a second image built with the draft baked
at `/opt/draft/LFM2.5-350M`; it uses exactly four draft tokens, draft TP 1,
and the existing 8192 target context. Its greedy output must be compared with
the non-speculative image before it reaches the portal. Never enable
`parallel_drafting`: the ordinary 350M checkpoint was not trained as a
parallel draft model.

Capture the parent with a separate v6 server, then start a fresh draft server
and run its clean 420-request workload **before** comparing greedy outputs.
The comparison sends requests and warms APC, so it must never precede the
candidate workload in the same server process. If comparison must happen
first, restart the draft server before the workload:

```powershell
conda run -n viettel python benchmark/compare_greedy.py `
  --base-url http://localhost:8000 `
  --output artifacts/v6/greedy-parent.json

# Restart with the speculative Compose/image, then:
conda run -n viettel python benchmark/compare_greedy.py `
  --base-url http://localhost:8000 `
  --expected artifacts/v6/greedy-parent.json `
  --output artifacts/speculative-draft/greedy-compare.json
```

## Building the custom images

The `Dockerfile` pins the official vLLM `v0.22.1` base by digest. It builds
either one isolated change: `ENABLE_SHORTCONV_FP8=1` for the narrow FP8 patch,
or `BAKE_DRAFT_MODEL=1` for the offline draft-only candidate.

### Required: public Docker Hub image

The contest rules require a custom image in a **public Docker Hub** repository;
a GHCR image is therefore not a valid portal artifact. A Docker Hub namespace
is unavoidable because it becomes part of the submitted image reference.

No local Docker daemon is needed. Create the public Docker Hub repository
`yahiisenpai/viettel-ai-vllm`, then add a read/write Docker Hub access token as
the repository Actions secret `DOCKERHUB_TOKEN` (never paste it into a notebook
or chat). Add the image-owning namespace once as repository variable
`DOCKERHUB_NAMESPACE`; if that namespace is an organization, also add the
member login as `DOCKERHUB_USERNAME`. Run
[`publish-shortconv-fp8.yml`](.github/workflows/publish-shortconv-fp8.yml)
from **Actions → Run workflow** and select `speculative-draft`.

The workflow builds either controlled variant, confirms a fresh anonymous
Docker Hub pull, validates the baked draft LICENSE/NOTICE manifest without
starting a GPU server, validates a digest-pinned review Compose, and uploads
that Compose as an Action artifact. It deliberately does **not** replace the
root Compose:
image publication alone is not evidence of a valid 420-request/GPQA candidate.
Until all gates pass, the root Compose stays on the valid v6 incumbent.

### Alternative: local Docker / Docker Hub

```powershell
docker build --build-arg ENABLE_SHORTCONV_FP8=1 `
  -t yahiisenpai/viettel-ai-vllm:shortconv-fp8 .
```

Or use the helper, which never edits `docker-compose.yml` and prints the
immutable registry reference after a successful push:

```powershell
.\scripts\build_and_push.ps1 -DockerHubUsername yahiisenpai `
  -Variant shortconv-fp8
```

The speculative variant is opt-in and downloads the immutable draft revision
at build time only. The runtime has `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

```powershell
docker build --build-arg ENABLE_SHORTCONV_FP8=0 --build-arg BAKE_DRAFT_MODEL=1 `
  -t yahiisenpai/viettel-ai-vllm:speculative-draft .
```

```powershell
.\scripts\build_and_push.ps1 -DockerHubUsername yahiisenpai `
  -Variant speculative-draft
```

After push, obtain the registry digest and use the full `image@sha256:...`
reference in `select_submission.py`. Never submit a mutable tag.

## Measure and record evidence

Run one workload against a healthy server:

```powershell
conda run -n viettel python benchmark/benchmark_ers.py `
  --trace 019e649f-4e27-74db-82da-920f57b13786/grading-workload-spec.json `
  --tokenizer-path /model --request-rate inf --seed 42 --runs 1 `
  --output artifacts/speculative-draft/benchmark.json
```

Run full GPQA only on the GPU runner/Colab:

```powershell
conda run -n viettel python benchmark/test_accuracy.py `
  --base-url http://localhost:8000 --mode gpqa --task gpqa_diamond `
  --output artifacts/speculative-draft/gpqa
```

Record the portal result and hashes in the tracked manifest (all artifact paths
are hashed; generated artifact content remains ignored):

```powershell
conda run -n viettel python scripts/record_submission.py `
  --candidate speculative-draft --submission-id '<portal-id>' `
  --compose artifacts/speculative-draft/docker-compose.speculative-draft.yml `
  --metrics artifacts/speculative-draft/ers-420.json `
  --gpqa artifacts/speculative-draft/gpqa_diamond.results.json `
  --greedy-comparison artifacts/speculative-draft/greedy-speculative-comparison.json `
  --startup-log artifacts/speculative-draft/vllm.log `
  --resolved-vllm-config artifacts/speculative-draft/startup_resolved_config.json `
  --run-manifest artifacts/speculative-draft/run_manifest.json `
  --healthcheck-passed --preflight-successful-requests 420 `
  --ers <portal-ers> --accuracy <gpqa-accuracy> --f-delta 1.0 --portal-valid
```

`benchmark/submission_results.json` is the tracked decision manifest. It only
recommends a non-v6 candidate when its hashed benchmark proves 420/420 with
zero failures, its GPQA artifact passes accuracy, and it strictly improves
v6's portal ERS. If ERS values are within 0.01, use higher accuracy and then
lower p95 TTFT as tie-breakers.

For `speculative-draft`, also pass
`--greedy-comparison artifacts/speculative-draft/greedy-compare.json`; the
manifest rejects it unless the recorded comparison says every greedy response
matched the non-speculative parent, and unless `/metrics` proves a run-scoped,
reset-free speculative counter delta with observed drafts and mean acceptance
length at least 3.5.

## Constraints

- Keep the required entrypoint, served model name, `/model`, and port 8000.
- Do not pre-bake a prefix cache, hard-code outputs, add an external service,
  or make the serving process download model assets.
- A candidate with an OOM, timeout, zero-token response, or fewer than 420
  successful requests is not eligible for portal selection.
- Preserve v6 when the new candidate does not clearly beat it on the portal.

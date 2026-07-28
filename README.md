# Viettel AI Race 2026 — LLM Inference Optimization

Serving and measurement toolkit for
`LiquidAI/LFM2.5-1.2B-Instruct` on one MiG H200 (18 GB VRAM, three CPU
cores, eight GB RAM) with vLLM 0.22.1.

The final score is:

```text
Score = 100 × ERS × f(accuracy drop)
```

The repository treats Colab T4 as a functional and accuracy gate. Only portal
submissions are used as H200 performance measurements.

[Open the validation notebook in Google Colab](https://colab.research.google.com/github/Platypus27-coder/viettel-ai-race-llm-serving/blob/main/notebooks/colab_benchmark.ipynb)

## Project layout

```text
ViettelAI/
├── docker-compose.yml          # the single active portal artifact
├── configs/                    # baseline and all candidate configurations
├── benchmark/                  # workload, accuracy, reports, result manifest
├── scripts/                    # run, generate, and record automation
├── tests/                      # unit and mock-server tests
├── notebooks/                  # Colab GPU validation workflows
│   └── colab_benchmark.ipynb
└── README.md / solution.md     # operating guide and selected strategy
```

This avoids duplicate active configurations: files under `configs/` are
references or candidates; only the root `docker-compose.yml` is submitted.

## What is implemented

- Faithful 70-conversation × six-turn benchmark (420 requests).
- Exact tokenizer-based input budgets and real growing assistant history.
- Deterministic Poisson arrivals with seed 42 and configurable request-rate
  sweeps.
- Robust fragmented SSE parsing, exact output-token validation, TTFT/TPOT/ERS,
  per-turn percentiles, vLLM metrics, and JSON reports.
- Quick accuracy sanity check plus a full `lm-eval` GPQA runner.
- Three remaining submission candidates with guarded FP8 weight selection and
  reproducible result manifests.
- Colab notebook pinned to `vllm==0.22.1`.

## Local environment

All local Python commands use the `viettel` Conda environment:

```powershell
conda run -n viettel python -m pip install -r requirements.txt
conda run -n viettel python -m unittest discover -s tests -v
```

The Windows machine runs clients and tests only. The vLLM server runs in
Docker/Linux on a GPU or in Colab.

## Colab workflow

1. Open the GitHub-hosted notebook using the Colab link above.
2. Select a T4 runtime.
3. Run the setup cell; it clones or fast-forwards the repository under
   `/content/viettel-ai-race-llm-serving` and installs the `cu129` wheel of
   `vllm==0.22.1`.
4. Start a clean server and run the 420-request workload.
5. Run the quick accuracy check and full GPQA for every quantized candidate.
6. Download the benchmark JSON, GPQA results, and `vllm.log`.

The default PyPI wheel for vLLM 0.22.1 requires CUDA 13 and fails on the Colab
T4 runtime with `libcudart.so.13` missing. The notebook therefore installs from
`https://wheels.vllm.ai/0.22.1/cu129` using `uv --torch-backend=cu129` and
verifies `vllm._C` in a subprocess before downloading the model. If a broken
CUDA 13 installation has already been imported, restart the Colab session and
rerun from the setup cell.

This compatibility choice applies only to Colab. Portal submissions continue
to use the official `vllm/vllm-openai:v0.22.1` image on H200.

Do not compare T4 latency between BF16/FP16 and FP8 to select the H200
submission. T4 does not execute the same native W8A8 path as Hopper.

## Faithful ERS benchmark

Run one cold-cache rate:

```powershell
conda run -n viettel python benchmark/benchmark_ers.py `
  --trace 019e649f-4e27-74db-82da-920f57b13786/grading-workload-spec.json `
  --tokenizer-path /model `
  --request-rate inf `
  --seed 42 `
  --runs 1 `
  --output benchmark/results/ers-inf.json
```

Run the rate sweep:

```powershell
conda run -n viettel python benchmark/benchmark_ers.py `
  --trace 019e649f-4e27-74db-82da-920f57b13786/grading-workload-spec.json `
  --tokenizer-path /model `
  --sweep-request-rates 0.5,1,2,4,8,inf `
  --output benchmark/results/rate-sweep.json
```

Between sweep runs, the client requires vLLM's prefix-cache reset endpoint.
If the endpoint is unavailable, restart the server and run each rate
separately; the benchmark stops instead of reporting warm-cache-biased data.

## Accuracy

Quick sanity test:

```powershell
conda run -n viettel python benchmark/test_accuracy.py --mode quick
```

Full GPQA gate on Colab:

```bash
python benchmark/test_accuracy.py \
  --mode gpqa \
  --task gpqa_diamond \
  --concurrency 4 \
  --output benchmark/gpqa_results
```

The selection floor is 0.32 and the preferred safety margin is 0.35. Confirm
the exact GPQA task alias used by the competition's `lm-eval` version.

## Remaining portal submissions

`docker-compose.yml` is the active portal artifact. Reference and candidate
configurations live together under `configs/`:

- `configs/docker-compose.baseline.yml`: BTC reference.
- `configs/docker-compose.observed-48.2-bf16-batch8192-seqs80.yml`: archived 48.2-point
  batch8192/seqs80 configuration.
- `configs/docker-compose.slot-03-bf16-batch4096-seqs64.yml`: run 3 recovery
  candidate and current root artifact.
- `configs/docker-compose.slot-04-bf16-batch2048-seqs64.yml`: run 4 TBT
  candidate.

Generate the fixed recovery candidates:

```powershell
conda run -n viettel python scripts/select_submission.py `
  --slot 3 --output configs/generated-03.yml

conda run -n viettel python scripts/select_submission.py `
  --slot 4 --output configs/generated-04.yml
```

Run 5 uses FP8 weights with the winning batch only after GPQA is at least 0.32
and the workload completes 420/420. Otherwise it uses BF16 with the winning
batch and 48 sequences. FP8 KV and batch 16384 are excluded. See
`configs/README.md` for exact commands and manifest recording. Validate every
generated file before upload:

```powershell
docker compose -f configs/generated-03.yml config --quiet
```

The 60.02-point historical run remains the incumbent. If normalized ERS differs
by less than 0.01, choose higher accuracy; if accuracy ties, choose lower p95
TTFT.

## Important constraints

- Required entrypoint:
  `python3 -m vllm.entrypoints.openai.api_server`
- Required endpoint: `0.0.0.0:8000`
- Required served model name: `LFM2.5-1.2B-Instruct`
- No pre-baked cache, hardcoded output, dual path, external service, or
  benchmark-specific response behavior.
- Never submit a configuration with an OOM, timeout, zero-token response, or
  fewer than 420 successful requests.

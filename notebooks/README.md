# Colab notebooks

`colab_benchmark.ipynb` is the repository's Colab validation workflow. It
clones `Platypus27-coder/viettel-ai-race-llm-serving` into `/content`, so no
project folder needs to be uploaded manually.

[Open the notebook in Colab](https://colab.research.google.com/github/Platypus27-coder/viettel-ai-race-llm-serving/blob/main/notebooks/colab_benchmark.ipynb)

Use a Tesla T4 runtime. Colab validates installation, server health, workload
stability, and accuracy only; its TTFT, TPOT, and ERS are not H200 performance
measurements and must not select a portal submission.

## Clean CUDA setup

The first code cell:

- clones/fetches the configured repository ref and records its resolved SHA;
- removes the old CUDA-13 vLLM/Torch packages and only the obsolete
  `libcudart.so.13` symlink left by the previous notebook;
- installs `vllm==0.22.1` from the official CUDA 12.9 wheel index using `uv`;
- uses a fresh subprocess to require `vllm._C`, CUDA 12.x, and a Tesla T4
  before downloading model weights;
- writes the Python package list, GPU details, environment JSON, and install
  diagnostics to a timestamped artifact directory.

If the current session has already imported `torch` or `vllm`, choose
**Runtime → Restart session** and run the notebook from the top. Do not create
or retain a CUDA-13 symlink as a workaround.

## Profiles

The server cell always keeps the v6 common settings
`--max-model-len=8192`, `--gpu-memory-utilization=0.97`, and
`--enable-prefix-caching`.

- `t4-fp16` is the default. It adds `--dtype=float16` and is the profile for
  the 420-request workload and GPQA tooling.
- `v6-fp8-smoke` adds the exact portal v6 flags `--quantization=fp8` and
  `--kv-cache-dtype=fp8_e4m3`. Set
  `os.environ['VIETTEL_COLAB_PROFILE'] = 'v6-fp8-smoke'` before the server
  cell to try it. A T4 has no native Hopper FP8 W8A8 path, so this is only an
  optional start/function/accuracy smoke test; a rejection is recorded in its
  server log and is not a portal failure.
- `shortconv-fp8-smoke` first applies the repository's fail-closed ShortConv
  patch to the installed vLLM source, then uses the same FP8 flags. Run the
  unpatched `t4-fp16` baseline first: the source patch persists for the rest
  of the Colab session, so restart from setup before returning to unpatched
  v6.
- `speculative-draft-smoke` is the optional FP16 functional smoke profile. It
  downloads the pinned
  `LiquidAI/LFM2.5-350M@1575d1b8b67d862834836087765bff2ef4020672` draft to
  `/content/LFM2.5-350M`, then fails closed unless its full vocabulary and
  special-token IDs match the target tokenizer. It starts the target with
  `--speculative-config` for `method=draft_model`, four draft tokens, draft
  TP=1, and draft context 8192. Set
  `os.environ['VIETTEL_COLAB_PROFILE'] = 'speculative-draft-smoke'` **before
  running the model-download cell**; changing the profile after that cell is
  rejected. This is a start/workload/accuracy smoke test only: T4 latency
  must not be compared with the H200 portal result.
- `speculative-draft-v6-fp8-smoke` has the same pinned draft and validation,
  but uses the exact portal target flags `--quantization=fp8` and
  `--kv-cache-dtype=fp8_e4m3` alongside the draft configuration. Set
  `os.environ['VIETTEL_COLAB_PROFILE'] = 'speculative-draft-v6-fp8-smoke'`
  **before the model-download cell**. This
  is the appropriate Colab startup/workload/accuracy preflight for the portal
  draft candidate; it is still not an H200 latency experiment.

Each server-cell execution terminates the prior process, starts a clean
server without prefix prewarming, validates `/health` and `/v1/models`, and
captures the resolved scheduler/chunked-prefill/Mamba settings from the vLLM
startup log. The notebook intentionally does **not** run a greedy-comparison
request before the 420-request workload: such a request would prewarm the
server and contaminate prefix-cache and speculative-counter evidence.

## Artifacts

The notebook calls `benchmark/benchmark_ers.py` and
`benchmark/test_accuracy.py` from the cloned repository. It stores server
configuration, startup log, health result, raw `/metrics`, the 420-request
JSON result, quick-accuracy log and JSON, optional full GPQA output, and a
manifest in `/content/viettel-artifacts/`, then downloads a zip. The
speculative profile additionally records the pinned draft-model manifest and
tokenizer compatibility result. It saves raw `/metrics` both immediately
before and after the workload, and rejects the speculative run unless the
benchmark's own run-scoped `benchmark_delta` metrics are measured without a
counter reset, observe drafts, and report mean acceptance length at least
3.5. A greedy parity check, if needed, must be run separately after this
workload/restart boundary—not before it.

Full GPQA defaults on for either speculative-draft profile and must complete
before that candidate can be considered. For ordinary baseline smoke profiles,
set `RUN_FULL_GPQA = True` in the final cell after the workload succeeds. The
downloaded zip is the record to attach
to a portal submission manifest; it does not replace H200 portal measurements.

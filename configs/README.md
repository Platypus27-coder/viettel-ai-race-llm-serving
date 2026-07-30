# Configuration references

`../docker-compose.yml` is the only active portal artifact. It remains v6.0
(61.41) until a reviewed candidate is deliberately promoted to the root.

This directory is not a second submission tree:

- `docker-compose.baseline.yml` is the BTC BF16 reference only.
- Temporary controlled candidates belong in ignored `../artifacts/`, generated
  from the root by `scripts/select_submission.py`.

## Render a controlled candidate

All candidates start from the root v6 Compose file. The renderer refuses a
source that already contains scheduler experiments and refuses a custom-image
tag without an immutable SHA-256 digest.

```powershell
# v6 plus the ShortConv FP8 custom image; no command flags change.
conda run -n viettel python ..\scripts\select_submission.py `
  --candidate shortconv-fp8 `
  --custom-image 'DOCKERHUB_USER/viettel-ai-vllm:shortconv-fp8@sha256:<64-hex>' `
  --output ..\artifacts\shortconv-fp8.yml

# v6 plus only the baked local draft model, 4 draft tokens, TP=1.
conda run -n viettel python ..\scripts\select_submission.py `
  --candidate speculative-draft `
  --custom-image 'DOCKERHUB_USER/viettel-ai-vllm:speculative-draft@sha256:<64-hex>' `
  --output ..\artifacts\speculative-draft.yml

# Scheduler-only experiments, run after choosing the better parent.
conda run -n viettel python ..\scripts\select_submission.py `
  --candidate batch1536 --output ..\artifacts\batch1536.yml
conda run -n viettel python ..\scripts\select_submission.py `
  --candidate batch1024 --output ..\artifacts\batch1024.yml
```

Before the portal, validate the exact generated file:

```powershell
docker compose -f ..\artifacts\shortconv-fp8.yml config --quiet
```

Use `--output ../docker-compose.yml --promote` only after review, immediately
before uploading that exact root artifact. Record its SHA-256, image digest,
startup log, GPQA artifact, 420-request preflight, portal ID, and score in
`../benchmark/submission_results.json`.

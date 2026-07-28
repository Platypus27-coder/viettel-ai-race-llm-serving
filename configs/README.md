# Configuration matrix

Only upload one file as `docker-compose.yml` for each portal run. The observed
incumbent is 60.02 points. The batch8192/seqs80 run scored 48.2 and is not a
candidate for final selection.

| Portal run | File | Purpose |
|---|---|---|
| 1 | Historical; exact compose unavailable | Incumbent, 60.02 points |
| 2 | `docker-compose.observed-48.2-bf16-batch8192-seqs80.yml` | Rejected run, 48.2 points |
| 3 | `docker-compose.slot-03-bf16-batch4096-seqs64.yml` | BF16 recovery |
| 4 | `docker-compose.slot-04-bf16-batch2048-seqs64.yml` | BF16 TBT priority |
| 5A | Generated FP8 candidate | Best slot 3/4 scheduler plus FP8 weights |
| 5B | Generated BF16 fallback | Best batch plus 48 sequences |

Generate each standalone candidate with the Conda environment:

```powershell
# Slots 3 and 4 are fixed BF16 recovery candidates.
conda run -n viettel python scripts/select_submission.py --slot 3 --output configs/generated-03.yml
conda run -n viettel python scripts/select_submission.py --slot 4 --output configs/generated-04.yml

# Slot 5A: only after full GPQA >= 0.32 and a 420/420 workload.
# Set batch-tokens to the winner from slots 3 and 4.
conda run -n viettel python scripts/select_submission.py --slot 5 --variant fp8 `
  --batch-tokens 4096 --accuracy 0.32 --successful-requests 420 `
  --output configs/generated-05.yml

# Slot 5B: use when FP8 fails either gate.
conda run -n viettel python scripts/select_submission.py --slot 5 --variant seqs48 `
  --batch-tokens 4096 --output configs/generated-05.yml
```

An ERS difference below 0.01 is treated as a tie and resolved in favor of
higher accuracy, then lower p95 TTFT. The recorder accepts either normalized
ERS (`0.6002`) or portal points (`60.02`).

Never judge H200 performance using the T4 latency numbers. Colab is the
functional and accuracy gate; portal results are the H200 performance signal.

Record every portal result with its exact compose digest:

```powershell
conda run -n viettel python scripts/record_submission.py `
  --slot 3 --submission-id PORTAL_ID --compose configs/generated-03.yml `
  --ers 61.5 --accuracy 0.36 --p95-ttft-ms 70 --successful-requests 420
```

The command writes the ignored local file `benchmark/submission_results.json` and
applies the ERS/accuracy/p95 selection policy automatically.

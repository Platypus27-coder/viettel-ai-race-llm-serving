"""
Viettel AI Race 2026 — Accuracy Validation Script
===================================================
Validates that the submission maintains accuracy >= 0.30 on GPQA Diamond
before submitting to the competition.

Usage:
    # Quick accuracy test (sends GPQA-style questions to the endpoint)
    python test_accuracy.py

    # For full GPQA evaluation, use lm_eval:
    # python -m lm_eval --model local-chat-completions \
    #     --model_args model=LFM2.5-1.2B-Instruct,base_url=http://localhost:8000/v1 \
    #     --tasks gpqa_diamond \
    #     --batch_size auto
"""

import asyncio
import json
import sys
import time
import argparse
import os
import subprocess
from pathlib import Path

import aiohttp

MODEL_NAME = "LFM2.5-1.2B-Instruct"
ACCURACY_BASELINE = 0.40
ACCURACY_SAFE = 0.30     # Δ ≤ 0.10 → f(Δ) = 1.0
ACCURACY_ZERO = 0.24     # Δ ≥ 0.16 → f(Δ) = 0.0

# GPQA Diamond-style questions for quick validation
# These are NOT the actual GPQA questions — just for sanity checking
VALIDATION_QUESTIONS = [
    {
        "question": "In quantum mechanics, what is the physical significance of the commutator [x, p] = iℏ?",
        "choices": [
            "A) It implies position and momentum can be measured simultaneously with arbitrary precision",
            "B) It is a mathematical artifact with no physical meaning",
            "C) It represents the fundamental uncertainty relation between conjugate variables",
            "D) It shows that position and momentum are the same observable"
        ],
        "answer": "C",
    },
    {
        "question": "Which of the following best describes the role of the Jacobian matrix in multivariable calculus?",
        "choices": [
            "A) It represents the best linear approximation of a differentiable function near a given point",
            "B) It is only used for computing determinants of square matrices",
            "C) It measures the total area under a multivariable function",
            "D) It is the inverse of the Hessian matrix"
        ],
        "answer": "A",
    },
    {
        "question": "What is the primary advantage of using a Transformer architecture over traditional RNNs for sequence modeling?",
        "choices": [
            "A) Transformers use less memory than RNNs",
            "B) Transformers can process all tokens in parallel through self-attention",
            "C) Transformers are always more accurate than RNNs",
            "D) Transformers do not require training data"
        ],
        "answer": "B",
    },
    {
        "question": "In organic chemistry, what type of reaction mechanism does an SN2 reaction follow?",
        "choices": [
            "A) A two-step mechanism with a carbocation intermediate",
            "B) A concerted, one-step mechanism with backside attack",
            "C) A radical chain mechanism",
            "D) An elimination reaction mechanism"
        ],
        "answer": "B",
    },
    {
        "question": "Which thermodynamic potential is minimized at equilibrium for a system at constant temperature and pressure?",
        "choices": [
            "A) Internal energy (U)",
            "B) Helmholtz free energy (F)",
            "C) Gibbs free energy (G)",
            "D) Enthalpy (H)"
        ],
        "answer": "C",
    },
    {
        "question": "In computational complexity theory, which class contains decision problems that can be verified in polynomial time?",
        "choices": [
            "A) P",
            "B) NP",
            "C) PSPACE",
            "D) EXPTIME"
        ],
        "answer": "B",
    },
    {
        "question": "What is the primary function of the endoplasmic reticulum in a eukaryotic cell?",
        "choices": [
            "A) DNA replication",
            "B) Protein synthesis, folding, and lipid metabolism",
            "C) ATP production through oxidative phosphorylation",
            "D) Cell division and chromosome segregation"
        ],
        "answer": "B",
    },
    {
        "question": "In general relativity, what does the Einstein field equation Gμν = 8πTμν fundamentally describe?",
        "choices": [
            "A) The relationship between spacetime curvature and energy-momentum distribution",
            "B) The conservation of angular momentum in rotating bodies",
            "C) The quantum behavior of gravitational fields",
            "D) The electromagnetic force between charged particles"
        ],
        "answer": "A",
    },
    {
        "question": "Which of the following is a key property of a Hilbert space that distinguishes it from a general Banach space?",
        "choices": [
            "A) Completeness",
            "B) The existence of an inner product that induces the norm",
            "C) Finite dimensionality",
            "D) Compactness"
        ],
        "answer": "B",
    },
    {
        "question": "In machine learning, what is the bias-variance tradeoff?",
        "choices": [
            "A) The tradeoff between training speed and model accuracy",
            "B) The tradeoff between underfitting (high bias) and overfitting (high variance)",
            "C) The tradeoff between model size and inference speed",
            "D) The tradeoff between supervised and unsupervised learning"
        ],
        "answer": "B",
    },
]


async def ask_question(
    session: aiohttp.ClientSession,
    base_url: str,
    question: dict,
    timeout: float = 30.0,
) -> tuple[str, bool]:
    """Send a GPQA-style question and check if the answer is correct."""
    url = f"{base_url}/v1/chat/completions"

    prompt = (
        f"Answer the following multiple-choice question. "
        f"Reply with ONLY the letter (A, B, C, or D) of the correct answer.\n\n"
        f"Question: {question['question']}\n"
        f"\n".join(question['choices']) + "\n\n"
        f"Answer:"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer multiple choice questions with just the letter."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "stream": False,
    }

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return "ERROR", False
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Extract the letter answer
            predicted = ""
            for char in content.upper():
                if char in "ABCD":
                    predicted = char
                    break

            correct = predicted == question["answer"]
            return predicted, correct
    except Exception as e:
        return f"ERROR: {e}", False


async def run_accuracy_test(base_url: str):
    """Run quick accuracy validation."""
    print(f"\n{'='*60}")
    print(f"  Viettel AI Race 2026 — Accuracy Validation")
    print(f"{'='*60}")
    print(f"  Endpoint: {base_url}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Questions: {len(VALIDATION_QUESTIONS)}")
    print(f"{'='*60}\n")

    # Health check
    print("[1/2] Checking server health...")
    async with aiohttp.ClientSession() as session:
        for attempt in range(15):
            try:
                async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        print("  ✅ Server is healthy!\n")
                        break
            except Exception:
                pass
            if attempt == 14:
                print("  ❌ Server not responding")
                return
            await asyncio.sleep(2)

    # Run questions
    print("[2/2] Running accuracy test...\n")
    correct_count = 0
    total = len(VALIDATION_QUESTIONS)

    async with aiohttp.ClientSession() as session:
        for i, q in enumerate(VALIDATION_QUESTIONS):
            predicted, is_correct = await ask_question(session, base_url, q)
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct_count += 1
            print(f"  Q{i+1}: {status}  Predicted={predicted}  "
                  f"Correct={q['answer']}  ({correct_count}/{i+1})")

    # Results
    accuracy = correct_count / total if total > 0 else 0
    delta = ACCURACY_BASELINE - accuracy

    print(f"\n{'='*60}")
    print(f"  📊 ACCURACY RESULTS")
    print(f"{'='*60}")
    print(f"  Correct: {correct_count}/{total}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Δ (vs baseline 0.40): {delta:.2f}")

    if delta <= 0.10:
        f_delta = 1.0
    elif delta < 0.16:
        f_delta = 1.0 - (delta - 0.10) / 0.06
    else:
        f_delta = 0.0

    print(f"  f(Δ): {f_delta:.3f}")

    if accuracy >= ACCURACY_SAFE:
        print(f"\n  ✅ SAFE — Accuracy >= {ACCURACY_SAFE:.0%}, no penalty expected")
    elif accuracy >= ACCURACY_ZERO:
        print(f"\n  ⚠️  WARNING — Accuracy between {ACCURACY_ZERO:.0%} and {ACCURACY_SAFE:.0%}")
        print(f"  Score will be reduced by f(Δ) = {f_delta:.3f}")
    else:
        print(f"\n  ❌ DANGER — Accuracy <= {ACCURACY_ZERO:.0%}, score will be ZERO!")

    print(f"\n  ⚠️  NOTE: This is a quick sanity check, NOT the official GPQA Diamond.")
    print(f"  For official accuracy, run:")
    print(f"    python -m lm_eval --model local-chat-completions \\")
    print(f"      --model_args model={MODEL_NAME},base_url={base_url}/v1 \\")
    print(f"      --tasks gpqa_diamond --batch_size auto")
    print()


def run_gpqa_full(
    base_url: str,
    output: str,
    task: str,
    limit: int | None,
    concurrency: int,
) -> int:
    """Run the official lm-eval task against vLLM's completions endpoint."""
    base_url = base_url.rstrip("/")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_args = ",".join(
        [
            f"model={MODEL_NAME}",
            f"base_url={base_url}/v1/completions",
            "tokenized_requests=False",
            "tokenizer=LiquidAI/LFM2.5-1.2B-Instruct",
            f"num_concurrent={concurrency}",
            "timeout=300",
            "max_retries=3",
        ]
    )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "local-completions",
        "--model_args",
        model_args,
        "--tasks",
        task,
        "--batch_size",
        "1",
        "--apply_chat_template",
        "--output_path",
        str(output_path),
        "--log_samples",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "not-required")
    print("Running full GPQA with lm-evaluation-harness:")
    print(" ".join(command))
    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode == 0:
        print(f"GPQA results saved under: {output_path}")
    return completed.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Viettel AI Race 2026 — Accuracy Validation"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Base URL of the vLLM server"
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "gpqa"),
        default="quick",
        help="Quick sanity questions or the full lm-eval GPQA gate",
    )
    parser.add_argument(
        "--task",
        default="gpqa_diamond",
        help="lm-eval task name used by the competition environment",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("gpqa_results")),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    if args.mode == "gpqa":
        return run_gpqa_full(
            args.base_url,
            args.output,
            args.task,
            args.limit,
            args.concurrency,
        )
    asyncio.run(run_accuracy_test(args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

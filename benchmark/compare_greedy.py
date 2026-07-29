"""Capture and compare deterministic greedy responses across two server runs.

Run this once against the non-speculative parent, then restart the server with
the speculative draft and pass the first artifact through ``--expected``. The
tool deliberately uses the OpenAI chat endpoint and the served contest model;
it does not inspect model weights or replace the workload benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import aiohttp


MODEL_NAME = "LFM2.5-1.2B-Instruct"
DEFAULT_PROMPTS: tuple[tuple[dict[str, str], ...], ...] = (
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {"role": "user", "content": "State the SI unit of force."},
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {"role": "user", "content": "What is 17 multiplied by 19?"},
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "Give one sentence explaining automatic prefix caching.",
        },
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "Name the capital city of Vietnam.",
        },
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "Complete the sequence: 2, 4, 8, 16, ...",
        },
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "Explain in one sentence why deterministic decoding helps A/B tests.",
        },
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "Write the word 'ready' followed by a period.",
        },
    ),
    (
        {"role": "system", "content": "Answer precisely and concisely."},
        {
            "role": "user",
            "content": "What data structure uses first-in, first-out order?",
        },
    ),
)


def request_payload(messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }


def _response_signature(data: dict[str, Any]) -> dict[str, Any]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Response has no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ValueError("Response choice has no text content")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def fetch_response(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    try:
        async with session.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=request_payload(messages, max_tokens),
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            body = await response.text()
            if response.status != 200:
                return {"ok": False, "error": f"HTTP {response.status}: {body[:500]}"}
            return {"ok": True, "signature": _response_signature(json.loads(body))}
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": str(exc)}


async def capture(
    base_url: str,
    prompts: list[list[dict[str, str]]],
    max_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    timeout_config = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_config) as session:
        return [
            await fetch_response(session, base_url, prompt, max_tokens, timeout)
            for prompt in prompts
        ]


def prompt_fingerprint(prompts: list[list[dict[str, str]]], max_tokens: int) -> str:
    serialized = json.dumps(
        {"model": MODEL_NAME, "prompts": prompts, "max_tokens": max_tokens},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compare_responses(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]]
) -> list[str]:
    """Return readable mismatches, including errors or changed greedy text."""
    errors: list[str] = []
    if len(expected) != len(actual):
        return [f"response count differs: expected {len(expected)}, got {len(actual)}"]
    for index, (expected_item, actual_item) in enumerate(zip(expected, actual), start=1):
        if expected_item != actual_item:
            errors.append(
                f"prompt {index}: expected {json.dumps(expected_item, ensure_ascii=False)}, "
                f"got {json.dumps(actual_item, ensure_ascii=False)}"
            )
    return errors


def load_prompts(path: Path | None) -> list[list[dict[str, str]]]:
    if path is None:
        return [[dict(message) for message in prompt] for prompt in DEFAULT_PROMPTS]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("--prompts must be a non-empty JSON list of message lists")
    prompts: list[list[dict[str, str]]] = []
    for index, prompt in enumerate(raw, start=1):
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Prompt {index} must be a non-empty message list")
        messages: list[dict[str, str]] = []
        for message in prompt:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                raise ValueError(f"Prompt {index} contains an invalid chat message")
            messages.append({"role": message["role"], "content": message["content"]})
        prompts.append(messages)
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--expected", type=Path, help="Capture JSON from the non-speculative parent")
    parser.add_argument("--prompts", type=Path, help="Optional JSON list of chat message lists")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_tokens < 1:
        raise SystemExit("error: --max-tokens must be positive")
    prompts = load_prompts(args.prompts)
    fingerprint = prompt_fingerprint(prompts, args.max_tokens)
    responses = asyncio.run(capture(args.base_url, prompts, args.max_tokens, args.timeout))
    artifact: dict[str, Any] = {
        "model": MODEL_NAME,
        "base_url": args.base_url,
        "max_tokens": args.max_tokens,
        "prompt_fingerprint": fingerprint,
        "prompts": prompts,
        "responses": responses,
    }

    mismatches: list[str] = []
    if args.expected:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        if expected.get("prompt_fingerprint") != fingerprint:
            raise SystemExit("error: --expected used different prompts or --max-tokens")
        expected_responses = expected.get("responses")
        if not isinstance(expected_responses, list):
            raise SystemExit("error: --expected does not contain responses")
        mismatches = compare_responses(expected_responses, responses)
        artifact["expected_path"] = str(args.expected)
        artifact["matches_expected"] = not mismatches
        artifact["mismatches"] = mismatches

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    if mismatches:
        print(f"Greedy comparison failed: {len(mismatches)} mismatch(es). See {args.output}")
        for mismatch in mismatches[:3]:
            print(mismatch)
        return 2
    if args.expected:
        print(f"Greedy comparison passed: {len(responses)} responses match. Saved {args.output}")
    else:
        print(f"Captured {len(responses)} greedy responses in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

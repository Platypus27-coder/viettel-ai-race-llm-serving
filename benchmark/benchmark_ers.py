"""
Faithful Viettel AI Race 2026 serving benchmark.

The benchmark models the published 70-conversation, six-turn workload, keeps
the real assistant output in each conversation, uses the model tokenizer for
token budgets, and calculates ERS with the official scoring formula.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import aiohttp
import numpy as np


F_TTFT = 0.010
C_TTFT = 0.400
F_TPOT = 0.001
C_TPOT = 0.010
GAMMA = 2
W_TTFT = 0.5
MODEL_NAME = "LFM2.5-1.2B-Instruct"
DEFAULT_RATES = (0.5, 1.0, 2.0, 4.0, 8.0, math.inf)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


def latency_score(value: float, floor: float, ceiling: float) -> float:
    """Return the official squared, clamped latency score."""
    normalized = max(0.0, min(1.0, (ceiling - value) / (ceiling - floor)))
    return normalized**GAMMA


def score_ttft(ttft: float) -> float:
    return latency_score(ttft, F_TTFT, C_TTFT)


def score_tpot(tpot: float) -> float:
    return latency_score(tpot, F_TPOT, C_TPOT)


@dataclass(frozen=True)
class ArrivalConfig:
    kind: str = "poisson"
    seed: int = 42
    request_rate: float = math.inf


@dataclass(frozen=True)
class TraceConfig:
    num_conversations: int = 70
    user_turns_per_conversation: int = 6
    shared_system_prefix_tokens: int = 1000
    per_conversation_prefix_tokens: int = 1000
    new_user_tokens_per_turn: int = 150
    output_tokens_per_turn_pinned: int = 300
    arrival: ArrivalConfig = ArrivalConfig()

    @property
    def total_requests(self) -> int:
        return self.num_conversations * self.user_turns_per_conversation


@dataclass
class StreamResult:
    ttft: float
    tpot: float
    output_tokens: int
    completion_text: str
    success: bool
    error: Optional[str] = None
    http_status: Optional[int] = None


@dataclass
class RequestResult:
    conversation_id: int
    turn: int
    ttft: float
    tpot: float
    total_time: float
    output_tokens: int
    expected_output_tokens: int
    input_tokens: int
    success: bool
    error: Optional[str] = None

    @property
    def s_ttft(self) -> float:
        return score_ttft(self.ttft) if self.success else 0.0

    @property
    def s_tpot(self) -> float:
        return score_tpot(self.tpot) if self.success else 0.0

    @property
    def s_request(self) -> float:
        if not self.success or self.output_tokens == 0:
            return 0.0
        return W_TTFT * self.s_ttft + (1.0 - W_TTFT) * self.s_tpot


class SSEDecoder:
    """Incremental UTF-8/SSE decoder that tolerates arbitrary network chunks."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += self._utf8.decode(chunk)
        self._buffer = self._buffer.replace("\r\n", "\n").replace("\r", "\n")
        events: list[str] = []
        while "\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split("\n\n", 1)
            data_lines = [
                line[5:].lstrip()
                for line in raw_event.splitlines()
                if line.startswith("data:")
            ]
            if data_lines:
                events.append("\n".join(data_lines))
        return events

    def finalize(self) -> list[str]:
        self._buffer += self._utf8.decode(b"", final=True)
        if self._buffer and not self._buffer.endswith("\n\n"):
            self._buffer += "\n\n"
        return self.feed(b"")


def _parse_arrival(raw: Any, seed_override: Optional[int] = None) -> ArrivalConfig:
    kind = "poisson"
    seed = 42
    rate = math.inf

    if isinstance(raw, dict):
        kind = str(raw.get("type", raw.get("kind", kind))).lower()
        seed = int(raw.get("seed", seed))
        raw_rate = raw.get("request_rate", raw.get("rate"))
        if raw_rate is not None:
            rate = parse_rate(raw_rate)
    elif isinstance(raw, str):
        kind = "poisson" if "poisson" in raw.lower() else raw.lower()
        match = re.search(r"seed\s*[:=]?\s*(\d+)", raw, re.IGNORECASE)
        if match:
            seed = int(match.group(1))

    if seed_override is not None:
        seed = seed_override
    return ArrivalConfig(kind=kind, seed=seed, request_rate=rate)


def load_trace(
    path: str | os.PathLike[str],
    request_rate: Optional[float] = None,
    seed: Optional[int] = None,
) -> TraceConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    arrival = _parse_arrival(data.get("arrival", {}), seed)
    if request_rate is not None:
        arrival = ArrivalConfig(arrival.kind, arrival.seed, request_rate)

    turns = int(data.get("user_turns_per_conversation", 6))
    conversations = int(data.get("num_conversations", 70))
    declared_total = data.get("total_requests", data.get("total_request"))
    calculated_total = conversations * turns
    if declared_total is not None and int(declared_total) != calculated_total:
        raise ValueError(
            f"Trace total_requests={declared_total} does not match "
            f"{conversations}*{turns}={calculated_total}"
        )

    return TraceConfig(
        num_conversations=conversations,
        user_turns_per_conversation=turns,
        shared_system_prefix_tokens=int(
            data.get("shared_system_prefix_tokens", 1000)
        ),
        per_conversation_prefix_tokens=int(
            data.get("per_conversation_prefix_tokens", 1000)
        ),
        new_user_tokens_per_turn=int(data.get("new_user_tokens_per_turn", 150)),
        output_tokens_per_turn_pinned=int(
            data.get("output_tokens_per_turn_pinned", 300)
        ),
        arrival=arrival,
    )


def parse_rate(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"inf", "infinity", "unlimited"}:
        return math.inf
    rate = float(text)
    if rate <= 0:
        raise ValueError("request rate must be positive or 'inf'")
    return rate


def arrival_offsets(count: int, arrival: ArrivalConfig) -> list[float]:
    """Generate deterministic conversation start offsets."""
    if count <= 0:
        return []
    if math.isinf(arrival.request_rate):
        return [0.0] * count
    if arrival.kind != "poisson":
        raise ValueError(f"Unsupported arrival kind: {arrival.kind}")

    rng = np.random.default_rng(arrival.seed)
    intervals = rng.exponential(1.0 / arrival.request_rate, max(0, count - 1))
    return [0.0, *np.cumsum(intervals).astype(float).tolist()]


def token_count(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def fit_text_to_tokens(
    tokenizer: TokenizerLike,
    seed_text: str,
    target_tokens: int,
) -> str:
    """Create deterministic text whose tokenizer length equals target_tokens."""
    if target_tokens <= 0:
        return ""
    seed_text = seed_text.strip() or "context"
    text = seed_text

    # Grow quickly to the requested budget.
    current = token_count(tokenizer, text)
    while current < target_tokens:
        multiplier = max(1, math.ceil((target_tokens - current) / max(1, current)))
        text += (" " + seed_text) * multiplier
        current = token_count(tokenizer, text)

    # Truncate on token boundaries, then correct boundary re-tokenization.
    for _ in range(64):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == target_tokens:
            return text
        if len(ids) > target_tokens:
            text = tokenizer.decode(
                ids[:target_tokens],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        else:
            text += " context"

    raise RuntimeError(
        f"Could not create exact {target_tokens}-token text "
        f"(last count={token_count(tokenizer, text)})"
    )


def build_prompts(
    trace: TraceConfig,
    tokenizer: TokenizerLike,
) -> tuple[str, list[str], list[list[str]]]:
    shared = fit_text_to_tokens(
        tokenizer,
        (
            "You are a careful technical assistant. Use evidence, explicit reasoning, "
            "and concise conclusions when answering questions about science, "
            "engineering, mathematics, economics, and computing."
        ),
        trace.shared_system_prefix_tokens,
    )
    raw_conversation_prefixes = [
        fit_text_to_tokens(
            tokenizer,
            (
                f"Private context for conversation {conversation_id}. "
                "The following background is unique to this conversation and must "
                "be considered in all later turns."
            ),
            trace.per_conversation_prefix_tokens,
        )
        for conversation_id in range(trace.num_conversations)
    ]
    user_turns = [
        [
            fit_text_to_tokens(
                tokenizer,
                (
                    f"Conversation {conversation_id}, turn {turn + 1}. Analyze the "
                    "question carefully, connect it to the prior conversation, and "
                    "provide a rigorous explanation with a concrete conclusion."
                ),
                trace.new_user_tokens_per_turn,
            )
            for turn in range(trace.user_turns_per_conversation)
        ]
        for conversation_id in range(trace.num_conversations)
    ]
    # Joining two separately exact text fragments can add/remove BPE tokens at
    # the boundary. Prebuild turn one at the exact combined content budget.
    first_turn_contents = [
        fit_text_to_tokens(
            tokenizer,
            f"{raw_conversation_prefixes[conversation_id]}\n\n"
            f"{user_turns[conversation_id][0]}",
            (
                trace.per_conversation_prefix_tokens
                + trace.new_user_tokens_per_turn
            ),
        )
        for conversation_id in range(trace.num_conversations)
    ]
    return shared, first_turn_contents, user_turns


def load_tokenizer(path: str) -> TokenizerLike:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for exact token budgets. "
            "Install requirements.txt in the viettel Conda environment."
        ) from exc
    return AutoTokenizer.from_pretrained(path, use_fast=True)


def estimate_chat_input_tokens(
    tokenizer: TokenizerLike,
    messages: list[dict[str, str]],
) -> int:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        rendered = apply_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(rendered)
    return sum(token_count(tokenizer, message["content"]) for message in messages)


async def send_streaming_request(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    tokenizer: TokenizerLike,
    request_timeout: float = 300.0,
) -> StreamResult:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.1,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "ignore_eos": True,
    }

    start = time.perf_counter()
    first_content_time: Optional[float] = None
    last_content_time: Optional[float] = None
    content_parts: list[str] = []
    usage_tokens: Optional[int] = None
    decoder = SSEDecoder()

    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request_timeout),
        ) as response:
            if response.status != 200:
                body = await response.text()
                return StreamResult(
                    0.0,
                    0.0,
                    0,
                    "",
                    False,
                    f"HTTP {response.status}: {body[:500]}",
                    response.status,
                )

            async for chunk in response.content.iter_chunked(4096):
                for event in decoder.feed(chunk):
                    if event == "[DONE]":
                        continue
                    try:
                        data = json.loads(event)
                    except json.JSONDecodeError:
                        continue

                    usage = data.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        usage_tokens = int(usage["completion_tokens"])

                    for choice in data.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            now = time.perf_counter()
                            if first_content_time is None:
                                first_content_time = now
                            last_content_time = now
                            content_parts.append(content)

            for event in decoder.finalize():
                if event not in {"", "[DONE]"}:
                    try:
                        data = json.loads(event)
                        usage = data.get("usage")
                        if usage and usage.get("completion_tokens") is not None:
                            usage_tokens = int(usage["completion_tokens"])
                    except json.JSONDecodeError:
                        pass

        completion_text = "".join(content_parts)
        if first_content_time is None or not completion_text:
            return StreamResult(0.0, 0.0, 0, completion_text, False, "No content")

        output_tokens = (
            usage_tokens
            if usage_tokens is not None
            else token_count(tokenizer, completion_text)
        )
        ttft = first_content_time - start
        if output_tokens >= 2 and last_content_time is not None:
            tpot = (last_content_time - first_content_time) / (output_tokens - 1)
        else:
            tpot = F_TPOT

        if output_tokens != max_tokens:
            return StreamResult(
                ttft,
                tpot,
                output_tokens,
                completion_text,
                False,
                f"Expected {max_tokens} output tokens, received {output_tokens}",
                200,
            )
        return StreamResult(
            ttft, tpot, output_tokens, completion_text, True, http_status=200
        )
    except asyncio.TimeoutError:
        return StreamResult(0.0, 0.0, 0, "", False, "Timeout")
    except aiohttp.ClientError as exc:
        return StreamResult(0.0, 0.0, 0, "", False, f"Connection error: {exc}")
    except Exception as exc:  # Keep one failed request from aborting the trace.
        return StreamResult(0.0, 0.0, 0, "", False, f"Unexpected error: {exc}")


async def run_conversation(
    session: aiohttp.ClientSession,
    base_url: str,
    conversation_id: int,
    trace: TraceConfig,
    tokenizer: TokenizerLike,
    shared_prompt: str,
    first_turn_content: str,
    user_turns: list[str],
    start_offset: float,
    benchmark_start: float,
    results: list[RequestResult],
) -> None:
    delay = benchmark_start + start_offset - time.perf_counter()
    if delay > 0:
        await asyncio.sleep(delay)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": shared_prompt}
    ]
    for turn, user_text in enumerate(user_turns):
        if turn == 0:
            user_text = first_turn_content
        messages.append({"role": "user", "content": user_text})
        input_tokens = estimate_chat_input_tokens(tokenizer, messages)
        request_start = time.perf_counter()
        stream = await send_streaming_request(
            session,
            base_url,
            messages,
            trace.output_tokens_per_turn_pinned,
            tokenizer,
        )
        total_time = time.perf_counter() - request_start
        results.append(
            RequestResult(
                conversation_id=conversation_id,
                turn=turn,
                ttft=stream.ttft,
                tpot=stream.tpot,
                total_time=total_time,
                output_tokens=stream.output_tokens,
                expected_output_tokens=trace.output_tokens_per_turn_pinned,
                input_tokens=input_tokens,
                success=stream.success,
                error=stream.error,
            )
        )

        if not stream.success:
            print(f"  ERROR conv={conversation_id} turn={turn + 1}: {stream.error}")
            return
        messages.append({"role": "assistant", "content": stream.completion_text})


async def wait_for_health(base_url: str, attempts: int = 60) -> None:
    async with aiohttp.ClientSession() as session:
        for attempt in range(attempts):
            try:
                async with session.get(
                    f"{base_url.rstrip('/')}/health",
                    timeout=aiohttp.ClientTimeout(total=3),
                ) as response:
                    if response.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            if attempt + 1 < attempts:
                await asyncio.sleep(2)
    raise RuntimeError(f"Server did not become healthy at {base_url}")


async def fetch_vllm_metrics(base_url: str) -> dict[str, float]:
    selected: dict[str, float] = {}
    patterns = (
        "time_to_first_token",
        "inter_token_latency",
        "prefix_cache_hit",
        "kv_cache_usage",
        "num_requests",
        "prompt_tokens",
        "generation_tokens",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url.rstrip('/')}/metrics",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    return selected
                body = await response.text()
        for line in body.splitlines():
            if not line or line.startswith("#") or not any(p in line for p in patterns):
                continue
            name, _, raw_value = line.rpartition(" ")
            try:
                value = float(raw_value)
                if math.isfinite(value):
                    selected[name] = value
            except ValueError:
                continue
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass
    return selected


async def reset_prefix_cache(base_url: str) -> None:
    """Reset APC between repeated/swept runs or fail rather than bias results."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{base_url.rstrip('/')}/reset_prefix_cache",
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(
                        f"Prefix-cache reset returned HTTP {response.status}: "
                        f"{body[:300]}"
                    )
                try:
                    reset_result = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    reset_result = {}
                if reset_result.get("success") is False:
                    raise RuntimeError(
                        "vLLM reported that the prefix cache could not be reset"
                    )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise RuntimeError(
                "Cannot reset prefix cache between benchmark runs. Restart the "
                "vLLM server and run one request rate at a time."
            ) from exc


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def summarize_results(
    results: list[RequestResult],
    trace: TraceConfig,
    duration: float,
    server_metrics: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]
    ers = (
        sum(result.s_request for result in results) / trace.total_requests
        if trace.total_requests
        else 0.0
    )
    per_turn: dict[str, Any] = {}
    for turn in range(trace.user_turns_per_conversation):
        turn_results = [result for result in results if result.turn == turn]
        turn_success = [result for result in turn_results if result.success]
        per_turn[str(turn + 1)] = {
            "requests": len(turn_results),
            "successful": len(turn_success),
            "ers": (
                sum(result.s_request for result in turn_results)
                / trace.num_conversations
                if trace.num_conversations
                else 0.0
            ),
            "ttft_ms": _percentiles(
                [result.ttft * 1000 for result in turn_success]
            ),
            "tpot_ms": _percentiles(
                [result.tpot * 1000 for result in turn_success]
            ),
        }

    return {
        "ers": ers,
        "score_if_accuracy_safe": 100 * ers,
        "duration_seconds": duration,
        "expected_requests": trace.total_requests,
        "observed_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": trace.total_requests - len(successful),
        "success_rate": (
            len(successful) / trace.total_requests if trace.total_requests else 0.0
        ),
        "ttft_ms": _percentiles([result.ttft * 1000 for result in successful]),
        "tpot_ms": _percentiles([result.tpot * 1000 for result in successful]),
        "input_tokens": _percentiles(
            [float(result.input_tokens) for result in successful]
        ),
        "output_tokens": _percentiles(
            [float(result.output_tokens) for result in successful]
        ),
        "per_turn": per_turn,
        "server_metrics": server_metrics or {},
        "failures": [
            {
                "conversation_id": result.conversation_id,
                "turn": result.turn + 1,
                "error": result.error,
                "output_tokens": result.output_tokens,
            }
            for result in failed
        ],
        "requests": [
            {
                "conversation_id": result.conversation_id,
                "turn": result.turn + 1,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "ttft_ms": result.ttft * 1000,
                "tpot_ms": result.tpot * 1000,
                "total_time_seconds": result.total_time,
                "s_ttft": result.s_ttft,
                "s_tpot": result.s_tpot,
                "s_request": result.s_request,
                "success": result.success,
                "error": result.error,
            }
            for result in sorted(
                results, key=lambda item: (item.conversation_id, item.turn)
            )
        ],
    }


def trace_as_json_dict(trace: TraceConfig) -> dict[str, Any]:
    data = asdict(trace)
    if math.isinf(trace.arrival.request_rate):
        data["arrival"]["request_rate"] = "inf"
    return data


def config_fingerprint(trace: TraceConfig, base_url: str) -> str:
    payload = json.dumps(
        {
            "trace": trace_as_json_dict(trace),
            "base_url": base_url,
            "model": MODEL_NAME,
        },
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


async def run_benchmark(
    base_url: str,
    trace: TraceConfig,
    tokenizer: TokenizerLike,
) -> dict[str, Any]:
    await wait_for_health(base_url)
    shared, prefixes, user_turns = build_prompts(trace, tokenizer)
    offsets = arrival_offsets(trace.num_conversations, trace.arrival)
    results: list[RequestResult] = []

    print(
        f"\nRunning {trace.total_requests} requests: "
        f"{trace.num_conversations} conversations x "
        f"{trace.user_turns_per_conversation} turns, "
        f"rate={trace.arrival.request_rate}"
    )
    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=max(100, trace.num_conversations + 10))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                run_conversation(
                    session,
                    base_url,
                    conversation_id,
                    trace,
                    tokenizer,
                    shared,
                    prefixes[conversation_id],
                    user_turns[conversation_id],
                    offsets[conversation_id],
                    start,
                    results,
                )
            )
            for conversation_id in range(trace.num_conversations)
        ]
        await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    metrics = await fetch_vllm_metrics(base_url)
    summary = summarize_results(results, trace, duration, metrics)
    summary["trace"] = trace_as_json_dict(trace)
    summary["config_fingerprint"] = config_fingerprint(trace, base_url)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    ttft = summary["ttft_ms"]
    tpot = summary["tpot_ms"]
    print("\n" + "=" * 72)
    print(
        f"ERS={summary['ers']:.6f}  "
        f"score_if_accuracy_safe={summary['score_if_accuracy_safe']:.2f}"
    )
    print(
        f"success={summary['successful_requests']}/"
        f"{summary['expected_requests']} "
        f"duration={summary['duration_seconds']:.2f}s"
    )
    if ttft:
        print(
            f"TTFT ms: mean={ttft['mean']:.2f} p50={ttft['p50']:.2f} "
            f"p95={ttft['p95']:.2f} p99={ttft['p99']:.2f}"
        )
        print(
            f"TPOT ms: mean={tpot['mean']:.3f} p50={tpot['p50']:.3f} "
            f"p95={tpot['p95']:.3f} p99={tpot['p99']:.3f}"
        )
    print("=" * 72)


def _rates_from_args(args: argparse.Namespace) -> list[float]:
    if args.sweep_request_rates:
        return [parse_rate(item) for item in args.sweep_request_rates.split(",")]
    return [parse_rate(args.request_rate)]


async def async_main(args: argparse.Namespace) -> int:
    tokenizer = load_tokenizer(args.tokenizer_path)
    rates = _rates_from_args(args)
    all_runs: list[dict[str, Any]] = []
    completed_runs = 0

    for rate in rates:
        for run_index in range(args.runs):
            if completed_runs:
                await reset_prefix_cache(args.base_url)
            if args.trace:
                trace = load_trace(args.trace, request_rate=rate, seed=args.seed)
            else:
                trace = TraceConfig(
                    arrival=ArrivalConfig("poisson", args.seed, rate)
                )
            summary = await run_benchmark(args.base_url, trace, tokenizer)
            summary["run_index"] = run_index + 1
            print_summary(summary)
            all_runs.append(summary)
            completed_runs += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_unix": time.time(),
        "base_url": args.base_url,
        "model": MODEL_NAME,
        "runs": all_runs,
    }
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved benchmark report to {output_path}")
    return 0 if all(run["success_rate"] == 1.0 for run in all_runs) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faithful Viettel AI Race 2026 ERS benchmark"
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--trace")
    parser.add_argument(
        "--tokenizer-path",
        default=(
            "/model"
            if os.path.isdir("/model")
            else "LiquidAI/LFM2.5-1.2B-Instruct"
        ),
    )
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument(
        "--sweep-request-rates",
        help="Comma-separated rates, e.g. 0.5,1,2,4,8,inf",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("benchmark_results.json")),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())

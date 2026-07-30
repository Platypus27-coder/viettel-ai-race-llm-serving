from __future__ import annotations

import json
import math
import time
import unittest
from pathlib import Path

from aiohttp import web

from benchmark.benchmark_ers import (
    ArrivalConfig,
    RequestResult,
    SSEDecoder,
    TraceConfig,
    arrival_offsets,
    build_prompts,
    fit_text_to_tokens,
    load_trace,
    parse_vllm_metrics_text,
    run_benchmark,
    run_conversation,
    score_tpot,
    score_ttft,
    summarize_speculative_decoding,
    summarize_results,
)


class WhitespaceTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        count = sum(len(message["content"].split()) for message in messages)
        return list(range(count + (1 if add_generation_prompt else 0)))


class ScoringAndTraceTests(unittest.TestCase):
    def test_official_score_boundaries(self) -> None:
        self.assertEqual(score_ttft(0.010), 1.0)
        self.assertEqual(score_ttft(0.400), 0.0)
        self.assertEqual(score_ttft(1.0), 0.0)
        self.assertEqual(score_tpot(0.001), 1.0)
        self.assertEqual(score_tpot(0.010), 0.0)
        self.assertAlmostEqual(score_ttft(0.205), 0.25)
        self.assertAlmostEqual(score_tpot(0.0055), 0.25)

    def test_loads_published_trace_and_string_arrival(self) -> None:
        trace_path = (
            Path(__file__).parents[1]
            / "019e649f-4e27-74db-82da-920f57b13786"
            / "grading-workload-spec.json"
        )
        trace = load_trace(trace_path, request_rate=2.0)
        self.assertEqual(trace.total_requests, 420)
        self.assertEqual(trace.shared_system_prefix_tokens, 1000)
        self.assertEqual(trace.per_conversation_prefix_tokens, 1000)
        self.assertEqual(trace.output_tokens_per_turn_pinned, 300)
        self.assertEqual(trace.arrival.kind, "poisson")
        self.assertEqual(trace.arrival.seed, 42)
        self.assertEqual(trace.arrival.request_rate, 2.0)

    def test_rejects_inconsistent_total_requests(self) -> None:
        path = Path(__file__).parent / "fixtures" / "invalid_trace.json"
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_trace(path)

    def test_poisson_offsets_are_seeded_and_monotonic(self) -> None:
        arrival = ArrivalConfig("poisson", 42, 2.0)
        first = arrival_offsets(10, arrival)
        second = arrival_offsets(10, arrival)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 0.0)
        self.assertTrue(all(a <= b for a, b in zip(first, first[1:])))
        self.assertEqual(
            arrival_offsets(4, ArrivalConfig("poisson", 42, math.inf)),
            [0.0] * 4,
        )

    def test_exact_token_text(self) -> None:
        tokenizer = WhitespaceTokenizer()
        text = fit_text_to_tokens(tokenizer, "alpha beta gamma", 150)
        self.assertEqual(len(tokenizer.encode(text)), 150)

    def test_first_turn_combined_budget_is_exact(self) -> None:
        tokenizer = WhitespaceTokenizer()
        trace = TraceConfig(
            num_conversations=1,
            user_turns_per_conversation=2,
            shared_system_prefix_tokens=10,
            per_conversation_prefix_tokens=12,
            new_user_tokens_per_turn=5,
            output_tokens_per_turn_pinned=3,
        )
        shared, first_turns, user_turns = build_prompts(trace, tokenizer)
        self.assertEqual(len(tokenizer.encode(shared)), 10)
        self.assertEqual(len(tokenizer.encode(first_turns[0])), 17)
        self.assertEqual(len(tokenizer.encode(user_turns[0][1])), 5)

    def test_sse_decoder_handles_fragmentation_and_utf8(self) -> None:
        decoder = SSEDecoder()
        payload = (
            'data: {"choices":[{"delta":{"content":"Việt"}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        events: list[str] = []
        for byte in payload:
            events.extend(decoder.feed(bytes([byte])))
        events.extend(decoder.finalize())
        self.assertEqual(len(events), 2)
        self.assertIn("Việt", events[0])
        self.assertEqual(events[1], "[DONE]")

    def test_summary_counts_unobserved_requests_as_failures(self) -> None:
        trace = TraceConfig(
            num_conversations=2,
            user_turns_per_conversation=2,
            shared_system_prefix_tokens=1,
            per_conversation_prefix_tokens=1,
            new_user_tokens_per_turn=1,
            output_tokens_per_turn_pinned=1,
        )
        result = RequestResult(0, 0, 0.01, 0.001, 0.01, 1, 1, 3, True)
        summary = summarize_results([result], trace, 1.0)
        self.assertEqual(summary["expected_requests"], 4)
        self.assertEqual(summary["failed_requests"], 3)
        self.assertAlmostEqual(summary["ers"], 0.25)

    def test_spec_decode_metrics_use_workload_counter_delta(self) -> None:
        before = parse_vllm_metrics_text(
            "\n".join(
                (
                    "# TYPE vllm:spec_decode_num_drafts counter",
                    'vllm:spec_decode_num_drafts_total{model_name="lfm draft"} 10 1700000000000',
                    'vllm:spec_decode_num_draft_tokens_total{model_name="lfm draft"} 37',
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="lfm draft"} 25',
                    'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="lfm draft",position="0"} 10',
                    'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="lfm draft",position="1"} 8',
                    # A Prometheus bookkeeping series must not be counted as a
                    # vLLM counter.
                    'vllm:spec_decode_num_drafts_created{model_name="lfm draft"} 1.7e9',
                )
            )
        )
        after = parse_vllm_metrics_text(
            "\n".join(
                (
                    'vllm:spec_decode_num_drafts_total{model_name="lfm draft"} 30',
                    'vllm:spec_decode_num_draft_tokens_total{model_name="lfm draft"} 105',
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="lfm draft"} 70',
                    'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="lfm draft",position="0"} 30',
                    'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="lfm draft",position="1"} 22',
                    'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="lfm draft",position="2"} 11',
                    "vllm:gpu_prefix_cache_hit_rate 0.75",
                )
            )
        )

        summary = summarize_speculative_decoding(
            before, after, duration=2.5, successful_requests=18
        )

        self.assertTrue(summary["available"])
        self.assertEqual(summary["counter_scope"], "benchmark_delta")
        self.assertEqual(summary["acceptance_status"], "measured")
        self.assertEqual(summary["counters"]["num_drafts"], 20.0)
        self.assertEqual(summary["counters"]["num_draft_tokens"], 68.0)
        self.assertEqual(summary["counters"]["num_accepted_tokens"], 45.0)
        self.assertEqual(
            summary["counters"]["num_accepted_tokens_per_position"],
            {"0": 20.0, "1": 14.0, "2": 11.0},
        )
        self.assertAlmostEqual(summary["mean_acceptance_length"], 3.25)
        self.assertAlmostEqual(summary["draft_token_acceptance_rate"], 45 / 68)
        self.assertEqual(
            summary["acceptance_rate_per_draft_position"],
            {"0": 1.0, "1": 0.7, "2": 0.55},
        )
        self.assertAlmostEqual(summary["accepted_tokens_per_second"], 18.0)

    def test_spec_decode_metrics_report_missing_and_counter_reset(self) -> None:
        no_metrics = summarize_speculative_decoding({}, {}, 1.0, 1)
        self.assertFalse(no_metrics["available"])
        self.assertEqual(no_metrics["acceptance_status"], "not_available")

        before = parse_vllm_metrics_text(
            "\n".join(
                (
                    "vllm:spec_decode_num_drafts_total 12",
                    "vllm:spec_decode_num_draft_tokens_total 40",
                    "vllm:spec_decode_num_accepted_tokens_total 20",
                )
            )
        )
        after = parse_vllm_metrics_text(
            "\n".join(
                (
                    "vllm:spec_decode_num_drafts_total 2",
                    "vllm:spec_decode_num_draft_tokens_total 6",
                    "vllm:spec_decode_num_accepted_tokens_total 3",
                )
            )
        )
        reset = summarize_speculative_decoding(before, after, 1.0, 1)
        self.assertTrue(reset["available"])
        self.assertTrue(reset["counter_reset_detected"])
        self.assertEqual(reset["acceptance_status"], "counter_reset")
        self.assertNotIn("mean_acceptance_length", reset)


class ConversationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[dict] = []
        self.output_token_override: int | None = None
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._completion)
        app.router.add_get("/health", self._health)
        app.router.add_get("/metrics", self._metrics)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets
        self.base_url = f"http://127.0.0.1:{sockets[0].getsockname()[1]}"

    async def asyncTearDown(self) -> None:
        await self.runner.cleanup()

    async def _completion(self, request: web.Request) -> web.StreamResponse:
        payload = await request.json()
        self.requests.append(payload)
        expected = int(payload["max_tokens"])
        actual = self.output_token_override or expected
        response = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        for index in range(actual):
            event = {
                "choices": [{"delta": {"content": f"answer{index} "}}],
                "usage": None,
            }
            await response.write(f"data: {json.dumps(event)}\n\n".encode())
        usage = {
            "choices": [],
            "usage": {"completion_tokens": actual},
        }
        await response.write(f"data: {json.dumps(usage)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    async def _metrics(self, request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "# TYPE vllm:gpu_prefix_cache_hit_rate gauge\n"
                "vllm:gpu_prefix_cache_hit_rate 0.75\n"
            )
        )

    async def test_end_to_end_benchmark_summary(self) -> None:
        tokenizer = WhitespaceTokenizer()
        trace = TraceConfig(
            num_conversations=1,
            user_turns_per_conversation=2,
            shared_system_prefix_tokens=2,
            per_conversation_prefix_tokens=2,
            new_user_tokens_per_turn=2,
            output_tokens_per_turn_pinned=3,
            arrival=ArrivalConfig("poisson", 42, math.inf),
        )
        summary = await run_benchmark(self.base_url, trace, tokenizer)
        self.assertEqual(summary["expected_requests"], 2)
        self.assertEqual(summary["successful_requests"], 2)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["trace"]["arrival"]["request_rate"], "inf")
        self.assertTrue(summary["config_fingerprint"])
        self.assertIn(
            "vllm:gpu_prefix_cache_hit_rate", summary["server_metrics"]
        )

    async def test_real_assistant_output_is_kept_in_history(self) -> None:
        from aiohttp import ClientSession

        tokenizer = WhitespaceTokenizer()
        trace = TraceConfig(
            num_conversations=1,
            user_turns_per_conversation=2,
            shared_system_prefix_tokens=2,
            per_conversation_prefix_tokens=2,
            new_user_tokens_per_turn=2,
            output_tokens_per_turn_pinned=3,
        )
        results: list[RequestResult] = []
        async with ClientSession() as session:
            await run_conversation(
                session,
                self.base_url,
                0,
                trace,
                tokenizer,
                "shared prefix",
                "private prefix first question",
                ["first question", "second question"],
                0.0,
                time.perf_counter(),
                results,
            )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.success for result in results))
        second_messages = self.requests[1]["messages"]
        assistants = [
            message for message in second_messages if message["role"] == "assistant"
        ]
        self.assertEqual(len(assistants), 1)
        self.assertIn("answer0", assistants[0]["content"])
        self.assertNotIn("[Response", assistants[0]["content"])

    async def test_short_output_fails_and_stops_conversation(self) -> None:
        from aiohttp import ClientSession

        self.output_token_override = 2
        tokenizer = WhitespaceTokenizer()
        trace = TraceConfig(
            num_conversations=1,
            user_turns_per_conversation=2,
            shared_system_prefix_tokens=1,
            per_conversation_prefix_tokens=1,
            new_user_tokens_per_turn=1,
            output_tokens_per_turn_pinned=3,
        )
        results: list[RequestResult] = []
        async with ClientSession() as session:
            await run_conversation(
                session,
                self.base_url,
                0,
                trace,
                tokenizer,
                "shared",
                "private first",
                ["first", "second"],
                0.0,
                time.perf_counter(),
                results,
            )
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertIn("Expected 3", results[0].error or "")


if __name__ == "__main__":
    unittest.main()

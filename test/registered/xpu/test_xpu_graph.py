"""XPU Full Graph integration test.

Verifies that:
1. Prefill Full Graph and Decode Full Graph both actually replay (no silent fallback to eager)
   across exact-bucket and bucket-padded multi-request workloads with request-slot padding.
2. Prefill logits and multi-step greedy decoded token sequences match eager execution.

Usage:
    python3 -m unittest test_xpu_graph.TestXPUGraph.test_full_graph_runs
"""

import time
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from sglang.benchmark.one_batch import (
    _init_process_global_configs,
    load_model,
    prepare_extend_inputs_for_correctness_test,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.runtime_context import SpawnRanks, publish
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_xpu_ci(est_time=45, suite="extra-test-1-gpu-xpu")


class TestXPUGraph(CustomTestCase):
    """Decode Full Graph plus Prefill Full Graph."""

    def test_full_graph_runs(self):
        # page_size=64 sets Xe20 decode split_threshold = 32 * 64 = 2048.
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
            device="xpu",
            attention_backend="intel_xpu",
            page_size=64,
            mem_fraction_static=0.6,
            cuda_graph_config={
                "decode": {"backend": "full", "bs": [1, 2]},
                "prefill": {
                    "backend": "full",
                    "bs": [64, 128],
                    "full_prefill_max_req": 4,
                    "max_context_size": 128,
                },
            },
        )
        server_args.resolve_once()
        _set_envs_and_config(server_args)
        port_args = PortArgs.init_new(server_args)
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(world_rank=0, gpu_id=0),
        )
        _init_process_global_configs()

        bench_runner, tokenizer = load_model(
            server_args, port_args, gpu_id=0, tp_rank=0
        )
        torch_runner = bench_runner.torch_runner
        self.assertIsNotNone(torch_runner.prefill_cuda_graph_runner)
        self.assertIsNotNone(torch_runner.decode_cuda_graph_runner)

        base_p0 = (
            tokenizer.encode(
                "The capital of France is Paris. The capital of Germany is Berlin. "
                "The capital of Italy is Rome. The capital of Japan is Tokyo. "
                "The capital of Spain is"
            )
            * 6
        )
        base_p1 = (
            tokenizer.encode(
                "One plus one equals two. Two plus two equals four. "
                "Three plus three equals six. Four plus four equals eight. "
                "Five plus five equals"
            )
            * 6
        )
        output_len = 8

        @contextmanager
        def capture_replays(use_graph: bool):
            saved_prefill = torch_runner.prefill_cuda_graph_runner
            saved_decode = torch_runner.decode_cuda_graph_runner
            prefill_replays, decode_replays = [], []
            orig_prefill_replay = saved_prefill.backend.replay
            orig_decode_replay = saved_decode.backend.replay
            try:
                if not use_graph:
                    torch_runner.prefill_cuda_graph_runner = None
                    torch_runner.decode_cuda_graph_runner = None
                else:
                    saved_prefill.backend.replay = lambda k, *a, **kw: (
                        prefill_replays.append(k) or orig_prefill_replay(k, *a, **kw)
                    )
                    saved_decode.backend.replay = lambda k, *a, **kw: (
                        decode_replays.append(k) or orig_decode_replay(k, *a, **kw)
                    )
                yield prefill_replays, decode_replays
            finally:
                saved_prefill.backend.replay = orig_prefill_replay
                saved_decode.backend.replay = orig_decode_replay
                torch_runner.prefill_cuda_graph_runner = saved_prefill
                torch_runner.decode_cuda_graph_runner = saved_decode

        def run_pass(custom_inputs, use_graph: bool):
            bench_runner.clear()
            reqs = prepare_synthetic_inputs_for_latency_test(
                batch_size=len(custom_inputs),
                input_len=len(custom_inputs[0]),
                custom_inputs=custom_inputs,
            )
            with capture_replays(use_graph) as (prefill_replays, decode_replays):
                bench_runner.synchronize()
                tic = time.perf_counter()
                next_token_ids, prefill_logits, batch = bench_runner.extend(reqs)
                bench_runner.synchronize()
                prefill_latency = time.perf_counter() - tic

                prefill_logits_cpu = prefill_logits.detach().float().cpu().clone()
                output_ids = [
                    list(custom_inputs[i]) + [int(next_token_ids[i].item())]
                    for i in range(len(reqs))
                ]
                decode_latencies = []
                for _ in range(output_len - 1):
                    bench_runner.synchronize()
                    tic = time.perf_counter()
                    next_token_ids, _ = bench_runner.decode(next_token_ids, batch)
                    bench_runner.synchronize()
                    decode_latencies.append(time.perf_counter() - tic)
                    for i in range(len(reqs)):
                        output_ids[i].append(int(next_token_ids[i].item()))
                decode_throughput = len(reqs) / float(np.median(decode_latencies))
                return (
                    prefill_logits_cpu,
                    output_ids,
                    prefill_replays,
                    decode_replays,
                    prefill_latency,
                    decode_throughput,
                )

        # Warmup once before steady-state measurement
        run_pass([base_p0[:32], base_p1[:32]], use_graph=True)
        run_pass([base_p0[:32], base_p1[:32]], use_graph=False)

        # Case 1 & 2: Exact bucket (64) and padded bucket (96 -> 128)
        for tok_len, expected_prefill_bucket in ((32, 64), (48, 128)):
            with self.subTest(tok_len=tok_len, bucket=expected_prefill_bucket):
                custom_inputs = [base_p0[:tok_len], base_p1[:tok_len]]
                (
                    graph_logits,
                    graph_ids,
                    prefill_replays,
                    decode_replays,
                    graph_prefill_latency,
                    graph_decode_throughput,
                ) = run_pass(custom_inputs, use_graph=True)
                (
                    eager_logits,
                    eager_ids,
                    eager_prefill_replays,
                    eager_decode_replays,
                    eager_prefill_latency,
                    eager_decode_throughput,
                ) = run_pass(custom_inputs, use_graph=False)

                print(
                    f"[tok_len={tok_len}, bucket={expected_prefill_bucket}] "
                    f"Graph prefill_latency={graph_prefill_latency * 1000:.2f}ms, "
                    f"Eager prefill_latency={eager_prefill_latency * 1000:.2f}ms | "
                    f"Graph decode_throughput={graph_decode_throughput:.2f} tok/s, "
                    f"Eager decode_throughput={eager_decode_throughput:.2f} tok/s",
                    flush=True,
                )

                self.assertGreater(graph_prefill_latency, 0)
                self.assertGreater(graph_decode_throughput, 0)
                self.assertEqual(
                    [k.size for k in prefill_replays], [expected_prefill_bucket]
                )
                self.assertEqual(
                    [(k.size, k.attention_variant) for k in decode_replays],
                    [(2, "short")] * (output_len - 1),
                )
                self.assertEqual(eager_prefill_replays, [])
                self.assertEqual(eager_decode_replays, [])

                cos_sim = torch.nn.functional.cosine_similarity(
                    graph_logits, eager_logits, dim=-1
                )
                self.assertTrue(torch.all(cos_sim > 0.999).item())
                self.assertEqual(graph_ids, eager_ids)

        # Case 3: Prefix-cached prefill at max_context_size=128 (replays) vs above cap (eager fallback).
        # Above-cap suffix tokens (72 + 56 = 128) fit bucket 128, so only max_context_size (136 > 128) triggers fallback.
        def run_prefix_pass(full_inputs, cut_len: int, use_graph: bool):
            bench_runner.clear()
            prefix_inputs = [inp[:cut_len] for inp in full_inputs]
            reqs = prepare_synthetic_inputs_for_latency_test(
                batch_size=len(prefix_inputs),
                input_len=cut_len,
                custom_inputs=prefix_inputs,
            )
            with capture_replays(use_graph) as (prefill_replays, _):
                bench_runner.extend(reqs)
                reqs = prepare_extend_inputs_for_correctness_test(
                    SimpleNamespace(cut_len=cut_len), full_inputs, reqs, torch_runner
                )
                next_token_ids, prefill_logits, batch = bench_runner.extend(reqs)
                prefill_logits_cpu = prefill_logits.detach().float().cpu().clone()
                output_ids = [
                    list(full_inputs[i]) + [int(next_token_ids[i].item())]
                    for i in range(len(reqs))
                ]
                for _ in range(3):
                    next_token_ids, _ = bench_runner.decode(next_token_ids, batch)
                    for i in range(len(reqs)):
                        output_ids[i].append(int(next_token_ids[i].item()))
                bench_runner.synchronize()
                return prefill_logits_cpu, output_ids, prefill_replays

        for case_name, full_inputs, expected_buckets in (
            ("at_cap", [base_p0[:128], base_p1[:128]], [128, 128]),
            ("above_cap", [base_p0[:136], base_p1[:120]], [128]),
        ):
            with self.subTest(case=f"prefix_extend_{case_name}"):
                g_logits, g_ids, prefill_replays = run_prefix_pass(
                    full_inputs, cut_len=64, use_graph=True
                )
                e_logits, e_ids, e_replays = run_prefix_pass(
                    full_inputs, cut_len=64, use_graph=False
                )
                self.assertEqual([k.size for k in prefill_replays], expected_buckets)
                self.assertEqual(e_replays, [])
                cos_sim = torch.nn.functional.cosine_similarity(
                    g_logits, e_logits, dim=-1
                )
                self.assertTrue(torch.all(cos_sim > 0.999).item())
                self.assertEqual(g_ids, e_ids)

        # Case 4: Single-sequence decode crossing split_threshold=2048 (2047..2050 -> short, short, long, long).
        boundary_prompt = [((base_p0 + base_p1) * 8)[:2046]]

        def run_boundary_decode(use_graph: bool):
            bench_runner.clear()
            reqs = prepare_synthetic_inputs_for_latency_test(
                batch_size=1, input_len=2046, custom_inputs=boundary_prompt
            )
            with capture_replays(use_graph) as (_, decode_replays):
                next_token_ids, _, batch = bench_runner.extend(reqs)
                tokens = [int(next_token_ids[0].item())]
                for _ in range(4):
                    next_token_ids, _ = bench_runner.decode(next_token_ids, batch)
                    tokens.append(int(next_token_ids[0].item()))
                bench_runner.synchronize()
                return tokens, decode_replays

        with self.subTest(case="decode_bucket_boundary"):
            g_tokens, decode_replays = run_boundary_decode(use_graph=True)
            e_tokens, e_decode_replays = run_boundary_decode(use_graph=False)
            self.assertEqual(
                [(k.size, k.attention_variant) for k in decode_replays],
                [(1, "short"), (1, "short"), (1, "long"), (1, "long")],
            )
            self.assertEqual(e_decode_replays, [])
            self.assertEqual(g_tokens, e_tokens)


if __name__ == "__main__":
    unittest.main()

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

register_xpu_ci(est_time=45, suite="stage-b-test-1-gpu-xpu")


class TestXPUGraph(CustomTestCase):
    """Decode Full Graph plus Prefill Full Graph."""

    def test_full_graph_runs(self):
        server_args = ServerArgs(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
            device="xpu",
            attention_backend="intel_xpu",
            mem_fraction_static=0.6,
            cuda_graph_config={
                "decode": {"backend": "full", "bs": [1, 2]},
                "prefill": {
                    "backend": "full",
                    "bs": [64, 128],
                    "full_prefill_max_req": 4,
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
            * 2
        )
        base_p1 = (
            tokenizer.encode(
                "One plus one equals two. Two plus two equals four. "
                "Three plus three equals six. Four plus four equals eight. "
                "Five plus five equals"
            )
            * 2
        )
        output_len = 8

        def run_pass(custom_inputs, use_graph: bool):
            bench_runner.clear()
            reqs = prepare_synthetic_inputs_for_latency_test(
                batch_size=len(custom_inputs),
                input_len=len(custom_inputs[0]),
                custom_inputs=custom_inputs,
            )
            saved_prefill = torch_runner.prefill_cuda_graph_runner
            saved_decode = torch_runner.decode_cuda_graph_runner
            prefill_replays = []
            decode_replays = []

            orig_prefill_replay = saved_prefill.backend.replay
            orig_decode_replay = saved_decode.backend.replay

            def wrapped_prefill_replay(shape_key, *args, **kwargs):
                prefill_replays.append(shape_key)
                return orig_prefill_replay(shape_key, *args, **kwargs)

            def wrapped_decode_replay(shape_key, *args, **kwargs):
                decode_replays.append(shape_key)
                return orig_decode_replay(shape_key, *args, **kwargs)

            try:
                if not use_graph:
                    torch_runner.prefill_cuda_graph_runner = None
                    torch_runner.decode_cuda_graph_runner = None
                else:
                    saved_prefill.backend.replay = wrapped_prefill_replay
                    saved_decode.backend.replay = wrapped_decode_replay

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
                med_decode_latency = float(np.median(decode_latencies))
                decode_throughput = len(reqs) / med_decode_latency
                return (
                    prefill_logits_cpu,
                    output_ids,
                    prefill_replays,
                    decode_replays,
                    prefill_latency,
                    decode_throughput,
                )
            finally:
                saved_prefill.backend.replay = orig_prefill_replay
                saved_decode.backend.replay = orig_decode_replay
                torch_runner.prefill_cuda_graph_runner = saved_prefill
                torch_runner.decode_cuda_graph_runner = saved_decode

        # Case 1: 2 requests x 32 tokens = 64 tokens (exact bucket=64, padded req slots 2->4)
        # Case 2: 2 requests x 48 tokens = 96 tokens (padded bucket 96->128, padded req slots 2->4)
        for tok_len, expected_prefill_bucket in ((32, 64), (48, 128)):
            with self.subTest(tok_len=tok_len, bucket=expected_prefill_bucket):
                custom_inputs = [base_p0[:tok_len], base_p1[:tok_len]]
                # Warmup both paths so timing reflects steady-state execution
                run_pass(custom_inputs, use_graph=True)
                run_pass(custom_inputs, use_graph=False)

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

                self.assertGreater(
                    graph_prefill_latency,
                    0,
                    "prefill latency must be > 0 with full prefill XPU graph",
                )
                self.assertGreater(
                    graph_decode_throughput,
                    0,
                    "decode throughput must be > 0 with full XPU graph",
                )
                self.assertEqual(
                    [k.size for k in prefill_replays],
                    [expected_prefill_bucket],
                    f"Expected prefill graph replay on bucket {expected_prefill_bucket}",
                )
                self.assertEqual(
                    [k.size for k in decode_replays],
                    [2] * (output_len - 1),
                    "Expected decode graph replay on batch size 2 for each decode step",
                )
                self.assertEqual(eager_prefill_replays, [])
                self.assertEqual(eager_decode_replays, [])

                cos_sim = torch.nn.functional.cosine_similarity(
                    graph_logits, eager_logits, dim=-1
                )
                self.assertTrue(
                    torch.all(cos_sim > 0.999).item(),
                    f"Prefill logits cosine similarity too low: {cos_sim}",
                )
                self.assertEqual(
                    graph_ids,
                    eager_ids,
                    f"Decoded token IDs mismatch between full graph and eager for tok_len={tok_len}",
                )

        # Case 3: Prefix-cached extend (32 prefix + 32 new = 64 total tokens per request,
        # 2 requests -> 64 extend tokens on bucket=64, cu_seqlens_q != cu_seqlens_k)
        full_inputs = [base_p0[:64], base_p1[:64]]
        cut_len = 32

        def run_prefix_pass(use_graph: bool):
            bench_runner.clear()
            prefix_inputs = [inp[:cut_len] for inp in full_inputs]
            reqs = prepare_synthetic_inputs_for_latency_test(
                batch_size=len(prefix_inputs),
                input_len=cut_len,
                custom_inputs=prefix_inputs,
            )
            saved_prefill = torch_runner.prefill_cuda_graph_runner
            saved_decode = torch_runner.decode_cuda_graph_runner
            prefill_replays = []
            decode_replays = []

            orig_prefill_replay = saved_prefill.backend.replay
            orig_decode_replay = saved_decode.backend.replay

            def wrapped_prefill_replay(shape_key, *args, **kwargs):
                prefill_replays.append(shape_key)
                return orig_prefill_replay(shape_key, *args, **kwargs)

            def wrapped_decode_replay(shape_key, *args, **kwargs):
                decode_replays.append(shape_key)
                return orig_decode_replay(shape_key, *args, **kwargs)

            try:
                if not use_graph:
                    torch_runner.prefill_cuda_graph_runner = None
                    torch_runner.decode_cuda_graph_runner = None
                else:
                    saved_prefill.backend.replay = wrapped_prefill_replay
                    saved_decode.backend.replay = wrapped_decode_replay

                # 1st extend: populate prefix KV cache (2 x 32 = 64 tokens)
                bench_runner.extend(reqs)
                # 2nd extend: extend with cached prefix (prefix_len=32, extend_len=32)
                reqs = prepare_extend_inputs_for_correctness_test(
                    SimpleNamespace(cut_len=cut_len), full_inputs, reqs, torch_runner
                )
                next_token_ids, prefill_logits, batch = bench_runner.extend(reqs)
                prefill_logits_cpu = prefill_logits.detach().float().cpu().clone()
                output_ids = [
                    list(full_inputs[i]) + [int(next_token_ids[i].item())]
                    for i in range(len(reqs))
                ]
                for _ in range(output_len - 1):
                    next_token_ids, _ = bench_runner.decode(next_token_ids, batch)
                    for i in range(len(reqs)):
                        output_ids[i].append(int(next_token_ids[i].item()))
                bench_runner.synchronize()
                return (
                    prefill_logits_cpu,
                    output_ids,
                    prefill_replays,
                    decode_replays,
                )
            finally:
                saved_prefill.backend.replay = orig_prefill_replay
                saved_decode.backend.replay = orig_decode_replay
                torch_runner.prefill_cuda_graph_runner = saved_prefill
                torch_runner.decode_cuda_graph_runner = saved_decode

        with self.subTest(case="prefix_extend", cut_len=cut_len, bucket=64):
            (
                graph_logits,
                graph_ids,
                prefill_replays,
                decode_replays,
            ) = run_prefix_pass(use_graph=True)
            (
                eager_logits,
                eager_ids,
                eager_prefill_replays,
                eager_decode_replays,
            ) = run_prefix_pass(use_graph=False)

            self.assertEqual(
                [k.size for k in prefill_replays],
                [64, 64],
                "Expected both prefix and suffix extends to replay on bucket 64",
            )
            self.assertEqual(
                [k.size for k in decode_replays],
                [2] * (output_len - 1),
            )
            self.assertEqual(eager_prefill_replays, [])
            self.assertEqual(eager_decode_replays, [])

            cos_sim = torch.nn.functional.cosine_similarity(
                graph_logits, eager_logits, dim=-1
            )
            self.assertTrue(
                torch.all(cos_sim > 0.999).item(),
                f"Prefix-extend prefill logits cosine similarity too low: {cos_sim}",
            )
            self.assertEqual(
                graph_ids,
                eager_ids,
                "Decoded token IDs mismatch between full graph and eager on prefix extend",
            )


if __name__ == "__main__":
    unittest.main()

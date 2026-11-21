"""CPU checks for benchmark evidence handling and TP checkpoint slicing."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cutile_nvfp4 as benchmark

sys.path.insert(0, str(benchmark.REPO_ROOT))
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

helpers = benchmark.load_test_helpers()

import torch


class TestCuTileBenchmark(CustomTestCase):
    def test_workspace_bucket_coverage_counts_tail_and_deduplicates(self):
        expected, visited, per_call = benchmark.workspace_bucket_coverage(
            [8192, 8193, 16384],
            8192,
            8192,
            lambda capacity: (1, 2, 8192),
            lambda count, capacity: min(count, capacity),
        )
        self.assertEqual(expected, [1, 2, 8192])
        self.assertEqual(visited, [1, 8192])
        self.assertEqual(per_call[8193], [8192, 1])
        self.assertEqual(per_call[16384], [8192, 8192])
        self.assertNotEqual(expected, visited)

    def test_workspace_inventory_deduplicates_aliases_and_breaks_cycles(self):
        tensor = torch.ones(8)
        # The fake object needs the same module boundary as the external runner.
        runner_type = type("FakeRunner", (), {"__module__": "flashinfer.test"})
        runner = runner_type()
        runner._workspace_cache = {(8192, 32): [tensor, tensor.view(2, 4)]}
        runner.parent = runner
        inventory = benchmark.execution_workspace_inventory({"runner": runner})
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["unique_storage_bytes"], 32)

    def arguments(self, *extra):
        return [
            "--synthetic",
            "--implementation-label",
            "unit-test",
            "--flashinfer-variant",
            "test-only",
            "--output-dir",
            "/unused",
            *extra,
        ]

    def test_peer_inputs_keep_preselected_routes_and_copy_all_fields(self):
        config = SimpleNamespace(hidden_size=32, num_experts=16, top_k=4)
        layer = SimpleNamespace(moe_runner_config=config)
        one = benchmark.make_inputs(
            layer, 65, 17, 0.5, sets=1, input_protocol="peer-compatible", device="cpu"
        )
        four = benchmark.make_inputs(
            layer, 65, 999, 0.5, sets=4, input_protocol="peer-compatible", device="cpu"
        )
        self.assertEqual(len(one), 1)
        self.assertEqual(len(four), 4)
        self.assertTrue(torch.equal(one[0].hidden_states, four[0].hidden_states))
        first, second = four[:2]
        self.assertFalse(torch.equal(first.hidden_states, second.hidden_states))
        for item in four:
            routes = item.topk_output
            self.assertTrue(all(len(set(row.tolist())) == 4 for row in routes.topk_ids))
            self.assertFalse(
                torch.allclose(routes.topk_weights.sum(-1), torch.ones(65))
            )
            torch.testing.assert_close(
                routes.router_logits.gather(1, routes.topk_ids.long()),
                routes.topk_weights,
                rtol=0,
                atol=0,
            )
        benchmark.copy_input(first, second)
        self.assertEqual(
            benchmark.input_full_hashes([first]), benchmark.input_full_hashes([second])
        )
        sliced = benchmark.slice_input(second, 7)
        self.assertTrue(
            torch.equal(
                sliced.topk_output.router_logits, second.topk_output.router_logits[:7]
            )
        )
        before = benchmark.input_full_hashes([second])
        prefix = benchmark.input_prefix_hashes([second])
        second.topk_output.router_logits[-1, 0] += 1
        self.assertNotEqual(before, benchmark.input_full_hashes([second]))
        self.assertEqual(prefix, benchmark.input_prefix_hashes([second]))

    def test_historical_protocol_is_still_normalized(self):
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(hidden_size=32, num_experts=8, top_k=4)
        )
        values = benchmark.make_inputs(layer, 7, 17, 0.5, sets=1, device="cpu")
        torch.testing.assert_close(
            values[0].topk_output.topk_weights.sum(-1), torch.ones(7)
        )

    def test_tp1_comparison_rejects_single_gpu_substitution(self):
        with self.assertRaisesRegex(ValueError, "TP2 or TP4"):
            benchmark.actual_tp1_comparison(SimpleNamespace(), None, [], [], 0, None, 1)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                benchmark.parse_args(self.arguments("--compare-tp1"))

    def test_peer_reference_does_not_replace_corrected_reference(self):
        args = benchmark.parse_args(
            self.arguments(
                "--input-protocol", "peer-compatible", "--reference-protocol", "both"
            )
        )
        self.assertEqual(args.max_relative_l2, 0.05)
        self.assertEqual(args.reference_protocol, "both")
        with self.assertRaises(benchmark.BenchmarkCheckError):
            benchmark.repeat_metrics("marlin", torch.full((4,), 1.002), torch.ones(4))
        with self.assertRaises(benchmark.BenchmarkCheckError):
            benchmark.repeat_metrics("cutile", torch.full((4,), 1.00001), torch.ones(4))

    def test_reference_sampling_covers_chunk_edges_and_tail(self):
        for tokens in (1, 64, 8191, 8192, 8193, 16384, 16385):
            with self.subTest(tokens=tokens):
                indices = helpers._reference_token_indices(tokens, 64)
                self.assertEqual(len(indices), min(tokens, 64))
                self.assertEqual(indices, sorted(set(indices)))
                self.assertIn(0, indices)
                self.assertIn(tokens - 1, indices)
                for edge in range(8192, tokens, 8192):
                    self.assertIn(edge - 1, indices)
                    self.assertIn(edge, indices)
        with self.assertRaisesRegex(ValueError, "budget"):
            helpers._reference_token_indices(16385, 4)

    def test_boundary_reference_detects_corrupted_tail_and_gathers_routes(self):
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatchOutput,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        rows = torch.arange(16385).float()
        dispatch = StandardDispatchOutput(
            hidden_states=rows[:, None].expand(-1, 2),
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_ids=rows.long()[:, None],
                topk_weights=rows[:, None],
                router_logits=rows[:, None].expand(-1, 3),
            ),
        )
        indices = helpers._reference_token_indices(16385, 64)
        selected = helpers._select_dispatch_rows(dispatch, indices)
        for field in ("topk_ids", "topk_weights", "router_logits"):
            torch.testing.assert_close(
                getattr(selected.topk_output, field),
                getattr(dispatch.topk_output, field)[indices],
            )
        no_logits = dispatch._replace(
            topk_output=dispatch.topk_output._replace(router_logits=None)
        )
        self.assertIsNone(
            helpers._select_dispatch_rows(no_logits, indices).topk_output.router_logits
        )
        expected = torch.ones(64, 2)
        actual = torch.ones(16385, 2)
        actual[-1] = 10
        # The old prefix gate would not detect this finite tail corruption.
        self.assertEqual(
            benchmark.error_metrics(actual[:64], expected)["relative_l2"], 0
        )
        sampled = benchmark.select_reference_output(
            actual, expected, {"reference_indices": indices}
        )
        self.assertGreater(
            benchmark.error_metrics(sampled, expected)["relative_l2"], 0.05
        )

    def test_same_partition_oracle_remains_strict(self):
        expected = torch.ones(8)
        small = expected + 0.001
        with self.assertRaises(benchmark.BenchmarkCheckError):
            benchmark.compare_direct_outputs(
                small, expected, same_partition=True, relative_l2_limit=0.05
            )
        comparison = benchmark.compare_direct_outputs(
            small, expected, same_partition=False, relative_l2_limit=0.05
        )
        self.assertFalse(comparison["bit_identity"])
        with self.assertRaises(benchmark.BenchmarkCheckError):
            benchmark.compare_direct_outputs(
                expected + 0.06, expected, same_partition=False, relative_l2_limit=0.05
            )

    def test_workspace_control_rejects_shared_execution_across_streams(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(
                self.arguments("--mode", "workspace", "--workspace-backend", "direct")
            )
        args = benchmark.parse_args(
            self.arguments(
                "--mode", "workspace", "--workspace-backend", "direct", "--streams", "1"
            )
        )
        self.assertEqual(args.workspace_backend, "direct")
        self.assertEqual(
            benchmark.parse_args(self.arguments()).reference_sampling, "prefix"
        )
        parsed = benchmark.parse_args(
            self.arguments(
                "--backends",
                "cutile",
                "direct-chunked",
                "direct",
                "--reference-sampling",
                "chunk-boundaries",
            )
        )
        self.assertEqual(parsed.reference_sampling, "chunk-boundaries")

    def test_percentiles_use_interpolation_and_endpoints(self):
        self.assertEqual(benchmark.percentile([8, 2], 0), 2)
        self.assertEqual(benchmark.percentile([8, 2], 50), 5)
        self.assertAlmostEqual(benchmark.percentile([8, 2], 95), 7.7)
        self.assertEqual(benchmark.percentile([8, 2], 100), 8)
        self.assertEqual(benchmark.percentile([7], 95), 7)

    def test_bit_identity_distinguishes_signed_zero(self):
        positive = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        negative = torch.tensor([-0.0, 1.0], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(positive, negative))
        self.assertFalse(benchmark.bit_equal(positive, negative))
        self.assertTrue(benchmark.bit_equal(positive, positive.clone()))

    def test_rank_aggregation_keeps_moving_stragglers(self):
        result = benchmark.summarize_rank_samples([[1, 10, 1, 10], [10, 1, 10, 1]])
        self.assertEqual(result["samples_ms"], [10, 10, 10, 10])
        self.assertEqual(result["p50_ms"], 10)
        # Max of independent rank medians would incorrectly report 5.5 ms.
        self.assertNotEqual(result["p50_ms"], 5.5)

    def test_invalid_measurements_are_rejected(self):
        for values in ([], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.percentile(values, 50)
        for values in ([], [[]], [[1], [1, 2]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.summarize_rank_samples(values)

    def test_cli_rejects_weak_or_invalid_protocol(self):
        invalid = [
            ("--samples", "99"),
            ("--warmup", "0"),
            ("--tokens", "0"),
            ("--tokens", "16385"),
            ("--input-std", "nan"),
            ("--max-relative-l2", "1"),
            ("--streams", "0"),
            ("--backends", "cutile", "cutile"),
            ("--distributed-timeout", "0"),
            ("--distributed-timeout", "-1"),
            ("--distributed-timeout", "nan"),
            ("--distributed-timeout", "inf"),
        ]
        for extra in invalid:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    benchmark.parse_args(self.arguments(*extra))
                self.assertEqual(caught.exception.code, 2)
        args = benchmark.parse_args(
            self.arguments("--samples", "100", "--cutile-api", "fused")
        )
        self.assertEqual(args.samples, 100)
        self.assertEqual(args.cutile_api, "fused")

    def test_helper_import_ignores_shadowing_test_package(self):
        shadow = ModuleType("test")
        with patch.dict(sys.modules, {"test": shadow}):
            loaded = benchmark.load_test_helpers()
            self.assertEqual(
                Path(loaded.__file__).resolve(),
                benchmark.REPO_ROOT
                / "test/manual/layers/moe/test_flashinfer_cutile_nvfp4.py",
            )
            self.assertIs(sys.modules["test"], shadow)

    def test_distributed_timeout_is_explicit_and_configurable(self):
        default = benchmark.parse_args(self.arguments())
        self.assertEqual(benchmark.distributed_timeout(default), timedelta(seconds=600))
        selected = benchmark.parse_args(self.arguments("--distributed-timeout", "12.5"))
        self.assertEqual(
            benchmark.distributed_timeout(selected), timedelta(seconds=12.5)
        )

    def test_full_output_check_rejects_nonfinite_tail_and_wrong_shape(self):
        output = torch.ones(65, 2)
        benchmark.validate_output(output, (65, 2))
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                broken = output.clone()
                broken[64, 0] = invalid
                # This is the former blind spot: reference prefix is finite,
                # and byte-wise comparison accepts a stable nonfinite pattern.
                self.assertTrue(torch.isfinite(broken[:64]).all().item())
                self.assertTrue(benchmark.bit_equal(broken, broken.clone()))
                with self.assertRaises(benchmark.BenchmarkCheckError) as caught:
                    benchmark.validate_output(broken, output.shape)
                self.assertEqual(caught.exception.metrics["nonfinite_elements"], 1)
        with self.assertRaises(benchmark.BenchmarkCheckError) as caught:
            benchmark.validate_output(output, (64, 2))
        self.assertEqual(caught.exception.metrics["actual_shape"], [65, 2])

    def test_repeat_gates_preserve_bit_identity_and_marlin_tolerance(self):
        expected = torch.ones(1)
        small = expected + 0.0005
        checked = benchmark.repeat_metrics("marlin", small, expected)
        self.assertFalse(checked["bit_identity"])
        self.assertLess(checked["relative_l2"], 0.001)
        with self.assertRaises(benchmark.BenchmarkCheckError):
            benchmark.repeat_metrics("cutile", small, expected)
        with self.assertRaises(benchmark.BenchmarkCheckError) as caught:
            benchmark.repeat_metrics("marlin", expected + 0.002, expected)
        self.assertGreater(caught.exception.metrics["relative_l2"], 0.001)

    def test_distributed_failure_keeps_stage_and_metrics_before_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            evidence = benchmark.RunEvidence(Path(tmp), rank=1)
            evidence.stage(
                "repeat-check",
                backend="marlin",
                tokens=1,
                execution="eager",
                check_index=7,
            )
            error = benchmark.BenchmarkCheckError(
                "repeat failed", {"relative_l2": 0.006}
            )
            with (
                patch.object(
                    benchmark, "exit_distributed_failure", side_effect=SystemExit(1)
                ) as terminate,
                patch.object(benchmark.traceback, "print_exception") as trace,
            ):
                with self.assertRaises(SystemExit):
                    benchmark.report_failure(evidence, error, distributed_worker=True)
            terminate.assert_called_once_with()
            trace.assert_called_once_with(type(error), error, error.__traceback__)
            saved = json.loads(evidence.path.read_text())
            self.assertFalse(saved["complete"])
            self.assertFalse(saved["running"])
            self.assertEqual(saved["phase"], "repeat-check")
            self.assertEqual(saved["backend"], "marlin")
            self.assertEqual(saved["tokens"], 1)
            self.assertEqual(saved["check_index"], 7)
            self.assertEqual(saved["metrics"], error.metrics)

    def test_graph_failure_preserves_completed_eager_samples(self):
        # Only CUDA operations are mocked; real CPU tensors, checking, sample
        # aggregation, per-mode persistence and failure reporting run normally.
        def dispatch(value):
            return SimpleNamespace(
                hidden_states=torch.full((65, 2), float(value)),
                topk_output=SimpleNamespace(
                    topk_ids=torch.zeros(65, 1, dtype=torch.int32),
                    topk_weights=torch.ones(65, 1),
                    router_logits=torch.zeros(65, 1),
                ),
            )

        inputs = [dispatch(value) for value in (1, 2, 3, 4)]
        references = [
            (item.hidden_states[:64].clone(), item.hidden_states[:64].clone(), {})
            for item in inputs
        ]
        call = Mock(side_effect=lambda item: item.hidden_states.clone())
        call.layer = object()
        event = Mock()
        event.elapsed_time.return_value = 1.0
        args = benchmark.parse_args(self.arguments())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            evidence = benchmark.RunEvidence(output, rank=0)
            evidence.stage("backend-checks", tokens=65, backend="cutile")
            rows = []
            with (
                patch.object(benchmark, "make_inputs", return_value=[dispatch(0)]),
                patch.object(torch.cuda, "memory_allocated", return_value=1024),
                patch.object(torch.cuda, "reset_peak_memory_stats"),
                patch.object(torch.cuda, "max_memory_allocated", return_value=2048),
                patch.object(torch.cuda, "Event", return_value=event),
                patch.object(torch.cuda, "CUDAGraph", return_value=Mock()),
                patch.object(
                    torch.cuda, "graph", side_effect=RuntimeError("capture failed")
                ),
                self.assertRaisesRegex(RuntimeError, "capture failed") as caught,
            ):
                benchmark.measure_backend(
                    "cutile",
                    call,
                    inputs,
                    references,
                    args,
                    None,
                    1,
                    Mock(),
                    evidence,
                    lambda row: benchmark.persist_latency_case(output, rows, row),
                )
            benchmark.report_failure(
                evidence, caught.exception, distributed_worker=False
            )
            saved = json.loads((output / "latency.json").read_text())
            self.assertFalse(saved["complete"])
            self.assertEqual(len(saved["cases"]), 1)
            row = saved["cases"][0]
            self.assertEqual(row["execution"], "eager")
            self.assertEqual(len(row["gpu"]["samples_ms"]), 100)
            self.assertEqual(len(row["repeat_checks"]), 20)
            self.assertEqual(row["repeat_max_relative_l2"], 0.0)
            failed = json.loads(evidence.path.read_text())
            self.assertFalse(failed["complete"])
            self.assertEqual(failed["execution"], "graph")
            self.assertEqual(failed["phase"], "capture")

    def test_real_weights_need_revision_provenance(self):
        args = self.arguments()
        args[0:1] = ["--checkpoint-dir", "/weights"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(args)
        parsed = benchmark.parse_args(args + ["--checkpoint-revision", "abc123"])
        self.assertEqual(parsed.checkpoint_revision, "abc123")

    def test_existing_artifacts_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "run"
            benchmark.create_output_dir(directory)
            evidence = directory / "metadata.json"
            benchmark.write_json(evidence, {"head": "original"})
            with self.assertRaises(FileExistsError):
                benchmark.create_output_dir(directory)
            self.assertEqual(json.loads(evidence.read_text()), {"head": "original"})
            before = benchmark.sha256_file(evidence)
            evidence.write_text("different")
            self.assertNotEqual(benchmark.sha256_file(evidence), before)

    def test_git_fingerprint_includes_working_diff(self):
        def command(argv, cwd):
            if "rev-parse" in argv:
                return {"stdout": "abc\n", "returncode": 0}
            if "status" in argv:
                return {"stdout": " M tracked.py\n?? new.py\n", "returncode": 0}
            return {"stdout": "patch-content", "returncode": 0}

        with patch.object(benchmark, "command_output", side_effect=command):
            first = benchmark.git_fingerprint(Path("/repo"))
        with patch.object(
            benchmark,
            "command_output",
            return_value={"stdout": "another-patch", "returncode": 0},
        ):
            second = benchmark.git_fingerprint(Path("/repo"))
        self.assertNotEqual(first["tracked_diff_sha256"], second["tracked_diff_sha256"])
        self.assertIn("?? new.py", first["status"]["stdout"])
        self.assertEqual(first["head"]["stdout"].strip(), "abc")

    def test_json_rejects_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                benchmark.write_json(Path(tmp) / "bad.json", {"ms": float("nan")})


class TestCuTileTensorParallelSlice(CustomTestCase):
    def setUp(self):
        super().setUp()
        experts, hidden, intermediate = 3, 128, 256

        def packed(shape):
            count = 1
            for size in shape:
                count *= size
            return (torch.arange(count).reshape(shape) % 251).to(torch.uint8)

        def scales(shape):
            return (packed(shape).float() / 16 + 1).to(torch.float8_e4m3fn)

        self.layer = helpers._layer(
            {
                "w13_weight_packed": packed((experts, intermediate * 2, hidden // 2)),
                "w13_weight_scale": scales((experts, intermediate * 2, hidden // 16)),
                "w13_weight_global_scale": torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ),
                "w13_input_global_scale": torch.ones(experts, 2),
                "w2_weight_packed": packed((experts, hidden, intermediate // 2)),
                "w2_weight_scale": scales((experts, hidden, intermediate // 16)),
                "w2_weight_global_scale": torch.tensor([7.0, 8.0, 9.0]),
                "w2_input_global_scale": torch.ones(experts),
            },
            top_k=2,
        )

    def test_backend_clone_has_marlin_group_size_and_independent_storage(self):
        clone = benchmark.clone_layer(self.layer, helpers)
        self.assertEqual(clone.quant_config.group_size, 16)
        self.assertEqual(clone.moe_runner_config.gate_up_interleaved, False)
        before = self.layer.w13_weight_packed.clone()
        clone.w13_weight_packed[0, 0, 0] = 255
        torch.testing.assert_close(self.layer.w13_weight_packed, before)

    def test_tp_slices_reconstruct_every_storage_axis(self):
        for tp_size in (1, 2, 4):
            with self.subTest(tp_size=tp_size):
                parts = [
                    helpers._slice_layer_for_tp(
                        self.layer, tp_rank=rank, tp_size=tp_size
                    )
                    for rank in range(tp_size)
                ]
                for rank, part in enumerate(parts):
                    self.assertEqual(part.num_experts, self.layer.num_experts)
                    self.assertEqual(part.moe_tp_rank, rank)
                    self.assertEqual(part.moe_tp_size, tp_size)
                    self.assertEqual(
                        part.intermediate_size_per_partition, 256 // tp_size
                    )
                    torch.testing.assert_close(
                        part.w13_weight_global_scale, self.layer.w13_weight_global_scale
                    )
                    torch.testing.assert_close(
                        part.w2_weight_global_scale, self.layer.w2_weight_global_scale
                    )
                for name in ("w13_weight_packed", "w13_weight_scale"):
                    branches = [
                        getattr(part, name).float().chunk(2, dim=1) for part in parts
                    ]
                    reconstructed = torch.cat(
                        [
                            torch.cat([pair[branch] for pair in branches], dim=1)
                            for branch in (0, 1)
                        ],
                        dim=1,
                    )
                    torch.testing.assert_close(
                        reconstructed, getattr(self.layer, name).float()
                    )
                for name in ("w2_weight_packed", "w2_weight_scale"):
                    reconstructed = torch.cat(
                        [getattr(part, name).float() for part in parts], dim=2
                    )
                    torch.testing.assert_close(
                        reconstructed, getattr(self.layer, name).float()
                    )

    def test_invalid_tp_geometry_is_rejected(self):
        for rank, size in ((-1, 2), (2, 2), (0, 0), (0, 3), (0, 8)):
            with self.subTest(rank=rank, size=size), self.assertRaises(ValueError):
                helpers._slice_layer_for_tp(self.layer, tp_rank=rank, tp_size=size)


if __name__ == "__main__":
    unittest.main()

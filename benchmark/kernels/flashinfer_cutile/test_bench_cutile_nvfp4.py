"""CPU checks for benchmark evidence handling and TP checkpoint slicing."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cutile_nvfp4 as benchmark

sys.path.insert(0, str(benchmark.REPO_ROOT))
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test.manual.layers.moe import test_flashinfer_cutile_nvfp4 as helpers

import torch


class TestCuTileBenchmark(CustomTestCase):
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

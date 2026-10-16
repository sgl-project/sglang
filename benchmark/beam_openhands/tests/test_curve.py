"""Safety and metric regressions. No GPU, network, model or verifier calls."""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from curve_config import SIMULATION, inside_repo, profile, server_args
from guarded_node import GIB, violations
from run_curve import plan
from summarize_rung import rung_metrics


class CurveTest(unittest.TestCase):
    def test_parallelisms_and_cache_boundary(self):
        a, b = profile("tp8", 16), profile("tep8", 16)
        pa, pb = a["resolved_args"], b["resolved_args"]
        self.assertEqual({k for k in pa if pa[k] != pb[k]}, {"ep_size"})
        self.assertEqual((pa["ep_size"], pb["ep_size"]), (1, 8))
        self.assertEqual(pa["hicache_size"], 88)
        self.assertNotIn("hicache_size", profile("tp8", 8)["resolved_args"])
        argv = server_args("tep8", 128, 1, "model", "draft", "head")
        self.assertNotIn("--speculative-draft-window-size", argv)
        self.assertNotIn("--moe-runner-backend", argv)
        self.assertNotIn("--moe-a2a-backend", argv)
        self.assertEqual(argv[argv.index("--speculative-num-draft-tokens") + 1], "8")
        self.assertEqual(SIMULATION["SGLANG_SIMULATE_ACC_LEN"], "4")
        self.assertEqual(
            SIMULATION["SGLANG_SIMULATE_ACC_TOKEN_MODE"], "real-draft-token"
        )

    def test_budgeted_hicache88_rerun_needs_no_extra_repeat(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "runtime"
        d = plan(
            {
                "work_dir": str(root),
                "draft_path": "/models/draft",
                "layouts": ["tep4"],
                "concurrencies_by_layout": {"tep4": [128, 64]},
                "hicache_size": 88,
                "repeat_cc128": False,
            },
            root / "results",
        )
        self.assertTrue(
            all(c["profile"]["resolved_args"]["hicache_size"] == 88 for c in d["cases"])
        )
        self.assertFalse(any(c["discard_first"] for c in d["cases"]))
        argv = server_args("tep4", 128, 0, "model", "draft", "head", hicache_size=88)
        self.assertEqual(argv[argv.index("--hicache-size") + 1], "88")
        self.assertNotIn("--moe-runner-backend", argv)
        self.assertNotIn("--moe-a2a-backend", argv)

    def test_descending_order_and_cc128_repeat(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "runtime"
        d = plan(
            {
                "work_dir": str(root),
                "draft_path": "/models/draft",
                "layouts": ["tep4", "tp4"],
                "repeat_cc128": True,
                "concurrencies_by_layout": {
                    "tep4": [4, 128, 64, 48, 32, 16, 8],
                    "tp4": [128, 64, 48, 32, 16, 8, 4, 1],
                },
            },
            root / "results",
        )
        self.assertEqual(
            [c["cc"] for c in d["cases"] if c["layout"] == "tep4"],
            [4, 128, 64, 48, 32, 16, 8],
        )
        self.assertEqual(
            [c["cc"] for c in d["cases"] if c["discard_first"]], [128, 128]
        )

    def test_footprint_tolerates_only_vanished_children(self):
        import tempfile
        from unittest.mock import patch

        from guarded_node import disk_footprint

        original = pathlib.Path.lstat
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "keep").write_bytes(b"x" * 1024)
            (root / "gone").write_bytes(b"tmp")

            def disappearing(path, *args, **kwargs):
                if path.name == "gone":
                    path.unlink()
                    raise FileNotFoundError(path)
                return original(path, *args, **kwargs)

            with patch.object(pathlib.Path, "lstat", disappearing):
                self.assertGreaterEqual(disk_footprint(root), 1024)

            def denied(path, *args, **kwargs):
                if path.name == "keep":
                    raise PermissionError(path)
                return original(path, *args, **kwargs)

            with (
                patch.object(pathlib.Path, "lstat", denied),
                self.assertRaises(PermissionError),
            ):
                disk_footprint(root)

    def test_footprint_counts_hardlinks_once_and_does_not_follow_symlinks(self):
        import os
        import tempfile

        from guarded_node import disk_footprint

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            f = root / "payload"
            f.write_bytes(b"x" * 1024)
            os.link(f, root / "hardlink")
            (root / "link").symlink_to("payload")
            self.assertEqual(
                disk_footprint(root),
                root.stat().st_size
                + f.stat().st_size
                + (root / "link").lstat().st_size,
            )
        with self.assertRaises(FileNotFoundError):
            disk_footprint(root)

    def test_released_beam_controls_are_normalized_even_when_not_special(self):
        import tempfile

        from sweep_tool import special_token_pattern

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            (path / "tokenizer.json").write_text(
                json.dumps(
                    {
                        "added_tokens": [
                            {"content": "<|channel|>", "special": False},
                            {"content": "<|message|>", "special": False},
                            {"content": "<think>", "special": False},
                        ]
                    }
                )
            )
            pattern = special_token_pattern(path)
            self.assertEqual(
                pattern.sub("", "<|channel|>analysis<|message|>text<think>"),
                "analysistext<think>",
            )

    def test_single_node_layout_and_gpu_normalization(self):
        for layout, ep in (("tp4", 1), ("tep4", 4)):
            p = profile(layout, 128)
            self.assertEqual(p["n_gpus"], 4)
            self.assertEqual(p["resolved_args"]["nnodes"], 1)
            self.assertEqual(p["resolved_args"]["tp_size"], 4)
            self.assertEqual(p["resolved_args"]["ep_size"], ep)
            with self.assertRaises(ValueError):
                server_args(layout, 128, 1, "model", "draft", "head")
        root = pathlib.Path(__file__).resolve().parents[1] / "runtime"
        d = plan(
            {
                "work_dir": str(root),
                "draft_path": "/models/draft",
                "layouts": ["tep4", "tp4"],
            },
            root / "results",
        )
        self.assertEqual([c["layout"] for c in d["cases"]], ["tep4"] * 5 + ["tp4"] * 5)

    def test_artifacts_cannot_escape_beam_checkout(self):
        with self.assertRaises(ValueError):
            inside_repo("/tmp/foreign-benchmark")

    def test_default_tp4_recipe_and_total_metric(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "runtime"
        d = plan(
            {"work_dir": str(root), "draft_path": "/models/draft"}, root / "results"
        )
        self.assertEqual(len(d["cases"]), 5)
        self.assertIn("cached input", d["primary_y"])
        self.assertEqual(
            [c["cc"] for c in d["cases"] if c["layout"] == "tp4"],
            [1, 4, 8, 16, 32],
        )

    def test_memory_disk_and_oom_stops(self):
        base = dict(
            working_bytes=400 * GIB,
            limit_bytes=626 * GIB,
            host_available_bytes=300 * GIB,
            disk_available_bytes=2000 * GIB,
            oom_events={"oom": 0, "oom_kill": 0},
            host_oom_kill=49,
        )
        self.assertEqual(violations(base, base, 10 * GIB), [])
        for key, value in [
            ("working_bytes", 489 * GIB),
            ("host_available_bytes", 199 * GIB),
            ("disk_available_bytes", 1023 * GIB),
            ("host_oom_kill", 50),
            ("oom_events", {"oom": 1, "oom_kill": 0}),
        ]:
            with self.subTest(key=key):
                self.assertTrue(violations({**base, key: value}, base, 10 * GIB))
        self.assertTrue(violations(base, base, 81 * GIB))

    def test_p90_is_reciprocal_of_latency_percentile(self):
        percentiles = [
            {"Percentile": f"{p}%", "TPOT (ms)": v, "TTFT (ms)": 100}
            for p, v in [(50, 10), (90, 25), (99, 50)]
        ]
        # Native percentile files use a string key for their percentile label.
        # Match the reference format, whose first field is Percentiles.
        for row in percentiles:
            row["Percentiles"] = row.pop("Percentile")
        r = rung_metrics(
            {"Total Throughput (tok/s)": 8000, "Output Throughput (tok/s)": 80},
            percentiles,
            {},
            8,
        )
        self.assertEqual(r["interactivity_p50"], 100)
        self.assertEqual(r["interactivity_p90"], 40)
        self.assertEqual(r["total_tok_s_per_gpu"], 1000)


if __name__ == "__main__":
    unittest.main()

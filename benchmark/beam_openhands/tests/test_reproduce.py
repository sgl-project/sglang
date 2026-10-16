"""Reproduction entry-point checks; no GPU or network operations."""

import argparse
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import reproduce
from guarded_node import GIB
from run_curve import node_job, plan


class ReproductionTest(unittest.TestCase):
    def setUp(self):
        runtime = HERE / "runtime"
        runtime.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="repro-test-", dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.args = argparse.Namespace(
            work_dir=str(self.work),
            model_path="/models/Beam target",
            draft_path="/models/Beam draft",
            server_python=sys.executable,
            client_python=sys.executable,
            layouts=["tp4"],
            concurrencies=[1, 4, 8, 16, 32],
            port=30080,
            dist_port=39746,
            runtime_pythonpath=[],
        )

    def test_init_produces_complete_matched_plan_and_quoted_site(self):
        path = reproduce.initialize(self.args)
        settings = json.loads(path.read_text())
        document = plan(settings, self.work / "results")
        self.assertEqual([c["cc"] for c in document["cases"]], [1, 4, 8, 16, 32])
        self.assertTrue(
            all(
                c["layout"] == "tp4" and not c["discard_first"]
                for c in document["cases"]
            )
        )
        self.assertEqual(
            [
                c["profile"]["resolved_args"].get("hicache_size", 0)
                for c in document["cases"]
            ],
            [0, 0, 0, 88, 88],
        )
        for line in (self.work / "site.env").read_text().splitlines():
            self.assertEqual(len(shlex.split(line)), 2)
        self.assertIn("N_GPUS=4", (self.work / "site.env").read_text())
        self.assertTrue((self.work / "bootstrap-profile.json").is_file())

    def test_init_preserves_existing_state_and_rejects_changed_settings(self):
        path = reproduce.initialize(self.args)
        original = path.read_bytes()
        reproduce.initialize(self.args)
        self.assertEqual(path.read_bytes(), original)
        self.args.concurrencies = [4]
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            reproduce.initialize(self.args)
        self.assertEqual(path.read_bytes(), original)

    def test_weights_cannot_consume_experiment_storage_budget(self):
        self.args.model_path = str(self.work / "weights")
        with self.assertRaisesRegex(ValueError, "outside work_dir"):
            reproduce.initialize(self.args)
        self.assertFalse((self.work / "settings.json").exists())

    def test_different_work_directories_share_gpu_ownership_lock(self):
        settings = json.loads(reproduce.initialize(self.args).read_text())
        case = {"layout": "tp4", "cc": 1}
        first = node_job(settings, case, 0, self.work / "first")
        second = node_job(
            {**settings, "work_dir": str(self.work / "second-work")},
            case,
            0,
            self.work / "second",
        )
        self.assertEqual(first["gpu_lock_file"], second["gpu_lock_file"])

    def test_prepare_job_does_not_persist_inherited_secrets(self):
        settings = json.loads(reproduce.initialize(self.args).read_text())
        with patch.dict(
            os.environ, {"HF_TOKEN": "sentinel-private", "PRIVATE_KEY": "sentinel-key"}
        ):
            job = reproduce.preparation_job(settings, "prepare-check")
        encoded = json.dumps(job)
        self.assertNotIn("sentinel", encoded)
        self.assertNotIn("PYTHONPATH", job["env"])
        self.assertIn("BEAM_SERVER_PYTHONPATH", job["env"])
        self.assertFalse(job["gpu"])
        self.assertEqual(job["env"]["SITE_ENV"], str(self.work / "site.env"))

    def test_prelaunch_budget_rejects_undersized_container(self):
        settings = json.loads(reproduce.initialize(self.args).read_text())
        baseline = dict(
            working_bytes=2 * GIB,
            limit_bytes=860 * GIB,
            host_available_bytes=815 * GIB,
            disk_available_bytes=2000 * GIB,
            oom_events={"oom": 0, "oom_kill": 0},
            host_oom_kill=0,
        )
        with (
            patch.object(reproduce, "snapshot", return_value=baseline),
            patch.object(reproduce, "disk_footprint", return_value=8 * GIB),
        ):
            reproduce.check_run_budget(settings)
        with (
            patch.object(
                reproduce,
                "snapshot",
                return_value={**baseline, "limit_bytes": 626 * GIB},
            ),
            patch.object(reproduce, "disk_footprint", return_value=8 * GIB),
        ):
            with self.assertRaisesRegex(RuntimeError, "cgroup headroom"):
                reproduce.check_run_budget(settings)
        self.assertEqual(len(list(self.work.glob("budget-*.json"))), 2)


if __name__ == "__main__":
    unittest.main()

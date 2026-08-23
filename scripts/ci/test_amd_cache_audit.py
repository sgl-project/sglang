"""Unit tests for amd_cache_audit extraction and join logic.

Pure-logic plus one end-to-end smoke test (stdlib only, no GPU, no sglang
import), mirroring test_list_stage_models.py so both run in the
ci-model-inventory workflow without installing dependencies:

    python -m unittest discover -s scripts/ci -p 'test_amd_cache_audit.py'

The end-to-end case is the point of this file. amd_cache_audit.py is a CLI
tool that no workflow executes, and it reuses list_stage_models.py for the
suite -> models half; without a test, a refactor there would break the audit
silently and nobody would notice until someone deleted the wrong checkpoints.
So `test_compute_smoke` runs the real join against the real repo and asserts
the contract -- not exact counts, which move every time a test is added.
"""

import importlib.util
import os
import tempfile
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODULE_PATH = os.path.join(REPO_ROOT, "scripts", "ci", "utils", "amd_cache_audit.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("amd_cache_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load_module()


class TestJobRunnerLabels(unittest.TestCase):
    """`--runner` scoping is only as good as the labels pulled off each job."""

    def test_plain_string(self):
        self.assertEqual(
            audit._job_runner_labels({"runs-on": "linux-mi35x-gpu-8"}),
            ["linux-mi35x-gpu-8"],
        )

    def test_list_form(self):
        self.assertEqual(
            audit._job_runner_labels({"runs-on": ["self-hosted", "linux-mi300"]}),
            ["self-hosted", "linux-mi300"],
        )

    def test_matrix_form(self):
        """AMD jobs commonly say `runs-on: ${{ matrix.runner }}`; the literal
        labels live in strategy.matrix, and the unexpanded expression itself
        must not be mistaken for a label."""
        job = {
            "runs-on": "${{ matrix.runner }}",
            "strategy": {"matrix": {"runner": ["linux-mi35x-gpu-1"]}},
        }
        self.assertEqual(audit._job_runner_labels(job), ["linux-mi35x-gpu-1"])

    def test_drops_unexpanded_expressions(self):
        job = {"runs-on": "${{ inputs.runner }}"}
        self.assertEqual(audit._job_runner_labels(job), [])

    def test_missing_and_malformed_keys(self):
        self.assertEqual(audit._job_runner_labels({}), [])
        self.assertEqual(audit._job_runner_labels({"strategy": "not-a-dict"}), [])


class TestJobText(unittest.TestCase):
    def test_concatenates_run_blocks_only(self):
        job = {
            "steps": [
                {"uses": "actions/checkout@v4"},
                {"run": "echo one"},
                {"name": "no run key"},
                {"run": "echo two"},
            ]
        }
        self.assertEqual(audit._job_text(job), "echo one\necho two")

    def test_no_steps(self):
        self.assertEqual(audit._job_text({}), "")


class TestReadCache(unittest.TestCase):
    """`models--org--name` is HuggingFace's on-disk spelling of `org/name`."""

    def test_listing_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hub.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(
                    "models--amd--Qwen3.8-2.4T-A95B-Quark-MXFP4\n"
                    "models--moonshotai--Kimi-K3\n"
                    "datasets--foo--bar\n"  # not a model dir
                    "\n"
                )
            self.assertEqual(
                audit._read_cache(None, path),
                {"amd/Qwen3.8-2.4T-A95B-Quark-MXFP4", "moonshotai/Kimi-K3"},
            )

    def test_only_first_separator_splits_org(self):
        """Repo names may themselves contain `--`; only the org boundary counts."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hub.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("models--org--name--with--dashes\n")
            self.assertEqual(audit._read_cache(None, path), {"org/name--with--dashes"})

    def test_directory_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "models--meta-llama--Llama-3.2-1B-Instruct"))
            os.mkdir(os.path.join(tmp, "tmp-not-a-model"))
            self.assertEqual(
                audit._read_cache(tmp, None), {"meta-llama/Llama-3.2-1B-Instruct"}
            )

    def test_missing_directory_is_not_fatal(self):
        self.assertEqual(audit._read_cache("/nonexistent/path", None), set())


class TestComputeSmoke(unittest.TestCase):
    """End-to-end against the real repo: the guard against silent rot."""

    @classmethod
    def setUpClass(cls):
        cls.result = audit.compute()

    def test_shape(self):
        for key in ("suite_count", "suite_reachability", "models", "suite_jobs"):
            self.assertIn(key, self.result)
        self.assertGreater(self.result["suite_count"], 0, "no AMD suites resolved")
        self.assertGreater(len(self.result["models"]), 0, "no checkpoints resolved")

    def test_per_commit_dispatch_is_counted(self):
        """The bug this tool exists to prevent: joining nightly only. If the
        per-commit bucket is ever empty, the reachability join has regressed
        and every per-commit-only checkpoint would read as reclaimable."""
        reach = self.result["suite_reachability"]
        self.assertGreater(
            len(reach["per_commit_only"]) + len(reach["both"]),
            0,
            "no AMD suite resolved to a per-commit workflow",
        )

    def test_constant_resolution_survives(self):
        """Models reached only through a shared constant must resolve. This one
        is named nowhere as a literal in an AMD test -- it arrives via
        DEFAULT_SMALL_MODEL_NAME_FOR_TEST -- so it catches the case where the
        extractor degrades to plain string matching."""
        self.assertIn("meta-llama/Llama-3.2-1B-Instruct", self.result["models"])

    def test_runner_scoping_narrows(self):
        """--runner must actually filter; without it a cluster gets compared
        against every AMD suite and the other cluster's models read as missing."""
        scoped = audit.compute(runner="mi35x")
        needed_all = {
            m
            for m, i in self.result["models"].items()
            if i["verdict"] != "undispatched"
        }
        needed_mi35x = {
            m for m, i in scoped["models"].items() if i["verdict"] != "undispatched"
        }
        self.assertTrue(needed_mi35x, "mi35x scoping resolved nothing")
        self.assertLess(len(needed_mi35x), len(needed_all))


if __name__ == "__main__":
    unittest.main(verbosity=2)

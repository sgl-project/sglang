"""Unit tests for /rerun-test backend resolution in slash_command_handler.

Needs requests and PyYAML, not the sglang package (PyGithub is stubbed).
Each case writes its test files into a temporary repo root; runner and suite
settings come from the real runner_configs.yml and NPU workflows.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.modules.setdefault("github", types.SimpleNamespace(Auth=object(), Github=object()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slash_command_handler as handler  # noqa: E402

CUDA = 'register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")\n'
NPU = 'register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")\n'


class TestNpuSuites(unittest.TestCase):
    def test_reads_the_real_workflows(self):
        suites = handler._npu_suites.load_suites()
        self.assertEqual(
            suites["nightly-4-npu-a3"],
            handler._npu_suites.NpuSuite("linux-aarch64-a3-4-", True, True),
        )
        # pr-test-npu.yml picks the runner by a coverage expression.
        self.assertEqual(suites["base-b-test-1-npu-a3"].runs_on, "linux-aarch64-a3-2-")
        # Run by _npu-single-node-test-stage.yml.
        self.assertNotIn("base-c-test-acc-2-npu-a3", suites)
        self.assertRegex(handler._npu_suites.a3_pr_image(), r"^[\w./:-]+-a3-[\w.-]+$")


class _TempRepo(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp.name)

    def write_test(self, name, registrations):
        path = f"registered/x/{name}"
        os.makedirs("test/registered/x", exist_ok=True)
        with open(f"test/{path}", "w") as f:
            f.write(f"from sglang.test.ci.ci_register import *\n\n{registrations}")
        return path

    def rerun(self, specs, hw=None):
        """Run /rerun-test; return (dispatched inputs, reply comment body)."""
        gh_repo = mock.MagicMock()
        gh_repo.get_workflows.return_value = [
            types.SimpleNamespace(name="Rerun Test", id=7)
        ]
        pr = mock.MagicMock()
        pr.head.repo.owner.login = gh_repo.owner.login
        with (
            mock.patch.object(handler, "_check_rebase_gate", return_value=(True, None)),
            mock.patch.object(handler, "find_workflow_run_url", return_value="u"),
            mock.patch.object(handler.requests, "post") as post,
        ):
            post.return_value.status_code = 204
            handler.handle_rerun_test(
                gh_repo,
                pr,
                mock.MagicMock(),
                {"cooldown_interval_minutes": 0},
                specs,
                token="t",
                hw=hw,
            )
        inputs = [c.kwargs["json"]["inputs"] for c in post.call_args_list]
        edit = pr.create_issue_comment.return_value.edit
        # A rejected command posts one comment; otherwise a placeholder is edited.
        reply = (edit if edit.called else pr.create_issue_comment).call_args.args[0]
        return inputs, reply


class TestRerunHw(_TempRepo):
    def test_rejections(self):
        for args, selector, error in (
            ('60, suite="base-c-test-acc-2-npu-a3"', "", "has no a3 job calling"),
            ('60, suite="base-b-test-1-npu-a3", disabled="flaky"', "", "(flaky)"),
            ('60, suite="base-b-test-1-npu-a3"', "::TestA.test_b", "drop the `::"),
            ('7200, suite="full-1-npu-a3", nightly=True', "", "60-minute"),
        ):
            with self.subTest(error=error):
                path = self.write_test("test_npu.py", f"register_npu_ci({args})\n")
                [entry] = handler._resolve_test_spec(path + selector)
                self.assertIn(error, entry["error"])

    def test_routing(self):
        npu_only = self.write_test("test_npu.py", NPU)
        both = self.write_test("test_both.py", CUDA + NPU)
        resolve = handler._resolve_test_spec
        [entry] = resolve(npu_only)
        self.assertEqual(
            (entry["mode"], entry["suite"], entry["runs_on"]),
            ("npu", "base-b-test-1-npu-a3", "linux-aarch64-a3-2-"),
        )
        self.assertEqual([e["mode"] for e in resolve(both)], ["cuda"])
        self.assertEqual([e["mode"] for e in resolve(both, "npu")], ["npu"])

    def test_hw_list(self):
        both = self.write_test("test_both.py", CUDA + NPU)
        cuda_only = self.write_test("test_cuda.py", CUDA)
        inputs, reply = self.rerun([both, cuda_only], hw="cuda,npu")
        self.assertEqual(
            [(i["mode"], i.get("suite")) for i in inputs],
            [("cuda", None), ("npu", "base-b-test-1-npu-a3")],
        )
        self.assertIn(f"python3 {both}  # base-b\n", reply)
        self.assertIn(f"python3 {cuda_only}  # base-b (npu is skipped)\n", reply)
        self.assertNotIn("⛔", reply)

        inputs, reply = self.rerun([cuda_only], hw="npu")
        self.assertEqual(inputs, [])
        self.assertIn("No NPU registration, so `--hw npu` has nothing to run.", reply)

    def test_hw_all(self):
        cuda_only = self.write_test("test_cuda.py", CUDA)
        amd_only = self.write_test(
            "test_amd.py", 'register_amd_ci(est_time=60, suite="a")\n'
        )
        inputs, reply = self.rerun([cuda_only, amd_only], hw="all")
        self.assertEqual([i["mode"] for i in inputs], ["cuda"])
        self.assertIn(f"python3 {cuda_only}  # base-b\n", reply)
        self.assertIn(
            f"⛔ `{amd_only}`: `test/{amd_only}` is registered for AMD", reply
        )

    def test_parse_errors(self):
        self.assertEqual(
            handler._pop_hw_flag(["a.py", "--hw", "npu", "-c"]), ("npu", ["a.py", "-c"])
        )
        self.assertEqual(handler._pop_hw_flag(["a.py", "--hw"]), ("", ["a.py"]))
        for hw in ("", "tpu", "cuda,", "npu,npu", "all,npu"):
            with self.subTest(hw=hw):
                inputs, reply = self.rerun(["a.py"], hw=hw)
                self.assertEqual(inputs, [])
                self.assertEqual(
                    reply,
                    "⛔ `--hw` takes `all` or a comma-separated list of `cuda`, `npu`.",
                )


if __name__ == "__main__":
    unittest.main()

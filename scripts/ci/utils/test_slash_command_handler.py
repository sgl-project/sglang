"""Unit tests for /rerun-test resolution and dispatch in slash_command_handler.

Runs without GPUs, NPUs, or the sglang package (PyGithub is stubbed; needs
requests and PyYAML):

    python -m pytest scripts/ci/utils/test_slash_command_handler.py

Test files are written into a temporary repo root, so each case pins its own
registrations; runner and suite settings come from the real
scripts/ci/runner_configs.yml and the NPU and AMD workflows.
"""

import json
import os
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.modules.setdefault("github", types.SimpleNamespace(Auth=object(), Github=object()))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slash_command_handler as handler  # noqa: E402

npu_suites = handler._npu_suites
amd_suites = handler._amd_suites
A3_PR_IMAGE = npu_suites.a3_pr_image()


class TestNpuSuites(unittest.TestCase):
    def test_reads_the_real_workflows(self):
        suites = npu_suites.load_suites()
        self.assertEqual(
            suites["nightly-4-npu-a3"],
            npu_suites.NpuSuite(
                "_npu-pr-test-stage.yml", "linux-aarch64-a3-4-", sgl_router=True
            ),
        )
        # pr-test-npu.yml picks a coverage runner by expression; the normal one wins.
        self.assertEqual(suites["base-b-test-1-npu-a3"].runs_on, "linux-aarch64-a3-2-")
        self.assertEqual(
            suites["base-c-test-acc-2-npu-a3"],
            npu_suites.NpuSuite(
                "_npu-single-node-test-stage.yml",
                "linux-aarch64-a3-2-",
                sgl_router=False,
            ),
        )
        self.assertNotIn("base-a-test-npu", suites)

    def test_pr_image_is_the_one_pr_ci_installs_into(self):
        workflow = yaml.safe_load(
            (npu_suites.WORKFLOWS_DIR / npu_suites.PR_WORKFLOW).read_text()
        )
        [step] = workflow["jobs"]["set-image-config"]["steps"]
        self.assertIn(f"CANN_image_a3=${{{npu_suites.A3_PR_IMAGE_ENV}}}", step["run"])

    def test_conflicting_jobs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, runner in zip(
                npu_suites.WORKFLOWS, ("linux-aarch64-a3-2-", "linux-aarch64-a3-4-")
            ):
                job = {
                    "uses": "./.github/workflows/_npu-pr-test-stage.yml",
                    "with": {"self_name": "s-npu-a3", "runner_config": runner},
                }
                Path(tmp, name).write_text(yaml.safe_dump({"jobs": {"j": job}}))
            Path(tmp, npu_suites.WORKFLOWS[2]).write_text("jobs: {}\n")
            with self.assertRaisesRegex(ValueError, "`s-npu-a3` runs with"):
                npu_suites.load_suites(Path(tmp))


class TestAmdSuites(unittest.TestCase):
    def test_reads_the_real_workflows(self):
        suites = amd_suites.load_suites()
        self.assertEqual(
            suites["stage-b-test-1-gpu-small-amd"],
            amd_suites.AmdSuite(
                runs_on="linux-mi300-1gpu-sglang",
                rocm_version="rocm10",
                sgl_router=False,
                exec_env=(),
            ),
        )
        # The PR runner, not the Spur one that accepts only main.
        self.assertEqual(
            suites["stage-c-test-large-8-gpu-amd-mi35x"].runs_on, "linux-mi35x-gpu-8"
        )
        self.assertEqual(
            suites["stage-c-test-4-gpu-amd"].exec_env,
            (
                "NCCL_CUMEM_ENABLE=0",
                "NCCL_NVLS_ENABLE=0",
                "RCCL_MSCCL_ENABLE=0",
                "SGLANG_USE_ROCM700A=1",
            ),
        )
        extra = suites["extra-a-test-2-gpu-large-amd"]
        self.assertEqual(
            (extra.runs_on, extra.rocm_version, extra.sgl_router),
            ("linux-mi300-2gpu-sglang", "rocm10", True),
        )
        disagg = "stage-b-test-large-8-gpu-mi35x-disaggregation-amd"
        self.assertNotIn(disagg, suites)
        self.assertIn(disagg, amd_suites.own_setup_jobs())

    def test_conflicting_jobs_fail(self):
        def job(runner):
            return {
                "runs-on": runner,
                "steps": [
                    {
                        "uses": amd_suites.SETUP_ACTION,
                        "with": {"rocm_version": "rocm10"},
                    },
                    {
                        "run": "bash scripts/ci/amd/amd_ci_exec.sh -w /x "
                        "python3 run_suite.py --hw amd --suite s-amd"
                    },
                ],
            }

        with tempfile.TemporaryDirectory() as tmp:
            for name, runner in zip(amd_suites.WORKFLOWS, ("r-1gpu", "r-2gpu")):
                Path(tmp, name).write_text(yaml.safe_dump({"jobs": {"j": job(runner)}}))
            with self.assertRaisesRegex(ValueError, "`s-amd` runs with"):
                amd_suites.load_suites(Path(tmp))


class _TempRepo(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def write_test(self, rel_path, registrations):
        """Write test/<rel_path> carrying the given registration lines."""
        path = os.path.join("test", rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(
                "from sglang.test.ci.ci_register import *\n\n"
                + textwrap.dedent(registrations).strip()
                + '\n\n\nif __name__ == "__main__":\n    main()\n'
            )
        return rel_path


class TestResolveNpu(_TempRepo):
    def test_pr_stage_suite_with_sgl_router(self):
        path = self.write_test(
            "registered/npu/test_pd.py",
            'register_npu_ci(est_time=600, suite="base-b-test-4-npu-a3")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertEqual(
            entry,
            {
                "spec": path,
                "test_command": path,
                "mode": "npu",
                "stage": "base-b-test-4-npu-a3",
                "runs_on": "linux-aarch64-a3-4-",
                "install_script": "",
                "install_timeout": "",
                "rdma_devices": "",
                **handler._MODE_INPUTS,
                "npu_suite": "base-b-test-4-npu-a3",
                "npu_image": A3_PR_IMAGE,
                "npu_sgl_router": True,
                "npu_collect_nightly_tests": False,
                "error": None,
            },
        )

    def test_nightly_registration_collects_nightly_tests(self):
        path = self.write_test(
            "registered/npu/test_full.py",
            'register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertEqual(entry["npu_suite"], "full-1-npu-a3")
        self.assertTrue(entry["npu_collect_nightly_tests"])

    def test_single_node_stage_suite(self):
        path = self.write_test(
            "registered/npu/test_acc.py",
            'register_npu_ci(est_time=900, suite="base-c-test-acc-2-npu-a3")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertEqual(entry["mode"], "npu_single_node")
        self.assertEqual(entry["runs_on"], "linux-aarch64-a3-2-")
        self.assertEqual(entry["npu_image"], A3_PR_IMAGE)

    def test_one_entry_per_registration(self):
        path = self.write_test(
            "registered/npu/test_two.py",
            """
            register_npu_ci(est_time=600, suite="base-b-test-1-npu-a3")
            register_npu_ci(est_time=600, suite="full-8-npu-a3", nightly=True)
            """,
        )
        entries = handler._resolve_test_spec(path)
        self.assertEqual(
            [(e["npu_suite"], e["runs_on"]) for e in entries],
            [
                ("base-b-test-1-npu-a3", "linux-aarch64-a3-2-"),
                ("full-8-npu-a3", "linux-aarch64-a3-8"),
            ],
        )

    def test_multi_node_registration_is_rejected(self):
        path = self.write_test(
            "registered/npu/test_multi_node.py",
            'register_npu_ci(est_time=3000, suite="", nightly=True)',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("empty suite", entry["error"])
        self.assertIn("nightly-test-npu-e2e-multi-node.yml", entry["error"])

    def test_suite_without_stage_job_is_rejected(self):
        path = self.write_test(
            "registered/unit/npu/test_cpu_side.py",
            'register_npu_ci(est_time=60, suite="base-a-test-npu")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("`base-a-test-npu`", entry["error"])
        self.assertIn("run by no NPU stage job", entry["error"])

    def test_non_a3_suite_is_rejected(self):
        path = self.write_test(
            "registered/npu/test_a5.py",
            'register_npu_ci(est_time=600, suite="nightly-acc-4-npu-a5", nightly=True)',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("runs on `linux-amd64-a5-4`", entry["error"])
        self.assertIn("only a3 suites", entry["error"])

    def test_disabled_registration_is_rejected(self):
        path = self.write_test(
            "registered/npu/test_disabled.py",
            'register_npu_ci(est_time=600, suite="base-b-test-1-npu-a3", disabled="flaky")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("disabled", entry["error"])
        self.assertIn("flaky", entry["error"])

    def test_selector_is_rejected(self):
        path = self.write_test(
            "registered/npu/test_sel.py",
            """
            register_npu_ci(est_time=600, suite="base-b-test-1-npu-a3")
            register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)
            """,
        )
        self.assertEqual(
            handler._resolve_test_spec(f"{path}::TestX.test_y"),
            [
                {
                    "spec": f"{path}::TestX.test_y",
                    "error": "NPU tests run as whole files through run_suite.py; "
                    "drop the `::TestX.test_y` selector.",
                }
            ],
        )

    def test_over_step_limit_is_rejected(self):
        path = self.write_test(
            "registered/npu/test_long.py",
            'register_npu_ci(est_time=7200, suite="full-acc-2-npu-a3", nightly=True)',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("longer than the 60-minute", entry["error"])

    def test_shell_unsafe_suite_is_rejected(self):
        suite = "x;curl"
        path = self.write_test(
            "registered/npu/test_unsafe.py",
            f'register_npu_ci(est_time=60, suite="{suite}")',
        )
        suites = {
            suite: npu_suites.NpuSuite(
                "_npu-pr-test-stage.yml", "linux-aarch64-a3-2-", sgl_router=False
            )
        }
        with mock.patch.object(npu_suites, "load_suites", return_value=suites):
            [entry] = handler._resolve_test_spec(path)
        self.assertIn("Disallowed NPU suite or image `x;curl`", entry["error"])


class TestResolveAmd(_TempRepo):
    def test_amd_only_file_resolves_to_amd(self):
        path = self.write_test(
            "registered/amd/test_amd.py",
            'register_amd_ci(est_time=60, stage="stage-c", runner_config="4-gpu-amd")',
        )
        [entry] = handler._resolve_test_spec(f"{path}::TestA.test_b")
        self.assertEqual(
            entry,
            {
                "spec": f"{path}::TestA.test_b",
                "test_command": f"{path} TestA.test_b",
                "mode": "amd",
                "stage": "stage-c-test-4-gpu-amd",
                "runs_on": "linux-mi300-4gpu-sglang",
                "install_script": "",
                "install_timeout": "",
                "rdma_devices": "",
                **handler._MODE_INPUTS,
                "amd_rocm_version": "rocm10",
                "amd_sgl_router": False,
                "amd_exec_env": "NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 "
                "RCCL_MSCCL_ENABLE=0 SGLANG_USE_ROCM700A=1",
                "error": None,
            },
        )

    def test_one_entry_per_registration(self):
        path = self.write_test(
            "registered/amd/test_two.py",
            """
            register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")
            register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")
            register_amd_ci(est_time=60, stage="extra-a", runner_config="2-gpu-large-amd")
            """,
        )
        self.assertEqual(
            [
                (e["stage"], e["runs_on"], e["amd_sgl_router"])
                for e in handler._resolve_test_spec(path)
            ],
            [
                ("stage-b-test-1-gpu-small-amd", "linux-mi300-1gpu-sglang", False),
                ("stage-b-test-1-gpu-small-amd-mi35x", "linux-mi35x-gpu-1", False),
                ("extra-a-test-2-gpu-large-amd", "linux-mi300-2gpu-sglang", True),
            ],
        )

    def test_disabled_registration_still_runs_directly(self):
        path = self.write_test(
            "registered/amd/test_disabled.py",
            'register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd", '
            'disabled="flaky")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertEqual(entry["mode"], "amd")

    def test_nightly_registration_is_rejected(self):
        path = self.write_test(
            "registered/amd/test_nightly.py",
            'register_amd_ci(est_time=600, suite="nightly-amd-1-gpu", nightly=True)',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("AMD nightly suite `nightly-amd-1-gpu`", entry["error"])

    def test_disaggregation_suite_is_rejected(self):
        suite = "stage-b-test-large-8-gpu-mi35x-disaggregation-amd"
        path = self.write_test(
            "registered/amd/test_disagg.py",
            f'register_amd_ci(est_time=600, suite="{suite}")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn(f"pr-test-amd.yml job `{suite}`", entry["error"])
        self.assertIn("sets up its own container", entry["error"])

    def test_suite_without_pr_job_is_rejected(self):
        path = self.write_test(
            "registered/amd/test_orphan.py",
            'register_amd_ci(est_time=60, suite="stage-z-test-amd")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("`stage-z-test-amd`", entry["error"])
        self.assertIn("run by no job in pr-test-amd.yml", entry["error"])

    def test_over_step_limit_is_rejected(self):
        path = self.write_test(
            "registered/amd/test_long.py",
            'register_amd_ci(est_time=7200, suite="stage-b-test-1-gpu-small-amd")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("on `linux-mi300-1gpu-sglang`, longer than", entry["error"])

    def test_shell_unsafe_value_is_rejected(self):
        path = self.write_test(
            "registered/amd/test_unsafe.py",
            'register_amd_ci(est_time=60, suite="s-amd")',
        )
        suites = {
            "s-amd": amd_suites.AmdSuite(
                "linux-mi300-1gpu-sglang", "rocm10", False, ("A=$(id)",)
            )
        }
        with mock.patch.object(amd_suites, "load_suites", return_value=suites):
            [entry] = handler._resolve_test_spec(path)
        self.assertIn(
            "Disallowed AMD runner, ROCm version, or container env `A=$(id)`",
            entry["error"],
        )


class TestResolveUnchanged(_TempRepo):
    def test_cuda_registration(self):
        path = self.write_test(
            "registered/core/test_cuda.py",
            'register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")',
        )
        [entry] = handler._resolve_test_spec(f"{path}::TestA.test_b")
        self.assertEqual(
            entry,
            {
                "spec": f"{path}::TestA.test_b",
                "test_command": f"{path} TestA.test_b",
                "mode": "cuda",
                "stage": "base-b",
                "runs_on": "1-gpu-h100",
                "install_script": "scripts/ci/cuda/ci_install_dependency.sh",
                "install_timeout": "20",
                "rdma_devices": "",
                **handler._MODE_INPUTS,
                "error": None,
            },
        )

    def test_cuda_wins_over_npu(self):
        path = self.write_test(
            "registered/core/test_both.py",
            """
            register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
            register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
            """,
        )
        self.assertEqual(
            [e["mode"] for e in handler._resolve_test_spec(path)], ["cuda"]
        )

    def test_cpu_registration(self):
        path = self.write_test(
            "registered/unit/test_cpu.py",
            'register_cpu_ci(est_time=10, suite="base-a-test-cpu")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertEqual(entry["mode"], "cpu")
        self.assertEqual(entry["runs_on"], "ubuntu-latest")
        self.assertEqual(entry["stage"], "base-a")
        self.assertEqual(entry["npu_suite"], "")

    def test_xpu_only_is_rejected(self):
        path = self.write_test(
            "registered/xpu/test_xpu.py",
            'register_xpu_ci(est_time=60, suite="per-commit-xpu")',
        )
        [entry] = handler._resolve_test_spec(path)
        self.assertIn("registered for XPU", entry["error"])
        self.assertIn("not for CUDA, CPU, NPU, or AMD", entry["error"])


class TestHwFlag(_TempRepo):
    def write_both(self):
        return self.write_test(
            "registered/core/test_both.py",
            """
            register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
            register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")
            """,
        )

    def test_npu_only_file_runs_on_npu_without_flag(self):
        path = self.write_test(
            "registered/npu/test_npu.py",
            'register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")',
        )
        self.assertEqual([e["mode"] for e in handler._resolve_test_spec(path)], ["npu"])

    def test_hw_npu_picks_npu_over_cuda(self):
        entries = handler._resolve_test_spec(self.write_both(), hw="npu")
        self.assertEqual(
            [(e["mode"], e["npu_suite"]) for e in entries],
            [("npu", "base-b-test-1-npu-a3")],
        )

    def test_hw_npu_without_npu_registration_is_rejected(self):
        path = self.write_test(
            "registered/core/test_cuda.py",
            'register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")',
        )
        [entry] = handler._resolve_test_spec(path, hw="npu")
        self.assertIn("has no `register_npu_ci(...)`", entry["error"])

    def test_hw_npu_rejects_multimodal(self):
        os.makedirs("python/sglang/multimodal_gen/test")
        Path("python/sglang/multimodal_gen/test/test_gen.py").write_text("")
        [entry] = handler._resolve_test_spec(
            "python/sglang/multimodal_gen/test/test_gen.py", hw="npu"
        )
        self.assertIn("multimodal_gen tests run on CUDA", entry["error"])

    def test_hw_cuda_keeps_cuda_and_never_falls_back_to_npu(self):
        self.assertEqual(
            [e["mode"] for e in handler._resolve_test_spec(self.write_both(), "cuda")],
            ["cuda"],
        )
        path = self.write_test(
            "registered/npu/test_npu.py",
            'register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")',
        )
        [entry] = handler._resolve_test_spec(path, hw="cuda")
        self.assertIn("drop `--hw cuda`", entry["error"])

    def write_cuda_and_amd(self):
        return self.write_test(
            "registered/core/test_cuda_amd.py",
            """
            register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
            register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")
            """,
        )

    def test_cuda_wins_over_amd_without_flag(self):
        self.assertEqual(
            [e["mode"] for e in handler._resolve_test_spec(self.write_cuda_and_amd())],
            ["cuda"],
        )

    def test_hw_amd_picks_amd_over_cuda(self):
        entries = handler._resolve_test_spec(self.write_cuda_and_amd(), hw="amd")
        self.assertEqual(
            [(e["mode"], e["runs_on"]) for e in entries],
            [("amd", "linux-mi300-1gpu-sglang")],
        )

    def test_hw_amd_without_amd_registration_is_rejected(self):
        [entry] = handler._resolve_test_spec(self.write_both(), hw="amd")
        self.assertIn("has no `register_amd_ci(...)`", entry["error"])

    def test_hw_cuda_never_falls_back_to_amd(self):
        path = self.write_test(
            "registered/amd/test_amd.py",
            'register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")',
        )
        [entry] = handler._resolve_test_spec(path, hw="cuda")
        self.assertIn("registered for AMD, not CUDA or CPU", entry["error"])
        self.assertIn("drop `--hw cuda`", entry["error"])

    def test_hw_amd_rejects_multimodal(self):
        os.makedirs("python/sglang/multimodal_gen/test")
        Path("python/sglang/multimodal_gen/test/test_gen.py").write_text("")
        [entry] = handler._resolve_test_spec(
            "python/sglang/multimodal_gen/test/test_gen.py", hw="amd"
        )
        self.assertIn("`register_amd_ci(...)`; multimodal_gen", entry["error"])

    def test_pop_hw_flag(self):
        self.assertEqual(
            handler._pop_hw_flag(["a.py", "--hw", "npu", "-c"]), ("npu", ["a.py", "-c"])
        )
        self.assertEqual(handler._pop_hw_flag(["a.py", "--hw"]), ("", ["a.py"]))
        self.assertEqual(handler._pop_hw_flag(["a.py"]), (None, ["a.py"]))

    def test_unsupported_hw_is_rejected_before_dispatch(self):
        for hw, reply in (
            ("rocm", "⛔ `--hw` takes one of `cuda`, `npu`, `amd`."),
            ("", "⛔ `--hw` takes one of `cuda`, `npu`, `amd`."),
        ):
            pr = mock.MagicMock()
            with mock.patch.object(handler.requests, "post") as post:
                ok = handler.handle_rerun_test(
                    mock.MagicMock(),
                    pr,
                    mock.MagicMock(),
                    {"cooldown_interval_minutes": 0},
                    ["a.py"],
                    token="t",
                    hw=hw,
                )
            self.assertFalse(ok)
            pr.create_issue_comment.assert_called_once_with(reply)
            post.assert_not_called()


class TestDispatch(_TempRepo):
    def _run(self, specs):
        gh_repo = mock.MagicMock(full_name="o/r")
        gh_repo.get_workflows.return_value = [
            types.SimpleNamespace(name="Rerun Test", id=7)
        ]
        pr = mock.MagicMock()
        pr.head.repo.owner.login = gh_repo.owner.login
        pr.head.ref = "feature"
        posts = []

        def post(url, json, headers):
            posts.append(json)
            return types.SimpleNamespace(status_code=204, text="")

        with (
            mock.patch.object(handler, "_check_rebase_gate", return_value=(True, None)),
            mock.patch.object(
                handler, "find_workflow_run_url", return_value="https://run"
            ),
            mock.patch.object(handler.requests, "post", side_effect=post),
        ):
            ok = handler.handle_rerun_test(
                gh_repo,
                pr,
                mock.MagicMock(),
                {"cooldown_interval_minutes": 0},
                specs,
                token="t",
            )
        self.assertTrue(ok)
        return [p["inputs"] for p in posts]

    def test_suites_on_one_runner_dispatch_separately(self):
        a = self.write_test(
            "registered/npu/test_a.py",
            'register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")',
        )
        b = self.write_test(
            "registered/npu/test_b.py",
            'register_npu_ci(est_time=60, suite="base-b-test-2-npu-a3")',
        )
        c = self.write_test(
            "registered/npu/test_c.py",
            'register_npu_ci(est_time=60, suite="base-b-test-1-npu-a3")',
        )
        inputs = self._run([a, b, c])
        self.assertEqual(
            [(i["npu_suite"], i["test_command"]) for i in inputs],
            [("base-b-test-1-npu-a3", f"{a}\n{c}"), ("base-b-test-2-npu-a3", b)],
        )
        self.assertEqual(
            {
                k: inputs[0][k]
                for k in (
                    "mode",
                    "runs_on",
                    "npu_image",
                    "npu_sgl_router",
                    "npu_collect_nightly_tests",
                )
            },
            {
                "mode": "npu",
                "runs_on": "linux-aarch64-a3-2-",
                "npu_image": A3_PR_IMAGE,
                "npu_sgl_router": "false",
                "npu_collect_nightly_tests": "false",
            },
        )

    def test_amd_dispatch_inputs(self):
        path = self.write_test(
            "registered/amd/test_pd.py",
            'register_amd_ci(est_time=60, stage="extra-a", runner_config="2-gpu-large-amd")',
        )
        [inputs] = self._run([path])
        self.assertEqual(
            {
                k: inputs[k]
                for k in ("mode", "runs_on", "test_command", *handler._MODE_INPUTS)
            },
            {
                "mode": "amd",
                "runs_on": "linux-mi300-2gpu-sglang",
                "test_command": path,
                **{k: str(v).lower() for k, v in handler._MODE_INPUTS.items()},
                "amd_rocm_version": "rocm10",
                "amd_sgl_router": "true",
                "amd_exec_env": "",
            },
        )

    def test_cuda_dispatch_sends_empty_mode_inputs(self):
        path = self.write_test(
            "registered/core/test_cuda.py",
            'register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")',
        )
        [inputs] = self._run([path])
        self.assertEqual(inputs["mode"], "cuda")
        self.assertEqual(
            {k: inputs[k] for k in handler._MODE_INPUTS},
            {
                "npu_suite": "",
                "npu_image": "",
                "npu_sgl_router": "false",
                "npu_collect_nightly_tests": "false",
                "amd_rocm_version": "",
                "amd_sgl_router": "false",
                "amd_exec_env": "",
            },
        )


class TestRerunWorkflowAmd(unittest.TestCase):
    def test_fork_reruns_get_no_docker_hub_secrets(self):
        workflow = yaml.safe_load(
            (amd_suites.WORKFLOWS_DIR / "rerun-test.yml").read_text()
        )
        env = workflow["jobs"]["rerun-test-amd"]["env"]
        for name in ("DOCKERHUB_AMD_USERNAME", "DOCKERHUB_AMD_TOKEN"):
            self.assertEqual(
                env[name],
                f"${{{{ inputs.pr_head_sha == '' && secrets.{name} || '' }}}}",
            )


class TestAmdEstimates(_TempRepo):
    def test_estimate_matches_the_runner(self):
        path = self.write_test(
            "registered/amd/test_est.py",
            """
            register_amd_ci(est_time=100, stage="stage-b", runner_config="1-gpu-small-amd")
            register_amd_ci(est_time=900, suite="stage-c-test-large-8-gpu-amd-mi35x")
            """,
        )
        estimate = handler._partition.estimate_seconds
        self.assertEqual(
            estimate(path, Path("."), "amd", "linux-mi300-1gpu-sglang"), 100
        )
        self.assertEqual(estimate(path, Path("."), "amd", "linux-mi35x-gpu-8"), 900)


class TestNpuStageInputs(unittest.TestCase):
    def test_partitions_and_repo_relative_files(self):
        matrix = {
            "include": [
                {"partition": 1, "test_command": "registered/npu/a.py"},
                {
                    "partition": 2,
                    "test_command": "registered/npu/b.py\nregistered/npu/c.py",
                },
            ]
        }
        partitions, only_tests = handler._partition.npu_stage_inputs(matrix)
        self.assertEqual(json.dumps(partitions), '{"size": 2, "arr": [0, 1]}')
        self.assertEqual(
            only_tests,
            "test/registered/npu/a.py\ntest/registered/npu/b.py\n"
            "test/registered/npu/c.py",
        )


if __name__ == "__main__":
    unittest.main()

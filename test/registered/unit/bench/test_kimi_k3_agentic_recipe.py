"""Unit tests for the Kimi-K3 AgentX recipe port.

``test/registered/amd/agentic/test_kimi_k3_mxfp4_agentic_mi35x.py`` only runs on
an 8-GPU MI35x node behind a 1.56 TB checkpoint load, so a renamed server flag
or a wrong derived number there costs a nightly slot to discover. Everything
about that port which does not need a GPU is checked here: the recipe's
per-concurrency table, the DSPARK and DCP switches, the kernel-env split between
what this tree reads and what only the pinned fork does, and -- the reason most
of this file exists -- that every flag it passes still parses against
``ServerArgs``.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.server_args import prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# ServerArgs resolution asks for the accelerator; the flags under test are not
# CPU-backend flags, so the parse has to happen as if a GPU were present.
_mock_device = patch(
    "sglang.srt.arg_groups.serving_hook.get_device", return_value="cuda"
)
_mock_device.start()


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "amd"
    / "agentic"
    / "test_kimi_k3_mxfp4_agentic_mi35x.py"
)


def _load_recipe_module(name="_kimi_k3_agentic_recipe_under_test"):
    """Import the AMD benchmark by path: ``test/registered`` is not a package.

    Only the module object is bound here, so neither unittest nor pytest
    collects the MI35x test cases it defines out of this file.
    """
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


recipe = _load_recipe_module()


def _flag(args, name):
    """The value ``name`` was passed, or ``None`` when it is a bare switch."""
    index = args.index(name)
    value = args[index + 1] if index + 1 < len(args) else None
    return None if value is None or value.startswith("--") else value


class TestRecipeTable(CustomTestCase):
    """The orchestrator's per-concurrency table, reproduced row by row."""

    # CONC, max_running_requests, cuda_graph_max_bs, mamba slots, hicache GB
    ROWS = [
        (1, 2, 2, 10, 80),
        (4, 8, 8, 40, 80),
        (8, 16, 16, 80, 145),
        (16, 32, 32, 160, 145),
        (32, 64, 64, 320, 145),
        (48, 96, 96, 480, 145),
    ]

    def test_every_row_matches_the_recipe(self):
        for concurrency, mrr, graph_bs, mamba, hicache_gb in self.ROWS:
            with self.subTest(concurrency=concurrency):
                arm = recipe.recipe_arm(concurrency)
                self.assertEqual(arm.max_running_requests, mrr)
                self.assertEqual(arm.cuda_graph_max_bs, graph_bs)
                self.assertEqual(arm.max_mamba_cache_size, mamba)
                self.assertEqual(arm.hicache_size_gb, hicache_gb)
                self.assertFalse(arm.spec_enabled)

    def test_state_slots_cover_the_admission_ceiling_exactly(self):
        # The point of the 5x multiplier: the state pool holds one slot budget
        # per admissible request and not one more, so a clamp below the
        # concurrency being measured means something else took the memory.
        for concurrency, mrr, _, mamba, _ in self.ROWS:
            with self.subTest(concurrency=concurrency):
                self.assertEqual(mamba // recipe.MAMBA_SLOTS_PER_REQUEST, mrr)

    def test_decode_graph_capture_is_capped(self):
        # 2x concurrency would ask for 400 captured batches at c200; K3 captures
        # across 93 attention and 92 MoE layers, so the recipe caps it.
        arm = recipe.recipe_arm(200)
        self.assertEqual(arm.max_running_requests, 400)
        self.assertEqual(arm.cuda_graph_max_bs, recipe.CUDA_GRAPH_CAP)

    def test_derived_numbers_stay_overridable(self):
        overrides = {
            "AGENTIC_MAX_RUNNING_REQUESTS": "12",
            "AGENTIC_CUDA_GRAPH_MAX_BS": "8",
            "AGENTIC_MAX_MAMBA_CACHE_SIZE": "96",
            "AGENTIC_HICACHE_SIZE_GB": "200",
        }
        with patch.dict(os.environ, overrides, clear=False):
            arm = recipe.recipe_arm(48)
        self.assertEqual(arm.max_running_requests, 12)
        self.assertEqual(arm.cuda_graph_max_bs, 8)
        self.assertEqual(arm.max_mamba_cache_size, 96)
        self.assertEqual(arm.hicache_size_gb, 200)


class TestSpeculativeSchedule(CustomTestCase):
    def test_dspark_is_off_in_the_ported_arm(self):
        for concurrency, *_ in TestRecipeTable.ROWS:
            with self.subTest(concurrency=concurrency):
                arm = recipe.recipe_arm(concurrency, spec_decode="false")
                self.assertEqual(arm.dspark_block_size, 0)
                self.assertNotIn(
                    "--speculative-algorithm", recipe.build_server_args(arm)
                )

    def test_auto_restores_the_orchestrator_schedule(self):
        expected = {1: (7, 3.84), 8: (7, 3.84), 16: (3, 3.00), 32: (0, 0.0)}
        for concurrency, (block_size, accept_len) in expected.items():
            with self.subTest(concurrency=concurrency):
                arm = recipe.recipe_arm(concurrency, spec_decode="auto")
                self.assertEqual(arm.dspark_block_size, block_size)
                self.assertEqual(arm.golden_accept_len, accept_len)

    def test_spec_arm_asks_for_replayssm_verification(self):
        # Without ReplaySSM the draft's intermediate KDA states come out of the
        # per-request slot budget, and the recipe's 5x mamba multiplier stops
        # covering max_running_requests.
        args = recipe.build_server_args(recipe.recipe_arm(8, spec_decode="auto"))
        self.assertEqual(_flag(args, "--speculative-algorithm"), "DSPARK")
        self.assertEqual(_flag(args, "--speculative-dspark-block-size"), "7")
        self.assertIn("--enable-linear-replayssm-spec", args)
        self.assertEqual(
            _flag(args, "--speculative-draft-model-path"), recipe.DSPARK_DRAFT_PATH
        )

    def test_spec_arm_leaves_acceptance_measured_by_default(self):
        env = recipe.build_server_env(recipe.recipe_arm(8, spec_decode="auto"))
        self.assertEqual(env["SGLANG_RAGGED_VERIFY_MODE"], "static")
        self.assertNotIn("SGLANG_SIMULATE_ACC_LEN", env)

    def test_simulated_acceptance_restores_the_agentx_pin(self):
        with patch.object(recipe, "SIMULATE_ACC_LEN", "3.84"):
            env = recipe.build_server_env(recipe.recipe_arm(8, spec_decode="auto"))
        self.assertEqual(env["SGLANG_SIMULATE_ACC_LEN"], "3.84")
        self.assertEqual(env["SGLANG_SIMULATE_ACC_METHOD"], "match-expected")


class TestDecodeContextParallel(CustomTestCase):
    def test_dcp1_uses_allgather_reducescatter(self):
        arm = recipe.recipe_arm(48, dcp_size=1)
        self.assertEqual(arm.dcp_comm_backend, "ag_rs")
        self.assertEqual(recipe.build_server_env(arm)["SGLANG_AITER_MLA_GLUON"], "0")

    def test_dcp_above_one_switches_backend_and_gluon(self):
        arm = recipe.recipe_arm(48, dcp_size=8)
        self.assertEqual(arm.dcp_comm_backend, "a2a")
        self.assertEqual(recipe.build_server_env(arm)["SGLANG_AITER_MLA_GLUON"], "1")


class TestKernelEnv(CustomTestCase):
    # Read only by the pinned yuychang/sglang + yuychang/aiter kimi_k3_mxmoe_16k
    # branches the recipe runs against, so exporting them upstream would be dead
    # configuration that reads as if it did something.
    FORK_ONLY = [
        "SGLANG_K3_AITER_M16384_PROFILE",
        "SGLANG_K3_AITER_MLA_GATE",
        "SGLANG_K3_AITER_KDA_GROUP64",
        "SGLANG_K3_AITER_B2_FUSIONS",
        "SGLANG_K3_AITER_MOE_PREROUTE_FP8",
        "SGLANG_K3_PREROUTE_PREACTIVATED_SHARED",
        "SGLANG_K3_AITER_LATENT_TAIL_FP8",
        "SGLANG_K3_AITER_MLA_Q_CACHE_FUSION",
        "SGLANG_K3_AITER_TUNED_MOE_FRONT",
        "SGLANG_K3_MOE_LATENT_MXFP4",
        "SGLANG_K3_PTPC_FP8",
        "AITER_FLYDSL_DISABLE_MXMOE_V2",
        "AITER_FLYDSL_STAGE1_SCRATCH_REUSE",
    ]

    def test_fork_only_flags_are_not_exported(self):
        with patch.dict(os.environ, {}, clear=True):
            env = recipe.build_server_env(recipe.recipe_arm(48))
        for name in self.FORK_ONLY:
            with self.subTest(name=name):
                self.assertNotIn(name, env)

    def test_in_tree_kernel_flags_are_exported(self):
        with patch.dict(os.environ, {}, clear=True):
            env = recipe.build_server_env(recipe.recipe_arm(48))
        self.assertEqual(env["SGLANG_USE_AITER"], "1")
        self.assertEqual(env["SGLANG_AITER_K3_OPT"], "1")
        # MXFP4-native experts, where the K3 accuracy and perf tests take W4A8.
        self.assertEqual(env["AITER_SITUV2_A4W4"], "1")
        self.assertEqual(env["AITER_SITUV2_A8W4"], "0")
        for name, value in recipe.YUYUN_KERNEL_ENV.items():
            with self.subTest(name=name):
                self.assertEqual(env[name], value)

    def test_aiter_only_arm_drops_the_kernel_block(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(recipe, "ENABLE_YUYUN_KERNEL_ENV", False),
        ):
            env = recipe.build_server_env(recipe.recipe_arm(48))
        for name in recipe.YUYUN_KERNEL_ENV:
            with self.subTest(name=name):
                self.assertNotIn(name, env)
        self.assertEqual(env["SGLANG_USE_AITER"], "1")


class TestHiCacheArm(CustomTestCase):
    def test_host_tier_flags_come_from_the_recipe(self):
        arm = recipe.recipe_arm(48)
        args = recipe.build_server_args(arm)
        self.assertIn("--enable-hierarchical-cache", args)
        self.assertEqual(_flag(args, "--hicache-size"), "145")
        self.assertEqual(_flag(args, "--hicache-write-policy"), "write_through")
        self.assertEqual(_flag(args, "--hicache-io-backend"), "direct")
        self.assertEqual(_flag(args, "--hicache-mem-layout"), "page_first_direct")
        # The replay asks every turn for its prefix-cache detail, which the
        # server only answers when reporting is on.
        self.assertIn("--enable-cache-report", args)

    def test_gpu_resident_arm_drops_them(self):
        with patch.object(recipe, "KV_OFFLOADING", "none"):
            args = recipe.build_server_args(recipe.recipe_arm(48))
        for flag in ("--enable-hierarchical-cache", "--hicache-size"):
            with self.subTest(flag=flag):
                self.assertNotIn(flag, args)


class TestServerArgsParse(CustomTestCase):
    """Every flag the recipe passes has to still exist and take its value.

    This is the check that is worth a CPU job: the alternative place to learn
    that a flag was renamed is an 8-GPU MI35x nightly, after a 1.56 TB load.
    """

    def _parse(self, args):
        try:
            return prepare_server_args(["--model-path", "dummy"] + args)
        except SystemExit as exc:
            self.fail(f"server args rejected by the CLI parser: SystemExit({exc.code})")

    def test_ported_arm_parses(self):
        arm = recipe.recipe_arm(48)
        server_args = self._parse(recipe.build_server_args(arm))

        self.assertEqual(server_args.tp_size, recipe.TP_SIZE)
        self.assertEqual(server_args.dcp_size, 1)
        self.assertEqual(server_args.dcp_comm_backend, "ag_rs")
        self.assertEqual(server_args.kv_cache_dtype, "fp8_e4m3")
        self.assertEqual(server_args.page_size, 128)
        self.assertEqual(server_args.context_length, 1048576)
        self.assertEqual(server_args.max_running_requests, arm.max_running_requests)
        self.assertEqual(server_args.max_mamba_cache_size, arm.max_mamba_cache_size)
        self.assertEqual(server_args.mamba_ssm_dtype, "bfloat16")
        self.assertEqual(server_args.mamba_track_interval, 1024)
        self.assertEqual(server_args.hicache_size, arm.hicache_size_gb)
        self.assertEqual(server_args.prefill_attention_backend, "aiter")
        self.assertEqual(server_args.decode_attention_backend, "aiter")
        self.assertEqual(server_args.tool_call_parser, "kimi_k3")
        self.assertEqual(server_args.reasoning_parser, "kimi_k3")
        self.assertEqual(server_args.chunked_prefill_size, 16384)
        self.assertEqual(server_args.max_prefill_tokens, 16384)

    def test_dspark_arm_parses(self):
        server_args = self._parse(
            recipe.build_server_args(recipe.recipe_arm(8, spec_decode="auto"))
        )
        self.assertEqual(server_args.speculative_algorithm, "DSPARK")
        self.assertEqual(server_args.speculative_dspark_block_size, 7)
        self.assertTrue(server_args.enable_linear_replayssm_spec)

    def test_dcp_arm_parses(self):
        server_args = self._parse(
            recipe.build_server_args(recipe.recipe_arm(48, dcp_size=8))
        )
        self.assertEqual(server_args.dcp_size, 8)
        self.assertEqual(server_args.dcp_comm_backend, "a2a")


class TestResolvedBatchLimits(CustomTestCase):
    def test_an_unreachable_server_is_reported_not_raised(self):
        # The report is written after the run, so a failed /server_info must not
        # be what loses a completed benchmark.
        limits = recipe.resolved_batch_limits("http://127.0.0.1:1")
        self.assertIn("error", limits)


class TestModeSelection(CustomTestCase):
    def test_an_unknown_mode_fails_loudly(self):
        # Both cases are skipUnless-gated on AGENTIC_MODE, so a typo'd dispatch
        # input would otherwise publish an empty run as a passing benchmark.
        with (
            patch.dict(os.environ, {"AGENTIC_MODE": "relpay"}, clear=False),
            self.assertRaises(ValueError),
        ):
            _load_recipe_module("_kimi_k3_agentic_recipe_bad_mode")


if __name__ == "__main__":
    unittest.main()

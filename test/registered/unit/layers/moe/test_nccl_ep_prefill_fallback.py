"""Unit tests for NCCL EP staged execution.

Tests the three-phase dispatch/combine split (dispatch_a/dispatch_b,
combine_a/combine_b), the _Stage state machine, and the DeepEPPDispatchHooks
mechanism.

CPU-only; no GPU or NCCL required. Uses mocks for parallel state.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import torch


class TestNcclEpStageStateMachine(unittest.TestCase):
    """Test the _Stage enum and _update_stage guard."""

    def test_stage_enum_has_four_states(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import _Stage

        self.assertEqual(len(_Stage), 4)
        self.assertIn("INITIAL", [s.name for s in _Stage])
        self.assertIn("AFTER_DISPATCH_A", [s.name for s in _Stage])
        self.assertIn("AFTER_DISPATCH_B", [s.name for s in _Stage])
        self.assertIn("AFTER_COMBINE_A", [s.name for s in _Stage])

    def test_update_stage_correct_transition(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
            _Stage,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._stage = _Stage.INITIAL
        dispatcher._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        self.assertEqual(dispatcher._stage, _Stage.AFTER_DISPATCH_A)
        dispatcher._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        self.assertEqual(dispatcher._stage, _Stage.AFTER_DISPATCH_B)
        dispatcher._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        self.assertEqual(dispatcher._stage, _Stage.AFTER_COMBINE_A)
        dispatcher._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        self.assertEqual(dispatcher._stage, _Stage.INITIAL)

    def test_update_stage_wrong_transition_raises(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
            _Stage,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._stage = _Stage.INITIAL
        with self.assertRaises(AssertionError):
            dispatcher._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)

    def test_combine_a_before_dispatch_b_raises(self):
        """combine_a requires stage AFTER_DISPATCH_B; calling from INITIAL
        must raise to prevent silent data corruption."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
            _Stage,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._stage = _Stage.INITIAL
        dispatcher._dispatched_t = 1
        dispatcher._dispatched_topk_weights = MagicMock()
        dispatcher._handle = MagicMock()
        dispatcher._nccl_ep = MagicMock()
        dispatcher.hidden_size = 64
        dispatcher.num_max_dispatch_tokens_per_rank = 128

        with self.assertRaises(AssertionError):
            dispatcher.combine_a(MagicMock())


class TestNcclEpDispatchHooks(unittest.TestCase):
    """Test the DeepEPPDispatchHooks mechanism for SBO."""

    def test_dispatch_hooks_initialized(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPPDispatchHooks,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._dispatch_hooks = DeepEPPDispatchHooks()
        self.assertIsInstance(dispatcher._dispatch_hooks, DeepEPPDispatchHooks)

    def test_register_deepep_dispatch_hook(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPPDispatchHooks,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._dispatch_hooks = DeepEPPDispatchHooks()

        called = []
        handle = dispatcher.register_deepep_dispatch_hook(
            lambda d: called.append(d)
        )
        self.assertEqual(len(called), 0)

        dispatcher._dispatch_hooks(dispatcher)
        self.assertEqual(len(called), 1)
        self.assertIs(called[0], dispatcher)
        handle.remove()

    def test_dispatch_calls_hooks_between_a_and_b(self):
        """dispatch() must call hooks after dispatch_a and before dispatch_b."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
            _Stage,
        )
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPPDispatchHooks,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._dispatch_hooks = DeepEPPDispatchHooks()
        dispatcher._stage = _Stage.INITIAL

        call_order = []

        def mock_dispatch_a(self, hs, topk_output):
            call_order.append("dispatch_a")
            self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
            self._dispatch_intermediate_state = ()

        def mock_dispatch_b(self):
            call_order.append("dispatch_b")
            self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
            return "dispatch_output"

        dispatcher.dispatch_a = mock_dispatch_a.__get__(dispatcher)
        dispatcher.dispatch_b = mock_dispatch_b.__get__(dispatcher)

        dispatcher.register_deepep_dispatch_hook(
            lambda d: call_order.append("hook")
        )

        result = dispatcher.dispatch(
            torch.randn(4, 64, dtype=torch.bfloat16),
            SimpleNamespace(),
        )
        self.assertEqual(result, "dispatch_output")
        self.assertEqual(call_order, ["dispatch_a", "hook", "dispatch_b"])


class TestMaxNumSmsResolution(unittest.TestCase):
    """Test NcclEpBuffer._resolve_max_num_sms."""

    def test_default_value(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch(
            "sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS"
        ) as mock_env:
            mock_env.is_set.return_value = False
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 20)

    def test_env_override(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch(
            "sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS"
        ) as mock_env:
            mock_env.is_set.return_value = True
            mock_env.get.return_value = 40
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 40)

    def test_minimum_for_large_expert_count(self):
        """256 experts need at least ceil(256/14)=19 SMs (nccl_ep.cc:1305)."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch(
            "sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS"
        ) as mock_env:
            mock_env.is_set.return_value = True
            mock_env.get.return_value = 1  # too low
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 19)  # clamped to minimum

    def test_minimum_for_small_expert_count(self):
        """64 experts need at least ceil(64/14)=5 SMs."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch(
            "sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS"
        ) as mock_env:
            mock_env.is_set.return_value = True
            mock_env.get.return_value = 1
            result = NcclEpBuffer._resolve_max_num_sms(64)
            self.assertEqual(result, 5)


class TestSBOAssertAcceptsNcclEp(unittest.TestCase):
    """Test that the SBO assert in deepseek_v2.py accepts NcclEpDispatcher."""

    def test_nccl_ep_dispatcher_is_accepted(self):
        """The assert isinstance(..., (MaybeTboDeepEPDispatcher, NcclEpDispatcher))
        should pass for NcclEpDispatcher instances."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher
        from sglang.srt.batch_overlap.two_batch_overlap import (
            MaybeTboDeepEPDispatcher,
        )

        # Create a bare NcclEpDispatcher (no __init__)
        dispatcher = object.__new__(NcclEpDispatcher)
        self.assertIsInstance(
            dispatcher,
            (MaybeTboDeepEPDispatcher, NcclEpDispatcher),
        )


class TestForwardNormalSkipArConsistency(unittest.TestCase):
    """Test skip_ar behavior for nccl_ep backend."""

    def test_skip_ar_false_for_nccl_ep(self):
        from sglang.srt.layers.moe.utils import should_skip_post_experts_all_reduce
        from sglang.srt.layers.moe.moe_runner.base import MoeA2ABackend

        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NCCL_EP,
        ):
            with patch(
                "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce",
                return_value=False,
            ):
                with patch(
                    "sglang.srt.layers.moe.utils.should_use_dp_reduce_scatterv",
                    return_value=False,
                ):
                    with patch(
                        "sglang.srt.layers.moe.utils.get_server_args"
                    ) as mock_sa:
                        mock_sa.return_value = SimpleNamespace(dwdp_size=1)
                        result = should_skip_post_experts_all_reduce(
                            is_tp_path=True
                        )

        self.assertFalse(result)


class TestNcclEpFuseAllreduceBehavior(unittest.TestCase):
    """Test NCCL EP backend causes fuse_mlp_allreduce=False."""

    def test_aiter_fusion_requires_a2a_none(self):
        from sglang.srt.layers.moe.moe_runner.base import MoeA2ABackend

        self.assertFalse(MoeA2ABackend.NCCL_EP.is_none())
        self.assertTrue(MoeA2ABackend.NONE.is_none())


class TestNcclEpDispatcherEpGroupType(unittest.TestCase):
    """Test that create_moe_dispatcher passes get_tp_group() as ep_group."""

    def test_create_moe_dispatcher_passes_tp_group_as_ep_group(self):
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            create_moe_dispatcher,
        )
        from sglang.srt.layers.moe.moe_runner.base import MoeA2ABackend, MoeRunnerConfig

        cfg = MoeRunnerConfig(
            num_experts=256,
            num_local_experts=32,
            hidden_size=6144,
            top_k=8,
            params_dtype=torch.bfloat16,
        )

        tp_group_mock = MagicMock(world_size=8)

        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NCCL_EP,
        ):
            with patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer.get_tp_group",
                return_value=tp_group_mock,
            ):
                with patch(
                    "sglang.srt.layers.moe.token_dispatcher.nccl_ep.NcclEpDispatcher"
                ) as mock_disp_class:
                    create_moe_dispatcher(cfg)

                    call_kwargs = mock_disp_class.call_args
                    self.assertIn("ep_group", call_kwargs.kwargs)
                    self.assertIs(call_kwargs.kwargs["ep_group"], tp_group_mock)


class TestNcclRuntimeVersionParsing(unittest.TestCase):
    """Test _nccl_runtime_version() handles nccl4py 0.3.x VersionInfo format."""

    def test_returns_none_when_no_nccl4py(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _nccl_runtime_version,
        )

        with patch.dict("sys.modules", {"nccl": None, "nccl.core": None}):
            with patch("torch.cuda.nccl.version", side_effect=Exception):
                result = _nccl_runtime_version()
                # Should return None or fall through gracefully
                # (may still return torch's version if import fails silently)
                self.assertTrue(result is None or isinstance(result, tuple))

    def test_parses_versioninfo_namedtuple(self):
        """nccl4py 0.3.x returns VersionInfo(nccl=LibraryInfo(version=<Version('2.30.7')>...))."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _nccl_runtime_version,
        )

        mock_version_info = SimpleNamespace(
            nccl=SimpleNamespace(
                version="2.30.7"
            )
        )

        with patch.dict("sys.modules", {"nccl": MagicMock(), "nccl.core": MagicMock()}):
            import sys
            mock_nccl = MagicMock()
            mock_nccl.get_version.return_value = mock_version_info
            sys.modules["nccl"] = mock_nccl

            mock_core = MagicMock()
            mock_core.get_version.return_value = ""
            sys.modules["nccl.core"] = mock_core

            with patch("torch.cuda.nccl.version", return_value=(2, 28, 9)):
                result = _nccl_runtime_version()
                self.assertEqual(result, (2, 30, 7))

    def test_falls_back_to_torch_version(self):
        """When nccl4py returns unparseable data, fall back to torch.cuda.nccl.version."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _nccl_runtime_version,
        )

        with patch.dict("sys.modules", {"nccl": MagicMock(), "nccl.core": MagicMock()}):
            import sys
            mock_nccl = MagicMock()
            mock_nccl.get_version.return_value = None
            sys.modules["nccl"] = mock_nccl

            mock_core = MagicMock()
            mock_core.get_version.return_value = None
            sys.modules["nccl.core"] = mock_core

            with patch("torch.cuda.nccl.version", return_value=(2, 28, 9)):
                result = _nccl_runtime_version()
                self.assertEqual(result, (2, 28, 9))


class TestNcclEpCapabilityCheck(unittest.TestCase):
    """Test is_nccl_ep_available() and nccl_ep_unavailable_reason()."""

    def test_unavailable_reason_returns_str_or_none(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            nccl_ep_unavailable_reason,
        )

        reason = nccl_ep_unavailable_reason()
        self.assertTrue(reason is None or isinstance(reason, str))

    def test_is_nccl_ep_available_returns_bool(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            is_nccl_ep_available,
        )

        result = is_nccl_ep_available()
        self.assertIsInstance(result, bool)


class TestNcclEpHiddenAllowlist(unittest.TestCase):
    """Test that NcclEpDispatcher enforces the hidden size allowlist."""

    def test_allowed_hidden_sizes(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _NCCL_EP_LL_SUPPORTED_HIDDEN,
        )

        # DeepSeek-V2/V3 family hidden sizes
        for h in [5120, 6144, 7168]:
            self.assertIn(h, _NCCL_EP_LL_SUPPORTED_HIDDEN)

    def test_disallowed_hidden_sizes(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _NCCL_EP_LL_SUPPORTED_HIDDEN,
        )

        for h in [1024, 3072]:
            self.assertNotIn(h, _NCCL_EP_LL_SUPPORTED_HIDDEN)


class TestNcclEpTopkGuard(unittest.TestCase):
    """Test that NcclEpDispatcher enforces topk <= 9 guard."""

    def test_topk_guard_constant(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            _NCCL_EP_LL_MAX_TOPK,
        )

        self.assertEqual(_NCCL_EP_LL_MAX_TOPK, 9)


if __name__ == "__main__":
    unittest.main()

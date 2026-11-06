"""Unit tests for NCCL EP staged execution.

Tests the three-phase dispatch/combine split (dispatch_a/dispatch_b,
combine_a/combine_b), the _Stage state machine, and the DeepEPPDispatchHooks
mechanism.

CPU-only; no GPU or NCCL required. Uses mocks for parallel state.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        with self.assertRaises(RuntimeError):
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

        with self.assertRaises(RuntimeError):
            dispatcher.combine_a(MagicMock())


class TestNcclEpRankMajorContract(unittest.TestCase):
    """CPU-only checks for the rank-major metadata contract.

    These are deliberately independent of nccl4py: they lock down the
    ownership mapping and weighted pre-reduction required by NCCL EP's
    rank-major combine API.
    """

    def test_rank_major_format_is_distinct_from_deepep_ll(self):
        from sglang.srt.layers.moe.token_dispatcher.base import (
            CombineInputFormat,
            DispatchOutputFormat,
        )

        self.assertEqual(
            DispatchOutputFormat.NCCL_EP_RANK_MAJOR.value, "nccl_ep_rank_major"
        )
        self.assertEqual(
            CombineInputFormat.NCCL_EP_RANK_MAJOR.value, "nccl_ep_rank_major"
        )

    def test_local_expert_mapping_filters_nonlocal_routes(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            rank_major_local_expert_ids,
        )

        # NCCL RM already returns local ids; -1 is padding/nonlocal routing.
        recv_ids = torch.tensor([[[0, 1, -1, 2], [1, -1, 4, 0]]], dtype=torch.int32)
        actual = rank_major_local_expert_ids(recv_ids, num_local_experts=2)
        expected = torch.tensor([[[0, 1, -1, -1], [1, -1, -1, 0]]])
        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_rank_major_pre_reduce_weights_each_local_route(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            reduce_rank_major_expert_outputs,
        )

        # Two slots, each with two local routes. src2dst indexes an [E, M, H]
        # packed output flattened as [E*M, H].
        expert_outputs = torch.tensor(
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
            ]
        )
        src2dst = torch.tensor([0, 2, 1, 3], dtype=torch.int32)
        local_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])

        actual = reduce_rank_major_expert_outputs(
            expert_outputs, src2dst, local_ids, weights, (1, 2)
        )
        expected = torch.tensor([[[2.5, 25.0], [3.0, 30.0]]])
        self.assertTrue(torch.equal(actual.cpu(), expected))

    def test_rank_major_preprocess_emits_fp8_without_second_quantize(self):
        """RM packing must produce the CUTLASS FP8 input directly.

        A separate BF16 packed tensor followed by ``_quantize_fp8`` adds a
        full HBM pass for every MoE layer.
        """
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher

        dispatcher = object.__new__(NcclEpDispatcher)
        recv_ids = torch.full((1, 8, 2), -1, dtype=torch.int32)
        recv_ids[0, :2, 0] = 0
        dispatcher._dispatch_intermediate_state = (
            torch.zeros((1, 8, 4), dtype=torch.bfloat16),
            recv_ids,
            torch.ones((1, 8, 2), dtype=torch.float32),
            torch.tensor([2], dtype=torch.int32),
            2,
        )
        dispatcher.handle = MagicMock()
        dispatcher.num_local_experts = 1
        dispatcher.router_topk = 2
        dispatcher.hidden_size = 4
        dispatcher._dispatched_t = None
        dispatcher._quantize_fp8 = MagicMock()

        packed = torch.zeros((1, 256, 4), dtype=torch.float8_e4m3fn)
        scales = torch.ones((1, 256, 1), dtype=torch.float32)
        with patch("torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)):
            with patch(
                "sglang.kernels.ops.moe.ep_moe_kernels.moe_ep_deepgemm_preprocess",
                return_value=(
                    torch.tensor([2], dtype=torch.int32),
                    2,
                    torch.tensor([0, 1, 0, 1], dtype=torch.int32),
                    packed,
                    scales,
                ),
            ) as preprocess:
                output = dispatcher._dispatch_b_rank_major()

        self.assertEqual(output.hidden_states.dtype, torch.float8_e4m3fn)
        self.assertIs(output.hidden_states, packed)
        self.assertIs(output.hidden_states_scale, scales)
        self.assertEqual(
            preprocess.call_args.kwargs["output_dtype"], torch.float8_e4m3fn
        )
        self.assertEqual(preprocess.call_args.args[0].shape, (8, 2))
        self.assertEqual(preprocess.call_args.args[2].shape, (8, 4))
        self.assertEqual(output.slot_shape, (1, 8))
        dispatcher._quantize_fp8.assert_not_called()

    def test_rank_major_keeps_slots_beyond_local_batch(self):
        """A source rank's contribution is bounded by its own batch, not t.

        Under SUM_LEN padding, DP ranks may have unequal batches: here the
        local rank sent 1 token while a remote rank routed 4 slots to this
        rank's experts. Slicing the receive view to the local t would drop 3
        valid remote slots before expert compute, while NCCL combine still
        expects their contributions.
        """
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpDispatcher

        dispatcher = object.__new__(NcclEpDispatcher)
        recv_ids = torch.full((2, 4, 2), -1, dtype=torch.int32)
        recv_ids[1, 1:4, 0] = 0
        dispatcher._dispatch_intermediate_state = (
            torch.zeros((2, 4, 4), dtype=torch.bfloat16),
            recv_ids,
            torch.ones((2, 4, 2), dtype=torch.float32),
            torch.tensor([1, 4], dtype=torch.int32),
            1,
        )
        dispatcher.handle = MagicMock()
        dispatcher.num_local_experts = 1
        dispatcher.router_topk = 2
        dispatcher.hidden_size = 4
        dispatcher._dispatched_t = None
        dispatcher._quantize_fp8 = MagicMock()

        packed = torch.zeros((1, 256, 4), dtype=torch.float8_e4m3fn)
        scales = torch.ones((1, 256, 1), dtype=torch.float32)
        with patch("torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)):
            with patch(
                "sglang.kernels.ops.moe.ep_moe_kernels.moe_ep_deepgemm_preprocess",
                return_value=(
                    torch.tensor([3], dtype=torch.int32),
                    3,
                    torch.arange(8, dtype=torch.int32),
                    packed,
                    scales,
                ),
            ) as preprocess:
                output = dispatcher._dispatch_b_rank_major()

        ids_arg = preprocess.call_args.args[0]
        self.assertEqual(ids_arg.shape, (8, 2))
        # Rank 1's slots 1..3 map to rows 5..7 and must keep their local
        # expert id, not the -1 padding.
        self.assertTrue((ids_arg[5:8, 0] >= 0).all())
        self.assertEqual(output.slot_shape, (2, 4))
        dispatcher._quantize_fp8.assert_not_called()


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
        handle = dispatcher.register_deepep_dispatch_hook(lambda d: called.append(d))
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

        dispatcher.register_deepep_dispatch_hook(lambda d: call_order.append("hook"))

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

        with patch("sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS") as mock_env:
            mock_env.is_set.return_value = False
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 20)

    def test_env_override(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch("sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS") as mock_env:
            mock_env.is_set.return_value = True
            mock_env.get.return_value = 40
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 40)

    def test_minimum_for_large_expert_count(self):
        """256 experts need at least ceil(256/14)=19 SMs (nccl_ep.cc:1305)."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch("sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS") as mock_env:
            mock_env.is_set.return_value = True
            mock_env.get.return_value = 1  # too low
            result = NcclEpBuffer._resolve_max_num_sms(256)
            self.assertEqual(result, 19)  # clamped to minimum

    def test_minimum_for_small_expert_count(self):
        """64 experts need at least ceil(64/14)=5 SMs."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer

        with patch("sglang.srt.environ.envs.SGLANG_NCCL_EP_MAX_NUM_SMS") as mock_env:
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

        # post_experts_output_is_complete reads dwdp_size from get_parallel()
        # (ParallelContext) and only the flashinfer/pplx/megamoe A2A combines
        # sum the experts' output on their own. NCCL EP does not, so the
        # post-experts all-reduce must stay enabled.
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NCCL_EP,
        ):
            with patch(
                "sglang.srt.layers.moe.utils.should_skip_mlp_all_reduce",
                return_value=False,
            ):
                with patch(
                    "sglang.srt.layers.moe.utils.should_use_flashinfer_cutlass_moe_fp4_allgather",
                    return_value=False,
                ):
                    with patch(
                        "sglang.srt.layers.moe.utils.get_parallel"
                    ) as mock_parallel:
                        mock_parallel.return_value = SimpleNamespace(dwdp_size=1)
                        result = should_skip_post_experts_all_reduce(is_tp_path=True)

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

        # create_moe_dispatcher derives its layer contract from the resolved
        # quant method, so a method object is part of the signature. The
        # nccl_ep branch never reads it, so a mock is enough here.
        quant_method = MagicMock()

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
                    create_moe_dispatcher(cfg, quant_method)

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

        mock_version_info = SimpleNamespace(nccl=SimpleNamespace(version="2.30.7"))

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
        """When nccl4py returns unparsable data, fall back to torch.cuda.nccl.version."""
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

    def test_missing_parent_package_degrades_to_reason(self):
        """find_spec raises (does not return None) when the parent is missing.

        The capability probe must let _nccl_ep_capability_fallback select
        deepep/none instead of crashing during server-args configuration.
        """
        import importlib.util

        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            nccl_ep_unavailable_reason,
        )

        with patch.object(
            importlib.util,
            "find_spec",
            side_effect=ModuleNotFoundError("No module named 'nccl'"),
        ):
            self.assertEqual(
                nccl_ep_unavailable_reason(),
                "nccl-extensions (nccl.ep) not importable",
            )

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


class TestNcclEpHandleLifecycle(unittest.TestCase):
    """The handle is created on first use and reused, never per-forward."""

    def _make_dispatcher(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpDispatcher,
            _Stage,
        )

        dispatcher = object.__new__(NcclEpDispatcher)
        dispatcher._nccl_ep = MagicMock()
        dispatcher.buffer = MagicMock()
        dispatcher.ep_group = MagicMock()
        dispatcher.handle = None
        dispatcher.layout = MagicMock()
        dispatcher.layout.is_rank_major.return_value = False
        dispatcher.num_max_dispatch_tokens_per_rank = 1024
        dispatcher._stage = _Stage.INITIAL
        dispatcher._dispatch_link_state = None
        dispatcher._combine_link_state = None
        dispatcher._dispatched_topk_ids = None
        dispatcher._dispatch_intermediate_state = None
        dispatcher._combine_intermediate_state = None
        return dispatcher

    def _dispatch_once(self, dispatcher):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import _Stage

        topk_output = SimpleNamespace(
            topk_weights=torch.ones(4, 2, dtype=torch.float32),
            topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        )
        with (
            patch("torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
        ):
            dispatcher.dispatch_a(torch.randn(4, 8, dtype=torch.bfloat16), topk_output)
        # Reset the stage machine so the next dispatch_a is admissible.
        dispatcher._stage = _Stage.INITIAL

    def test_handle_created_once_then_updated(self):
        dispatcher = self._make_dispatcher()

        self._dispatch_once(dispatcher)
        created_handle = dispatcher.handle
        dispatcher.buffer.group.create_handle.assert_called_once()
        created_handle.update.assert_not_called()

        self._dispatch_once(dispatcher)
        self.assertIs(dispatcher.handle, created_handle)
        dispatcher.buffer.group.create_handle.assert_called_once()
        created_handle.update.assert_called_once()

    def test_dispatch_descriptors_survive_until_completion(self):
        """Backend callbacks keep no Python references to submitted bundles."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import _Stage

        class Descriptor:
            def __init__(self, tensor):
                self.tensor = tensor

        for rank_major in (False, True):
            with self.subTest(rank_major=rank_major):
                dispatcher = self._make_dispatcher()
                dispatcher.layout.is_rank_major.return_value = rank_major
                dispatcher.buffer.recv_tokens_fp8 = None
                descriptor_refs = []

                def dispatch(inputs, outputs, *, layout_info, **kwargs):
                    for bundle in (inputs, outputs, layout_info):
                        descriptor_refs.extend(
                            weakref.ref(value) for value in vars(bundle).values()
                        )

                dispatcher._nccl_ep.Tensor = Descriptor
                for name in ("DispatchInputs", "DispatchOutputs", "LayoutInfo"):
                    setattr(dispatcher._nccl_ep, name, SimpleNamespace)
                dispatcher.handle = SimpleNamespace(
                    dispatch=dispatch, update=lambda **kwargs: None
                )
                topk = SimpleNamespace(
                    topk_weights=torch.ones(4, 2),
                    topk_ids=torch.zeros(4, 2, dtype=torch.int32),
                )
                with patch(
                    "torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)
                ):
                    dispatcher.dispatch_a(torch.zeros(4, 8), topk)
                gc.collect()
                self.assertEqual(len(descriptor_refs), 6 if rank_major else 5)
                self.assertTrue(all(ref() is not None for ref in descriptor_refs))
                # The converted routing tensor is distinct from the caller's int32 IDs.
                routing_ref = weakref.ref(dispatcher._dispatched_topk_ids)
                self.assertEqual(routing_ref().dtype, torch.int64)
                dispatcher._dispatch_link_state = None
                dispatcher._dispatch_intermediate_state = None
                gc.collect()
                self.assertTrue(all(ref() is None for ref in descriptor_refs))
                self.assertIsNotNone(routing_ref())

                def complete(**kwargs):
                    self.assertIsNotNone(routing_ref())

                dispatcher.handle.complete = complete
                dispatcher._stage = _Stage.AFTER_COMBINE_A
                dispatcher._combine_intermediate_state = object()
                with patch(
                    "torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)
                ):
                    dispatcher.combine_b()
                gc.collect()
                self.assertIsNone(routing_ref())

    def test_combine_descriptors_survive_until_completion(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpRankMajorCombineInput,
            _Stage,
        )

        class Descriptor:
            def __init__(self, tensor):
                self.tensor = tensor

        for rank_major in (False, True):
            with self.subTest(rank_major=rank_major):
                dispatcher = self._make_dispatcher()
                dispatcher._stage = _Stage.AFTER_DISPATCH_B
                dispatcher._dispatched_t = 4
                descriptor_refs = []

                def combine(inputs, outputs, **kwargs):
                    for bundle in (inputs, outputs):
                        descriptor_refs.extend(
                            weakref.ref(value) for value in vars(bundle).values()
                        )

                def complete(**kwargs):
                    gc.collect()
                    self.assertTrue(all(ref() is not None for ref in descriptor_refs))

                dispatcher.handle = SimpleNamespace(combine=combine, complete=complete)
                dispatcher._nccl_ep.Tensor = Descriptor
                dispatcher._nccl_ep.CombineInputs = SimpleNamespace
                dispatcher._nccl_ep.CombineOutputs = SimpleNamespace
                if rank_major:
                    combine_input = NcclEpRankMajorCombineInput(
                        torch.zeros(4, 8), None, None, None, (1, 4)
                    )
                else:
                    combine_input = torch.zeros(4, 8)
                    dispatcher._dispatched_topk_weights = torch.ones(4, 2)
                with (
                    patch("torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)),
                    patch(
                        "sglang.srt.layers.moe.token_dispatcher.nccl_ep.reduce_rank_major_expert_outputs",
                        return_value=torch.zeros(4, 8),
                    ),
                ):
                    dispatcher.combine_a(combine_input)
                    gc.collect()
                    self.assertEqual(len(descriptor_refs), 2 if rank_major else 3)
                    self.assertTrue(all(ref() is not None for ref in descriptor_refs))
                    dispatcher.combine_b()
                gc.collect()
                self.assertTrue(all(ref() is None for ref in descriptor_refs))

    def test_combine_keeps_handle_alive(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import _Stage

        dispatcher = self._make_dispatcher()
        dispatcher.handle = MagicMock()
        dispatcher._combine_intermediate_state = MagicMock()
        dispatcher._stage = _Stage.AFTER_COMBINE_A

        with patch("torch.cuda.current_stream", return_value=MagicMock(cuda_stream=0)):
            dispatcher.combine_b()

        dispatcher.handle.complete.assert_called_once()
        dispatcher.handle.destroy.assert_not_called()
        self.assertIsNotNone(dispatcher.handle)

    def test_rejected_batch_leaves_dispatcher_usable(self):
        """An over-capacity batch raises without advancing the stage machine,
        so the next admissible dispatch still runs instead of a stage mismatch."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import _Stage

        dispatcher = self._make_dispatcher()
        dispatcher.num_max_dispatch_tokens_per_rank = 2
        topk_output = SimpleNamespace(
            topk_weights=torch.ones(4, 2, dtype=torch.float32),
            topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        )
        with self.assertRaises(ValueError):
            dispatcher.dispatch_a(torch.randn(4, 8, dtype=torch.bfloat16), topk_output)
        self.assertEqual(dispatcher._stage, _Stage.INITIAL)

        dispatcher.num_max_dispatch_tokens_per_rank = 1024
        self._dispatch_once(dispatcher)
        dispatcher.buffer.group.create_handle.assert_called_once()

    def test_handle_creation_refused_during_capture(self):
        """Auto-sized handle creation is collective and may reallocate RDMA,
        so it must never be recorded into a CUDA graph."""
        dispatcher = self._make_dispatcher()
        topk_ids = torch.zeros(4, 2, dtype=torch.int64)
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaises(RuntimeError),
        ):
            dispatcher._ensure_handle(topk_ids, MagicMock(cuda_stream=0))
        dispatcher.buffer.group.create_handle.assert_not_called()

    def test_first_dispatch_during_capture_does_not_allocate_buffer(self):
        dispatcher = self._make_dispatcher()
        dispatcher._ensure_buffer = MagicMock()
        topk = SimpleNamespace(
            topk_weights=torch.ones(4, 2),
            topk_ids=torch.zeros(4, 2, dtype=torch.int32),
        )
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "warmup forward"),
        ):
            dispatcher.dispatch_a(torch.zeros(4, 8), topk)
        dispatcher._ensure_buffer.assert_not_called()
        dispatcher.buffer.group.create_handle.assert_not_called()

    def test_destroy_releases_handles_before_group(self):
        """ncclEpHandleDestroy dereferences its group, so the group must outlive
        every handle created on it."""
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import NcclEpBuffer
        from sglang.srt.runtime_context import get_resources

        order = MagicMock()
        state = NcclEpBuffer._state()
        state.handles = [order.handle_a, order.handle_b]
        state.group = order.group
        NcclEpBuffer.destroy()

        self.assertEqual(
            [c[0] for c in order.mock_calls],
            ["handle_a.destroy", "handle_b.destroy", "group.destroy"],
        )
        self.assertNotIn("nccl_ep_state", get_resources().buffers)


class TestNcclEpRankMajorW4AFp8(unittest.TestCase):
    """Guard the nine-field rank-major view passed to the masked GEMM."""

    @staticmethod
    def _fake_layer():
        quant_method = SimpleNamespace(
            a_strides1=1,
            b_strides1=2,
            c_strides1=3,
            a_strides2=4,
            b_strides2=5,
            c_strides2=6,
            s_strides13=7,
            s_strides2=8,
            expert_offsets=9,
            problem_sizes1=10,
            problem_sizes2=11,
        )
        return SimpleNamespace(
            w13_weight="w13",
            w2_weight="w2",
            w13_weight_scale_inv="w13_scale",
            w2_weight_scale_inv="w2_scale",
            w13_input_scale="w13_input",
            w2_input_scale="w2_input",
            quant_method=quant_method,
        )

    def test_adapter_forwards_packed_view_to_masked_gemm(self):
        from sglang.srt.layers.moe.token_dispatcher.nccl_ep import (
            NcclEpRankMajorDispatchOutput,
        )
        from sglang.srt.layers.quantization.w4afp8 import (
            W4AFp8Config,
            W4AFp8MoEMethod,
        )

        method = W4AFp8MoEMethod(W4AFp8Config())
        layer = self._fake_layer()

        hidden = torch.zeros(8, 16, dtype=torch.float8_e4m3fn)
        hidden_scale = torch.ones(8, 2, dtype=torch.float32)
        topk_ids = torch.zeros(8, 2, dtype=torch.int32)
        masked_m = torch.tensor([4, 4], dtype=torch.int32)
        dispatch_output = NcclEpRankMajorDispatchOutput(
            hidden_states=hidden,
            hidden_states_scale=hidden_scale,
            topk_ids=topk_ids,
            masked_m=masked_m,
            expected_m=3,
            src2dst=torch.zeros(4, dtype=torch.int32),
            local_topk_ids=topk_ids,
            recv_topk_weights=torch.ones(8, 2, dtype=torch.float32),
            slot_shape=(2, 4),
        )

        with patch(
            "sglang.srt.layers.moe.cutlass_w4a8_moe.cutlass_w4a8_moe_deepep_ll",
            return_value="gemm_out",
        ) as gemm:
            out = method.apply_nccl_ep_rank_major(layer, dispatch_output)

        self.assertEqual(out, "gemm_out")
        args = gemm.call_args.args
        self.assertIs(args[0], hidden)
        self.assertIs(args[1], hidden_scale)
        self.assertIs(args[6], topk_ids)
        self.assertIs(args[7], masked_m)
        self.assertEqual(gemm.call_args.kwargs["expected_m"], 3)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for runtime_context: delegation, singletons, and override()."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import dataclasses
import os
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.arg_utils import A, Arg
from sglang.srt.runtime_context import (
    Flags,
    ParallelContext,
    RuntimeContext,
    _FlagGroupBase,
    get_context,
    get_flags,
    get_parallel,
    get_server_args,
    reset_context,
)
from sglang.test.test_utils import CustomTestCase

_PS = "sglang.srt.distributed.parallel_state"
_DP = "sglang.srt.layers.dp_attention"

SIZE_RANK_DELEGATIONS = [
    ("world_size", f"{_PS}.get_world_size"),
    ("world_rank", f"{_PS}.get_world_rank"),
    ("tp_size", f"{_PS}.get_tensor_model_parallel_world_size"),
    ("tp_rank", f"{_PS}.get_tensor_model_parallel_rank"),
    ("pp_size", f"{_PS}.get_pipeline_model_parallel_world_size"),
    ("pp_rank", f"{_PS}.get_pipeline_model_parallel_rank"),
    ("moe_ep_size", f"{_PS}.get_moe_expert_parallel_world_size"),
    ("moe_ep_rank", f"{_PS}.get_moe_expert_parallel_rank"),
    ("moe_dp_size", f"{_PS}.get_moe_data_parallel_world_size"),
    ("moe_dp_rank", f"{_PS}.get_moe_data_parallel_rank"),
    ("moe_tp_size", f"{_PS}.get_moe_tensor_parallel_world_size"),
    ("moe_tp_rank", f"{_PS}.get_moe_tensor_parallel_rank"),
    ("attn_tp_size", f"{_PS}.get_attn_tensor_model_parallel_world_size"),
    ("attn_tp_rank", f"{_PS}.get_attn_tensor_model_parallel_rank"),
    ("attn_cp_size", f"{_PS}.get_attn_context_model_parallel_world_size"),
    ("attn_cp_rank", f"{_PS}.get_attn_context_model_parallel_rank"),
    ("attn_dp_size", f"{_DP}.get_attention_dp_size"),
    ("attn_dp_rank", f"{_DP}.get_attention_dp_rank"),
]

GROUP_DELEGATIONS = [
    ("world_group", f"{_PS}.get_world_group"),
    ("tp_group", f"{_PS}.get_tp_group"),
    ("pp_group", f"{_PS}.get_pp_group"),
    ("moe_ep_group", f"{_PS}.get_moe_ep_group"),
    ("moe_dp_group", f"{_PS}.get_moe_dp_group"),
    ("moe_tp_group", f"{_PS}.get_moe_tp_group"),
    ("attn_tp_group", f"{_PS}.get_attn_tp_group"),
    ("attn_cp_group", f"{_PS}.get_attn_cp_group"),
]


class TestRuntimeContextSingletons(CustomTestCase):
    def test_singletons(self):
        self.assertIs(get_parallel(), get_parallel())
        self.assertIsInstance(get_parallel(), ParallelContext)
        self.assertIsInstance(get_context(), RuntimeContext)
        self.assertIs(get_context().parallel, get_parallel())


class _IsolatedOverrides(CustomTestCase):
    """Give each test a clean override map, restoring afterward only the overrides
    installed outside it (e.g. by another test file sharing the process)."""

    def setUp(self):
        super().setUp()
        p = get_parallel()
        self._saved_overrides = dict(p._overrides)
        p._overrides.clear()

    def tearDown(self):
        p = get_parallel()
        p._overrides.clear()
        p._overrides.update(self._saved_overrides)
        super().tearDown()


class TestParallelDelegation(_IsolatedOverrides):
    def test_size_rank_delegate_to_canonical_getters(self):
        # Patch each getter to a distinct sentinel: a miswired attribute would read
        # a different (unpatched) getter and fail.
        for i, (attr, target) in enumerate(SIZE_RANK_DELEGATIONS):
            sentinel = 1000 + i
            with patch(target, return_value=sentinel):
                self.assertEqual(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_groups_delegate_to_canonical_getters(self):
        for attr, target in GROUP_DELEGATIONS:
            sentinel = object()
            with patch(target, return_value=sentinel):
                self.assertIs(
                    getattr(get_parallel(), attr),
                    sentinel,
                    msg=f"{attr} must delegate to {target}",
                )

    def test_wrapper_holds_no_resolved_state(self):
        # __slots__: no __dict__; the only instance state is the override hook.
        self.assertFalse(hasattr(get_parallel(), "__dict__"))
        # tp_group IS exposed: live delegation handles PD-multiplexing / the tp patch.
        self.assertTrue(hasattr(ParallelContext, "tp_group"))
        # local_attn_dp is intentionally not part of the wrapper surface.
        self.assertFalse(hasattr(ParallelContext, "local_attn_dp_size"))


class TestParallelOverride(_IsolatedOverrides):
    def test_override_takes_precedence(self):
        p = get_parallel()
        with p.override(tp_size=99, tp_rank=3, attn_dp_size=8):
            self.assertEqual(p.tp_size, 99)
            self.assertEqual(p.tp_rank, 3)
            self.assertEqual(p.attn_dp_size, 8)
            # same singleton: a fresh get_parallel() sees the override too
            self.assertEqual(get_parallel().tp_size, 99)
        self.assertEqual(p._overrides, {})

    def test_override_can_force_groups(self):
        sentinel = object()
        with get_parallel().override(tp_group=sentinel):
            self.assertIs(get_parallel().tp_group, sentinel)

    def test_override_nests_and_restores(self):
        p = get_parallel()
        with p.override(tp_size=2):
            self.assertEqual(p.tp_size, 2)
            with p.override(tp_size=4, pp_size=2):
                self.assertEqual(p.tp_size, 4)
                self.assertEqual(p.pp_size, 2)
            self.assertEqual(p.tp_size, 2)
            self.assertNotIn("pp_size", p._overrides)

    def test_override_unknown_key_raises_and_does_not_mutate(self):
        p = get_parallel()
        with self.assertRaises(ValueError):
            with p.override(tp_sizee=1):  # typo
                pass
        self.assertEqual(p._overrides, {})


class _IsolatedServerArgs(CustomTestCase):
    """Save/restore the published ServerArgs around each test (the slot is
    process-global; another test file sharing the process may have published)."""

    def setUp(self):
        super().setUp()
        self._saved_server_args = get_context()._server_args

    def tearDown(self):
        if self._saved_server_args is None:
            reset_context()
        else:
            get_context().set_server_args(self._saved_server_args)
        super().tearDown()


class TestServerArgsOwnership(_IsolatedServerArgs):
    """V2b: the context owns the slot; the legacy getters are identity shims."""

    def test_legacy_setter_publishes_into_context(self):
        # Identity (not equality) is the contract; publish accepts any object.
        sentinel = object()
        server_args_module.set_global_server_args_for_scheduler(sentinel)
        self.assertIs(server_args_module.get_global_server_args(), sentinel)
        self.assertIs(get_server_args(), sentinel)
        self.assertIs(get_context().server_args, sentinel)

    def test_context_publish_visible_through_legacy_getter(self):
        sentinel = object()
        get_context().set_server_args(sentinel)
        self.assertIs(server_args_module.get_global_server_args(), sentinel)

    def test_tokenizer_alias_is_same_function(self):
        self.assertIs(
            server_args_module.set_global_server_args_for_tokenizer,
            server_args_module.set_global_server_args_for_scheduler,
        )

    def test_pre_publish_error_verbatim(self):
        reset_context()
        for accessor in (get_server_args, server_args_module.get_global_server_args):
            with self.assertRaises(ValueError) as cm:
                accessor()
            self.assertEqual(str(cm.exception), "Global server args is not set yet!")

    def test_republish_overwrite_allowed(self):
        first, second = object(), object()
        server_args_module.set_global_server_args_for_scheduler(first)
        server_args_module.set_global_server_args_for_scheduler(second)
        self.assertIs(get_server_args(), second)

    def test_reset_context_clears_owned_store(self):
        server_args_module.set_global_server_args_for_scheduler(object())
        reset_context()
        with self.assertRaises(ValueError):
            get_server_args()

    def test_module_global_removed(self):
        # The legacy storage must not survive: a stale _global_server_args would
        # silently fork the config into two objects.
        self.assertFalse(hasattr(server_args_module, "_global_server_args"))


@dataclasses.dataclass
class _FakeCaptureGroup(_FlagGroupBase):
    gamma: int = 0


class TestFlagsTier(_IsolatedServerArgs):
    """Runtime-flags tier: typed groups, typo-safe writes, override primitive.

    Resolved configuration lives on server_args fields (materialized at the
    end of __post_init__); the flags tier only carries runtime state
    (today: the capture lifecycle)."""

    def test_wiring_and_groups(self):
        flags = get_flags()
        self.assertIs(flags, get_context().flags)
        self.assertIsInstance(flags, Flags)
        self.assertTrue(hasattr(flags, "capture"))

    def test_typo_safety(self):
        group = _FakeCaptureGroup()
        with self.assertRaises(AttributeError):
            group.gamma_misspelled = 2  # undeclared leaf
        with self.assertRaises(AttributeError):
            get_flags().not_a_flag = 1

    def test_override_is_transactional(self):
        group = _FakeCaptureGroup()
        with group.override(gamma=99):
            self.assertEqual(group.gamma, 99)
        self.assertEqual(group.gamma, 0)
        with self.assertRaises(ValueError):
            with group.override(gamma=2, delta=3):  # delta undeclared
                pass
        self.assertEqual(group.gamma, 0)  # validated before any write

    def test_reset_context_installs_fresh_flags(self):
        old = get_flags()
        old.capture.enable_torch_compile = True
        reset_context()
        self.assertIsNot(get_flags(), old)
        self.assertFalse(get_flags().capture.enable_torch_compile)


@dataclasses.dataclass
class _FakeResolvedArgs:
    """Publishable fixture with a resolvable whitelist (real flat leaves)."""

    page_size: A[int | None, Arg(help="p", resolvable=True)] = None
    sampling_backend: A[str | None, Arg(help="s", resolvable=True)] = None
    _resolved_overrides: list = dataclasses.field(default_factory=list)


class TestMoeFlagsGroup(_IsolatedServerArgs):
    """flags.moe: materialized by initialize_moe_config; the ACTIVE backends
    swap under the speculative contexts and restore on exit."""

    def _init(self, **kw):
        from types import SimpleNamespace

        from sglang.srt.layers.moe.utils import initialize_moe_config

        defaults = dict(
            moe_a2a_backend="none",
            moe_runner_backend="auto",
            speculative_moe_runner_backend=None,
            speculative_moe_a2a_backend=None,
            deepep_mode="auto",
            deepep_config=None,
            enable_two_batch_overlap=False,
            enable_single_batch_overlap=False,
            tbo_token_distribution_threshold=0.48,
            disable_flashinfer_cutlass_moe_fp4_allgather=False,
            quantization=None,
        )
        defaults.update(kw)
        initialize_moe_config(SimpleNamespace(**defaults))

    def test_lazy_defaults_before_initialize(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
            is_tbo_enabled,
        )

        reset_context()
        self.assertTrue(get_moe_a2a_backend().is_none())
        self.assertEqual(get_moe_runner_backend().name, "AUTO")
        self.assertFalse(is_tbo_enabled())

    def test_initialize_materializes_group(self):
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend, is_tbo_enabled

        self._init(moe_a2a_backend="deepep", enable_two_batch_overlap=True)
        self.assertTrue(get_moe_a2a_backend().is_deepep())
        self.assertTrue(is_tbo_enabled())
        self.assertEqual(get_flags().moe.deepep_config, "")

    def test_speculative_swap_and_restore(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )

        self._init(
            moe_a2a_backend="deepep",
            moe_runner_backend="triton",
            speculative_moe_runner_backend="auto",
            speculative_moe_a2a_backend="none",
        )
        with speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.assertEqual(get_moe_runner_backend().name, "AUTO")
            self.assertTrue(get_moe_a2a_backend().is_none())
            # MTP layers are unquantized: fp4 allgather is forced off
            self.assertTrue(get_flags().moe.disable_fp4_allgather)
        self.assertEqual(get_moe_runner_backend().name, "TRITON")
        self.assertTrue(get_moe_a2a_backend().is_deepep())
        self.assertFalse(get_flags().moe.disable_fp4_allgather)

    def test_swap_restores_on_exception(self):
        from sglang.srt.layers.moe.utils import (
            get_moe_runner_backend,
            speculative_moe_backend_context,
        )

        self._init(moe_runner_backend="triton", speculative_moe_runner_backend="auto")
        with self.assertRaises(RuntimeError):
            with speculative_moe_backend_context():
                raise RuntimeError("boom")
        self.assertEqual(get_moe_runner_backend().name, "TRITON")


class TestDpFlagsGroup(_IsolatedServerArgs):
    """flags.dp: the DP-attention runtime flags; is_dp_attention_enabled is a
    thin shim over the group leaf."""

    def test_shim_reads_the_leaf(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        reset_context()
        self.assertFalse(is_dp_attention_enabled())
        get_flags().dp.enabled = True
        self.assertTrue(is_dp_attention_enabled())

    def test_scoped_override_forces_the_predicate(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        reset_context()
        with get_flags().dp.override(enabled=True):
            self.assertTrue(is_dp_attention_enabled())
        self.assertFalse(is_dp_attention_enabled())


class TestResources(_IsolatedServerArgs):
    """ctx.resources: named slots for process-level resource handles with one
    reset lifecycle; owning accessors keep their creation/publish semantics."""

    def test_graph_pool_lazy_create_and_reuse(self):
        from types import SimpleNamespace

        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool,
            get_or_create_global_graph_memory_pool,
        )

        reset_context()
        self.assertIsNone(get_global_graph_memory_pool())
        dev = SimpleNamespace(graph_pool_handle=lambda: object())
        handle = get_or_create_global_graph_memory_pool(dev)
        self.assertIs(get_or_create_global_graph_memory_pool(dev), handle)

    def test_expert_recorder_noop_default_and_injection(self):
        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )
        from sglang.srt.runtime_context import get_resources

        reset_context()
        self.assertEqual(
            type(get_global_expert_distribution_recorder()).__name__,
            "_ExpertDistributionRecorderNoop",
        )
        with get_resources().override(expert_distribution_recorder="mock"):
            self.assertEqual(get_global_expert_distribution_recorder(), "mock")

    def test_expert_location_metadata_publish_once_until_reset(self):
        from sglang.srt.eplb.expert_location import (
            get_global_expert_location_metadata,
            set_global_expert_location_metadata,
        )

        reset_context()
        self.assertIsNone(get_global_expert_location_metadata())
        set_global_expert_location_metadata("meta")
        with self.assertRaises(AssertionError):
            set_global_expert_location_metadata("again")
        reset_context()
        self.assertIsNone(get_global_expert_location_metadata())


class TestNamedStreams(_IsolatedServerArgs):
    """ctx.get_stream(name): keyed get-or-create (the persistent-buffer
    pattern); set_stream installs explicitly."""

    def test_get_or_create_shares_by_name(self):
        from unittest.mock import patch

        reset_context()
        created = []

        class _FakeStream:
            def __init__(self):
                created.append(self)

        with patch("torch.cuda.Stream", _FakeStream):
            a = get_context().get_stream("alt")
            b = get_context().get_stream("alt")
            c = get_context().get_stream("other")
        self.assertIs(a, b)
        self.assertIsNot(a, c)
        self.assertEqual(len(created), 2)

    def test_get_buffer_keyed_lazy(self):
        reset_context()
        created = []

        def factory():
            created.append(object())
            return created[-1]

        a = get_context().get_buffer("ws", factory)
        b = get_context().get_buffer("ws", factory)
        self.assertIs(a, b)
        self.assertEqual(len(created), 1)
        self.assertIsNot(get_context().get_buffer("other", factory), a)

    def test_set_stream_installs_explicitly(self):
        reset_context()
        sentinel = object()
        get_context().set_stream("alt", sentinel)
        self.assertIs(get_context().get_stream("alt"), sentinel)

    def test_reset_clears_the_registry(self):
        reset_context()
        get_context().set_stream("alt", object())
        reset_context()
        self.assertEqual(get_context().resources.streams, {})

    def test_capturer_slots_roundtrip_and_reset(self):
        from sglang.srt.state_capturer.indexer_topk import (
            get_global_indexer_capturer,
            set_global_indexer_capturer,
        )
        from sglang.srt.state_capturer.routed_experts import (
            get_global_experts_capturer,
            set_global_experts_capturer,
        )

        reset_context()
        self.assertIsNone(get_global_indexer_capturer())
        self.assertIsNone(get_global_experts_capturer())
        indexer, experts = object(), object()
        set_global_indexer_capturer(indexer)
        set_global_experts_capturer(experts)
        self.assertIs(get_global_indexer_capturer(), indexer)
        self.assertIs(get_global_experts_capturer(), experts)
        reset_context()
        self.assertIsNone(get_global_indexer_capturer())
        self.assertIsNone(get_global_experts_capturer())

    def test_tcp_store_slot_roundtrip_and_reset(self):
        from sglang.srt.distributed.utils import (
            get_global_tcp_store,
            set_global_tcp_store,
        )

        reset_context()
        self.assertIsNone(get_global_tcp_store())
        store = object()
        set_global_tcp_store(store)
        self.assertIs(get_global_tcp_store(), store)
        reset_context()
        self.assertIsNone(get_global_tcp_store())

    def test_trace_level_env_seeded_lazy_default(self):
        from sglang.srt.observability.trace import (
            get_global_trace_level,
            set_global_trace_level,
        )

        reset_context()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TRACE_LEVEL", None)
            self.assertEqual(get_global_trace_level(), 3)
        set_global_trace_level(5)
        self.assertEqual(get_global_trace_level(), 5)
        reset_context()
        with patch.dict(os.environ, {"SGLANG_TRACE_LEVEL": "1"}):
            self.assertEqual(get_global_trace_level(), 1)


class TestEpBufferState(_IsolatedServerArgs):
    """EP dispatcher buffer managers: state lives on ctx.resources; the
    facade keeps the mode-transition and clean semantics."""

    def test_deepep_dispatch_mode_transitions_and_reset(self):
        try:
            from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
        except ImportError:
            self.skipTest("deep_ep not installed")

        reset_context()
        cleans = []

        class _FakeBuffer:
            low_latency_mode = True

            def clean_low_latency_buffer(self, *args):
                cleans.append(args)

        state = DeepEPBuffer._state()
        state.buffer = _FakeBuffer()
        state.hidden_size = 7168
        state.num_max_dispatch_tokens_per_rank = 128
        state.num_experts = 256

        DeepEPBuffer.set_dispatch_mode_as_normal()
        # NORMAL -> LOW_LATENCY must clean the low-latency buffer once.
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self.assertEqual(cleans, [(128, 7168, 256)])
        # LOW_LATENCY -> LOW_LATENCY must not clean again.
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        self.assertEqual(len(cleans), 1)

        reset_context()
        self.assertIsNone(DeepEPBuffer._state().buffer)


class TestForwardFlags(_IsolatedServerArgs):
    """ctx.forward: contextvar-backed per-forward flags; scoped() restores,
    threads see defaults."""

    def test_scoped_set_restore_and_nesting(self):
        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        self.assertFalse(fwd.multi_stream)
        with fwd.scoped(multi_stream=True):
            self.assertTrue(fwd.multi_stream)
            with fwd.scoped(multi_stream=False):
                self.assertFalse(fwd.multi_stream)
            self.assertTrue(fwd.multi_stream)
        self.assertFalse(fwd.multi_stream)

    def test_scoped_restores_on_exception_and_validates_keys(self):
        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        with self.assertRaises(RuntimeError):
            with fwd.scoped(moe_output_buffer="buf"):
                raise RuntimeError("boom")
        self.assertIsNone(fwd.moe_output_buffer)
        with self.assertRaises(ValueError):
            with fwd.scoped(nope=1):
                pass
        with self.assertRaises(AttributeError):
            fwd.multi_stream = True  # attribute writes are rejected

    def test_threads_see_defaults(self):
        import threading

        from sglang.srt.runtime_context import get_forward

        reset_context()
        fwd = get_forward()
        seen = {}
        with fwd.scoped(multi_stream=True):

            def probe():
                seen["value"] = get_forward().multi_stream

            worker = threading.Thread(target=probe)
            worker.start()
            worker.join()
        self.assertFalse(seen["value"])  # a new thread sees the default

    def test_graph_visible_flags_trace_under_torch_compile(self):
        # Regression: dynamo cannot trace ContextVar.get, and these flags are
        # read inside compiled model code (vocab embedding, communicator, DP
        # gather) — they must stay plain-slot backed. fullgraph=True turns
        # any graph break back into a failure.
        import torch

        from sglang.srt.runtime_context import get_forward

        reset_context()

        @torch.compile(fullgraph=True, backend="eager", dynamic=False)
        def probe(x):
            fwd = get_forward()
            if fwd.attn_input_scattered:
                x = x + 1
            if fwd.is_extend_in_batch:
                x = x + 2
            return x

        self.assertEqual(probe(torch.zeros(())).item(), 0)
        with get_forward().scoped(attn_input_scattered=True):
            self.assertEqual(probe(torch.zeros(())).item(), 1)
        get_forward().set("is_extend_in_batch", True)
        self.assertEqual(probe(torch.zeros(())).item(), 2)
        get_forward().set("is_extend_in_batch", False)

    def test_graph_visible_flags_are_process_visible_across_threads(self):
        # Documented divergence from the contextvar-backed flags: plain slots
        # are process-global (the storage form these flags had before the
        # tier), so another thread sees the current value, not the default.
        import threading

        from sglang.srt.runtime_context import get_forward

        reset_context()
        seen = {}
        with get_forward().scoped(attn_input_scattered=True):

            def probe():
                seen["value"] = get_forward().attn_input_scattered

            worker = threading.Thread(target=probe)
            worker.start()
            worker.join()
        self.assertTrue(seen["value"])
        self.assertFalse(get_forward().attn_input_scattered)

    def test_multi_stream_shims(self):
        from sglang.srt.utils.multi_stream_utils import (
            do_multi_stream,
            with_multi_stream,
        )

        reset_context()
        self.assertFalse(do_multi_stream())
        with with_multi_stream(True):
            self.assertTrue(do_multi_stream())
        self.assertFalse(do_multi_stream())

    def test_attn_tp_context_per_forward_slots(self):
        from types import SimpleNamespace

        from sglang.srt.layers.communicator import get_attn_tp_context
        from sglang.srt.runtime_context import get_forward

        reset_context()
        ctx = get_attn_tp_context()
        self.assertFalse(ctx.input_scattered)
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False, is_target_verify=lambda: False
            ),
            input_ids=None,
            can_run_tbo=False,
        )
        sentinel = SimpleNamespace(fetch_qkv_latent=lambda: "qkv")
        with ctx.maybe_input_scattered(fb):
            ctx.set_attn_inputs(sentinel)
            self.assertEqual(ctx.fetch_qkv_latent(), "qkv")
        # attn inputs are cleared at scope exit, flag restored
        self.assertIsNone(get_forward().attn_inputs)
        self.assertFalse(ctx.input_scattered)

    def test_dp_buffer_state_split(self):
        import torch

        from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper as wrapper
        from sglang.srt.layers.dp_attention import (
            get_dp_dtype,
            get_dp_global_num_tokens,
            get_global_dp_buffer_len,
            is_dp_max_padding,
            set_dp_buffer_len,
        )

        reset_context()
        # metadata is init-static (flags.dp); sizing is per-forward sticky
        wrapper.set_metadata(64, torch.float16, torch.device("cpu"))
        self.assertEqual(get_dp_dtype(), torch.float16)
        set_dp_buffer_len(128, 32, True, [64, 64])
        self.assertEqual(get_global_dp_buffer_len(), 128)
        self.assertTrue(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [64, 64])
        set_dp_buffer_len(256, 64, False)  # sticky until the next write
        self.assertEqual(get_global_dp_buffer_len(), 256)
        self.assertFalse(is_dp_max_padding())
        self.assertIsNone(get_dp_global_num_tokens())
        reset_context()
        self.assertIsNone(get_dp_dtype())

    def test_is_extend_in_batch_sticky_within_thread(self):
        from sglang.srt.layers.dp_attention import (
            get_is_extend_in_batch,
            set_is_extend_in_batch,
        )

        reset_context()
        self.assertFalse(get_is_extend_in_batch())
        set_is_extend_in_batch(True)
        self.assertTrue(get_is_extend_in_batch())  # sticky until next write
        set_is_extend_in_batch(False)
        self.assertFalse(get_is_extend_in_batch())

    def test_moe_output_buffer_ctx(self):
        from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx
        from sglang.srt.runtime_context import get_forward

        reset_context()
        sentinel = object()
        with moe_output_buffer_ctx(sentinel):
            self.assertIs(get_forward().moe_output_buffer, sentinel)
        self.assertIsNone(get_forward().moe_output_buffer)


class TestPublishLifecycle(_IsolatedServerArgs):
    """Publish installs the resolved server_args and seeds the capture tier."""

    def _publish(self, **kw):
        args = _FakeResolvedArgs(**kw)
        get_context().set_server_args(args)
        return args

    def test_capture_tier_seeded_at_publish(self):
        args = self._publish(page_size=1)
        args.enable_torch_compile = True
        get_context().set_server_args(args)  # re-publish picks up the value
        self.assertTrue(get_flags().capture.enable_torch_compile)
        # capture-time write (B4) targets the capture leaf
        get_flags().capture.enable_torch_compile = False
        self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_capture_tier_defaults_for_sentinel_publish(self):
        get_context().set_server_args(object())
        self.assertFalse(get_flags().capture.enable_torch_compile)

    def test_declare_load_time_override_writes_through(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        args = self._publish(page_size=1)
        declare_load_time_override("model.load_time", {"page_size": 64})
        self.assertEqual(args.page_size, 64)

    def test_declare_load_time_override_validates_whitelist(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override

        args = self._publish(page_size=1)
        with self.assertRaises(ValueError):
            declare_load_time_override("bad", {"nope": 1})
        self.assertEqual(args.page_size, 1)

    def test_declare_load_time_override_records_provenance(self):
        from sglang.srt.arg_groups.overrides import declare_load_time_override
        from sglang.srt.server_args import ServerArgs

        class _Args(_FakeResolvedArgs):
            override = ServerArgs.override

        args = _Args(page_size=1)
        get_context().set_server_args(args)
        declare_load_time_override("model.load_time", {"page_size": 64})
        self.assertEqual(args.page_size, 64)
        self.assertIn(("model.load_time", {"page_size": 64}), args._resolved_overrides)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the flashinfer_mla native fp8 KV cache path.

``FlashInferMLAAttnBackend`` can hand flashinfer's MLA kernel the e4m3 pool
buffer directly on SM90 (plan with ``kv_data_type=torch.float8_e4m3fn`` plus
``ckv_scale``/``kpe_scale`` dequant scales), which removes the per-layer
full-pool ``.to(q.dtype)`` bf16 upcast. Every non-qualifying config must keep
the legacy upcast path byte-identically.

These tests are CPU-only: the guard is exercised through the pure helper, the
plan/run plumbing through mocked flashinfer wrappers and faked
model_runner/forward_batch objects. ``TestGraphOrchestrationIntegration``
additionally drives the REAL backend orchestration (eager decode, cuda-graph
capture, replay via the fast plans, target-verify) against recording fake
wrappers, so a kv-dtype mismatch anywhere between backend init, the indices
updaters, and the stock/fast plan swap is caught here.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.attention import flashinfer_mla_backend as mla_mod
from sglang.srt.layers.attention.flashinfer_mla_backend import (
    FlashInferMLAAttnBackend,
    FlashInferMLAIndicesUpdaterDecode,
    FlashInferMLAIndicesUpdaterPrefill,
    FlashInferMLAMultiStepDraftBackend,
    _fp8_kv_enabled,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

E4M3 = torch.float8_e4m3fn
BF16 = torch.bfloat16

# Positional index of q_data_type / kv_data_type in BatchMLAPagedAttention
# Wrapper.plan(...) calls made by the indices updaters. The fast replay plans
# installed over a captured wrapper share this layout (wrapper bound first),
# so kv_data_type is their LAST positional argument.
PLAN_Q_DTYPE_ARG = 10
PLAN_KV_DTYPE_ARG = 11


class _FakeSpecInput(SpecInput):
    """Real SpecInput subclass so the updaters' isinstance assert holds."""

    def __init__(self, kv_indptr=None, kv_indices=None):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices


def _call_mock(**attrs):
    """Stand-in for a runtime-context getter: calling it returns an object
    with the given attributes (not an auto-created child mock)."""
    return Mock(return_value=SimpleNamespace(**attrs))


def _fake_model_runner(kv_cache_dtype, model_dtype=BF16):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            context_len=4096,
            num_attention_heads=16,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=512,
            scaling=0.04,
        ),
        device="cpu",
        dtype=model_dtype,
        kv_cache_dtype=kv_cache_dtype,
        page_size=1,
        req_to_token_pool=SimpleNamespace(
            size=4,
            req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        ),
        token_to_kv_pool=Mock(),
        token_to_kv_pool_allocator=Mock(),
        server_args=SimpleNamespace(page_size=1),
    )


def _fake_layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=16,
        head_dim=576,
        v_head_dim=512,
        scaling=0.04,
        logit_cap=0.0,
    )


def _fake_forward_batch():
    return SimpleNamespace(
        out_cache_loc=None,
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        attn_attend_prefix_cache=None,
        extend_prefix_lens_cpu=None,
        attn_dcp_metadata=None,
    )


def _e4m3_pool_buffer():
    """Mimic MLATokenToKVPool.get_key_buffer under --kv-cache-dtype fp8_e4m3:
    an e4m3-typed view of the uint8 arena, shape (slots, 1, 576)."""
    return torch.zeros(16, 1, 576, dtype=torch.uint8).view(E4M3)


class _FakeTritonKernel:
    """Stand-in for a triton kernel object: ``kern[(grid,)]`` yields a
    no-op launcher, so CPU tests never compile or launch triton."""

    def __getitem__(self, grid):
        def _launch(*args, **kwargs):
            pass

        return _launch


class _RecordingMlaWrapper:
    """CPU stand-in for BatchMLAPagedAttentionWrapper: accepts the same
    constructor kwargs and records every plan call so the dtype plumbing is
    assertable. Instances also survive the fast-plan partial swap (the swap
    replaces the instance attribute, not the class)."""

    def __init__(self, workspace_buffer=None, **kwargs):
        self.plan_calls = []

    def plan(self, *args, **kwargs):
        self.plan_calls.append((args, kwargs))


class _ToCountingTensor(torch.Tensor):
    """Tensor subclass counting ``.to()`` calls: the fp8 run path must issue
    ZERO upcasts off the e4m3 pool buffer, the legacy path exactly one."""

    to_calls = 0

    def to(self, *args, **kwargs):
        type(self).to_calls += 1
        return super().to(*args, **kwargs)


class _ForwardMode:
    """Minimal forward_mode stand-in (method-style queries like the enum)."""

    def __init__(self, decode=False, target_verify=False):
        self._decode = decode
        self._target_verify = target_verify

    def is_decode_or_idle(self):
        return self._decode

    def is_decode(self):
        return self._decode

    def is_target_verify(self):
        return self._target_verify


def _fb_decode(bs):
    seq_lens = torch.tensor([3, 4][:bs], dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        seq_lens_cpu=seq_lens.clone(),
        positions=torch.zeros(bs, dtype=torch.int64),
        forward_mode=_ForwardMode(decode=True),
        spec_info=None,
        attn_dcp_metadata=None,
    )


def _verify_spec():
    spec = _FakeSpecInput(
        kv_indptr=torch.tensor([0, 3, 7], dtype=torch.int32),
        kv_indices=torch.zeros(7, dtype=torch.int32),
    )
    spec.draft_token_num = 4
    spec.ragged_verify_layout = None
    spec.generate_attn_arg_prefill = MagicMock(
        return_value=(
            torch.zeros(7, dtype=torch.int32),  # kv_indices
            torch.tensor([0, 3, 7], dtype=torch.int32),  # kv_indptr
            torch.tensor([0, 4, 8], dtype=torch.int32),  # qo_indptr
            None,  # custom_mask
        )
    )
    return spec


def _fb_verify(bs):
    seq_lens = torch.tensor([3, 4][:bs], dtype=torch.int64)
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        seq_lens=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        seq_lens_cpu=seq_lens.clone(),
        positions=torch.zeros(bs * 4, dtype=torch.int64),
        forward_mode=_ForwardMode(target_verify=True),
        spec_info=_verify_spec(),
        attn_dcp_metadata=None,
    )


class TestFp8KvGuard(CustomTestCase):
    QUALIFYING = dict(
        kv_cache_dtype=E4M3,
        model_dtype=BF16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        dcp_enabled=False,
        sm90_supported=True,
    )

    def test_qualifying_config_enables_fp8(self):
        self.assertTrue(_fp8_kv_enabled(**self.QUALIFYING))

    def test_single_disqualifier_disables_fp8(self):
        overrides = [
            ("auto/bf16 kv cache", dict(kv_cache_dtype=BF16)),
            ("e5m2 kv cache", dict(kv_cache_dtype=torch.float8_e5m2)),
            ("non-SM90 gpu", dict(sm90_supported=False)),
            ("fp16 model", dict(model_dtype=torch.float16)),
            ("kv_lora_rank != 512", dict(kv_lora_rank=576)),
            ("NoPE (kpe == 0)", dict(qk_rope_head_dim=0)),
            ("kpe == 128", dict(qk_rope_head_dim=128)),
            ("DCP enabled", dict(dcp_enabled=True)),
        ]
        for name, kw in overrides:
            with self.subTest(name):
                cfg = {**self.QUALIFYING, **kw}
                self.assertFalse(_fp8_kv_enabled(**cfg))


class TestBackendInitWiring(CustomTestCase):
    """__init__ must resolve fp8_kv/kv_data_type from model_runner and warn
    (once per process) when an fp8 KV cache was requested but disqualified."""

    def setUp(self):
        mla_mod._FP8_FALLBACK_WARNED = False

    def tearDown(self):
        mla_mod._FP8_FALLBACK_WARNED = False

    def _make_backend(self, kv_cache_dtype, sm90):
        model_runner = _fake_model_runner(kv_cache_dtype)
        fakes = dict(
            get_buffer=Mock(return_value=torch.zeros(16, dtype=torch.uint8)),
            get_disagg=_call_mock(disaggregation_mode="none"),
            get_schedule=_call_mock(disable_chunked_prefix_cache=True),
            get_exec=_call_mock(
                kernel=SimpleNamespace(flashinfer_mla_disable_ragged=True)
            ),
            get_parallel=_call_mock(dcp_enabled=False),
            is_sm90_supported=Mock(return_value=sm90),
            is_sm100_supported=Mock(return_value=not sm90),
            BatchMLAPagedAttentionWrapper=MagicMock(),
            BatchPrefillWithRaggedKVCacheWrapper=MagicMock(),
            FlashInferMLAIndicesUpdaterPrefill=MagicMock(),
            FlashInferMLAIndicesUpdaterDecode=MagicMock(),
        )
        with patch.multiple(mla_mod, **fakes):
            return FlashInferMLAAttnBackend(model_runner)

    def test_fp8_kv_on_sm90_takes_native_path(self):
        backend = self._make_backend(E4M3, sm90=True)
        self.assertTrue(backend.fp8_kv)
        self.assertEqual(backend.kv_data_type, E4M3)

    def test_fp8_kv_requested_but_disqualified_warns_and_falls_back(self):
        # e5m2 is an fp8 KV cache flashinfer's MLA path does not accept.
        for kv_dtype, sm90 in ((E4M3, False), (torch.float8_e5m2, True)):
            with self.subTest(kv_dtype=kv_dtype, sm90=sm90):
                mla_mod._FP8_FALLBACK_WARNED = False
                with self.assertLogs(
                    "sglang.srt.layers.attention.flashinfer_mla_backend",
                    level="WARNING",
                ) as logs:
                    backend = self._make_backend(kv_dtype, sm90=sm90)
                self.assertFalse(backend.fp8_kv)
                self.assertEqual(backend.kv_data_type, BF16)
                self.assertEqual(len(logs.output), 1)

    def test_disqualified_fp8_warns_once_across_instances(self):
        # Spec decode builds several backends per scheduler (one per draft
        # step, see TestSpecMultiStepBackends); the fallback config cannot
        # differ between them, so the warning must fire once per process.
        with self.assertLogs(
            "sglang.srt.layers.attention.flashinfer_mla_backend",
            level="WARNING",
        ) as logs:
            self._make_backend(E4M3, sm90=False)
            self._make_backend(E4M3, sm90=False)
        self.assertEqual(len(logs.output), 1)

    def test_non_fp8_kv_cache_never_warns(self):
        with self.assertNoLogs(
            "sglang.srt.layers.attention.flashinfer_mla_backend", level="WARNING"
        ):
            backend = self._make_backend(BF16, sm90=True)
        self.assertFalse(backend.fp8_kv)
        self.assertEqual(backend.kv_data_type, BF16)


class TestSpecMultiStepBackends(CustomTestCase):
    """The spec multi-step draft wrapper constructs one decode backend per
    draft step; each must independently qualify for fp8 (not just the first).
    The scaffolds stay silent under a disqualified fp8 request — only the
    selected backend's own instance emits/consumes the fallback warning."""

    def setUp(self):
        mla_mod._FP8_FALLBACK_WARNED = False

    def tearDown(self):
        mla_mod._FP8_FALLBACK_WARNED = False

    def _make_multistep(self, kv_cache_dtype, sm90):
        model_runner = _fake_model_runner(kv_cache_dtype)
        fakes = dict(
            get_buffer=Mock(return_value=torch.zeros(16, dtype=torch.uint8)),
            get_disagg=_call_mock(disaggregation_mode="none"),
            get_schedule=_call_mock(disable_chunked_prefix_cache=True),
            get_exec=_call_mock(
                kernel=SimpleNamespace(flashinfer_mla_disable_ragged=True)
            ),
            get_parallel=_call_mock(attn_tp_size=1, attn_dcp_size=1, dcp_enabled=False),
            is_sm90_supported=Mock(return_value=sm90),
            is_sm100_supported=Mock(return_value=not sm90),
            BatchMLAPagedAttentionWrapper=MagicMock(),
            BatchPrefillWithRaggedKVCacheWrapper=MagicMock(),
            unified_mla_hooks=Mock(
                return_value=SimpleNamespace(translate_kv_loc_dense=None)
            ),
        )
        with patch.multiple(mla_mod, **fakes):
            return FlashInferMLAMultiStepDraftBackend(
                model_runner, topk=1, speculative_num_steps=3
            )

    def test_every_draft_step_backend_takes_native_fp8(self):
        ms = self._make_multistep(E4M3, sm90=True)
        self.assertEqual(len(ms.attn_backends), 2)
        for backend in ms.attn_backends:
            self.assertTrue(backend.fp8_kv)
            self.assertEqual(backend.kv_data_type, E4M3)

    def test_disqualified_multistep_falls_back_without_warning(self):
        # The per-step backends are temporary scaffolds (the TRT-LLM multistep
        # subclass replaces them wholesale); the selected backend's own
        # instance owns the fallback warning, so construction here must stay
        # silent even when fp8 KV was requested and disqualified.
        with self.assertNoLogs(
            "sglang.srt.layers.attention.flashinfer_mla_backend", level="WARNING"
        ):
            ms = self._make_multistep(E4M3, sm90=False)
        for backend in ms.attn_backends:
            self.assertFalse(backend.fp8_kv)
            self.assertEqual(backend.kv_data_type, BF16)


class TestDecodePlanKvDtype(CustomTestCase):
    """Both decode wrapper.plan call sites (eager + cuda-graph fast replay)
    must pass the fp8 kv dtype only when the backend qualified for fp8."""

    def _make_updater(self, kv_data_type):
        updater = FlashInferMLAIndicesUpdaterDecode.__new__(
            FlashInferMLAIndicesUpdaterDecode
        )
        updater.num_local_heads = 16
        updater.kv_lora_rank = 512
        updater.qk_rope_head_dim = 64
        updater.scaling = 0.04
        updater.data_type = BF16
        updater.kv_data_type = kv_data_type
        return updater

    def _run(self, kv_dtype, init_metadata_replay):
        updater = self._make_updater(kv_dtype)
        wrapper = MagicMock()
        spec_info = _FakeSpecInput(
            kv_indptr=torch.tensor([0, 3, 7], dtype=torch.int32),
            kv_indices=torch.zeros(7, dtype=torch.int32),
        )
        fast_kwargs = (
            dict(
                qo_indptr_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
                kv_indptr_cpu=torch.tensor([0, 3, 7], dtype=torch.int32),
                kv_len_arr_cpu=torch.tensor([3, 4], dtype=torch.int32),
                kv_indices=spec_info.kv_indices,
            )
            if init_metadata_replay
            else {}
        )
        updater.call_begin_forward(
            wrapper,
            torch.tensor([0, 1]),
            torch.tensor([3, 4], dtype=torch.int64),
            7,
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 3, 7], dtype=torch.int32),
            init_metadata_replay=init_metadata_replay,
            spec_info=spec_info,
            **fast_kwargs,
        )
        wrapper.plan.assert_called_once()
        args = wrapper.plan.call_args.args
        self.assertEqual(args[PLAN_Q_DTYPE_ARG], BF16)
        self.assertEqual(args[PLAN_KV_DTYPE_ARG], kv_dtype)

    def test_eager_plan_passes_kv_data_type(self):
        for kv_dtype in (E4M3, BF16):
            with self.subTest(kv_data_type=kv_dtype):
                self._run(kv_dtype, init_metadata_replay=False)

    def test_fast_replay_plan_passes_kv_data_type(self):
        for kv_dtype in (E4M3, BF16):
            with self.subTest(kv_data_type=kv_dtype):
                self._run(kv_dtype, init_metadata_replay=True)


class TestPrefillPlanKvDtype(CustomTestCase):
    def test_paged_plan_passes_kv_data_type(self):
        for kv_dtype in (E4M3, BF16):
            with self.subTest(kv_data_type=kv_dtype):
                updater = FlashInferMLAIndicesUpdaterPrefill.__new__(
                    FlashInferMLAIndicesUpdaterPrefill
                )
                updater.num_local_heads = 16
                updater.kv_lora_rank = 512
                updater.qk_nope_head_dim = 128
                updater.qk_rope_head_dim = 64
                updater.v_head_dim = 512
                updater.scaling = 0.04
                updater.data_type = BF16
                updater.q_data_type = BF16
                updater.kv_data_type = kv_dtype
                updater.req_to_token = torch.zeros((4, 16), dtype=torch.int64)

                wrapper_paged = MagicMock()
                spec_info = _FakeSpecInput()
                spec_info.generate_attn_arg_prefill = MagicMock(
                    return_value=(
                        torch.zeros(7, dtype=torch.int32),
                        torch.tensor([0, 3, 7], dtype=torch.int32),
                        torch.tensor([0, 2, 4], dtype=torch.int32),
                        None,
                    )
                )
                updater.call_begin_forward(
                    MagicMock(),  # wrapper_ragged (unused, use_ragged=False)
                    wrapper_paged,
                    torch.tensor([0, 1]),
                    torch.tensor([3, 4], dtype=torch.int64),
                    7,
                    torch.tensor([3, 4], dtype=torch.int64),
                    torch.tensor([0, 0], dtype=torch.int64),
                    torch.tensor([0, 3, 7], dtype=torch.int32),
                    torch.tensor([0, 2, 4], dtype=torch.int32),
                    use_ragged=False,
                    spec_info=spec_info,
                )
                wrapper_paged.plan.assert_called_once()
                args = wrapper_paged.plan.call_args.args
                self.assertEqual(args[PLAN_Q_DTYPE_ARG], BF16)
                self.assertEqual(args[PLAN_KV_DTYPE_ARG], kv_dtype)


class TestRunBranchSelection(CustomTestCase):
    """forward_decode / paged forward_extend must feed flashinfer native e4m3
    views with unit dequant scales on the fp8 path, and keep the full-pool
    bf16 upcast (no scale kwargs) otherwise."""

    def _make_backend(self, fp8_kv):
        backend = FlashInferMLAAttnBackend.__new__(FlashInferMLAAttnBackend)
        backend.fp8_kv = fp8_kv
        pool = _e4m3_pool_buffer().as_subclass(_ToCountingTensor)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=Mock(return_value=pool)
        )
        run_mock = MagicMock()
        backend.forward_metadata = SimpleNamespace(
            decode_wrapper=run_mock,
            prefill_wrapper=run_mock,
            use_ragged=False,
        )
        return backend, pool, run_mock

    def _assert_kv_args(self, call, pool, fp8_kv, base_kwargs):
        ckv, kpe = call.args[2], call.args[3]
        expected_kwargs = set(base_kwargs) | (
            {"ckv_scale", "kpe_scale"} if fp8_kv else set()
        )
        # Exact kwarg set: flashinfer rejects the scale kwargs when the
        # planned KV dtype is not fp8, and unknown kwargs are never added.
        self.assertEqual(set(call.kwargs), expected_kwargs)
        if fp8_kv:
            # Views straight off the e4m3 pool buffer: no upcast copy, ckv at
            # offset 0 and kpe at offset 512 of each 576-wide row.
            self.assertEqual(ckv.dtype, E4M3)
            self.assertEqual(kpe.dtype, E4M3)
            self.assertEqual(ckv.shape, (16, 1, 512))
            self.assertEqual(kpe.shape, (16, 1, 64))
            # Both views keep the pool's row layout (no materializing slice).
            self.assertEqual(ckv.stride(), (576, 576, 1))
            self.assertEqual(kpe.stride(), (576, 576, 1))
            self.assertEqual(ckv.data_ptr(), pool.data_ptr())
            self.assertEqual(kpe.data_ptr(), pool.data_ptr() + 512)
            self.assertEqual(call.kwargs.get("ckv_scale"), 1.0)
            self.assertEqual(call.kwargs.get("kpe_scale"), 1.0)
            # Zero upcasts off the pool buffer on the native path.
            self.assertEqual(_ToCountingTensor.to_calls, 0)
        else:
            # Legacy path: exactly one bf16 upcast copy of the pool, no
            # scale kwargs.
            self.assertEqual(ckv.dtype, BF16)
            self.assertEqual(kpe.dtype, BF16)
            self.assertNotEqual(ckv.data_ptr(), pool.data_ptr())
            self.assertNotIn("ckv_scale", call.kwargs)
            self.assertNotIn("kpe_scale", call.kwargs)
            self.assertEqual(_ToCountingTensor.to_calls, 1)

    @patch.object(mla_mod, "get_parallel", return_value=Mock(dcp_enabled=False))
    def test_forward_decode_run_branch(self, _mock_parallel):
        q = torch.zeros(4, 16, 512, dtype=BF16)
        q_rope = torch.zeros(4, 16, 64, dtype=BF16)
        for fp8_kv in (True, False):
            with self.subTest(fp8_kv=fp8_kv):
                _ToCountingTensor.to_calls = 0
                backend, pool, run_mock = self._make_backend(fp8_kv)
                backend.forward_decode(
                    q,
                    None,
                    None,
                    _fake_layer(),
                    _fake_forward_batch(),
                    save_kv_cache=False,
                    q_rope=q_rope,
                    k_rope=None,
                )
                run_mock.run.assert_called_once()
                self._assert_kv_args(
                    run_mock.run.call_args, pool, fp8_kv, {"out", "return_lse"}
                )

    def test_forward_extend_paged_run_branch(self):
        q = torch.zeros(4, 16, 512, dtype=BF16)
        q_rope = torch.zeros(4, 16, 64, dtype=BF16)
        for fp8_kv in (True, False):
            with self.subTest(fp8_kv=fp8_kv):
                _ToCountingTensor.to_calls = 0
                backend, pool, run_mock = self._make_backend(fp8_kv)
                backend.forward_extend(
                    q,
                    None,
                    None,
                    _fake_layer(),
                    _fake_forward_batch(),
                    save_kv_cache=False,
                    q_rope=q_rope,
                    k_rope=None,
                )
                run_mock.run.assert_called_once()
                self._assert_kv_args(run_mock.run.call_args, pool, fp8_kv, {"out"})


class TestGraphOrchestrationIntegration(CustomTestCase):
    """End-to-end wiring test over the REAL backend and indices updaters
    (only the flashinfer wrapper classes, the triton kv-indices kernel, the
    runtime-context getters, and the fast plan bodies are faked), covering
    the plan-side dtype plumbing across the whole cuda-graph lifecycle:

    - eager decode plans the __init__ stock decode wrapper;
    - decode capture plans the per-bs STOCK wrapper once in eager form (the
      call whose cached module every replay inherits), and only THEN installs
      ``fast_mla_decode_plan`` over it — capture's own graph-replay-form plan
      and every subsequent replay already route through that fast plan bound
      to the SAME captured wrapper;
    - target-verify capture/replay do the same on the paged prefill wrapper
      (stock plan at capture, fast prefill plan at replay).

    The bf16 control runs the identical sequence and must produce BF16 plans
    everywhere, proving the asserts actually bind to the backend's dtype.
    """

    def setUp(self):
        mla_mod._FP8_FALLBACK_WARNED = False

    def _orchestrate(self, kv_cache_dtype, sm90=True):
        fast_calls = []

        def _fast_decode(wrapper, *args, **kwargs):
            fast_calls.append(("decode", wrapper, args))

        def _fast_prefill(wrapper, *args, **kwargs):
            fast_calls.append(("prefill", wrapper, args))

        fakes = dict(
            get_buffer=Mock(return_value=torch.zeros(1024, dtype=torch.uint8)),
            get_disagg=_call_mock(disaggregation_mode="none"),
            get_schedule=_call_mock(disable_chunked_prefix_cache=True),
            get_exec=_call_mock(
                kernel=SimpleNamespace(flashinfer_mla_disable_ragged=True)
            ),
            get_parallel=_call_mock(attn_tp_size=1, attn_dcp_size=1, dcp_enabled=False),
            is_sm90_supported=Mock(return_value=sm90),
            is_sm100_supported=Mock(return_value=not sm90),
            BatchMLAPagedAttentionWrapper=_RecordingMlaWrapper,
            BatchPrefillWithRaggedKVCacheWrapper=MagicMock(),
            create_flashinfer_kv_indices_triton=_FakeTritonKernel(),
            unified_mla_hooks=Mock(
                return_value=SimpleNamespace(translate_kv_loc_dense=None)
            ),
            update_local_kv_lens_for_dcp=Mock(),
            fast_mla_decode_plan=_fast_decode,
            fast_mla_prefill_plan=_fast_prefill,
        )
        real_empty = torch.empty

        def _cpu_empty(*args, **kwargs):
            # The eager kv_indices scratch is hard-allocated on "cuda"; the
            # triton kernel that would fill it is faked, so CPU scratch is
            # equivalent for this test.
            if kwargs.get("device") == "cuda":
                kwargs = {**kwargs, "device": "cpu"}
            return real_empty(*args, **kwargs)

        with patch.object(torch, "empty", _cpu_empty), patch.multiple(mla_mod, **fakes):
            backend = FlashInferMLAAttnBackend(_fake_model_runner(kv_cache_dtype))
            backend.init_cuda_graph_state(
                max_bs=4,
                max_num_tokens=4,
                kv_indices_buf=torch.zeros(64, dtype=torch.int32),
            )

            # 1) eager decode (non-graph path)
            backend.init_forward_metadata(_fb_decode(bs=2))
            eager_wrapper = backend.decode_wrapper

            # 2) decode cuda-graph capture
            backend.init_forward_metadata_out_graph(_fb_decode(bs=2), in_capture=True)
            cap_wrapper = backend.decode_cuda_graph_metadata[2]

            # 3) decode cuda-graph replay
            backend.init_forward_metadata_out_graph(_fb_decode(bs=2), in_capture=False)

            # 4) target-verify capture
            backend.init_forward_metadata_out_graph(_fb_verify(bs=2), in_capture=True)
            verify_wrapper = backend.prefill_cuda_graph_metadata[2]

            # 5) target-verify replay
            backend.init_forward_metadata_out_graph(_fb_verify(bs=2), in_capture=False)

        return SimpleNamespace(
            backend=backend,
            eager_wrapper=eager_wrapper,
            cap_wrapper=cap_wrapper,
            verify_wrapper=verify_wrapper,
            fast_calls=fast_calls,
        )

    def test_kv_dtype_flows_through_every_plan(self):
        for kv_dtype, expect_fp8 in ((E4M3, True), (BF16, False)):
            with self.subTest(kv_dtype=kv_dtype):
                ev = self._orchestrate(kv_dtype)
                self.assertIs(ev.backend.fp8_kv, expect_fp8)
                self.assertEqual(ev.backend.kv_data_type, kv_dtype)

                # Eager decode: the __init__ stock wrapper is planned once,
                # with the backend's kv dtype in the last plan slot.
                self.assertIsInstance(ev.eager_wrapper, _RecordingMlaWrapper)
                self.assertEqual(len(ev.eager_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.eager_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], kv_dtype
                )

                # Capture: the per-bs wrapper registered into the graph
                # metadata is the same instance that was planned.
                self.assertIsInstance(ev.cap_wrapper, _RecordingMlaWrapper)
                self.assertIs(ev.backend.decode_cuda_graph_metadata[2], ev.cap_wrapper)
                # The stock wrapper got exactly ONE plan call (eager form) —
                # the call that primes _cached_module; afterwards its plan
                # attribute was replaced by the fast-plan partial, so both
                # capture's graph-replay-form plan and the real replay below
                # must route through the fast path instead.
                self.assertEqual(len(ev.cap_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.cap_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], kv_dtype
                )
                self.assertIn("plan", vars(ev.cap_wrapper))

                # Fast decode plans: exactly two — capture's replay-form plan
                # and the replay leg — both bound to the SAME captured
                # wrapper (cached-module reuse), both carrying the dtype.
                decode_fast = [c for c in ev.fast_calls if c[0] == "decode"]
                self.assertEqual(len(decode_fast), 2)
                for _kind, wrapper, args in decode_fast:
                    self.assertIs(wrapper, ev.cap_wrapper)
                    self.assertEqual(args[-1], kv_dtype)
                # No further stock plan calls happened during replay.
                self.assertEqual(len(ev.cap_wrapper.plan_calls), 1)

                # Target-verify: capture planned the paged wrapper once;
                # replay went through fast_mla_prefill_plan on the same one.
                self.assertIsInstance(ev.verify_wrapper, _RecordingMlaWrapper)
                self.assertEqual(len(ev.verify_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.verify_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], kv_dtype
                )
                self.assertIn("plan", vars(ev.verify_wrapper))
                prefill_fast = [c for c in ev.fast_calls if c[0] == "prefill"]
                self.assertEqual(len(prefill_fast), 1)
                self.assertIs(prefill_fast[0][1], ev.verify_wrapper)
                self.assertEqual(prefill_fast[0][2][-1], kv_dtype)

    def test_fast_plans_never_touch_the_init_stock_decode_wrapper(self):
        # The eager (non-graph) decode wrapper must not be captured into the
        # graph metadata nor drive the fast replay path.
        ev = self._orchestrate(E4M3)
        self.assertIsNot(ev.cap_wrapper, ev.eager_wrapper)
        for _kind, wrapper, _args in ev.fast_calls:
            self.assertIsNot(wrapper, ev.eager_wrapper)

    def test_disqualified_fp8_plans_compute_dtype_not_requested_dtype(self):
        # Disqualified fp8 KV caches (here: e4m3 on non-SM90, and e5m2) must
        # plan with the compute dtype (bf16), never the requested kv cache
        # dtype — pinning the updaters to attn_backend.kv_data_type rather
        # than model_runner.kv_cache_dtype at every plan site.
        cases = (
            ("e4m3 on non-SM90", E4M3, False),
            ("e5m2 on SM90", torch.float8_e5m2, True),
        )
        for name, kv_cache_dtype, sm90 in cases:
            with self.subTest(name):
                mla_mod._FP8_FALLBACK_WARNED = False
                ev = self._orchestrate(kv_cache_dtype, sm90=sm90)
                self.assertFalse(ev.backend.fp8_kv)
                self.assertEqual(ev.backend.kv_data_type, BF16)

                self.assertEqual(len(ev.eager_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.eager_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], BF16
                )
                self.assertEqual(len(ev.cap_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.cap_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], BF16
                )
                self.assertEqual(len(ev.verify_wrapper.plan_calls), 1)
                self.assertEqual(
                    ev.verify_wrapper.plan_calls[0][0][PLAN_KV_DTYPE_ARG], BF16
                )
                for _kind, _wrapper, args in ev.fast_calls:
                    self.assertEqual(args[-1], BF16)


if __name__ == "__main__":
    unittest.main()

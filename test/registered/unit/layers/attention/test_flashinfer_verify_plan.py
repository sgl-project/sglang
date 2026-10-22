"""FlashInfer target-verify planning - CPU only.

* FlashInfer keys its target-verify CUDA-graph wrappers by (bs, num_tokens),
  so a runner that captures several verify widths per batch size replays each
  graph with the wrappers it was captured with.
* A target-verify replay whose spec input knows its KV lengths on the host
  (``SpecInput.attn_kv_lens_cpu``) is planned by ``fast_prefill_plan`` from
  host arrays, with no device read; a one-request tree mask is packed to its
  host-known size.
* A spec input may build its graph's verify wrappers with its own FlashInfer
  attention variant (``SpecInput.flashinfer_verify_variant``), through the
  public ``jit_args`` / ``variant_owns_mask`` constructor options; such
  wrappers are planned without a mask.

Backends are built with ``__new__`` and only the attributes each method reads.
"""

import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import flashinfer_backend as fi_mod
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferIndicesUpdaterPrefill,
    fast_prefill_plan,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.spec_info import (
    FlashInferVerifyVariant,
    SpecInput,
    SpecInputType,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Wrapper:
    """A FlashInfer wrapper stand-in (hashable and weak-referenceable)."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _flashinfer_backend():
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.prefill_cuda_graph_metadata = {}
    backend.target_verify_cuda_graph_wrappers = {}
    backend.use_sliding_window_kv_pool = False
    backend.prefill_backend = "fa2"
    backend.dispatch_reason = None
    backend.indices_updater_prefill = mock.MagicMock()
    backend.variant_mask_wrappers = weakref.WeakSet()
    created = []

    def create(bs, use_custom_mask=False, variant=None):
        wrappers = [_Wrapper(begin_forward=None)]
        created.append(wrappers)
        if variant is not None:
            backend.variant_mask_wrappers.update(wrappers)
        return wrappers

    backend._create_prefill_wrappers = create
    return backend, created


def _replay_view(bs, num_tokens, spec_info=None):
    return SimpleNamespace(
        batch_size=bs,
        req_pool_indices=torch.zeros(bs, dtype=torch.int64),
        seq_lens=torch.ones(bs, dtype=torch.int64),
        seq_lens_cpu=None,
        seq_lens_sum=bs,
        encoder_lens=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=spec_info,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        out_cache_loc=None,
        kv_loc_plan=None,
    )


class TestWidthKeyedVerifyWrappers(CustomTestCase):
    def test_each_verify_width_replays_its_own_wrappers(self):
        backend, created = _flashinfer_backend()
        # Capture widths 4 then 2 for bs=1 (widest first, as a provider does).
        backend._prepare_cuda_graph_metadata(1, 4, ForwardMode.TARGET_VERIFY, None)
        backend._prepare_cuda_graph_metadata(1, 2, ForwardMode.TARGET_VERIFY, None)
        wide, narrow = created

        backend.init_forward_metadata_out_graph(_replay_view(1, 4))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], wide)

        backend.init_forward_metadata_out_graph(_replay_view(1, 2))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], narrow)

    def test_recapturing_a_width_keeps_the_wrappers_of_earlier_graphs(self):
        backend, created = _flashinfer_backend()
        # Two output variants can capture the same attention shape. Each graph
        # retains its wrapper's plan buffers, so replay must refresh both.
        backend._prepare_cuda_graph_metadata(1, 4, ForwardMode.TARGET_VERIFY, None)
        first = backend.forward_metadata.prefill_wrappers
        backend._prepare_cuda_graph_metadata(1, 2, ForwardMode.TARGET_VERIFY, None)
        backend._prepare_cuda_graph_metadata(1, 4, ForwardMode.TARGET_VERIFY, None)
        second = backend.forward_metadata.prefill_wrappers
        backend.init_forward_metadata_out_graph(_replay_view(1, 4))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], first)
        self.assertIs(used["prefill_wrappers"], second)
        self.assertEqual(len(created), 2)

    def test_one_width_per_bs_matches_the_bs_keyed_wrappers(self):
        backend, created = _flashinfer_backend()
        backend._prepare_cuda_graph_metadata(2, 8, ForwardMode.TARGET_VERIFY, None)
        backend.init_forward_metadata_out_graph(_replay_view(2, 8))
        used = backend.indices_updater_prefill.update.call_args.kwargs
        self.assertIs(used["prefill_wrappers"], backend.prefill_cuda_graph_metadata[2])
        self.assertIs(used["prefill_wrappers"], created[0])


def _packbits(bits, bitorder):
    """flashinfer.packbits on the CPU."""
    assert bitorder == "little"
    bits = bits.to(torch.uint8)
    bits = torch.nn.functional.pad(bits, (0, -len(bits) % 8)).view(-1, 8)
    return (bits << torch.arange(8, dtype=torch.uint8)).sum(-1).to(torch.uint8)


class _HostLayoutInput(SpecInput):
    """A verify input whose generate_attn_arg_prefill widens by ``width``."""

    def __init__(self, width, custom_mask=None, host_known=True):
        super().__init__(SpecInputType.EAGLE_VERIFY)
        self.num_tokens_per_req = width
        self.custom_mask = custom_mask
        self.host_known = host_known

    def attn_kv_lens_cpu(self, seq_lens_cpu):
        return seq_lens_cpu + self.num_tokens_per_req if self.host_known else None

    def generate_attn_arg_prefill(self, *, req_pool_indices, paged_kernel_lens, **kw):
        kv_lens = paged_kernel_lens + self.num_tokens_per_req
        kv_indptr = torch.zeros(len(kv_lens) + 1, dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(kv_lens, 0)
        qo_indptr = torch.arange(len(kv_lens) + 1, dtype=torch.int32) * (
            self.num_tokens_per_req
        )
        kv_indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32)
        return kv_indices, kv_indptr, qo_indptr, self.custom_mask


def _capture(backend, bs, spec_info, seq_len=5):
    view = _replay_view(bs, bs * spec_info.num_tokens_per_req, spec_info)
    view.seq_lens_cpu = torch.full((bs,), seq_len)
    backend.init_forward_metadata_out_graph(view, in_capture=True)
    return backend.target_verify_cuda_graph_wrappers[(bs, view.positions.numel())]


def _is_fast(wrapper):
    return getattr(wrapper.begin_forward, "func", None) is fast_prefill_plan


class TestHostLayoutReplayPlan(CustomTestCase):
    def test_eagle_verify_kv_lens_add_the_verify_tokens(self):
        verify = EagleVerifyInput.create_idle_input(
            topk=4, spec_steps=3, num_verify_tokens=6, device="cpu"
        )
        self.assertEqual(
            verify.attn_kv_lens_cpu(torch.tensor([10, 3])).tolist(), [16, 9]
        )
        self.assertIsNone(SpecInput.attn_kv_lens_cpu(verify, torch.tensor([1])))

    def test_capture_installs_the_host_plan_where_the_layout_is_host_known(self):
        mask = torch.ones(64, dtype=torch.bool)
        cases = [
            (1, _HostLayoutInput(4, custom_mask=mask), True),  # tree mask, one request
            (
                2,
                _HostLayoutInput(4, custom_mask=mask),
                False,
            ),  # mask size not host-known
            (2, _HostLayoutInput(4), True),  # chain verify, no mask
            (1, _HostLayoutInput(4, host_known=False), False),
        ]
        for bs, spec_info, fast in cases:
            backend, _ = _flashinfer_backend()
            (wrapper,) = _capture(backend, bs, spec_info)
            self.assertIs(_is_fast(wrapper), fast, (bs, spec_info.host_known))

    def test_replay_plans_from_host_arrays_and_packs_the_mask(self):
        width, seq_len = 3, 4
        kv_len = seq_len + width
        mask = torch.rand(width * kv_len + 9) > 0.5  # padded past the request
        spec_info = _HostLayoutInput(width, custom_mask=mask)
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(reads_are_translated=False),
            variant_mask_wrappers=weakref.WeakSet(),
        )
        updater.kv_last_page_len = torch.ones(4, dtype=torch.int32)
        updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = 8, 2, 64
        updater.q_data_type = updater.data_type = torch.bfloat16
        calls = []
        with mock.patch.object(fi_mod, "packbits", _packbits, create=True):
            updater.call_begin_forward(
                wrapper_ragged=None,
                wrapper_paged=_FastPlanRecorder(calls),
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
                paged_kernel_lens=torch.tensor([seq_len]),
                paged_kernel_lens_sum=seq_len,
                seq_lens=torch.tensor([seq_len]),
                prefix_lens=None,
                kv_start_idx=None,
                kv_indptr=None,
                qo_indptr=None,
                use_ragged=False,
                spec_info=spec_info,
                seq_lens_cpu=torch.tensor([seq_len]),
                plan=None,
            )
        ((_, kwargs),) = calls
        self.assertEqual(kwargs["qo_indptr_host"].tolist(), [0, width])
        self.assertEqual(kwargs["kv_indptr_host"].tolist(), [0, kv_len])
        self.assertEqual(kwargs["kv_lens_host"].tolist(), [kv_len])
        self.assertEqual(kwargs["max_kv_len"], kv_len)
        bits = width * kv_len
        torch.testing.assert_close(
            kwargs["packed_custom_mask"], _packbits(mask[:bits], "little")
        )
        self.assertEqual(kwargs["mask_indptr_host"].tolist(), [0, (bits + 7) // 8])

    def test_fast_plan_refreshes_the_packed_mask_buffers(self):
        wrapper = SimpleNamespace(
            is_cuda_graph_enabled=True,
            device=torch.device("cpu"),
            _backend="fa2",
            _cached_module=SimpleNamespace(plan=mock.MagicMock(return_value="plan")),
            _max_total_num_rows=None,
            _float_workspace_buffer=None,
            _int_workspace_buffer=None,
            _pin_memory_int_workspace_buffer=None,
            _qo_indptr_buf=torch.zeros(2, dtype=torch.int32),
            _paged_kv_indptr_buf=torch.zeros(2, dtype=torch.int32),
            _paged_kv_last_page_len_buf=torch.zeros(1, dtype=torch.int32),
            _paged_kv_indices_buf=torch.zeros(16, dtype=torch.int32),
            _custom_mask_buf=torch.zeros(8, dtype=torch.uint8),
            _mask_indptr_buf=torch.zeros(2, dtype=torch.int32),
        )
        packed = torch.tensor([7, 9, 11], dtype=torch.uint8)
        fast_prefill_plan(
            wrapper,
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 9], dtype=torch.int32),
            torch.arange(9, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            8,
            2,
            64,
            1,
            qo_indptr_host=torch.tensor([0, 2], dtype=torch.int32),
            kv_indptr_host=torch.tensor([0, 9], dtype=torch.int32),
            kv_lens_host=torch.tensor([9], dtype=torch.int32),
            max_q_len=2,
            max_kv_len=9,
            packed_custom_mask=packed,
            mask_indptr_host=torch.tensor([0, 3], dtype=torch.int32),
        )
        self.assertEqual(wrapper._custom_mask_buf[:3].tolist(), [7, 9, 11])
        self.assertEqual(wrapper._mask_indptr_buf.tolist(), [0, 3])
        self.assertEqual(wrapper._plan_info, "plan")


class _VariantInput(_HostLayoutInput):
    """A verify input that builds its bs-1 graphs with its own variant."""

    def __init__(self, width):
        super().__init__(width, custom_mask=torch.ones(64, dtype=torch.bool))
        self.variant = FlashInferVerifyVariant(
            jit_args=("uri",), mask_buf=torch.zeros(64, dtype=torch.uint8)
        )
        self.variant_calls = []

    def flashinfer_verify_variant(self, bs, num_tokens, **dtypes_and_head_dim):
        self.variant_calls.append((bs, num_tokens, dtypes_and_head_dim))
        return self.variant if bs == 1 else None


class TestVerifyAttentionVariant(CustomTestCase):
    def test_capture_builds_the_variant_wrappers_of_the_graph(self):
        backend, _ = _flashinfer_backend()
        backend.num_wrappers = 1
        backend.workspace_buffer = torch.zeros(1)
        backend.cuda_graph_qo_indptr = backend.kv_indptr = [torch.zeros(8)]
        backend.cuda_graph_qk_indptr = [torch.zeros(8)]
        backend.cuda_graph_kv_indices = [torch.zeros(8)]
        backend.cuda_graph_custom_mask = torch.zeros(8, dtype=torch.uint8)
        backend.kv_last_page_len = torch.ones(8)
        backend.indices_updater_prefill = SimpleNamespace(
            q_data_type=torch.bfloat16, data_type=torch.float8_e4m3fn, head_dim=128
        )
        built = []
        spec_info = _VariantInput(4)
        layout = dict(
            q_dtype=torch.bfloat16, kv_dtype=torch.float8_e4m3fn, head_dim=128
        )
        with mock.patch.object(
            fi_mod,
            "BatchPrefillWithPagedKVCacheWrapper",
            side_effect=lambda *a, **kw: built.append(kw) or _Wrapper(**kw),
            create=True,
        ):
            del backend._create_prefill_wrappers  # the real one
            backend._prepare_cuda_graph_metadata(
                1, 4, ForwardMode.TARGET_VERIFY, spec_info
            )
            backend._prepare_cuda_graph_metadata(
                2, 8, ForwardMode.TARGET_VERIFY, spec_info
            )
        self.assertEqual(
            spec_info.variant_calls,
            [(bs, n, layout) for bs, n in ((1, 4), (2, 8))],
        )
        variant_kw, default_kw = built
        self.assertEqual(variant_kw["jit_args"], ["uri"])
        self.assertIs(variant_kw["custom_mask_buf"], spec_info.variant.mask_buf)
        self.assertTrue(variant_kw["variant_owns_mask"])
        self.assertNotIn("jit_args", default_kw)
        self.assertIs(default_kw["custom_mask_buf"], backend.cuda_graph_custom_mask)
        (variant_wrapper,) = backend.target_verify_cuda_graph_wrappers[(1, 4)]
        (default_wrapper,) = backend.target_verify_cuda_graph_wrappers[(2, 8)]
        self.assertIn(variant_wrapper, backend.variant_mask_wrappers)
        self.assertNotIn(default_wrapper, backend.variant_mask_wrappers)

    def test_variant_wrappers_are_planned_without_a_mask(self):
        spec_info = _VariantInput(4)
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.kv_last_page_len = torch.ones(4, dtype=torch.int32)
        updater.num_qo_heads, updater.num_kv_heads, updater.head_dim = 8, 2, 64
        updater.q_data_type = updater.data_type = torch.bfloat16
        for in_variant_set in (True, False):
            calls = []
            wrapper = _FastPlanRecorder(calls)
            wrapper.begin_forward = lambda *a, **kw: calls.append(kw)
            wrapper._custom_mask_buf = torch.zeros(8, dtype=torch.uint8)
            variant_wrappers = weakref.WeakSet([wrapper] if in_variant_set else [])
            updater.attn_backend = SimpleNamespace(
                kv_index_translator=SimpleNamespace(reads_are_translated=False),
                variant_mask_wrappers=variant_wrappers,
            )
            updater.call_begin_forward(
                wrapper_ragged=None,
                wrapper_paged=wrapper,
                req_pool_indices=torch.zeros(1, dtype=torch.int64),
                paged_kernel_lens=torch.tensor([5]),
                paged_kernel_lens_sum=5,
                seq_lens=torch.tensor([5]),
                prefix_lens=None,
                kv_start_idx=None,
                kv_indptr=None,
                qo_indptr=None,
                use_ragged=False,
                spec_info=spec_info,
                plan=None,
            )
            (kwargs,) = calls
            expected = None if in_variant_set else spec_info.custom_mask
            self.assertIs(kwargs["custom_mask"], expected)


class _RecordingPartial:
    """Stands in for the partial(fast_prefill_plan, wrapper) capture installs."""

    func = staticmethod(fast_prefill_plan)

    def __init__(self, calls):
        self.calls = calls

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FastPlanRecorder:
    """A cuda-graph wrapper whose begin_forward is the installed fast plan."""

    def __init__(self, calls):
        self.is_cuda_graph_enabled = True
        self._custom_mask_buf = None
        self.begin_forward = _RecordingPartial(calls)


if __name__ == "__main__":
    unittest.main()

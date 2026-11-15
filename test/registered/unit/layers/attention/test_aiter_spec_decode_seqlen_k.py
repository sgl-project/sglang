"""AITER unified-attention decode KV lengths under EAGLE draft decode, built on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import aiter_backend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_SEQ_LENS = [32, 64, 65]
_BATCH_SIZE = len(_SEQ_LENS)
_HEAD_DIM = 8
_MAX_BLOCKS = 128


class _NoopKernel:
    def __getitem__(self, _grid):
        return lambda *_args, **_kwargs: None


def _spec_info(step: int):
    """EAGLE draft backend `step` owns rows of seq_lens + step + 1 tokens: the
    committed prefix plus one drafted token per step taken so far, including
    this step's own token."""
    row_lens = torch.tensor([s + step + 1 for s in _SEQ_LENS], dtype=torch.int32)
    kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(row_lens, dim=0)
    return SimpleNamespace(
        kv_indptr=kv_indptr,
        kv_indices=torch.arange(int(row_lens.sum()), dtype=torch.int32),
    )


def _decode_batch(spec_info):
    seq_lens = torch.tensor(_SEQ_LENS, dtype=torch.int64)
    return SimpleNamespace(
        batch_size=_BATCH_SIZE,
        req_pool_indices=torch.arange(_BATCH_SIZE, dtype=torch.int32),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        seq_lens_sum=int(seq_lens.sum()),
        encoder_lens=None,
        spec_info=spec_info,
        forward_mode=ForwardMode.DECODE,
        out_cache_loc=None,
    )


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=_HEAD_DIM,
        v_head_dim=_HEAD_DIM,
        scaling=1.0,
        logit_cap=0.0,
        sliding_window_size=None,
        k_scale=None,
        v_scale=None,
    )


def _unified_decode_backend():
    backend = aiter_backend.AiterAttnBackend.__new__(aiter_backend.AiterAttnBackend)
    backend.device = torch.device("cpu")
    backend.use_mla = False
    backend.use_sliding_window_kv_pool = False
    backend.decode_uses_page_table = True
    backend.use_triton_unified_attention = True
    backend.kv_cache_is_vectorized_5d = False
    backend.dcp_world_size = 1
    backend.page_size = 1
    backend.max_context_len = _MAX_BLOCKS
    backend.input_dtype = torch.float32
    backend.kv_cache_dtype = torch.float32
    backend.kv_indptr = torch.zeros(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.qo_indptr_unified_decode = torch.arange(_BATCH_SIZE + 1, dtype=torch.int32)
    backend.req_to_token = torch.zeros((_BATCH_SIZE, _MAX_BLOCKS), dtype=torch.int32)
    backend._decode_partition_buckets = [("default", 1 << 30, 1)]
    backend.max_num_partitions = 1

    cache = torch.zeros(_MAX_BLOCKS, 1, 1, _HEAD_DIM)
    backend.token_to_kv_pool = SimpleNamespace(
        layer_router=SimpleNamespace(
            read=lambda _layer_id: SimpleNamespace(kv=lambda: (cache, cache))
        )
    )

    def build_page_table(spec_info, bs, dest_buf=None, swa_dest_buf=None):
        table = (
            dest_buf
            if dest_buf is not None
            else torch.zeros((bs, _MAX_BLOCKS), dtype=torch.int32)
        )
        return table, None

    backend._build_unified_page_table_from_spec = build_page_table
    return backend


class _CaptureUnifiedAttention:
    """Stands in for the aiter kernel so the test can read back what the backend
    told it about each row's KV length."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs


class TestAiterSpecDecodeSeqlenK(CustomTestCase):
    """EAGLE draft decode appends a token per step, so a draft row is longer than
    seq_lens. The spec page table carries the only exact per-row length; passing
    seq_lens instead hides the drafted tokens, including the query's own key."""

    def setUp(self):
        self.capture = _CaptureUnifiedAttention()
        for target, replacement in (
            ("reset_verify_attn_plan_cache", lambda: None),
            ("create_flashmla_kv_indices_triton", _NoopKernel()),
            ("is_gfx95_supported", lambda: False),
            ("unified_attention", self.capture),
        ):
            patcher = patch.object(aiter_backend, target, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _seqused_k(self, backend, forward_batch):
        q = torch.zeros(_BATCH_SIZE, _HEAD_DIM)
        backend.forward_decode(
            q, None, None, _layer(), forward_batch, save_kv_cache=False
        )
        return self.capture.kwargs["seqused_k"].tolist()

    def test_eager_metadata_lengths_come_from_the_spec_page_table(self):
        for step in range(3):
            backend = _unified_decode_backend()
            backend.init_forward_metadata(_decode_batch(_spec_info(step)))
            self.assertEqual(
                backend.forward_metadata.decode_seqlen_k.tolist(),
                [s + step + 1 for s in _SEQ_LENS],
                f"step {step}",
            )

    def test_graph_metadata_lengths_reuse_captured_storage(self):
        backend = _unified_decode_backend()
        backend.num_draft_tokens = None
        backend.skip_prefill = True
        backend.use_aiter_triton_mla_decode = False
        backend.muse_reuse_verify_lengths = False
        backend.init_cuda_graph_state(_BATCH_SIZE, _BATCH_SIZE)

        captured = None
        for step in range(3):
            backend.init_forward_metadata_out_graph(
                _decode_batch(_spec_info(step)), in_capture=(step == 0)
            )
            seqlen_k = backend.forward_metadata.decode_seqlen_k
            self.assertEqual(
                seqlen_k.tolist(), [s + step + 1 for s in _SEQ_LENS], f"step {step}"
            )
            if captured is None:
                captured = seqlen_k
            # A replay cannot rebind the kernel's argument, so every step has to
            # land in the storage the capture saw.
            self.assertEqual(seqlen_k.data_ptr(), captured.data_ptr())

    def test_forward_decode_passes_spec_lengths_to_the_kernel(self):
        step = 2
        backend = _unified_decode_backend()
        forward_batch = _decode_batch(_spec_info(step))
        backend.init_forward_metadata(forward_batch)
        self.assertEqual(
            self._seqused_k(backend, forward_batch),
            [s + step + 1 for s in _SEQ_LENS],
        )

    def test_forward_decode_without_spec_passes_seq_lens(self):
        backend = _unified_decode_backend()
        forward_batch = _decode_batch(None)
        backend.init_forward_metadata(forward_batch)
        self.assertEqual(self._seqused_k(backend, forward_batch), _SEQ_LENS)


if __name__ == "__main__":
    unittest.main()

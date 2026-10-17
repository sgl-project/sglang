"""FlashInfer prefill with DP-attention row padding."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_HEADS, HEAD_DIM = 2, 4


class _PagedWrapper:
    """Rejects a query whose row count differs from the plan, as FlashInfer does."""

    def __init__(self, planned_rows: int):
        self.planned_rows = planned_rows

    def forward(self, q, kv_cache, **kwargs):
        if q.shape[0] != self.planned_rows:
            raise ValueError(
                f"q.shape[0] ({q.shape[0]}) does not match qo_indptr[-1] "
                f"({self.planned_rows})."
            )
        return torch.ones_like(q)


class _Pool:
    def __init__(self):
        self.written_rows = None

    def get_kv_buffer(self, layer_id):
        return None

    def set_kv_buffer(self, layer, loc, k, v, *scales):
        self.written_rows = k.shape[0]


def _run(forward_mode, extend_seq_lens_cpu, padded_rows, planned_rows):
    backend = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
    backend.num_wrappers = 1
    backend.prefill_uses_dequant_workspace = False
    backend.token_to_kv_pool = _Pool()
    backend.kv_cache_quant_method = SimpleNamespace(needs_global_scale=lambda: True)
    backend.forward_metadata = SimpleNamespace(
        prefill_wrappers=[_PagedWrapper(planned_rows)],
        use_ragged=False,
        multi_item_params=None,
        swa_out_cache_loc=None,
    )
    layer = SimpleNamespace(
        layer_id=0,
        logit_cap=0.0,
        scaling=1.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=-1,
        tp_q_head_num=NUM_HEADS,
        head_dim=HEAD_DIM,
        k_scale_float=None,
        v_scale_float=None,
    )
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        out_cache_loc=torch.zeros(padded_rows, dtype=torch.int64),
        out_cache_loc_is_physical=False,
    )
    qkv = torch.zeros(padded_rows, NUM_HEADS * HEAD_DIM)
    out = backend.forward_extend(qkv, qkv, qkv, layer, forward_batch)
    return out, backend.token_to_kv_pool.written_rows


class TestFlashInferDpPaddedPrefill(unittest.TestCase):
    def test_padded_rows_are_dropped_for_the_kernel_and_restored_as_zeros(self):
        """A 1-token prefill padded to attn_tp_size=2 crashed the scheduler with
        "q.shape[0] (2) does not match qo_indptr[-1] (1)"."""
        out, written_rows = _run(ForwardMode.EXTEND, [1], padded_rows=2, planned_rows=1)
        self.assertEqual(out.shape, (2, NUM_HEADS * HEAD_DIM))
        self.assertTrue(torch.all(out[0] == 1))
        self.assertTrue(torch.all(out[1] == 0))
        self.assertEqual(written_rows, 2)

    def test_target_verify_rows_are_not_trimmed(self):
        out, _ = _run(ForwardMode.TARGET_VERIFY, [1], padded_rows=2, planned_rows=2)
        self.assertTrue(torch.all(out == 1))


if __name__ == "__main__":
    unittest.main()

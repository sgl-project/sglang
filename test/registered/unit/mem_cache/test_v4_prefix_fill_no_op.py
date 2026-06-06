"""Unit tests for the prefix-fill stage of ``cp_allgather_and_save_kv_cache``
(DESIGN_kv_reshard.md §6, Part 4b).

The stage (implemented as the private helper
``_cp_fill_remote_prefix_pool_rows``) must short-circuit (no NCCL launch,
no buffer writes) when:
  - ``cp_owner_per_pages`` is empty/None (no CP-admitted request).
  - ``cp_transient_prefix_rows`` is None (no non-owned prefix positions
    pre-allocated, e.g. fresh prefill / CP-resharding off).
  - All requests have ``prefix_len == 0`` (fresh prefill).

CUDA / NCCL paths are exercised by the E2E test_cp_single_layer
multi-GPU run; these CPU tests only verify the early-return behavior.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils.cp_transient import CpTransientState
from sglang.srt.layers.utils.cp_utils import _cp_fill_remote_prefix_pool_rows
from sglang.test.test_utils import CustomTestCase


class _ChangeTrackingBuffer:
    """Tracks index_put writes so a test can confirm no scatter happened."""

    def __init__(self, shape):
        self._t = torch.zeros(shape, dtype=torch.bfloat16)
        self.write_count = 0

    @property
    def shape(self):
        return self._t.shape

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def device(self):
        return self._t.device

    def __getitem__(self, idx):
        return self._t[idx]

    def __setitem__(self, idx, val):
        self.write_count += 1
        self._t[idx] = val


class _FakePool:
    def __init__(
        self, num_layers: int = 4, size: int = 64, head_num: int = 2, head_dim: int = 8
    ):
        self.k_buffers = [
            _ChangeTrackingBuffer((size, head_num, head_dim)) for _ in range(num_layers)
        ]
        self.v_buffers = [
            _ChangeTrackingBuffer((size, head_num, head_dim)) for _ in range(num_layers)
        ]

    def get_kv_buffer(self, layer_id: int):
        return self.k_buffers[layer_id], self.v_buffers[layer_id]


def _make_fb_for_prefix_fill(
    cp_owner_per_pages,
    extend_prefix_lens_cpu,
    req_pool_indices,
    cp_transient_prefix_rows,
):
    req_to_token = torch.zeros((8, 32), dtype=torch.int64)
    pool = SimpleNamespace(req_to_token=req_to_token)
    cp_transient = (
        CpTransientState(
            owner_per_pages=cp_owner_per_pages,
            prefix_rows=cp_transient_prefix_rows,
        )
        if cp_owner_per_pages is not None or cp_transient_prefix_rows is not None
        else None
    )
    return SimpleNamespace(
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        req_pool_indices=torch.tensor(req_pool_indices, dtype=torch.int64),
        req_to_token_pool=pool,
        token_to_kv_pool=_FakePool(),
        cp_transient=cp_transient,
    )


class _StubLayer:
    layer_id = 0
    k_scale = None
    v_scale = None
    is_cross_attention = False


class TestPrefixFillNoOp(CustomTestCase):
    def test_skips_when_cp_owner_per_pages_is_none(self):
        fb = _make_fb_for_prefix_fill(
            cp_owner_per_pages=None,
            extend_prefix_lens_cpu=[4],
            req_pool_indices=[0],
            cp_transient_prefix_rows=torch.tensor([100], dtype=torch.int64),
        )
        _cp_fill_remote_prefix_pool_rows(fb, _StubLayer(), cp_size=2)
        self.assertEqual(fb.token_to_kv_pool.k_buffers[0].write_count, 0)
        self.assertEqual(fb.token_to_kv_pool.v_buffers[0].write_count, 0)

    def test_skips_when_cp_owner_per_pages_all_none(self):
        # No CP-admitted request in the batch.
        fb = _make_fb_for_prefix_fill(
            cp_owner_per_pages=[None, None],
            extend_prefix_lens_cpu=[4, 4],
            req_pool_indices=[0, 1],
            cp_transient_prefix_rows=None,
        )
        _cp_fill_remote_prefix_pool_rows(fb, _StubLayer(), cp_size=2)
        self.assertEqual(fb.token_to_kv_pool.k_buffers[0].write_count, 0)

    def test_skips_when_no_prefix_transient_rows(self):
        # cp_owner_per_pages is set, but no prefix-range transient rows
        # were pre-allocated (e.g. fresh prefill).
        fb = _make_fb_for_prefix_fill(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=[4],
            req_pool_indices=[3],
            cp_transient_prefix_rows=None,
        )
        _cp_fill_remote_prefix_pool_rows(fb, _StubLayer(), cp_size=2)
        self.assertEqual(fb.token_to_kv_pool.k_buffers[0].write_count, 0)

    def test_skips_when_all_prefix_lens_zero(self):
        # CP-admitted request but prefix is empty -> nothing to gather.
        fb = _make_fb_for_prefix_fill(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=[0],
            req_pool_indices=[3],
            cp_transient_prefix_rows=torch.tensor([100], dtype=torch.int64),
        )
        _cp_fill_remote_prefix_pool_rows(fb, _StubLayer(), cp_size=2)
        self.assertEqual(fb.token_to_kv_pool.k_buffers[0].write_count, 0)

    def test_skips_when_prefix_lens_cpu_is_none(self):
        fb = _make_fb_for_prefix_fill(
            cp_owner_per_pages=[torch.tensor([0, 1], dtype=torch.int8)],
            extend_prefix_lens_cpu=None,
            req_pool_indices=[3],
            cp_transient_prefix_rows=torch.tensor([100], dtype=torch.int64),
        )
        _cp_fill_remote_prefix_pool_rows(fb, _StubLayer(), cp_size=2)
        self.assertEqual(fb.token_to_kv_pool.k_buffers[0].write_count, 0)


if __name__ == "__main__":
    unittest.main()

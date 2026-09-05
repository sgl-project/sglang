# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV-cache row addressing must come from tensor strides, not shapes.

A per-layer KV buffer may be a strided view into a larger buffer (slot stride
larger than ``head_num * head_dim``). Every helper that computes a slot address
must then use ``stride(0)``; deriving it from ``prod(shape[1:])`` silently
addresses the wrong slot, and ``.contiguous()`` silently snapshots instead of
aliasing the live pool. These tests pin the stride-derived behaviour on views
that differ from contiguous buffers.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_kv_stride_hygiene.py -v
"""

import unittest

import torch

from sglang.kernels.ops.attention.utils import canonicalize_stride
from sglang.kernels.ops.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.pool_patcher.buffer_alloc import make_row_source
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _strided_rows(num_rows: int, row_elems: int, slot_elems: int, dtype, *, offset=0):
    """A [num_rows, row_elems] view whose row stride is ``slot_elems`` (> row_elems)."""
    backing = torch.zeros(num_rows * slot_elems + offset, dtype=dtype)
    view = backing.as_strided((num_rows, row_elems), (slot_elems, 1), offset)
    assert not view.is_contiguous()
    return backing, view


class TestCanonicalizeStride(unittest.TestCase):
    def test_degenerate_head_dim_is_fixed(self):
        t = torch.empty(4 * 64 * 128).as_strided((4, 1, 64, 128), (8192, 128, 128, 1))
        out = canonicalize_stride(t)
        self.assertEqual(tuple(out.stride()), (8192, 8192, 128, 1))
        self.assertEqual(out.data_ptr(), t.data_ptr())

    def test_strided_view_without_degenerate_dims_is_untouched(self):
        ps, H, D, E = 4, 8, 128, 8 * 128 + 512  # slot stride wider than a row
        t = torch.empty(3 * ps * E).as_strided((3, ps, H, D), (ps * E, E, D, 1))
        self.assertIs(canonicalize_stride(t), t)

    def test_strided_view_keeps_non_degenerate_strides(self):
        # Degenerate head dim on a strided view: only that stride is rewritten.
        ps, D, E = 4, 128, 2048
        t = torch.empty(3 * ps * E).as_strided((3, 1, ps, D), (ps * E, E, E, 1))
        out = canonicalize_stride(t)
        self.assertEqual(tuple(out.stride()), (ps * E, ps * E, E, 1))
        for p in range(3):
            for s in range(ps):
                self.assertEqual(out[p, 0, s].data_ptr(), t[p, 0, s].data_ptr())

    def test_single_head_with_distinct_strides_is_untouched(self):
        ps, D, E = 4, 128, 2048
        t = torch.empty(3 * ps * E).as_strided((3, ps, 1, D), (ps * E, E, D, 1))
        self.assertIs(canonicalize_stride(t), t)


class TestRealKvSourceStrides(unittest.TestCase):
    def test_accepts_strided_rows(self):
        _, view = _strided_rows(8, 64, 96, torch.uint8)
        src = RealKvSource(
            tensor=view, page_size=1, num_bytes_per_token=64, read_bytes=16
        )
        self.assertEqual(src.tensor.stride(0), 96)

    def test_rejects_unaligned_row_stride(self):
        _, view = _strided_rows(8, 64, 72, torch.uint8)
        with self.assertRaisesRegex(ValueError, "row stride must be a multiple of 16"):
            RealKvSource(
                tensor=view, page_size=1, num_bytes_per_token=64, read_bytes=16
            )


class TestMakeRowSourceAliases(unittest.TestCase):
    def test_strided_layer_view_is_aliased_not_copied(self):
        N, H, D, E = 6, 2, 16, 2 * 16 + 32  # bf16 elems; slot stride wider than a row
        backing = torch.zeros(N * E, dtype=torch.bfloat16)
        layer = backing.as_strided((N, H, D), (E, D, 1))
        (src,) = make_row_source(layer_buffer=layer, read_bytes=16)
        self.assertEqual(src.tensor.data_ptr(), layer.data_ptr())
        self.assertEqual(tuple(src.tensor.shape), (N, H * D * 2))
        self.assertEqual(src.tensor.stride(0), E * 2)
        layer[3, 1, 0] = 1.0  # a later pool write must be visible to the canary
        written = src.tensor[3, D * 2 : D * 2 + 2].view(torch.bfloat16)[0]
        self.assertNotEqual(int(written), 0)

    def test_non_viewable_layout_raises_instead_of_snapshotting(self):
        layer = torch.zeros(4, 16, 2, dtype=torch.bfloat16).transpose(1, 2)
        with self.assertRaises(RuntimeError):
            make_row_source(layer_buffer=layer, read_bytes=16)


class TestDataStridesFollowViews(unittest.TestCase):
    def test_data_strides_use_stride0(self):
        # K and V of both layers share one slot of E elements.
        L, N, H, D, E = 2, 5, 2, 16, 2 * (2 * 16) + 64
        itemsize = 2
        backing = torch.zeros(N * E, dtype=torch.float16)
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        pool.device = "cpu"
        pool.k_buffer = [
            backing.as_strided((N, H, D), (E, D, 1), l * 2 * H * D) for l in range(L)
        ]
        pool.v_buffer = [
            backing.as_strided((N, H, D), (E, D, 1), (l * 2 + 1) * H * D)
            for l in range(L)
        ]
        pool._init_data_ptrs_and_strides()
        self.assertEqual(pool.data_strides.tolist(), [E * itemsize] * (2 * L))
        self.assertNotEqual(E * itemsize, H * D * itemsize)
        self.assertEqual(
            pool.data_ptrs.tolist(),
            [x.data_ptr() for x in pool.k_buffer + pool.v_buffer],
        )


if __name__ == "__main__":
    unittest.main()

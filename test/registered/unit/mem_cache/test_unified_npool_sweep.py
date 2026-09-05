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
"""N-sub-pool construction sweep for ``UnifiedKVPool``.

The pool accepts N sub-pool specs -- exactly one grow-up END, exactly one
grow-down END, and >= 0 "float" MIDDLE pools between their frontiers -- and
sorts them into the canonical chain order
``[up end, floats (input order), down end]``. Every sub-pool view spans the
whole buffer at anchor 0; keeping the bands disjoint is the allocators' job.

Pure CPU geometry -- no allocator, no GPU.
"""

import unittest

import torch

from sglang.srt.mem_cache.unified_memory_pool import (
    MambaSubPoolSpec,
    MHASubPoolSpec,
    MLASubPoolSpec,
    UnifiedKVPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

# Hermetic convention of this directory's pool tests: plain unittest.TestCase,
# only ci_register imported (no heavy sglang.test.test_utils chain).
register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_DEV = "cpu"


def _mha(
    name: str,
    grow_direction: str,
    *,
    layer_num: int = 2,
    head_num: int = 2,
    head_dim: int = 8,
) -> MHASubPoolSpec:
    return MHASubPoolSpec(
        name=name,
        layer_num=layer_num,
        grow_direction=grow_direction,
        head_num=head_num,
        head_dim=head_dim,
        store_dtype=torch.bfloat16,
    )


def _mla(name: str, grow_direction: str, *, layer_num: int = 2) -> MLASubPoolSpec:
    return MLASubPoolSpec(
        name=name,
        layer_num=layer_num,
        grow_direction=grow_direction,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        store_dtype=torch.bfloat16,
    )


def _mamba(name: str, grow_direction: str, *, layer_num: int = 2) -> MambaSubPoolSpec:
    return MambaSubPoolSpec(
        name=name,
        layer_num=layer_num,
        grow_direction=grow_direction,
        conv_state_shapes=((4, 6),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(2, 4, 4),
        temporal_dtype=torch.float32,
    )


def _make_pool(specs, *, total_bytes: int = 1 << 20, page_size: int = 1):
    return UnifiedKVPool(
        total_bytes=total_bytes,
        sub_pool_specs=specs,
        device=_DEV,
        enable_memory_saver=False,
        page_size=page_size,
    )


def _chain_names(pool: UnifiedKVPool):
    return [s.name for s in pool.sub_pool_specs]


class TestNPoolCanonicalOrder(unittest.TestCase):
    def test_chain_order_is_canonical(self):
        """Ends canonical (up first, down last) whatever the input order;
        floats keep INPUT order between them, on every cache-spec kind."""
        for specs, expect in (
            ([_mha("full", "down"), _mamba("mamba", "up")], ["mamba", "full"]),
            ([_mamba("mamba", "up"), _mha("full", "down")], ["mamba", "full"]),
            (
                [_mha("full", "down"), _mha("swa", "float"), _mamba("conv", "up")],
                ["conv", "swa", "full"],
            ),
            (
                [_mha("swa", "float"), _mamba("conv", "up"), _mha("full", "down")],
                ["conv", "swa", "full"],
            ),
            (
                [_mamba("conv", "up"), _mha("full", "down"), _mha("swa", "float")],
                ["conv", "swa", "full"],
            ),
            (
                [
                    _mha("full", "down"),
                    _mha("f1", "float"),
                    _mamba("state", "up"),
                    _mha("f0", "float", layer_num=1),
                ],
                ["state", "f1", "f0", "full"],
            ),
            (
                [
                    _mamba("state", "up"),
                    _mha("f_mha", "float"),
                    _mla("f_mla", "float", layer_num=1),
                    _mamba("f_mamba", "float", layer_num=1),
                    _mha("full", "down"),
                ],
                ["state", "f_mha", "f_mla", "f_mamba", "full"],
            ),
        ):
            with self.subTest(inputs=[s.name for s in specs]):
                self.assertEqual(_chain_names(_make_pool(specs)), expect)

    def test_by_name_geometry_independent_of_n(self):
        two = _make_pool([_mha("full", "down"), _mamba("mamba", "up")])
        three = _make_pool(
            [_mha("full", "down"), _mha("swa", "float"), _mamba("mamba", "up")]
        )
        for name in ("full", "mamba"):
            self.assertEqual(
                two.max_slots(name),
                two.total_bytes // two.spec(name).entry_bytes(),
            )
            self.assertEqual(two.max_slots(name), three.max_slots(name))


class TestNPoolValidation(unittest.TestCase):
    def test_duplicate_names_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unique"):
            _make_pool([_mha("x", "down"), _mamba("x", "up")])

    def test_fewer_than_two_specs_rejected(self):
        with self.assertRaisesRegex(AssertionError, ">= 2 sub-pools"):
            _make_pool([_mha("full", "down")])

    def test_end_direction_counts_rejected(self):
        """Exactly one grow-up END and one grow-down END, no more, no less."""
        for case, specs in (
            ("two_ups", [_mha("a", "up"), _mamba("b", "up")]),
            ("missing_down", [_mha("a", "up"), _mha("b", "float")]),
            (
                "missing_up",
                [_mha("a", "down"), _mha("b", "float"), _mha("c", "float")],
            ),
        ):
            with self.subTest(case=case):
                with self.assertRaisesRegex(AssertionError, "exactly one grow-up"):
                    _make_pool(specs)

    def test_bogus_direction_rejected_at_spec_level(self):
        with self.assertRaisesRegex(AssertionError, "grow_direction"):
            _mha("a", "sideways")


class TestReservedFloorWithFloats(unittest.TestCase):
    def test_float_page_envelope_extends_the_sink(self):
        # The float MHA has the largest page-0 envelope; every pool's
        # min_slot_index must clear it (mamba is page_size=1 and excluded from
        # the page-aware term, but still must clear the byte floor).
        page_size = 4
        big_float = _mha("swa", "float", layer_num=8, head_num=4, head_dim=32)
        specs = [_mamba("state", "up"), big_float, _mha("full", "down")]
        pool = _make_pool(specs, total_bytes=1 << 22, page_size=page_size)
        floor = max(
            max(s.entry_bytes() for s in specs),
            page_size * big_float.entry_bytes(),
            page_size * specs[2].entry_bytes(),
        )
        for s in specs:
            e = s.entry_bytes()
            self.assertEqual(pool.min_slot_index(s.name), (floor + e - 1) // e)

    def test_too_small_buffer_fails_loud(self):
        # 2048 B with page_size=16 and 128 B/entry MHA specs: the page-0 sink
        # (16*128 = 2048 B) consumes the whole buffer, leaving no slot.
        with self.assertRaisesRegex(RuntimeError, "no room"):
            _make_pool(
                [_mamba("state", "up"), _mha("swa", "float"), _mha("full", "down")],
                total_bytes=2048,
                page_size=16,
            )


class TestFloatViews(unittest.TestCase):
    def test_float_mha_views_shape_and_roundtrip(self):
        page_size = 2
        spec = _mha("swa", "float", layer_num=3, head_num=2, head_dim=8)
        pool = _make_pool(
            [_mamba("state", "up"), spec, _mha("full", "down")],
            total_bytes=1 << 20,
            page_size=page_size,
        )
        k_views, v_views = pool.mha_views_for("swa")
        self.assertEqual(len(k_views), spec.layer_num)
        self.assertEqual(len(v_views), spec.layer_num)
        num_pages = pool.max_slots("swa") // page_size
        blocks = 2 * spec.layer_num  # K at block 2l, V at 2l+1
        n_rows = num_pages * blocks * page_size
        for k in (*k_views, *v_views):
            # Stock 3-D per-layer MHA signature; the row index is the
            # kernel-facing id, each view's storage_offset folding in its block
            # origin (see `build_mha_views`).
            self.assertEqual(tuple(k.shape), (n_rows, spec.head_num, spec.head_dim))
        # Round-trip: a float view is a real strided window into _raw.
        slot = pool.min_slot_index("swa")
        row = (slot // page_size) * (page_size * blocks) + slot % page_size
        pattern = (
            torch.arange(spec.head_num * spec.head_dim, dtype=torch.float32)
            .reshape(spec.head_num, spec.head_dim)
            .to(torch.bfloat16)
        )
        k_views[1][row] = pattern
        torch.testing.assert_close(k_views[1][row], pattern)


if __name__ == "__main__":
    unittest.main()

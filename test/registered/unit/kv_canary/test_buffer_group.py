"""Tests for sglang.srt.kv_canary.buffer_group: PoolKind enum and CanaryBufferGroup."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.test.test_utils import CustomTestCase

_DEVICE = torch.device("cpu")

# Minimal canary buffer shape: [num_slots, canary_bytes]
_SHAPE = (10, 16)


def _buf():
    return torch.zeros(*_SHAPE, dtype=torch.uint8, device=_DEVICE)


def _make_group(
    *,
    kind=PoolKind.FULL,
    v_head=None,
    v_tail=None,
    swa_index_lut=None,
    kv_offset=0,
    real_kv_k=(),
    real_kv_v=(),
):
    return CanaryBufferGroup(
        kind=kind,
        k_head=_buf(),
        k_tail=_buf(),
        v_head=v_head,
        v_tail=v_tail,
        real_kv_sources_k=real_kv_k,
        real_kv_sources_v=real_kv_v,
        swa_index_lut=swa_index_lut,
        kv_token_id_vs_position_offset=kv_offset,
    )


class TestPoolKindEnum(CustomTestCase):
    def test_full_value(self):
        self.assertEqual(int(PoolKind.FULL), 0)

    def test_swa_value(self):
        self.assertEqual(int(PoolKind.SWA), 1)

    def test_members_count(self):
        self.assertEqual(len(list(PoolKind)), 2)

    def test_is_int_enum(self):
        import enum

        self.assertIsInstance(PoolKind.FULL, int)
        self.assertTrue(issubclass(PoolKind, enum.IntEnum))

    def test_full_less_than_swa(self):
        self.assertLess(PoolKind.FULL, PoolKind.SWA)


class TestCanaryBufferGroupHasVHalf(CustomTestCase):
    def test_has_v_half_true_when_v_head_set(self):
        bg = _make_group(v_head=_buf(), v_tail=_buf())
        self.assertTrue(bg.has_v_half)

    def test_has_v_half_false_when_v_head_none(self):
        bg = _make_group(v_head=None, v_tail=None)
        self.assertFalse(bg.has_v_half)

    def test_has_v_half_false_returns_bool(self):
        bg = _make_group()
        self.assertIsInstance(bg.has_v_half, bool)

    def test_has_v_half_true_returns_bool(self):
        bg = _make_group(v_head=_buf(), v_tail=_buf())
        self.assertIsInstance(bg.has_v_half, bool)


class TestCanaryBufferGroupFields(CustomTestCase):
    def test_kind_full(self):
        bg = _make_group(kind=PoolKind.FULL)
        self.assertEqual(bg.kind, PoolKind.FULL)

    def test_kind_swa(self):
        bg = _make_group(kind=PoolKind.SWA)
        self.assertEqual(bg.kind, PoolKind.SWA)

    def test_k_head_stored(self):
        k = _buf()
        bg = CanaryBufferGroup(
            kind=PoolKind.FULL,
            k_head=k,
            k_tail=_buf(),
            v_head=None,
            v_tail=None,
            real_kv_sources_k=(),
            real_kv_sources_v=(),
            swa_index_lut=None,
            kv_token_id_vs_position_offset=0,
        )
        self.assertIs(bg.k_head, k)

    def test_k_tail_stored(self):
        t = _buf()
        bg = CanaryBufferGroup(
            kind=PoolKind.FULL,
            k_head=_buf(),
            k_tail=t,
            v_head=None,
            v_tail=None,
            real_kv_sources_k=(),
            real_kv_sources_v=(),
            swa_index_lut=None,
            kv_token_id_vs_position_offset=0,
        )
        self.assertIs(bg.k_tail, t)

    def test_v_head_none_when_not_set(self):
        bg = _make_group()
        self.assertIsNone(bg.v_head)

    def test_v_tail_none_when_not_set(self):
        bg = _make_group()
        self.assertIsNone(bg.v_tail)

    def test_real_kv_sources_k_empty_tuple(self):
        bg = _make_group()
        self.assertEqual(bg.real_kv_sources_k, ())

    def test_real_kv_sources_v_empty_tuple(self):
        bg = _make_group()
        self.assertEqual(bg.real_kv_sources_v, ())

    def test_swa_index_lut_none_for_full_group(self):
        bg = _make_group(kind=PoolKind.FULL, swa_index_lut=None)
        self.assertIsNone(bg.swa_index_lut)

    def test_swa_index_lut_stored(self):
        lut = torch.zeros(100, dtype=torch.int64)
        bg = _make_group(kind=PoolKind.SWA, swa_index_lut=lut)
        self.assertIs(bg.swa_index_lut, lut)

    def test_kv_token_id_vs_position_offset_zero(self):
        bg = _make_group(kv_offset=0)
        self.assertEqual(bg.kv_token_id_vs_position_offset, 0)

    def test_kv_token_id_vs_position_offset_one(self):
        bg = _make_group(kv_offset=1)
        self.assertEqual(bg.kv_token_id_vs_position_offset, 1)

    def test_frozen_dataclass_rejects_mutation(self):
        bg = _make_group()
        with self.assertRaises((AttributeError, TypeError)):
            bg.kind = PoolKind.SWA


if __name__ == "__main__":
    unittest.main(verbosity=3)

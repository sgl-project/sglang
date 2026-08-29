"""SWA ring scatter tests for the two-pool fp8 unified_kv layout.

``store_swa_into_unified`` writes one latent row per token. Under
SGLANG_DSV4_UNIFIED_KV_FP8 that row is split over a packed fp8 nope pool and a
bf16 rope pool, so the property these tests pin is that the ring row index --
derived from state_slot/positions alone -- stays identical to the bf16 layout's
and identical between the two pools.

Needs a GPU for triton but only a few MB of it, so it runs fine next to a busy
card.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV4_FP8_NOPE_ROW_BYTES
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the scatter itself is a plain row move, but the layout it pins is gfx95-only, so
# run it where the feature lives rather than on the default mi300 runner
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

# 448 values + 14 E8M0 scales + 50 pad, in bytes
NOPE_ROW_BYTES = DSV4_FP8_NOPE_ROW_BYTES
ROPE_DIM = 64
# V4-Pro latent, in elements -- same number as NOPE_ROW_BYTES, different unit
BF16_LATENT = 448 + ROPE_DIM

RING_STRIDE = 16
WIN = 8
N_PAGES = 64


def _inputs(n_rows=12):
    """state_slot/positions whose ring rows are all distinct, so a row's writer is unambiguous"""
    state_slot = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3][:n_rows],
        device=DEVICE,
        dtype=torch.int32,
    )
    positions = torch.tensor(
        [0, 1, 2, 16, 17, 18, 32, 33, 34, 48, 49, 50][:n_rows],
        device=DEVICE,
        dtype=torch.int32,
    )
    return state_slot, positions


def _expected_rows(state_slot, positions, final_pos=None):
    loc = state_slot.long() * RING_STRIDE + positions.long() % RING_STRIDE
    if final_pos is None:
        keep = torch.ones_like(loc, dtype=torch.bool)
    else:
        keep = positions.long() > final_pos.long() - WIN
    return loc, keep


def _packed_nope(n_rows):
    """random packed fp8 rows; byte 0 is forced nonzero so a written row is detectable"""
    raw = torch.randint(
        0, 256, (n_rows, NOPE_ROW_BYTES), device=DEVICE, dtype=torch.uint8
    )
    raw[:, 0] = torch.arange(1, n_rows + 1, device=DEVICE, dtype=torch.uint8)
    return raw.view(torch.float8_e4m3fn), raw


def _bf16_rope(n_rows):
    rope = torch.randn(n_rows, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
    rope[:, 0] = torch.arange(1, n_rows + 1, device=DEVICE, dtype=torch.bfloat16)
    return rope.contiguous()


def _store(kv, pool, state_slot, positions, final_pos=None, **kw):
    runtime.store_swa_into_unified(
        kv=kv,
        state_slot=state_slot,
        positions=positions,
        unified_kv=pool,
        win=WIN,
        ring_stride=RING_STRIDE,
        final_pos=final_pos,
        **kw,
    )


class TestUnifiedFp8SwaScatter(CustomTestCase):
    def setUp(self):
        torch.manual_seed(20)
        self.state_slot, self.positions = _inputs()
        self.n_rows = self.state_slot.shape[0]

    def _run_two_pool(self, final_pos=None):
        kv_nope, nope_bytes = _packed_nope(self.n_rows)
        kv_rope = _bf16_rope(self.n_rows)
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        pool_rope = torch.zeros(N_PAGES, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
        _store(
            kv_nope,
            pool_nope,
            self.state_slot,
            self.positions,
            final_pos=final_pos,
            kv_rope=kv_rope,
            unified_kv_rope=pool_rope,
        )
        return pool_nope, pool_rope, nope_bytes, kv_rope

    def _run_bf16(self, final_pos=None):
        kv = torch.randn(
            self.n_rows, BF16_LATENT, device=DEVICE, dtype=torch.bfloat16
        ).contiguous()
        kv[:, 0] = torch.arange(1, self.n_rows + 1, device=DEVICE, dtype=torch.bfloat16)
        pool = torch.zeros(N_PAGES, BF16_LATENT, device=DEVICE, dtype=torch.bfloat16)
        _store(kv, pool, self.state_slot, self.positions, final_pos=final_pos)
        return pool, kv

    def test_bf16_single_pool_unchanged(self):
        """the bf16 path still writes exactly the expected ring rows"""
        pool, kv = self._run_bf16()
        loc, keep = _expected_rows(self.state_slot, self.positions)
        expected = torch.zeros_like(pool)
        expected[loc[keep]] = kv[keep]
        self.assertTrue(torch.equal(pool, expected))

    def test_two_pool_bytes_exact(self):
        """each pool gets its half verbatim -- nope byte-for-byte, rope bit-for-bit"""
        pool_nope, pool_rope, nope_bytes, kv_rope = self._run_two_pool()
        loc, keep = _expected_rows(self.state_slot, self.positions)

        exp_nope = torch.zeros_like(pool_nope).view(torch.uint8)
        exp_nope[loc[keep]] = nope_bytes[keep]
        self.assertTrue(torch.equal(pool_nope.view(torch.uint8), exp_nope))

        exp_rope = torch.zeros_like(pool_rope)
        exp_rope[loc[keep]] = kv_rope[keep]
        self.assertTrue(torch.equal(pool_rope, exp_rope))

    def test_two_pool_rows_match_bf16(self):
        """same state_slot/positions -> same ring rows as bf16, and the same in both pools"""
        pool_nope, pool_rope, _, _ = self._run_two_pool()
        pool_bf16, _ = self._run_bf16()

        rows_nope = (pool_nope.view(torch.uint8) != 0).any(dim=1)
        rows_rope = (pool_rope != 0).any(dim=1)
        rows_bf16 = (pool_bf16 != 0).any(dim=1)

        self.assertTrue(torch.equal(rows_nope, rows_bf16))
        self.assertTrue(torch.equal(rows_rope, rows_bf16))
        self.assertEqual(int(rows_bf16.sum()), self.n_rows)

    def test_final_pos_skips_both_pools(self):
        """tokens already outside the window are skipped in nope and rope alike"""
        # positions[t] <= final_pos[t] - WIN skips; give the first half a far
        # final_pos and the second half its own position
        final_pos = self.positions.clone()
        final_pos[: self.n_rows // 2] = self.positions.max() + WIN
        pool_nope, pool_rope, nope_bytes, kv_rope = self._run_two_pool(
            final_pos=final_pos
        )
        loc, keep = _expected_rows(self.state_slot, self.positions, final_pos)
        self.assertTrue(bool((~keep).any()), "test would be vacuous without a skip")

        rows_nope = (pool_nope.view(torch.uint8) != 0).any(dim=1)
        rows_rope = (pool_rope != 0).any(dim=1)
        expected_rows = torch.zeros(N_PAGES, device=DEVICE, dtype=torch.bool)
        expected_rows[loc[keep]] = True
        self.assertTrue(torch.equal(rows_nope, expected_rows))
        self.assertTrue(torch.equal(rows_rope, expected_rows))

    def test_rope_tensor_and_pool_come_together(self):
        kv_nope, _ = _packed_nope(self.n_rows)
        kv_rope = _bf16_rope(self.n_rows)
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        pool_rope = torch.zeros(N_PAGES, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            _store(
                kv_nope,
                pool_nope,
                self.state_slot,
                self.positions,
                kv_rope=kv_rope,
            )
        with self.assertRaises(AssertionError):
            _store(
                kv_nope,
                pool_nope,
                self.state_slot,
                self.positions,
                unified_kv_rope=pool_rope,
            )

    def test_short_rope_pool_rejected(self):
        """the kernel doesn't bound-check the ring row, so a rope pool with fewer
        rows than the nope pool writes into whatever tensor follows it"""
        kv_nope, _ = _packed_nope(self.n_rows)
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        # ring rows reach state_slot 3 -> row 48, well past this
        pool_rope = torch.zeros(8, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            _store(
                kv_nope,
                pool_nope,
                self.state_slot,
                self.positions,
                kv_rope=_bf16_rope(self.n_rows),
                unified_kv_rope=pool_rope,
            )

    def test_rope_row_width_mismatch_rejected(self):
        """row width is read off src, so a wider pool would place row i at i * D"""
        kv_nope, _ = _packed_nope(self.n_rows)
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        pool_rope = torch.zeros(
            N_PAGES, ROPE_DIM * 2, device=DEVICE, dtype=torch.bfloat16
        )
        with self.assertRaises(AssertionError):
            _store(
                kv_nope,
                pool_nope,
                self.state_slot,
                self.positions,
                kv_rope=_bf16_rope(self.n_rows),
                unified_kv_rope=pool_rope,
            )

    def test_dtype_mismatch_rejected(self):
        """a bf16 row must not land in an fp8 pool (the DSpark-under-fp8 case)"""
        kv = torch.randn(
            self.n_rows, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.bfloat16
        )
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        with self.assertRaises(AssertionError):
            _store(kv, pool_nope, self.state_slot, self.positions)

    def test_empty_batch_is_a_noop(self):
        empty_slot = torch.zeros(0, device=DEVICE, dtype=torch.int32)
        kv_nope, _ = _packed_nope(0)
        pool_nope = torch.zeros(
            N_PAGES, NOPE_ROW_BYTES, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        pool_rope = torch.zeros(N_PAGES, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
        _store(
            kv_nope,
            pool_nope,
            empty_slot,
            empty_slot,
            kv_rope=_bf16_rope(0),
            unified_kv_rope=pool_rope,
        )
        self.assertEqual(int((pool_nope.view(torch.uint8) != 0).sum()), 0)
        self.assertEqual(int((pool_rope != 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The backend's own prefill wiring: what reaches the reader, and what lands in the ring.

The pieces on either side of this are covered elsewhere -- the scatter primitive by
test_dsv4_unified_fp8_scatter, the model->backend kwargs by the q_pair test -- but
the middle, where _forward_unified_kv picks the fp8 arm and hands the packed pair to
both attention and the ring write, had nothing running through it.

Losing the rope half of that write is silent: the nope pool gets this chunk's rows,
the rope pool keeps stale ones, and later chunks plus decode read a wrong RoPE with
no crash and no NaN. So these run the real store against real (small) pools and pin
that both pools got written, on the same ring row. The attention reader is stubbed:
it is covered by test_dsv4_unified_fp8_prefill, and the store is what is at stake.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.deepseek_v4_backend_hip_radix as backend_mod
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV4_FP8_NOPE_ROW_BYTES
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the store is a plain row move, but the two-pool layout it pins is gfx95-only, so
# run it where the feature lives rather than on the default mi300 runner
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

NOPE_ROW_BYTES = DSV4_FP8_NOPE_ROW_BYTES
ROPE_DIM = 64
V_HEAD_DIM = 512
NUM_HEADS = 16

WIN = 8
RING_STRIDE = 8
SWA_PAGES = 24  # ring rows are state_slot * RING_STRIDE + pos % RING_STRIDE, so < 24
POOL_ROWS = 32

# distinctive fill, so "the store never ran here" and "the store wrote zeros" are
# different failures
NOPE_SENTINEL = 0xEE
ROPE_SENTINEL = -7.0

# two requests on ring slots 1 and 2, three tokens each at positions 0..2
STATE_SLOT = [1, 1, 1, 2, 2, 2]
POSITIONS = [0, 1, 2, 0, 1, 2]
CU_Q = [0, 0, 0, 3, 3, 3]
EXPECTED_ROWS = [8, 9, 10, 16, 17, 18]

_needs_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the two-pool fp8 layout is gfx95-only",
)


def _ints(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE).contiguous()


class _Pool:
    """Just the surface _forward_unified_kv touches."""

    def __init__(self, fp8):
        self.unified_swa_window = WIN
        self.unified_swa_ring_size = RING_STRIDE
        self.unified_swa_pages = SWA_PAGES
        if fp8:
            self.nope = torch.full(
                (POOL_ROWS, NOPE_ROW_BYTES),
                NOPE_SENTINEL,
                dtype=torch.uint8,
                device=DEVICE,
            ).view(torch.float8_e4m3fn)
        else:
            self.nope = torch.full(
                (POOL_ROWS, V_HEAD_DIM),
                ROPE_SENTINEL,
                dtype=torch.bfloat16,
                device=DEVICE,
            )
        self.rope = torch.full(
            (POOL_ROWS, ROPE_DIM), ROPE_SENTINEL, dtype=torch.bfloat16, device=DEVICE
        )

    def get_unified_kv(self, layer_id):
        return self.nope

    def get_unified_kv_rope(self, layer_id):
        return self.rope


def _chunk(fp8):
    """This fwd's K, one row per token, every row a different value."""
    tokens = len(STATE_SLOT)
    if fp8:
        rows = torch.arange(1, tokens + 1, dtype=torch.uint8, device=DEVICE)
        nope = rows[:, None].expand(tokens, NOPE_ROW_BYTES).contiguous()
        nope = nope.view(torch.float8_e4m3fn)
    else:
        rows = torch.arange(1, tokens + 1, dtype=torch.bfloat16, device=DEVICE)
        nope = rows[:, None].expand(tokens, V_HEAD_DIM).contiguous()
    rope = (
        torch.arange(1, tokens + 1, dtype=torch.bfloat16, device=DEVICE)[:, None]
        .expand(tokens, ROPE_DIM)
        .contiguous()
    )
    return nope, rope


class TestUnifiedFp8BackendPrefill(CustomTestCase):
    def _run(self, fp8=True, save_kv_cache=True):
        tokens = len(STATE_SLOT)
        pool = _Pool(fp8)
        k_nope, k_rope = _chunk(fp8)
        if fp8:
            q = torch.zeros(
                tokens, NUM_HEADS, NOPE_ROW_BYTES, dtype=torch.uint8, device=DEVICE
            ).view(torch.float8_e4m3fn)
            q_rope = torch.zeros(
                tokens, NUM_HEADS, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE
            )
        else:
            q = torch.zeros(
                tokens, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )
            q_rope, k_rope = None, None

        unified_meta = SimpleNamespace(
            pf_state_slot=_ints(STATE_SLOT),
            pf_chunk_start=_ints([0] * tokens),
            pf_cu_q=_ints(CU_Q),
            pf_final_pos=_ints([max(POSITIONS)] * tokens),
        )
        core_meta = SimpleNamespace(
            unified=unified_meta,
            c128_page_indices=None,
            c4_sparse_page_indices=None,
        )
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            positions=torch.tensor(POSITIONS, dtype=torch.int64, device=DEVICE),
            req_pool_indices=_ints(STATE_SLOT),
        )
        fake_self = SimpleNamespace(
            token_to_kv_pool=pool, softmax_scale=V_HEAD_DIM**-0.5
        )
        reader_calls = []

        def _fake_reader(**kwargs):
            reader_calls.append(kwargs)
            return torch.zeros(
                tokens, NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
            )

        target = "prefill_fp8_2buff" if fp8 else "prefill"
        with (
            patch.object(runtime, target, _fake_reader),
            patch.object(
                backend_mod,
                "get_parallel",
                return_value=SimpleNamespace(attn_cp_size=1, attn_cp_rank=0),
            ),
        ):
            backend_mod.DeepseekV4HipRadixBackend._forward_unified_kv(
                fake_self,
                q=q,
                kv=k_nope,
                layer=SimpleNamespace(layer_id=0, v_head_dim=V_HEAD_DIM),
                forward_batch=forward_batch,
                compress_ratio=0,
                attn_sink=torch.zeros(NUM_HEADS, dtype=torch.float32, device=DEVICE),
                core_attn_metadata=core_meta,
                save_kv_cache=save_kv_cache,
                q_rope=q_rope,
                k_rope=k_rope,
            )
        self.assertEqual(len(reader_calls), 1)
        return pool, k_nope, k_rope, reader_calls[0]

    def _untouched(self):
        return sorted(set(range(POOL_ROWS)) - set(EXPECTED_ROWS))

    @_needs_gfx950
    def test_both_pools_get_this_chunk_on_the_same_ring_row(self):
        """the regression this file exists for: a rope pool left holding stale rows"""
        pool, k_nope, k_rope, _ = self._run()

        for token, row in enumerate(EXPECTED_ROWS):
            self.assertTrue(
                torch.equal(
                    pool.nope[row].view(torch.uint8), k_nope[token].view(torch.uint8)
                ),
                f"nope pool row {row} does not hold token {token}",
            )
            self.assertTrue(
                torch.equal(pool.rope[row], k_rope[token]),
                f"rope pool row {row} does not hold token {token} -- "
                f"got {pool.rope[row][0].item()}, want {k_rope[token][0].item()}",
            )

    @_needs_gfx950
    def test_rows_outside_the_window_are_left_alone(self):
        """both scatters take the same row, so neither may spray past it"""
        pool, _, _, _ = self._run()
        rest = self._untouched()

        self.assertTrue(
            bool((pool.nope[rest].view(torch.uint8) == NOPE_SENTINEL).all())
        )
        self.assertTrue(bool((pool.rope[rest] == ROPE_SENTINEL).all()))

    @_needs_gfx950
    def test_the_reader_gets_the_same_pair_the_ring_write_does(self):
        _, k_nope, k_rope, call = self._run()

        self.assertIs(call["kv_extend"], k_nope)
        self.assertIs(call["kv_extend_rope"], k_rope)
        self.assertIsNotNone(call["unified_kv_rope"])

    @_needs_gfx950
    def test_nothing_is_written_when_the_model_already_stored(self):
        pool, _, _, _ = self._run(save_kv_cache=False)

        self.assertTrue(bool((pool.nope.view(torch.uint8) == NOPE_SENTINEL).all()))
        self.assertTrue(bool((pool.rope == ROPE_SENTINEL).all()))

    @_needs_gfx950
    def test_the_bf16_arm_never_touches_the_rope_pool(self):
        """one pool, one write -- the rope pool only exists under the fp8 layout"""
        pool, k_nope, _, _ = self._run(fp8=False)

        for token, row in enumerate(EXPECTED_ROWS):
            self.assertTrue(torch.equal(pool.nope[row], k_nope[token]))
        self.assertTrue(bool((pool.rope == ROPE_SENTINEL).all()))


if __name__ == "__main__":
    unittest.main()

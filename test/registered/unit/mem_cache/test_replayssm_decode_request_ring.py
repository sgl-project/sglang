"""GDN decode ReplaySSM request-row ring (SGLANG_OPT_REPLAYSSM_DECODE_REQUEST_RING).

The decode ring and its write cursor are sized by request rows, the cursor
restarts in `StateStorage.reset_request_rows` (not on slot acquire or copy), and
the backend indexes it by the same translated request row as the kernel's ring.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.attention_hook import handle_linear_attn_backend
from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage, acquire_rows
from sglang.srt.runtime_context import override_platform
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NUM_REQS = 4
NUM_SLOTS = 12
RING_LEN = 4
LAYERS = [0, 2]
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)


def _gdn_params():
    shape = Mamba2StateShape(
        conv=[(8, 3)],
        temporal=(4, 8, 8),
        intermediate_size=0,
        conv_dim=0,
        ssm_state_size=0,
        num_heads=0,
        head_dim=0,
        state_size=0,
        conv_kernel=0,
        num_k_heads_per_tp=2,
    )
    return Mamba2CacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _pool(*, request_ring, params=None):
    return HybridReqToTokenPool(
        size=NUM_REQS,
        mamba_size=NUM_SLOTS,
        mamba_spec_state_size=NUM_REQS,
        max_context_len=16,
        device="cpu",
        enable_memory_saver=False,
        cache_params=params or _gdn_params(),
        mamba_layer_ids=LAYERS,
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=True,
        linear_replayssm_cache_len=RING_LEN,
        replayssm_request_ring=request_ring,
    )


def _req(rid):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q"),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


def _storage(cls, pool):
    """A storage holding only the allocator view `settle_replay` reads."""
    storage = object.__new__(cls)
    storage._allocator = SimpleNamespace(
        state=pool.state,
        state_ids=lambda sub_pool: IdSpace(key=(sub_pool, None)),
    )
    return storage


class TestDecodeRequestRingPool(CustomTestCase):
    def test_ring_rows_follow_request_rows(self):
        for request_ring, rows in ((False, NUM_SLOTS + 1), (True, NUM_REQS + 1)):
            with self.subTest(request_ring=request_ring):
                mamba_pool = _pool(request_ring=request_ring).mamba_pool
                self.assertEqual(mamba_pool.replayssm_request_ring, request_ring)
                cache = mamba_pool.mamba_cache
                for ring in (cache.replayssm_d, cache.replayssm_k, cache.replayssm_g):
                    self.assertEqual(ring.shape[1], rows)
                self.assertEqual(tuple(mamba_pool.replayssm_write_pos.shape), (rows,))

    def test_kda_decode_ring_stays_per_slot(self):
        mamba_pool = _pool(request_ring=True, params=_kda_params()).mamba_pool
        self.assertFalse(mamba_pool.replayssm_request_ring)
        self.assertEqual(mamba_pool.mamba_cache.replayssm_d.shape[1], NUM_SLOTS + 1)


class TestDecodeRequestRingCursor(CustomTestCase):
    def _bind_one(self, pool):
        req = _req(0)
        rows = acquire_rows(pool, pool.state, [req])
        return req, rows[0], int(req.kv.mamba_pool_idx)

    def test_slot_cursor_restarts_on_acquire(self):
        pool = _pool(request_ring=False)
        write_pos = pool.mamba_pool.replayssm_write_pos
        write_pos.fill_(3)
        _, row, slot = self._bind_one(pool)
        self.assertEqual(int(write_pos[slot]), 0)
        # Row resets leave the per-slot cursor alone.
        write_pos.fill_(3)
        pool.state.reset_request_rows([row], device="cpu")
        self.assertTrue(bool((write_pos == 3).all()))

    def test_row_cursor_restarts_on_row_reset_only(self):
        pool = _pool(request_ring=True)
        write_pos = pool.mamba_pool.replayssm_write_pos
        write_pos.fill_(3)
        _, row, _ = self._bind_one(pool)
        # Acquire runs on the schedule stream and must not touch the row
        # cursor, which the row's previous request may still be advancing.
        self.assertTrue(bool((write_pos == 3).all()))
        pool.state.reset_request_rows([row], device="cpu")
        expected = torch.full_like(write_pos, 3)
        expected[row] = 0
        self.assertTrue(torch.equal(write_pos, expected))

    def test_copy_from_resets_only_slot_cursors(self):
        src = torch.tensor([1, 2])
        dst = torch.tensor([3, 4])
        for request_ring in (False, True):
            with self.subTest(request_ring=request_ring):
                mamba_pool = _pool(request_ring=request_ring).mamba_pool
                write_pos = mamba_pool.replayssm_write_pos
                write_pos.zero_()
                write_pos[dst] = 5
                mamba_pool.copy_from(src, dst)
                # A slot cursor follows the copied checkpoint (empty ring); a
                # row cursor belongs to a request, not to either slot.
                self.assertEqual(
                    write_pos[dst].tolist(), [5, 5] if request_ring else [0, 0]
                )

    def test_settle_and_restart_replay(self):
        for cls in (StaticKVStorage, KVStorage):
            for request_ring in (False, True):
                with self.subTest(storage=cls.__name__, request_ring=request_ring):
                    pool = _pool(request_ring=request_ring)
                    write_pos = pool.mamba_pool.replayssm_write_pos
                    storage = _storage(cls, pool)
                    row, slot = 2, 3
                    write_pos.zero_()
                    write_pos[row if request_ring else slot] = 6
                    lag = storage.settle_replay("mamba", torch.tensor(slot), row=row)
                    self.assertEqual(lag, 6)
                    self.assertTrue(bool((write_pos == 0).all()))
                    # A loaded slot carries no row cursor to restart.
                    write_pos.fill_(1)
                    storage.restart_replay("mamba", torch.tensor([slot]))
                    self.assertEqual(int(write_pos[slot]), 1 if request_ring else 0)


class TestDecodeRequestRingBackendCursor(CustomTestCase):
    def _backend(self, request_ring):
        backend = object.__new__(MambaAttnBackendBase)
        backend.state_pool = SimpleNamespace(replayssm_request_ring=request_ring)
        # A translating request id space (as under unified memory) shows the
        # cursor rows go through the same translation as the ring rows.
        backend._ring_ids = IdSpace(key=("request", None), write=lambda ids: ids + 10)
        return backend

    def test_cursor_indices(self):
        slots = torch.tensor([3, -1, 5])
        req_pool_indices = torch.tensor([1, 2, 4])
        self.assertEqual(
            self._backend(False)
            ._replayssm_cursor_indices(slots, req_pool_indices)
            .tolist(),
            [3, 0, 5],
        )
        # Padded rows (negative slot) read the reserved row 0.
        self.assertEqual(
            self._backend(True)
            ._replayssm_cursor_indices(slots, req_pool_indices)
            .tolist(),
            [11, 0, 14],
        )


GDN_CONFIG = SimpleNamespace(
    hf_config=SimpleNamespace(
        architectures=["Qwen3NextForCausalLM"],
        linear_attn_config={"linear_attention_layers": [0]},
    )
)
KDA_CONFIG = SimpleNamespace(
    hf_config=SimpleNamespace(
        architectures=["KimiK3LinearForCausalLM"],
        linear_attn_config={"kda_layers": [0]},
    )
)


class TestDecodeRequestRingArgs(CustomTestCase):
    def _check(self, *, request_ring, model_config, **kwargs):
        args = ServerArgs(
            model_path="dummy",
            linear_attn_decode_backend="triton",
            enable_linear_replayssm=True,
            **kwargs,
        )
        with (
            envs.SGLANG_OPT_REPLAYSSM_DECODE_REQUEST_RING.override(request_ring),
            patch(
                "sglang.srt.arg_groups.attention_hook.model_config_of",
                return_value=model_config,
            ),
            override_platform(is_sm100=False),
            override_platform(is_cuda=False),
        ):
            handle_linear_attn_backend(args)

    def test_extra_buffer_needs_the_gdn_request_ring(self):
        extra_buffer = dict(mamba_radix_cache_strategy="extra_buffer")
        with self.assertRaisesRegex(ValueError, "no_buffer"):
            self._check(request_ring=False, model_config=GDN_CONFIG, **extra_buffer)
        self._check(request_ring=True, model_config=GDN_CONFIG, **extra_buffer)
        with self.assertRaisesRegex(ValueError, "no_buffer"):
            self._check(request_ring=True, model_config=KDA_CONFIG, **extra_buffer)

    def test_request_ring_rejects_uncovered_pools(self):
        for flag, message in (
            ("enable_hierarchical_cache", "hierarchical"),
            ("enable_unified_memory", "unified"),
        ):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, message):
                    self._check(
                        request_ring=True, model_config=GDN_CONFIG, **{flag: True}
                    )


if __name__ == "__main__":
    unittest.main()

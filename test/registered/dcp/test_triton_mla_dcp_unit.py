"""Single-GPU unit test for the DCP MLA KV write/read layout.

Fast (no server, no model) proof that the one new data-structure invariant
holds: ``MLATokenToKVPool`` stores a rank-owned token (``loc % dcp == rank``)
at the rank-local physical slot ``loc // dcp``, bitwise identical to what
``create_triton_kv_indices_for_dcp_triton`` reads back — including sparse
virtual locations beyond the physical rank-local capacity. End-to-end decode
correctness (lse merge, fp8, chunked prefill) is covered by the parity suite.
"""

import contextlib
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.dcp.kernels import create_triton_kv_indices_for_dcp_triton
from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

DCP_SIZE = 2
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_DIM


class _Layer:
    layer_id = 0


def _make_pool(dtype: torch.dtype, size: int = 126) -> MLATokenToKVPool:
    return MLATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_DIM,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )


def _dcp_patches(module_path: str, rank: int, with_rank: bool = True):
    # memory_pool no longer imports the rank accessor (ownership masking is
    # in-kernel), so its namespace patches without the rank symbol.
    patches = [
        patch(f"{module_path}.dcp_enabled", lambda: True),
        patch(f"{module_path}.get_attention_dcp_world_size", lambda: DCP_SIZE),
    ]
    if with_rank:
        patches.append(patch(f"{module_path}.get_attention_dcp_rank", lambda: rank))
    return patches


def _read_back_indices(virtual_locs: torch.Tensor, rank: int) -> torch.Tensor:
    """Rank-local physical kv_indices for a request whose token positions
    0..L-1 live at ``virtual_locs``, via the production read-side kernel."""
    seq_len = virtual_locs.numel()
    req_to_token = torch.zeros((1, seq_len), dtype=torch.int64, device="cuda")
    req_to_token[0, :] = virtual_locs
    dcp_len = int(get_dcp_lens(torch.tensor([seq_len]), DCP_SIZE, rank).item())
    kv_indptr = torch.tensor([0, dcp_len], dtype=torch.int64, device="cuda")
    kv_indices = torch.empty(dcp_len, dtype=torch.int64, device="cuda")
    create_triton_kv_indices_for_dcp_triton[(1,)](
        req_to_token,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        torch.tensor([dcp_len], dtype=torch.int64, device="cuda"),
        kv_indptr,
        None,
        kv_indices,
        req_to_token.stride(0),
        DCP_SIZE,
        rank,
    )
    return kv_indices


class TestDcpKvLayoutRoundTrip(CustomTestCase):
    """Pool writes land where the DCP read indices expect, bitwise."""

    # Sparse, parity-aligned, several beyond the 127-row physical buffer
    # (size=126) — only in bounds after the // dcp translation.
    VIRTUAL_LOCS = [4, 5, 10, 11, 200, 201, 250, 251, 126, 127, 60, 61]

    def _cache_rows(self, dtype: torch.dtype, n: int) -> torch.Tensor:
        rows = torch.arange(n * KV_DIM, device="cuda", dtype=torch.float32)
        rows = (rows.reshape(n, 1, KV_DIM) % 61) * 0.25 - 7.0
        return rows.to(dtype)

    def _roundtrip(self, dtype: torch.dtype, use_mla_api: bool):
        virtual = torch.tensor(self.VIRTUAL_LOCS, dtype=torch.int64, device="cuda")
        cache_k = self._cache_rows(dtype, virtual.numel())
        for rank in range(DCP_SIZE):
            pool = _make_pool(dtype)
            patches = _dcp_patches(
                "sglang.srt.mem_cache.memory_pool", rank, with_rank=False
            )
            patches += _dcp_patches("sglang.srt.mem_cache.triton_ops.mla_buffer", rank)
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                if use_mla_api:
                    pool.set_mla_kv_buffer(
                        _Layer(),
                        virtual,
                        cache_k[..., :KV_LORA_RANK],
                        cache_k[..., KV_LORA_RANK:],
                    )
                else:
                    pool.set_kv_buffer(_Layer(), virtual, cache_k, None)
            got = pool.kv_buffer[0][_read_back_indices(virtual, rank)].view(torch.uint8)
            owned = cache_k[virtual % DCP_SIZE == rank]
            expected = (
                owned.view(pool.store_dtype)
                if pool.store_dtype != pool.dtype
                else owned
            ).view(torch.uint8)
            self.assertTrue(
                torch.equal(got, expected),
                f"round-trip mismatch: rank={rank} dtype={dtype} mla_api={use_mla_api}",
            )

    def test_set_kv_buffer_bf16(self):
        self._roundtrip(torch.bfloat16, use_mla_api=False)

    def test_set_kv_buffer_fp8(self):
        self._roundtrip(torch.float8_e4m3fn, use_mla_api=False)

    def test_set_mla_kv_buffer_bf16(self):
        self._roundtrip(torch.bfloat16, use_mla_api=True)

    def test_set_mla_kv_buffer_fp8(self):
        self._roundtrip(torch.float8_e4m3fn, use_mla_api=True)

    def test_raw_virtual_loc_write_fails_roundtrip(self):
        """The pre-fix behavior — ownership mask without the // dcp divide —
        does not land where the read indices expect."""
        rank = 0
        virtual = torch.tensor([4, 5, 10, 11, 60, 61, 126, 127], device="cuda")
        cache_k = self._cache_rows(torch.bfloat16, virtual.numel())
        pool = _make_pool(torch.bfloat16)
        mask = virtual % DCP_SIZE == rank
        pool.kv_buffer[0][virtual[mask]] = cache_k[mask]  # raw loc, no divide
        got = pool.kv_buffer[0][_read_back_indices(virtual, rank)]
        self.assertFalse(
            torch.equal(got.view(torch.uint8), cache_k[mask].view(torch.uint8)),
            "raw virtual-loc writes must not satisfy the DCP read layout",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

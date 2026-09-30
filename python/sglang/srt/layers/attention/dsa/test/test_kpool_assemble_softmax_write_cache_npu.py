"""Accuracy tests for the Ascend BF16 KPool assembly kernel."""

import unittest
from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from sglang.srt.layers.attention.dsa.kpool_index_npu import (
    kpool_assemble_softmax_write_cache_npu,
)
from sglang.test.test_utils import CustomTestCase


HEAD_DIM = 128
PAGE_SIZE = 128
RTOL = 1e-2
ATOL = 1e-2


def _npu_is_available() -> bool:
    if torch_npu is None or not hasattr(torch, "npu"):
        return False
    try:
        return torch.npu.is_available()
    except RuntimeError:
        return False


@torch.no_grad()
def kpool_assemble_softmax_write_cache_torch(
    index_k_cache: torch.Tensor,
    chunk_k: torch.Tensor,
    chunk_score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    req_pool_idx: torch.Tensor,
    n_from_tail: torch.Tensor,
    chunk_src_start: torch.Tensor,
    tail_logical_base: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    write_mask: Optional[torch.Tensor] = None,
) -> None:
    """Torch numerical reference for BF16 KPool assembly and cache write."""

    n_pools = req_pool_idx.numel()
    if n_pools == 0:
        return

    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    if write_mask is None:
        active_rows = torch.arange(n_pools, dtype=torch.int64, device=loc.device)
    else:
        active_rows = torch.nonzero(write_mask, as_tuple=False).flatten()

    req = req_pool_idx.index_select(0, active_rows)
    n_tail = n_from_tail.index_select(0, active_rows).to(torch.int64)
    chunk_src = chunk_src_start.index_select(0, active_rows)
    tail_base = tail_logical_base.index_select(0, active_rows).to(torch.int64)

    slots = torch.arange(pool_size, dtype=torch.int64, device=chunk_k.device)
    slots = slots.unsqueeze(0)
    use_tail = slots < n_tail.unsqueeze(1)

    tail_phys = torch.remainder(tail_base.unsqueeze(1) + slots, tail_size)
    tail_req = req.unsqueeze(1).expand_as(tail_phys)
    assembled_tail_k = tail_k[tail_req, tail_phys]
    assembled_tail_score = tail_score[tail_req, tail_phys]

    # torch.where evaluates both branches, so keep unselected chunk indices valid.
    chunk_offsets = torch.clamp(slots - n_tail.unsqueeze(1), min=0)
    chunk_indices = chunk_src.unsqueeze(1) + chunk_offsets
    assembled_chunk_k = chunk_k[chunk_indices]
    assembled_chunk_score = chunk_score[chunk_indices]

    assembled_k = torch.where(
        use_tail.unsqueeze(-1), assembled_tail_k, assembled_chunk_k
    )
    assembled_score = torch.where(
        use_tail.unsqueeze(-1), assembled_tail_score, assembled_chunk_score
    )

    logits = assembled_score.float() + ape.float().unsqueeze(0)
    probs = torch.softmax(logits, dim=1)
    pooled = torch.sum(assembled_k.float() * probs, dim=1).to(torch.bfloat16)

    cache_2d = index_k_cache.view(-1, HEAD_DIM)
    active_locs = loc.index_select(0, active_rows)
    cache_2d.index_copy_(0, active_locs, pooled)


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is required")
class TestKPoolBF16IndexNPU(CustomTestCase):
    device = "npu"

    @staticmethod
    def _make_noncontiguous(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            result = torch.stack((tensor, tensor), dim=1)[:, 0]
        else:
            result = tensor.transpose(0, 1).contiguous().transpose(0, 1)
        assert result.shape == tensor.shape and not result.is_contiguous()
        return result

    def _make_inputs(
        self,
        *,
        seed: int,
        n_pools: int,
        pool_size: int,
        tail_size: int,
        n_from_tail_values: list[int],
        score_dtype: torch.dtype,
        use_write_mask: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self.assertEqual(len(n_from_tail_values), n_pools)
        self.assertTrue(all(0 <= value <= pool_size for value in n_from_tail_values))
        self.assertLessEqual(max(n_from_tail_values), tail_size)

        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)

        chunk_src_starts = []
        chunk_rows = 0
        for n_tail in n_from_tail_values:
            chunk_src_starts.append(chunk_rows)
            chunk_rows += pool_size - n_tail

        # The Torch reference gathers both torch.where branches. Keep one
        # sentinel row valid for pools assembled entirely from tail state.
        chunk_rows += 1
        n_requests = max(4, (n_pools + 1) // 2)
        cache_pages = max(2, (n_pools + PAGE_SIZE) // PAGE_SIZE)
        cache_rows = cache_pages * PAGE_SIZE

        def randn(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.randn(shape, device=self.device, dtype=torch.float32).to(dtype)

        initial_cache = randn(
            (cache_pages, PAGE_SIZE, 1, HEAD_DIM), torch.bfloat16
        )
        chunk_k = randn((chunk_rows, HEAD_DIM), torch.bfloat16)
        tail_k = randn((n_requests, tail_size, HEAD_DIM), torch.bfloat16)
        chunk_score = randn((chunk_rows, HEAD_DIM), score_dtype) * 4
        tail_score = randn((n_requests, tail_size, HEAD_DIM), score_dtype) * 4
        ape = randn((pool_size, HEAD_DIM), torch.float32) * 2

        req_pool_idx = torch.tensor(
            [(row * 3 + 1) % n_requests for row in range(n_pools)],
            device=self.device,
            dtype=torch.int64,
        )
        n_from_tail = torch.tensor(
            n_from_tail_values, device=self.device, dtype=torch.int32
        )
        chunk_src_start = torch.tensor(
            chunk_src_starts, device=self.device, dtype=torch.int64
        )
        tail_logical_base = torch.tensor(
            [(tail_size - 2 + row * 3) % tail_size for row in range(n_pools)],
            device=self.device,
            dtype=torch.int32,
        )
        loc = torch.tensor(
            [(row * 7 + 3) % cache_rows for row in range(n_pools)],
            device=self.device,
            dtype=torch.int64,
        )
        write_mask = None
        if use_write_mask:
            write_mask = torch.tensor(
                [row % 3 != 1 for row in range(n_pools)],
                device=self.device,
                dtype=torch.bool,
            )

        inputs = {
            "chunk_k": chunk_k.contiguous(),
            "chunk_score": chunk_score.contiguous(),
            "tail_k": tail_k.contiguous(),
            "tail_score": tail_score.contiguous(),
            "req_pool_idx": req_pool_idx.contiguous(),
            "n_from_tail": n_from_tail.contiguous(),
            "chunk_src_start": chunk_src_start.contiguous(),
            "tail_logical_base": tail_logical_base.contiguous(),
            "ape": ape.contiguous(),
            "loc": loc.contiguous(),
        }
        if write_mask is not None:
            inputs["write_mask"] = write_mask.contiguous()
        return initial_cache.contiguous(), inputs

    def _run_accuracy_case(
        self,
        *,
        seed: int,
        n_pools: int,
        pool_size: int,
        tail_size: int,
        n_from_tail_values: list[int],
        score_dtype: torch.dtype,
        use_write_mask: bool = False,
        use_noncontiguous_dynamic_inputs: bool = False,
        num_programs: Optional[int] = None,
    ) -> None:
        initial_cache, inputs = self._make_inputs(
            seed=seed,
            n_pools=n_pools,
            pool_size=pool_size,
            tail_size=tail_size,
            n_from_tail_values=n_from_tail_values,
            score_dtype=score_dtype,
            use_write_mask=use_write_mask,
        )
        if use_noncontiguous_dynamic_inputs:
            dynamic_names = ["chunk_k", "chunk_score", "ape", "loc"]
            if "write_mask" in inputs:
                dynamic_names.append("write_mask")
            for name in dynamic_names:
                inputs[name] = self._make_noncontiguous(inputs[name])

        torch_cache = initial_cache.clone()
        triton_cache = initial_cache.clone()

        kpool_assemble_softmax_write_cache_torch(
            index_k_cache=torch_cache,
            **inputs,
        )
        kpool_assemble_softmax_write_cache_npu(
            index_k_cache=triton_cache,
            num_programs=num_programs,
            **inputs,
        )
        torch.npu.synchronize()

        expected = torch_cache.float().cpu()
        actual = triton_cache.float().cpu()
        max_abs_diff = (actual - expected).abs().max().item()
        torch.testing.assert_close(
            actual,
            expected,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda msg: f"max_abs_diff={max_abs_diff}\n{msg}",
        )

        write_mask = inputs.get("write_mask")
        if write_mask is not None:
            inactive_locs = inputs["loc"][~write_mask]
            initial_2d = initial_cache.view(-1, HEAD_DIM)
            triton_2d = triton_cache.view(-1, HEAD_DIM)
            torch.testing.assert_close(
                triton_2d.index_select(0, inactive_locs).cpu(),
                initial_2d.index_select(0, inactive_locs).cpu(),
                rtol=0,
                atol=0,
            )

    def test_all_chunk_score_dtypes(self):
        for index, score_dtype in enumerate(
            (torch.bfloat16, torch.float16, torch.float32)
        ):
            with self.subTest(score_dtype=score_dtype):
                self._run_accuracy_case(
                    seed=100 + index,
                    n_pools=5,
                    pool_size=8,
                    tail_size=13,
                    n_from_tail_values=[0] * 5,
                    score_dtype=score_dtype,
                )

    def test_mixed_tail_chunk_wraparound_mask_and_noncontiguous_inputs(self):
        self._run_accuracy_case(
            seed=200,
            n_pools=9,
            pool_size=8,
            tail_size=11,
            n_from_tail_values=[0, 1, 3, 7, 8, 2, 5, 8, 4],
            score_dtype=torch.float32,
            use_write_mask=True,
            use_noncontiguous_dynamic_inputs=True,
            num_programs=4,
        )

    def test_grid_stride_when_pools_exceed_programs(self):
        n_pools = 37
        self._run_accuracy_case(
            seed=300,
            n_pools=n_pools,
            pool_size=16,
            tail_size=19,
            n_from_tail_values=[(row * 5) % 17 for row in range(n_pools)],
            score_dtype=torch.bfloat16,
            num_programs=3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

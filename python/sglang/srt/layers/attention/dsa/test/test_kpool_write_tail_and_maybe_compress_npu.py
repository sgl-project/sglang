"""Accuracy tests for the Ascend KPool tail-write/compress kernel."""

import unittest
from typing import Optional

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

from sglang.srt.layers.attention.dsa.kpool_index_npu import (
    kpool_write_tail_and_maybe_compress_npu,
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
def kpool_write_tail_and_maybe_compress_torch(
    index_k_cache: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    tail_k: torch.Tensor,
    tail_score: torch.Tensor,
    ape: torch.Tensor,
    req_pool_indices: torch.Tensor,
    write_start: torch.Tensor,
    tail_logical_start: torch.Tensor,
    write_loc: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_draft_tokens: int,
    effective_n_per_batch: Optional[torch.Tensor] = None,
) -> None:
    """Torch reference matching the Triton target-verify kernel semantics."""

    total_rows = key.shape[0]
    if total_rows == 0:
        return

    batch_size = total_rows // num_draft_tokens
    pool_size = ape.shape[0]
    tail_size = tail_k.shape[1]
    cache_2d = index_k_cache.view(-1, HEAD_DIM)

    for batch in range(batch_size):
        if out_cache_loc[batch * num_draft_tokens].item() == 0:
            continue

        req = int(req_pool_indices[batch].item())
        start = int(write_start[batch].item())
        for draft_offset in range(num_draft_tokens):
            row = batch * num_draft_tokens + draft_offset
            tail_slot = (start + draft_offset) % tail_size
            tail_k[req, tail_slot] = key[row]
            tail_score[req, tail_slot] = score[row]

        effective_n = (
            num_draft_tokens
            if effective_n_per_batch is None
            else int(effective_n_per_batch[batch].item())
        )
        effective_n = max(0, min(effective_n, num_draft_tokens))
        base_pool = start // pool_size
        completed_pools = (start + effective_n) // pool_size - base_pool
        if completed_pools == 0:
            continue

        first_logical_start = int(tail_logical_start[batch].item())
        slots = torch.arange(pool_size, dtype=torch.int64, device=tail_k.device)
        for pool_offset in range(completed_pools):
            logical_start = first_logical_start + pool_offset * pool_size
            tail_slots = torch.remainder(logical_start + slots, tail_size)
            pool_keys = tail_k[req].index_select(0, tail_slots).float()
            pool_scores = tail_score[req].index_select(0, tail_slots).float()
            probs = torch.softmax(pool_scores + ape.float(), dim=0)
            pooled = torch.sum(pool_keys * probs, dim=0).to(torch.bfloat16)
            cache_2d[int(write_loc[batch, pool_offset].item())] = pooled


@unittest.skipUnless(_npu_is_available(), "Ascend NPU is required")
class TestKPoolWriteTailAndMaybeCompressNPU(CustomTestCase):
    device = "npu"

    @staticmethod
    def _make_noncontiguous(tensor: torch.Tensor) -> torch.Tensor:
        result = torch.stack((tensor, tensor), dim=-1)[..., 0]
        assert result.shape == tensor.shape and not result.is_contiguous()
        return result

    def _make_inputs(
        self,
        *,
        seed: int,
        batch_size: int,
        num_draft_tokens: int,
        pool_size: int,
        write_start_values: list[int],
        score_dtype: torch.dtype,
        effective_n_values: Optional[list[int]] = None,
        invalid_batches: tuple[int, ...] = (),
        use_noncontiguous_inputs: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor | int],
    ]:
        self.assertEqual(len(write_start_values), batch_size)
        if effective_n_values is not None:
            self.assertEqual(len(effective_n_values), batch_size)

        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)

        total_rows = batch_size * num_draft_tokens
        tail_size = pool_size + num_draft_tokens
        max_closed_pools = (num_draft_tokens + pool_size - 1) // pool_size
        required_cache_rows = batch_size * max_closed_pools + 3
        cache_pages = max(
            2, (required_cache_rows + PAGE_SIZE - 1) // PAGE_SIZE
        )
        cache_rows = cache_pages * PAGE_SIZE

        def randn(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return torch.randn(shape, device=self.device, dtype=torch.float32).to(dtype)

        initial_cache = randn(
            (cache_pages, PAGE_SIZE, 1, HEAD_DIM), torch.bfloat16
        ).contiguous()
        initial_tail_k = randn(
            (batch_size, tail_size, HEAD_DIM), torch.bfloat16
        ).contiguous()
        initial_tail_score = randn(
            (batch_size, tail_size, HEAD_DIM), score_dtype
        ).contiguous()

        key = randn((total_rows, HEAD_DIM), torch.bfloat16)
        score = randn((total_rows, HEAD_DIM), score_dtype) * 4
        ape = randn((pool_size, HEAD_DIM), torch.float32) * 2
        req_pool_indices = torch.arange(
            batch_size, dtype=torch.int64, device=self.device
        )
        write_start = torch.tensor(
            write_start_values, dtype=torch.int32, device=self.device
        )
        tail_logical_start = torch.tensor(
            [(value // pool_size) * pool_size for value in write_start_values],
            dtype=torch.int32,
            device=self.device,
        )
        write_loc = (
            torch.arange(
                batch_size * max_closed_pools,
                dtype=torch.int64,
                device=self.device,
            ).view(batch_size, max_closed_pools)
            + 3
        )
        self.assertLess(int(write_loc.max().item()), cache_rows)
        out_cache_loc = torch.arange(
            1, total_rows + 1, dtype=torch.int64, device=self.device
        )
        for batch in invalid_batches:
            start = batch * num_draft_tokens
            out_cache_loc[start : start + num_draft_tokens] = 0

        inputs: dict[str, torch.Tensor | int] = {
            "key": key.contiguous(),
            "score": score.contiguous(),
            "ape": ape.contiguous(),
            "req_pool_indices": req_pool_indices.contiguous(),
            "write_start": write_start.contiguous(),
            "tail_logical_start": tail_logical_start.contiguous(),
            "write_loc": write_loc.contiguous(),
            "out_cache_loc": out_cache_loc.contiguous(),
            "num_draft_tokens": num_draft_tokens,
        }
        if effective_n_values is not None:
            inputs["effective_n_per_batch"] = torch.tensor(
                effective_n_values, dtype=torch.int32, device=self.device
            ).contiguous()

        if use_noncontiguous_inputs:
            dynamic_names = [
                "key",
                "score",
                "ape",
                "req_pool_indices",
                "write_start",
                "tail_logical_start",
                "write_loc",
                "out_cache_loc",
            ]
            if "effective_n_per_batch" in inputs:
                dynamic_names.append("effective_n_per_batch")
            for name in dynamic_names:
                tensor = inputs[name]
                assert isinstance(tensor, torch.Tensor)
                inputs[name] = self._make_noncontiguous(tensor)

        return initial_cache, initial_tail_k, initial_tail_score, inputs

    def _run_accuracy_case(
        self,
        *,
        seed: int,
        batch_size: int,
        num_draft_tokens: int,
        pool_size: int,
        write_start_values: list[int],
        score_dtype: torch.dtype,
        effective_n_values: Optional[list[int]] = None,
        invalid_batches: tuple[int, ...] = (),
        use_noncontiguous_inputs: bool = False,
        num_programs: Optional[int] = None,
    ) -> None:
        initial_cache, initial_tail_k, initial_tail_score, inputs = self._make_inputs(
            seed=seed,
            batch_size=batch_size,
            num_draft_tokens=num_draft_tokens,
            pool_size=pool_size,
            write_start_values=write_start_values,
            score_dtype=score_dtype,
            effective_n_values=effective_n_values,
            invalid_batches=invalid_batches,
            use_noncontiguous_inputs=use_noncontiguous_inputs,
        )

        torch_cache = initial_cache.clone()
        torch_tail_k = initial_tail_k.clone()
        torch_tail_score = initial_tail_score.clone()
        triton_cache = initial_cache.clone()
        triton_tail_k = initial_tail_k.clone()
        triton_tail_score = initial_tail_score.clone()

        kpool_write_tail_and_maybe_compress_torch(
            index_k_cache=torch_cache,
            tail_k=torch_tail_k,
            tail_score=torch_tail_score,
            **inputs,
        )
        kpool_write_tail_and_maybe_compress_npu(
            index_k_cache=triton_cache,
            tail_k=triton_tail_k,
            tail_score=triton_tail_score,
            num_programs=num_programs,
            **inputs,
        )
        torch.npu.synchronize()

        torch.testing.assert_close(
            triton_tail_k.cpu(), torch_tail_k.cpu(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            triton_tail_score.cpu(), torch_tail_score.cpu(), rtol=0, atol=0
        )

        expected_cache = torch_cache.float().cpu()
        actual_cache = triton_cache.float().cpu()
        max_abs_diff = (actual_cache - expected_cache).abs().max().item()
        torch.testing.assert_close(
            actual_cache,
            expected_cache,
            rtol=RTOL,
            atol=ATOL,
            msg=lambda msg: f"max_abs_diff={max_abs_diff}\n{msg}",
        )

        inactive_locs = []
        write_loc = inputs["write_loc"]
        assert isinstance(write_loc, torch.Tensor)
        max_closed_pools = write_loc.shape[1]
        for batch, start in enumerate(write_start_values):
            valid = batch not in invalid_batches
            gate_n = (
                num_draft_tokens
                if effective_n_values is None
                else effective_n_values[batch]
            )
            gate_n = max(0, min(gate_n, num_draft_tokens))
            completed = (start + gate_n) // pool_size - start // pool_size
            first_inactive = completed if valid else 0
            inactive_locs.extend(
                write_loc[batch, first_inactive:max_closed_pools].tolist()
            )

        if inactive_locs:
            inactive_index = torch.tensor(
                inactive_locs, dtype=torch.int64, device=self.device
            )
            torch.testing.assert_close(
                triton_cache.view(-1, HEAD_DIM)
                .index_select(0, inactive_index)
                .cpu(),
                initial_cache.view(-1, HEAD_DIM)
                .index_select(0, inactive_index)
                .cpu(),
                rtol=0,
                atol=0,
            )

        for batch in invalid_batches:
            torch.testing.assert_close(
                triton_tail_k[batch].cpu(),
                initial_tail_k[batch].cpu(),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                triton_tail_score[batch].cpu(),
                initial_tail_score[batch].cpu(),
                rtol=0,
                atol=0,
            )

    def test_score_dtypes_and_ring_wrap_compression(self):
        for index, score_dtype in enumerate(
            (torch.bfloat16, torch.float16, torch.float32)
        ):
            with self.subTest(score_dtype=score_dtype):
                self._run_accuracy_case(
                    seed=100 + index,
                    batch_size=4,
                    num_draft_tokens=2,
                    pool_size=4,
                    write_start_values=[2, 3, 6, 7],
                    score_dtype=score_dtype,
                )

    def test_no_compression_and_invalid_batch(self):
        self._run_accuracy_case(
            seed=200,
            batch_size=5,
            num_draft_tokens=2,
            pool_size=8,
            write_start_values=[0, 5, 6, 7, 6],
            score_dtype=torch.float32,
            invalid_batches=(4,),
            num_programs=2,
        )

    def test_effective_n_noncontiguous_and_grid_stride(self):
        batch_size = 17
        self._run_accuracy_case(
            seed=300,
            batch_size=batch_size,
            num_draft_tokens=4,
            pool_size=8,
            write_start_values=[(batch * 3) % 24 for batch in range(batch_size)],
            score_dtype=torch.bfloat16,
            effective_n_values=[batch % 5 for batch in range(batch_size)],
            use_noncontiguous_inputs=True,
            num_programs=3,
        )

    def test_multiple_completed_pools(self):
        self._run_accuracy_case(
            seed=400,
            batch_size=4,
            num_draft_tokens=10,
            pool_size=4,
            write_start_values=[0, 3, 6, 7],
            score_dtype=torch.float32,
            effective_n_values=[10, 10, 9, 0],
            num_programs=2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

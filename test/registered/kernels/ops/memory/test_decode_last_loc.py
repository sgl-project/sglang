import itertools
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.memory.common import (
    get_last_loc_triton_safe,
    get_last_loc_triton_safe_i32,
)
from sglang.srt.mem_cache.allocation import last_loc_uses_triton_dispatch
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")
# The dispatch-predicate cases below are pure Python. Register CPU so the
# backend guard keeps coverage on runners that have no accelerator at all --
# `ascend` and `torch_native` are precisely the backends a GPU runner cannot
# exercise.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# req_to_token is int32 (mem_cache/memory_pool.py ReqToTokenPool.__init__), so the
# reference expression below also yields int32 -- the _i32 helper returning int32 is
# parity with today's numerics, not a narrowing.
REQ_TO_TOKEN_DTYPE = torch.int32
# _get_last_loc_safe_kernel's BLOCK_SIZE. Batch sizes straddle it so the masked
# tail of a multi-block grid is exercised rather than assumed correct.
BLOCK_SIZE = 256


def _make_req_to_token(num_reqs, max_context_len, device):
    """Distinct, non-negative slot ids so a wrong gather cannot alias a right one."""
    total = num_reqs * max_context_len
    return (
        torch.arange(total, dtype=REQ_TO_TOKEN_DTYPE, device=device).view(
            num_reqs, max_context_len
        )
        + 1
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or HIP")
class TestGetLastLocTritonSafe(CustomTestCase):
    """Contract of the single-launch `last_loc` helpers used by paged decode alloc.

    `get_last_loc_triton_safe_i32` replaces the two-launch torch expression
    `req_to_token[req_pool_indices, prefix_lens - 1]` on the per-step decode path,
    so the property that matters is bit-equality with that expression -- a wrong
    `last_loc` silently points attention at another request's KV rather than
    raising.
    """

    DEVICE = "cuda"

    def test_matches_advanced_indexing(self):
        """Bit-identical to the torch expression it replaces, across dtypes/shapes."""
        generator = torch.Generator().manual_seed(0)
        for bs, max_context_len, prefix_dtype, index_dtype in itertools.product(
            (1, 7, BLOCK_SIZE, BLOCK_SIZE + 1),
            (1, 13, 4096),
            (torch.int32, torch.int64),
            (torch.int32, torch.int64),
        ):
            with self.subTest(
                bs=bs,
                max_context_len=max_context_len,
                prefix_dtype=prefix_dtype,
                index_dtype=index_dtype,
            ):
                num_reqs = bs + 3
                req_to_token = _make_req_to_token(
                    num_reqs, max_context_len, self.DEVICE
                )
                # Permuted, not arange: a kernel ignoring req_pool_indices entirely
                # still passes against an identity mapping.
                req_pool_indices = torch.randperm(num_reqs, generator=generator)[
                    :bs
                ].to(device=self.DEVICE, dtype=index_dtype)
                # >= 1 here; the prefix_lens == 0 divergence is its own test.
                prefix_lens = torch.randint(
                    1,
                    max_context_len + 1,
                    (bs,),
                    generator=generator,
                ).to(device=self.DEVICE, dtype=prefix_dtype)

                expected = req_to_token[req_pool_indices.long(), prefix_lens.long() - 1]
                actual = get_last_loc_triton_safe_i32(
                    req_to_token, req_pool_indices, prefix_lens
                )

                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(torch.equal(actual, expected))

    def test_zero_prefix_yields_minus_one_instead_of_wraparound(self):
        """The one documented divergence from plain indexing, with its control.

        A row with `prefix_lens == 0` indexes column -1 under torch semantics, which
        silently returns the LAST column of that request. The kernel returns -1
        instead. Asserting only the -1 would pass against a kernel that returned -1
        for every row, so the torch value is asserted to be the wraparound -- that is
        the negative control on the claim that this is a deliberate divergence.
        """
        max_context_len = 32
        num_reqs = 8
        req_to_token = _make_req_to_token(num_reqs, max_context_len, self.DEVICE)
        req_pool_indices = torch.tensor(
            [5, 2, 7, 0], device=self.DEVICE, dtype=torch.int64
        )
        prefix_lens = torch.tensor([0, 4, 0, 1], device=self.DEVICE, dtype=torch.int64)

        actual = get_last_loc_triton_safe_i32(
            req_to_token, req_pool_indices, prefix_lens
        )
        torch_reference = req_to_token[req_pool_indices, prefix_lens - 1]

        zero = prefix_lens == 0
        self.assertTrue(torch.equal(actual[zero], torch.full_like(actual[zero], -1)))
        # Control: torch really does wrap to the last column for those same rows.
        self.assertTrue(
            torch.equal(
                torch_reference[zero],
                req_to_token[req_pool_indices[zero], max_context_len - 1],
            )
        )
        # Non-zero rows still agree with torch.
        self.assertTrue(torch.equal(actual[~zero], torch_reference[~zero]))

    def test_i32_core_never_promotes(self):
        """The whole point of the _i32 split: no trailing promotion launch."""
        req_to_token = _make_req_to_token(4, 16, self.DEVICE)
        req_pool_indices = torch.arange(4, device=self.DEVICE)
        for prefix_dtype in (torch.int32, torch.int64):
            with self.subTest(prefix_dtype=prefix_dtype):
                prefix_lens = torch.full(
                    (4,), 3, device=self.DEVICE, dtype=prefix_dtype
                )
                result = get_last_loc_triton_safe_i32(
                    req_to_token, req_pool_indices, prefix_lens
                )
                self.assertEqual(result.dtype, REQ_TO_TOKEN_DTYPE)

    def test_wrapper_promotes_to_index_dtype(self):
        """`get_last_loc_triton_safe` keeps its pre-split signature and dtype."""
        req_to_token = _make_req_to_token(6, 64, self.DEVICE)
        req_pool_indices = torch.tensor(
            [3, 1, 4, 5], device=self.DEVICE, dtype=torch.int64
        )
        for prefix_dtype in (torch.int32, torch.int64):
            with self.subTest(prefix_dtype=prefix_dtype):
                prefix_lens = torch.tensor(
                    [1, 17, 64, 2], device=self.DEVICE, dtype=prefix_dtype
                )
                wrapped = get_last_loc_triton_safe(
                    req_to_token, req_pool_indices, prefix_lens
                )
                core = get_last_loc_triton_safe_i32(
                    req_to_token, req_pool_indices, prefix_lens
                )

                self.assertEqual(wrapped.dtype, prefix_dtype)
                self.assertTrue(torch.equal(wrapped, core.to(prefix_dtype)))

    def test_honours_req_to_token_row_stride(self):
        """The kernel is handed `stride(0)`, so a row-strided view must still work."""
        base = _make_req_to_token(16, 32, self.DEVICE)
        strided = base[::2]
        self.assertNotEqual(strided.stride(0), strided.shape[1])

        req_pool_indices = torch.arange(strided.shape[0], device=self.DEVICE)
        prefix_lens = torch.randint(
            1, strided.shape[1] + 1, (strided.shape[0],), device=self.DEVICE
        )

        actual = get_last_loc_triton_safe_i32(strided, req_pool_indices, prefix_lens)
        expected = strided[req_pool_indices, prefix_lens - 1]
        self.assertTrue(torch.equal(actual, expected))


class TestLastLocUsesTritonDispatch(CustomTestCase):
    """Backend guard for the paged-decode `last_loc` computation.

    `alloc_for_decode` replaced plain torch indexing with a Triton launch, and
    reaching that branch does not imply a Triton runtime: the generic paged
    allocator's `alloc_decode` does launch a Triton kernel, but the NPU
    allocator's is pure torch, so `ascend` arrives there with none. The
    predicate must therefore agree with the one `get_last_loc` already applies
    to the identical computation on the extend side.

    Both answers are pinned. A guard only tested on the backends that return
    True would pass just as well if it were hardwired to True.
    """

    TRITON_BACKENDS = ("fa3", "triton", "aiter", "flashinfer")
    NON_TRITON_BACKENDS = ("ascend", "torch_native")

    @staticmethod
    def _predicate_with(prefill, decode):
        with mock.patch(
            "sglang.srt.mem_cache.allocation.attention_backends",
            return_value=(prefill, decode),
        ):
            return last_loc_uses_triton_dispatch()

    def test_triton_backends_allow_the_fused_launch(self):
        for prefill, decode in itertools.product(self.TRITON_BACKENDS, repeat=2):
            with self.subTest(prefill=prefill, decode=decode):
                self.assertTrue(self._predicate_with(prefill, decode))

    def test_either_half_being_non_triton_forces_the_torch_fallback(self):
        # Either half is disqualifying: the pair is read as "this server has a
        # Triton runtime at all", not "the decode half happens to have one".
        for non_triton in self.NON_TRITON_BACKENDS:
            for other in self.TRITON_BACKENDS:
                with self.subTest(non_triton=non_triton, other=other):
                    self.assertFalse(self._predicate_with(non_triton, other))
                    self.assertFalse(self._predicate_with(other, non_triton))

    def test_both_halves_non_triton(self):
        for prefill, decode in itertools.product(self.NON_TRITON_BACKENDS, repeat=2):
            with self.subTest(prefill=prefill, decode=decode):
                self.assertFalse(self._predicate_with(prefill, decode))

    def test_matches_the_extend_side_routing(self):
        """The decode guard and `get_last_loc` must not drift apart.

        `get_last_loc` sends non-Triton backends to `get_last_loc_torch`; the
        decode branch must make the same call for the same pair, since it is
        the same computation over the same tensors.
        """
        for prefill, decode in itertools.product(
            self.TRITON_BACKENDS + self.NON_TRITON_BACKENDS, repeat=2
        ):
            with self.subTest(prefill=prefill, decode=decode):
                expected = prefill not in self.NON_TRITON_BACKENDS and (
                    decode not in self.NON_TRITON_BACKENDS
                )
                self.assertEqual(self._predicate_with(prefill, decode), expected)


if __name__ == "__main__":
    unittest.main()

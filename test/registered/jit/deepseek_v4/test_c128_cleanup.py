"""Correctness tests for fused offline C128 speculative-state cleanup."""

from __future__ import annotations

import gc
import unittest
import weakref

import torch

from sglang.jit_kernel.dsv4 import (
    C128DraftCleanup,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_RING_SIZE = 256


def _make_states(
    num_states: int,
    *,
    num_req_slots: int,
    half: int,
    dtype: torch.dtype,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return [
        torch.randn(
            num_req_slots * _RING_SIZE,
            half * 2,
            dtype=dtype,
            device="cuda",
            generator=generator,
        )
        for _ in range(num_states)
    ]


def _reference_cleanup(
    states: list[torch.Tensor],
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    *,
    num_draft_tokens: int,
) -> list[torch.Tensor]:
    """Apply the cleanup contract with ordinary PyTorch indexing."""
    expected = [state.clone() for state in states]
    req_pool_indices_cpu = req_pool_indices.cpu().tolist()
    seq_lens_cpu = seq_lens.cpu().tolist()
    accept_lens_cpu = accept_lens.cpu().tolist()

    for req_pool_idx, seq_len, accept_len in zip(
        req_pool_indices_cpu, seq_lens_cpu, accept_lens_cpu
    ):
        for draft_offset in range(accept_len, num_draft_tokens):
            row = req_pool_idx * _RING_SIZE + ((seq_len + draft_offset) % _RING_SIZE)
            for state in expected:
                half = state.shape[-1] // 2
                state[row, :half].zero_()
                state[row, half:].fill_(float("-inf"))

    return expected


class TestC128DraftCleanup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def _run_case(
        self,
        *,
        num_states: int,
        batch_size: int,
        num_draft_tokens: int,
        half: int,
        dtype: torch.dtype,
    ) -> None:
        if batch_size == 1:
            req_values = [2]
            seq_values = [511]
            accept_values = [1]
        elif batch_size == 2:
            req_values = [0, 3]
            seq_values = [255, 256]
            accept_values = [1, num_draft_tokens]
        elif batch_size == 4:
            req_values = [0, 2, 5, 7]
            seq_values = [255, 256, 511, 512]
            accept_values = [1, 3, num_draft_tokens, 2]
        else:
            raise AssertionError(f"Unexpected test batch size: {batch_size}")

        states = _make_states(
            num_states,
            num_req_slots=max(req_values) + 1,
            half=half,
            dtype=dtype,
            seed=1000 + num_states * 100 + batch_size * 10 + half,
        )
        req_pool_indices = torch.tensor(req_values, dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor(seq_values, dtype=torch.int64, device="cuda")
        accept_lens = torch.tensor(accept_values, dtype=torch.int32, device="cuda")
        expected = _reference_cleanup(
            states,
            req_pool_indices,
            seq_lens,
            accept_lens,
            num_draft_tokens=num_draft_tokens,
        )

        cleanup = C128DraftCleanup(states, ring_size=_RING_SIZE)
        self.assertIsInstance(cleanup, C128DraftCleanup)
        cleanup.clear(
            req_pool_indices,
            seq_lens,
            accept_lens,
            num_draft_tokens=num_draft_tokens,
        )
        torch.cuda.synchronize()

        for layer_id, (actual, reference) in enumerate(zip(states, expected)):
            self.assertTrue(
                torch.equal(actual, reference),
                f"state differs from reference at layer {layer_id}",
            )

    @torch.inference_mode()
    def test_matches_pytorch_reference(self):
        cases = [
            # Single-layer and feature-tail coverage, with ring wrap at 255.
            (1, 2, 2, 257),
            # Sparse request slots, mixed acceptance, and several ring wraps.
            (3, 4, 6, 512),
            # Exact layer/batch/draft/width geometry from the decode trace.
            (31, 1, 4, 512),
        ]
        for dtype in (torch.float32, torch.bfloat16):
            for num_states, batch_size, num_draft_tokens, half in cases:
                with self.subTest(
                    dtype=dtype,
                    num_states=num_states,
                    batch_size=batch_size,
                    num_draft_tokens=num_draft_tokens,
                    half=half,
                ):
                    self._run_case(
                        num_states=num_states,
                        batch_size=batch_size,
                        num_draft_tokens=num_draft_tokens,
                        half=half,
                        dtype=dtype,
                    )

    @torch.inference_mode()
    def test_full_accept_and_empty_batch_are_noops(self):
        states = _make_states(
            3,
            num_req_slots=4,
            half=512,
            dtype=torch.float32,
            seed=2001,
        )
        cleanup = C128DraftCleanup(states, ring_size=_RING_SIZE)
        original = [state.clone() for state in states]

        req_pool_indices = torch.tensor([0, 3], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([255, 511], dtype=torch.int64, device="cuda")
        accept_lens = torch.full((2,), 4, dtype=torch.int32, device="cuda")
        cleanup.clear(
            req_pool_indices,
            seq_lens,
            accept_lens,
            num_draft_tokens=4,
        )

        empty_indices = torch.empty(0, dtype=torch.int64, device="cuda")
        empty_accept_lens = torch.empty(0, dtype=torch.int32, device="cuda")
        cleanup.clear(
            empty_indices,
            empty_indices,
            empty_accept_lens,
            num_draft_tokens=4,
        )
        torch.cuda.synchronize()

        for layer_id, (actual, reference) in enumerate(zip(states, original)):
            self.assertTrue(
                torch.equal(actual, reference),
                f"no-op cleanup changed layer {layer_id}",
            )

    @torch.inference_mode()
    def test_cuda_graph_replay_reads_updated_metadata(self):
        num_draft_tokens = 4
        states = _make_states(
            3,
            num_req_slots=5,
            half=512,
            dtype=torch.bfloat16,
            seed=3001,
        )
        baseline = [state.clone() for state in states]
        cleanup = C128DraftCleanup(states, ring_size=_RING_SIZE)

        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        seq_lens = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
        accept_lens = torch.full(
            (2,), num_draft_tokens, dtype=torch.int32, device="cuda"
        )

        # Compile before capture. Full-accept metadata makes warmup a no-op.
        cleanup.clear(
            req_pool_indices,
            seq_lens,
            accept_lens,
            num_draft_tokens=num_draft_tokens,
        )
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cleanup.clear(
                req_pool_indices,
                seq_lens,
                accept_lens,
                num_draft_tokens=num_draft_tokens,
            )
        torch.cuda.synchronize()

        for state, original in zip(states, baseline):
            state.copy_(original)
        req_pool_indices.copy_(torch.tensor([2, 4], dtype=torch.int64, device="cuda"))
        seq_lens.copy_(torch.tensor([255, 511], dtype=torch.int64, device="cuda"))
        accept_lens.copy_(torch.tensor([1, 3], dtype=torch.int32, device="cuda"))

        expected = _reference_cleanup(
            baseline,
            req_pool_indices,
            seq_lens,
            accept_lens,
            num_draft_tokens=num_draft_tokens,
        )
        graph.replay()
        torch.cuda.synchronize()

        for layer_id, (actual, reference) in enumerate(zip(states, expected)):
            self.assertTrue(
                torch.equal(actual, reference),
                f"graph replay used stale metadata at layer {layer_id}",
            )

    @torch.inference_mode()
    def test_cleanup_rejects_invalid_state_lists(self):
        valid = torch.empty(_RING_SIZE, 1024, dtype=torch.float32, device="cuda")

        invalid_lists = [
            [],
            [torch.empty(_RING_SIZE, 1024, dtype=torch.float32)],
            [valid, torch.empty(_RING_SIZE, 1024, dtype=torch.float32)],
            [torch.empty_like(valid, dtype=torch.float16)],
            [valid, torch.empty_like(valid, dtype=torch.bfloat16)],
            [valid, torch.empty(_RING_SIZE, 1026, device="cuda")],
            [torch.empty(_RING_SIZE, 1023, device="cuda")],
            [torch.empty(1, _RING_SIZE, 1024, device="cuda")],
            [torch.empty(_RING_SIZE, 2048, device="cuda")[:, ::2]],
        ]

        for state_list in invalid_lists:
            description = [
                (tuple(state.shape), state.dtype, state.device, state.is_contiguous())
                for state in state_list
            ]
            with self.subTest(states=description):
                with self.assertRaises(ValueError):
                    C128DraftCleanup(state_list, ring_size=_RING_SIZE)

        with self.assertRaises(ValueError):
            C128DraftCleanup([valid], ring_size=0)

    @torch.inference_mode()
    def test_cleanup_owns_state_allocations(self):
        states = _make_states(
            3,
            num_req_slots=1,
            half=257,
            dtype=torch.float32,
            seed=4001,
        )
        state_refs = tuple(weakref.ref(state) for state in states)
        cleanup = C128DraftCleanup((state for state in states), ring_size=_RING_SIZE)

        del states
        gc.collect()

        self.assertTrue(all(state_ref() is not None for state_ref in state_refs))

        # Keep the owner live through the assertion; deleting it should release
        # the only remaining strong references to the tensors.
        del cleanup
        gc.collect()
        self.assertTrue(all(state_ref() is None for state_ref in state_refs))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""Correctness checks for the Hopper FP4 decode indexer."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import (
    fp4_index_logits_decode,
    fp4_index_logits_decode_paged,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_sm90 = unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90 (Hopper)",
)


@requires_sm90
class TestSM90FP4Indexer(CustomTestCase):
    def _inputs(self, batch_size: int, lmax: int, ratio: int):
        device = "cuda"
        num_requests = 11
        page_size = 64
        num_pages = 64
        row_padding = 17
        width = lmax * ratio
        generator = torch.Generator(device=device).manual_seed(
            1000 + batch_size * 100 + lmax * 3 + ratio
        )

        # This view has a padded row stride but unit inner stride. It mirrors the
        # pool contract without relying on a fully contiguous allocation.
        req_to_token_storage = torch.randint(
            0,
            num_pages * page_size * ratio,
            (num_requests, width + row_padding),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        req_to_token = req_to_token_storage[:, :width]
        req = torch.arange(batch_size, device=device, dtype=torch.int64) % 5
        visible_lengths = [
            0,
            min(1, lmax),
            max(lmax - 1, 0),
            lmax,
            min(63, lmax),
            min(64, lmax),
            min(511, lmax),
        ]
        lens = torch.tensor(
            (visible_lengths * ((batch_size + 6) // 7))[:batch_size],
            device=device,
            dtype=torch.int64,
        )
        if batch_size == 1:
            lens.fill_(lmax)
        q = torch.randn(
            batch_size,
            32,
            128,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        weights = torch.randn(
            batch_size,
            32,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        table = torch.randint(
            0,
            256,
            (num_pages, page_size * 68),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        # E8M0 scales with arbitrary bytes can intentionally overflow to inf;
        # use a finite range so exact output comparisons test the indexer math.
        table[:, page_size * 64 :] = torch.randint(
            124,
            129,
            (num_pages, page_size * 4),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        positions = torch.arange(lmax, device=device)
        slots = req_to_token[req[:, None], (positions * ratio)[None, :]].to(torch.int64)
        slots = (slots // ratio).masked_fill(positions[None, :] >= lens[:, None], 0)
        return q, weights, req_to_token, req, lens, table, slots, page_size

    def test_paged_matches_materialized_slot_map(self):
        for ratio in (1, 2):
            for batch_size in (1, 7, 64):
                for lmax in (1, 63, 64, 65, 513):
                    with self.subTest(ratio=ratio, batch_size=batch_size, lmax=lmax):
                        (
                            q,
                            weights,
                            req_to_token,
                            req,
                            lens,
                            table,
                            slots,
                            page_size,
                        ) = self._inputs(batch_size, lmax, ratio)
                        expected = fp4_index_logits_decode(
                            q, weights, slots, lens, table, page_size
                        )
                        actual = fp4_index_logits_decode_paged(
                            q,
                            weights,
                            req_to_token,
                            req,
                            ratio,
                            lmax,
                            lens,
                            table,
                            page_size,
                        )
                        torch.cuda.synchronize()
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int32), expected.view(torch.int32)
                            )
                        )

    def test_graph_replay_with_changed_lengths_and_request_ids(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                lmax = 513
                q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
                    7, lmax, ratio
                )
                lens.fill_(64)

                def paged():
                    return fp4_index_logits_decode_paged(
                        q,
                        weights,
                        req_to_token,
                        req,
                        ratio,
                        lmax,
                        lens,
                        table,
                        page_size,
                    )

                # Compile and warm up before entering graph capture.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        paged()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = paged()

                positions = torch.arange(lmax, device=q.device)
                for visible_length in (0, 1, 63, 64, 512, lmax):
                    lens.fill_(visible_length)
                    req.copy_(req.roll(1))
                    graph.replay()
                    slots = (
                        req_to_token[req[:, None], (positions * ratio)[None, :]].to(
                            torch.int64
                        )
                        // ratio
                    )
                    slots.masked_fill_(positions[None, :] >= lens[:, None], 0)
                    expected = fp4_index_logits_decode(
                        q, weights, slots, lens, table, page_size
                    )
                    torch.cuda.synchronize()
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int32), expected.view(torch.int32)
                        )
                    )

    def test_empty_output(self):
        for batch_size, lmax in ((0, 65), (7, 0)):
            with self.subTest(batch_size=batch_size, lmax=lmax):
                q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
                    batch_size, lmax, 1
                )
                actual = fp4_index_logits_decode_paged(
                    q, weights, req_to_token, req, 1, lmax, lens, table, page_size
                )
                self.assertEqual(actual.shape, (batch_size, lmax))
                self.assertEqual(actual.dtype, torch.float32)

    def test_paged_rejects_nonunit_req_to_token_inner_stride(self):
        q, weights, req_to_token, req, lens, table, _, page_size = self._inputs(
            1, 64, 1
        )
        strided = torch.empty(
            req_to_token.shape[0],
            req_to_token.shape[1] * 2,
            dtype=torch.int32,
            device="cuda",
        )[:, ::2]
        strided.copy_(req_to_token)
        self.assertEqual(strided.stride(1), 2)
        with self.assertRaises(AssertionError):
            fp4_index_logits_decode_paged(
                q, weights, strided, req, 1, 64, lens, table, page_size
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)

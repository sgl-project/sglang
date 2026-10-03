"""Tests for the SM90 grouped FP4 indexer."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSm90Fp4GroupedIndexer(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
        "requires an SM90 GPU",
    )
    def test_matches_triton(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            fp4_index_logits_decode,
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
            fp4_index_logits_grouped_sm90,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(11)
        heads, width, page_size = 64, 131, 64
        for group_size in (3, 6):
            rows = 2 * group_size
            q = fake_quant_fp4(
                torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
            )
            weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
            req = torch.arange(2, device="cuda", dtype=torch.int64).repeat_interleave(
                group_size
            )
            lens = torch.tensor(
                [width - i for _ in range(2) for i in range(group_size)],
                device="cuda",
                dtype=torch.int64,
            )
            slots = torch.stack(
                [torch.randperm(4 * page_size, device="cuda")[:width] for _ in range(2)]
            ).to(torch.int32)
            table = torch.empty((4, page_size * 68), device="cuda", dtype=torch.uint8)
            store_fp4_index_k_cache(
                torch.randn(
                    4 * page_size,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                table,
                torch.arange(4 * page_size, device="cuda", dtype=torch.int32),
                page_size=page_size,
                rne=True,
            )

            for ratio in (1, 2):
                with self.subTest(group_size=group_size, ratio=ratio):
                    req_to_token = torch.zeros(
                        2, width * ratio, device="cuda", dtype=torch.int32
                    )
                    req_to_token[:, ::ratio] = slots * ratio
                    explicit_slots = (
                        req_to_token[
                            req[:, None], torch.arange(width, device="cuda") * ratio
                        ]
                        // ratio
                    )
                    expected = fp4_index_logits_decode(
                        q, weights, explicit_slots, lens, table, page_size
                    )
                    actual = fp4_index_logits_grouped_sm90(
                        q,
                        weights,
                        req_to_token,
                        req,
                        lens,
                        table,
                        page_size,
                        ratio,
                        width,
                        group_size,
                    )
                    torch.testing.assert_close(actual, expected, equal_nan=True)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
        "requires an SM90 GPU",
    )
    def test_skips_invisible_group_tiles(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            fp4_index_logits_decode,
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
            fp4_index_logits_grouped_sm90,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(23)
        group_size, groups = 6, 2
        rows, heads, width, page_size = group_size * groups, 64, 4096, 64
        q = fake_quant_fp4(
            torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
        req = torch.arange(groups, device="cuda", dtype=torch.int64).repeat_interleave(
            group_size
        )
        lens = torch.tensor(
            [0, 1, 31, 62, 63, 17] + [0] * group_size,
            device="cuda",
            dtype=torch.int64,
        )
        table = torch.empty((2, page_size * 68), device="cuda", dtype=torch.uint8)
        store_fp4_index_k_cache(
            torch.randn(
                2 * page_size,
                128,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            table,
            torch.arange(2 * page_size, device="cuda", dtype=torch.int32),
            page_size=page_size,
            rne=True,
        )

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                req_to_token = torch.full(
                    (groups, width * ratio),
                    -ratio,
                    device="cuda",
                    dtype=torch.int32,
                )
                req_to_token[0, : 63 * ratio : ratio] = (
                    torch.arange(63, device="cuda", dtype=torch.int32) * ratio
                )
                explicit_slots = (
                    req_to_token[
                        req[:, None], torch.arange(width, device="cuda") * ratio
                    ]
                    // ratio
                )
                expected = fp4_index_logits_decode(
                    q, weights, explicit_slots, lens, table, page_size
                )
                actual = fp4_index_logits_grouped_sm90(
                    q,
                    weights,
                    req_to_token,
                    req,
                    lens,
                    table,
                    page_size,
                    ratio,
                    width,
                    group_size,
                )
                torch.testing.assert_close(actual, expected, equal_nan=True)
                invalid = torch.arange(width, device="cuda")[None, :] >= lens[:, None]
                self.assertTrue(torch.isneginf(actual[invalid]).all().item())

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
        "requires an SM90 GPU",
    )
    def test_graph_replay_with_changing_visibility(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            fp4_index_logits_decode,
            store_fp4_index_k_cache,
        )
        from sglang.kernels.ops.attention.dsv4.sm90_fp4_grouped_indexer import (
            fp4_index_logits_grouped_sm90,
        )
        from sglang.kernels.ops.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(37)
        groups, group_size, width, page_size, ratio = 17, 6, 4097, 16, 2
        rows = groups * group_size - 1
        q = fake_quant_fp4(
            torch.randn(rows, 64, 128, device="cuda", dtype=torch.bfloat16)
        )
        weights = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
        req = torch.arange(groups, device="cuda").repeat_interleave(group_size)[:rows]
        lens = torch.full((rows,), width, device="cuda", dtype=torch.int64)
        slots = torch.randint(128, (groups, width), device="cuda", dtype=torch.int32)
        req_to_token = torch.zeros(
            groups, width * ratio, device="cuda", dtype=torch.int32
        )
        req_to_token[:, ::ratio] = slots * ratio
        table = torch.empty((8, page_size * 68), device="cuda", dtype=torch.uint8)
        store_fp4_index_k_cache(
            torch.randn(128, 128, device="cuda", dtype=torch.bfloat16),
            table,
            torch.arange(128, device="cuda", dtype=torch.int32),
            page_size=page_size,
            rne=True,
        )

        def run(implementation):
            return fp4_index_logits_grouped_sm90(
                q,
                weights,
                req_to_token,
                req,
                lens,
                table,
                page_size,
                ratio,
                width,
                group_size,
                implementation=implementation,
            )

        for implementation in ("auto", "persistent", "pipeline"):
            run(implementation)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = run(implementation)
            for visible in (width, 0, 1, 63, 65, 513, width):
                with self.subTest(implementation=implementation, visible=visible):
                    lens.fill_(visible)
                    lens[::group_size] = 0
                    # Entire groups can disappear and return on later replays.
                    lens[group_size : 2 * group_size] = 0
                    req_to_token.fill_(-2147483647)
                    req_to_token[:, : visible * ratio : ratio] = (
                        slots[:, :visible] * ratio
                    )
                    expected = fp4_index_logits_decode(
                        q, weights, slots[req], lens, table, page_size
                    )
                    actual.fill_(123456)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(actual, expected, equal_nan=True)


if __name__ == "__main__":
    unittest.main()

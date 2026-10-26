import itertools
import unittest

import torch

from sglang.kernels.ops.memory.allocator import get_and_clear_swa_pages
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA or HIP")
class TestSwaPageFree(CustomTestCase):
    def test_resolve_and_clear_matches_reference(self):
        generator = torch.Generator().manual_seed(0)
        for page_size, num_pages, dtype, stride in itertools.product(
            (1, 4, 64, 128),
            (0, 1, 3, 65, 257),
            (torch.int32, torch.int64),
            (1, 3),
        ):
            with self.subTest(
                page_size=page_size,
                num_pages=num_pages,
                dtype=dtype,
                stride=stride,
            ):
                pool_pages = 2 * num_pages + 3
                pages = (
                    torch.randperm(pool_pages - 1, generator=generator)[:num_pages] + 1
                )
                representatives = pages * page_size + (
                    torch.arange(num_pages) % page_size
                )
                mapping_cpu = torch.arange(pool_pages * page_size) + 7 * page_size
                if num_pages > 1:
                    mapping_cpu[representatives[num_pages // 2]] = 0

                expected_peers = mapping_cpu[representatives]
                expected_mapping = mapping_cpu.clone()
                for page in pages.tolist():
                    expected_mapping[page * page_size : (page + 1) * page_size] = 0

                indices = torch.empty(num_pages * stride, dtype=dtype, device="cuda")[
                    ::stride
                ]
                indices.copy_(representatives)
                mapping = mapping_cpu.cuda()
                swa_pages, peers_mapped, page_mappings_valid = get_and_clear_swa_pages(
                    indices, mapping, page_size
                )

                self.assertIsNone(page_mappings_valid)
                self.assertTrue(
                    torch.equal(swa_pages.cpu(), expected_peers // page_size)
                )
                self.assertTrue(torch.equal(peers_mapped.cpu(), expected_peers > 0))
                self.assertTrue(torch.equal(mapping.cpu(), expected_mapping))

    def test_int32_mapping_last_page_before_sentinel(self):
        for page_size, check_page_mappings in itertools.product((1, 16), (False, True)):
            with self.subTest(
                page_size=page_size, check_page_mappings=check_page_mappings
            ):
                mapping_cpu = torch.arange(5 * page_size + 1, dtype=torch.int32)
                mapping_cpu[-1] = -1
                representative = mapping_cpu.numel() - 2
                expected_mapping = mapping_cpu.clone()
                expected_mapping[4 * page_size : 5 * page_size] = 0
                mapping = mapping_cpu.cuda()

                swa_pages, peers_mapped, page_mappings_valid = get_and_clear_swa_pages(
                    torch.tensor([representative], dtype=torch.int32, device="cuda"),
                    mapping,
                    page_size,
                    check_page_mappings=check_page_mappings,
                )

                self.assertEqual(swa_pages.dtype, torch.int32)
                self.assertEqual(swa_pages.item(), 4)
                self.assertTrue(peers_mapped.item())
                if check_page_mappings:
                    self.assertTrue(page_mappings_valid.item())
                self.assertTrue(torch.equal(mapping.cpu(), expected_mapping))

    def test_debug_rejects_out_of_bounds(self):
        mapping = torch.arange(33, device="cuda")
        expected_mapping = mapping.clone()
        for representative in (-1, 32, 33):
            with self.subTest(representative=representative):
                with self.assertRaisesRegex(
                    AssertionError, "FULL page representative out of bounds"
                ):
                    get_and_clear_swa_pages(
                        torch.tensor([representative], device="cuda"), mapping, 4, True
                    )
                self.assertTrue(torch.equal(mapping, expected_mapping))

    def test_page_mapping_validation(self):
        page_size = 4
        full_page = 3
        representative = full_page * page_size + 1
        swa_page = 7
        base_mapping = torch.zeros(12 * page_size, dtype=torch.int64)
        base_mapping[full_page * page_size : (full_page + 1) * page_size] = (
            torch.arange(swa_page * page_size, (swa_page + 1) * page_size)
        )
        base_mapping[full_page * page_size] = 0

        mixed_peer = full_page * page_size + 3
        for name, updates, expected_page, expected_peer, expected_valid in (
            ("valid", (), swa_page, True, True),
            ("missing_representative", ((representative, 0),), 0, False, False),
            (
                "multiple_peer_pages",
                ((mixed_peer, base_mapping[mixed_peer] + page_size),),
                swa_page,
                True,
                False,
            ),
        ):
            with self.subTest(name=name):
                mapping = base_mapping.clone()
                for index, value in updates:
                    mapping[index] = value
                expected_mapping = mapping.clone()
                expected_mapping[
                    full_page * page_size : (full_page + 1) * page_size
                ] = 0
                mapping = mapping.cuda()

                swa_pages, peers_mapped, page_mappings_valid = get_and_clear_swa_pages(
                    torch.tensor([representative], device="cuda"),
                    mapping,
                    page_size,
                    check_page_mappings=True,
                )

                self.assertEqual(swa_pages.item(), expected_page)
                self.assertEqual(peers_mapped.item(), expected_peer)
                self.assertIsNotNone(page_mappings_valid)
                self.assertEqual(page_mappings_valid.item(), expected_valid)
                self.assertTrue(torch.equal(mapping.cpu(), expected_mapping))


if __name__ == "__main__":
    unittest.main()

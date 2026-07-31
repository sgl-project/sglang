import unittest

import torch

from sglang.kernels.ops.kvcache.kv_indices import translate_full_to_swa_int32
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)


class TestTranslateFullToSwaInt32(CustomTestCase):
    def test_cpu_fallback(self):
        mapping = torch.arange(101, 202, dtype=torch.int64)
        mapping[-1] = -7
        full_locs = torch.tensor([0, 99, 4, -1], dtype=torch.int64)

        actual = translate_full_to_swa_int32(mapping, full_locs)

        self.assertEqual(actual.dtype, torch.int32)
        self.assertTrue(torch.equal(actual, mapping[full_locs].to(torch.int32)))

    def test_cuda_parity_for_dtypes_and_strided_locations(self):
        for mapping_dtype in (torch.int32, torch.int64):
            for loc_dtype in (torch.int32, torch.int64):
                mapping = torch.arange(
                    101,
                    202,
                    dtype=mapping_dtype,
                    device="cuda",
                )
                mapping[-1] = -7
                full_locs_base = torch.tensor(
                    [0, 100, 4, 98, 17, 97, -1, 96],
                    dtype=loc_dtype,
                    device="cuda",
                )
                full_locs = full_locs_base[::2]

                actual = translate_full_to_swa_int32(mapping, full_locs)
                expected = mapping[full_locs].to(torch.int32)

                with self.subTest(
                    mapping_dtype=mapping_dtype,
                    loc_dtype=loc_dtype,
                ):
                    self.assertEqual(actual.dtype, torch.int32)
                    self.assertTrue(torch.equal(actual, expected))

    def test_empty_cuda_input(self):
        mapping = torch.tensor([3, 5, -1], dtype=torch.int64, device="cuda")
        full_locs = torch.empty(0, dtype=torch.int64, device="cuda")

        actual = translate_full_to_swa_int32(mapping, full_locs)

        self.assertEqual(actual.shape, (0,))
        self.assertEqual(actual.dtype, torch.int32)
        self.assertEqual(actual.device, full_locs.device)

    def test_cuda_multi_block_tail(self):
        mapping = torch.arange(2049, dtype=torch.int64, device="cuda")
        mapping[-1] = -1
        full_locs_base = torch.arange(1026, dtype=torch.int32, device="cuda")
        full_locs_base[-2] = -1
        full_locs = full_locs_base[::2]

        actual = translate_full_to_swa_int32(mapping, full_locs)

        self.assertEqual(full_locs.numel(), 513)
        self.assertTrue(torch.equal(actual, mapping[full_locs].to(torch.int32)))

    def test_cuda_graph_replay_reads_live_inputs(self):
        mapping = torch.arange(257, dtype=torch.int64, device="cuda")
        mapping[-1] = -1
        full_locs = torch.tensor([1, 2, 3, -1], dtype=torch.int64, device="cuda")

        translate_full_to_swa_int32(mapping, full_locs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            translated = translate_full_to_swa_int32(mapping, full_locs)

        output_ptr = translated.data_ptr()
        for mapping_offset, replay_full_locs in (
            (1000, [7, 8, 9, 10]),
            (2000, [100, 200, 255, -1]),
        ):
            next_mapping = torch.arange(
                mapping_offset,
                mapping_offset + mapping.numel(),
                dtype=mapping.dtype,
                device=mapping.device,
            )
            next_mapping[-1] = -1
            mapping.copy_(next_mapping)
            full_locs.copy_(
                torch.tensor(
                    replay_full_locs,
                    dtype=full_locs.dtype,
                    device=full_locs.device,
                )
            )

            graph.replay()

            self.assertTrue(
                torch.equal(
                    translated,
                    mapping[full_locs].to(torch.int32),
                )
            )
            self.assertEqual(translated.data_ptr(), output_ptr)


if __name__ == "__main__":
    unittest.main()

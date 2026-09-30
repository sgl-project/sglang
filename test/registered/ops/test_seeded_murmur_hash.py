"""Device regression tests for seeded MurmurHash and Gumbel sampling."""

import math
import unittest

import torch

from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.utils.common import is_npu
from sglang.test.ci.ci_register import register_cuda_ci, register_npu_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
register_npu_ci(est_time=30, suite="base-b-test-1-npu-a3")

MASK = (1 << 32) - 1


def reference_hash(seed: int, position: int, column: int) -> int:
    seed &= (1 << 64) - 1
    value = 0
    for key in (seed & MASK, seed >> 32, position & MASK, column & MASK):
        key = (key * 0xCC9E2D51) & MASK
        key = ((key << 15) | (key >> 17)) & MASK
        key = (key * 0x1B873593) & MASK
        value ^= key
        value = ((value << 13) | (value >> 19)) & MASK
        value = (value * 5 + 0xE6546B64) & MASK
    value ^= 16
    value ^= value >> 16
    value = (value * 0x85EBCA6B) & MASK
    value ^= value >> 13
    value = (value * 0xC2B2AE35) & MASK
    return value ^ (value >> 16)


@unittest.skipUnless(is_npu() or torch.cuda.is_available(), "requires NPU or CUDA")
class TestSeededMurmurHash(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "npu" if is_npu() else "cuda"
        self.seeds = torch.tensor(
            [0, 1, -1, -(2**63), 2**63 - 1, 2**32, 2**40], dtype=torch.long
        )
        self.positions = torch.tensor(
            [1707985137, 2**31 - 1, 2**31, MASK, 2**32, -1, 2**40]
        )
        self.columns = torch.tensor([0, 1, 2**31 - 1, 2**31, MASK, 2**32, 2**40])

    def expected_hashes(self, seeds: torch.Tensor) -> list[list[int]]:
        return [
            [reference_hash(int(seed), int(pos), int(col)) for col in self.columns]
            for seed, pos in zip(seeds, self.positions)
        ]

    def test_integer_boundaries(self) -> None:
        actual = murmur_hash32(
            self.seeds.to(self.device).view(torch.uint64),
            self.positions.to(self.device),
            self.columns.to(self.device),
        )
        self.assertEqual(actual.cpu().tolist(), self.expected_hashes(self.seeds))

    def test_graph_replay_reads_updated_seeds(self) -> None:
        module = torch.get_device_module(self.device)
        seeds = self.seeds.to(self.device).view(torch.uint64)
        positions = self.positions.to(self.device)
        columns = self.columns.to(self.device)
        for _ in range(3):
            murmur_hash32(seeds, positions, columns)
        module.synchronize()
        graph = module.NPUGraph() if self.device == "npu" else module.CUDAGraph()
        with module.graph(graph):
            actual = murmur_hash32(seeds, positions, columns)
        for value in (0, 1234, -1):
            self.seeds[0] = value
            seeds.copy_(self.seeds.view(torch.uint64))
            graph.replay()
            module.synchronize()
            self.assertEqual(actual.cpu().tolist(), self.expected_hashes(self.seeds))

    def test_sampler_preserves_columns_and_double_noise(self) -> None:
        scores = torch.randn(7, 7, generator=torch.Generator().manual_seed(73))
        scores[0, 0] = -torch.inf
        for name, columns in (
            ("default", torch.arange(scores.shape[1])),
            ("compact", self.columns),
        ):
            with self.subTest(columns=name):
                expected = []
                for row, seed in enumerate(self.seeds):
                    values = []
                    for col, column in enumerate(columns):
                        hashed = reference_hash(
                            int(seed), int(self.positions[row]), int(column)
                        )
                        uniform = hashed / MASK
                        log_uniform = math.log(uniform) if uniform else -math.inf
                        log_uniform = max(
                            min(log_uniform, -(2.0**-32)),
                            -torch.finfo(torch.float64).max,
                        )
                        values.append(float(scores[row, col]) - math.log(-log_uniform))
                    expected.append(max(range(len(values)), key=values.__getitem__))
                if name == "default":
                    actual = multinomial_with_seed(
                        scores.to(self.device),
                        self.seeds.to(self.device),
                        self.positions.to(self.device),
                    )
                else:
                    actual = multinomial_with_seed(
                        scores.to(self.device),
                        self.seeds.to(self.device),
                        self.positions.to(self.device),
                        token_ids=columns.to(self.device),
                    )
                self.assertEqual(actual.view(-1).cpu().tolist(), expected)


if __name__ == "__main__":
    unittest.main()

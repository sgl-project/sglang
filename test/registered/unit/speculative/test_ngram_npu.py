"""CPU oracle for Ascend's device-native NGRAM tree metadata."""

import unittest

import torch

from sglang.srt.speculative.ngram_npu import reconstruct_indices_from_tree_mask_npu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNgramNpuMetadata(CustomTestCase):
    def test_tree_links_against_parent_walk_oracle(self):
        for parents in ([[-1, 0, 1, 2, 3]], [[-1, 0, 0, 1, 2], [-1, 0, 1, -1, -1]]):
            batch_size, width = len(parents), len(parents[0])
            tree = torch.zeros(batch_size, width, width, dtype=torch.bool)
            prefix_lens = torch.tensor([257 + b * 131 for b in range(batch_size)])
            expected_positions = torch.empty(batch_size, width, dtype=torch.int64)
            expected_next = torch.full_like(expected_positions, -1)
            expected_sibling = torch.full_like(expected_positions, -1)
            for batch, links in enumerate(parents):
                for node in range(width):
                    depth, ancestor = 0, node
                    while ancestor >= 0:
                        tree[batch, node, ancestor] = True
                        ancestor = links[ancestor]
                        depth += 1
                    expected_positions[batch, node] = prefix_lens[batch] + depth - 1
                    children = [
                        child
                        for child in range(node + 1, width)
                        if links[child] == node
                    ]
                    if children:
                        expected_next[batch, node] = min(children)
                    if links[node] >= 0:
                        siblings = [
                            sibling
                            for sibling in range(node + 1, width)
                            if links[sibling] == links[node]
                        ]
                        if siblings:
                            expected_sibling[batch, node] = min(siblings)
            positions = torch.empty(batch_size * width, dtype=torch.int64)
            retrieve = torch.empty(batch_size, width, dtype=torch.int64)
            next_nodes, siblings = (
                torch.empty_like(retrieve),
                torch.empty_like(retrieve),
            )
            reconstruct_indices_from_tree_mask_npu(
                tree.flatten(),
                prefix_lens,
                positions,
                retrieve,
                next_nodes,
                siblings,
                batch_size,
                width,
            )
            torch.testing.assert_close(
                positions.view(batch_size, width), expected_positions
            )
            torch.testing.assert_close(next_nodes, expected_next)
            torch.testing.assert_close(siblings, expected_sibling)
            torch.testing.assert_close(
                retrieve, torch.arange(batch_size * width).view(batch_size, width)
            )


if __name__ == "__main__":
    unittest.main()

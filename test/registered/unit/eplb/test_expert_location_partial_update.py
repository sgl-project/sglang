"""Layer commits preserve untouched rows and all captured map addresses."""

import unittest

import torch

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def metadata(physical):
    physical = torch.tensor(physical)
    logical = torch.full((3, 4, 6), -1, dtype=torch.long)
    for layer in range(3):
        for expert in range(4):
            slots = (physical[layer] == expert).nonzero().flatten()
            logical[layer, expert, : len(slots)] = slots
    return ExpertLocationMetadata._init_raw(2, physical, logical, moe_ep_rank=0)


class TestPartialUpdate(unittest.TestCase):
    def test_exact_rows_and_cached_graph_pointers(self):
        for algorithm in ("static", "dynamic"):
            for selected in ([], [1], [2, 0], [0, 1, 2], [1, 1, -1, 99]):
                with self.subTest(algorithm=algorithm, selected=selected):
                    with get_context().override_server_args(
                        ep_dispatch_algorithm=algorithm, ep_size=2, nnodes=1
                    ):
                        current = metadata([[0, 1, 2, 3, 0, 1]] * 3)
                        incoming = metadata([[3, 2, 1, 0, 3, 2]] * 3)
                    gate_map, gate_count = current.mega_gate_maps(algorithm)
                    fields = {
                        name: tensor
                        for name, tensor in vars(current).items()
                        if isinstance(tensor, torch.Tensor)
                    }
                    before = {name: value.clone() for name, value in fields.items()}
                    pointers = {
                        name: value.data_ptr() for name, value in fields.items()
                    }
                    cache_pointers = (gate_map.data_ptr(), gate_count.data_ptr())
                    current.update(incoming, selected)
                    for name, value in fields.items():
                        expected = before[name]
                        for layer in range(3):
                            if layer in selected:
                                expected[layer] = getattr(incoming, name)[layer]
                        torch.testing.assert_close(value, expected)
                        self.assertEqual(value.data_ptr(), pointers[name])
                    self.assertEqual(
                        cache_pointers, (gate_map.data_ptr(), gate_count.data_ptr())
                    )
                    if algorithm == "static":
                        torch.testing.assert_close(
                            gate_map[..., 0],
                            current.logical_to_rank_dispatch_physical_map.to(
                                torch.int32
                            ),
                        )
                        self.assertTrue(torch.all(gate_count == 1))
                    else:
                        torch.testing.assert_close(
                            gate_map,
                            current.logical_to_all_physical_map.to(torch.int32),
                        )
                        torch.testing.assert_close(
                            gate_count,
                            current.logical_to_all_physical_map_num_valid.to(
                                torch.int32
                            ),
                        )


if __name__ == "__main__":
    unittest.main()

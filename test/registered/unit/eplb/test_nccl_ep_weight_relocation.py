"""Real Gloo expert migration on CPU, including replicas and layer chunks."""

import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _relocate(rank, rendezvous):
    from nccl_ep_test.eplb import metadata, tensor_addresses

    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
    from sglang.srt.layers.moe.utils import MoeA2ABackend
    from sglang.srt.runtime_context import get_flags, get_resources

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        layouts = [
            [[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]],
            [[3, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 1]],
            [[1, 2, 3, 0, 1, 2], [0, 1, 2, 3, 2, 3]],
        ]
        old = metadata(layouts[0], ep_size=2, rank=rank, device="cpu")
        get_resources().expert_location_metadata = old
        # Byte payloads exercise packed/FP8 weight transport; float tensors
        # stand in for independent scale blocks and must move with the weights.
        logical = [
            [
                (torch.arange(24).reshape(4, 2, 3) + 40 * layer).to(torch.uint8),
                torch.arange(8).reshape(4, 2).float() * 0.25 + layer,
            ]
            for layer in range(2)
        ]
        weights = {
            layer: [
                value[
                    old.physical_to_logical_map[layer, rank * 3 : (rank + 1) * 3]
                ].clone()
                for value in logical[layer]
            ]
            for layer in range(2)
        }
        addresses = tensor_addresses(old, weights)
        updater = ExpertLocationUpdater()
        # The CPU path has no GPU cache or capture state to inspect. Actual
        # metadata mutation, P2P scheduling, sends/receives and copies are real.
        updater._first_execution = False
        with get_flags().moe.override(a2a_backend=MoeA2ABackend.NCCL_EP), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            for mapping in layouts[1:] + layouts[:1]:
                target = metadata(mapping, ep_size=2, rank=rank, device="cpu")
                for layer in range(2):
                    assert (
                        updater.update(weights, target, [layer], nnodes=1, rank=rank)
                        == {}
                    )
                    # Every layer remains internally consistent, including the
                    # not-yet-updated layer between successive chunk updates.
                    for check_layer in range(2):
                        slots = old.physical_to_logical_map[
                            check_layer, rank * 3 : (rank + 1) * 3
                        ]
                        for actual, source in zip(
                            weights[check_layer], logical[check_layer]
                        ):
                            torch.testing.assert_close(
                                actual, source[slots], rtol=0, atol=0
                            )
                        mapping_for_rank = old.logical_to_rank_dispatch_physical_map[
                            check_layer
                        ]
                        torch.testing.assert_close(
                            old.physical_to_logical_map[check_layer, mapping_for_rank],
                            torch.arange(4),
                            rtol=0,
                            atol=0,
                        )
                    assert tensor_addresses(old, weights) == addresses
                dist.barrier()
    finally:
        dist.destroy_process_group()


class TestNcclEpWeightRelocation(CustomTestCase):
    def test_two_rank_relocation_preserves_weights_scales_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _relocate,
                args=((Path(directory) / "rendezvous").as_uri(),),
                nprocs=2,
                join=True,
            )

    def test_physical_count_oracle_includes_idle_and_duplicate_routes(self):
        from nccl_ep_test.eplb import received_counts

        mapping = [[0, 1, 2, 3, 0, 1]]
        empty = torch.empty((0, 2), dtype=torch.int64)
        for routes, expected in (
            ((empty, torch.tensor([[0, 3]])), [[0, 0, 0, 1, 1, 0]]),
            ((torch.tensor([[1, 1]]), empty), [[0, 2, 0, 0, 0, 0]]),
            ((torch.tensor([[-1, -1]]), empty), [[0, 0, 0, 0, 0, 0]]),
        ):
            torch.testing.assert_close(
                received_counts(mapping, routes),
                torch.tensor(expected, dtype=torch.int32),
            )


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

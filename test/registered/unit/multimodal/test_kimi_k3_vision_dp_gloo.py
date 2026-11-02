"""Vision-DP validation must not leave an empty peer in a collective."""

import math
import time
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


class _GlooVisionGroup:
    """Use real CPU collectives through the vision helper's group interface."""

    def broadcast(self, tensor, src):
        dist.broadcast(tensor, src=src)

    def all_gather(self, tensor, dim):
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)


class _CPUVisionTower(torch.nn.Module):
    device = torch.device("cpu")
    merge_kernel_size = (2, 2)
    patch_size = 2

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2)
        self.patch_embed = SimpleNamespace(
            proj=SimpleNamespace(weight=torch.empty(1, dtype=torch.float32))
        )
        self.calls = 0

    def forward(self, pixels, *, grid_hw, grid_thw_list, max_seqlen):
        self.calls += 1
        assert grid_hw.tolist() == grid_thw_list
        assert max_seqlen == max(math.prod(grid) for grid in grid_thw_list)
        outputs = []
        offset = 0
        for temporal, height, width in grid_thw_list:
            # Like K3's tower, pool time away and preserve the packed 2x2 axes.
            value = pixels[offset, 0].item()
            outputs.append(torch.full((height * width // 4, 4, 2), value))
            offset += temporal * height * width
        assert offset == len(pixels)
        return outputs


def _run_vision_rank(rank, init_file):
    torch.set_num_threads(1)
    # Keep optional CUDA kernels out of a test that only runs CPU control flow.
    maybe_stub_sgl_kernel()
    from sglang.srt.environ import envs
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
    from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration
    from sglang.srt.runtime_context import get_parallel

    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        model = KimiK3ForConditionalGeneration.__new__(KimiK3ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.use_data_parallel = True
        model.vision_tower = _CPUVisionTower()
        model.mm_projector = torch.nn.Identity()

        def item(grid, value):
            return MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=torch.full((math.prod(grid), 2), value),
                model_specific_data={"image_grid_thw": torch.tensor([grid])},
            )

        with (
            envs.SGLANG_K3_VISION_MAX_FRAMES_PER_BATCH.override(3),
            get_parallel().override(
                attn_tp_size=2, attn_tp_rank=rank, attn_tp_group=_GlooVisionGroup()
            ),
        ):
            try:
                model.get_image_feature([item([4, 2, 2], 5.0)])
            except ValueError as exc:
                message = str(exc)
                assert "temporal item does not fit" in message
            else:
                raise AssertionError("oversized temporal item was accepted")

            messages = [None, None]
            dist.all_gather_object(messages, message)
            assert messages == [message, message]
            assert model.vision_tower.calls == 0

            # The same ranks must still finish a broadcast with one empty peer.
            actual = model.get_image_feature([item([2, 2, 2], 7.0)])
            torch.testing.assert_close(actual, torch.full((1, 4, 2), 7.0))

            # Uneven rank outputs exercise padding, a real gather and order repair.
            actual = model.get_image_feature(
                [
                    item([3, 2, 2], 11.0),
                    item([1, 4, 2], 13.0),
                    item([2, 2, 2], 17.0),
                ]
            )
            expected = torch.cat(
                [
                    torch.full((1, 4, 2), 11.0),
                    torch.full((2, 4, 2), 13.0),
                    torch.full((1, 4, 2), 17.0),
                ]
            )
            torch.testing.assert_close(actual, expected)
            dist.barrier()
    finally:
        dist.destroy_process_group()


class TestKimiK3VisionDPGloo(CustomTestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs Gloo")
    def test_invalid_budget_then_valid_requests_finish_on_all_ranks(self):
        context = mp.get_context("spawn")
        processes = []
        with TemporaryDirectory() as directory:
            try:
                for rank in range(2):
                    process = context.Process(
                        target=_run_vision_rank,
                        args=(rank, str(Path(directory) / "gloo-init")),
                    )
                    process.start()
                    processes.append(process)
                deadline = time.monotonic() + 60
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(
                    any(process.is_alive() for process in processes),
                    "a vision-DP rank did not finish within 60 seconds",
                )
                self.assertEqual([process.exitcode for process in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join(5)
                    if process.is_alive():
                        process.kill()
                        process.join(5)
                    process.close()


if __name__ == "__main__":
    unittest.main()

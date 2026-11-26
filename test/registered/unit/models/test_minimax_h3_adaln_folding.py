"""AdaLN checkpoint loading and gathers must use its group, not DiT TP."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, stage="base-a", runner_config="cpu")


class TestAdaLNFolding(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy", device="cpu"), role="test")
        self.addCleanup(reset_context)

    def test_batch_adaln_uses_resolved_group_size(self):
        for dit_tp_size, adaln_tp_size in ((1, 1), (2, 2), (1, 4), (2, 4)):
            with self.subTest(dit_tp_size=dit_tp_size, adaln_tp_size=adaln_tp_size):
                model = SimpleNamespace(
                    adaln_cache=None,
                    adaln_parallel_group=SimpleNamespace(world_size=adaln_tp_size),
                    blocks=[],
                )
                with (
                    patch.object(h3, "get_tp_world_size", return_value=dit_tp_size),
                    patch.object(torch.compiler, "is_compiling", return_value=False),
                    patch.object(h3.envs, "SGLANG_CACHE_DIT_ENABLED", False),
                    patch.object(
                        h3, "is_layerwise_offloaded_module", return_value=False
                    ),
                ):
                    self.assertEqual(
                        h3.MiniMaxH3DiTModel._can_batch_block_adaln(model),
                        adaln_tp_size > 1,
                    )

    def test_folded_checkpoint_and_output_order(self):
        arch = SimpleNamespace(hidden_size=8, time_embed_dim=4, adaln_curve_grid=None)
        for modalities, expand in ((3, 6), (1, 2)):
            width = modalities * expand * arch.hidden_size
            weight = torch.arange(width * 4).reshape(width, 4).to(torch.bfloat16) / 128
            bias = torch.arange(width).to(torch.bfloat16) / 64
            x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.bfloat16)
            projections = []
            outputs = []
            for rank in range(4):
                group = SimpleNamespace(rank_in_group=rank, world_size=4)

                with patch(
                    "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
                    return_value=SimpleNamespace(rank_in_group=rank % 2, world_size=2),
                ):
                    projection = h3.MiniMaxH3AdalnProj(
                        arch,
                        width,
                        None,
                        prefix="adaln_proj",
                        expand_ratio=expand,
                        modality_num=modalities,
                        parallel_group=group,
                    )
                    # The ambient DiT group is TP2; the projection owns TP4.
                    projection.linear.weight_loader(projection.linear.weight, weight)
                    projection.linear.weight_loader(projection.linear.bias, bias)
                    torch.testing.assert_close(
                        projection.linear.weight,
                        weight.chunk(4, dim=0)[rank],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        projection.linear.bias, bias.chunk(4)[rank], rtol=0, atol=0
                    )
                    outputs.append(projection.project_local(x))
                    projections.append(projection)
            gathered = torch.cat(outputs, dim=-1)
            reference = torch.nn.functional.linear(x, weight, bias)
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)
            expected = reference.view(2 * modalities, expand * arch.hidden_size).chunk(
                expand, dim=-1
            )
            for projection in projections:

                def gather(local, dim):
                    torch.testing.assert_close(
                        local, projection.project_local(x), rtol=0, atol=0
                    )
                    return torch.cat(outputs, dim=dim)

                projection.parallel_group.all_gather = gather
                for actual, target in zip(projection(x), expected):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

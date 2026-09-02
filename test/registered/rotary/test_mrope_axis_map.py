import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
    apply_interleaved_rope,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def build_mrope(
    mrope_section: list[int],
    rotary_dim: int,
    interleaved: bool = False,
    glm: bool = False,
) -> MRotaryEmbedding:
    return MRotaryEmbedding(
        head_size=rotary_dim,
        rotary_dim=rotary_dim,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=mrope_section,
        mrope_interleaved=interleaved,
        mrope_interleaved_glm=glm,
    )


def select_by_axis(table: torch.Tensor, axis_map: torch.Tensor) -> torch.Tensor:
    """Take every lane from the axis the map names, the way the kernel does."""
    lanes = torch.arange(table.shape[2])
    return table[axis_map, :, lanes].T


class TestMRopeAxisMap(CustomTestCase):
    def setUp(self):
        cpu_patch = patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        cpu_patch.start()
        self.addCleanup(cpu_patch.stop)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)

    def test_interleaved_matches_apply_interleaved_rope(self):
        # Under [1, 1, 30] the reference places only 10 of the 30 lanes it was asked
        # for, so this pins that the map reproduces that loss.
        for section, rotary_dim in (
            ([24, 20, 20], 128),
            ([11, 11, 10], 64),
            ([1, 1, 30], 64),
        ):
            with self.subTest(section=section):
                rope = build_mrope(section, rotary_dim, interleaved=True)
                table = torch.randn(3, 7, rotary_dim // 2)
                torch.testing.assert_close(
                    select_by_axis(table, rope.axis_map),
                    apply_interleaved_rope(table, section),
                    atol=0,
                    rtol=0,
                )

    def test_contiguous_matches_section_split(self):
        section, rotary_dim = [24, 20, 20], 128
        rope = build_mrope(section, rotary_dim)
        table = torch.randn(3, 7, rotary_dim // 2)
        torch.testing.assert_close(
            select_by_axis(table, rope.axis_map),
            torch.cat(
                [m[i] for i, m in enumerate(table.split(section, dim=-1))], dim=-1
            ),
            atol=0,
            rtol=0,
        )

    def test_glm_map_keeps_its_round_robin_order(self):
        """GLM's kernel is out of tree, so the order is pinned rather than compared."""
        rope = build_mrope([8, 12, 12], 64, interleaved=True, glm=True)
        want = [0, 1, 2] * 8 + [1, 1, 2, 1, 1, 2, 2, 2]
        self.assertEqual(rope.axis_map.tolist(), want)

    def test_only_glm_reaches_the_older_kernels(self):
        self.assertIsNone(
            build_mrope([24, 20, 20], 128, interleaved=True)._legacy_axis_map
        )
        glm = build_mrope([8, 12, 12], 64, interleaved=True, glm=True)
        torch.testing.assert_close(glm._legacy_axis_map, glm.axis_map, atol=0, rtol=0)

    def test_ernie_has_no_map(self):
        ernie = Ernie4_5_VLRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=64,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
            mrope_section=[16, 16, 32],
        )
        self.assertIsNone(ernie.axis_map)


if __name__ == "__main__":
    unittest.main()

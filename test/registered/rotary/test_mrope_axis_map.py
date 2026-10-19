import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.rotary_embedding import mrope
from sglang.srt.layers.rotary_embedding.mrope import (
    Ernie4_5_VLRotaryEmbedding,
    MRotaryEmbedding,
    apply_interleaved_rope,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def test_npu_dispatch_preserves_multimodal_positions(self):
        rope = build_mrope([11, 11, 10], 64, interleaved=True)
        rope.head_size = 256
        positions = torch.tensor([[0, 1], [2, 3], [4, 5]])
        query, key = torch.randn(2, 3072), torch.randn(2, 512)
        expected = rope.forward_native(positions, query, key)
        with patch.object(mrope, "torch_npu", create=True) as npu:
            npu.npu_mrope.return_value = (query, key)
            actual = rope.forward_npu(positions, query, key)
            npu.npu_mrope.assert_not_called()
            torch.testing.assert_close(actual, expected)
            text_positions = positions[0]
            rope.forward_npu(text_positions, query, key)
            npu.npu_mrope.assert_called_once_with(
                text_positions,
                query,
                key,
                rope.cos_sin_cache,
                256,
                mrope_section=[0, 0, 0],
                rotary_mode="half",
            )

    def test_qwen35_npu_prefill_and_decode_dispatch(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import qwen3_5

        layer = SimpleNamespace(
            attn_output_gate=True,
            forward_prepare_npu_prefill=Mock(return_value="prefill"),
            forward_prepare_npu=Mock(return_value="decode"),
        )
        with patch.multiple(
            qwen3_5,
            _is_npu=True,
            _is_cuda=False,
            _is_hip=False,
            _is_cpu=False,
            _is_xpu=False,
        ):
            for mode, expected in (
                (ForwardMode.EXTEND, "prefill"),
                (ForwardMode.DECODE, "decode"),
            ):
                result = qwen3_5.Qwen3_5AttentionDecoderLayer._prepare_qkv_gate(
                    layer, None, None, SimpleNamespace(forward_mode=mode)
                )
                self.assertEqual(result, expected)

    def test_qwen35_npu_prefill_mrope_contract(self):
        from sglang.srt.models import qwen3_5

        # Qwen3.5-27B at TP2: 12 Q/gate heads, 2 K/V heads, partial RoPE.
        qkv = torch.empty(2, 28 * 256, dtype=torch.bfloat16)
        rope = build_mrope([11, 11, 10], 64, interleaved=True)
        layer = SimpleNamespace(
            qkv_proj=Mock(return_value=(qkv, None)),
            rotary_emb=rope,
            num_heads=12,
            num_kv_heads=2,
            head_dim=256,
            attn_output_gate=True,
            q_norm=SimpleNamespace(
                gemma_weight=qkv.new_full((256,), 1.25), variance_epsilon=1e-6
            ),
            k_norm=SimpleNamespace(gemma_weight=qkv.new_full((256,), 0.75)),
        )
        multimodal = torch.tensor([[0, 1], [2, 3], [4, 5]])
        with patch.object(
            qwen3_5,
            "triton_split_qkv_rmsnorm_mrope",
            create=True,
            return_value=(None, None, None, None),
        ) as fused:
            for positions in (multimodal, multimodal[0]):
                qwen3_5.Qwen3_5AttentionDecoderLayer.forward_prepare_npu_prefill(
                    layer, positions, None
                )
                args = fused.call_args.kwargs.copy()
                self.assertIs(args.pop("qkv"), qkv)
                self.assertIs(args.pop("q_weight"), layer.q_norm.gemma_weight)
                self.assertIs(args.pop("k_weight"), layer.k_norm.gemma_weight)
                torch.testing.assert_close(
                    args.pop("cos_sin"),
                    rope.cos_sin_cache[positions.expand(3, -1)].to(qkv),
                )
                self.assertEqual(
                    args,
                    dict(
                        num_q_heads=12,
                        num_kv_heads=2,
                        head_size=256,
                        eps=1e-6,
                        mrope_section=[11, 11, 10],
                        is_interleaved=True,
                        rope_dim=64,
                        has_gate=True,
                    ),
                )


if __name__ == "__main__":
    unittest.main()

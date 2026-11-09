import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16TritonMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HIDDEN = 128
INTERMEDIATE = 640
GROUP = 128


def _triton_scheme(*, sym=True):
    # Bypass __init__, which needs a full CompressedTensorsConfig.
    scheme = CompressedTensorsWNA16TritonMoE.__new__(CompressedTensorsWNA16TritonMoE)
    scheme.group_size = GROUP
    scheme.sym = sym
    scheme.actorder = None
    scheme.strategy = "group"
    scheme.packed_factor = 8
    return scheme


def _fused_moe_shell(*, tp_size, tp_rank, experts):
    """FusedMoE with real loader methods but no distributed/device init."""
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    for name, value in dict(
        moe_tp_size=tp_size,
        moe_tp_rank=tp_rank,
        intermediate_size_per_partition=INTERMEDIATE // tp_size,
        use_presharded_weights=False,
        use_padded_loading=False,
        use_triton_kernels=False,
        use_flashinfer_trtllm_moe=False,
        uses_expert_location=False,
        _has_fused_shared=False,
        _expert_storage_rank=0,
        _num_local_routed=experts,
        _num_global_routed=experts,
        num_local_experts=experts,
        layer_id=0,
        quant_config=None,
        moe_runner_config=SimpleNamespace(is_gated=True),
    ).items():
        setattr(layer, name, value)
    return layer


def _load_w2(*, tp_size, tp_rank, packed, scales, experts):
    """Load w2 through the real FusedMoE loader and return the Triton quant info."""
    layer = _fused_moe_shell(tp_size=tp_size, tp_rank=tp_rank, experts=experts)
    scheme = _triton_scheme()
    layer.scheme = layer.quant_method = scheme
    scheme.create_weights(
        layer,
        experts,
        HIDDEN,
        INTERMEDIATE // tp_size,
        torch.float32,
        weight_loader=layer.weight_loader,
    )
    for expert in range(experts):
        for param, source, name in (
            (layer.w2_weight_packed, packed + expert, "w2_weight_packed"),
            (layer.w2_weight_scale, scales + expert, "w2_weight_scale"),
        ):
            param.weight_loader(param, source, name, "w2", expert)
    layer.w13_weight_packed.data.zero_()
    scheme.process_weights_after_loading(layer)
    return scheme.get_triton_quant_info(layer)


class TestWNA16TritonMoETPSplitScales(CustomTestCase):
    def test_each_k_element_gets_its_checkpoint_group_scale(self):
        """Every local K element of w2 must get its own checkpoint group's scale.

        At moe_tp_size=4, K=640 shards to 160 = 128 + 32. Narrowing the 5 checkpoint
        groups by integer division gave each rank the scales of the wrong K range.
        """
        experts = 2
        packed = torch.arange(HIDDEN * INTERMEDIATE // 8, dtype=torch.int32).view(
            HIDDEN, INTERMEDIATE // 8
        )
        scales = torch.arange(HIDDEN * INTERMEDIATE // GROUP, dtype=torch.float32)
        scales = scales.view(HIDDEN, INTERMEDIATE // GROUP) + 1
        for tp_size in (1, 2, 4, 8):
            shard = INTERMEDIATE // tp_size
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    info = _load_w2(
                        tp_size=tp_size,
                        tp_rank=rank,
                        packed=packed,
                        scales=scales,
                        experts=experts,
                    )
                    local_group = info.w2_block_shape[1]
                    self.assertEqual(shard % local_group, 0)
                    self.assertEqual(GROUP % local_group, 0)
                    k_range = slice(rank * shard, (rank + 1) * shard)
                    for expert in range(experts):
                        expected_packed = (packed + expert)[
                            :, rank * shard // 8 : (rank + 1) * shard // 8
                        ]
                        torch.testing.assert_close(
                            info.w2_weight[expert],
                            expected_packed.contiguous().view(torch.uint8),
                        )
                        per_element = info.w2_scale[expert].repeat_interleave(
                            local_group, dim=1
                        )
                        expected = (scales + expert).repeat_interleave(GROUP, dim=1)
                        torch.testing.assert_close(per_element, expected[:, k_range])

    def test_split_rejects_layouts_it_cannot_remap(self):
        """Group splitting assumes full symmetric checkpoint scales; fail loudly otherwise.

        Presharded checkpoints are already narrowed, and asymmetric zero points are
        not split, so either would silently pair weights with the wrong scales.
        """
        for name, presharded, sym in (
            ("presharded", True, True),
            ("asymmetric", False, False),
        ):
            with self.subTest(name):
                layer = _fused_moe_shell(tp_size=4, tp_rank=1, experts=1)
                layer.use_presharded_weights = presharded
                with self.assertRaisesRegex(ValueError, "Splitting W2 scale groups"):
                    _triton_scheme(sym=sym).create_weights(
                        layer,
                        1,
                        HIDDEN,
                        INTERMEDIATE // 4,
                        torch.float32,
                        weight_loader=layer.weight_loader,
                    )


if __name__ == "__main__":
    unittest.main()

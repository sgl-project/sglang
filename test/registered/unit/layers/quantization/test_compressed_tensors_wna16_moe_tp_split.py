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


def _triton_scheme(*, actorder=None):
    # Bypass __init__, which needs a full CompressedTensorsConfig.
    scheme = CompressedTensorsWNA16TritonMoE.__new__(CompressedTensorsWNA16TritonMoE)
    scheme.group_size = GROUP
    scheme.sym = True
    scheme.actorder = actorder
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


def _create(scheme, layer, tp_size, experts):
    scheme.create_weights(
        layer,
        experts,
        HIDDEN,
        INTERMEDIATE // tp_size,
        torch.float32,
        weight_loader=layer.weight_loader,
    )


class TestWNA16MoETPGroupAlignment(CustomTestCase):
    def test_rejects_shards_that_split_a_scale_group(self):
        """640 / tp is not a multiple of 128 at tp 2, 4 and 8; loading must fail early.

        Without the check each rank narrows whole scale groups and pairs its K range
        with the wrong scales (or reads past them in the kernel).
        """
        for tp_size in (2, 4, 8):
            shard = INTERMEDIATE // tp_size
            with self.subTest(tp_size=tp_size):
                layer = _fused_moe_shell(tp_size=tp_size, tp_rank=0, experts=1)
                with self.assertRaisesRegex(
                    ValueError, rf"\({shard}\).*group_size \({GROUP}\)"
                ):
                    _create(_triton_scheme(), layer, tp_size, experts=1)

    def test_full_w2_scales_are_not_rejected(self):
        """Grouped actorder loads the full w2 scales on every rank, so no split happens."""
        layer = _fused_moe_shell(tp_size=4, tp_rank=1, experts=1)
        _create(_triton_scheme(actorder="group"), layer, 4, experts=1)
        self.assertEqual(layer.w2_weight_scale.shape[1], INTERMEDIATE // GROUP)

    def test_aligned_shards_load_their_own_scales(self):
        """At tp 1 and 5 every shard holds whole groups and gets its own scales."""
        experts = 2
        packed = torch.arange(HIDDEN * INTERMEDIATE // 8, dtype=torch.int32).view(
            HIDDEN, INTERMEDIATE // 8
        )
        scales = torch.arange(HIDDEN * INTERMEDIATE // GROUP, dtype=torch.float32)
        scales = scales.view(HIDDEN, INTERMEDIATE // GROUP) + 1
        for tp_size in (1, 5):
            shard = INTERMEDIATE // tp_size
            for rank in range(tp_size):
                with self.subTest(tp_size=tp_size, rank=rank):
                    layer = _fused_moe_shell(
                        tp_size=tp_size, tp_rank=rank, experts=experts
                    )
                    scheme = _triton_scheme()
                    layer.scheme = layer.quant_method = scheme
                    _create(scheme, layer, tp_size, experts)
                    for expert in range(experts):
                        for param, source, name in (
                            (
                                layer.w2_weight_packed,
                                packed + expert,
                                "w2_weight_packed",
                            ),
                            (layer.w2_weight_scale, scales + expert, "w2_weight_scale"),
                        ):
                            param.weight_loader(param, source, name, "w2", expert)
                    layer.w13_weight_packed.data.zero_()
                    scheme.process_weights_after_loading(layer)
                    info = scheme.get_triton_quant_info(layer)
                    groups = slice(rank * shard // GROUP, (rank + 1) * shard // GROUP)
                    for expert in range(experts):
                        torch.testing.assert_close(
                            info.w2_scale[expert], (scales + expert)[:, groups]
                        )


if __name__ == "__main__":
    unittest.main()

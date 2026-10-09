"""With AITER on, a block-FP8 MoE that the Triton runner serves must compute what
it computes with AITER off: AITER's expert layout is for AITER's kernels only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

MODULE = "sglang.srt.layers.quantization.fp8"
BLOCK = 128
EXPERTS, HIDDEN, INTERMEDIATE, TOKENS, TOP_K = 4, 256, 256, 64, 2


def _on_fnuz_gpu() -> bool:
    if not (is_hip() and torch.cuda.is_available()):
        return False
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    return is_fp8_fnuz()


def _aiter_shuffle():
    try:
        from aiter.ops.shuffle import shuffle_weight
    except ImportError:
        return None
    return shuffle_weight


def _finalized(runner_backend, *, use_aiter):
    """A block-FP8 MoE layer after the finalization the given runner would see."""
    from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    method = object.__new__(Fp8MoEMethod)
    method.quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True, weight_block_size=[BLOCK, BLOCK]
    )
    method.block_quant = True
    method.use_mxfp8 = False
    method.convert_mxfp8_to_block = False
    method.is_fp4_expert = False
    method.is_checkpoint_fp8_serialized = True
    method.runner = SimpleNamespace(runner_backend=runner_backend)
    layer = torch.nn.Module()
    generator = torch.Generator(device="cuda").manual_seed(0)
    for name, shape in (
        ("w13_weight", (EXPERTS, 2 * INTERMEDIATE, HIDDEN)),
        ("w2_weight", (EXPERTS, HIDDEN, INTERMEDIATE)),
    ):
        weight = (
            ((torch.rand(shape, generator=generator, device="cuda") - 0.5) * 2 * 448)
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        scale = (
            torch.rand(
                (EXPERTS, shape[1] // BLOCK, shape[2] // BLOCK),
                generator=generator,
                device="cuda",
            )
            * 1e-2
        )
        layer.register_parameter(name, torch.nn.Parameter(weight, requires_grad=False))
        layer.register_parameter(
            name + "_scale_inv", torch.nn.Parameter(scale, requires_grad=False)
        )
    with (
        patch(f"{MODULE}._use_aiter", use_aiter),
        patch(f"{MODULE}._use_hip_int4", False),
        patch(f"{MODULE}.shuffle_weight", _aiter_shuffle(), create=True),
    ):
        method.process_weights_after_loading_block_quant(layer)
    return layer


def _triton_moe(layer, hidden_states, topk_output):
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe

    return fused_moe(
        hidden_states.clone(),  # fused_moe writes its output into its input
        layer.w13_weight,
        layer.w2_weight,
        topk_output,
        use_fp8_w8a8=True,
        w1_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
        block_shape=[BLOCK, BLOCK],
    )


@unittest.skipUnless(
    _on_fnuz_gpu() and _aiter_shuffle() is not None,
    "requires a gfx94x (E4M3FNUZ) GPU with AITER",
)
class TestFp8MoeTritonRunnerWithAiter(unittest.TestCase):
    def test_the_triton_runner_computes_what_it_computes_without_aiter(self):
        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.layers.moe.topk import TopKConfig, select_experts
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        hidden_states = (
            torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda") / 10
        )
        router_logits = torch.randn(
            TOKENS, EXPERTS, dtype=torch.bfloat16, device="cuda"
        )
        topk_output = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=TOP_K, renormalize=False),
        )

        with torch.inference_mode():
            reference = _triton_moe(
                _finalized(MoeRunnerBackend.TRITON, use_aiter=False),
                hidden_states,
                topk_output,
            )
            served_layer = _finalized(MoeRunnerBackend.TRITON, use_aiter=True)
            served = _triton_moe(served_layer, hidden_states, topk_output)
            # What the Triton runner computed when it was handed AITER's layout.
            aiter_layout = _triton_moe(
                _finalized(MoeRunnerBackend.AITER, use_aiter=True),
                hidden_states,
                topk_output,
            )

        torch.testing.assert_close(served, reference, rtol=0, atol=0)
        self.assertFalse(getattr(served_layer.w13_weight, "is_shuffled", False))
        # The layout matters to the Triton kernel: fed AITER's, it is not even close.
        error = (aiter_layout.float() - reference.float()).abs().mean()
        self.assertGreater(error / reference.float().abs().mean(), 0.5)


if __name__ == "__main__":
    unittest.main()

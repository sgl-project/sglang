import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.moe.silu_and_mul_block_quant import (
    silu_and_mul_block_quant_two_group,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import init_single_process_dist
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_BLOCK = 128
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def _quantize_per_token_groups(x: torch.Tensor):
    """Torch reference for dynamic FP8 quantization with a partial K group."""
    logical_k = x.shape[-1]
    padded = F.pad(x, (0, (-logical_k) % _BLOCK)).float()
    groups = padded.reshape(-1, padded.shape[-1] // _BLOCK, _BLOCK)
    scales = groups.abs().amax(dim=-1).clamp(min=1e-10) / _FP8_MAX
    quantized = (
        (groups / scales.unsqueeze(-1))
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(_FP8_DTYPE)
        .reshape(*x.shape[:-1], padded.shape[-1])
    )
    return quantized[..., :logical_k].contiguous(), scales.reshape(*x.shape[:-1], -1)


def _quantize_weight_blocks(weight: torch.Tensor):
    """Quantize an (N, K) weight while retaining its logical shape."""
    n, k = weight.shape
    padded_n = (n + _BLOCK - 1) // _BLOCK * _BLOCK
    padded_k = (k + _BLOCK - 1) // _BLOCK * _BLOCK
    padded = F.pad(weight, (0, padded_k - k, 0, padded_n - n)).float()
    blocks = padded.reshape(
        padded_n // _BLOCK,
        _BLOCK,
        padded_k // _BLOCK,
        _BLOCK,
    )
    scales = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-10) / _FP8_MAX
    quantized = (
        (blocks / scales[:, None, :, None])
        .clamp(min=-_FP8_MAX, max=_FP8_MAX)
        .to(_FP8_DTYPE)
        .reshape(padded_n, padded_k)
    )
    return quantized[:n, :k].contiguous(), scales


def _block_scaled_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
):
    """Small independent reference for the Triton block-scaled GEMM."""
    m, k = activation.shape
    n = weight.shape[0]
    output = torch.zeros((m, n), device=activation.device, dtype=torch.float32)

    for k_block, k_start in enumerate(range(0, k, _BLOCK)):
        k_end = min(k_start + _BLOCK, k)
        activation_tile = activation[:, k_start:k_end].float()
        for n_block, n_start in enumerate(range(0, n, _BLOCK)):
            n_end = min(n_start + _BLOCK, n)
            weight_tile = weight[n_start:n_end, k_start:k_end].float()
            output[:, n_start:n_end] += (
                (activation_tile @ weight_tile.T)
                * activation_scale[:, k_block, None]
                * weight_scale[n_block, k_block]
            )

    return output


def _torch_block_fp8_moe_reference(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    topk_weights,
    topk_ids,
):
    assert topk_ids.shape[1] == 1
    output = torch.zeros_like(hidden_states)

    for expert in range(w1.shape[0]):
        token_mask = topk_ids[:, 0] == expert
        if not torch.any(token_mask):
            continue

        activation, activation_scale = _quantize_per_token_groups(
            hidden_states[token_mask]
        )
        gate_up = _block_scaled_mm(
            activation, w1[expert], activation_scale, w1_scale[expert]
        ).to(hidden_states.dtype)
        gate, up = gate_up.chunk(2, dim=-1)
        activation = (F.silu(gate) * up).to(hidden_states.dtype)

        # This is the BailingMoeV3 TP8 case under test: logical K=160 produces two
        # scale groups, with only 32 live values in the second group.
        assert activation.shape[-1] == 160
        activation, activation_scale = _quantize_per_token_groups(activation)
        down = _block_scaled_mm(
            activation, w2[expert], activation_scale, w2_scale[expert]
        )
        routed_weight = topk_weights[token_mask, 0].unsqueeze(-1)
        output[token_mask] = (down * routed_weight).to(hidden_states.dtype)

    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTritonMoEBlockFP8Tail(CustomTestCase):
    def test_native_fused_silu_block_quant_k160(self):
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            self.skipTest("FP8 requires compute capability 8.9 or newer")

        torch.manual_seed(0)
        rows, logical_k = 6, 160
        gate_up = (
            torch.randn(
                rows,
                2 * logical_k,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.5
        )
        expert_ids = torch.tensor([0, -1, 1], device="cuda", dtype=torch.int32)

        actual, actual_scale = silu_and_mul_block_quant_two_group(
            gate_up,
            _BLOCK,
            expert_ids=expert_ids,
            expert_step=2,
        )

        gate, up = gate_up.chunk(2, dim=-1)
        activation = (F.silu(gate) * up).to(gate_up.dtype)
        activation[2:4].zero_()
        expected, expected_scale = _quantize_per_token_groups(activation)

        self.assertEqual(actual.shape, (rows, logical_k))
        self.assertEqual(actual_scale.shape, (rows, 2))
        self.assertEqual(actual.dtype, _FP8_DTYPE)
        self.assertEqual(actual_scale.dtype, torch.float32)
        torch.testing.assert_close(
            actual_scale,
            expected_scale,
            rtol=0.02,
            atol=1e-7,
        )

        scale_columns = actual_scale.repeat_interleave(_BLOCK, dim=-1)[:, :logical_k]
        expected_scale_columns = expected_scale.repeat_interleave(_BLOCK, dim=-1)[
            :, :logical_k
        ]
        actual_dequant = actual.float() * scale_columns
        expected_dequant = expected.float() * expected_scale_columns
        relative_error = (actual_dequant - expected_dequant).abs().mean() / (
            expected_dequant.abs().mean().clamp(min=1e-6)
        )
        self.assertLess(relative_error.item(), 0.01)
        self.assertEqual(torch.count_nonzero(actual[2:4].float()).item(), 0)

    def test_native_fused_silu_block_quant_shared_limit_k160(self):
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            self.skipTest("FP8 requires compute capability 8.9 or newer")

        torch.manual_seed(1)
        rows, logical_k = 6, 160
        gate_up = torch.randn(
            rows,
            2 * logical_k,
            device="cuda",
            dtype=torch.bfloat16,
        )
        # Each expert id covers two TMA-layout rows: routed, shared, padding.
        expert_ids = torch.tensor([0, 2, -1], device="cuda", dtype=torch.int32)
        shared_limit = 0.25

        actual, actual_scale = silu_and_mul_block_quant_two_group(
            gate_up,
            _BLOCK,
            expert_ids=expert_ids,
            expert_step=2,
            shared_limit=shared_limit,
            num_routed_experts=2,
        )

        gate, up = gate_up.chunk(2, dim=-1)
        activation = (F.silu(gate) * up).to(gate_up.dtype)
        activation[2:4] = (
            F.silu(gate[2:4]).clamp(max=shared_limit)
            * up[2:4].clamp(min=-shared_limit, max=shared_limit)
        ).to(gate_up.dtype)
        activation[4:6].zero_()
        expected, expected_scale = _quantize_per_token_groups(activation)

        torch.testing.assert_close(
            actual_scale,
            expected_scale,
            rtol=0.02,
            atol=1e-7,
        )
        scale_columns = actual_scale.repeat_interleave(_BLOCK, dim=-1)[:, :logical_k]
        expected_scale_columns = expected_scale.repeat_interleave(_BLOCK, dim=-1)[
            :, :logical_k
        ]
        actual_dequant = actual.float() * scale_columns
        expected_dequant = expected.float() * expected_scale_columns
        relative_error = (actual_dequant - expected_dequant).abs().mean() / (
            expected_dequant.abs().mean().clamp(min=1e-6)
        )
        self.assertLess(relative_error.item(), 0.01)
        self.assertEqual(torch.count_nonzero(actual[4:6].float()).item(), 0)

    def test_native_fused_down_gemm_k160(self):
        capability = torch.cuda.get_device_capability()
        if capability < (8, 9):
            self.skipTest("FP8 requires compute capability 8.9 or newer")

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        init_single_process_dist(master_port=29642)
        torch.manual_seed(0)

        device = "cuda"
        dtype = torch.bfloat16
        num_tokens, num_experts = 3, 2
        hidden_size, local_intermediate_size = 4096, 160

        hidden_states = (
            torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
        )
        w1_bf16 = (
            torch.randn(
                num_experts,
                2 * local_intermediate_size,
                hidden_size,
                device=device,
                dtype=dtype,
            )
            * 0.1
        )
        w2_bf16 = (
            torch.randn(
                num_experts,
                hidden_size,
                local_intermediate_size,
                device=device,
                dtype=dtype,
            )
            * 0.1
        )

        w1_and_scales = [_quantize_weight_blocks(weight) for weight in w1_bf16]
        w2_and_scales = [_quantize_weight_blocks(weight) for weight in w2_bf16]
        w1 = torch.stack([weight for weight, _ in w1_and_scales])
        w1_scale = torch.stack([scale for _, scale in w1_and_scales])
        w2 = torch.stack([weight for weight, _ in w2_and_scales])
        w2_scale = torch.stack([scale for _, scale in w2_and_scales])
        self.assertEqual(w2.shape, (num_experts, hidden_size, 160))

        router_logits = torch.tensor(
            [[4.0, -4.0], [-4.0, 4.0], [4.0, -4.0]],
            device=device,
            dtype=dtype,
        )
        topk_output = select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            topk_config=TopKConfig(top_k=1, renormalize=False),
        )
        topk_weights, topk_ids, _ = topk_output

        with torch.inference_mode():
            expected = _torch_block_fp8_moe_reference(
                hidden_states,
                w1,
                w2,
                w1_scale,
                w2_scale,
                topk_weights,
                topk_ids,
            )
            actual = fused_moe(
                hidden_states,
                w1,
                w2,
                topk_output,
                moe_runner_config=MoeRunnerConfig(inplace=False),
                use_fp8_w8a8=True,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                block_shape=[_BLOCK, _BLOCK],
            )

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        relative_error = (actual.float() - expected.float()).abs().mean() / (
            expected.float().abs().mean().clamp(min=1e-6)
        )
        self.assertLess(relative_error.item(), 0.03)


if __name__ == "__main__":
    unittest.main()

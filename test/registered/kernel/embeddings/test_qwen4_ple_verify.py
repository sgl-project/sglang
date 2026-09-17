import itertools
import math
import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.qwen4_ple import (
    can_fuse_qwen4_gate_reduce,
    can_fuse_qwen4_verify_conv,
    fused_qwen4_gate_reduce,
    fused_qwen4_verify_conv,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


def gate_reference(key, query, value):
    gate = (key * query).sum(-1, keepdim=True) / math.sqrt(key.shape[-1])
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    return torch.sigmoid(gate) * value.unsqueeze(-2)


def conv_reference(x, residual, weight, state, indices, valid, width, dilation):
    channels = x.shape[1]
    state_len = state.shape[2]
    padded = x.reshape(-1, width, channels)
    conv_input = torch.cat([state[indices].to(x.dtype), padded.transpose(1, 2)], -1)
    conv = F.conv1d(conv_input, weight, dilation=dilation, groups=channels)
    output = F.silu(conv.transpose(1, 2).reshape_as(x)) + residual
    output = torch.where(valid[:, None], output, torch.zeros_like(output))
    checkpoint = (
        conv_input.unfold(2, state_len, 1)[:, :, 1 : width + 1].permute(0, 2, 1, 3)
        if state_len
        else x.new_empty((indices.numel(), width, channels, 0))
    )
    checkpoint = torch.where(
        valid.reshape(-1, width, 1, 1), checkpoint, torch.zeros_like(checkpoint)
    )
    return output, checkpoint


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestQwen4PLEVerify(CustomTestCase):
    def test_gate_reduction_rounding_and_zero(self):
        torch.manual_seed(42)
        for tokens in (0, 1, 4, 17, 192):
            with self.subTest(tokens=tokens):
                key = torch.randn(tokens, 4, 2560, device="cuda", dtype=torch.bfloat16)
                query = torch.randn_like(key)
                if tokens:
                    query[0, 0].zero_()
                    query[0, 1] = -key[0, 1]
                value = torch.randn(tokens, 2560, device="cuda", dtype=torch.bfloat16)
                expected = gate_reference(key, query, value)
                actual = fused_qwen4_gate_reduce(key, query, value)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_verify_output_and_checkpoints(self):
        torch.manual_seed(123)
        for width, dilation, state_dtype in itertools.product(
            (1, 4, 7), (1, 3), (torch.bfloat16, torch.float32)
        ):
            with self.subTest(width=width, dilation=dilation, state_dtype=state_dtype):
                requests, channels, state_len = 3, 264, 2 * dilation
                x = torch.randn(
                    requests * width, channels, device="cuda", dtype=torch.bfloat16
                )
                residual = torch.randn_like(x)
                weight = torch.randn(channels, 1, 3, device="cuda", dtype=x.dtype)
                state = torch.randn(
                    9, channels, state_len * 2, device="cuda", dtype=state_dtype
                )[:, :, ::2]
                original_state = state.clone()
                indices = torch.tensor([5, 8, 0, 8, 0, 8], device="cuda")[::2]
                lengths = torch.tensor([width, max(0, width - 1), 0], device="cuda")
                valid = (
                    torch.arange(width, device="cuda")[None, :] < lengths[:, None]
                ).flatten()
                cache = torch.full(
                    (requests + 1, width + 1, channels, state_len * 2),
                    -17,
                    device="cuda",
                    dtype=state_dtype,
                )[:, :, :, ::2]
                expected, checkpoint = conv_reference(
                    x, residual, weight, state, indices, valid, width, dilation
                )
                actual = fused_qwen4_verify_conv(
                    x, residual, weight, state, indices, valid, width, dilation, cache
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    cache[:requests, :width], checkpoint.to(state_dtype), rtol=0, atol=0
                )
                torch.testing.assert_close(state, original_state, rtol=0, atol=0)
                self.assertTrue(torch.all(cache[requests:] == -17))
                self.assertTrue(torch.all(cache[:, width:] == -17))

    def test_graph_replay_updates_validity_and_state_selection(self):
        width, channels = 4, 10240
        x = torch.randn(8, channels, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        weight = torch.randn(channels, 1, 4, device="cuda", dtype=x.dtype)
        state = torch.randn(5, channels, 9, device="cuda", dtype=x.dtype)
        indices = torch.tensor([1, 3], device="cuda")
        valid = torch.ones(8, device="cuda", dtype=torch.bool)
        cache = torch.empty(2, width, channels, 9, device="cuda", dtype=x.dtype)
        args = (x, residual, weight, state, indices, valid, width, 3, cache)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused_qwen4_verify_conv(*args)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = fused_qwen4_verify_conv(*args)
        indices.copy_(torch.tensor([4, 0], device="cuda"))
        valid[2:4] = False
        x.add_(1)
        graph.replay()
        expected, checkpoint = conv_reference(*args[:-1])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache, checkpoint, rtol=0, atol=0)

    def test_missing_checkpoint_and_zero_state(self):
        for tokens in (0, 4):
            x = torch.randn(tokens, 16, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(16, 1, 1, device="cuda", dtype=x.dtype)
            state = torch.empty(2, 16, 0, device="cuda", dtype=x.dtype)
            indices = torch.zeros(tokens // 4, device="cuda", dtype=torch.int32)
            valid = torch.ones(tokens, device="cuda", dtype=torch.bool)
            out = fused_qwen4_verify_conv(
                x, x, weight, state, indices, valid, 4, 3, None
            )
            if tokens:
                ref, _ = conv_reference(x, x, weight, state, indices, valid, 4, 3)
                torch.testing.assert_close(out, ref, rtol=0, atol=0)
            else:
                self.assertEqual(out.shape, x.shape)

    def test_unsupported_layout_falls_back(self):
        key = torch.empty(4, 4, 2560, device="cuda", dtype=torch.bfloat16)
        value = torch.empty(4, 2560, device="cuda", dtype=key.dtype)
        self.assertFalse(can_fuse_qwen4_gate_reduce(key, key, value.t()))
        x = key.flatten(1)
        weight = torch.empty(10240, 1, 3, device="cuda", dtype=x.dtype)
        state = torch.empty(2, 10240, 6, device="cuda", dtype=x.dtype)
        indices = torch.zeros(1, device="cuda", dtype=torch.long)
        valid = torch.ones(4, device="cuda", dtype=torch.bool)
        self.assertFalse(
            can_fuse_qwen4_verify_conv(x, x, weight, state, indices, valid, 3, 3, None)
        )


if __name__ == "__main__":
    unittest.main()

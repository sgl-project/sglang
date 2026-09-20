import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.diffusion import can_use_joint_qkv_cat, joint_qkv_cat
from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.models.dits import joy_image
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def make_inputs(batch, image_tokens, text_tokens, heads, dim, dtype):
    inputs = []
    for tokens in (image_tokens, text_tokens):
        packed = torch.randn(batch, tokens, 3, heads, dim, device="cuda", dtype=dtype)
        q, k, v = packed.unbind(2)
        inputs.extend((q.contiguous(), k.contiguous(), v))
    return tuple(inputs)


def reference(inputs):
    return tuple(torch.cat((inputs[i], inputs[i + 3]), dim=1) for i in range(3))


class TestJointQKVCat(CustomTestCase):
    def assert_bits_equal(self, actual, expected):
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
            self.assertTrue(a.is_contiguous())

    def test_packed_v_and_batch_strides(self):
        cases = [(1, 8048, 1004, 32, 128), (2, 257, 13, 8, 128), (1, 1, 1, 3, 17)]
        for dtype in (torch.bfloat16, torch.float16):
            for shape in cases:
                with self.subTest(dtype=dtype, shape=shape):
                    inputs = make_inputs(*shape, dtype)
                    # Include signed zero, subnormals, infinities and NaN payloads.
                    bits = torch.tensor(
                        [0, -32768, 1, -32767, 32767, -1, 32640, -128],
                        device="cuda",
                        dtype=torch.int16,
                    )
                    for value in inputs:
                        value.view(torch.int16)[0, 0, 0, :8].copy_(bits)
                    before = tuple(x.clone() for x in inputs)
                    self.assertTrue(can_use_joint_qkv_cat(*inputs))
                    out = joint_qkv_cat(*inputs)
                    self.assert_bits_equal(out, reference(inputs))
                    self.assert_bits_equal(
                        tuple(x.contiguous() for x in inputs), before
                    )
                    # Q/K/V share an allocation but do not overlap.
                    saved = tuple(x.clone() for x in out[1:])
                    out[0].zero_()
                    self.assert_bits_equal(out[1:], saved)

    def test_changed_input_graph_replay(self):
        inputs = make_inputs(2, 257, 13, 8, 128, torch.bfloat16)
        gate = BitExactFusionGate("test", per_signature=True)
        with patch.object(joy_image, "_JOY_QKV_CAT", gate):
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), reference(inputs))
            self.assertTrue(gate.verified)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    joy_image._joy_joint_qkv(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out = joy_image._joy_joint_qkv(*inputs)
            for seed in range(8):
                torch.manual_seed(seed)
                for value in inputs:
                    value.copy_(torch.randn_like(value))
                graph.replay()
                self.assert_bits_equal(out, reference(inputs))

    def test_unsupported_inputs_and_autograd_fallback(self):
        inputs = make_inputs(1, 13, 5, 4, 32, torch.bfloat16)
        self.assertFalse(can_use_joint_qkv_cat(*inputs[:5]))
        self.assertFalse(can_use_joint_qkv_cat(*(x.float() for x in inputs)))
        self.assertFalse(can_use_joint_qkv_cat(*(x[..., ::2] for x in inputs)))
        self.assertFalse(
            can_use_joint_qkv_cat(inputs[0], inputs[1][:, :2], *inputs[2:])
        )
        with patch("torch.compiler.is_compiling", return_value=True):
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), reference(inputs))
        cpu = tuple(x.cpu().requires_grad_() for x in inputs)
        self.assertFalse(can_use_joint_qkv_cat(*cpu))
        out = joy_image._joy_joint_qkv(*cpu)
        sum(x.float().sum() for x in out).backward()
        for value in cpu:
            self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))

    def test_gate_mismatch_exception_and_unverified_capture(self):
        inputs = make_inputs(1, 13, 5, 4, 32, torch.bfloat16)
        expected = reference(inputs)
        wrong = tuple(x.clone() for x in expected)
        wrong[0].view(torch.int16)[0, 0, 0, 0] ^= 1
        gate = BitExactFusionGate("test", per_signature=True)
        with (
            patch.object(joy_image, "_JOY_QKV_CAT", gate),
            patch.object(joy_image, "joint_qkv_cat", return_value=wrong) as fused,
        ):
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), expected)
            self.assertTrue(gate.disabled)
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), expected)
            fused.assert_called_once()
        gate = BitExactFusionGate("test", per_signature=True)
        with (
            patch.object(joy_image, "_JOY_QKV_CAT", gate),
            patch.object(
                joy_image, "joint_qkv_cat", side_effect=RuntimeError("unavailable")
            ),
        ):
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), expected)
            self.assertTrue(gate.disabled)
        gate = BitExactFusionGate("test", per_signature=True)
        with (
            patch.object(joy_image, "_JOY_QKV_CAT", gate),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch.object(joy_image, "joint_qkv_cat") as fused,
        ):
            self.assert_bits_equal(joy_image._joy_joint_qkv(*inputs), expected)
            fused.assert_not_called()
            self.assertFalse(gate.verified)


if __name__ == "__main__":
    unittest.main()


from sglang.test.test_utils import CustomTestCase
from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualNormScaleShift
import random
import unittest
import torch
from torch import Tensor


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def allclose_with_tolerance(x: Tensor, y: Tensor, atol: float, rtol: float, max_ratio=0.005) -> Tuple[bool, str]:
    diff = torch.abs(x - y)
    th = atol + rtol * torch.abs(y)

    # out-of-tolerance mask
    bad_mask = diff > th
    bad_ratio = bad_mask.float().mean().item()

    return bad_ratio <= max_ratio, f"{bad_ratio:.6f} > {max_ratio}"

################################################################################
# Accuracy Test
################################################################################


class TestScaleResidualNormScaleShiftAccuracy(CustomTestCase):
    DTYPES = [torch.float32, torch.float16, torch.bfloat16]
    PARAM_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
    BATCH_NUM = [1, 2]
    SEQ = [83, 1024, 2047, 32760]
    FRAME = [4, 8, 16]
    HIDDEN_SIZES = [83, 1024, 1338, 1536, 3072, 4096]
    USE_AFFINE = [False, True]
    USE_BIAS = [False, True]
    NORM_TYPE = ["rms", "layer"]
    GATE_SHAPE = ["1", "1D", "B1D", "BF1D"]
    SCALE_SHIFT_SHAPE = ["1.2", "[1]", "BD",
                         "1D", "BSD", "B1D", "1SD", "11D", "BF1D"]
    SEEDS = [0]

    args = [
        (DTYPES, "dtype"),
        (PARAM_DTYPES, "param_type"),
        (BATCH_NUM, "batch"),
        (SEQ, "seq"),
        (FRAME, "frame"),
        (HIDDEN_SIZES, "hidden_size"),
        (USE_AFFINE, "use_affine"),
        (USE_BIAS, "use_bias"),
        (NORM_TYPE, "norm_type"),
        (GATE_SHAPE, "gate_shape"),
        (SCALE_SHIFT_SHAPE, "scale_shift_shape"),
        (SEEDS, "seed"),
    ]

    def gen_cases(self, num):
        num_of_args = len(self.args)

        while num > 0:
            yield {
                self.args[i][1]: random.choice(self.args[i][0])
                for i in range(num_of_args)
            }
            num -= 1

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_fused_test(
        self, **params,
    ):
        dtype = params["dtype"]
        param_type = params["param_type"]
        batch = params["batch"]
        seq = params["seq"]
        frame = params["frame"]
        hidden_size = params["hidden_size"]
        use_affine = params["use_affine"]
        use_bias = params["use_bias"]
        norm_type = params["norm_type"]
        gate_shape = params["gate_shape"]
        scale_shift_shape = params["scale_shift_shape"]
        seed = params["seed"]

        torch.manual_seed(seed)
        layer = ScaleResidualNormScaleShift(
            hidden_size, elementwise_affine=use_affine, bias=use_bias, dtype=param_type, norm_type=norm_type
        )
        if use_affine:
            w = torch.empty_like(layer.norm.weight)
            w.normal_(mean=1.0, std=0.1)
            layer.norm.weight.data.copy_(w)
            if norm_type == "layer" and use_bias:
                b = torch.empty_like(layer.norm.bias)
                b.normal_(mean=1.0, std=0.1)
                layer.norm.bias.data.copy_(b)

        residual = torch.randn(batch, seq, hidden_size, dtype=dtype)
        x = torch.randn(batch, seq, hidden_size, dtype=dtype)

        if gate_shape == "1":
            gate = 1
        elif gate_shape == "1D":
            gate = torch.randn(1, hidden_size, dtype=dtype)
        elif gate_shape == "BF1D":
            if seq % frame != 0:
                return
            gate = torch.randn(batch, frame, 1, hidden_size, dtype=dtype)
        elif gate_shape == "B1D":
            gate = torch.randn(batch, 1, hidden_size, dtype=dtype)
        else:
            raise ValueError("Unknown gate shape.")

        if is_float(scale_shift_shape):
            scale = torch.tensor(1.0, dtype=dtype) * float(scale_shift_shape)
            shift = torch.tensor(1.0, dtype=dtype) * float(scale_shift_shape)
        elif scale_shift_shape == "[1]":
            scale = torch.ones(1, dtype=dtype) * torch.rand(1).item()
            shift = torch.ones(1, dtype=dtype) * torch.rand(1).item()
        elif scale_shift_shape == "BD":
            scale = torch.randn(batch, hidden_size, dtype=dtype)
            shift = torch.randn(batch, hidden_size, dtype=dtype)
        elif scale_shift_shape == "1D":
            scale = torch.randn(1, hidden_size, dtype=dtype)
            shift = torch.randn(1, hidden_size, dtype=dtype)
        elif scale_shift_shape == "BSD":
            scale = torch.randn(batch, seq, hidden_size, dtype=dtype)
            shift = torch.randn(batch, seq, hidden_size, dtype=dtype)
        elif scale_shift_shape == "B1D":
            scale = torch.randn(batch, 1, hidden_size, dtype=dtype)
            shift = torch.randn(batch, 1, hidden_size, dtype=dtype)
        elif scale_shift_shape == "1SD":
            scale = torch.randn(1, seq, hidden_size, dtype=dtype)
            shift = torch.randn(1, seq, hidden_size, dtype=dtype)
        elif scale_shift_shape == "11D":
            scale = torch.randn(1, 1, hidden_size, dtype=dtype)
            shift = torch.randn(1, 1, hidden_size, dtype=dtype)
        elif scale_shift_shape == "BF1D":
            if seq % frame != 0:
                return
            scale = torch.randn(batch, frame, 1, hidden_size, dtype=dtype)
            shift = torch.randn(batch, frame, 1, hidden_size, dtype=dtype)

        with torch.inference_mode():
            ref_out_mod, ref_out_resi = layer.forward_native(
                residual, x, gate, shift, scale)
            out_mod, out_resi = layer(residual, x, gate, shift, scale)

        if dtype == torch.float32 and param_type == torch.float32:
            self.assertTrue(*allclose_with_tolerance(
                out_mod, ref_out_mod, atol=1e-6, rtol=1e-4))
            self.assertTrue(*allclose_with_tolerance(
                out_resi, ref_out_resi, atol=1e-6, rtol=1e-4))
        else:
            self.assertTrue(*allclose_with_tolerance(
                out_mod, ref_out_mod, atol=5e-2, rtol=1e-2))
            self.assertTrue(*allclose_with_tolerance(
                out_resi, ref_out_resi, atol=5e-2, rtol=1e-2))

    def test_fused(self):
        for params in self.gen_cases(num=300):
            with self.subTest(**params):
                self._run_fused_test(**params)
                torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)

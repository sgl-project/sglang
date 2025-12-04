
from sglang.multimodal_gen.runtime.layers.triton_ops import fuse_scale_shift_kernel
from sglang.multimodal_gen.test.utils import allclose_with_tolerance
from sglang.test.test_utils import CustomTestCase
from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualNormScaleShift
from sglang.multimodal_gen.runtime.layers.layernorm import ScaleResidualLayerNormScaleShift, LayerNorm
import torch.nn as nn
import random
import unittest
import itertools

import torch
import time
import os
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch sees", torch.cuda.device_count(), "GPUs")
print("Using device:", torch.cuda.get_device_name(0))


################################################################################
# Benchmark
################################################################################
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def benchmark(fn, *args, **kwargs):
    warmup = 2
    iters = 20
    # Make sure everything is clean
    with torch.inference_mode():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Warmup ---
        for _ in range(warmup):
            fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # --- Timed runs ---
        t0 = time.time()
        for _ in range(iters):
            fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()

    # Average time per iteration (ms)
    avg_time_ms = (t1 - t0) * 1000 / iters
    return avg_time_ms


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


################################################################################
# Performance Test
################################################################################
class TestScaleResidualNormScaleShiftPerf(CustomTestCase):
    DTYPES = [torch.float32, torch.float16]
    PARAM_DTYPES = [torch.float32]
    BATCH_NUM = [1, 2]
    SEQ = [1024, 2048, 4096, 16380, 32760, 115200]
    FRAME = [4]
    HIDDEN_SIZES = [512, 1024, 1536, 2048, 3072]
    USE_AFFINE = [True, False]
    USE_BIAS = [True]
    NORM_TYPE = ["layer"]
    GATE_SHAPE = ["B1D"]
    SCALE_SHIFT_SHAPE = ["11D"]
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

    def gen_cases(self):
        keys = [arg[1] for arg in self.args]
        value_lists = [arg[0] for arg in self.args]

        for values in itertools.product(*value_lists):
            yield dict(zip(keys, values))

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device(f"cuda")

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

        if norm_type == "layer" and use_bias == False:
            return

        ref_layer = ScaleResidualLayerNormScaleShift(
            hidden_size, elementwise_affine=use_affine, dtype=param_type, norm_type=norm_type
        )
        layer = ScaleResidualNormScaleShift(
            hidden_size, elementwise_affine=use_affine, bias=use_bias, dtype=param_type, norm_type=norm_type
        )
        if use_affine:
            w = torch.empty_like(layer.norm.weight)
            w.normal_(mean=1.0, std=0.1)
            layer.norm.weight.data.copy_(w)
            ref_layer.norm.weight.data.copy_(w)
            if norm_type == "layer" and use_bias:
                b = torch.empty_like(layer.norm.bias)
                b.normal_(mean=1.0, std=0.1)
                layer.norm.bias.data.copy_(b)
                ref_layer.norm.bias.data.copy_(b)

        residual = torch.randn(batch, seq, hidden_size, dtype=dtype)
        x = torch.randn(batch, seq, hidden_size, dtype=dtype)

        if gate_shape == "1":
            gate = 1
        elif gate_shape == "BF1D":
            if seq % frame != 0:
                return
            gate = torch.randn(batch, frame, 1, hidden_size, dtype=dtype)
        elif gate_shape == "B1D":
            gate = torch.randn(batch, 1, hidden_size, dtype=dtype)
        else:
            raise ValueError("Unknown gate shape.")

        if is_float(scale_shift_shape):
            scale = torch.ones(1, dtype=dtype) * float(scale_shift_shape)
            shift = torch.ones(1, dtype=dtype) * float(scale_shift_shape)
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
            ref_out_mod, ref_out_resi = ref_layer(
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

        # Perf
        fused_kernel_time = benchmark(layer, residual, x, gate, shift, scale)
        naive_kernel_time = benchmark(
            ref_layer, residual, x, gate, shift, scale)
        speedup = naive_kernel_time / fused_kernel_time
        print(
            f"[speedup]={speedup:.2f}x ({naive_kernel_time:.3f}ms/{fused_kernel_time:.3f}ms), "
            f"dtype={dtype}, param_type={param_type}, batch={batch}, seq={seq}, "
            f"frame={frame}, hidden={hidden_size}, use_affine={use_affine}, "
            f"use_bias={use_bias}, norm_type={norm_type}, gate_shape={gate_shape}, "
            f"scale_shift_shape={scale_shift_shape}, seed={seed}"
        )
        return speedup

    def test_fused(self):
        speedup = []
        for params in self.gen_cases():
            with self.subTest(**params):
                torch.cuda.synchronize()
                speedup.append(self._run_fused_test(**params))
                torch.cuda.synchronize()
        avg_speedup = sum(speedup) / len(speedup)
        print(f"Average Speedup = {avg_speedup}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

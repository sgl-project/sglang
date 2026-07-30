"""Correctness and sharing tests for the PyTorch/MLX tensor bridge."""

import gc
import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch
from packaging.version import Version

from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_SUPPORTED_RUNTIME = (
    importlib.util.find_spec("mlx") is not None
    and torch.backends.mps.is_available()
    and Version(torch.__version__) >= Version("2.13.0")
)


class TestTensorBridgeImport(unittest.TestCase):
    def test_import_does_not_eagerly_import_mlx(self):
        script = """
import sys
from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx
assert mlx_to_torch is not None and torch_to_mlx is not None
assert not any(name == "mlx" or name.startswith("mlx.") for name in sys.modules)
"""
        env = os.environ.copy()
        env.pop("SGLANG_USE_MLX", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


@unittest.skipUnless(_HAS_SUPPORTED_RUNTIME, "requires MLX >= 0.32 and Torch MPS")
class TestTensorBridgeMetalSharing(unittest.TestCase):
    def test_common_inference_dtypes_round_trip_losslessly(self):
        import mlx.core as mx

        cases = [
            (torch.float32, mx.float32, [0.0, 1.0, -2.0]),
            (torch.float16, mx.float16, [0.0, 1.0, -2.0]),
            (torch.bfloat16, mx.bfloat16, [0.0, 1.0, -2.0]),
            (torch.int32, mx.int32, [0, 1, -2]),
            (torch.bool, mx.bool_, [False, True, False]),
        ]
        for torch_dtype, mlx_dtype, values in cases:
            with self.subTest(dtype=torch_dtype):
                source = torch.tensor(values, device="mps", dtype=torch_dtype)
                array = torch_to_mlx(source)
                round_tripped = mlx_to_torch(array)

                self.assertEqual(array.dtype, mlx_dtype)
                self.assertEqual(round_tripped.dtype, torch_dtype)
                self.assertEqual(round_tripped.device.type, "mps")
                self.assertTrue(torch.equal(round_tripped.cpu(), source.cpu()))

    def test_torch_mps_to_mlx_shares_strided_bfloat16_storage(self):
        import mlx.core as mx

        tensor = torch.arange(24, device="mps", dtype=torch.float32)
        tensor = tensor.to(torch.bfloat16).reshape(4, 6).T
        array = torch_to_mlx(tensor)

        tensor.zero_()
        torch.mps.synchronize()
        self.assertTrue(mx.all(array == 0).item())

        del tensor
        gc.collect()
        self.assertTrue(mx.all(array == 0).item())

    def test_mlx_to_torch_mps_shares_storage_and_lifetime(self):
        import mlx.core as mx

        array = mx.arange(16, dtype=mx.float32).reshape(4, 4)[:, 1:3]
        tensor = mlx_to_torch(array)

        self.assertEqual(tensor.device.type, "mps")
        tensor.zero_()
        torch.mps.synchronize()
        self.assertTrue(mx.all(array == 0).item())

        del array
        gc.collect()
        self.assertEqual(torch.count_nonzero(tensor).item(), 0)

    def test_mps_round_trip_reuses_the_same_allocation(self):
        tensor = torch.arange(16, device="mps", dtype=torch.float32)

        round_tripped = mlx_to_torch(torch_to_mlx(tensor))

        self.assertEqual(round_tripped.data_ptr(), tensor.data_ptr())

    def test_bridge_detaches_autograd_and_synchronizes_producers(self):
        import mlx.core as mx

        tensor = torch.arange(8, device="mps", dtype=torch.float32)
        tensor.requires_grad_()
        with mock.patch.object(
            torch.mps, "synchronize", wraps=torch.mps.synchronize
        ) as synchronize:
            array = torch_to_mlx(tensor)
        synchronize.assert_called_once_with()

        with mock.patch.object(mx, "eval", wraps=mx.eval) as evaluate:
            round_tripped = mlx_to_torch(array)
        evaluate.assert_called_once_with(array)
        self.assertFalse(round_tripped.requires_grad)

    def test_torch_cpu_input_is_an_explicit_copy(self):
        import mlx.core as mx

        tensor = torch.arange(8, dtype=torch.bfloat16)
        array = torch_to_mlx(tensor)
        tensor.zero_()
        mx.eval(array)

        self.assertEqual(
            array.astype(mx.float32).tolist(),
            [float(value) for value in range(8)],
        )

    def test_explicit_cpu_target_shares_storage(self):
        import mlx.core as mx

        array = mx.arange(16, dtype=mx.float32).reshape(4, 4)[:, ::2]
        tensor = mlx_to_torch(array, device="cpu")

        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tensor.stride(), (4, 2))
        tensor.zero_()
        self.assertTrue(mx.all(array == 0).item())

        del array
        gc.collect()
        self.assertEqual(torch.count_nonzero(tensor).item(), 0)


if __name__ == "__main__":
    unittest.main()

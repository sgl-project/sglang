"""Correctness and sharing tests for the PyTorch/MLX tensor bridge."""

import gc
import os
import subprocess
import sys
import unittest
from importlib.metadata import PackageNotFoundError, version
from unittest import mock

import torch
from packaging.version import Version

from sglang.srt.utils.tensor_bridge import (
    MlxTensorView,
    borrow_torch_tensors,
    mlx_call,
    mlx_call_multi,
    mlx_to_torch,
    torch_to_mlx,
)
from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")


def _has_stable_version_at_least(distribution: str, minimum: Version) -> bool:
    try:
        installed = Version(version(distribution))
    except (PackageNotFoundError, ValueError):
        return False
    return not installed.is_prerelease and installed >= minimum


_HAS_MLX = _has_stable_version_at_least("mlx", Version("0.32.0"))
_HAS_SUPPORTED_RUNTIME = (
    _HAS_MLX
    and torch.backends.mps.is_available()
    and not Version(torch.__version__).is_prerelease
    and Version(torch.__version__).release[:2] == (2, 13)
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


@unittest.skipUnless(_HAS_MLX, "requires MLX >= 0.32")
class TestTensorBridgeCpu(unittest.TestCase):
    def test_mlx_call_multi_preserves_cpu_float64(self):
        import mlx.core as mx

        source = torch.tensor([1.25, -2.5, 4.0, 8.0], dtype=torch.float64)
        with mock.patch.object(mx, "eval", wraps=mx.eval) as evaluate:
            first, second = mlx_call_multi(
                lambda x: (x + 1, x * 2),
                source,
                device="cpu",
            )

        evaluate.assert_called_once()
        self.assertEqual(first.dtype, torch.float64)
        self.assertEqual(second.dtype, torch.float64)
        torch.testing.assert_close(first, source + 1)
        torch.testing.assert_close(second, source * 2)

    def test_mlx_call_multi_materializes_cpu_negative_strides_safely(self):
        import mlx.core as mx

        source = torch.arange(8, dtype=torch.float32)
        with mock.patch.object(mx, "eval", wraps=mx.eval) as evaluate:
            first, reversed_ = mlx_call_multi(
                lambda x: (x + 1, x[::-1]), source, device="cpu"
            )

        # The ordinary graph is evaluated once; all negative-stride results
        # share one additional materialization boundary before DLPack export.
        self.assertEqual(evaluate.call_count, 2)
        torch.testing.assert_close(first, source + 1)
        torch.testing.assert_close(reversed_, source.flip(0))

    def test_mlx_call_multi_rejects_invalid_target_before_work(self):
        operation = mock.Mock()
        with self.assertRaisesRegex(ValueError, "CPU and MPS targets"):
            mlx_call_multi(
                operation,
                torch.ones(1),
                device="cuda",
            )
        operation.assert_not_called()

    def test_mlx_call_rejects_invalid_target_before_work(self):
        operation = mock.Mock()
        with self.assertRaisesRegex(ValueError, "CPU and MPS targets"):
            mlx_call(
                operation,
                torch.ones(1),
                device="cuda",
            )
        operation.assert_not_called()

    def test_mlx_call_multi_rejects_non_mlx_outputs(self):
        with self.assertRaisesRegex(TypeError, "outputs must be MLX arrays"):
            mlx_call_multi(
                lambda _x: (torch.ones(1),),
                torch.ones(1),
                device="cpu",
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

    def test_torch_mps_to_mlx_is_an_explicit_copy(self):
        import mlx.core as mx

        tensor = torch.arange(24, device="mps", dtype=torch.float32)
        tensor = tensor.to(torch.bfloat16).reshape(4, 6).T
        expected = tensor.cpu().clone()
        array = torch_to_mlx(tensor)

        tensor.zero_()
        torch.mps.synchronize()
        mx.eval(array)
        round_tripped = mlx_to_torch(array, device="cpu")
        self.assertTrue(torch.equal(round_tripped, expected))

        del tensor
        gc.collect()
        self.assertTrue(torch.equal(round_tripped, expected))

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

    def test_mps_round_trip_uses_independent_input_storage(self):
        tensor = torch.arange(16, device="mps", dtype=torch.float32)

        round_tripped = mlx_to_torch(torch_to_mlx(tensor))

        self.assertNotEqual(round_tripped.data_ptr(), tensor.data_ptr())

    def test_mlx_call_keeps_zero_copy_borrows_alive(self):
        import mlx.core as mx

        tensor = torch.randn(2, 8, device="mps", dtype=torch.float32)
        weight = torch.randn(8, device="mps", dtype=torch.float32)
        before = tensor.cpu().clone()
        weight_before = weight.cpu().clone()
        reference = torch.nn.functional.rms_norm(before, (8,), weight_before, 1e-6)
        result = mlx_call(lambda x, w: mx.fast.rms_norm(x, w, 1e-6), tensor, weight)

        torch.mps.synchronize()
        self.assertTrue(torch.equal(tensor.cpu(), before))
        self.assertTrue(torch.equal(weight.cpu(), weight_before))
        self.assertNotEqual(result.data_ptr(), tensor.data_ptr())

        del tensor, weight
        gc.collect()
        torch.testing.assert_close(result.cpu(), reference)

    def test_persistent_view_keeps_torch_storage_alive(self):
        import mlx.core as mx

        source = torch.arange(16, device="mps", dtype=torch.float32).reshape(4, 4)
        view = MlxTensorView(source)
        self.assertTrue(view.matches(source))
        del source
        gc.collect()

        result = mlx_call(lambda x: x + 1, view, device="mps")
        torch.testing.assert_close(
            result.cpu(), torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
        )
        # The view is still the owner after the result has been exported.
        self.assertEqual(view.array.shape, (4, 4))
        mx.eval(view.array)

    def test_mlx_call_synchronizes_persistent_view_producers(self):
        source = torch.zeros(8, device="mps", dtype=torch.float32)
        view = MlxTensorView(source)
        source.fill_(3)
        with mock.patch.object(
            torch.mps, "synchronize", wraps=torch.mps.synchronize
        ) as synchronize:
            result = mlx_call(lambda x: x + 1, view, device="mps")
        synchronize.assert_called_once_with()
        torch.testing.assert_close(result.cpu(), torch.full((8,), 4.0))

    def test_batch_borrow_syncs_once_and_preserves_sources(self):
        first = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)
        second = torch.randn(8, 8, device="mps", dtype=torch.bfloat16)
        first_before = first.cpu().clone()
        second_before = second.cpu().clone()
        with mock.patch.object(
            torch.mps, "synchronize", wraps=torch.mps.synchronize
        ) as synchronize:
            views = borrow_torch_tensors(first, second)
        synchronize.assert_called_once_with()
        self.assertTrue(torch.equal(first.cpu(), first_before))
        self.assertTrue(torch.equal(second.cpu(), second_before))
        self.assertTrue(views[0].matches(first))
        self.assertTrue(views[1].matches(second))

    def test_invalid_batch_borrow_does_not_synchronize(self):
        mps_tensor = torch.ones(1, device="mps")
        cpu_tensor = torch.ones(1)
        with mock.patch.object(torch.mps, "synchronize") as synchronize:
            with self.assertRaisesRegex(ValueError, "requires MPS tensors"):
                borrow_torch_tensors(mps_tensor, cpu_tensor)
        synchronize.assert_not_called()

    def test_mlx_call_borrows_noncontiguous_view_for_call_scope(self):
        import mlx.core as mx

        base = torch.randn(4, 6, device="mps", dtype=torch.bfloat16)
        tensor = base.T
        weight = torch.randn(4, device="mps", dtype=torch.bfloat16)
        base_before = base.cpu().clone()
        tensor_before = tensor.cpu().clone()

        result = mlx_call(lambda x, w: mx.fast.rms_norm(x, w, 1e-6), tensor, weight)

        torch.mps.synchronize()
        self.assertTrue(torch.equal(base.cpu(), base_before))
        reference = torch.nn.functional.rms_norm(
            tensor_before, (4,), weight.cpu(), 1e-6
        )
        torch.testing.assert_close(result.cpu(), reference)

    def test_mlx_call_multi_fences_and_evaluates_once(self):
        """A multi-output island must not evaluate each result independently."""
        import mlx.core as mx

        source = torch.arange(8, device="mps", dtype=torch.float32)
        source_before = source.cpu().clone()
        captured = {}
        events = []
        real_synchronize = torch.mps.synchronize
        real_eval = mx.eval
        real_from_dlpack = torch.utils.dlpack.from_dlpack

        def synchronize_then_record():
            real_synchronize()
            events.append("fence returned")

        def operation(x):
            events.append("operation")
            captured["arrays"] = (x + 1, x * 2)
            return list(captured["arrays"])

        def evaluate_and_record(*arrays):
            events.append("eval")
            return real_eval(*arrays)

        def import_and_record(*args, **kwargs):
            events.append("dlpack")
            return real_from_dlpack(*args, **kwargs)

        with (
            mock.patch.object(
                torch.mps, "synchronize", side_effect=synchronize_then_record
            ) as synchronize,
            mock.patch.object(mx, "eval", side_effect=evaluate_and_record) as evaluate,
            mock.patch.object(
                torch.utils.dlpack,
                "from_dlpack",
                side_effect=import_and_record,
            ) as from_dlpack,
        ):
            first, second = mlx_call_multi(
                operation,
                source,
                device="mps",
            )

        synchronize.assert_called_once_with()
        self.assertEqual(
            events,
            ["fence returned", "operation", "eval", "dlpack", "dlpack"],
        )
        evaluate.assert_called_once()
        self.assertEqual(len(evaluate.call_args.args), 2)
        self.assertEqual(from_dlpack.call_count, 2)
        torch.testing.assert_close(first.cpu(), source_before + 1)
        torch.testing.assert_close(second.cpu(), source_before * 2)
        self.assertEqual(first.device.type, "mps")
        self.assertEqual(second.device.type, "mps")

        # Mutation through the Torch result remains visible from the original
        # MLX result allocation, proving that the positive-stride export did
        # not insert a copy.
        first.fill_(7)
        torch.mps.synchronize()
        self.assertTrue(mx.all(captured["arrays"][0] == 7).item())

    def test_mlx_call_multi_keeps_borrowed_inputs_alive_until_all_exports(self):
        source = torch.arange(8, device="mps", dtype=torch.float32)
        view = MlxTensorView(source)
        expected = source.cpu()

        first, second = mlx_call_multi(
            lambda x: (x + 3, x - 3),
            view,
            device="mps",
        )
        del source, view
        gc.collect()

        torch.testing.assert_close(first.cpu(), expected + 3)
        torch.testing.assert_close(second.cpu(), expected - 3)

    def test_concurrent_bridge_calls_are_serialized(self):
        """Concurrent bridge entry points must not race Metal command buffers."""
        script = """
from concurrent.futures import ThreadPoolExecutor
import torch
from sglang.srt.utils.tensor_bridge import mlx_call

source = torch.arange(8, device="mps", dtype=torch.float32)

def worker(iterations):
    for _ in range(iterations):
        result = mlx_call(lambda x: x + 1, source, device="mps")
        assert result.device.type == "mps"
        del result
    return True

with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(worker, 64), pool.submit(worker, 64)]
    assert all(future.result() for future in futures)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )

    def test_mlx_call_multi_rejects_non_sequence_output(self):
        source = torch.ones(2, device="mps", dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "non-empty tuple or list"):
            mlx_call_multi(lambda x: x + 1, source, device="mps")

    def test_mlx_call_multi_cpu_input_does_not_fence_mps(self):
        source = torch.arange(8, dtype=torch.float32)

        with mock.patch.object(torch.mps, "synchronize") as synchronize:
            (result,) = mlx_call_multi(
                lambda x: (x + 1,),
                source,
                device="mps",
            )

        synchronize.assert_not_called()
        torch.testing.assert_close(result.cpu(), source + 1)

    def test_mlx_call_multi_sync_failure_precedes_graph_build(self):
        source = torch.ones(1, device="mps")
        operation = mock.Mock()

        with (
            mock.patch.object(
                torch.mps,
                "synchronize",
                side_effect=RuntimeError("producer fence failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "producer fence failed"),
        ):
            mlx_call_multi(
                operation,
                source,
                device="mps",
            )

        operation.assert_not_called()

    def test_mlx_call_multi_propagates_operation_failure(self):
        source = torch.ones(1, device="mps")

        def operation(_source):
            raise RuntimeError("graph build failed")

        with self.assertRaisesRegex(RuntimeError, "graph build failed"):
            mlx_call_multi(
                operation,
                source,
                device="mps",
            )

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

    def test_torch_cpu_float64_does_not_silently_downcast(self):
        import mlx.core as mx

        tensor = torch.tensor([1.25, -2.5], dtype=torch.float64)
        array = torch_to_mlx(tensor)
        mx.eval(array)
        self.assertEqual(array.dtype, mx.float64)
        round_tripped = mlx_to_torch(array, device="cpu")
        self.assertEqual(round_tripped.dtype, torch.float64)
        torch.testing.assert_close(round_tripped, tensor)

    def test_unsupported_cpu_dtype_fails_instead_of_narrowing(self):
        tensor = torch.tensor([1 + 2j], dtype=torch.complex128)
        with self.assertRaisesRegex(ValueError, "complex128"):
            torch_to_mlx(tensor)

    def test_cpu_float64_export_is_materialized_on_cpu(self):
        import mlx.core as mx

        with mx.stream(mx.cpu):
            array = mx.array([1.25, -2.5], dtype=mx.float64)
        tensor = mlx_to_torch(array, device="cpu")
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tensor.dtype, torch.float64)
        torch.testing.assert_close(
            tensor, torch.tensor([1.25, -2.5], dtype=torch.float64)
        )

    def test_cpu_float64_positive_stride_export_remains_zero_copy(self):
        import mlx.core as mx

        with mx.stream(mx.cpu):
            base = mx.arange(8).astype(mx.float64)
            array = base[::2]
        tensor = mlx_to_torch(array, device="cpu")
        self.assertEqual(tensor.stride(), (2,))

        tensor.fill_(11)
        torch.testing.assert_close(
            torch.utils.dlpack.from_dlpack(array.__dlpack__(dl_device=(1, 0))),
            torch.full((4,), 11, dtype=torch.float64),
        )

    def test_cpu_export_consumes_the_dlpack_capsule_once(self):
        """A DLPack capsule is single-use and must not be imported twice."""
        import mlx.core as mx

        with mx.stream(mx.cpu):
            array = mx.array([1.25, -2.5], dtype=mx.float32)

        with mock.patch.object(
            torch.utils.dlpack,
            "from_dlpack",
            wraps=torch.utils.dlpack.from_dlpack,
        ) as from_dlpack:
            tensor = mlx_to_torch(array, device="cpu")

        self.assertEqual(from_dlpack.call_count, 1)
        torch.testing.assert_close(
            tensor, torch.tensor([1.25, -2.5], dtype=torch.float32)
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

    def test_negative_stride_views_materialize_without_aborting(self):
        script = """
import mlx.core as mx
import torch
from sglang.srt.utils.tensor_bridge import mlx_call, mlx_to_torch

expected = torch.arange(15, -1, -1, dtype=torch.float32)
for target in ("cpu", "mps"):
    array = mx.arange(16, dtype=mx.float32)[::-1]
    tensor = mlx_to_torch(array, device=target)
    torch.testing.assert_close(tensor.cpu(), expected)
    tensor.zero_()
    if target == "mps":
        torch.mps.synchronize()
    assert mx.array_equal(array, mx.arange(16, dtype=mx.float32)[::-1]).item()

with mx.stream(mx.cpu):
    array = mx.arange(16).astype(mx.float64)[::-1]
tensor = mlx_to_torch(array, device="cpu")
torch.testing.assert_close(
    tensor, torch.arange(15, -1, -1, dtype=torch.float64)
)
try:
    mlx_to_torch(array, device="mps")
except ValueError as exc:
    assert "float64" in str(exc)
else:
    raise AssertionError("float64 MLX export to MPS must fail explicitly")

source = torch.arange(16, device="mps", dtype=torch.float32)
result = mlx_call(lambda x: x[::-1], source, device="mps")
torch.testing.assert_close(result.cpu(), expected)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout={completed.stdout}\nstderr={completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()

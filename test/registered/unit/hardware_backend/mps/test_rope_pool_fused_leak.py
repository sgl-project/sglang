"""rope_pool_fused must return MLX arrays and release every output.

The binding once placed each output over an mx.array(0) without destroying it,
leaking one scalar per output on every call, evaluated or not.
"""

import gc
import importlib.util
import unittest

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")

T, HQ, HKV, D, POOL, BASE = 3, 4, 2, 64, 16, 10000.0
CALLS = 1000


def _has_metal_kernels():
    if importlib.util.find_spec("mlx") is None:
        return False
    try:
        from sgl_kernel import metal
    except ImportError:
        return False
    return metal._metal is not None


def _inputs():
    import mlx.core as mx

    mx.random.seed(0)
    q = mx.random.normal((T, HQ, D))
    k = mx.random.normal((T, HKV, D))
    v = mx.random.normal((T, HKV, D))
    positions = mx.array([0, 5, 11], dtype=mx.int32)
    slots = mx.array([2, -1, 9], dtype=mx.int32)
    k_pool = mx.zeros((POOL, HKV, D))
    v_pool = mx.zeros((POOL, HKV, D))
    mx.eval(q, k, v, positions, slots, k_pool, v_pool)
    return q, k, v, positions, slots, k_pool, v_pool


def _call(args):
    from sgl_kernel.metal import rope_pool_fused

    return rope_pool_fused(
        *args, head_dim=D, num_qo_heads=HQ, num_kv_heads=HKV, rope_base=BASE
    )


def _neox(x, positions):
    import mlx.core as mx

    half = D // 2
    inv = BASE ** (-mx.arange(half, dtype=mx.float32) * 2 / D)
    angle = positions.astype(mx.float32)[:, None] * inv[None, :]
    cos, sin = mx.cos(angle)[:, None, :], mx.sin(angle)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _active():
    import mlx.core as mx

    gc.collect()
    mx.synchronize()
    return mx.get_active_memory()


@unittest.skipUnless(
    _has_metal_kernels(), "Requires MLX and the sgl_kernel Metal extension"
)
class TestRopePoolFusedLeak(CustomTestCase):
    def test_outputs_are_mlx_arrays_and_match_reference(self):
        import mlx.core as mx

        args = _inputs()
        q, k, v, positions, slots, k_pool, v_pool = args
        outs = _call(args)
        self.assertIsInstance(outs, tuple)
        self.assertEqual(len(outs), 4)
        for out in outs:
            self.assertIsInstance(out, mx.array)
        q_rot, k_rot, k_pool_new, v_pool_new = outs
        mx.eval(*outs)
        k_ref = _neox(k, positions)
        k_pool_ref, v_pool_ref = mx.array(k_pool), mx.array(v_pool)
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                k_pool_ref[slot] = k_ref[token]
                v_pool_ref[slot] = v[token]
        self.assertTrue(mx.allclose(q_rot, _neox(q, positions), atol=1e-4).item())
        self.assertTrue(mx.allclose(k_rot, k_ref, atol=1e-4).item())
        self.assertTrue(mx.allclose(k_pool_new, k_pool_ref, atol=1e-4).item())
        self.assertTrue(mx.allclose(v_pool_new, v_pool_ref, atol=1e-4).item())

    def _assert_no_leak(self, evaluate):
        import mlx.core as mx

        args = _inputs()
        # Warm up the library, kernels and allocator before the baseline.
        for _ in range(10):
            mx.eval(*_call(args))
        baseline = _active()
        for _ in range(CALLS):
            outs = _call(args)
            for out in outs:
                self.assertIsInstance(out, mx.array)
            if evaluate:
                mx.eval(*outs)
            del outs, out
        delta = _active() - baseline
        self.assertEqual(
            delta, 0, f"active memory grew {delta} bytes over {CALLS} calls"
        )

    def test_no_leak_unevaluated(self):
        self._assert_no_leak(evaluate=False)

    def test_no_leak_evaluated(self):
        self._assert_no_leak(evaluate=True)


if __name__ == "__main__":
    unittest.main()

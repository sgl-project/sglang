from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_mlx_ci

register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_PROVIDER = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_vlm.speculative.drafters.gemma4_assistant")
    is not None
)

if _HAS_PROVIDER:
    import mlx.core as mx
    from registered.unit.hardware_backend.mlx.gemma4_test_utils import (
        build_runner,
        tiny_gemma4,
        write_tiny_assistant_checkpoint,
    )

    from sglang.srt.hardware_backend.mlx.gemma4_mtp import (
        Gemma4MTPAssistantLoader,
        Gemma4MTPKVSharingPlan,
    )


@unittest.skipUnless(_HAS_PROVIDER, "requires mlx-vlm Gemma 4 assistant provider")
class TestGemma4MTPKVSharing(unittest.TestCase):
    def _runtime(self, root: Path):
        target = tiny_gemma4()
        write_tiny_assistant_checkpoint(root)
        loader = Gemma4MTPAssistantLoader(target)
        return target, loader, loader.load(str(root))

    def test_logical_owner_compact_mapping_and_runner_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            target, _loader, runtime = self._runtime(Path(temp))
            plan = runtime.sharing_plan
            self.assertEqual(plan.target_logical_layers, (2, 3, 2, 3))
            self.assertEqual(plan.target_owner_layers, (0, 1, 0, 1))
            self.assertEqual(plan.compact_cache_indices, (0, 1, 0, 1))
            self.assertEqual(plan.compact_owner_layers, (0, 1))

            runner = build_runner(target)
            self.assertEqual(
                runner._cache_layout.native_cache_owner_by_layer, (0, 1, 0, 1)
            )
            self.assertEqual(runner._cache_layout.native_cache_index(2), 0)
            self.assertEqual(runner._cache_layout.native_cache_index(3), 1)

    def test_nested_or_out_of_range_owner_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            target, _loader, runtime = self._runtime(Path(temp))
            backbone = target.model
            original = list(backbone.previous_kvs)
            for invalid in ([0, 1, 3, 1], [0, 1, 99, 1]):
                with self.subTest(invalid=invalid):
                    backbone.previous_kvs = invalid
                    with self.assertRaises(ValueError):
                        Gemma4MTPKVSharingPlan.from_target(target, runtime.metadata)
            backbone.previous_kvs = original

    def test_rotating_view_restores_absolute_positions_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            target, _loader, runtime = self._runtime(Path(temp))
            cache = target.make_cache()
            prompt = list(range(1, 18))
            target(mx.array([prompt], dtype=mx.int32), cache=cache)
            mx.eval(*[item for entry in cache for item in entry.state])
            identities = tuple(map(id, cache))
            offsets = tuple(entry.offset for entry in cache)
            ring_indices = tuple(getattr(entry, "_idx", None) for entry in cache)

            view = runtime.bind_request("r", cache)
            shared = view.shared_kv_states()
            local_k, local_v = shared["sliding_attention"]
            full_k, full_v = shared["full_attention"]
            mx.eval(local_k, local_v, full_k, full_v)

            self.assertEqual(view.position, 17)
            self.assertEqual(local_k.shape[-2], 17)
            self.assertEqual(full_k.shape[-2], 17)
            # The first nine local positions are masked padding; the temporal
            # target window occupies the absolute suffix.
            np.testing.assert_array_equal(np.array(local_k[..., :9, :]), 0)
            self.assertEqual(tuple(map(id, cache)), identities)
            self.assertEqual(tuple(entry.offset for entry in cache), offsets)
            self.assertEqual(
                tuple(getattr(entry, "_idx", None) for entry in cache), ring_indices
            )

    def test_request_isolation_and_lifecycle_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            target, loader, runtime = self._runtime(Path(temp))
            cache_a = target.make_cache()
            cache_b = target.make_cache()
            target(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache_a)
            target(mx.array([[7, 8, 9, 10]], dtype=mx.int32), cache=cache_b)
            view_a = runtime.bind_request("a", cache_a)
            view_b = runtime.bind_request("b", cache_b)
            self.assertEqual(view_a.position, 3)
            self.assertEqual(view_b.position, 4)
            self.assertFalse(
                np.array_equal(
                    np.array(view_a.shared_kv_states()["full_attention"][0]),
                    np.array(view_b.shared_kv_states()["full_attention"][0]),
                )
            )

            runtime.release_request("a")
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = view_a.position
            self.assertEqual(view_b.position, 4)
            loader.clear_request_bindings()
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _ = view_b.position


if __name__ == "__main__":
    unittest.main()

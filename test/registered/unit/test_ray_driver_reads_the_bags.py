"""The Ray driver sizes its actors from the published configuration.

`RayEngine` publishes as part of `Engine._launch_subprocesses` and *then* lays
out the actors, so the placement arithmetic reads the `parallel` bag. That is
where a resolution decision lives: a launch that leaves `dp_size` to resolution
has it in the `parallel` bag, and the override case below is what tells the two
apart.

There is no CI coverage of the Ray path (`test/manual/test_ray_engine.py` boots a
real cluster), so these cases drive the two pure helpers directly against a
published config -- including the override direction, which is what tells a bag
read from a record read.
"""

import importlib.util
import unittest

from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# `sglang.srt.ray.engine` imports `ray` at module scope, and the CPU runner has
# no ray wheel.
_HAS_RAY = importlib.util.find_spec("ray") is not None
_needs_ray = unittest.skipUnless(_HAS_RAY, "ray is not installed")


class TestRayDriverReadsTheBags(CustomTestCase):
    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    @_needs_ray
    def test_world_size_multiplies_the_published_sizes(self):
        from sglang.srt.ray.engine import _compute_world_size

        self._publish(tp_size=2, pp_size=3, dp_size=4, enable_dp_attention=False)
        self.assertEqual(_compute_world_size(), 24)

    @_needs_ray
    def test_dp_attention_folds_dp_into_tp(self):
        from sglang.srt.ray.engine import _compute_world_size

        self._publish(tp_size=4, pp_size=2, dp_size=4, enable_dp_attention=True)
        # DP attention folds DP into TP, so dp_size drops out of the product.
        self.assertEqual(_compute_world_size(), 8)

    @_needs_ray
    def test_the_world_size_follows_a_post_publish_override(self):
        """The direction that separates a bag read from a record read.

        `override` writes the bag and never the record, so a driver still
        reading `server_args.tp_size` would keep answering with the old size.
        """
        from sglang.srt.ray.engine import _compute_world_size

        self._publish(tp_size=2, pp_size=1, dp_size=1, enable_dp_attention=False)
        self.assertEqual(_compute_world_size(), 2)
        get_context().override("test.ray_driver", tp_size=8)
        self.assertEqual(get_parallel().tp_size, 8)
        self.assertEqual(_compute_world_size(), 8)


if __name__ == "__main__":
    unittest.main()

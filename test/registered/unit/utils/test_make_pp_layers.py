import unittest

from torch import nn

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import make_pp_layers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_LAYERS = 6


def _layer(idx, prefix):
    return nn.Linear(1, 1)


def _stage():
    """Build the layers here; return the stage range and the indices built."""
    layers, start, end = make_pp_layers(NUM_LAYERS, _layer, prefix="model.layers")
    built = [
        i for i, layer in enumerate(layers) if not isinstance(layer, PPMissingLayer)
    ]
    return (start, end), built


class TestMakePPLayers(CustomTestCase):
    def test_each_stage_builds_its_own_slice(self):
        for pp_rank, stage in ((0, (0, 3)), (1, (3, 6))):
            with (
                self.subTest(pp_rank=pp_rank),
                published_topology(pp_size=2, ranks=dict(world_rank=pp_rank)),
            ):
                self.assertEqual(_stage(), (stage, list(range(*stage))))

    def test_a_single_stage_scope_builds_every_layer(self):
        # A draft is built with its pipeline narrowed to one stage.
        with published_topology(pp_size=2, ranks=dict(world_rank=1)):
            with get_parallel().override(pp_size=1, pp_rank=0):
                stage, built = _stage()
        self.assertEqual(stage, (0, NUM_LAYERS))
        self.assertEqual(built, list(range(NUM_LAYERS)))


if __name__ == "__main__":
    unittest.main()

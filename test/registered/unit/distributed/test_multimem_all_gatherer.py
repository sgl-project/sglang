"""Building a model runs no collectives in MultimemAllGatherer: a weight-update replica builds the engine's model
over mock process groups, where a node-topology all-reduce cannot run."""

import unittest
from unittest.mock import patch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMultimemAllGathererConstruction(CustomTestCase):
    def test_a_multi_node_deployment_probes_no_topology_at_construction(self):
        """A TP16 engine on two nodes made the replica's LogitsProcessor run in_the_same_node_as on a mock group."""

        def no_collectives(*args, **kwargs):
            raise AssertionError("constructing the gatherer ran a collective")

        with (
            get_parallel().override(nnodes=2),
            patch.object(parallel_state, "in_the_same_node_as", no_collectives),
            patch.object(parallel_state, "get_tp_group", no_collectives),
        ):
            MultimemAllGatherer(max_tokens=8)


if __name__ == "__main__":
    unittest.main()

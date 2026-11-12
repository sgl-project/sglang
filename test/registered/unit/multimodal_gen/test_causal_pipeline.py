# SPDX-License-Identifier: Apache-2.0
"""Causal pipeline stage partitioning, placement and configuration contracts."""

import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.runtime.distributed.causal_pipeline import (
    CausalPipelinePlacement,
    CausalPipelinePlan,
)
from sglang.multimodal_gen.runtime.distributed.causal_pipeline_config import (
    configure_causal_pipeline_parallelism,
    validate_causal_pipeline_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestCausalPipeline(CustomTestCase):
    def test_partition_covers_each_step_layer_once_in_order(self):
        for mode, degrees in (("timestep", (1, 2, 3)), ("block", (1, 2, 3, 4, 7, 15))):
            for degree in degrees:
                with self.subTest(mode=mode, degree=degree):
                    plan = CausalPipelinePlan(3, 5, degree, mode)
                    visited = [
                        (part.timestep, layer)
                        for rank in range(degree)
                        for part in plan.slices(rank)
                        for layer in range(part.block_start, part.block_end)
                    ]
                    self.assertEqual(
                        visited,
                        [(step, layer) for step in range(3) for layer in range(5)],
                    )
        for mode, degree in (("timestep", 4), ("block", 16)):
            with self.assertRaises(ValueError):
                CausalPipelinePlan(3, 5, degree, mode)

    def test_vae_placement_rejects_nonpositive_degrees(self):
        for degree in (0, -1):
            with self.assertRaises(ValueError):
                CausalPipelinePlacement(degree)

    def test_parallel_defaults_do_not_spend_decoder_rank_on_sequence_parallelism(self):
        args = SimpleNamespace(
            causal_pipeline_mode="timestep",
            causal_pipeline_parallel_size=4,
            num_gpus=5,
            tp_size=None,
            sp_degree=None,
            ulysses_degree=None,
            ring_degree=None,
            kv_gather_degree=None,
            dp_size=1,
            enable_cfg_parallel=None,
            cfg_parallel_degree=None,
            encoder_parallel="auto",
        )
        configure_causal_pipeline_parallelism(args)
        self.assertEqual(
            (args.sp_degree, args.tp_size, args.cfg_parallel_degree), (1, 1, 1)
        )
        self.assertEqual(args.encoder_parallel, "replicate")
        args.num_gpus = 4
        with self.assertRaisesRegex(ValueError, "must be 5"):
            configure_causal_pipeline_parallelism(args)

    def test_disabled_mode_cannot_silently_accept_pipeline_options(self):
        args = SimpleNamespace(
            causal_pipeline_mode="off",
            causal_pipeline_parallel_size=2,
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            validate_causal_pipeline_args(args)


if __name__ == "__main__":
    unittest.main()

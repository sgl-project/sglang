"""A pipeline neighbour layer is built on the meta device and allocates nothing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe
from sglang.srt.models.qwen4_exp import _offloads_ple
from sglang.srt.utils.common import (
    building_neighbour_layer,
    is_building_neighbour_layer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNeighbourLayerBuild(CustomTestCase):
    def test_the_flag_holds_only_inside_the_build(self):
        self.assertFalse(is_building_neighbour_layer())
        with building_neighbour_layer():
            self.assertTrue(is_building_neighbour_layer())
            self.assertEqual(torch.empty(1).device.type, "meta")
        self.assertFalse(is_building_neighbour_layer())

    def test_a_gated_residual_allocates_nothing(self):
        config = HyperConnectionConfig(hc_count=4, hidden_size=64, hc_lowrank=16)
        with building_neighbour_layer():
            module = GatedResidual(config, use_mix=True, use_combine=True)
        tensors = [*module.parameters(), *module.buffers()]
        self.assertTrue(tensors)
        self.assertEqual({t.device.type for t in tensors}, {"meta"})

    def test_a_neighbour_keeps_its_ple_table_off_the_host(self):
        offloaded = SimpleNamespace(ple_offload_embedding=True)
        self.assertTrue(_offloads_ple(offloaded))
        with building_neighbour_layer():
            self.assertFalse(_offloads_ple(offloaded))
        self.assertFalse(_offloads_ple(SimpleNamespace(ple_offload_embedding=False)))

    def test_a_neighbour_moe_sets_up_no_all_to_all(self):
        class FlashinferA2A:
            def is_flashinfer(self):
                return True

            def __getattr__(self, name):
                if name.startswith("is_"):
                    return lambda: False
                raise AttributeError(name)

        def unexpected(**kwargs):
            raise AssertionError("a neighbour layer built an all-to-all dispatcher")

        config = SimpleNamespace(
            top_k=2, num_experts=4, num_local_experts=4, hidden_size=8
        )
        parallel = SimpleNamespace(tp_group=SimpleNamespace(device_group=None))
        standard = object()
        with (
            patch.object(fused_moe, "get_moe_a2a_backend", FlashinferA2A),
            patch.object(fused_moe, "get_parallel", lambda: parallel),
            patch.object(fused_moe, "FlashinferDispatcher", unexpected),
            patch.object(fused_moe, "StandardDispatcher", lambda config: standard),
        ):
            with building_neighbour_layer():
                self.assertIs(fused_moe.create_moe_dispatcher(config, None), standard)
            with self.assertRaisesRegex(AssertionError, "all-to-all dispatcher"):
                fused_moe.create_moe_dispatcher(config, None)


if __name__ == "__main__":
    unittest.main()

"""Muse Spark collectives: the automatic projection dispatch stays inside its validated
envelope, and peer memory stays off when the TP group spans hosts."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.communication import muse_spark_all_gather as gather
from sglang.srt.distributed.device_communicators.symm_workspace import (
    WorkspaceLease,
)
from sglang.srt.environ import envs
from sglang.srt.layers.muse_spark import muse_spark_dynamic_guests as dg
from sglang.srt.layers.muse_spark import muse_spark_sp_experts as spe
from sglang.srt.layers.muse_spark.fused import glue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")


class TestMuseSparkProjectionSupport(CustomTestCase):
    def setUp(self):
        self.comm = Mock(spec=WorkspaceLease)
        self.comm.world_size = 8
        self.comm.has_multicast = True
        self.comm.request = SimpleNamespace(push_slot_bytes=512 * 8192 * 2)

    def supported(self, m=16, n=4096, capability=(10, 0), *, x=None, weight=None):
        x = torch.empty(m, 64, dtype=torch.bfloat16) if x is None else x
        weight = torch.empty(n, 64, dtype=torch.bfloat16) if weight is None else weight
        with (
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(gather, "current_arch") as arch,
        ):
            arch.return_value.is_sm.side_effect = lambda *sm: sm == capability
            return gather.gemm_ag_supported(self.comm, x, weight)

    def test_b200_tp8_token_and_output_boundaries(self):
        for n in (4096, 8192):
            for m in (9, 16, 128):
                with self.subTest(m=m, n=n):
                    self.assertTrue(self.supported(m, n))
            self.assertFalse(self.supported(129, n))
        self.assertFalse(self.supported(16, 2048))
        self.assertFalse(self.supported(0))

    def test_existing_hardware_and_small_projection_paths(self):
        for tp in (4, 8):
            self.comm.world_size = tp
            self.assertTrue(self.supported(512, capability=(10, 3)))
            self.assertFalse(self.supported(513, capability=(10, 3)))
        for tp in (2, 4):
            self.comm.world_size = tp
            self.assertFalse(self.supported(9))
            self.assertTrue(self.supported(8))
        for tp in (4, 8):  # SM90 pooler projections
            self.comm.world_size = tp
            for n in (2048, 4096, 8192):
                self.assertTrue(self.supported(512, n, capability=(9, 0)))
            self.assertFalse(self.supported(513, capability=(9, 0)))
            self.assertFalse(self.supported(9, 1024, capability=(9, 0)))
        self.comm.world_size = 8
        for capability in ((10, 1), (12, 0)):
            self.assertFalse(self.supported(9, capability=capability))

    def test_sm90_switch_off_keeps_the_stock_projection_dispatch(self):
        """SGLANG_MUSE_SPARK_HOPPER_KERNELS=0 on SM90: the pooler / unpooler workspace
        and gemm_ag stay as without the Hopper path (only the small projection)."""
        self.addCleanup(glue.hopper_kernels_enabled.cache_clear)
        for on in (True, False):
            glue.hopper_kernels_enabled.cache_clear()
            with (
                self.subTest(on=on),
                envs.SGLANG_MUSE_SPARK_HOPPER_KERNELS.override(on),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(torch.version, "hip", None),
                patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            ):
                self.assertEqual(glue.sliced_projection_gpu(), on)
                self.assertEqual(self.supported(512, capability=(9, 0)), on)
                self.assertTrue(self.supported(8, capability=(9, 0)))
        with (
            envs.SGLANG_MUSE_SPARK_HOPPER_KERNELS.override(False),
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3)),
        ):
            self.assertTrue(glue.sliced_projection_gpu())

    def test_dtype_shape_and_layout_fallbacks(self):
        self.assertFalse(self.supported(x=torch.empty(16, 64, dtype=torch.float16)))
        self.assertFalse(self.supported(weight=torch.empty(4096, 64)))
        self.assertFalse(self.supported(x=torch.empty(1, 16, 64, dtype=torch.bfloat16)))
        self.assertFalse(
            self.supported(weight=torch.empty(4096, 32, dtype=torch.bfloat16))
        )
        weight = torch.empty(64, 4096, dtype=torch.bfloat16).T
        self.assertFalse(self.supported(weight=weight))

    def test_communicator_and_workspace_fallbacks(self):
        with patch.object(self, "comm", None):
            self.assertFalse(self.supported())
        with patch.object(self.comm, "has_multicast", False):
            self.assertFalse(self.supported())
        with patch.object(self.comm, "world_size", 1):
            self.assertFalse(self.supported())
        with patch.object(self.comm.request, "push_slot_bytes", 16 * 4096 * 2 - 1):
            self.assertFalse(self.supported())


class _Group:  # hashable, as GroupCoordinator
    def __init__(self, name):
        self.world_size, self.rank_in_group, self.cpu_group = 8, 0, object()
        self.device_group = SimpleNamespace(group_name=name)


class TestMuseSparkSpOneHost(CustomTestCase):
    def setUp(self):
        dg.tp_group_on_one_host.cache_clear()
        spe._state.clear()

    def _prepare_guests(self, group, nnodes, same_node):
        layers = [
            SimpleNamespace(
                use_fused_w4a16=True,
                global_num_experts=256,
                num_local_experts=32,
                dynamic_guest_setup=None,
            )
        ]
        parallel = SimpleNamespace(tp_group=group, nnodes=nnodes)
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.distributed.parallel_state.in_the_same_node_as",
                return_value=same_node,
            ) as probe,
            patch.object(
                torch._C, "_accelerator_getAllocatorSettings", return_value=""
            ),
        ):
            dg.maybe_prepare_dynamic_guests(layers)
        return layers[0].dynamic_guest_setup, probe.call_count

    def test_tp8_over_two_hosts_keeps_dynamic_guests_off(self):
        """TP8 on two 4-GPU hosts: a peer's CUDA IPC handle would not open."""
        setup, probes = self._prepare_guests(
            _Group("tp:0"), 2, [True] * 4 + [False] * 4
        )
        self.assertIsNone(setup)
        self.assertEqual(probes, 1)

    def test_one_host_enables_dynamic_guests(self):
        setup, probes = self._prepare_guests(_Group("tp:1"), 1, None)
        self.assertIsInstance(setup, dg._Setup)
        self.assertEqual(probes, 0)  # one node: no probe collective
        setup, _ = self._prepare_guests(_Group("tp:2"), 2, [True] * 8)
        self.assertIsInstance(setup, dg._Setup)

    def test_sp_buffers_fall_back_to_nccl_across_hosts(self):
        """Symmetric memory rendezvous needs every rank on one host."""
        parallel = SimpleNamespace(nnodes=2)
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch(
                "sglang.srt.distributed.parallel_state.in_the_same_node_as",
                return_value=[True] * 4 + [False] * 4,
            ),
            patch.object(spe, "SPExpertBuffers") as buffers,
        ):
            self.assertIsNone(spe.prepare(_Group("tp:3"), torch.device("cpu"), 4096))
        buffers.assert_not_called()


if __name__ == "__main__":
    unittest.main()

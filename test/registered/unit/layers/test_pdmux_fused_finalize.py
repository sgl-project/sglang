import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.communication import all_reduce_fusion
from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPdmuxFusedFinalize(unittest.TestCase):
    def test_pdmux_does_not_register_the_shared_push_plane(self):
        with (
            patch.object(
                mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=True)
            ),
            patch.object(mxfp4, "get_parallel") as parallel,
            patch.object(all_reduce_fusion, "register_comm") as register,
        ):
            mxfp4.register_fused_all_reduce_comm()
            parallel.assert_not_called()
            register.assert_not_called()

    def test_pdmux_capability_check_stops_before_workspace_lookup(self):
        with (
            patch.object(
                mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=True)
            ),
            patch.object(all_reduce_fusion, "get_registered_comm") as lookup,
        ):
            self.assertFalse(mxfp4.can_fuse_all_reduce(8, 5120))
            lookup.assert_not_called()

    def test_non_pdmux_keeps_existing_registered_comm(self):
        comm = SimpleNamespace(
            config=SimpleNamespace(num_push_blocks=384), max_push_size=123
        )
        with (
            patch.object(
                mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=False)
            ),
            patch.object(
                mxfp4,
                "get_parallel",
                return_value=SimpleNamespace(
                    tp_size=4, tp_group=SimpleNamespace(ca_comm=comm)
                ),
            ),
            patch.object(
                all_reduce_fusion, "get_registered_comm", return_value=object()
            ),
            patch.object(
                all_reduce_fusion, "fits_push_slot", return_value=True
            ) as fits,
        ):
            self.assertTrue(mxfp4.can_fuse_all_reduce(8, 5120))
            fits.assert_called_once_with(123, 8, 5120)
            self.assertFalse(mxfp4.can_fuse_all_reduce(385, 5120))


if __name__ == "__main__":
    unittest.main()

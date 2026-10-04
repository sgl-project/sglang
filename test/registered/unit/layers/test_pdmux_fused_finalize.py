import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.layers.quantization import mxfp4_flashinfer_trtllm_moe as mxfp4
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPdmuxFusedFinalize(unittest.TestCase):
    def test_pdmux_does_not_probe_or_reuse_shared_push_plane(self):
        for initialized in (False, True):
            with (
                self.subTest(initialized=initialized),
                patch.object(
                    mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=True)
                ),
                patch.object(mxfp4, "get_parallel") as parallel,
                patch.object(mxfp4, "_fused_finalize_all_reduce_probed", initialized),
                patch.object(
                    mxfp4,
                    "_fused_finalize_all_reduce_world_size",
                    4 if initialized else None,
                ),
                patch.object(
                    mxfp4.torch.cuda, "is_current_stream_capturing"
                ) as capturing,
            ):
                self.assertIsNone(mxfp4._fused_finalize_all_reduce_comm_world_size())
                self.assertEqual(mxfp4._fused_finalize_all_reduce_probed, initialized)
                parallel.assert_not_called()
                capturing.assert_not_called()

    def test_pdmux_capability_check_stops_before_workspace_initialization(self):
        with (
            patch.object(
                mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=True)
            ),
            patch.object(
                mxfp4, "_fused_finalize_all_reduce_comm_world_size"
            ) as reserve,
        ):
            self.assertFalse(mxfp4.should_use_fuse_finalize_all_reduce(Mock(), 8, 5120))
            reserve.assert_not_called()

    def test_non_pdmux_keeps_existing_reserved_world_size(self):
        with (
            patch.object(
                mxfp4, "get_disagg", return_value=SimpleNamespace(enable_pdmux=False)
            ),
            patch.object(mxfp4, "_fused_finalize_all_reduce_probed", True),
            patch.object(mxfp4, "_fused_finalize_all_reduce_world_size", 4),
        ):
            self.assertEqual(mxfp4._fused_finalize_all_reduce_comm_world_size(), 4)


if __name__ == "__main__":
    unittest.main()

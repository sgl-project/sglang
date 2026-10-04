"""Full-device decode may overlap a capped PDMux prefill Green Context."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_config(body):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pdmux.yaml"
        path.write_text(body)
        return pdmux_context.load_pdmux_config(str(path))


class TestPDMuxOverlapStreams(unittest.TestCase):
    def setUp(self):
        self.cuda_streams = []

        def make_cuda_stream(device, priority=0):
            stream = SimpleNamespace(
                cuda_stream=f"cuda-{len(self.cuda_streams)}", priority=priority
            )
            self.cuda_streams.append(stream)
            return stream

        spatial = Mock()
        spatial.get_sm_available.return_value = 132
        spatial.create_greenctx_stream_by_value.return_value = (
            SimpleNamespace(cuda_stream="prefill-green"),
            SimpleNamespace(cuda_stream="unused-green"),
        )
        kernel = ModuleType("sgl_kernel")
        kernel.spatial = spatial
        self.spatial = spatial
        self.patches = [
            patch.dict(sys.modules, {"sgl_kernel": kernel}),
            patch.object(pdmux_context.torch.cuda, "current_device", return_value=0),
            patch.object(
                pdmux_context.torch.cuda,
                "Stream",
                side_effect=make_cuda_stream,
            ),
            patch.object(pdmux_context, "STREAM_GROUPS", []),
            patch.object(pdmux_context, "SM_COUNTS", []),
            patch.object(pdmux_context, "_RESERVED_GREEN_STREAMS", []),
            patch.object(pdmux_context, "_FULL_DEVICE_DECODE_STREAMS", set()),
            patch.object(pdmux_context, "CURRENT_STREAM_IDX", 0),
            patch.object(pdmux_context, "CURRENT_STREAM_GROUP", None),
        ]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def test_full_device_decode_uses_primary_stream(self):
        config = load_config("""sm_group_num: 3
manual_divisions:
  - [104, 132, 0]
split_forward_token_budget: 65536
overlap_decode_full_sm: true
""")

        pdmux_context.initialize_stream_groups(0, config)

        self.spatial.create_greenctx_stream_by_value.assert_called_once_with(104, 28, 0)
        self.assertEqual(pdmux_context.get_sm_counts()[1], (104, 132))
        self.assertEqual(
            pdmux_context.get_stream_groups()[1][0].cuda_stream, "prefill-green"
        )
        self.assertEqual(pdmux_context.get_stream_groups()[1][1].priority, -1)
        self.assertEqual(
            [s.cuda_stream for s in pdmux_context._RESERVED_GREEN_STREAMS],
            ["unused-green"],
        )

    def test_helper_selection_uses_capture_stream_instead_of_group_index(self):
        config = pdmux_context.PDMuxConfig(
            sm_group_num=3,
            manual_divisions=[[104, 132, 0]],
            overlap_decode_full_sm=True,
        )
        pdmux_context.initialize_stream_groups(0, config)
        self.assertEqual(pdmux_context.get_current_stream_idx(), 0)
        helper = object()
        for lane, expected in ((0, None), (1, helper)):
            with patch.object(
                pdmux_context.torch.cuda,
                "current_stream",
                return_value=pdmux_context.get_stream_groups()[1][lane],
            ):
                self.assertIs(
                    pdmux_context.get_pdmux_decode_alt_stream(helper), expected
                )

        # An unrelated full-device stream is not a decode lane.
        with patch.object(
            pdmux_context.torch.cuda,
            "current_stream",
            return_value=pdmux_context.get_stream_groups()[0][0],
        ):
            self.assertIsNone(pdmux_context.get_pdmux_decode_alt_stream(helper))

    def test_exclusive_decode_keeps_green_partition(self):
        config = pdmux_context.PDMuxConfig(
            sm_group_num=3, manual_divisions=[[104, 28, 0]]
        )
        pdmux_context.initialize_stream_groups(0, config)
        helper = object()
        for idx, expected in ((1, None), (2, helper)):
            with patch.object(
                pdmux_context.torch.cuda,
                "current_stream",
                return_value=pdmux_context.get_stream_groups()[idx][1],
            ):
                self.assertIs(
                    pdmux_context.get_pdmux_decode_alt_stream(helper), expected
                )

    def test_oversubscribed_exclusive_division_is_rejected_early(self):
        config = load_config("""sm_group_num: 3
manual_divisions:
  - [104, 132, 0]
""")

        with self.assertRaisesRegex(ValueError, "must equal the device SM count"):
            pdmux_context.initialize_stream_groups(0, config)
        self.spatial.create_greenctx_stream_by_value.assert_not_called()

    def test_overlap_requires_manual_division(self):
        with self.assertRaisesRegex(ValueError, "requires manual_divisions"):
            load_config("sm_group_num: 3\noverlap_decode_full_sm: true\n")


if __name__ == "__main__":
    unittest.main()

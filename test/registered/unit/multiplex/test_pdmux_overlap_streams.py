"""Full-device decode may overlap a capped PDMux prefill Green Context."""

import os
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
            patch.dict(os.environ, {}, clear=True),
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
            patch.object(pdmux_context, "_GREEN_CONTEXT_STREAMS", set()),
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
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "28")
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
                self.assertEqual(
                    pdmux_context.is_pdmux_green_context_stream(), lane == 0
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
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "104")
        for idx, expected in ((1, None), (2, helper)):
            with patch.object(
                pdmux_context.torch.cuda,
                "current_stream",
                return_value=pdmux_context.get_stream_groups()[idx][1],
            ):
                self.assertIs(
                    pdmux_context.get_pdmux_decode_alt_stream(helper), expected
                )
                self.assertEqual(
                    pdmux_context.is_pdmux_green_context_stream(), idx == 1
                )

    def test_green_detection_without_pdmux_does_not_access_cuda(self):
        with patch.object(pdmux_context.torch.cuda, "current_stream") as current:
            self.assertFalse(pdmux_context.is_pdmux_green_context_stream())
        current.assert_not_called()

    def test_cooperative_routing_fits_every_group(self):
        pdmux_context._configure_flashinfer_moe_sm_reservation(
            148, [(148, 0), (128, 148), (112, 148), (0, 148)]
        )
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "36")

    def test_cooperative_routing_preserves_safer_user_reservation(self):
        os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS] = "40"
        pdmux_context._configure_flashinfer_moe_sm_reservation(148, [(128, 148)])
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "40")

    def test_cooperative_routing_increases_unsafe_user_reservation(self):
        os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS] = "0"
        pdmux_context._configure_flashinfer_moe_sm_reservation(148, [(128, 148)])
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "20")

    def test_cooperative_routing_keeps_default_when_it_fits(self):
        pdmux_context._configure_flashinfer_moe_sm_reservation(148, [(144, 148)])
        self.assertEqual(os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS], "8")

    def test_cooperative_routing_rejects_invalid_reservation(self):
        for value in ("-1", "148", "149", "1.5", "invalid"):
            with self.subTest(value=value):
                os.environ[pdmux_context._FLASHINFER_MOE_RESERVED_SMS] = value
                with self.assertRaisesRegex(ValueError, "FLASHINFER_TRTLLM_MOE"):
                    pdmux_context._configure_flashinfer_moe_sm_reservation(
                        148, [(128, 148)]
                    )

    def test_layer_cap_loads_and_defaults_to_zero(self):
        config_body = """sm_group_num: 3
manual_divisions:
  - [112, 0, 1]
split_forward_token_budget: 65536
overlap_decode_full_sm: true
"""
        self.assertEqual(pdmux_context.PDMuxConfig().max_split_forward_layers, 0)
        self.assertEqual(load_config(config_body).max_split_forward_layers, 0)
        config = load_config(config_body + "max_split_forward_layers: 2\n")
        self.assertEqual(config.max_split_forward_layers, 2)
        self.assertEqual(config.split_forward_token_budget, 65536)

    def test_layer_cap_rejects_invalid_yaml_values(self):
        for value in ("-1", "1.5", "true", "false", '"2"', "null"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    load_config(f"sm_group_num: 3\nmax_split_forward_layers: {value}\n")

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

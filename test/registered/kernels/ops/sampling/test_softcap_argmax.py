"""XPU reduction must preserve transformed ties, NaNs and graph replay."""

import unittest

import torch

from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
from sglang.kernels.ops.sampling import softcap_argmax
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, stage="stage-b", runner_config="1-gpu-xpu")


def reference(raw, cap, sanitize):
    logits = raw.float()
    if cap:
        softcap_inplace_logits(logits, cap)
    if sanitize:
        torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=-1e30)
    return logits.argmax(-1)


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class TestSoftcapArgmax(CustomTestCase):
    def test_transformed_ties_nonfinites_and_chunk_boundaries(self):
        for width in (17, 4095, 4096, 4097, 262144):
            storage = torch.randn(8, width + 16, device="xpu", dtype=torch.bfloat16)
            raw = storage[:, :width]
            raw[0].fill_(-float("inf"))
            raw[0, 0], raw[0, -1] = 1000, 1004
            raw[1, 0] = raw[1, -1] = float("nan")
            raw[2].fill_(-float("inf"))
            raw[3].fill_(-0.0)
            raw[4, 1] = raw[4, -1] = float("inf")
            raw[5].fill_(1.0)
            raw[6, 0], raw[6, -1] = float("nan"), float("inf")
            raw[7].fill_(float("nan"))
            original = storage.clone()
            for cap in (None, 0.0, 30.0):
                for sanitize in (False, True):
                    with self.subTest(width=width, cap=cap, sanitize=sanitize):
                        expected = reference(raw, cap, sanitize)
                        actual = softcap_argmax(raw, softcap=cap, sanitize=sanitize)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                storage, original, atol=0, rtol=0, equal_nan=True
            )

    def test_capture_reads_current_inputs_and_reduces_first_global_tie(self):
        raw = torch.zeros(3, 262144, device="xpu", dtype=torch.bfloat16)
        raw[:, 5000] = 10.0
        expected = reference(raw, 30.0, False)
        softcap_argmax(raw, softcap=30.0)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            out = softcap_argmax(raw, softcap=30.0)
        out.fill_(-1)
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        raw[:, 0] = 10.0
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_lazy_import_keeps_public_function_callable(self):
        # A same-named implementation submodule used to overwrite the package
        # function after first dispatch and crash the next verify iteration.
        raw = torch.tensor([[1.0, 2.0]], device="xpu", dtype=torch.bfloat16)
        softcap_argmax(raw)
        from sglang.kernels.ops.sampling import softcap_argmax as reimported

        actual = reimported(raw)
        torch.testing.assert_close(actual, torch.ones_like(actual), atol=0, rtol=0)

    def test_invalid_layout_and_meta_shape(self):
        for raw in (
            torch.zeros(2, 3, device="xpu"),
            torch.zeros(2, 0, device="xpu", dtype=torch.bfloat16),
        ):
            with self.assertRaises(ValueError):
                softcap_argmax(raw)
        raw = torch.empty(7, 1024, device="meta", dtype=torch.bfloat16)
        output = softcap_argmax(raw)
        self.assertEqual(output.shape, (7,))
        self.assertEqual(output.dtype, torch.int64)
        self.assertEqual(output.device.type, "meta")


if __name__ == "__main__":
    unittest.main()

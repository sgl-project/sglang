"""Exact streaming GDN output/state parity without materialized reversed video."""

import unittest
from unittest.mock import patch

import torch

import sglang.multimodal_gen.runtime.models.dits.sana_wm_components as wm
from sglang.kernels.ops.diffusion import BitExactFusionGate
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSanaWMReverseScan(CustomTestCase):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        original = wm._SANA_WM_GDN_REVERSE
        wm._SANA_WM_GDN_REVERSE = BitExactFusionGate("test", per_signature=True)
        self.addCleanup(setattr, wm, "_SANA_WM_GDN_REVERSE", original)
        torch.manual_seed(42)

    def inputs(self, frames, transposed=False, frame_beta=False):
        b, h, d, s = 1, 20, 112, 880
        shape = (b, h, frames * s, d) if transposed else (b, h, d, frames * s)
        xs = [torch.randn(shape, device="cuda") * 0.03 for _ in range(5)]
        if transposed:
            xs = [x.transpose(-1, -2) for x in xs]
        beta_shape = (b, h, frames) if frame_beta else (b, h, frames, s)
        return (
            *xs,
            torch.rand(beta_shape, device="cuda") * 0.02,
            torch.rand(b, h, frames, device="cuda") * 0.5 + 0.4,
        )

    def assert_nested_equal(self, a, b):
        if isinstance(a, tuple):
            for x, y in zip(a, b, strict=True):
                self.assert_nested_equal(x, y)
        else:
            self.assertTrue(torch.equal(a, b))

    @torch.inference_mode()
    def test_multiple_chunks_outputs_and_carried_states(self):
        for transposed in (False, True):
            for frame_beta in (False, True):
                reference_state = candidate_state = (None, None)
                reference_cam = candidate_cam = None
                for frames in (1, 4, 3, 2):
                    q, k, v, qr, kr, beta, decay = self.inputs(
                        frames, transposed, frame_beta
                    )
                    with patch.object(wm._SANA_WM_GDN_REVERSE, "disabled", True):
                        a, reference_state = wm._gdn_scan_cached(
                            q,
                            k,
                            v,
                            qr,
                            kr,
                            beta,
                            decay,
                            init_state_kv=reference_state[0],
                            init_state_z=reference_state[1],
                        )
                        c, reference_cam = wm._single_path_delta_scan_cached(
                            qr, kr, v, beta, decay, init_state_kv=reference_cam
                        )
                    b, candidate_state = wm._gdn_scan_cached(
                        q,
                        k,
                        v,
                        qr,
                        kr,
                        beta,
                        decay,
                        init_state_kv=candidate_state[0],
                        init_state_z=candidate_state[1],
                    )
                    d, candidate_cam = wm._single_path_delta_scan_cached(
                        qr, kr, v, beta, decay, init_state_kv=candidate_cam
                    )
                    self.assert_nested_equal(
                        (a, reference_state, c, reference_cam),
                        (b, candidate_state, d, candidate_cam),
                    )
                    self.assertFalse(wm._SANA_WM_GDN_REVERSE.disabled)
        self.assertTrue(wm._SANA_WM_GDN_REVERSE.verified)

    @torch.inference_mode()
    def test_changed_graph_inputs_and_initial_states(self):
        q, k, v, qr, kr, beta, decay = self.inputs(3, True)
        state = torch.randn(1, 20, 112, 112, device="cuda") * 0.001
        z = torch.randn(1, 20, 112, 1, device="cuda") * 0.001

        def run():
            return (
                wm._gdn_scan_cached(
                    q, k, v, qr, kr, beta, decay, init_state_kv=state, init_state_z=z
                ),
                wm._single_path_delta_scan_cached(
                    qr, kr, v, beta, decay, init_state_kv=state
                ),
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        q.add_(0.01)
        qr.mul_(0.7)
        beta.mul_(0.8)
        state.add_(0.001)
        z.neg_()
        graph.replay()
        with patch.object(wm._SANA_WM_GDN_REVERSE, "disabled", True):
            expected = run()
        self.assert_nested_equal(actual, expected)

    @torch.inference_mode()
    def test_unverified_capture_and_mismatch_fallback(self):
        _, _, v, qr, kr, beta, decay = self.inputs(2)

        def reference():
            return wm._single_path_delta_scan_backward_reference(qr, kr, v, beta, decay)

        expected = reference()
        with patch.object(wm, "_sana_wm_reverse_scan_impl") as fast:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = wm._sana_wm_reverse_scan(
                    qr, kr, v, beta, decay, reference=reference
                )
            graph.replay()
            fast.assert_not_called()
        self.assertTrue(torch.equal(actual, expected))
        with patch.object(
            wm, "_sana_wm_reverse_scan_impl", return_value=torch.ones_like(expected)
        ):
            actual = wm._sana_wm_reverse_scan(
                qr, kr, v, beta, decay, reference=reference
            )
        self.assertTrue(wm._SANA_WM_GDN_REVERSE.disabled)
        self.assertTrue(torch.equal(actual, expected))

    @torch.inference_mode()
    def test_synthetic_zero_update_keeps_nonfinite_query_behavior(self):
        q, k, v, qr, kr, beta, decay = self.inputs(1)
        qr[..., 0] = float("nan")
        q[..., 1] = float("inf")
        expected = wm._gdn_scan_backward_reference(q, k, v, qr, kr, beta, decay, 1e-6)
        actual = wm._sana_wm_reverse_scan_impl(qr, kr, v, beta, decay, q, k)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


if __name__ == "__main__":
    unittest.main()

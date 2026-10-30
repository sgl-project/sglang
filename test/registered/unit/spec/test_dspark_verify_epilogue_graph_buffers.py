import unittest

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import RefreshMode
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

STRIDE = 4


def _epilogue(max_bs=4) -> DsparkVerifyEpilogue:
    return DsparkVerifyEpilogue(
        max_bs=max_bs,
        verify_num_draft_tokens=STRIDE,
        device="cpu",
        tp_sync=None,
    )


class TestDsparkVerifyEpilogueGraphBuffers(unittest.TestCase):
    def test_buffers_are_registered_with_their_refresh_mode(self):
        epi = _epilogue()
        reg = epi.graph_buffers
        staged = {
            "sim_temperatures_buf",
            "sim_seed_buf",
            "inject_gate_buf",
            "verify_lens_buf",
        }
        for name in reg.buffer_names():
            self.assertIs(getattr(epi, name), reg.get_buffer(name).tensor)
            expected = (
                RefreshMode.BEFORE_REPLAY if name in staged else RefreshMode.IN_GRAPH
            )
            self.assertIs(reg.get_buffer(name).refresh, expected, name)
        self.assertTrue(staged <= set(reg.buffer_names()))
        self.assertIn("draft_tokens_buf", reg.buffer_names())

    def test_static_step_restages_after_a_capture_reseed(self):
        epi = _epilogue()
        epi.begin_static_step(2, armed=True)
        self.assertEqual(epi.verify_lens_buf.tolist(), [STRIDE, STRIDE, 0, 0])
        self.assertEqual(epi.inject_gate_buf.tolist(), [1])

        epi.draft_tokens_buf.fill_(5)
        epi.graph_buffers.prepare_capture()
        self.assertEqual(epi.verify_lens_buf.tolist(), [0, 0, 0, 0])
        self.assertTrue(bool((epi.draft_tokens_buf == 5).all()))
        self.assertEqual(epi.inject_gate_buf.tolist(), [0])

        epi.begin_static_step(2, armed=True)
        self.assertEqual(epi.verify_lens_buf.tolist(), [STRIDE, STRIDE, 0, 0])
        self.assertEqual(epi.inject_gate_buf.tolist(), [1])

    def test_greedy_sim_bonus_clears_sampled_temperatures(self):
        epi = _epilogue()
        epi.begin_sim_bonus(2, None)
        epi.begin_sim_bonus(2, torch.tensor([0.5, 0.75]))
        self.assertEqual(epi.sim_temperatures_buf[:2].tolist(), [0.5, 0.75])
        epi.begin_sim_bonus(2, None)
        self.assertEqual(epi.sim_temperatures_buf.tolist(), [0.0] * 4)

    def test_ragged_step_zeroes_rows_past_the_layout(self):
        epi = _epilogue()
        epi.begin_static_step(4, armed=False)
        epi.begin_step(torch.tensor([3, 1]), armed=True)
        self.assertEqual(epi.verify_lens_buf.tolist(), [3, 1, 0, 0])
        epi.begin_step(None, armed=False)
        self.assertEqual(epi.verify_lens_buf.tolist(), [0, 0, 0, 0])
        self.assertEqual(epi.inject_gate_buf.tolist(), [0])

    def test_rebinding_a_buffer_trips_the_pointer_check(self):
        epi = _epilogue()
        epi.graph_buffers.check_pointers()
        epi.verify_lens_buf = torch.zeros_like(epi.verify_lens_buf)
        with self.assertRaises(RuntimeError):
            epi.graph_buffers.check_pointers()

    def test_strided_outputs_register_once_and_keep_their_shape(self):
        epi = _epilogue(max_bs=2)
        compact = torch.zeros((3, 8), dtype=torch.float32)
        logits = epi._ensure_out("strided_logits", compact)
        self.assertIs(epi.strided_logits, logits)
        self.assertEqual(tuple(logits.shape), (2 * STRIDE, 8))
        self.assertIs(epi._ensure_out("strided_logits", compact), logits)
        with self.assertRaises(RuntimeError):
            epi._ensure_out("strided_logits", compact.to(torch.bfloat16))


if __name__ == "__main__":
    unittest.main()

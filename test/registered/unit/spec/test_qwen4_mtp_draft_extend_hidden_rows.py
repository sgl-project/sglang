import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen4MTPDraftExtendHiddenRows(CustomTestCase):
    """The draft-extend graph path selects the last accepted row per request in
    the logits processor and the worker no longer re-indexes; the HC hidden
    states this model substitutes must be selected the same way, or every draft
    round starts from the wrong row."""

    def _run(self, select_index):
        module = SimpleNamespace(hc_count=2, hidden_size=3)
        hc = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 6)
        out = SimpleNamespace(hidden_states=None)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=SimpleNamespace(select_index=select_index),
            extend_seq_lens=None,
        )
        Qwen4ExpForCausalLMMTP._set_hc_logits_hidden_states(module, out, hc, fb)
        return out.hidden_states, hc

    def test_graph_path_selects_the_processor_rows(self):
        selected, hc = self._run(torch.tensor([3, 7]))
        torch.testing.assert_close(selected, hc[[3, 7]])

    def test_eager_path_keeps_every_row_for_the_worker(self):
        kept, hc = self._run(None)
        self.assertIs(kept, hc)


if __name__ == "__main__":
    unittest.main()

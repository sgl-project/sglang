import unittest

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestEagleDraftExtendLogitsPruning(unittest.TestCase):
    def setUp(self):
        self.hidden_states = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        self.select_index = torch.tensor([2, 4], dtype=torch.int64)

    def _metadata(self, capture_hidden_mode=CaptureHiddenMode.LAST):
        return LogitsMetadata(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            capture_hidden_mode=capture_hidden_mode,
            draft_extend_select_index=self.select_index,
        )

    def _stored_hidden_states(
        self,
        *,
        hidden_states_before_norm=None,
        aux_hidden_states=None,
        capture_hidden_mode=CaptureHiddenMode.LAST,
    ):
        metadata = self._metadata(capture_hidden_mode)
        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            _,
            _,
        ) = LogitsProcessor._get_pruned_states(
            None,
            self.hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            metadata,
        )
        return LogitsProcessor._get_hidden_states_to_store(
            None,
            self.hidden_states,
            hidden_states_before_norm,
            aux_hidden_states,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            metadata,
        )

    def test_last_hidden_states_use_selected_rows(self):
        actual = self._stored_hidden_states()
        torch.testing.assert_close(actual, self.hidden_states[self.select_index])

    def test_last_pre_norm_hidden_states_use_selected_rows(self):
        hidden_states_before_norm = self.hidden_states + 100
        actual = self._stored_hidden_states(
            hidden_states_before_norm=hidden_states_before_norm
        )
        torch.testing.assert_close(actual, hidden_states_before_norm[self.select_index])

    def test_last_aux_hidden_states_use_selected_rows(self):
        aux_hidden_states = [self.hidden_states + 100, self.hidden_states + 200]
        actual = self._stored_hidden_states(aux_hidden_states=aux_hidden_states)
        expected = pack_aux_hidden_states(
            [hidden[self.select_index] for hidden in aux_hidden_states]
        )
        torch.testing.assert_close(actual, expected)

    def test_last_packed_aux_hidden_states_use_selected_rows(self):
        aux_hidden_states = torch.cat(
            [self.hidden_states + 100, self.hidden_states + 200], dim=-1
        )
        actual = self._stored_hidden_states(aux_hidden_states=aux_hidden_states)
        torch.testing.assert_close(actual, aux_hidden_states[self.select_index])

    def test_full_hidden_capture_stays_unpruned(self):
        aux_hidden_states = [self.hidden_states + 100, self.hidden_states + 200]
        actual = self._stored_hidden_states(
            aux_hidden_states=aux_hidden_states,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        torch.testing.assert_close(actual, pack_aux_hidden_states(aux_hidden_states))


if __name__ == "__main__":
    unittest.main()

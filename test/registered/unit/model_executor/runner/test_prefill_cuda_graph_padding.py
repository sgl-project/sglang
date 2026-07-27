import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _pad_draft_extend_spec_info,
    _pad_tokens_to_static,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))


class TestPadTokensToStatic(CustomTestCase):
    """Regression for a crash where EAGLE3/MTP draft models (e.g.
    qwen3_5_mtp.py) hit `assert input_embeds is not None` during prefill
    CUDA-graph replay of a multimodal spec-decoding request. The graph
    runner pads every other prefill tensor (input_ids, positions, ...) to
    the captured bucket size, but `mm_input_embeds` -- the target model's
    precomputed embeddings threaded into the draft model -- was left at its
    raw (unpadded) token count.
    """

    def test_pads_to_static_token_count(self):
        raw = torch.arange(6, dtype=torch.float32).reshape(3, 2)

        padded = _pad_tokens_to_static(raw, static_num_tokens=5)

        self.assertEqual(padded.shape, (5, 2))
        torch.testing.assert_close(padded[:3], raw)
        torch.testing.assert_close(padded[3:], torch.zeros(2, 2))

    def test_none_stays_none(self):
        self.assertIsNone(_pad_tokens_to_static(None, static_num_tokens=5))


class TestPadDraftExtendSpecInfo(CustomTestCase):
    """Regression for a second, distinct shape crash hit right after fixing
    the mm_input_embeds padding above: with mm_input_embeds correctly padded
    to the bucket size, EAGLE3/MTP draft models' `torch.cat([input_embeds,
    hidden_states], dim=-1)` then failed instead ("Expected size 48 but got
    size 39") because `forward_batch.spec_info` -- and therefore
    `spec_info.hidden_states` -- was passed straight through unpadded from
    the raw forward_batch into static_forward_batch, unlike every other
    prefill field. Only EagleDraftExtendInput (the draft-extend spec_info
    type) carries hidden_states that needs this; other spec_info types, or a
    None hidden_states, must pass through unchanged.
    """

    def test_pads_hidden_states_to_static_token_count(self):
        raw_hidden_states = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        spec_info = EagleDraftExtendInput(hidden_states=raw_hidden_states)

        padded_spec_info = _pad_draft_extend_spec_info(spec_info, static_num_tokens=5)

        self.assertEqual(padded_spec_info.hidden_states.shape, (5, 2))
        torch.testing.assert_close(
            padded_spec_info.hidden_states[:3], raw_hidden_states
        )
        torch.testing.assert_close(
            padded_spec_info.hidden_states[3:], torch.zeros(2, 2)
        )
        # The raw spec_info (and its tensor) must not be mutated in place --
        # other code may still read it after load_batch() returns.
        torch.testing.assert_close(spec_info.hidden_states, raw_hidden_states)

    def test_none_spec_info_stays_none(self):
        self.assertIsNone(_pad_draft_extend_spec_info(None, static_num_tokens=5))

    def test_non_draft_extend_spec_info_passes_through(self):
        sentinel = object()

        self.assertIs(
            _pad_draft_extend_spec_info(sentinel, static_num_tokens=5), sentinel
        )


class TestPrefillMambaTrackGate(CustomTestCase):
    """The captured prefill path must track mamba state whenever the eager one
    does. Spec decoding replaces only the DECODE-side track-save (the verify
    commit); the extend-side checkpoint writer has no spec-side substitute, and
    prepare_for_extend populates batch.mamba_track_mask regardless of spec. If
    the captured path drops the track slots, its padded ForwardBatch carries no
    mask, the writer silently no-ops, and later prefix-cache hits restore a
    stale mamba state.
    """

    def _gate(self, *, extra_buffer: bool, disable_radix: bool, spec_none: bool):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.model_runner = SimpleNamespace(
            server_args=SimpleNamespace(
                enable_mamba_extra_buffer=lambda: extra_buffer,
                disable_radix_cache=disable_radix,
            ),
            spec_algorithm=SimpleNamespace(is_none=lambda: spec_none),
        )
        return runner._is_mamba_track_enabled()

    def test_enabled_with_speculative_decoding(self):
        # The regression: EAGLE must not switch extend-side tracking off.
        self.assertTrue(
            self._gate(extra_buffer=True, disable_radix=False, spec_none=False)
        )

    def test_enabled_without_speculative_decoding(self):
        self.assertTrue(
            self._gate(extra_buffer=True, disable_radix=False, spec_none=True)
        )

    def test_disabled_without_extra_buffer(self):
        self.assertFalse(
            self._gate(extra_buffer=False, disable_radix=False, spec_none=False)
        )

    def test_disabled_without_radix_cache(self):
        # Nothing consumes the checkpoints when the radix cache is off.
        self.assertFalse(
            self._gate(extra_buffer=True, disable_radix=True, spec_none=False)
        )


if __name__ == "__main__":
    unittest.main()

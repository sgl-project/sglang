"""Unit tests for Nemotron 3.5 ASR's in-model greedy RNN-T decoding.

The engine drives decoding one emitted token per step, with requests joining,
leaving and being re-prefilled at arbitrary points. These cases pin that the
per-slot state machine still reproduces transformers' greedy RNN-T `generate`
under those schedules.
"""

import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Nemotron3_5AsrConfig, Nemotron3_5AsrForRNNT, ParakeetTokenizer
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderModelOutput,
)

from sglang.srt.models.nemotron3_5_asr import RNNTDecodeState
from sglang.srt.multimodal.processors.nemotron3_5_asr import resolve_prompt_language
from sglang.srt.utils.hf_transformers.tokenizer import _fix_parakeet_group_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_BLANK = 31
_MAX_SYMBOLS = 3


def _tiny_model() -> Nemotron3_5AsrForRNNT:
    config = Nemotron3_5AsrConfig(
        vocab_size=32,
        blank_token_id=_BLANK,
        decoder_hidden_size=16,
        num_decoder_layers=2,
        max_symbols_per_step=_MAX_SYMBOLS,
        num_prompts=8,
        prompt_intermediate_size=16,
        encoder_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_mel_bins": 16,
            "subsampling_conv_channels": 8,
        },
    )
    model = Nemotron3_5AsrForRNNT(config).eval()
    torch.manual_seed(1)
    with torch.no_grad():
        # Default init leaves the joint inputs zero; random weights plus this
        # blank bias give runs of blanks, of tokens, and of forced advances.
        for module in (model.decoder, model.joint):
            for param in module.parameters():
                param.normal_(0, 0.5)
        model.joint.head.bias[_BLANK] = 2.0
    return model


def _transformers_greedy(model, frames: torch.Tensor) -> list[int]:
    """transformers' greedy RNN-T over given projected encoder frames."""
    # The encoder is not under test; feed the decoder loop the frames directly.
    # generate() deletes this instance attribute when it returns.
    model.get_audio_features = lambda **kwargs: NemotronAsrStreamingEncoderModelOutput(
        last_hidden_state=frames[None], pooler_output=frames[None]
    )
    mel_frames = torch.zeros(1, 8 * frames.shape[0], 16)
    with torch.no_grad():
        sequence = model.generate(
            input_features=mel_frames, decoder_start_token_id=_BLANK
        ).sequences[0]
    return sequence.tolist()


class TestRNNTDecodeState(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _tiny_model()
        cls.frames, cls.expected = [], []
        longest_run = 0
        for num_frames in (20, 13, 31):
            torch.manual_seed(num_frames)
            frames = torch.randn(num_frames, 16)
            sequence = _transformers_greedy(cls.model, frames)
            cls.frames.append(frames)
            cls.expected.append([t for t in sequence if t != _BLANK])
            run = 0
            for token_id in sequence[1:]:
                run = run + 1 if token_id != _BLANK else 0
                longest_run = max(longest_run, run)
        # More non-blanks in a row than max_symbols_per_step means the
        # forced frame advance is exercised.
        assert longest_run > _MAX_SYMBOLS, longest_run

    def _state(self) -> RNNTDecodeState:
        return RNNTDecodeState(
            decoder=self.model.decoder,
            joint=self.model.joint,
            blank_token_id=_BLANK,
            max_symbols_per_step=_MAX_SYMBOLS,
        )

    def _run(self, state, clips, *, joined_at=None, resume=None):
        """Drive the engine loop over requests {slot: clip index}.

        A request joins at step joined_at[slot], and one in `resume` is
        re-prefilled with those already-emitted tokens. Returns each slot's
        emitted tokens; the blank that ends a request is excluded.
        """
        joined_at = joined_at or {}
        resume = resume or {}
        out = {slot: list(resume.get(slot, [])) for slot in clips}
        last = {}
        step = 0
        with torch.no_grad():
            while len(last) < len(clips) or any(t != _BLANK for t in last.values()):
                active = [s for s in clips if joined_at.get(s, 0) <= step]
                fresh = [s for s in active if s not in last]
                running = [s for s in active if s in last and last[s] != _BLANK]
                if fresh:
                    state.start(
                        slots=fresh, frames=[self.frames[clips[s]] for s in fresh]
                    )
                for slot in fresh:
                    if slot in resume:
                        state.replay(slot=slot, token_ids=resume[slot])
                if running:
                    state.commit(
                        slots=torch.tensor(running),
                        token_ids=torch.tensor([last[s] for s in running]),
                    )
                stepping = fresh + running
                if stepping:
                    next_ids = state.predict(
                        slots=torch.tensor(stepping), slots_cpu=stepping
                    ).tolist()
                    for slot, token_id in zip(stepping, next_ids):
                        last[slot] = token_id
                        if token_id != _BLANK:
                            out[slot].append(token_id)
                step += 1
                # A request emits at most max_symbols_per_step per frame.
                self.assertLess(step, 2 + _MAX_SYMBOLS * 2 * 31, "decoding never ended")
        return out

    def test_single_request_matches_transformers_greedy(self):
        for clip in range(len(self.frames)):
            with self.subTest(clip=clip):
                out = self._run(self._state(), {0: clip})
                self.assertEqual(out[0], self.expected[clip])

    def test_staggered_batch_matches_isolated_requests(self):
        # Slots are not contiguous and requests join mid-flight, so any
        # cross-row leak in the gathered state shows up as a wrong token.
        clips = {7: 0, 0: 1, 70: 2}
        out = self._run(self._state(), clips, joined_at={0: 3, 70: 5})
        for slot, clip in clips.items():
            self.assertEqual(out[slot], self.expected[clip], f"slot {slot}")

    def test_reprefill_resumes_where_the_request_left_off(self):
        # A retracted request is re-prefilled with its emitted tokens; the
        # replay must leave the decoder exactly where it was.
        for clip in range(len(self.frames)):
            cut = len(self.expected[clip]) // 2
            with self.subTest(clip=clip, cut=cut):
                resume = {0: self.expected[clip][:cut]}
                out = self._run(self._state(), {0: clip}, resume=resume)
                self.assertEqual(out[0], self.expected[clip])

    def test_steps_after_end_of_audio_keep_emitting_blank(self):
        # The overlap scheduler can run one more step after the stop token.
        state = self._state()
        self._run(state, {2: 2})
        slots = torch.tensor([2])
        for _ in range(2):
            state.commit(slots=slots, token_ids=torch.tensor([_BLANK]))
            self.assertEqual(state.predict(slots=slots, slots_cpu=[2]).item(), _BLANK)


class TestPromptLanguage(CustomTestCase):
    _PROMPTS = {"auto": 101, "en": 0, "en-US": 0, "th-TH": 32, "pt-BR": 12, "pt-PT": 13}

    def test_resolution(self):
        cases = [
            (None, "auto"),
            ("en", "en"),
            # Bare codes the checkpoint lacks map to their locale.
            ("th", "th-TH"),
        ]
        for language, expected in cases:
            with self.subTest(language=language):
                self.assertEqual(
                    resolve_prompt_language(language, self._PROMPTS), expected
                )

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_prompt_language("xx", self._PROMPTS)


class TestParakeetTokenizerDecode(CustomTestCase):
    def test_repeated_tokens_survive_decode(self):
        # ParakeetTokenizer merges repeats by default (CTC); generated RNN-T
        # ids are final, so "yeah yeah" must not decode as "yeah".
        vocab = {"<unk>": 0, "yeah": 1, "<pad>": 2}
        backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer = ParakeetTokenizer(
            tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
        )
        _fix_parakeet_group_tokens(tokenizer)
        self.assertEqual(tokenizer.decode([1, 1]).split(), ["yeah", "yeah"])
        self.assertEqual(tokenizer.batch_decode([[1, 1]])[0].split(), ["yeah", "yeah"])


if __name__ == "__main__":
    unittest.main()

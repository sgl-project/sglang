"""CPU coverage for multi-token bans, transport, and request validation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import random
import unittest
from array import array
from types import SimpleNamespace

import msgspec
import torch

from sglang.srt.sampling.custom_logit_processor import (
    BadWordsLogitsProcessor,
    CustomLogitProcessor,
    encode_bad_words,
    prepare_bad_words_request,
    validate_bad_words_request,
)
from sglang.srt.sampling.sampling_params import SamplingParams


def params(words, history=(), prompt=()):
    return {
        "bad_words_token_ids": words,
        "__req__": SimpleNamespace(
            output_ids=array("q", history), origin_input_ids=array("q", prompt)
        ),
    }


class TestBadWordsProcessor(unittest.TestCase):
    def test_suffixes_and_batch_isolation(self):
        entries = [
            params([[1, 2], [3]], [1]),
            params([[1, 2]], [4]),
            params([[0, 1, 2], [1, 3], [1, 3]], [0, 1]),
            params([[1, 2]], [], [1]),
            params([], [1]),
        ]
        before = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        expected = before.clone()
        expected[0, [2, 3]] = -float("inf")
        expected[2, [2, 3]] = -float("inf")
        actual = BadWordsLogitsProcessor()(before.clone(), entries)
        self.assertTrue(torch.equal(actual, expected))
        # Reordering/removing requests must not reuse another request's state.
        order = [4, 2, 0]
        actual = BadWordsLogitsProcessor()(
            before[order].clone(), [entries[i] for i in order]
        )
        self.assertTrue(torch.equal(actual, expected[order]))

    def test_random_reference_equivalence(self):
        rng = random.Random(17)
        for _ in range(300):
            histories = [
                [rng.randrange(16) for _ in range(rng.randrange(15))] for _ in range(4)
            ]
            words = [
                [
                    [rng.randrange(16) for _ in range(rng.randrange(1, 6))]
                    for _ in range(rng.randrange(12))
                ]
                for _ in histories
            ]
            logits = torch.randn(4, 16)
            expected = logits.clone()
            # Reference semantics: ban precisely the last ID of a sequence
            # whose preceding IDs are a suffix of generated history.
            for row, (history, sequences) in enumerate(zip(histories, words)):
                for sequence in sequences:
                    n = len(sequence) - 1
                    if (
                        n <= len(history)
                        and (history[-n:] if n else []) == sequence[:-1]
                    ):
                        expected[row, sequence[-1]] = -float("inf")
            actual = BadWordsLogitsProcessor()(
                logits, [params(w, h) for w, h in zip(words, histories)]
            )
            self.assertTrue(torch.equal(actual, expected))

    def test_greedy_completion_is_blocked(self):
        processor = CustomLogitProcessor.from_str(BadWordsLogitsProcessor.to_str())
        entry = params([[1, 2]])
        logits = torch.tensor([[0.0, 10.0, 9.0, 8.0]])
        first = processor(logits.clone(), [entry]).argmax(-1).item()
        self.assertEqual(first, 1)
        entry["__req__"].output_ids.append(first)
        logits = torch.tensor([[0.0, 1.0, 10.0, 9.0]])
        second = processor(logits, [entry]).argmax(-1).item()
        self.assertEqual(second, 3)

    def test_nested_ids_survive_typed_msgpack(self):
        original = SamplingParams(custom_params={"bad_words_token_ids": [[1, 2], [3]]})
        restored = msgspec.msgpack.decode(
            msgspec.msgpack.encode(original), type=SamplingParams
        )
        self.assertEqual(restored.custom_params, original.custom_params)

    def validate(self, words, **overrides):
        options = dict(
            enabled=True, speculative_algorithm=None, disable_overlap_schedule=True
        )
        options.update(overrides)
        validate_bad_words_request(
            BadWordsLogitsProcessor.to_str(),
            SamplingParams(custom_params={"bad_words_token_ids": words}),
            16,
            **options,
        )

    def test_validation(self):
        self.validate([])
        self.validate([[0], [1, 15]])
        for words in (
            None,
            "foo",
            [1],
            [[]],
            [[True]],
            [[1.2]],
            [[-1]],
            [[16]],
            [[i] for i in range(16)],
        ):
            with self.subTest(words=words), self.assertRaises(ValueError):
                self.validate(words)

    def test_unsupported_modes(self):
        with self.assertRaisesRegex(ValueError, "speculative"):
            self.validate([[1]], speculative_algorithm="EAGLE")
        from unittest.mock import patch

        with patch.dict("os.environ", {"SGLANG_SIMULATE_ACC_LEN": "3"}):
            with self.assertRaisesRegex(ValueError, "simulated"):
                self.validate([[1]], speculative_algorithm="DFLASH")
        self.validate([[1]], disable_overlap_schedule=False)
        self.validate(
            [[1]], speculative_algorithm="DFLASH", disable_overlap_schedule=False
        )

    def test_other_constraints_rejected(self):
        for extra in (
            {"regex": "a"},
            {"logit_bias": {"1": -100}},
            {"min_new_tokens": 1},
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(
                ValueError, "constraints"
            ):
                validate_bad_words_request(
                    BadWordsLogitsProcessor.to_str(),
                    SamplingParams(
                        custom_params={"bad_words_token_ids": [[1]]}, **extra
                    ),
                    16,
                    enabled=True,
                    speculative_algorithm=None,
                    disable_overlap_schedule=True,
                )

    def test_string_encoding_policy(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return {
                    "blue moon": [1, 2],
                    " blue moon": [3, 2],
                    "中文": [4, 5],
                    " 中文": [6, 4, 5],
                    "same": [7],
                    " same": [7],
                    "": [],
                    " ": [],
                }[text]

        tokenizer = Tokenizer()
        self.assertEqual(
            encode_bad_words(["blue moon", "中文", "same", " blue moon"], tokenizer),
            [[1, 2], [3, 2], [4, 5], [7]],
        )
        self.assertEqual(encode_bad_words([], None), [])
        for words in ("blue moon", [""], [1], ["   "]):
            with self.subTest(words=words), self.assertRaises(ValueError):
                encode_bad_words(words, tokenizer)
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            encode_bad_words(["blue moon"], None)

    def test_string_prepare_conflicts(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        sampling = SamplingParams(custom_params={"unrelated": 4})
        serialized = prepare_bad_words_request(
            sampling,
            Tokenizer(),
            None,
            bad_words=["word"],
            enabled=True,
        )
        self.assertIsInstance(
            CustomLogitProcessor.from_str(serialized), BadWordsLogitsProcessor
        )
        self.assertEqual(
            sampling.custom_params, {"unrelated": 4, "bad_words_token_ids": [[1, 2]]}
        )
        with self.assertRaisesRegex(ValueError, "not both"):
            prepare_bad_words_request(
                sampling, Tokenizer(), None, bad_words=["word"], enabled=True
            )
        with self.assertRaisesRegex(ValueError, "custom_logit_processor"):
            prepare_bad_words_request(
                SamplingParams(),
                Tokenizer(),
                serialized,
                bad_words=["word"],
                enabled=True,
            )
        with self.assertRaisesRegex(ValueError, "enable"):
            prepare_bad_words_request(
                SamplingParams(), Tokenizer(), None, bad_words=["word"], enabled=False
            )

    def test_openai_chat_forwards_bad_words(self):

        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            bad_words=["blue moon"],
        )
        self.assertEqual(
            request.to_sampling_params(stop=[], model_generation_config={})[
                "bad_words"
            ],
            ["blue moon"],
        )

    def test_sampler_mixed_batch(self):
        from sglang.srt.layers.sampler import apply_custom_logit_processor

        class Batch(SimpleNamespace):
            def __len__(self):
                return len(self.custom_params)

        batch = Batch(
            custom_params=[params([[1, 2]], [1]), {}, params([[1, 3]], [1])],
            custom_logit_processor={
                0: (BadWordsLogitsProcessor(), torch.tensor([True, False, True]))
            },
        )
        logits = torch.zeros(3, 5)
        apply_custom_logit_processor(logits, batch)
        expected = torch.zeros(3, 5)
        expected[0, 2] = expected[2, 3] = -float("inf")
        self.assertTrue(torch.equal(logits, expected))

    def test_request_preprocessing_and_transport(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.managers import tokenizer_manager as tm
        from sglang.srt.managers.io_struct import (
            GenerateReqInput,
            _msgpack_decoder,
            _msgpack_encoder,
        )

        manager = object.__new__(tm.TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=16)
        request = GenerateReqInput(
            input_ids=[4, 5],
            sampling_params={"custom_params": {"bad_words_token_ids": [[1, 2]]}},
            custom_logit_processor=BadWordsLogitsProcessor.to_str(),
        )
        request.normalize_batch_and_arguments()
        manager.rid_to_state = {request.rid: SimpleNamespace(time_stats=MagicMock())}
        with (
            patch.object(
                tm,
                "get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_custom_logit_processor=True)
                ),
            ),
            patch.object(
                tm, "get_spec", return_value=SimpleNamespace(speculative_algorithm=None)
            ),
            patch.object(
                tm,
                "get_schedule",
                return_value=SimpleNamespace(disable_overlap_schedule=True),
            ),
            patch.object(
                tm,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_transfer_backend="nixl"),
            ),
        ):
            result = manager._create_tokenized_object(request, "", [4, 5])
            result.time_stats = None
            decoded = _msgpack_decoder.decode(_msgpack_encoder.encode(result))
            self.assertEqual(
                decoded.sampling_params.custom_params["bad_words_token_ids"], [[1, 2]]
            )
            request.sampling_params["custom_params"]["bad_words_token_ids"] = [[16]]
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                manager._create_tokenized_object(request, "", [4, 5])


class TestBadWordsDFlash(unittest.TestCase):
    def test_verify_positions_and_no_history_mutation(self):
        from sglang.srt.layers.sampler import apply_custom_logit_processor

        class Info(SimpleNamespace):
            def __len__(self):
                return len(self.custom_params)

        entries = [
            params([[1, 2], [2, 3], [3, 4]], [1]),
            None,
            params([[7, 1, 2], [4]], [7, 1]),
        ]
        info = Info(
            custom_params=entries,
            custom_logit_processor={
                0: (BadWordsLogitsProcessor(), torch.tensor([True, False, True]))
            },
        )
        candidates = torch.tensor([[1, 2, 3], [9, 2, 3], [1, 2, 3]])
        logits = torch.zeros(9, 10)
        apply_custom_logit_processor(logits, info, 3, draft_tokens=candidates)
        expected = torch.zeros_like(logits)
        expected[0, 2] = expected[1, 3] = expected[2, 4] = -float("inf")
        expected[6, 2] = -float("inf")
        expected[6:9, 4] = -float("inf")
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual(list(entries[0]["__req__"].output_ids), [1])
        self.assertEqual(list(entries[2]["__req__"].output_ids), [7, 1])

    def test_random_verify_reference(self):
        from sglang.srt.layers.sampler import apply_custom_logit_processor

        class Info(SimpleNamespace):
            def __len__(self):
                return len(self.custom_params)

        rng = random.Random(73)
        for width in (1, 2, 4, 16):
            for _ in range(30):
                histories = [
                    [rng.randrange(8) for _ in range(rng.randrange(1, 12))]
                    for _ in range(3)
                ]
                words = [
                    [
                        [rng.randrange(8) for _ in range(rng.randrange(1, 6))]
                        for _ in range(15)
                    ]
                    for _ in histories
                ]
                candidates = [
                    [h[-1]] + [rng.randrange(8) for _ in range(width - 1)]
                    for h in histories
                ]
                info = Info(
                    custom_params=[params(w, h) for w, h in zip(words, histories)],
                    custom_logit_processor={
                        0: (BadWordsLogitsProcessor(), torch.ones(3, dtype=torch.bool))
                    },
                )
                actual = torch.randn(3 * width, 8)
                expected = actual.clone()
                for req, h in enumerate(histories):
                    for pos in range(width):
                        effective = h + candidates[req][1 : pos + 1]
                        for word in words[req]:
                            n = len(word) - 1
                            if (
                                n <= len(effective)
                                and (effective[-n:] if n else []) == word[:-1]
                            ):
                                expected[req * width + pos, word[-1]] = -float("inf")
                apply_custom_logit_processor(
                    actual, info, width, draft_tokens=torch.tensor(candidates)
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_rejection_and_full_accept_bonus(self):
        from sglang.srt.speculative.dflash_utils import (
            apply_dflash_verify_logits_adjustments,
            compute_dflash_correct_drafts_and_bonus,
        )

        class Info(SimpleNamespace):
            def __len__(self):
                return len(self.custom_params)

        for banned, expected_accept, expected_bonus in [([2, 3], 1, 0), ([3, 4], 2, 0)]:
            entry = params([banned], [1])
            info = Info(
                custom_params=[entry],
                has_custom_logit_processor=True,
                custom_logit_processor={
                    0: (BadWordsLogitsProcessor(), torch.tensor([True]))
                },
                acc_linear_penalties=None,
                penalizer_orchestrator=None,
                grammar_mask=None,
                logit_bias=None,
            )
            logits = torch.zeros(3, 8)
            logits[0, 2] = logits[1, 3] = logits[2, 4] = 10
            candidates = torch.tensor([[1, 2, 3]])
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits,
                sampling_info=info,
                draft_token_num=3,
                draft_tokens=candidates,
            )
            accepted, bonus = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates, target_predict=logits.argmax(-1).view(1, 3)
            )
            self.assertEqual(accepted.item(), expected_accept)
            self.assertEqual(bonus.item(), expected_bonus)
            self.assertEqual(list(entry["__req__"].output_ids), [1])

    def test_overlap_sync_only_for_bad_words_with_pending_results(self):
        from unittest.mock import patch
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.sampling.custom_logit_processor import has_bad_words_processor

        info = SimpleNamespace(
            custom_logit_processor={
                0: (BadWordsLogitsProcessor(), torch.tensor([True]))
            }
        )
        batch = SimpleNamespace(
            sampling_info=info,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )
        scheduler = SimpleNamespace(require_mlp_sync=False, result_queue=[object()])
        with patch(
            "sglang.srt.managers.scheduler.envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get",
            return_value=False,
        ):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(scheduler, batch, None)
            )
            scheduler.result_queue = []
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(scheduler, batch, None)
            )
            scheduler.result_queue = [object()]
            info.custom_logit_processor = None
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(scheduler, batch, None)
            )
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(scheduler, None, None)
            )
        self.assertFalse(has_bad_words_processor(None))


class TestBadWordsOptimized(unittest.TestCase):
    def test_request_cache_lifetime_and_replacement(self):
        processor = BadWordsLogitsProcessor()
        entry = params([[1, 2]], [1])
        processor(torch.zeros(1, 8), [entry])
        cache = entry["__req__"]._bad_words_prefix_index
        entry["__req__"].output_ids.append(3)
        actual = processor(torch.zeros(1, 8), [entry])
        self.assertFalse(torch.isneginf(actual).any())
        self.assertIs(entry["__req__"]._bad_words_prefix_index, cache)
        entry["bad_words_token_ids"] = [[3, 4]]
        actual = processor(torch.zeros(1, 8), [entry])
        self.assertTrue(torch.isneginf(actual[0, 4]))
        self.assertIsNot(entry["__req__"]._bad_words_prefix_index, cache)

    def test_no_hit_skips_sparse_write(self):
        from unittest.mock import patch

        logits = torch.randn(2, 8)
        expected = logits.clone()
        with patch("torch.tensor", side_effect=AssertionError("unexpected transfer")):
            actual = BadWordsLogitsProcessor()(
                logits, [params([[1, 2]], [3]), params([], [])]
            )
        self.assertIs(actual, logits)
        self.assertTrue(torch.equal(actual, expected))

    def test_sparse_update_preserves_storage_and_other_logits(self):
        logits = torch.randn(2, 8)
        expected = logits.clone()
        expected[0, 2] = expected[0, 4] = expected[1, 7] = -float("inf")
        actual = BadWordsLogitsProcessor()(
            logits,
            [params([[1, 2], [1, 2], [2], [4]], [1]), params([[7]])],
        )
        self.assertIs(actual, logits)
        self.assertTrue(torch.equal(actual, expected))

    def test_overlapping_prefixes_produce_unique_coordinates(self):
        processor = BadWordsLogitsProcessor()
        entry = params([[1, 2], [2], [0, 1, 2]], [0, 1])
        index, _ = processor._compiled(entry)
        rows, tokens = [], []
        processor._collect(index, [0, 1], 3, rows, tokens)
        self.assertEqual(rows, [3])
        self.assertEqual(tokens, [2])

    def test_batch_tokenizer_policy(self):
        class Tokenizer:
            def batch_encode_plus(self, texts, **kwargs):
                self.texts = texts
                return {"input_ids": [[1, 2], [3, 2], [4], [5, 4]]}

        tokenizer = Tokenizer()
        self.assertEqual(
            encode_bad_words([" x", "y"], tokenizer), [[1, 2], [3, 2], [4]]
        )
        self.assertEqual(tokenizer.texts, ["x", " x", "y", " y"])

    def test_tokenizer_cache_is_bounded_isolated_and_returns_copies(self):
        class Tokenizer:
            def __init__(self, token):
                self.token = token
                self.calls = 0

            def encode(self, text, add_special_tokens=False):
                self.calls += 1
                return [self.token, len(text)]

        first, second = Tokenizer(1), Tokenizer(2)
        original = encode_bad_words(["word"], first)
        original[0][0] = 9
        self.assertEqual(encode_bad_words(["word"], first)[0][0], 1)
        self.assertEqual(first.calls, 2)
        self.assertEqual(encode_bad_words(["word"], second)[0][0], 2)
        encode_bad_words([str(i) for i in range(1100)], first)
        self.assertEqual(len(first._sglang_bad_words_cache), 1024)
        self.assertNotIn("word", first._sglang_bad_words_cache)
        encode_bad_words(["x" * 1025], first)
        self.assertNotIn("x" * 1025, first._sglang_bad_words_cache)

    def test_generic_processor_mixed_batch(self):
        from sglang.srt.layers.sampler import apply_custom_logit_processor
        from sglang.srt.sampling.custom_logit_processor import (
            DisallowedTokensLogitsProcessor,
        )

        class Info(SimpleNamespace):
            def __len__(self):
                return len(self.custom_params)

        info = Info(
            custom_params=[{"token_ids": [2]}, None, {"token_ids": [2]}],
            custom_logit_processor={
                0: (
                    DisallowedTokensLogitsProcessor(),
                    torch.tensor([True, False, True]),
                )
            },
        )
        logits = torch.zeros(6, 8)
        apply_custom_logit_processor(logits, info, 2)
        expected = torch.zeros_like(logits)
        expected[[0, 1, 4, 5], 2] = -float("inf")
        self.assertTrue(torch.equal(logits, expected))


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tokenizers import AddedToken, Tokenizer, decoders
from tokenizers.models import BPE, WordLevel
from transformers import PreTrainedTokenizerFast

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MASK_ID = 156895
MASK_TOKEN = "<|mask|>"
CONTROL_TOKEN = "<|control|>"


def _make_contextual_tokenizer(*, mask_is_special=True):
    characters = sorted(set(f"before {MASK_TOKEN} after{CONTROL_TOKEN}"))
    vocabulary = {
        "[UNK]": 0,
        **{character: index + 1 for index, character in enumerate(characters)},
    }
    vocabulary[" <"] = len(vocabulary)
    vocabulary["><"] = len(vocabulary)

    backend = Tokenizer(
        BPE(
            vocab=vocabulary,
            merges=[(" ", "<"), (">", "<")],
            unk_token="[UNK]",
        )
    )
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    )
    tokenizer.add_tokens(
        [
            AddedToken(
                MASK_TOKEN,
                normalized=False,
                special=mask_is_special,
            )
        ],
        special_tokens=mask_is_special,
    )
    tokenizer.add_tokens([AddedToken(CONTROL_TOKEN, normalized=False, special=False)])
    return tokenizer


def _make_unsplittable_tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, MASK_TOKEN: 1}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    )
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


def _make_manager(*, dllm_algorithm="joint_threshold", tokenizer=None):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(dllm_algorithm=dllm_algorithm)
    manager.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LLaDA2MoeModelLM"]),
        is_embedding_gemma=False,
    )
    manager.tokenizer = tokenizer
    manager.async_dynamic_batch_tokenizer = None
    return manager


def _init_manager(manager, mask_id):
    with patch(
        "sglang.srt.managers.tokenizer_manager.get_dllm_model_params",
        return_value={"mask_id": mask_id},
    ):
        manager._init_dllm_prompt_handling()


def _encode_mask_as_text(tokenizer, text):
    literal_tokenizer = copy.deepcopy(tokenizer)
    literal_tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return literal_tokenizer.encode(
        text,
        add_special_tokens=False,
        split_special_tokens=True,
    )


class TestDllmPromptHandling(unittest.TestCase):
    def test_text_prompt_uses_contextual_tokens_for_literal_mask(self):
        tokenizer = _make_contextual_tokenizer()
        mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        manager = _make_manager(tokenizer=tokenizer)
        _init_manager(manager, mask_id)

        prompt = f"before {MASK_TOKEN} after"
        input_ids, token_type_ids = get_or_create_event_loop().run_until_complete(
            manager._tokenize_texts(prompt)
        )
        expected_ids = _encode_mask_as_text(tokenizer, prompt)
        fixed_replacement_count = (
            len(tokenizer.encode(prompt, add_special_tokens=False))
            - 1
            + len(_encode_mask_as_text(tokenizer, MASK_TOKEN))
        )

        self.assertEqual(input_ids, expected_ids)
        self.assertIsNone(token_type_ids)
        self.assertEqual(fixed_replacement_count, len(expected_ids) + 1)

        manager.context_len = fixed_replacement_count
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager._validate_one_request(SimpleNamespace(sampling_params={}), input_ids)

    def test_normalization_preserves_added_tokens_and_adjacent_mask_context(self):
        tokenizer = _make_contextual_tokenizer()
        mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        control_id = tokenizer.convert_tokens_to_ids(CONTROL_TOKEN)
        manager = _make_manager(tokenizer=tokenizer)
        _init_manager(manager, mask_id)

        cases = (
            (
                f"{MASK_TOKEN}{MASK_TOKEN}",
                _encode_mask_as_text(tokenizer, f"{MASK_TOKEN}{MASK_TOKEN}"),
            ),
            (
                f"{CONTROL_TOKEN}before {MASK_TOKEN} after",
                [
                    control_id,
                    *_encode_mask_as_text(tokenizer, f"before {MASK_TOKEN} after"),
                ],
            ),
        )
        for prompt, expected_ids in cases:
            with self.subTest(prompt=prompt):
                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                normalized_ids = manager.normalize_dllm_prompt_token_ids(input_ids)

                self.assertEqual(normalized_ids, expected_ids)
                self.assertNotIn(mask_id, normalized_ids)
                self.assertEqual(
                    tokenizer.decode(
                        normalized_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    prompt,
                )

    def test_non_special_mask_metadata_is_supported_without_mutation(self):
        tokenizer = _make_contextual_tokenizer(mask_is_special=False)
        mask_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
        manager = _make_manager(tokenizer=tokenizer)

        self.assertFalse(tokenizer.added_tokens_decoder[mask_id].special)
        _init_manager(manager, mask_id)
        normalized_ids = manager.normalize_dllm_prompt_token_ids(
            tokenizer.encode(MASK_TOKEN, add_special_tokens=False)
        )

        self.assertNotIn(mask_id, normalized_ids)
        self.assertFalse(tokenizer.added_tokens_decoder[mask_id].special)
        self.assertFalse(tokenizer.split_special_tokens)

    def test_non_dllm_prompt_handling_is_a_noop(self):
        manager = _make_manager(dllm_algorithm=None, tokenizer=object())

        manager._init_dllm_prompt_handling()
        input_ids = [1, MASK_ID, 2]

        self.assertIs(manager.normalize_dllm_prompt_token_ids(input_ids), input_ids)

    def test_direct_input_ids_reject_reserved_mask_without_tokenizer(self):
        manager = _make_manager(tokenizer=None)
        manager._init_dllm_prompt_handling()

        with self.assertRaisesRegex(
            ValueError,
            "Supply ordinary pretokenized IDs.*restart without --skip-tokenizer-init",
        ):
            manager._validate_one_request(None, [1, MASK_ID, 2])

    def test_startup_rejects_unsplittable_mask_token(self):
        tokenizer = _make_unsplittable_tokenizer()
        manager = _make_manager(tokenizer=tokenizer)

        with self.assertRaisesRegex(
            RuntimeError,
            "could not encode the reserved dLLM mask token .* as ordinary text",
        ):
            _init_manager(manager, tokenizer.mask_token_id)


if __name__ == "__main__":
    unittest.main()

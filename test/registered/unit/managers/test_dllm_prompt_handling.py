import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

MASK_ID = 156895
LITERAL_MASK_TOKEN_IDS = [41, 42, 43]


class _TestTokenizer:
    is_fast = True
    eos_token_id = None

    def __init__(self, literal_mask_token_ids=None):
        self.backend_tokenizer = SimpleNamespace(encode_special_tokens=False)
        self.literal_mask_token_ids = (
            LITERAL_MASK_TOKEN_IDS
            if literal_mask_token_ids is None
            else literal_mask_token_ids
        )
        self.encode_calls = []

    def convert_ids_to_tokens(self, token_id):
        if token_id != MASK_ID:
            raise AssertionError(f"unexpected token ID: {token_id}")
        return "<|mask|>"

    def encode(self, text, **kwargs):
        self.encode_calls.append((text, kwargs))
        self.backend_tokenizer.encode_special_tokens = kwargs.get(
            "split_special_tokens", False
        )
        return self.literal_mask_token_ids

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[156899, MASK_ID, 156900] for _ in texts]}


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


class TestDllmPromptHandling(unittest.TestCase):
    def test_text_prompt_expands_literal_mask_without_mutating_tokenizer(self):
        tokenizer = _TestTokenizer()
        manager = _make_manager(tokenizer=tokenizer)

        manager._init_dllm_prompt_handling()
        input_ids, token_type_ids = get_or_create_event_loop().run_until_complete(
            manager._tokenize_texts("before <|mask|> after")
        )

        self.assertEqual(input_ids, [156899, *LITERAL_MASK_TOKEN_IDS, 156900])
        self.assertIsNone(token_type_ids)
        self.assertEqual(
            tokenizer.encode_calls,
            [
                (
                    "<|mask|>",
                    {
                        "add_special_tokens": False,
                        "split_special_tokens": True,
                    },
                )
            ],
        )
        self.assertFalse(tokenizer.backend_tokenizer.encode_special_tokens)

    def test_non_dllm_prompt_handling_is_a_noop(self):
        manager = _make_manager(dllm_algorithm=None, tokenizer=object())

        manager._init_dllm_prompt_handling()
        input_ids = [1, MASK_ID, 2]

        self.assertIs(manager.expand_dllm_literal_mask_tokens(input_ids), input_ids)

    def test_direct_input_ids_reject_reserved_mask_without_tokenizer(self):
        manager = _make_manager(tokenizer=None)
        manager._init_dllm_prompt_handling()

        with self.assertRaisesRegex(
            ValueError,
            "dLLM prompt input_ids must not contain the reserved mask token ID 156895",
        ):
            manager._validate_one_request(None, [1, MASK_ID, 2])

    def test_startup_rejects_unsplittable_mask_token(self):
        manager = _make_manager(tokenizer=_TestTokenizer([MASK_ID]))

        with self.assertRaisesRegex(
            RuntimeError,
            "could not encode the reserved dLLM mask token .* as ordinary text",
        ):
            manager._init_dllm_prompt_handling()


if __name__ == "__main__":
    unittest.main()

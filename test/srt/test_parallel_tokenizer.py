import unittest

from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    try_cached_model,
)


def make_long_mixed_text(min_length: int = 20000) -> str:
    base = (
        "Hello, 世界！This is a long text for ParallelTokenizer; 混合符号: "
        "1234567890 ~!@#$%^&*()_+-=[]{}|;:'\"<>,./?\\ ` “中文引号” ‘单引号’ ，。！？；：\n\t"
    )
    s = []
    while sum(len(x) for x in s) < min_length:
        s.append(base)
    return "".join(s)


MODEL_LIST = [
    DEFAULT_SMALL_CROSS_ENCODER_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
]


class TestParallelTokenizer(unittest.TestCase):

    def test_short_text_consistency(self):
        text = "短文 mixed EN/中文, punctuation: ,.;!?()[]{}\n"

        for model in MODEL_LIST:
            with self.subTest(model=model):
                model_path = try_cached_model(model)
                tok = get_tokenizer(model_path)
                pt = get_tokenizer(model_path, enable_parallel_tokenizer=True)

                # encode path
                for add_special_tokens in [True, False]:
                    with self.subTest(add_special_tokens=add_special_tokens):
                        ref_ids = tok.encode(
                            text, add_special_tokens=add_special_tokens
                        )
                        pt_ids = pt.encode(text, add_special_tokens=add_special_tokens)
                        self.assertEqual(ref_ids, pt_ids)

                # __call__ path (dict compatibility)
                for add_special_tokens in [True, False]:
                    with self.subTest(add_special_tokens=add_special_tokens):
                        ref_dict = tok(text, add_special_tokens=add_special_tokens)
                        pt_dict = pt(text, add_special_tokens=add_special_tokens)
                        self.assertEqual(ref_dict["input_ids"], pt_dict["input_ids"])

    def test_batch_input_passthrough(self):
        texts = [
            "Hello",
            "你好，世界！",
        ]
        for model in MODEL_LIST:
            with self.subTest(model=model):
                model_path = try_cached_model(model)
                tok = get_tokenizer(model_path)
                ptok = get_tokenizer(model_path, enable_parallel_tokenizer=True)
                for add_special_tokens in [True, False]:
                    with self.subTest(add_special_tokens=add_special_tokens):
                        ref = tok(
                            texts,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        pt = ptok(
                            texts,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        self.assertEqual(ref["input_ids"], pt["input_ids"])
                        if "token_type_ids" in ref:
                            self.assertEqual(
                                ref["token_type_ids"], pt.get("token_type_ids")
                            )

    def test_long_text_merge_input_ids_and_token_type_ids(self):
        text = make_long_mixed_text(20000)

        for model in MODEL_LIST:
            with self.subTest(model=model):
                model_path = try_cached_model(model)
                tok = get_tokenizer(model_path)
                ptok = get_tokenizer(model_path, enable_parallel_tokenizer=True)

                for add_special_tokens in [True, False]:
                    with self.subTest(add_special_tokens=add_special_tokens):
                        ref = tok(
                            text,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        pt = ptok(
                            text,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        self.assertEqual(ref["input_ids"], pt["input_ids"])
                        if "token_type_ids" in ref:
                            self.assertEqual(
                                ref["token_type_ids"], pt.get("token_type_ids")
                            )

    def test_long_text_list_merge_input_ids_and_token_type_ids(self):
        texts = [
            make_long_mixed_text(12000),
            make_long_mixed_text(15000),
        ]

        for model in MODEL_LIST:
            with self.subTest(model=model):
                model_path = try_cached_model(model)
                tok = get_tokenizer(model_path)
                ptok = get_tokenizer(model_path, enable_parallel_tokenizer=True)

                for add_special_tokens in [True, False]:
                    with self.subTest(add_special_tokens=add_special_tokens):
                        ref = tok(
                            texts,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        pt = ptok(
                            texts,
                            add_special_tokens=add_special_tokens,
                            return_token_type_ids=True,
                        )
                        self.assertEqual(ref["input_ids"], pt["input_ids"])
                        if "token_type_ids" in ref:
                            self.assertEqual(
                                ref["token_type_ids"], pt.get("token_type_ids")
                            )

    def test_attribute_delegation(self):
        for model in MODEL_LIST:
            with self.subTest(model=model):
                model_path = try_cached_model(model)
                tok = get_tokenizer(model_path)
                pt = get_tokenizer(model_path, enable_parallel_tokenizer=True)
                self.assertEqual(pt.pad_token_id, tok.pad_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)

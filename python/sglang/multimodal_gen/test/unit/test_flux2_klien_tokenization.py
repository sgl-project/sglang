import unittest

from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2KleinPipelineConfig


class _FakeTokenizer:
    def __init__(self):
        self.last_tokenize_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        _ = kwargs
        return f"TEMPLATE::{messages[0]['content']}"

    def __call__(self, texts, **kwargs):
        self.last_tokenize_kwargs = kwargs
        padding = kwargs.get("padding", "max_length")
        truncation = kwargs.get("truncation", True)
        max_length = kwargs.get("max_length", 512)

        tokenized_lengths = [len(text.split()) for text in texts]
        if truncation:
            tokenized_lengths = [min(length, max_length) for length in tokenized_lengths]

        if padding == "max_length":
            padded_length = max_length
        elif padding in (True, "longest"):
            padded_length = max(tokenized_lengths) if tokenized_lengths else 0
        else:
            padded_length = None

        input_ids = []
        attention_mask = []
        for length in tokenized_lengths:
            if padded_length is None:
                row_len = length
            else:
                row_len = padded_length
            input_ids.append(list(range(row_len)))
            attention_mask.append([1] * length + [0] * (row_len - length))

        return {"texts": texts, "input_ids": input_ids, "attention_mask": attention_mask}


class TestFlux2KleinTokenization(unittest.TestCase):
    def test_flux2_klein_config_max_length_is_512(self):
        cfg = Flux2KleinPipelineConfig()
        self.assertEqual(cfg.text_encoder_extra_args[0]["max_length"], 512)

    def test_flux2_klein_tokenize_prompt_uses_512_even_if_kwargs_lower(self):
        cfg = Flux2KleinPipelineConfig()
        tok = _FakeTokenizer()

        out = cfg.tokenize_prompt(
            ["a prompt"],
            tok,
            tok_kwargs={"max_length": 77, "padding": "max_length", "truncation": True},
        )

        self.assertIn("input_ids", out)
        self.assertEqual(tok.last_tokenize_kwargs["max_length"], 512)

    def test_mixed_length_prompt_batch_tokenization_is_padded_and_stable(self):
        cfg = Flux2KleinPipelineConfig()
        tok = _FakeTokenizer()

        out = cfg.tokenize_prompt(
            ["short prompt", " ".join(["long"] * 300)],
            tok,
            tok_kwargs={"padding": "max_length", "truncation": True, "max_length": 77},
        )

        self.assertEqual(len(out["input_ids"]), 2)
        self.assertEqual(len(out["input_ids"][0]), 512)
        self.assertEqual(len(out["input_ids"][1]), 512)
        self.assertEqual(sum(out["attention_mask"][0]), 2)
        self.assertEqual(sum(out["attention_mask"][1]), 301)

    def test_cfg_on_off_mixed_prompt_negative_lengths_keep_compatible_shapes(self):
        cfg = Flux2KleinPipelineConfig()
        tok = _FakeTokenizer()

        pos_out = cfg.tokenize_prompt(
            [" ".join(["positive"] * 180)],
            tok,
            tok_kwargs={"padding": "max_length", "truncation": True, "max_length": 77},
        )
        neg_out = cfg.tokenize_prompt(
            ["negative short"],
            tok,
            tok_kwargs={"padding": "max_length", "truncation": True, "max_length": 77},
        )

        # CFG off: positive branch alone has expected shape.
        self.assertEqual(len(pos_out["input_ids"]), 1)
        self.assertEqual(len(pos_out["input_ids"][0]), 512)

        # CFG on: positive/negative branches can be concatenated on batch dimension.
        self.assertEqual(len(pos_out["input_ids"][0]), len(neg_out["input_ids"][0]))
        cfg_concat = neg_out["input_ids"] + pos_out["input_ids"]
        self.assertEqual(len(cfg_concat), 2)
        self.assertEqual(len(cfg_concat[0]), 512)
        self.assertEqual(len(cfg_concat[1]), 512)

    def test_long_prompt_truncates_at_512_boundary(self):
        cfg = Flux2KleinPipelineConfig()
        tok = _FakeTokenizer()

        very_long_prompt = " ".join(["token"] * 900)
        out = cfg.tokenize_prompt(
            [very_long_prompt],
            tok,
            tok_kwargs={"padding": "max_length", "truncation": True, "max_length": 77},
        )

        self.assertEqual(len(out["input_ids"][0]), 512)
        self.assertEqual(sum(out["attention_mask"][0]), 512)


if __name__ == "__main__":
    unittest.main()



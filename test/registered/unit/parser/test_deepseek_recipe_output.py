import unittest

from sglang.srt.parser import deepseek_recipe_renderer as renderer
from sglang.srt.parser.deepseek_recipe_output import DeepSeekRecipeOutputParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
TEXTS = (
    "Electrons pass through the cytochrome b₆f complex and pump H⁺ ions.",
    "r = 2 × 6.674 × 10⁻¹¹ m³",
    "Nice work \U0001f642\U0001f680 and a naïve café crème brûlée.",
    "∫₀¹ x² dx ≈ ⅓, ∑ₖ k ≤ n²/2 → ∞ and ∀ε ∃δ ∴ ⇒ ⊆ ∪ ∩.",
)


def _tokenizer_file():
    """The V4.1 tokenizer; only outside CI does an unreachable Hub skip the test."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    except Exception as error:  # noqa: BLE001
        if is_in_ci():
            raise
        raise unittest.SkipTest(f"the tokenizer is not available: {error}")


class TestSplitCharacterAtFinish(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tokenizers import Tokenizer

        cls.tokenizer = Tokenizer.from_file(_tokenizer_file())
        cls.deepseek_recipe = renderer.DeepSeekRecipe("dsv41", cls.tokenizer)

    def _parser(self):
        return DeepSeekRecipeOutputParser(
            self.deepseek_recipe, thinking_stage=None, parse_tool_calls=True
        )

    def _split_outputs(self):
        for text in TEXTS:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            for end in range(1, len(ids)):
                decoded = self.tokenizer.decode(ids[:end])
                if decoded.endswith("�"):
                    yield ids[:end], decoded.rstrip("�")

    def test_split_outputs_exist(self):
        self.assertGreaterEqual(len(list(self._split_outputs())), 3)

    def test_whole_output_keeps_the_complete_characters(self):
        for ids, expected in self._split_outputs():
            with self.subTest(ids=ids):
                delta = self._parser().finish(ids, {"type": "length"})
                self.assertEqual(delta.content, expected)

    def test_split_character_fed_before_the_finish(self):
        for ids, expected in self._split_outputs():
            with self.subTest(ids=ids):
                parser = self._parser()
                content = parser.feed(ids).content
                content += parser.finish([], {"type": "length"}).content
                self.assertEqual(content, expected)

    def test_streamed_output_keeps_the_complete_characters(self):
        for ids, expected in self._split_outputs():
            with self.subTest(ids=ids):
                parser = self._parser()
                content = ""
                for start in range(0, len(ids) - 1, 2):
                    content += parser.feed(
                        ids[start : min(start + 2, len(ids) - 1)]
                    ).content
                content += parser.finish(ids[-1:], {"type": "length"}).content
                self.assertEqual(content, expected)

    def test_output_ending_on_a_whole_character_is_unchanged(self):
        for text in TEXTS:
            ids = self.tokenizer.encode(text, add_special_tokens=False).ids
            delta = self._parser().finish(ids, {"type": "length"})
            self.assertEqual(delta.content, text)


if __name__ == "__main__":
    unittest.main()

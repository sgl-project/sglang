import unittest

from sglang.multimodal_gen.runtime.models.registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    def test_mistral3_alias_resolves_to_text_encoder_wrapper(self):
        model_cls, arch = ModelRegistry.resolve_model_cls("Mistral3Model")

        self.assertEqual(model_cls.__name__, "Mistral3ForConditionalGeneration")
        self.assertEqual(arch, "Mistral3ForConditionalGeneration")


if __name__ == "__main__":
    unittest.main()

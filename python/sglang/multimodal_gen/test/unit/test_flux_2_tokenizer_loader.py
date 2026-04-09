import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    TokenizerLoader,
)


class Flux2PipelineConfigStub:
    pass


class OtherPipelineConfigStub:
    pass


class TestFlux2TokenizerLoader(unittest.TestCase):
    def test_flux2_uses_pixtral_processor_path(self):
        loader = TokenizerLoader()
        server_args = SimpleNamespace(pipeline_config=Flux2PipelineConfigStub())

        with patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.Flux2PipelineConfig",
            Flux2PipelineConfigStub,
        ), patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader.AutoProcessor.from_pretrained",
            return_value="flux2-processor",
        ) as mock_from_pretrained:
            tokenizer = loader.load_customized(
                "black-forest-labs/FLUX.2-dev",
                server_args,
                "tokenizer",
            )

        self.assertEqual(tokenizer, "flux2-processor")
        mock_from_pretrained.assert_called_once_with("black-forest-labs/FLUX.2-dev")


if __name__ == "__main__":
    unittest.main()

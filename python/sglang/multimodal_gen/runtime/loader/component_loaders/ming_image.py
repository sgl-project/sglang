# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    checkpoint_weights_iterator,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)


class MingImageEncoderLoader(TextEncoderLoader):
    component_names = []

    def load_customized(self, component_model_path, server_args, component_name):
        root = Path(component_model_path)
        config = server_args.pipeline_config.text_encoder_configs[0]
        mllm = get_diffusers_component_config(component_path=str(root / "mllm"))
        config.arch_config.llm_config = mllm["llm_config"]
        config.arch_config.vision_config = mllm["vision_config"]
        config.arch_config.connector_config = get_diffusers_component_config(
            component_path=str(root / "connector")
        )
        config.arch_config.projection_config = get_diffusers_component_config(
            component_path=str(root / "mlp")
        )
        server_args.model_paths[component_name] = component_model_path
        return self.load_model(
            component_model_path,
            config,
            server_args,
            dtype=server_args.pipeline_config.text_encoder_precisions[0],
            component_name=component_name,
        )

    def _get_all_weights(self, model, model_path, to_cpu):
        for component in ("mllm", "connector", "mlp"):
            for name, weight in checkpoint_weights_iterator(
                str(Path(model_path) / component), to_cpu=to_cpu
            ):
                yield (
                    (f"connector.{name}" if component == "connector" else name),
                    weight,
                )


class MingImageTokenizerLoader(ComponentLoader):
    expected_library = "transformers"

    def load_customized(self, component_model_path, server_args, component_name):
        return PreTrainedTokenizerFast.from_pretrained(
            component_model_path, padding_side="right"
        )

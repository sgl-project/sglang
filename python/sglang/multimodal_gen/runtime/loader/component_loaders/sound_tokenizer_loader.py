# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_avae import (
    Cosmos3AVAEAudioTokenizer,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class SoundTokenizerLoader(ComponentLoader):
    component_names = ["sound_tokenizer"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ) -> bool:
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        try:
            precision = server_args.pipeline_config.vae_precision
        except AttributeError:
            precision = "bf16"
        dtype = PRECISION_TO_TYPE[precision]
        target_device = self.target_device(self.should_offload(server_args))
        model = Cosmos3AVAEAudioTokenizer.from_pretrained(
            component_model_path, dtype=dtype, device=target_device
        )
        server_args.model_paths[component_name] = component_model_path
        return model

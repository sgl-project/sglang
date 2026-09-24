# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage21 import QwenImage21SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21InputValidationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

if current_platform.is_mps():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21_mlx import (
        QwenImage21MLXGenerationStage,
        QwenImage21MLXGenerator,
    )


class QwenImage21MLXPipeline(ComposedPipelineBase):
    pipeline_name = "QwenImage21MLXPipeline"
    pipeline_config_cls = QwenImage21PipelineConfig
    sampling_params_cls = QwenImage21SamplingParams
    _required_config_modules = []

    def build_executor(self, server_args):
        return SyncExecutor(server_args)

    def load_modules(self, server_args, loaded_modules=None):
        if not current_platform.is_mps() or server_args.num_gpus != 1:
            raise ValueError(
                "Qwen-Image 2.1 MLX requires a single Apple Silicon device"
            )
        if (
            server_args.lora_path
            or server_args.cache_dit_config
            or server_args.enable_torch_compile
        ):
            raise ValueError("MLX does not support LoRA, Cache-DiT or torch.compile")
        if server_args.is_arg_explicitly_set("attention_backend"):
            raise ValueError(
                "MLX selects its native attention kernels; omit --attention-backend"
            )
        if server_args.pipeline_config.vae_tiling:
            raise ValueError("Qwen-Image 2.1 MLX does not yet support VAE tiling")
        self.generator = QwenImage21MLXGenerator(
            self.model_path, revision=server_args.revision
        )
        return {}

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage(QwenImage21MLXGenerationStage(self.generator))


EntryClass = QwenImage21MLXPipeline

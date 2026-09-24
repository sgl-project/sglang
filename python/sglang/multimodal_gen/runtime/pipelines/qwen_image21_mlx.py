# SPDX-License-Identifier: Apache-2.0

import numpy as np

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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21InputValidationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import MemorySnapshot

if current_platform.is_mps():
    import mlx.core as mx

    from sglang.multimodal_gen.runtime.hardware_backend.mlx.qwen_image21_pipeline import (
        QwenImage21MLXPipeline as MLXTensorPipeline,
    )

logger = init_logger(__name__)


class QwenImage21MLXGenerationStage(PipelineStage):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, batch, server_args):
        prompts = batch.prompt if isinstance(batch.prompt, list) else [batch.prompt]
        negatives = (
            batch.negative_prompt
            if isinstance(batch.negative_prompt, list)
            else [batch.negative_prompt] * len(prompts)
        )
        images = batch.condition_image
        images = (
            [] if images is None else images if isinstance(images, list) else [images]
        )
        guidance = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, batch.guidance_scale
        )
        outputs = []
        peak_memory = 0

        def progress(index, total, seconds):
            if batch.metrics is not None:
                batch.metrics.record_step(seconds)
            if not batch.is_warmup and not batch.suppress_logs:
                logger.info("MLX denoising step %d/%d: %.3fs", index, total, seconds)

        for prompt_index, (prompt, negative) in enumerate(
            zip(prompts, negatives, strict=True)
        ):
            for output_index in range(batch.num_outputs_per_prompt):
                seed_index = prompt_index * batch.num_outputs_per_prompt + output_index
                image, timings = self.pipeline.generate(
                    prompt=prompt,
                    negative_prompt=negative or "",
                    images=images,
                    width=batch.width,
                    height=batch.height,
                    num_inference_steps=batch.num_inference_steps,
                    seed=batch.seeds[seed_index],
                    guidance_scale=guidance,
                    progress=progress,
                )
                outputs.append(np.array(image))
                peak_memory = max(peak_memory, timings["peak_memory_bytes"])
                logger.debug(
                    "MLX phase timings (including component loading): %s", timings
                )
        if batch.metrics is not None:
            batch.metrics.record_memory_snapshot(
                "mlx",
                MemorySnapshot(
                    allocated_mb=mx.get_active_memory() / 1024**2,
                    reserved_mb=(mx.get_active_memory() + mx.get_cache_memory())
                    / 1024**2,
                    peak_allocated_mb=peak_memory / 1024**2,
                    peak_reserved_mb=0.0,
                ),
            )
        return OutputBatch(
            output=outputs,
            metrics=batch.metrics,
            usage=batch.usage,
            peak_memory_mb=peak_memory / 1024**2,
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
        self.tensor_pipeline = MLXTensorPipeline(
            self.model_path, revision=server_args.revision
        )
        return {}

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage(QwenImage21MLXGenerationStage(self.tensor_pipeline))


EntryClass = QwenImage21MLXPipeline

# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Base class for composed pipelines.

This module defines the base class for pipelines that are composed of multiple stages.
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, cast

import torch
from tqdm import tqdm

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    ImageEncodingStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    LatentPreparationStage,
    PipelineStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ComposedPipelineBase(ABC):
    """
    Base class for pipelines composed of multiple stages.

    This class provides the framework for creating pipelines by composing multiple
    stages together. Each stage is responsible for a specific part of the diffusion
    process, and the pipeline orchestrates the execution of these stages.
    """

    is_video_pipeline: bool = False  # To be overridden by video pipelines
    # should contains only the modules to be loaded
    _required_config_modules: list[str] = []
    _extra_config_module_map: dict[str, str] = {}
    server_args: ServerArgs | None = None
    modules: dict[str, Any] = {}
    executor: PipelineExecutor | None = None

    # the name of the pipeline it associated with, in diffusers
    pipeline_name: str

    def is_lora_effective(self):
        return False

    def is_lora_set(self):
        return False

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ):
        """
        Initialize the pipeline. After __init__, the pipeline should be ready to
        use. The pipeline should be stateless and not hold any batch state.
        """
        self.server_args = server_args

        self.model_path: str = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.executor = executor or self.build_executor(server_args=server_args)

        if required_config_modules is not None:
            self._required_config_modules = required_config_modules

        if self._required_config_modules is None:
            raise NotImplementedError("Subclass must set _required_config_modules")

        # [module_name, gpu memory usage]
        self.memory_usages: dict[str, float] = {}
        # Load modules directly in initialization
        logger.info("Loading pipeline modules...")
        self.modules = self.load_modules(server_args, loaded_modules)

        self.__post_init__()

    def build_executor(self, server_args: ServerArgs):
        # TODO
        from sglang.multimodal_gen.runtime.pipelines_core.executors.parallel_executor import (
            ParallelExecutor,
        )

        # return SyncExecutor(server_args=server_args)
        return ParallelExecutor(server_args=server_args)

    def __post_init__(self) -> None:
        assert self.server_args is not None, "server_args must be set"
        self.initialize_pipeline(self.server_args)

        logger.info("Creating pipeline stages...")
        self.create_pipeline_stages(self.server_args)

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        return self.modules.get(module_name, default_value)

    def add_module(self, module_name: str, module: Any):
        self.modules[module_name] = module

    def _load_config(self) -> dict[str, Any]:
        model_path = maybe_download_model(self.model_path, force_diffusers_model=True)
        self.model_path = model_path
        logger.info("Model path: %s", model_path)
        config = verify_model_config_and_directory(model_path)
        return cast(dict[str, Any], config)

    @property
    def required_config_modules(self) -> list[str]:
        """
        List of modules that are required by the pipeline. The names should match
        the diffusers directory and model_index.json file. These modules will be
        loaded using the PipelineComponentLoader and made available in the
        modules dictionary. Access these modules using the get_module method.

        class ConcretePipeline(ComposedPipelineBase):
            _required_config_modules = ["vae", "text_encoder", "transformer", "scheduler", "tokenizer"]


            @property
            def required_config_modules(self):
                return self._required_config_modules
        """
        return self._required_config_modules

    @property
    def stages(self) -> list[PipelineStage]:
        """
        List of stages in the pipeline.
        """
        return self._stages

    @abstractmethod
    def create_pipeline_stages(self, server_args: ServerArgs):
        """
        Create the inference pipeline stages.
        """
        raise NotImplementedError

    def initialize_pipeline(self, server_args: ServerArgs):
        """
        Initialize the pipeline.
        """
        return

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        override_path = server_args.component_paths.get(module_name)
        if override_path is not None:
            # overridden with args like --vae-path
            component_model_path = maybe_download_model(override_path)
        else:
            component_model_path = os.path.join(self.model_path, load_module_name)

        logger.debug("Resolved component path: %s", component_model_path)
        return component_model_path

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Load the modules from the config.
        loaded_modules: Optional[Dict[str, torch.nn.Module]] = None,
        If provided, loaded_modules will be used instead of loading from config/pretrained weights.
        """

        model_index = self._load_config()
        logger.info("Loading pipeline modules from config: %s", model_index)

        # remove keys that are not pipeline modules
        model_index.pop("_class_name")
        model_index.pop("_diffusers_version")
        if (
            "boundary_ratio" in model_index
            and model_index["boundary_ratio"] is not None
        ):
            has_transformer = (
                "transformer" in model_index
                or "transformer_2" in model_index
                or "transformer" in self.required_config_modules
                or "transformer_2" in self.required_config_modules
            )
            if has_transformer:
                logger.info(
                    "MoE pipeline detected. Adding transformer_2 to self.required_config_modules..."
                )
                if "transformer_2" not in self.required_config_modules:
                    self.required_config_modules.append("transformer_2")
            else:
                logger.info(
                    "Boundary ratio found in model_index.json without transformers; "
                    "using it for pipeline config only."
                )
            logger.info(
                "Setting boundary ratio to %s",
                model_index["boundary_ratio"],
            )
            server_args.pipeline_config.dit_config.boundary_ratio = model_index[
                "boundary_ratio"
            ]

        model_index.pop("boundary_ratio", None)
        # used by Wan2.2 ti2v
        model_index.pop("expand_timesteps", None)

        # some sanity checks
        assert (
            len(model_index) > 1
        ), "model_index.json must contain at least one pipeline module"

        model_index = {
            required_module: model_index[required_module]
            for required_module in self.required_config_modules
        }

        for module_name in self.required_config_modules:
            if (
                module_name not in model_index
                and module_name in self._extra_config_module_map
            ):
                extra_module_value = self._extra_config_module_map[module_name]
                logger.warning(
                    "model_index.json does not contain a %s module, but found {%s: %s} in _extra_config_module_map, adding to model_index.",
                    module_name,
                    module_name,
                    extra_module_value,
                )
                if extra_module_value in model_index:
                    logger.info(
                        "Using module %s for %s", extra_module_value, module_name
                    )
                    model_index[module_name] = model_index[extra_module_value]
                    continue
                else:
                    raise ValueError(
                        f"Required module key: {module_name} value: {model_index.get(module_name)} was not found in loaded modules {model_index.keys()}"
                    )

        # all the component models used by the pipeline
        required_modules = self.required_config_modules
        logger.info("Loading required components: %s", required_modules)

        loaded_components = {}
        for module_name, (
            transformers_or_diffusers,
            architecture,
        ) in tqdm(iterable=model_index.items(), desc="Loading required modules"):
            if transformers_or_diffusers is None:
                logger.warning(
                    "Module %s in model_index.json has null value, removing from required_config_modules",
                    module_name,
                )
                if module_name in self.required_config_modules:
                    self.required_config_modules.remove(module_name)
                continue
            if module_name not in required_modules:
                logger.info("Skipping module %s", module_name)
                continue
            if loaded_modules is not None and module_name in loaded_modules:
                logger.info("Using module %s already provided", module_name)
                loaded_components[module_name] = loaded_modules[module_name]
                continue

            # we load the module from the extra config module map if it exists
            if module_name in self._extra_config_module_map:
                load_module_name = self._extra_config_module_map[module_name]
            else:
                load_module_name = module_name

            component_model_path = self._resolve_component_path(
                server_args, module_name, load_module_name
            )
            module, memory_usage = PipelineComponentLoader.load_component(
                component_name=load_module_name,
                component_model_path=component_model_path,
                transformers_or_diffusers=transformers_or_diffusers,
                server_args=server_args,
            )

            self.memory_usages[load_module_name] = memory_usage

            if module_name in loaded_components:
                logger.warning("Overwriting module %s", module_name)
            loaded_components[module_name] = module

        # Check if all required modules were loaded
        for module_name in required_modules:
            if (
                module_name not in loaded_components
                or loaded_components[module_name] is None
            ):
                raise ValueError(
                    f"Required module: {module_name} was not found in loaded modules: {list(loaded_components.keys())}"
                )

        logger.debug(
            "Memory usage of loaded modules (GiB): %s. Available memory: %s",
            self.memory_usages,
            round(current_platform.get_available_gpu_memory(), 2),
        )

        return loaded_components

    @staticmethod
    def _infer_stage_name(stage: PipelineStage) -> str:
        class_name = stage.__class__.__name__
        # snake_case
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()
        if not name.endswith("_stage"):
            name += "_stage"
        return name

    def add_stage(
        self, stage: PipelineStage, stage_name: str | None = None
    ) -> "ComposedPipelineBase":

        assert self.modules is not None, "No modules are registered"

        if stage_name is None:
            stage_name = self._infer_stage_name(stage)
        if stage_name in self._stage_name_mapping:
            raise ValueError(f"Duplicate stage name detected: {stage_name}")

        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        return self

    def add_stages(
        self, stages: list[PipelineStage | tuple[PipelineStage, str]]
    ) -> "ComposedPipelineBase":

        for item in stages:
            if isinstance(item, tuple):
                stage, name = item
                self.add_stage(stage, name)
            else:
                self.add_stage(item)
        return self

    def add_stage_if(
        self,
        condition: bool | Callable[[], bool],
        stage: PipelineStage,
    ) -> "ComposedPipelineBase":
        should_add = condition() if callable(condition) else condition
        if should_add:
            self.add_stage(stage)
        return self

    def get_stage(self, stage_name: str) -> PipelineStage | None:
        """Get a stage by name."""
        return self._stage_name_mapping.get(stage_name)

    def add_standard_text_encoding_stage(
        self,
        text_encoder_key: str = "text_encoder",
        tokenizer_key: str = "tokenizer",
    ) -> "ComposedPipelineBase":
        return self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module(text_encoder_key)],
                tokenizers=[self.get_module(tokenizer_key)],
            ),
        )

    def add_standard_timestep_preparation_stage(
        self,
        scheduler_key: str = "scheduler",
        prepare_extra_kwargs: list[Callable] | None = [],
    ) -> "ComposedPipelineBase":
        return self.add_stage(
            TimestepPreparationStage(
                scheduler=self.get_module(scheduler_key),
                prepare_extra_set_timesteps_kwargs=prepare_extra_kwargs,
            ),
        )

    def add_standard_latent_preparation_stage(
        self,
        scheduler_key: str = "scheduler",
        transformer_key: str = "transformer",
    ) -> "ComposedPipelineBase":
        return self.add_stage(
            LatentPreparationStage(
                scheduler=self.get_module(scheduler_key),
                transformer=self.get_module(transformer_key),
            ),
        )

    def add_standard_denoising_stage(
        self,
        transformer_key: str = "transformer",
        transformer_2_key: str | None = "transformer_2",
        scheduler_key: str = "scheduler",
        vae_key: str | None = "vae",
    ) -> "ComposedPipelineBase":

        kwargs = {
            "transformer": self.get_module(transformer_key),
            "scheduler": self.get_module(scheduler_key),
        }

        if transformer_2_key:
            transformer_2 = self.get_module(transformer_2_key, None)
            if transformer_2 is not None:
                kwargs["transformer_2"] = transformer_2

        if vae_key:
            vae = self.get_module(vae_key, None)
            if vae is not None:
                kwargs["vae"] = vae
                kwargs["pipeline"] = self

        return self.add_stage(DenoisingStage(**kwargs))

    def add_standard_decoding_stage(
        self,
        vae_key: str = "vae",
    ) -> "ComposedPipelineBase":

        return self.add_stage(
            DecodingStage(vae=self.get_module(vae_key), pipeline=self),
        )

    def add_standard_t2i_stages(
        self,
        include_input_validation: bool = True,
        prepare_extra_timestep_kwargs: list[Callable] | None = [],
    ) -> "ComposedPipelineBase":

        if include_input_validation:
            self.add_stage(InputValidationStage())

        self.add_standard_text_encoding_stage()

        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=prepare_extra_timestep_kwargs
        )
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()

        return self

    def add_standard_ti2i_stages(
        self,
        *,
        include_input_validation: bool = True,
        vae_image_processor: Any | None = None,
        prompt_encoding: Literal["text", "image_encoding"] = "text",
        text_encoder_key: str = "text_encoder",
        tokenizer_key: str = "tokenizer",
        image_processor_key: str = "processor",
        prompt_text_encoder_key: str = "text_encoder",
        image_vae_key: str = "vae",
        image_vae_stage_kwargs: dict[str, Any] | None = None,
        prepare_extra_timestep_kwargs: list[Callable] | None = [],
    ) -> "ComposedPipelineBase":
        if include_input_validation:
            self.add_stage(
                InputValidationStage(vae_image_processor=vae_image_processor)
            )

        if prompt_encoding == "text":
            self.add_standard_text_encoding_stage(
                text_encoder_key=text_encoder_key,
                tokenizer_key=tokenizer_key,
            )
        elif prompt_encoding == "image_encoding":
            self.add_stage(
                ImageEncodingStage(
                    image_processor=self.get_module(image_processor_key),
                    text_encoder=self.get_module(prompt_text_encoder_key),
                ),
            )
        else:
            raise ValueError(f"Unknown prompt_encoding: {prompt_encoding}")

        self.add_stage(
            ImageVAEEncodingStage(
                vae=self.get_module(image_vae_key),
                **(image_vae_stage_kwargs or {}),
            ),
        )

        self.add_standard_latent_preparation_stage()

        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=prepare_extra_timestep_kwargs
        )
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()
        return self

    def add_standard_ti2v_stages(
        self,
        *,
        include_input_validation: bool = True,
        vae_image_processor: Any | None = None,
        text_encoder_key: str = "text_encoder",
        tokenizer_key: str = "tokenizer",
        image_encoder_key: str = "image_encoder",
        image_processor_key: str = "image_processor",
        image_vae_key: str = "vae",
        image_vae_stage_kwargs: dict[str, Any] | None = None,
        image_vae_encoding_position: Literal[
            "before_timestep", "after_latent"
        ] = "before_timestep",
        prepare_extra_timestep_kwargs: list[Callable] | None = [],
        denoising_stage_factory: Callable[[], PipelineStage] | None = None,
    ) -> "ComposedPipelineBase":
        if include_input_validation:
            self.add_stage(
                InputValidationStage(vae_image_processor=vae_image_processor)
            )

        self.add_standard_text_encoding_stage(
            text_encoder_key=text_encoder_key,
            tokenizer_key=tokenizer_key,
        )

        image_encoder = self.get_module(image_encoder_key, None)
        image_processor = self.get_module(image_processor_key, None)
        self.add_stage_if(
            image_encoder is not None and image_processor is not None,
            ImageEncodingStage(
                image_encoder=image_encoder,
                image_processor=image_processor,
            ),
        )

        if image_vae_encoding_position == "before_timestep":
            self.add_stage(
                ImageVAEEncodingStage(
                    vae=self.get_module(image_vae_key),
                    **(image_vae_stage_kwargs or {}),
                )
            )

        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=prepare_extra_timestep_kwargs
        )
        if image_vae_encoding_position == "after_latent":
            self.add_stage(
                ImageVAEEncodingStage(
                    vae=self.get_module(image_vae_key),
                    **(image_vae_stage_kwargs or {}),
                )
            )
        elif image_vae_encoding_position != "before_timestep":
            raise ValueError(
                f"Unknown image_vae_encoding_position: {image_vae_encoding_position}"
            )

        if denoising_stage_factory is None:
            self.add_standard_denoising_stage()
        else:
            self.add_stage(denoising_stage_factory())

        self.add_standard_decoding_stage()
        return self

    # TODO(will): don't hardcode no_grad
    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        """
        Generate a video or image using the pipeline.

        Args:
            batch: The batch to generate from.
            server_args: The inference arguments.
        Returns:
            Req: The batch with the generated video or image.
        """

        if self.is_lora_set() and not self.is_lora_effective():
            logger.warning(
                "LoRA adapter is set, but not effective. Please make sure the LoRA weights are merged"
            )

        # Execute each stage
        if not batch.is_warmup and not batch.suppress_logs:
            logger.info(
                "Running pipeline stages: %s",
                list(self._stage_name_mapping.keys()),
                main_process_only=True,
            )

        return self.executor.execute_with_profiling(self.stages, batch, server_args)

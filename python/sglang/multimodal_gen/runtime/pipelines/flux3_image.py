# SPDX-License-Identifier: Apache-2.0
"""Served FLUX.3 Image Klein package pipeline."""

import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    Flux3ImagePipelineConfig,
    Flux3ImageSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.flux3_image_weights import (
    flux3_image_dit_config_from_package,
    load_flux3_image_dit,
    resolve_flux3_image_package_quant_config,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def require_full_state_dict(model, state_dict, *, name: str) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"{name} checkpoint does not match the model: "
            f"{len(result.missing_keys)} missing key(s) {result.missing_keys[:6]}, "
            f"{len(result.unexpected_keys)} unexpected key(s) "
            f"{result.unexpected_keys[:6]}"
        )


class Flux3ImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Flux3ImagePipeline"
    pipeline_config_cls = Flux3ImagePipelineConfig
    sampling_params_cls = Flux3ImageSamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Flux3ImagePipeline supports same-process execution only; "
                "disaggregated roles are not supported."
            )

    def load_modules(self, server_args: ServerArgs, loaded_modules=None) -> dict:
        package = server_args.pipeline_config.flux3_package
        self._flux3_package_config = package.config
        if loaded_modules is not None:
            return loaded_modules

        from safetensors.torch import load_file

        from sglang.multimodal_gen.configs.models.vaes.flux3_image import (
            Flux3ImageVAEConfig,
        )
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
        from sglang.multimodal_gen.runtime.models.encoders.flux3_text_encoder import (
            Flux3TextEncoder,
        )
        from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
            Flux3ImageVAE,
        )

        if server_args.quantization is not None:
            raise ValueError(
                "FLUX.3 Klein quantization is selected by --model-variant; "
                "do not also set --quantization"
            )

        pipeline_config: Flux3ImagePipelineConfig = server_args.pipeline_config
        device = get_local_torch_device()
        cpu = torch.device("cpu")
        quant_config = resolve_flux3_image_package_quant_config(package)
        enable_prims_ts = server_args.lora_path is None
        if not enable_prims_ts and quant_config is not None:
            logger.info(
                "FLUX.3 PrimsTS is disabled because dynamic LoRA requires "
                "the generic quantized linear layout."
            )
        dit = load_flux3_image_dit(
            str(package.weights),
            flux3_image_dit_config_from_package(package),
            device,
            quant_config=quant_config,
            enable_prims_ts=enable_prims_ts,
        )
        dit.eval()
        if server_args.dit_cpu_offload:
            dit.to(cpu)

        vae_device = cpu if server_args.vae_cpu_offload else device
        vae_dtype = getattr(torch, package.config.vae.dtype)
        vae = Flux3ImageVAE(Flux3ImageVAEConfig()).to(vae_device, vae_dtype)
        require_full_state_dict(
            vae,
            load_file(str(package.vae), device=str(vae_device)),
            name="FLUX.3 image VAE",
        )
        vae.eval()

        text_config = package.config.text_encoder
        text_device = cpu if server_args.text_encoder_cpu_offload else device
        text_encoder = Flux3TextEncoder(
            str(package.text_encoder),
            output_layers=tuple(text_config.layers),
            pad_multiple=text_config.pad_multiple,
            max_length=text_config.max_tokens,
            dtype=getattr(torch, text_config.dtype),
            tokenizer_spec=str(package.tokenizer),
            weight_only_fp8=pipeline_config.flux3_text_encoder_fp8,
            compile_language_model=pipeline_config.flux3_compile_text_encoder,
        ).to(text_device)
        return {
            "transformer": dit,
            "vae": vae,
            "text_encoder": text_encoder,
        }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_image import (
            Flux3ImageConditionEncodeStage,
            Flux3ImageDecodeStage,
            Flux3ImageDenoiseStage,
            Flux3ImageTextEncodeStage,
        )

        package_config = self._flux3_package_config
        self.add_stage(
            Flux3ImageTextEncodeStage(
                text_encoder=self.get_module("text_encoder"),
                package_config=package_config,
            ),
            "flux3_text_encode",
        )
        self.add_stage(
            Flux3ImageConditionEncodeStage(
                vae=self.get_module("vae"),
                package_config=package_config,
            ),
            "flux3_condition_encode",
        )
        self.add_stage(
            Flux3ImageDenoiseStage(
                transformer=self.get_module("transformer"),
                package_config=package_config,
            ),
            "flux3_denoise",
        )
        self.add_stage(
            Flux3ImageDecodeStage(
                vae=self.get_module("vae"),
                package_config=package_config,
            ),
            "flux3_decode",
        )


EntryClass = Flux3ImagePipeline

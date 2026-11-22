# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from time import perf_counter

import torch
from safetensors import safe_open

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingBotVLAV2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_vla_v2 import (
    LingBotVLAV2SamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.models.vlas.lingbot_vla_v2.modeling_lingbot_vla_v2 import (
    LingBotVLAV2ForActionPrediction,
)
from sglang.multimodal_gen.runtime.models.vlas.lingbot_vla_v2.processor_lingbot_vla_v2 import (
    LingBotVLAV2Processor,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionPostprocessStage,
    vla_options,
    vla_state,
    vla_timings,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.vla.cuda_graph import VLATensorGraphRunner


class LingBotVLAV2PolicyStage(PipelineStage):
    def __init__(
        self,
        policy: LingBotVLAV2ForActionPrediction,
        processor: LingBotVLAV2Processor,
        *,
        enable_action_cuda_graph: bool,
    ):
        super().__init__()
        self.policy = policy
        self.processor = processor
        self.denoise_graph = VLATensorGraphRunner(
            "lingbot_vla_v2_denoise", enabled=enable_action_cuda_graph, max_entries=1
        )

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        request = vla_state(batch)
        observation = request["observation"]
        images = observation.get("images") or {}
        state = observation.get("state")
        prompt = observation.get("prompt")
        model_config = self.processor.config
        start = perf_counter()
        inputs = self.processor.build_model_inputs(
            {
                **{
                    key
                    if key.startswith("observation.images.")
                    else f"observation.images.{key}": value
                    for key, value in images.items()
                },
                "state": state,
                "prompt": prompt,
            }
        )
        preprocessed = perf_counter()
        device = inputs["state"].device
        noise = observation.get("noise")
        if noise is not None:
            noise = torch.tensor(noise, dtype=torch.float32, device=device).unsqueeze(0)
            if not torch.isfinite(noise).all():
                raise ValueError("LingBot initial noise must be finite")
        else:
            generator_device = batch.generator_device or device
            generator = torch.Generator(device=generator_device).manual_seed(batch.seed)
            noise = torch.randn(
                (1, model_config.chunk_size, model_config.max_action_dim),
                dtype=torch.float32,
                device=generator_device,
                generator=generator,
            ).to(device)

        graph = None
        if (
            self.denoise_graph.enabled
            and vla_options(batch).get("enable_cuda_graph", True)
            and batch.num_inference_steps == model_config.num_steps
        ):
            graph = self.denoise_graph
        actions = self.policy.sample_actions(
            **inputs,
            num_steps=batch.num_inference_steps,
            noise=noise,
            denoise_graph=graph,
        )
        calibrated = self.processor.build_model_outputs(actions)
        request["actions"] = calibrated[None]
        vla_timings(batch).update(
            preprocess_ms=(preprocessed - start) * 1000,
            policy_ms=(perf_counter() - preprocessed) * 1000,
        )
        return batch


class LingBotVLAV2Pipeline(ComposedPipelineBase):
    pipeline_name = "LingBotVLAV2Pipeline"
    pipeline_config_cls = LingBotVLAV2PipelineConfig
    sampling_params_cls = LingBotVLAV2SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError("LingBot-VLA 2.0 requires monolithic serving")

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        checkpoint_dir = Path(
            maybe_download_model(
                self.model_path,
                revision=server_args.revision,
                allow_patterns=["*.json", "*.safetensors", "*.jinja"],
            )
        )
        config = server_args.pipeline_config.policy
        device = get_local_torch_device()
        if device.type != "cuda":
            raise ValueError(f"LingBot-VLA 2.0 requires CUDA, got {device}")
        self.processor = LingBotVLAV2Processor(config, checkpoint_dir, device)
        if loaded_modules is not None:
            return loaded_modules

        # note (0xtoward): the expert weight cache needs parameter version counters.
        with torch.inference_mode(False):
            policy = LingBotVLAV2ForActionPrediction(config).float()
            with safe_open(
                checkpoint_dir / "model.safetensors", framework="pt", device="cpu"
            ) as weights:
                policy.load_weights(
                    (key, weights.get_tensor(key)) for key in weights.keys()
                )
            policy.to(device).eval().requires_grad_(False)
        return {"policy": policy}

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            LingBotVLAV2PolicyStage(
                self.get_module("policy"),
                self.processor,
                enable_action_cuda_graph=server_args.pipeline_config.enable_action_cuda_graph,
            ),
            "lingbot_policy",
        )
        self.add_stage(VLAActionPostprocessStage(), "lingbot_postprocess")


EntryClass = LingBotVLAV2Pipeline

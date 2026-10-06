# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 video super-resolution (VSR): source video -> upscaled video (+ audio).

Stage chain (all monolithic)::

    input (stream-decode, fps / frame budget, pre-upscale, spatial-factor alignment pad,
           source audio)
      -> encode (LU path: whole-video KVAE encode)
      -> latent_prep (LU or tile encode + upscale -> initial noisy latent chunks)
      -> denoising (the bundle's own scheduler steps every chunk)
      -> decode (KVAE decode every chunk into uint8 tiles)
      -> output (Hann stitch, crop back to the requested size, optional resize, fp16 [0, 1]
                 video + source audio)

Four model phases (encode / upscale-or-encode / denoise / decode), one stage per phase, so the
component-residency manager moves each component on and off the GPU once per request instead of
once per tile; the denoising stage additionally subclasses the shared
``pipelines_core.stages.denoising.DenoisingStage`` instead of a free-standing loop, to reuse its
cache-dit / torch.compile hooks (see ``denoising_stage.py``).

The model is an official Diffusers ``Kandinsky6SRPipeline`` repo, loaded directly like
the K6 TI2VA repos::

    model_index.json              _class_name = Kandinsky6SRPipeline
    transformer/                  Kandinsky6SRTransformer3DModel (text-free SR DiT, ``sr_params``)
    vae/                          Kandinsky6SRVAE (causal video KVAE)
    scheduler/                    FlowMatchEulerDiscreteScheduler or PiflowScheduler
    latent_upscaler/  (optional)  Kandinsky6SRLatentUpscalerBank (x2 / x4 entries)

The ``scheduler`` component selects the sampler, and now actually drives the step loop (via
``scheduler.set_timesteps`` / ``scheduler.step``) instead of a hand-written Euler / pi-Flow
loop re-deriving the same schedule.  There are two official repos:

* ``kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers``: ``FlowMatchEulerDiscreteScheduler``,
  flow-Euler with ``num_inference_steps`` DiT calls per tile;
* ``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers``: ``PiflowScheduler``
  with ``nfe`` 2, pi-Flow with ``nfe`` DiT calls per tile (``num_inference_steps`` is ignored,
  with a warning if it was explicitly set to something other than ``nfe``).

``num_inference_steps`` counts DiT calls per tile here, the convention used everywhere else in
this repo; the upstream Diffusers pipeline counts timestep grid points instead (its default 5
is 4 calls here).
"""

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.decode_stage import (
    Kandinsky6SRDecodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.denoising_stage import (
    Kandinsky6SRDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.encode_stage import (
    Kandinsky6SREncodeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.input_stage import (
    Kandinsky6SRInputStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latent_prep_stage import (
    Kandinsky6SRLatentPrepStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.output_stage import (
    Kandinsky6SROutputStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Kandinsky6SRPipeline(ComposedPipelineBase):
    pipeline_name = "Kandinsky6SRPipeline"
    is_video_pipeline = True
    pipeline_config_cls = Kandinsky6SRPipelineConfig
    sampling_params_cls = Kandinsky6SRSamplingParams

    _required_config_modules = ["transformer", "vae", "scheduler", "latent_upscaler"]
    # Repos without a latent-upscaler bank run the pixel path only.
    _optional_config_modules = ("latent_upscaler",)

    def validate_disagg_role(self, role: RoleType) -> None:
        # The tiled loop keeps tile latents / decoded tiles in one process.
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Kandinsky6SRPipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def load_modules(self, server_args: ServerArgs, loaded_modules=None):
        model_index = self._load_config()
        if "dit" in model_index and "transformer" not in model_index:
            raise ValueError(
                f"{self.model_path} is a k6_video SR bundle (it has a 'dit' component), "
                "not an official Diffusers Kandinsky6SRPipeline repo. Use "
                "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers or "
                "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"
            )
        for name in self._optional_config_modules:
            if model_index.get(name) is None and name in self._required_config_modules:
                self._required_config_modules.remove(name)
                model_index.pop(name, None)
        return super().load_modules(server_args, loaded_modules)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        latent_upscaler = self.get_module("latent_upscaler")
        vae = self.get_module("vae")
        transformer = self.get_module("transformer")
        scheduler = self.get_module("scheduler")
        self.add_stage(stage_name="input_stage", stage=Kandinsky6SRInputStage())
        self.add_stage(
            stage_name="encode_stage",
            stage=Kandinsky6SREncodeStage(vae=vae, latent_upscaler=latent_upscaler),
        )
        self.add_stage(
            stage_name="latent_prep_stage",
            stage=Kandinsky6SRLatentPrepStage(
                vae=vae,
                transformer=transformer,
                latent_upscaler=latent_upscaler,
                scheduler=scheduler,
            ),
        )
        self.add_stage(
            stage_name="denoising_stage",
            stage=Kandinsky6SRDenoisingStage(
                transformer=transformer, scheduler=scheduler, pipeline=self
            ),
        )
        self.add_stage(
            stage_name="decode_stage", stage=Kandinsky6SRDecodeStage(vae=vae)
        )
        self.add_stage(stage_name="output_stage", stage=Kandinsky6SROutputStage())


EntryClass = [Kandinsky6SRPipeline]

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.simple_pipelines.simple_pipeline import (
    SimplePipeline,
)


class QwenImageEditPlusPipeline(SimplePipeline):
    pipeline_name = "QwenImageEditPlusPipeline"

    def __init__(self, model_path: str, server_args: ServerArgs):
        super().__init__(model_path, server_args)
        # self.config is already loaded in the parent class

        self.device = get_local_torch_device()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, subfolder="text_encoder"
        )
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            model_path, subfolder="vae"
        ).to(self.device)
        self.processor = Qwen2VLProcessor.from_pretrained(
            model_path, subfolder="processor"
        )
        # self.transformer = QwenImageTransformer2DModel(self.load_submodules_config(submodule_name="transformer"))
        self.negtive_prompt = " "
        self.true_cfg_scale = 4.0
        self.num_inference_steps = 40
        self.max_sequence_length = 512

    def forward(
        self,
        req,
        server_args,
        # prompt: Union[str, list[str]] = "",
        # height: int | None = None,
        # width: int | None = None,
        # sigmas: Optional[list[float]] = None,
        # guidance_scale: float = 1.0,
        # num_images_per_prompt: int = 1,
        # generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        # latents: Optional[torch.Tensor] = None,
        # prompt_embeds: Optional[torch.Tensor] = None,
        # prompt_embeds_mask: Optional[torch.Tensor] = None,
        # negative_prompt_embeds: Optional[torch.Tensor] = None,
        # negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        # output_type: Optional[str] = "pil",
        # attention_kwargs: Optional[dict[str, Any]] = None,
        # callback_on_step_end_tensor_inputs: list[str] = ["latents"],
    ):
        trajectory_decoded = None
        import torch

        height, width = 256, 256
        y = torch.linspace(0, 1, height).unsqueeze(1).expand(height, width)
        x = torch.linspace(0, 1, width).unsqueeze(0).expand(height, width)
        r = x
        g = y
        b = 1 - x

        # 拼成 (c, h, w)
        frames = [torch.stack([r, g, b], dim=0)]
        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=req.trajectory_timesteps,
            trajectory_latents=req.trajectory_latents,
            trajectory_decoded=trajectory_decoded,
            timings=req.timings,
        )
        return output_batch

from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLProcessor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class QwenImageEditPlusPipeline(nn.Module):
    pipeline_name = "QwenImageEditPlusPipeline"

    def __init__(self, model_path: str, server_args: ServerArgs):
        super().__init__()
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
        config = {}
        hf_config = {}
        self.transformer = QwenImageTransformer2DModel(config, hf_config)
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
        pass

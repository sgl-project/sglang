import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class NemotronH_Nano_VL_V2(nn.Module):
    def __init__(
        self,
        config: NemotronH_Nano_VL_V2_Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config

        # Language model (NemotronH)
        self.language_model = NemotronHForCausalLM(
            config=config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        # Vision tower (HF Radio-based) - use HF impl for now
        self.vision_model = AutoModel.from_config(
            config.vision_config, trust_remote_code=True
        )

        # Ensure external preprocessing and dtype alignment
        if hasattr(self.vision_model, "model") and hasattr(
            self.vision_model.model, "_init_weights"
        ):
            # Align init method name if needed (HF naming differences)
            self.vision_model.model._initialize_weights = (
                self.vision_model.model._init_weights
            )
        if hasattr(self.vision_model, "radio_model"):
            try:
                self.vision_model.radio_model.make_preprocessor_external()
            except Exception as e:
                logger.warning(f"Failed to externalize radio preprocessor: {e}")
        self.vision_model = self.vision_model.to(
            dtype=self.language_model.config.torch_dtype
        )

        # Vision projection (RMSNorm -> Linear -> activation -> Linear)
        vit_hidden_size = self.config.vit_hidden_size
        downsample_ratio = self.config.downsample_ratio
        proj_hidden = self.config.projector_hidden_size
        llm_hidden = self.config.llm_config.hidden_size
        in_feat = vit_hidden_size * int(1 / downsample_ratio) ** 2

        self.vision_projector = nn.Sequential(
            RMSNorm(in_feat, eps=1e-5),
            nn.Linear(in_feat, proj_hidden, bias=False),
            nn.GELU(),
            nn.Linear(proj_hidden, llm_hidden, bias=False),
        ).to(self.language_model.config.torch_dtype)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        x = x.view(
            n,
            w,
            int(h * scale_factor),
            int(c / scale_factor),
        )
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if getattr(self.config, "ps_version", "v2") != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vit_embeds = self.vision_model(pixel_values).features
        # Expect shape: [N, T, C] -> reshape to HxW grid
        vit_embeds = vit_embeds.to(dtype=self.language_model.config.torch_dtype)
        hw_tokens = vit_embeds.shape[1]
        h = w = int(hw_tokens**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(
            vit_embeds, scale_factor=self.config.downsample_ratio
        )
        vit_embeds = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        vit_embeds = self.vision_projector(vit_embeds)
        return vit_embeds

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        image_features = self.extract_feature(pixel_values)
        return image_features

    def get_video_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the video model into language model space.

        Returns:
            video_features (`torch.Tensor`): Video feature tensor of shape `(num_videos, video_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        video_features = self.extract_feature(pixel_values)
        return video_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
            },
            positions=positions,
        )

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for language model, vision model, and projector.

        Expected prefixes in checkpoint names:
        - "language_model." for NemotronH LLM weights
        - "vision_model." for HF Radio vision tower (buffers/params)
        - "mlp1." for projector weights (mapped to our "vision_projector")
        """

        def is_vision_model_weights(weight: Tuple[str, torch.Tensor]) -> bool:
            return weight[0].startswith("vision_model")

        def is_adapter_weights(weight: Tuple[str, torch.Tensor]) -> bool:
            # Check legacy vLLM projector prefix
            return weight[0].startswith("mlp1")

        # Prepare lookup dicts
        vision_params = dict(self.vision_model.named_parameters())
        vision_buffers = dict(self.vision_model.named_buffers())
        projector_params = dict(self.vision_projector.named_parameters())

        def llm_weights_generator():
            for name, w in weights:
                if is_vision_model_weights((name, w)):
                    # Load vision encoder weights/buffers directly
                    trimmed = ".".join(name.split(".")[1:])  # drop "vision_model."
                    if "input_conditioner" in trimmed:
                        continue
                    if trimmed in vision_buffers:
                        buf = vision_buffers[trimmed]
                        with torch.no_grad():
                            buf.copy_(w.to(dtype=buf.dtype, device=buf.device))
                    elif trimmed in vision_params:
                        param = vision_params[trimmed]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    else:
                        logger.warning(f"Vision weight not found: {trimmed}")
                elif is_adapter_weights((name, w)):
                    # Map "mlp1.*" to our projector submodule names (which are relative)
                    trimmed = ".".join(name.split(".")[1:])  # e.g., "0.weight"
                    if trimmed in projector_params:
                        param = projector_params[trimmed]
                        with torch.no_grad():
                            default_weight_loader(param, w)
                    else:
                        logger.warning(f"Projector weight not found: {trimmed}")
                else:
                    # LLM weights -> yield without the top-level prefix
                    assert name.startswith("language_model"), name
                    trimmed = ".".join(name.split(".")[1:])
                    yield (trimmed, w)

        # Delegate LLM weight loading to the language model
        self.language_model.load_weights(llm_weights_generator())


EntryClass = [NemotronH_Nano_VL_V2]

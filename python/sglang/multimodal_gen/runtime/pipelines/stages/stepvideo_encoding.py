# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# The dedicated stepvideo prompt encoding stage.
class StepvideoPromptEncodingStage(PipelineStage):
    """
    Stage for encoding prompts using the remote caption API.

    This stage applies the magic string transformations and calls
    the remote caption service asynchronously to get:
      - primary prompt embeddings,
      - an attention mask,
      - and a clip embedding.
    """

    def __init__(self, stepllm, clip) -> None:
        super().__init__()
        # self.caption_client = caption_client  # This should have a call_caption(prompts: List[str]) method.
        self.stepllm = stepllm
        self.clip = clip

    @torch.no_grad()
    def forward(self, batch: Req, server_args) -> Req:

        prompts = [batch.prompt + server_args.pipeline_config.pos_magic]
        bs = len(prompts)
        prompts += [server_args.pipeline_config.neg_magic] * bs
        with set_forward_context(current_timestep=0, attn_metadata=None):
            y, y_mask = self.stepllm(prompts)
            clip_emb, _ = self.clip(prompts)
            len_clip = clip_emb.shape[1]
            y_mask = torch.nn.functional.pad(y_mask, (len_clip, 0), value=1)
        pos_clip, neg_clip = clip_emb[:bs], clip_emb[bs:]

        # split positive vs negative text
        batch.prompt_embeds = y[:bs]  # [bs, seq_len, dim]
        batch.negative_prompt_embeds = y[bs : 2 * bs]  # [bs, seq_len, dim]
        batch.prompt_attention_mask = y_mask[:bs]  # [bs, seq_len]
        batch.negative_attention_mask = y_mask[bs : 2 * bs]  # [bs, seq_len]
        batch.clip_embedding_pos = pos_clip
        batch.clip_embedding_neg = neg_clip
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify stepvideo encoding stage inputs."""
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_not_empty)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify stepvideo encoding stage outputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, [V.is_tensor, V.with_dims(3)]
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            [V.is_tensor, V.with_dims(3)],
        )
        result.add_check(
            "prompt_attention_mask",
            batch.prompt_attention_mask,
            [V.is_tensor, V.with_dims(2)],
        )
        result.add_check(
            "negative_attention_mask",
            batch.negative_attention_mask,
            [V.is_tensor, V.with_dims(2)],
        )
        result.add_check(
            "clip_embedding_pos",
            batch.clip_embedding_pos,
            [V.is_tensor, V.with_dims(2)],
        )
        result.add_check(
            "clip_embedding_neg",
            batch.clip_embedding_neg,
            [V.is_tensor, V.with_dims(2)],
        )
        return result

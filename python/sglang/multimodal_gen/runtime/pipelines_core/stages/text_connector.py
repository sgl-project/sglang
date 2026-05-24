import torch

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LTX2TextConnectorStage(PipelineStage):
    """
    Stage for applying LTX-2 Text Connectors to split/transform text embeddings
    into video and audio contexts.
    """

    def __init__(self, connectors):
        super().__init__()
        self.connectors = connectors

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Input: batch.prompt_embeds (from Gemma, [B, S, D])
        # Output: batch.prompt_embeds (Video Context), batch.audio_prompt_embeds (Audio Context)

        prompt_embeds = batch.prompt_embeds
        prompt_attention_mask = batch.prompt_attention_mask
        neg_prompt_embeds = batch.negative_prompt_embeds
        neg_prompt_attention_mask = batch.negative_attention_mask

        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0] if len(prompt_embeds) > 0 else None

        if isinstance(prompt_attention_mask, list):
            prompt_attention_mask = (
                prompt_attention_mask[0] if len(prompt_attention_mask) > 0 else None
            )

        if isinstance(neg_prompt_embeds, list):
            neg_prompt_embeds = (
                neg_prompt_embeds[0] if len(neg_prompt_embeds) > 0 else None
            )

        if isinstance(neg_prompt_attention_mask, list):
            neg_prompt_attention_mask = (
                neg_prompt_attention_mask[0]
                if len(neg_prompt_attention_mask) > 0
                else None
            )

        if prompt_embeds is None or prompt_attention_mask is None:
            raise ValueError(
                "LTX2TextConnectorStage requires prompt embeddings and "
                "attention mask."
            )

        if batch.do_classifier_free_guidance:
            if neg_prompt_embeds is None or neg_prompt_attention_mask is None:
                raise ValueError(
                    "LTX2TextConnectorStage requires negative prompt embeddings "
                    "and attention mask when classifier-free guidance is enabled."
                )

            # Official LTX-2.3 processes positive and negative prompts through
            # the connector independently; batching shifts output numerics.
            dtype = prompt_embeds.dtype
            pos_additive_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max
            neg_additive_mask = (neg_prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                pos_embeds, pos_audio_embeds, pos_mask = self.connectors(
                    prompt_embeds, pos_additive_mask, additive_mask=True
                )
                neg_embeds, neg_audio_embeds, neg_mask = self.connectors(
                    neg_prompt_embeds, neg_additive_mask, additive_mask=True
                )

            batch.prompt_embeds = [pos_embeds]
            batch.audio_prompt_embeds = [pos_audio_embeds]
            batch.prompt_attention_mask = pos_mask
            batch.negative_prompt_embeds = [neg_embeds]
            batch.negative_audio_prompt_embeds = [neg_audio_embeds]
            batch.negative_attention_mask = neg_mask
        else:
            # Prepare additive mask for connectors (as per diffusers implementation)
            dtype = prompt_embeds.dtype
            additive_attention_mask = (prompt_attention_mask.to(torch.int64) - 1).to(
                dtype
            ) * torch.finfo(dtype).max

            with set_forward_context(current_timestep=None, attn_metadata=None):
                (
                    connector_prompt_embeds,
                    connector_audio_prompt_embeds,
                    connector_mask,
                ) = self.connectors(
                    prompt_embeds, additive_attention_mask, additive_mask=True
                )

            batch.prompt_embeds = [connector_prompt_embeds]
            batch.audio_prompt_embeds = [connector_audio_prompt_embeds]
            batch.prompt_attention_mask = connector_mask

        return batch

# SPDX-License-Identifier: Apache-2.0
"""
Prompt enhancement stage for ErnieImage pipeline.
"""

import json

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PromptEnhancementStage(PipelineStage):

    def __init__(self, pe_model, pe_tokenizer):
        super().__init__()
        self.pe_model = pe_model
        self.pe_tokenizer = pe_tokenizer

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # Skip if use_pe is disabled or tokenizer unavailable
        use_pe = getattr(batch, "use_pe", True)
        if not use_pe or self.pe_model is None:
            return batch

        if self.pe_tokenizer is None:
            logger.warning(
                "pe_tokenizer is None, skipping prompt enhancement. "
                "Check PE model loading logs for errors."
            )
            return batch

        prompt = batch.prompt
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        height = getattr(batch, "height", 1024)
        width = getattr(batch, "width", 1024)

        enhanced = []
        for p in prompts:
            enhanced_p = self._enhance_single_prompt(p, width, height)
            enhanced.append(enhanced_p)

        if isinstance(batch.prompt, str):
            batch.prompt = enhanced[0]
        else:
            batch.prompt = enhanced

        logger.info("PE enhanced prompt: %s", batch.prompt)
        return batch

    def _enhance_single_prompt(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        max_new_tokens: int = 1536,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> str:
        user_content = json.dumps(
            {"prompt": prompt, "width": width, "height": height},
            ensure_ascii=False,
        )
        messages = [{"role": "user", "content": user_content}]

        input_text = self.pe_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Use srt Engine for optimized generation
        output = self.pe_model.generate(
            prompt=input_text,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        )

        return output["text"].strip()

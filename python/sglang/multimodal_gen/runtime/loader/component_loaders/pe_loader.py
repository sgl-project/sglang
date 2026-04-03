# SPDX-License-Identifier: Apache-2.0
"""Loader for prompt enhancement (PE) causal language model.

Loads a Ministral-3B causal LM via ``AutoModelForCausalLM`` with
**Flash Attention 2** (fused attention kernel, ~2x faster than SDPA).

Note: ``srt.Engine`` cannot be used here because the multimodal_gen
pipeline runs inside a daemon subprocess, and daemon processes are not
allowed to spawn children (which Engine requires).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PEModelWrapper:
    """Wraps an ``AutoModelForCausalLM`` + tokenizer for use by
    :class:`PromptEnhancementStage`.

    The wrapper provides the same ``.generate()`` / ``.pe_tokenizer``
    interface that ``PromptEnhancementStage`` expects.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.pe_tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, sampling_params: dict) -> dict:
        """Generate text using the HuggingFace model.

        Args:
            prompt: Already-formatted input text (chat template applied).
            sampling_params: Dict with ``max_new_tokens``, ``temperature``,
                ``top_p``, etc.

        Returns:
            Dict with a ``"text"`` key containing the generated text.
        """
        inputs = self.pe_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=sampling_params.get("max_new_tokens", 1536),
            temperature=sampling_params.get("temperature", 0.6),
            top_p=sampling_params.get("top_p", 0.95),
            do_sample=True,
        )

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0, input_len:]
        text = self.pe_tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"text": text}

    def to(self, *args, **kwargs):
        """Move underlying model to device."""
        self.model = self.model.to(*args, **kwargs)
        if args:
            device = args[0]
            if isinstance(device, (str, torch.device)):
                self.device = torch.device(device)
        return self


class PELoader(ComponentLoader):
    """Loader for prompt-enhancement causal LM (Ministral-3 based).

    Loads via ``AutoModelForCausalLM`` in bf16 on GPU with
    Flash Attention 2 (or SDPA fallback).
    """

    component_names = ["pe"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        logger.info("Loading PE model from %s ...", component_model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            component_model_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        # Pad token required for batched generation; use eos if unset.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Flash Attention 2 (fall back to SDPA)
        attn_impl = "flash_attention_2"
        try:
            model = AutoModelForCausalLM.from_pretrained(
                component_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=server_args.trust_remote_code,
                attn_implementation=attn_impl,
            )
            logger.info("PE model: using Flash Attention 2")
        except (ValueError, ImportError):
            logger.warning(
                "Flash Attention 2 not available, falling back to SDPA"
            )
            attn_impl = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                component_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=server_args.trust_remote_code,
                attn_implementation=attn_impl,
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()

        logger.info(
            "PE model loaded on %s: %s (attn=%s)",
            device,
            model.__class__.__name__,
            attn_impl,
        )

        return PEModelWrapper(model=model, tokenizer=tokenizer, device=device)

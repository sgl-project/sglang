# SPDX-License-Identifier: Apache-2.0
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _read_model_max_length(model_path: str) -> int | None:
    """Read model_max_length from tokenizer_config.json in the given directory."""
    config_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            val = config.get("model_max_length")
            if val is not None:
                return int(val)
        except Exception as e:
            logger.warning(
                "Failed to read tokenizer_config.json from %s: %s", model_path, e
            )
    return None


class PEModelWrapper:

    def __init__(self, model, tokenizer, device, model_max_length: int):
        self.model = model
        self.pe_tokenizer = tokenizer
        self.device = device
        self.model_max_length = model_max_length

    def generate(self, prompt: str, sampling_params: dict) -> dict:
        inputs = self.pe_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model_max_length,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=sampling_params.get("max_new_tokens", self.model_max_length),
            do_sample=True,
        )
        temperature = sampling_params.get("temperature")
        top_p = sampling_params.get("top_p")
        if temperature is not None:
            generate_kwargs["temperature"] = temperature
        if top_p is not None:
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            output_ids = self.model.generate(**generate_kwargs)

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
    """Loader for prompt-enhancement causal LM (Ministral-3 based)."""

    component_names = ["pe"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        logger.info("Loading PE model from %s ...", component_model_path)

        pe_tokenizer_dir = os.path.join(
            os.path.dirname(component_model_path), "pe_tokenizer"
        )
        if not os.path.exists(
            os.path.join(component_model_path, "tokenizer_config.json")
        ) and os.path.exists(os.path.join(pe_tokenizer_dir, "tokenizer_config.json")):
            tokenizer_path = pe_tokenizer_dir
            logger.info(
                "PE tokenizer files not found in %s, using %s",
                component_model_path,
                tokenizer_path,
            )
        else:
            tokenizer_path = component_model_path

        model_max_length = _read_model_max_length(tokenizer_path)
        if model_max_length is None:
            raise RuntimeError(
                f"Cannot load PE model: 'model_max_length' not found in "
                f"{os.path.join(tokenizer_path, 'tokenizer_config.json')}. "
                "Please ensure the PE component directory (or its sibling "
                "pe_tokenizer/ directory) contains a valid tokenizer_config.json "
                "with a 'model_max_length' field."
            )
        logger.info(
            "PE model_max_length=%d (from tokenizer_config.json)", model_max_length
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=server_args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

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
            logger.warning("Flash Attention 2 not available, falling back to SDPA")
            attn_impl = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                component_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=server_args.trust_remote_code,
                attn_implementation=attn_impl,
            )

        device = get_local_torch_device()
        model = model.to(device).eval()

        logger.info(
            "PE model loaded on %s: %s (attn=%s)",
            device,
            model.__class__.__name__,
            attn_impl,
        )

        return PEModelWrapper(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_max_length=model_max_length,
        )

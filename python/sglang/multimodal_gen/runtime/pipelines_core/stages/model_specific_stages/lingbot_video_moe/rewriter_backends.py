# SPDX-License-Identifier: Apache-2.0

import base64
import io

import PIL.Image
import requests

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class HTTPRewriterBackend:
    """Call an OpenAI-compatible endpoint, selecting a model per turn."""

    def __init__(
        self,
        url: str,
        map_url: str,
        expand_model: str,
        map_model: str,
        timeout: float,
    ):
        self.url = url.rstrip("/")
        self.map_url = map_url.rstrip("/")
        self.expand_model = expand_model
        self.map_model = map_model
        self.timeout = timeout

    def generate(self, text: str, image: PIL.Image.Image | None, use_lora: bool) -> str:
        content: list[dict] = [{"type": "text", "text": text}]
        if image is not None:
            content.insert(
                0, {"type": "image_url", "image_url": {"url": _data_url(image)}}
            )
        response = requests.post(
            f"{self.map_url if use_lora else self.url}/v1/chat/completions",
            json={
                "model": self.map_model if use_lora else self.expand_model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                # The mapping turn must answer with JSON and nothing else.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class TransformersRewriterBackend:
    """Base VLM plus the rewriter adapter, loaded in this process on first use.

    The adapter ships a regex ``target_modules`` over the unfused Qwen3.5
    projections, which peft applies directly and the SGLang LoRA path does not
    accept, so serving it needs this loader or a pre-merged checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str,
        device_map: str,
        max_new_tokens: int,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.processor = None
        self.model = None

    def _load(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(
            "Loading the prompt rewriter from %s with adapter %s ...",
            self.model_path,
            self.adapter_path,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        base = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, self.adapter_path).eval()

    def generate(self, text: str, image: PIL.Image.Image | None, use_lora: bool) -> str:
        import contextlib

        import torch

        if self.model is None:
            self._load()
        content = [{"type": "image", "image": image}] if image is not None else []
        content.append({"type": "text", "text": text})
        chat = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(
            text=[chat],
            images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(self.model.device)
        adapter = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with torch.no_grad(), adapter:
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[
            0
        ].strip()


def build_rewriter_backend(config):
    if config.rewriter_url is not None:
        return HTTPRewriterBackend(
            url=config.rewriter_url,
            map_url=config.rewriter_map_url or config.rewriter_url,
            expand_model=config.rewriter_expand_model,
            map_model=config.rewriter_map_model,
            timeout=config.rewriter_timeout,
        )
    if config.rewriter_model_path is None:
        return None
    if config.rewriter_adapter_path is None:
        raise ValueError(
            "rewriter_model_path needs rewriter_adapter_path: the mapping turn is "
            "the base VLM with the rewriter adapter enabled."
        )
    return TransformersRewriterBackend(
        model_path=config.rewriter_model_path,
        adapter_path=config.rewriter_adapter_path,
        device_map=config.rewriter_device_map,
        max_new_tokens=config.rewriter_max_new_tokens,
    )


def _data_url(image: PIL.Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

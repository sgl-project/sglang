import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

_RESERVED_GENERATION_KWARGS = {
    "model",
    "messages",
    "stream",
    "n",
    "tools",
    "tool_choice",
}


def _check_generation_kwargs(value: dict[str, Any]) -> dict[str, Any]:
    # one non-streaming completion per diffusion request
    reserved = value.keys() & _RESERVED_GENERATION_KWARGS
    if reserved:
        raise ValueError(f"generation_kwargs cannot override {sorted(reserved)}")
    return value


class PromptEnhancerTaskConfig(BaseModel):
    """How one task talks to the enhancer; unset fields fall back to the top level."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = Field(default=None, min_length=1)
    # The user message text; "{prompt}" is replaced by the request prompt.
    user_template: str | None = None
    # Send the reference images before the text, the order some enhancers were trained on.
    images_first: bool = False
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("user_template")
    @classmethod
    def validate_user_template(cls, value: str | None) -> str | None:
        if value is not None and "{prompt}" not in value:
            raise ValueError('user_template must contain "{prompt}"')
        return value

    @field_validator("generation_kwargs")
    @classmethod
    def validate_generation_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_generation_kwargs(value)


class PromptEnhancerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: HttpUrl
    model: str = Field(min_length=1)
    system_prompt: str = Field(
        default=(
            "You expand prompts for image and video generation. The user supplies "
            "JSON with task and prompt fields. Expand the prompt into a concise, "
            "visually detailed description, clarifying composition, lighting and "
            "appearance where appropriate instead of simply repeating it. Preserve "
            "the requested subjects, count, quoted text, actions, style and language. "
            "Preserve special tokens and reference labels exactly. "
            "Clarify visual details without inventing new subjects or changing intent. "
            "For image editing, describe only the requested edits and preserve "
            "everything else. For video, preserve the requested motion and timing. "
            "Do not invent facts about reference media you cannot see. "
            "Return only the rewritten prompt, without surrounding quotes, "
            "commentary or markdown."
        ),
        min_length=1,
    )
    include_images: bool = False
    timeout: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    api_key_env: str | None = None
    generation_kwargs: dict[str, Any] = Field(default_factory=dict)
    tasks: dict[Literal["image", "image_edit", "video"], PromptEnhancerTaskConfig] = (
        Field(default_factory=dict)
    )
    # A built-in protocol fills in the tasks and sampling it defines; explicit
    # fields above take precedence.
    protocol: Literal["flux3-klein-pup"] | None = None
    # Local directory or Hugging Face repo id with the protocol's files.
    protocol_source: str | None = None

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: HttpUrl) -> HttpUrl:
        if value.username or value.password or value.query or value.fragment:
            raise ValueError(
                "base_url must not contain credentials, a query or fragment"
            )
        return value

    @field_validator("generation_kwargs")
    @classmethod
    def validate_generation_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _check_generation_kwargs(value)


class PromptEnhancer:
    """HTTP frontend client; owns its connection pool, never runs in GPU workers."""

    def __init__(self, config: PromptEnhancerConfig):
        if config.protocol is not None:
            config = _apply_protocol(config)
        self.config = config
        headers = {}
        if config.api_key_env is not None:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"Prompt enhancer API key env {config.api_key_env} is unset"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.AsyncClient(
            base_url=str(config.base_url).rstrip("/") + "/",
            headers=headers,
            timeout=config.timeout,
            trust_env=False,
        )

    @classmethod
    def from_file(cls, path: str) -> "PromptEnhancer":
        return cls(PromptEnhancerConfig.model_validate_json(Path(path).read_text()))

    async def close(self) -> None:
        await self.client.aclose()

    async def enhance(
        self,
        prompt: str,
        *,
        task: Literal["image", "image_edit", "video"],
        image_paths: list[str],
    ) -> str:
        task_config = self.config.tasks.get(task, PromptEnhancerTaskConfig())
        if task_config.user_template is None:
            text = json.dumps({"task": task, "prompt": prompt}, ensure_ascii=False)
        else:
            text = task_config.user_template.replace("{prompt}", prompt)
        content: str | list[dict[str, Any]] = text
        if self.config.include_images and image_paths:
            images = []
            for path in image_paths:
                if path.startswith(("https://", "http://", "data:image/")):
                    url = path
                elif self.config.protocol == "flux3-klein-pup":
                    url = await asyncio.to_thread(
                        _flux3_reference_url, path, len(image_paths)
                    )
                else:
                    url = await asyncio.to_thread(_local_image_url, path)
                images.append({"type": "image_url", "image_url": {"url": url}})
            text_part = [{"type": "text", "text": text}]
            content = (
                images + text_part if task_config.images_first else text_part + images
            )

        payload = {
            "max_tokens": 512,
            "temperature": 0,
            **self.config.generation_kwargs,
            **task_config.generation_kwargs,
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": task_config.system_prompt or self.config.system_prompt,
                },
                {"role": "user", "content": content},
            ],
            "stream": False,
            "n": 1,
        }
        try:
            response = await self.client.post("chat/completions", json=payload)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            raise HTTPException(504, "Prompt enhancer timed out") from e
        except httpx.HTTPError as e:
            # do not expose upstream response bodies, prompts or auth headers
            raise HTTPException(502, "Prompt enhancer request failed") from e

        try:
            choice = response.json()["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("Incomplete completion")
            rewritten = choice["message"]["content"]
            if not isinstance(rewritten, str) or not rewritten.strip():
                raise ValueError("Empty completion")
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise HTTPException(
                502,
                "Prompt enhancer did not return a complete, nonempty text response; "
                "check its template and token limit",
            ) from e
        return rewritten.strip()


# FLUX.3 Image Klein's prompt upsampler, as BFL's reference code drives it
# (github.com/black-forest-labs/flux-3-image-klein, src/flux3/pup.py).
_FLUX3_PUP_REPO = "black-forest-labs/flux-3-image-klein-pup"
_FLUX3_PUP_SAMPLING = {
    "max_tokens": 3584,
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
_FLUX3_PUP_EDIT_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {prompt}\n"
    "Output the caption."
)
_FLUX3_REFERENCE_MULTIPLE = 32
_FLUX3_REFERENCE_PIXELS = {1: 1024 * 1024, 2: 724 * 724}  # 512 * 512 for 3+


def _apply_protocol(config: PromptEnhancerConfig) -> PromptEnhancerConfig:
    source = config.protocol_source or _FLUX3_PUP_REPO
    if os.path.isdir(source):
        prompts_path = os.path.join(source, "system_prompts.json")
    else:
        from huggingface_hub import hf_hub_download

        prompts_path = hf_hub_download(source, "system_prompts.json")
    prompts = json.loads(Path(prompts_path).read_text())
    tasks = {
        "image": PromptEnhancerTaskConfig(
            system_prompt=prompts["t2i"],
            user_template="{prompt}",
            generation_kwargs={"presence_penalty": 1.5},
        ),
        "image_edit": PromptEnhancerTaskConfig(
            system_prompt=prompts["i2i"],
            user_template=_FLUX3_PUP_EDIT_SCAFFOLD,
            images_first=True,
            generation_kwargs={"presence_penalty": 0.0},
        ),
    }
    return config.model_copy(
        update={
            "include_images": True,
            "generation_kwargs": {**_FLUX3_PUP_SAMPLING, **config.generation_kwargs},
            "tasks": {**tasks, **config.tasks},
        }
    )


def _floor_multiple(value: float) -> int:
    m = _FLUX3_REFERENCE_MULTIPLE
    return max(m, int(math.floor(value / m)) * m)


def _flux3_reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    m = _FLUX3_REFERENCE_MULTIPLE
    cap = _FLUX3_REFERENCE_PIXELS.get(n_refs, 512 * 512)
    w = max(m, round(width / m) * m)
    h = max(m, round(height / m) * m)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = _floor_multiple(width / beta), _floor_multiple(height / beta)
    if w * h > cap:
        if w == m:
            h = _floor_multiple(cap / w)
        else:
            w = _floor_multiple(cap / h)
    return w, h


def _flux3_reference_url(path: str, n_refs: int) -> str:
    """The reference as the upsampler saw it in training: RGB over white, resized, JPEG q92."""
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in ("RGBA", "LA", "PA") or (
                image.mode == "P" and "transparency" in image.info
            ):
                rgba = image.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(white, rgba)
            image = image.convert("RGB")
            size = _flux3_reference_size(image.width, image.height, n_refs)
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=92)
    except (OSError, ValueError) as e:
        raise HTTPException(
            400, "Cannot read reference image for prompt enhancement"
        ) from e
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _local_image_url(path: str) -> str:
    # uploads live on the diffusion host, not necessarily on the SRT host
    try:
        with Image.open(path) as image:
            mime = image.get_format_mimetype()
            image.verify()
        if mime is None:
            raise ValueError("Unknown image MIME type")
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    except (OSError, ValueError, SyntaxError) as e:
        raise HTTPException(
            400, "Cannot read reference image for prompt enhancement"
        ) from e
    return f"data:{mime};base64,{encoded}"


async def maybe_enhance_prompt(
    request: Request,
    prompt: str,
    *,
    enabled: bool,
    task: Literal["image", "image_edit", "video"],
    image_paths: list[str] | None = None,
) -> str:
    if not enabled:
        return prompt
    enhancer: PromptEnhancer | None = request.app.state.prompt_enhancer
    if enhancer is None:
        raise HTTPException(400, "Prompt enhancement requires --prompt-enhancer-config")
    return await enhancer.enhance(prompt, task=task, image_paths=image_paths or [])

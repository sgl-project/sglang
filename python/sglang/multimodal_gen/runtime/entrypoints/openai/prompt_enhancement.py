import asyncio
import base64
import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import HTTPException, Request
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


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
        # one non-streaming completion per diffusion request
        reserved = value.keys() & {
            "model",
            "messages",
            "stream",
            "n",
            "tools",
            "tool_choice",
        }
        if reserved:
            raise ValueError(f"generation_kwargs cannot override {sorted(reserved)}")
        return value


class PromptEnhancer:
    """HTTP frontend client; owns its connection pool, never runs in GPU workers."""

    def __init__(self, config: PromptEnhancerConfig):
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
        content: str | list[dict[str, Any]] = json.dumps(
            {"task": task, "prompt": prompt}, ensure_ascii=False
        )
        if self.config.include_images and image_paths:
            content = [{"type": "text", "text": content}]
            for path in image_paths:
                if path.startswith(("https://", "http://", "data:image/")):
                    url = path
                else:
                    url = await asyncio.to_thread(_local_image_url, path)
                content.append({"type": "image_url", "image_url": {"url": url}})

        payload = {
            "max_tokens": 512,
            "temperature": 0,
            **self.config.generation_kwargs,
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
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

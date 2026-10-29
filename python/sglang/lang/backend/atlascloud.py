import os
from typing import Optional

from sglang.lang.backend.openai import OpenAI
from sglang.lang.chat_template import ChatTemplate

ATLASCLOUD_BASE_URL = "https://api.atlascloud.ai/v1"


class AtlasCloud(OpenAI):
    """SGLang backend for Atlas Cloud.

    Atlas Cloud exposes an OpenAI-compatible API, so this is a thin
    wrapper around the OpenAI backend that handles Atlas Cloud-specific
    defaults.

    Args:
        model_name: The model to use, e.g. "deepseek-ai/DeepSeek-V3.1-Terminus".
        api_key: Atlas Cloud API key. Defaults to ATLASCLOUD_API_KEY env var.
        base_url: Override the Atlas Cloud endpoint. Defaults to the Atlas Cloud API.
        chat_template: Optional custom chat template.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        chat_template: Optional[ChatTemplate] = None,
        **kwargs,
    ):
        resolved_api_key = api_key or os.environ.get("ATLASCLOUD_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Atlas Cloud API key required. Pass api_key= or set ATLASCLOUD_API_KEY."
            )

        super().__init__(
            model_name=model_name,
            chat_template=chat_template,
            api_key=resolved_api_key,
            base_url=base_url or ATLASCLOUD_BASE_URL,
            **kwargs,
        )

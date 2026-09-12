import os
from typing import Optional

from sglang.lang.backend.openai import OpenAI
from sglang.lang.chat_template import ChatTemplate

CRUSOE_BASE_URL = "https://managed-inference-api-proxy.crusoecloud.com/v1/"


class Crusoe(OpenAI):
    """SGLang backend for Crusoe managed inference.

    Crusoe exposes an OpenAI-compatible API, so this is a thin wrapper
    around the OpenAI backend that handles Crusoe-specific defaults.

    Args:
        model_name: The model to use, e.g. "meta-llama/Llama-3.1-8B-Instruct".
        api_key: Crusoe API key. Defaults to CRUSOE_API_KEY env var.
        base_url: Override the Crusoe endpoint. Defaults to the Crusoe API.
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
        resolved_api_key = api_key or os.environ.get("CRUSOE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Crusoe API key required. Pass api_key= or set CRUSOE_API_KEY."
            )

        super().__init__(
            model_name=model_name,
            chat_template=chat_template,
            api_key=resolved_api_key,
            base_url=base_url or CRUSOE_BASE_URL,
            **kwargs,
        )

import os
from typing import Optional

from sglang.lang.backend.openai import OpenAI
from sglang.lang.chat_template import ChatTemplate
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams


MINIMAX_DEFAULT_BASE_URL = "https://api.minimax.io/v1"


class MiniMax(OpenAI):
    """Backend for MiniMax models via the OpenAI-compatible API.

    MiniMax provides an OpenAI-compatible API endpoint. This backend
    configures the OpenAI client with MiniMax's default base URL and
    API key, and adjusts parameters to comply with MiniMax constraints.

    Usage:
        import sglang as sgl
        backend = sgl.MiniMax("MiniMax-M2.5")
        sgl.set_default_backend(backend)
    """

    def __init__(
        self,
        model_name: str = "MiniMax-M2.5",
        is_chat_model: Optional[bool] = None,
        chat_template: Optional[ChatTemplate] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *args,
        **kwargs,
    ):
        resolved_api_key = api_key or os.environ.get("MINIMAX_API_KEY")
        resolved_base_url = base_url or os.environ.get(
            "MINIMAX_BASE_URL", MINIMAX_DEFAULT_BASE_URL
        )

        super().__init__(
            model_name=model_name,
            is_chat_model=is_chat_model if is_chat_model is not None else True,
            chat_template=chat_template,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            *args,
            **kwargs,
        )

    def _clamp_temperature(self, sampling_params: SglSamplingParams):
        """MiniMax requires temperature in (0.0, 1.0]. Clamp 0 to a small value."""
        if sampling_params.temperature <= 0:
            sampling_params = sampling_params.clone()
            sampling_params.temperature = 0.01
        return sampling_params

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
        spec_var_name: str = None,
    ):
        sampling_params = self._clamp_temperature(sampling_params)
        return super().generate(s, sampling_params, spec_var_name)

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        sampling_params = self._clamp_temperature(sampling_params)
        return super().generate_stream(s, sampling_params)

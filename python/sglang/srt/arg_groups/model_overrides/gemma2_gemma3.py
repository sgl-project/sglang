"""Config-time override declarations for gemma2_gemma3.

Architectures: Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3nForCausalLM, Gemma3nForConditionalGeneration.
"""

import logging
from typing import Any

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
)

logger = logging.getLogger(__name__)


@_register_for(
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3ForConditionalGeneration",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
)
def _gemma2_gemma3_overrides(server_args: Any, hf_config: Any) -> dict:
    # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
    # It failed at this test: https://github.com/sgl-project/sglang/actions/runs/16255155597/job/45890331952#step:4:736
    logger.warning(
        f"Disable hybrid SWA memory for {hf_config.architectures[0]} as it is not yet supported."
    )
    return {"disable_hybrid_swa_memory": True}

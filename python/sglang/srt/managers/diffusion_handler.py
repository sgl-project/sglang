import logging
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import Scheduler

from sglang.srt.sampling.diffusion import diffusion_sample

logger = logging.getLogger(__name__)


def is_rnd1_model(scheduler: "Scheduler") -> bool:
    """Check if the loaded model is RND1."""
    try:
        model_type = getattr(scheduler.model_config.hf_text_config, "model_type", None)
        return model_type == "rnd1"
    except:
        return False


def diffusion_generation(scheduler: "Scheduler", req: "Req") -> List[int]:
    """
    Generate tokens for RND1 model (TODO: more models will be supported in the future) using diffusion sampling.

    This function is called after prefill for RND1 models. Instead of entering
    the standard autoregressive decode loop, it generates all remaining tokens
    at once using entropy-based diffusion sampling.
    """

    # Get generation parameters
    max_new_tokens = req.sampling_params.max_new_tokens

    temperature = req.sampling_params.temperature
    # Handle temperature == 0.0 case (greedy)
    if temperature is None or temperature < 1e-6:
        temperature = 1.0
        greedy = True
    else:
        greedy = False

    top_k = req.sampling_params.top_k if req.sampling_params.top_k > 0 else None
    top_p = req.sampling_params.top_p if req.sampling_params.top_p < 1.0 else None

    # Get RND1-specific config
    config = scheduler.model_config.hf_text_config
    mask_token_id = getattr(config, "mask_token_id", 151669)
    eos_token_id = getattr(config, "eos_token_id", 151645)
    pad_token_id = getattr(config, "pad_token_id", 151643)
    num_diffusion_steps = getattr(req.sampling_params, "num_diffusion_steps", None)

    if not num_diffusion_steps:
        num_diffusion_steps = getattr(config, "num_diffusion_steps", 256)

    # Prepare input - use ONLY the original prompt (no generated tokens)
    input_ids = torch.tensor(
        [req.origin_input_ids], dtype=torch.long, device=scheduler.device
    )
    prefix_len = len(req.origin_input_ids)
    seq_len = prefix_len + max_new_tokens

    # Run diffusion sampling
    generated_sequence = diffusion_sample(
        model_runner=scheduler.tp_worker.model_runner,
        input_ids=input_ids,
        seq_len=seq_len,
        num_steps=num_diffusion_steps,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        mask_token_id=mask_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

    # Extract generated tokens (skip the prefix)
    generated_tokens = generated_sequence[0, prefix_len:].tolist()

    # Find EOS and truncate if present
    if eos_token_id in generated_tokens:
        eos_idx = generated_tokens.index(eos_token_id)
        generated_tokens = generated_tokens[: eos_idx + 1]

    return generated_tokens

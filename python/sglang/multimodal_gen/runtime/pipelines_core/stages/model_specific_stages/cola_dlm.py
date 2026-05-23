"""Custom pipeline stages for Cola-DLM text diffusion model.

Three stages implement the Cola-DLM inference algorithm:
1. ColaTokenizationStage — tokenize prompt, VAE encode, prepare first-block data
2. ColaBlockDenoisingStage — block-wise ODE loop with CFG, VAE decode, token sampling
3. ColaTextDecodingStage — detokenize generated tokens to text

Reference: https://github.com/ByteDance-Seed/Cola-DLM/blob/main/cola_dlm/inference.py (generate_task_repaint_inference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-request state (carried in batch.extra["cola_state"])
# ---------------------------------------------------------------------------


@dataclass
class ColaRequestState:
    """State for one request across diffusion blocks."""

    # Prompt latents from VAE encode (fp32, after scaling)
    prefix_latents: torch.Tensor  # (n_prefix, latent_dim)
    prefix_len: int

    # Generation progress
    blocks_generated: int = 0
    generated_token_ids: List[int] = field(default_factory=list)

    # Current cumulative latent sequence length (prefix + generated blocks)
    kv_len: int = 0

    # First-block clean-guidance data
    first_block_done: bool = False
    first_block_latents: Optional[torch.Tensor] = None  # (block_size, latent_dim)
    first_block_labels: Optional[torch.Tensor] = None  # (block_size,)
    first_block_prompt_token_count: int = 0

    # CFG scale for first block (1.0 when prefix is empty)
    cfg_scale_first_block: float = 7.0

    finished: bool = False


# ---------------------------------------------------------------------------
# Sampling helper
# ---------------------------------------------------------------------------


def _sample_with_strategies(
    logits: torch.Tensor,
    generated_ids: Optional[torch.Tensor] = None,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> torch.Tensor:
    """Sample tokens from logits. Handles (batch, vocab) and (batch, block, vocab)."""
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    is_3d = logits.dim() == 3
    if is_3d:
        bs, blk, vocab = logits.shape
        logits = logits.reshape(-1, vocab)
    else:
        bs, vocab = logits.shape
        blk = 1

    if generated_ids is not None and repetition_penalty != 1.0:
        target = generated_ids.repeat_interleave(blk, dim=0) if is_3d else generated_ids
        score = torch.gather(logits, 1, target)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(1, target, score)

    if temperature < 1e-5:
        tokens = torch.argmax(logits, dim=-1)
        return tokens.view(bs, blk) if is_3d else tokens.view(bs)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, top_k)
        min_vals = topk_vals[:, -1].unsqueeze(-1)
        logits = torch.where(
            logits < min_vals, torch.full_like(logits, float("-inf")), logits
        )

    if 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        remove = remove.scatter(1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if torch.isnan(probs).any():
        probs = torch.nan_to_num(probs, nan=1.0 / vocab)
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return tokens.view(bs, blk) if is_3d else tokens.view(bs)


def _shape_tensor(lens: list[int], device: torch.device) -> torch.LongTensor:
    """Build a (B, 1) shape tensor from a Python list of per-sample lengths."""
    return torch.tensor([[l] for l in lens], dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Stage 1: Tokenization + VAE Encode + First-Block Setup
# ---------------------------------------------------------------------------


class ColaTokenizationStage(PipelineStage):
    """Tokenize prompt, VAE encode to latents, prepare first-block data.

    Populates batch.extra with Cola-DLM state for the denoising stage.
    """

    def __init__(self, vae, tokenizer, pipeline_config):
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = (
            batch.latents.device if batch.latents is not None else torch.device("cuda")
        )
        config = self.pipeline_config

        block_size = config.block_size
        patch_size = config.patch_size
        chunk = patch_size * block_size
        scale = config.scaling_factor
        shift = config.shifting_factor
        pad_token_id = config.pad_token_id

        # 1. Tokenize prompt
        prompt_text = batch.prompt or ""
        if hasattr(self.tokenizer, "encode"):
            # tokenizers.Tokenizer (from HuggingFace tokenizers library)
            ids = self.tokenizer.encode(prompt_text).ids
        else:
            # transformers.AutoTokenizer
            ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        prompt_ids = ids

        # 2. Block-align pad (same as official inference.py)
        p_pad_len = (chunk - len(prompt_ids) % chunk) % chunk
        token_labels = [1] * len(prompt_ids) + [3] * p_pad_len
        padded_ids = prompt_ids + [pad_token_id] * p_pad_len
        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=device)

        # 3. VAE encode
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            enc = self.vae.encode([input_tensor])
            latents = ((enc.latents_list[0] - shift) * scale).float()

        # 4. Latent labels
        t_labels = torch.tensor(token_labels, dtype=torch.long, device=device)
        n_patches = t_labels.shape[0] // patch_size
        reshaped = t_labels.view(n_patches, patch_size)
        c1 = (reshaped == 1).any(dim=1)
        c2 = (reshaped == 2).any(dim=1)
        lat_labels = torch.full((n_patches,), 3, dtype=torch.long, device=device)
        lat_labels[c2] = 2
        lat_labels[c1] = 1

        num_prompt_latents = int((lat_labels == 1).sum().item())
        lat_total = latents.shape[0]

        # 5. Split into prefix + first-block
        if num_prompt_latents % block_size != 0:
            start_idx = (num_prompt_latents // block_size) * block_size
            if start_idx + block_size <= lat_total:
                block_lat = latents[start_idx : start_idx + block_size].clone()
                block_lab = lat_labels[start_idx : start_idx + block_size].clone()
                block_lab[block_lab == 3] = 2
                token_start = start_idx * patch_size
                token_end = min(
                    token_start + block_size * patch_size, len(token_labels)
                )
                first_prompt_tok_count = sum(
                    1 for t in token_labels[token_start:token_end] if t == 1
                )
                prefix_latents = latents[:start_idx].clone()
                first_block_latents = block_lat
                first_block_labels = block_lab
            else:
                prefix_latents = latents[:num_prompt_latents].clone()
                first_block_latents = latents[
                    lat_total - block_size : lat_total
                ].clone()
                first_block_labels = torch.full(
                    (block_size,), 2, dtype=torch.long, device=device
                )
                first_prompt_tok_count = 0
        else:
            prefix_latents = latents[:num_prompt_latents].clone()
            first_block_latents = latents[lat_total - block_size : lat_total].clone()
            first_block_labels = torch.full(
                (block_size,), 2, dtype=torch.long, device=device
            )
            first_prompt_tok_count = 0

        prefix_len = prefix_latents.shape[0]

        # 6. CFG scale for first block
        sp = batch.sampling_params
        guidance_scale = sp.guidance_scale if sp else 7.0
        cfg_first = guidance_scale if prefix_len > 0 else 1.0

        # 7. Create state
        state = ColaRequestState(
            prefix_latents=prefix_latents,
            prefix_len=prefix_len,
            kv_len=prefix_len,
            first_block_latents=first_block_latents,
            first_block_labels=first_block_labels,
            first_block_prompt_token_count=first_prompt_tok_count,
            cfg_scale_first_block=cfg_first,
        )

        # Store state and config in batch.extra
        batch.extra["cola_state"] = state
        batch.extra["cola_prompt_ids"] = prompt_ids

        return batch


# ---------------------------------------------------------------------------
# Stage 2: Block-wise Denoising (ODE + CFG + Token Sampling)
# ---------------------------------------------------------------------------


class ColaBlockDenoisingStage(PipelineStage):
    """Block-wise diffusion denoising for Cola-DLM.

    Each call generates one block of block_size tokens via:
    1. Sample noise ~ N(0, I)
    2. Euler ODE integration (timestep_num steps from T to 0) with CFG
    3. VAE decode to logits
    4. Sample tokens
    5. Update state, check finish conditions
    """

    def __init__(self, dit, vae, pipeline_config):
        super().__init__()
        self.dit = dit
        self.vae = vae
        self.pipeline_config = pipeline_config

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        state: ColaRequestState = batch.extra["cola_state"]
        device = (
            batch.latents.device if batch.latents is not None else torch.device("cuda")
        )

        config = self.pipeline_config
        block_size = config.block_size
        patch_size = config.patch_size
        latent_dim = config.latent_dim

        # Get sampling params from batch (SamplingParams auto-delegates)
        sp = batch.sampling_params
        guidance_scale = sp.guidance_scale if sp else 7.0
        timestep_num = sp.num_inference_steps if sp else 16
        T = getattr(sp, "T", 1000.0) if sp else 1000.0
        temperature = sp.temperature if sp else 0.0
        top_k = sp.top_k if sp else 50
        top_p = sp.top_p if sp else 0.9
        repetition_penalty = sp.repetition_penalty if sp else 1.1
        max_new_tokens = sp.max_new_tokens if sp else 256

        # Enable KV cache on both models
        self.dit.set_kv_cache(True)
        self.vae.set_kv_cache(True)

        # Timestep schedule
        timesteps = torch.linspace(
            int(T), 0, timestep_num + 1, dtype=torch.float32, device=device
        )

        def _dt(t_curr, t_next):
            return (float(t_curr) - float(t_next)) / max(T, 1.0)

        # --- Prefix KV prefetch (DiT + VAE decoder) ---
        # Warm the KV caches with the prefix latents so subsequent block-wise
        # calls only attend over the newly added Q tokens.
        prefix_len = state.prefix_len
        if prefix_len > 0:
            txt_prefix = state.prefix_latents.to(torch.bfloat16)
            txt_shape_prefix = _shape_tensor([prefix_len], device)
            ts_prefix = torch.zeros(prefix_len, device=device, dtype=torch.bfloat16)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = self.dit(
                    txt=txt_prefix,
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    timestep=ts_prefix,
                    update_kv=True,
                    use_kv_cache=True,
                )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _ = self.vae.decode(
                    z=state.prefix_latents,
                    txt_shape=txt_shape_prefix,
                    txt_q_shape=txt_shape_prefix,
                    update_kv=True,
                )

        # Per-sample cumulative K length (initially = prefix length)
        txt_shape_cum = _shape_tensor([prefix_len], device)
        txt_q_shape = _shape_tensor([block_size], device)

        # CFG scale for first block: 1.0 when prefix is empty (avoids
        # amplifying bf16 noise when conditional == unconditional)
        cfg_scale_first = guidance_scale if prefix_len > 0 else 1.0

        # Loop until all blocks are generated
        while not state.finished:
            # Update cumulative K length
            txt_shape_cum = txt_shape_cum + block_size

            # Sample noise for this block
            txt = torch.randn(block_size, latent_dim, device=device)

            # Clean-guidance setup for first block
            is_first_block = not state.first_block_done
            flat_mask = None
            if is_first_block and state.first_block_labels is not None:
                flat_mask = state.first_block_labels == 1

            # Euler ODE integration
            for t_curr, t_next in zip(timesteps[:-1], timesteps[1:]):
                ts_batch = torch.full((block_size,), t_curr, device=device)
                dt = _dt(t_curr, t_next)

                # Clean guidance: pin prompt positions at t=0
                if is_first_block and flat_mask is not None and flat_mask.any():
                    ts_batch[flat_mask] = 0
                    txt[flat_mask] = state.first_block_latents[flat_mask]

                txt_bf16 = txt.to(torch.bfloat16)
                ts_bf16 = ts_batch.to(torch.bfloat16)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Conditional pass — reads from KV cache (prefix + prior blocks)
                    drift_cond = self.dit(
                        txt=txt_bf16,
                        txt_shape=txt_shape_cum,
                        txt_q_shape=txt_q_shape,
                        timestep=ts_bf16,
                        update_kv=False,
                        use_kv_cache=True,
                    ).txt_sample

                    # Unconditional pass — no KV cache, block only
                    drift_uncond = self.dit(
                        txt=txt_bf16,
                        txt_shape=txt_q_shape,
                        txt_q_shape=txt_q_shape,
                        timestep=ts_bf16,
                        update_kv=False,
                        use_kv_cache=False,
                    ).txt_sample

                # CFG combination
                s = cfg_scale_first if is_first_block else guidance_scale
                drift = s * (drift_cond - drift_uncond) + drift_uncond
                txt_next = txt - drift * dt

                # Pin prompt positions after update
                if is_first_block and flat_mask is not None and flat_mask.any():
                    txt_next[flat_mask] = state.first_block_latents[flat_mask]

                txt = txt_next

            if is_first_block:
                state.first_block_done = True

            # VAE decode → logits (uses KV cache for prefix context)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                decoded = self.vae.decode(
                    z=txt,
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    update_kv=True,
                )
            decoded_logits = decoded[0, -block_size * patch_size :, :].float()

            # Sample tokens
            gen_tensor = None
            if state.generated_token_ids:
                gen_tensor = torch.tensor(
                    [state.generated_token_ids], dtype=torch.long, device=device
                )

            one_block = _sample_with_strategies(
                decoded_logits.unsqueeze(0),
                generated_ids=gen_tensor,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            token_ids = one_block[0].tolist()

            # Trim first-block prompt tokens
            if is_first_block and state.first_block_prompt_token_count > 0:
                trim = int(state.first_block_prompt_token_count)
                token_ids = token_ids[trim:]

            # Write the just-denoised block into the DiT KV cache
            txt_bf16 = txt.to(torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = self.dit(
                    txt=txt_bf16,
                    txt_shape=txt_shape_cum,
                    txt_q_shape=txt_q_shape,
                    timestep=torch.zeros(
                        block_size, device=device, dtype=torch.bfloat16
                    ),
                    update_kv=True,
                    use_kv_cache=True,
                )

            # Update state
            state.blocks_generated += 1
            state.generated_token_ids.extend(token_ids)

            # Check finish conditions
            eos_id = config.eos_token_id
            if eos_id is not None and eos_id in token_ids:
                state.finished = True
            if len(state.generated_token_ids) >= max_new_tokens:
                state.finished = True

        # Clean up KV cache
        self.dit.set_kv_cache(False)
        self.vae.set_kv_cache(False)

        batch.extra["cola_output_ids"] = state.generated_token_ids
        return batch


# ---------------------------------------------------------------------------
# Stage 3: Text Decoding (Detokenize)
# ---------------------------------------------------------------------------


class ColaTextDecodingStage(PipelineStage):
    """Detokenize generated token IDs to text string and return OutputBatch."""

    def __init__(self, tokenizer, pipeline_config):
        super().__init__()
        self.tokenizer = tokenizer
        self.pipeline_config = pipeline_config

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output_ids = batch.extra.get("cola_output_ids", [])

        if not output_ids:
            text = ""
        elif hasattr(self.tokenizer, "decode"):
            # tokenizers.Tokenizer or transformers.AutoTokenizer
            text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        else:
            text = str(output_ids)

        return OutputBatch(output=[text])

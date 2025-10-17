import logging
from typing import Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position_torch,
)

logger = logging.getLogger(__name__)


def apply_top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits: with non-top-k values set to -inf"""
    top_k_values, top_k_indices = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, top_k_indices, top_k_values)
    return filtered_logits


def apply_top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Apply top-p (nucleus) filtering to logits: with tokens beyond threshold set to -inf"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 0] = False  # Keep at least o token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


@torch.no_grad()
def diffusion_sample(
    model_runner,
    input_ids: torch.Tensor,
    seq_len: int,
    num_steps: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = True,
    mask_token_id: int = 151669,
    eos_token_id: int = 151645,
    pad_token_id: int = 0,
    bos_token_id: Optional[int] = None,
    add_eos_token_at_end: bool = True,
):
    """
    Perform masked diffusion sampling with entropy-based token selection.

    This is the SGLang-integrated version that uses the model_runner to perform
    forward passes.

    Args:
        model_runner: SGLang ModelRunner instance
        input_ids: Input prompt token IDs [batch_size, prompt_len]
        seq_len: Target sequence length
        num_steps: Number of denoising steps
        temperature: Temperature for sampling
        top_k: Optional top-k filtering
        top_p: Optional nucleus (top-p) filtering
        greedy: Whether to use greedy sampling
        mask_token_id: Token ID for masked positions
        eos_token_id: End of sequence token ID
        pad_token_id: Padding token ID
        bos_token_id: Beginning of sequence token ID
        add_eos_token_at_end: Whether to add the EOS token at the end of the sequence

    Returns:
        Generated token IDs as LongTensor [batch_size, seq_len]
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]
    prefix_len = input_ids.shape[1]

    # Build initial masked sequence
    # Format: [prefix_ids | mask_tokens | eos_token]
    x = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long, device=device)

    # Place prefix
    x[:, :prefix_len] = input_ids

    # Place mask tokens for generation
    gen_start = prefix_len
    gen_end = seq_len - 1  # Reserve last position for EOS
    x[:, gen_start:gen_end] = mask_token_id

    # Place EOS token at the end
    if add_eos_token_at_end and eos_token_id is not None:
        x[:, -1] = eos_token_id

    # Create maskable positions indicator
    init_maskable = torch.zeros_like(x, dtype=torch.bool)
    init_maskable[:, gen_start:gen_end] = True

    # Ensure special tokens are not maskable
    if bos_token_id is not None:
        init_maskable[:, 0] = False
    if eos_token_id is not None:
        init_maskable &= x.ne(eos_token_id)
    init_maskable &= x.ne(pad_token_id)

    maskable = init_maskable.clone()
    xt = x.clone()

    batch_size, seq_len = xt.shape

    # Build correct per-sequence positions using built-in utility
    extend_seq_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    extend_prefix_lens = torch.zeros(batch_size, dtype=torch.int32, device=device)
    positions, extend_start_loc = compute_position_torch(
        extend_prefix_lens, extend_seq_lens
    )

    # Create ForwardBatch with NO KV cache allocation for DLLMs for now
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=batch_size,
        input_ids=xt.reshape(-1),
        req_pool_indices=torch.zeros(batch_size, dtype=torch.int32, device=device),
        seq_lens=extend_seq_lens,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=model_runner.attn_backend,
        out_cache_loc=None,  # NO KV cache allocation for DLLMs
        seq_lens_sum=int(batch_size * seq_len),
        return_logprob=False,
        positions=positions,
        extend_num_tokens=int(seq_len),
        extend_seq_lens=extend_seq_lens,
        extend_prefix_lens=extend_prefix_lens,
        extend_prefix_lens_cpu=[0] * batch_size,
        extend_seq_lens_cpu=[int(seq_len)] * batch_size,
        extend_logprob_start_lens_cpu=[0] * batch_size,
        top_logprobs_nums=[],
        capture_hidden_mode=CaptureHiddenMode.FULL,
    )
    # Initialize attention backend metadata
    try:
        model_runner.attn_backend.init_forward_metadata(forward_batch)
    except Exception:
        pass

    def forward_scores(tokens):
        """Compute predictions and entropy scores for next tokens."""
        with torch.no_grad():

            # update forward batch with new tokens
            forward_batch.input_ids = tokens.reshape(-1)

            # Use model to compute hidden states directly (bypass logits processor)
            model = model_runner.model
            hidden_states = model.model(  # Use model.model to bypass logits processor
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
            )

            # Obtain full-sequence logits via LogitsProcessor to handle TP correctly
            logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
            logits_flat = model.logits_processor._get_logits(
                hidden_states, model.lm_head, logits_metadata
            )

            # Reshape logits to [batch_size, seq_len, vocab_size]
            logits = logits_flat.view(batch_size, seq_len, -1)

            # Apply temperature scaling
            safe_temperature = max(temperature, 1e-8)
            logits = logits / safe_temperature

            # Apply filtering strategies
            if top_k is not None and top_k > 0:
                logits = apply_top_k_filtering(logits, top_k)

            if top_p is not None and 0 < top_p < 1.0:
                logits = apply_top_p_filtering(logits, top_p)

            # Convert to log probabilities
            logp = torch.log_softmax(logits, dim=-1)

            # Greedy or stochastic sampling
            if greedy:
                pred_next = logp.argmax(-1)
            else:
                pred_next = torch.distributions.Categorical(logits=logp).sample()

            conf_next = torch.gather(logp, -1, pred_next.unsqueeze(-1)).squeeze(-1)

            p = logp.exp()
            ent_next = -(p * logp).sum(-1)

            # Shift predictions: pos i predicts token i+1
            pred_i = tokens.clone()
            conf_i = torch.full_like(conf_next, torch.finfo(conf_next.dtype).min)
            ent_i = torch.zeros_like(ent_next)

            pred_i[:, 1:] = pred_next[:, :-1]
            conf_i[:, 1:] = conf_next[:, :-1]
            ent_i[:, 1:] = ent_next[:, :-1]

            return pred_i, conf_i, ent_i

    # Initial forward pass
    pred_i, conf_i, ent_i = forward_scores(xt)
    total_masked = init_maskable.sum(1, keepdim=True)
    finf = torch.finfo(conf_i.dtype)

    # Diffusion loop: gradually unmask tokens
    for step in range(num_steps - 1, 0, -1):
        rate = step / num_steps
        cutoff_len = (total_masked * rate).long().clamp(min=0)

        # Choose HIGH-entropy tokens to keep masked
        sel_scores = ent_i.masked_fill(~maskable, -finf.max)
        B, L = sel_scores.shape
        k_max = cutoff_len.max().item()
        if k_max > 0:
            sss, idx = torch.topk(sel_scores, k_max, dim=-1, largest=True)
            keep_mask = torch.zeros_like(sel_scores, dtype=torch.bool)
            for b in range(B):
                k_b = int(cutoff_len[b].item())
                if k_b > 0:
                    keep_mask[b, idx[b, :k_b]] = True
        else:
            keep_mask = torch.zeros_like(sel_scores, dtype=torch.bool)

        to_unmask = maskable & ~keep_mask
        if to_unmask.any():
            xt[to_unmask] = pred_i[to_unmask]
            maskable[to_unmask] = False

        # If there are still masked tokens, run another forward pass
        if maskable.any():
            pred_i, conf_i, ent_i = forward_scores(xt)

    # Final unmasking
    if maskable.any():
        xt[maskable] = pred_i[maskable]

    return xt

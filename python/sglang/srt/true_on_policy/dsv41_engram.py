"""DeepSeek-V4.1 Engram as one TOP program for serving and the trainer.

Hash ids come from SGLang's Triton hash kernels (integer arithmetic; serving's extend, decode and verify modes and the
trainer's packed prefill all reduce to the same n-gram of compressed ids per position). The table lookup dequantizes
E4M3 rows with UE8M0 scales exactly. ``wkv`` is the 32x32 FP8 projection (fp8_b32_linear). The gate is the fused
Triton kernel only (one program per token and stream copy, FP32, no FP fusion), measured row-invariant on H200; the
torch fallback is a different reduction order and is refused.
"""

import torch


def gate(
    x, kv, q_weight, k_weight, eps, clamp_value, input_ids=None, image_token_id=None
):
    """x [T, hc_mult, dim] + sigmoid(signed-sqrt normalized dot) * value, per stream copy.

    With ``input_ids`` [T], rows holding ``image_token_id`` keep x: the kernel selects after the same FP32 math, so
    every other row is the text program's."""
    if (
        not x.is_cuda
        or x.ndim != 3
        or kv.shape != (x.shape[0], (x.shape[1] + 1) * x.shape[2])
        or x.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or q_weight.dtype != torch.bfloat16
        or k_weight.dtype != torch.bfloat16
        or (input_ids is None) != (image_token_id is None)
    ):
        raise NotImplementedError(
            "[top] V4.1 engram gate runs the fused kernel on CUDA BF16 streams, keys and q/k weights only"
        )
    from sglang.kernels.ops.embeddings.engram_gate import fused_engram_gate

    return fused_engram_gate(
        x.contiguous(),
        kv.contiguous(),
        q_weight.contiguous(),
        k_weight.contiguous(),
        eps,
        clamp_value,
        input_ids,
        image_token_id,
    )


def hash_packed(hasher, input_ids, lengths, *, image_token_id=None):
    """Hash ids [T, n_engram_layers, n_hash_cols] of requests packed back to back, each starting at position 0.

    The trainer's form of a cold prefill: no history before any request. ``image_token_id`` blocks n-grams at image
    positions, as serving's hasher does for a model with a vision tower.
    """
    from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND, engram_hash_ids
    from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

    if not input_ids.is_cuda or input_ids.ndim != 1:
        raise NotImplementedError("[top] V4.1 engram hashes packed 1-D CUDA token ids")
    device = input_ids.device
    lens = torch.as_tensor(lengths, dtype=torch.int64, device=device)
    starts = torch.cumsum(lens, 0) - lens
    row = torch.repeat_interleave(
        torch.arange(lens.shape[0], device=device), lens, output_size=input_ids.shape[0]
    )
    positions = torch.arange(input_ids.shape[0], device=device) - starts[row]
    history = torch.zeros(
        (lens.shape[0], hasher.max_ngram_size - 1), dtype=torch.int32, device=device
    )
    hash_ids, _ = engram_hash_ids(
        input_ids,
        positions,
        mode=MODE_EXTEND,
        history=history,
        token_map=hasher.token_map,
        multipliers=hasher.multipliers,
        primes=hasher.primes,
        offsets=hasher.offsets,
        pad_id=hasher.pad_id,
        num_real=input_ids.shape[0],
        req_slots=None,
        row=row,
        starts=starts,
        image_token_id=image_token_id,
        # Serving's value: only multimodal pad ids read as images; the kernel's default 0 reads every id as one.
        mm_pad_shift=MM_PAD_SHIFT_VALUE,
    )
    return hash_ids

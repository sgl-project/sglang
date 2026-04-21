"""Per-expert Hessian sensitivity via HVP double-backward.

Computes ½·dᵀHd per expert, where d = W_BF16 − dequant(W_INT4_from_HF).
Everything stays on GPU: model loaded direct to device via
``from_pretrained(device_map=...)``; INT4 checkpoints streamed via safetensors
``device=f'cuda:N'`` (reuses existing ``_load_int4_dequantized``).

Distributed: torchrun --nproc-per-node=N. Each rank processes a disjoint
subset of calibration samples; per-expert scalars are AllReduce'd at the end.

Layer-chunked to bound grad memory: only one chunk's expert params hold
``requires_grad=True`` at a time (others are frozen — no 2nd-order graph
for them). Forward re-runs per chunk; the autograd graph for the chunk is
rebuilt each pass.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List


def _rss_gb() -> float:
    """Process resident-set size in GB (excludes file-backed mmap that kernel can reclaim)."""
    try:
        with open(f"/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return -1.0

import torch
import torch.distributed as dist

# Reuse existing GPTQ dequant + calibration loader from the per-expert script.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "sensitivity" / "per_expert"),
)
from sensitivity import (  # noqa: E402
    _load_calibration_data,
    _load_int4_dequantized,
    _resolve_hf_path,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def _freeze_all(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _chunk_params(model, layer_ids: Iterable[int]) -> List[torch.nn.Parameter]:
    """Deterministic param ordering for a chunk: [gate_up_L0, down_L0, gate_up_L1, ...]."""
    params = []
    for L in layer_ids:
        layer = model.model.layers[L]
        params.append(layer.mlp.experts.gate_up_proj)
        params.append(layer.mlp.experts.down_proj)
    return params


def _set_chunk_requires_grad(model, layer_ids: Iterable[int], flag: bool) -> None:
    for p in _chunk_params(model, layer_ids):
        p.requires_grad_(flag)


def _compute_chunk_d(
    model,
    int4_ckpt: str,
    layer_ids: Iterable[int],
    num_experts: int,
    hidden: int,
    moe_intermediate: int,
    group_size: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Returns [d_gate_up_L0, d_down_L0, d_gate_up_L1, ...] matching _chunk_params order."""
    ds: List[torch.Tensor] = []
    for L in layer_ids:
        layer = model.model.layers[L]
        bf16_w13 = layer.mlp.experts.gate_up_proj.data  # [E, 2I, H]
        bf16_w2 = layer.mlp.experts.down_proj.data  # [E, H, I]

        int4_w13, int4_w2 = _load_int4_dequantized(
            int4_ckpt, L, num_experts, hidden, moe_intermediate, group_size, device
        )

        d_w13 = (bf16_w13 - int4_w13).detach()
        d_w2 = (bf16_w2 - int4_w2).detach()

        del int4_w13, int4_w2
        ds.append(d_w13)
        ds.append(d_w2)

    torch.cuda.empty_cache()
    return ds


# ---------------------------------------------------------------------------
# HVP core
# ---------------------------------------------------------------------------

def _per_expert_reduce(t: torch.Tensor) -> torch.Tensor:
    """Sum over all non-expert dims. t shape [E, ...] → returns [E]."""
    return t.flatten(1).sum(dim=1)


def _run_hvp(model, input_ids: torch.Tensor, layer_ids, ds):
    """One HVP pass for a chunk.

    Returns:
        fo_per_param: list of [E] tensors — first-order ``(d · g)`` per expert,
                      one entry per param (gate_up and down interleaved per layer).
        hvp_per_param: list of [E] tensors — ``(d · Hd)`` per expert.
        loss_val: scalar float, the NLL for this forward pass.
    """
    params = _chunk_params(model, layer_ids)

    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = out.loss

    # First backward (with double-backward graph retained).
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # gᵀd — scalar, triggers tangent path to H·d.
    gd = sum((g * d).sum() for g, d in zip(grads, ds))

    # Second backward: H·d (block-diagonal — only chunk params have requires_grad).
    Hd = torch.autograd.grad(gd, params)

    fo = [_per_expert_reduce(d * g.detach()) for d, g in zip(ds, grads)]
    hvp = [_per_expert_reduce(d * h) for d, h in zip(ds, Hd)]

    loss_val = float(loss.detach().item())

    # Free 2nd-order graph refs.
    del grads, gd, Hd, loss, out
    return fo, hvp, loss_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-expert Hessian sensitivity via HVP")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--int4_checkpoint", required=True)
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--chunk_size", type=int, default=2,
                    help="Layers per HVP chunk. Smaller = lower peak mem, slower.")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    # Distributed init (torchrun sets env vars; single-GPU falls through).
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    rank_tag = f"[rank{rank}/{world_size}] " if world_size > 1 else ""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {rank_tag}%(levelname)s %(message)s",
    )

    # --- Load model direct to this rank's GPU (no CPU staging).
    from transformers import AutoModelForCausalLM  # noqa: E402

    hf_name = _resolve_hf_path(args.model_path)
    logger.info(f"Loading BF16 model direct to cuda:{local_rank}  ({hf_name})")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        device_map=f"cuda:{local_rank}",
        attn_implementation="eager",
    )
    model.eval()
    load_peak = torch.cuda.max_memory_allocated(device) / 1e9
    logger.info(f"  loaded in {time.time()-t0:.1f}s; peak GPU mem {load_peak:.1f} GB")

    cfg = model.config
    num_experts = cfg.num_experts
    hidden = cfg.hidden_size
    moe_intermediate = cfg.moe_intermediate_size
    num_layers = cfg.num_hidden_layers
    logger.info(
        f"  E={num_experts}  H={hidden}  I={moe_intermediate}  layers={num_layers}"
    )

    _freeze_all(model)

    # --- Calibration samples: load once, split across ranks round-robin.
    all_ids = _load_calibration_data(hf_name, args.nsamples, args.seqlen, args.seed)
    my_idx = list(range(rank, args.nsamples, world_size))
    my_ids = all_ids[my_idx].to(device)
    logger.info(f"  {len(my_idx)} of {args.nsamples} calib samples assigned to this rank")

    # --- Layer chunks.
    layer_chunks = [
        list(range(s, min(s + args.chunk_size, num_layers)))
        for s in range(0, num_layers, args.chunk_size)
    ]
    logger.info(f"  {len(layer_chunks)} chunks x {args.chunk_size} layers")

    # --- On-GPU accumulators [num_layers, num_experts], float64 for numerical safety.
    score_accum = torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64)
    fo_accum = torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64)
    d_norm_sq = torch.zeros(num_layers, num_experts, device=device, dtype=torch.float64)
    loss_sum = 0.0
    sample_count = 0

    total_t0 = time.time()
    rss_baseline = _rss_gb()
    logger.info(f"  host RSS baseline (pre-HVP): {rss_baseline:.2f} GB")
    for c_idx, chunk in enumerate(layer_chunks):
        t_chunk = time.time()
        _set_chunk_requires_grad(model, chunk, True)

        # Compute d once per chunk.
        ds = _compute_chunk_d(
            model, args.int4_checkpoint, chunk,
            num_experts, hidden, moe_intermediate, args.group_size, device,
        )

        # d_norm_sq is data-independent — fill once per chunk.
        for L_idx, L in enumerate(chunk):
            dw13 = ds[2 * L_idx]
            dw2 = ds[2 * L_idx + 1]
            d_norm_sq[L] = (
                _per_expert_reduce(dw13.pow(2)) + _per_expert_reduce(dw2.pow(2))
            ).to(torch.float64)

        # Iterate samples on this rank for this chunk.
        for s_idx in range(my_ids.shape[0]):
            input_ids = my_ids[s_idx : s_idx + 1]  # [1, seqlen]
            try:
                fo_pp, hvp_pp, loss_val = _run_hvp(model, input_ids, chunk, ds)
            except torch.cuda.OutOfMemoryError:
                peak = torch.cuda.max_memory_allocated(device) / 1e9
                logger.error(
                    f"OOM at chunk {c_idx} (layers {chunk}) sample {s_idx}. "
                    f"Peak {peak:.1f} GB. Try smaller --chunk_size or --seqlen."
                )
                raise

            # Only add loss on the first chunk — loss is chunk-independent; summing
            # over all chunks would just inflate by a factor of len(layer_chunks).
            if c_idx == 0:
                loss_sum += loss_val
                sample_count += 1
                # RSS safeguard: every 16 samples on chunk 0, log host memory.
                if s_idx == 0 or (s_idx + 1) % 16 == 0 or s_idx + 1 == my_ids.shape[0]:
                    rss = _rss_gb()
                    delta = rss - rss_baseline if rss_baseline > 0 else 0
                    logger.info(
                        f"    [sample {s_idx+1}/{my_ids.shape[0]}] host RSS={rss:.2f} GB "
                        f"(Δ={delta:+.2f} GB vs baseline)"
                    )

            for L_idx, L in enumerate(chunk):
                fo_layer = (fo_pp[2 * L_idx] + fo_pp[2 * L_idx + 1]).to(torch.float64)
                hvp_layer = (hvp_pp[2 * L_idx] + hvp_pp[2 * L_idx + 1]).to(torch.float64)
                fo_accum[L] += fo_layer
                score_accum[L] += 0.5 * hvp_layer

            del fo_pp, hvp_pp

        _set_chunk_requires_grad(model, chunk, False)
        del ds
        gc.collect()
        torch.cuda.empty_cache()

        peak = torch.cuda.max_memory_allocated(device) / 1e9
        logger.info(
            f"  chunk {c_idx+1}/{len(layer_chunks)} layers {chunk[0]}..{chunk[-1]} "
            f"done in {time.time()-t_chunk:.1f}s | peak {peak:.1f} GB"
        )

    total_elapsed = time.time() - total_t0
    logger.info(f"All chunks done in {total_elapsed:.1f}s")

    # --- AllReduce across ranks.
    if world_size > 1:
        dist.all_reduce(score_accum, op=dist.ReduceOp.SUM)
        dist.all_reduce(fo_accum, op=dist.ReduceOp.SUM)

        count_t = torch.tensor([sample_count], device=device, dtype=torch.float64)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        total_samples = int(count_t.item())

        loss_t = torch.tensor([loss_sum], device=device, dtype=torch.float64)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_sum = loss_t.item()

        # d_norm_sq is identical across ranks (same weights) — broadcast from 0 is fine.
        dist.broadcast(d_norm_sq, src=0)
    else:
        total_samples = sample_count

    score_accum /= max(total_samples, 1)
    fo_accum /= max(total_samples, 1)
    avg_loss = loss_sum / max(total_samples, 1)

    # --- Write results (rank 0 only).
    if rank == 0:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        per_layer = {}
        for L in range(num_layers):
            experts_d = {
                str(E): {
                    "hessian_score": float(score_accum[L, E].item()),
                    "first_order_score": float(fo_accum[L, E].item()),
                    "d_norm_sq": float(d_norm_sq[L, E].item()),
                }
                for E in range(num_experts)
            }
            per_layer[str(L)] = {"experts": experts_d}

        result = {
            "model": args.model_path,
            "int4_checkpoint": args.int4_checkpoint,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "seed": args.seed,
            "group_size": args.group_size,
            "chunk_size": args.chunk_size,
            "world_size": world_size,
            "total_samples": total_samples,
            "avg_loss": avg_loss,
            "elapsed_seconds": total_elapsed,
            "metric": "half_dTHd",
            "per_layer": per_layer,
        }

        out_path = out_dir / "hessian_scores.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote {out_path}")
        logger.info(f"  avg_loss={avg_loss:.4f}  total_samples={total_samples}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

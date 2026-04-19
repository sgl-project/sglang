"""Per-expert INT4 sensitivity analysis using sglang fused MoE kernels.

Loads the BF16 model to CPU, processes one MoE layer at a time on GPU.
For each layer, dequantizes INT4 GPTQ weights to BF16.  For each expert,
swaps that expert's BF16 weights with the dequantized INT4 version and
runs outplace_fused_experts to measure ||swapped_output - baseline||_2.

No Marlin kernel — uses only the Triton BF16 fused MoE kernel.

Multi-GPU: run.sh launches one process per GPU, each with a disjoint
expert range (--expert_start / --expert_end).

Usage:
    python sensitivity.py --gpus 4 ...
    python sensitivity.py --gpu 0 --expert_start 0 --expert_end 32 --rank 0 ...
    python sensitivity.py --merge --out_dir results/
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _resolve_hf_path(path):
    m = re.match(r"^(.+)/hub/models--(.+?)--(.+?)/snapshots/[a-f0-9]+$", path)
    if m:
        os.environ.setdefault("HF_HOME", m.group(1))
        return f"{m.group(2)}/{m.group(3)}"
    return path


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def _load_calibration_data(tokenizer_path, nsamples, seqlen, seed):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    random.seed(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tok("\n\n".join(data["text"]), return_tensors="pt")
    ids = enc.input_ids[0]
    samples = []
    for _ in range(nsamples):
        i = random.randint(0, ids.shape[0] - seqlen - 1)
        samples.append(ids[i : i + seqlen])
    return torch.stack(samples)


# ---------------------------------------------------------------------------
# INT4 GPTQ → BF16 dequantization
# ---------------------------------------------------------------------------

def _dequant_gptq_int4(qweight, scales, qzeros, group_size):
    """Dequantize GPTQ INT4 packed weights to BF16.

    Args:
        qweight: [K//8, N] INT32 (8 × INT4 packed)
        scales:  [K//group_size, N] FP16
        qzeros:  [K//group_size, N//8] INT32 (packed zeros)
        group_size: int

    Returns:
        [K, N] BF16 dequantized weight
    """
    K = qweight.shape[0] * 8
    N = qweight.shape[1]
    device = qweight.device

    # Unpack INT4 weights → [K, N]
    w = torch.zeros(K, N, dtype=torch.int32, device=device)
    for bit in range(8):
        w[bit::8] = (qweight >> (4 * bit)) & 0xF

    # Unpack INT4 zeros → [num_groups, N]
    num_groups = qzeros.shape[0]
    zp = torch.zeros(num_groups, N, dtype=torch.int32, device=device)
    for bit in range(8):
        zp[:, bit::8] = (qzeros >> (4 * bit)) & 0xF

    # Dequantize: (int4_val - zero_point) * scale, per group
    w_float = w.to(torch.float32)
    for g in range(num_groups):
        row_start = g * group_size
        row_end = min(row_start + group_size, K)
        w_float[row_start:row_end] = (
            (w_float[row_start:row_end] - zp[g].float()) * scales[g].float()
        )

    return w_float.to(torch.bfloat16)


def _load_int4_dequantized(checkpoint_path, layer_id, E, H, I, group_size, device):
    """Load GPTQ INT4 weights for one layer, dequantize to BF16.

    Returns (w13_deq, w2_deq):
        w13_deq: [E, 2*I, H] BF16  (gate_up fused)
        w2_deq:  [E, H, I]  BF16  (down)
    """
    from safetensors import safe_open

    pack = 8

    # Per-expert GPTQ tensors — load to CPU first, dequant on GPU
    # gate_proj / up_proj → fused into w13
    # down_proj → w2
    expert_data = {}  # expert_id -> {proj -> {attr -> tensor}}

    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".safetensors"):
        st_files = [checkpoint_path]
    else:
        st_files = sorted(
            f for f in os.listdir(checkpoint_path) if f.endswith(".safetensors")
        )
        st_files = [os.path.join(checkpoint_path, f) for f in st_files]

    prefix = f"model.layers.{layer_id}.mlp.experts."
    for st_file in st_files:
        with safe_open(st_file, framework="pt", device=str(device)) as sf:
            for key in sf.keys():
                if not key.startswith(prefix):
                    continue
                m = re.match(
                    r"(\d+)\.(gate_proj|up_proj|down_proj)\.(qweight|scales|qzeros)",
                    key[len(prefix) :],
                )
                if not m:
                    continue
                eid, proj, attr = int(m.group(1)), m.group(2), m.group(3)
                if eid not in expert_data:
                    expert_data[eid] = {}
                if proj not in expert_data[eid]:
                    expert_data[eid][proj] = {}
                expert_data[eid][proj][attr] = sf.get_tensor(key)

    # Dequantize per expert → fused [E, 2*I, H] and [E, H, I]
    w13_deq = torch.zeros(E, 2 * I, H, dtype=torch.bfloat16, device=device)
    w2_deq = torch.zeros(E, H, I, dtype=torch.bfloat16, device=device)

    for eid in range(E):
        if eid not in expert_data:
            continue
        ed = expert_data[eid]

        # gate_proj: qweight [H//8, I], scales [H//gs, I], qzeros [H//gs, I//8]
        if "gate_proj" in ed:
            gp = ed["gate_proj"]
            gate = _dequant_gptq_int4(
                gp["qweight"], gp["scales"], gp["qzeros"], group_size
            )  # [H, I]
            w13_deq[eid, :I, :] = gate.T  # → [I, H]

        # up_proj: same shape
        if "up_proj" in ed:
            up = ed["up_proj"]
            up_w = _dequant_gptq_int4(
                up["qweight"], up["scales"], up["qzeros"], group_size
            )  # [H, I]
            w13_deq[eid, I:, :] = up_w.T  # → [I, H]

        # down_proj: qweight [I//8, H], scales [I//gs, H], qzeros [I//gs, H//8]
        if "down_proj" in ed:
            dp = ed["down_proj"]
            down = _dequant_gptq_int4(
                dp["qweight"], dp["scales"], dp["qzeros"], group_size
            )  # [I, H]
            w2_deq[eid, :, :] = down.T  # → [H, I]

    del expert_data
    return w13_deq, w2_deq


# ---------------------------------------------------------------------------
# Sensitivity measurement
# ---------------------------------------------------------------------------

def _measure_expert_sensitivity(
    moe_input, topk_weights, topk_ids,
    bf16_w13, bf16_w2, int4_w13, int4_w2,
    num_experts, expert_start, expert_end,
):
    """Swap one expert at a time to INT4-dequantized weights, measure error.

    Uses only outplace_fused_experts (Triton BF16 kernel) — no Marlin.
    """
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts

    # All-BF16 baseline
    baseline = outplace_fused_experts(
        moe_input, bf16_w13, bf16_w2, topk_weights, topk_ids
    )
    baseline_norm = torch.norm(baseline).item()

    results = {}

    for e in tqdm(
        range(expert_start, expert_end),
        desc=f"  E[{expert_start}:{expert_end})",
        leave=False,
    ):
        mask = topk_ids == e
        token_count = mask.any(dim=1).sum().item()

        if token_count == 0:
            results[e] = {"sensitivity": 0.0, "token_count": 0}
            continue

        # Swap expert E's weights with dequantized INT4
        orig_w13_e = bf16_w13[e].clone()
        orig_w2_e = bf16_w2[e].clone()
        bf16_w13[e] = int4_w13[e]
        bf16_w2[e] = int4_w2[e]

        # Single BF16 kernel call with swapped weights
        mixed = outplace_fused_experts(
            moe_input, bf16_w13, bf16_w2, topk_weights, topk_ids
        )

        # Restore
        bf16_w13[e] = orig_w13_e
        bf16_w2[e] = orig_w2_e

        sensitivity = torch.norm(mixed - baseline).item()
        results[e] = {"sensitivity": sensitivity, "token_count": token_count}

    return results, baseline_norm


# ---------------------------------------------------------------------------
# Merge mode
# ---------------------------------------------------------------------------

def _merge(args):
    import glob

    partials = sorted(glob.glob(os.path.join(args.out_dir, "partial_rank*.json")))
    if not partials:
        logger.error(f"No partial_rank*.json files in {args.out_dir}")
        sys.exit(1)

    data = []
    for p in partials:
        with open(p) as f:
            data.append(json.load(f))
        logger.info(f"  loaded {p}")

    merged = {k: v for k, v in data[0].items()
              if k not in ("per_layer", "expert_start", "expert_end")}
    merged["num_ranks"] = len(data)
    merged["per_layer"] = {}

    all_layers = sorted(
        {lid for d in data for lid in d.get("per_layer", {})}, key=int
    )
    for lid in all_layers:
        experts = {}
        baseline_norm = None
        for d in data:
            if lid not in d.get("per_layer", {}):
                continue
            layer_data = d["per_layer"][lid]
            baseline_norm = layer_data["baseline_norm"]
            experts.update(layer_data["experts"])
        merged["per_layer"][lid] = {
            "baseline_norm": baseline_norm,
            "experts": experts,
        }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(merged, f, indent=2)

    total_experts = sum(
        len(merged["per_layer"][lid]["experts"]) for lid in all_layers
    )
    logger.info(
        f"Merged {len(data)} partials × {len(all_layers)} layers "
        f"({total_experts} expert entries) → {summary_path}"
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def _run_sweep(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    logger.info(f"Loading model to CPU: {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoConfig

    hf_name = _resolve_hf_path(args.model_path)
    config = AutoConfig.from_pretrained(
        hf_name, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True,
    )
    model.eval()

    E = config.num_experts
    H = config.hidden_size
    I = config.moe_intermediate_size
    top_k = config.num_experts_per_tok
    num_layers = config.num_hidden_layers
    end_layer = num_layers if args.end_layer < 0 else args.end_layer
    expert_start = args.expert_start
    expert_end = E if args.expert_end < 0 else args.expert_end

    logger.info(
        f"  E={E}  H={H}  I={I}  top_k={top_k}  layers={num_layers}  "
        f"experts=[{expert_start},{expert_end})"
    )

    logger.info(f"Loading {args.nsamples} calibration samples (seqlen={args.seqlen})")
    input_ids = _load_calibration_data(hf_name, args.nsamples, args.seqlen, args.seed)
    input_ids = input_ids.to(device)

    logger.info("Computing embeddings and RoPE")
    model.model.embed_tokens.to(device)
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
    model.model.embed_tokens.to("cpu")

    B, S, _ = hidden_states.shape
    position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(S, device=device)

    model.model.rotary_emb.to(device)
    with torch.no_grad():
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    model.model.rotary_emb.to("cpu")

    del input_ids
    torch.cuda.empty_cache()
    logger.info(
        f"Activations on GPU: {hidden_states.shape}  "
        f"({hidden_states.nelement() * 2 / 1e9:.2f} GB)"
    )

    # Output path
    if args.rank >= 0:
        out_path = os.path.join(args.out_dir, f"partial_rank{args.rank}.json")
    else:
        out_path = os.path.join(args.out_dir, "summary.json")

    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
        done = len(all_results.get("per_layer", {}))
        logger.info(f"Resuming — {done} layers already done")
    else:
        all_results = {
            "model": args.model_path,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "total_tokens": args.nsamples * args.seqlen,
            "seed": args.seed,
            "group_size": args.group_size,
            "expert_start": expert_start,
            "expert_end": expert_end,
            "metric": "l2_norm",
            "per_layer": {},
        }

    total_t0 = time.time()

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        in_range = args.start_layer <= layer_idx < end_layer
        already_done = str(layer_idx) in all_results["per_layer"]

        layer.to(device)

        if not in_range or already_done:
            with torch.no_grad():
                out = layer(
                    hidden_states,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                    cache_position=cache_position,
                )
                hidden_states = out if isinstance(out, torch.Tensor) else out[0]
            layer.to("cpu")
            torch.cuda.empty_cache()
            if already_done:
                logger.info(f"Layer {layer_idx}: skip (done)")
            continue

        logger.info(f"=== Layer {layer_idx}/{num_layers} ===")
        t0 = time.time()

        # Capture MoE input
        moe_capture = []

        def _capture_hook(mod, inp):
            moe_capture.append(inp[0].detach().clone())

        hook = layer.mlp.register_forward_pre_hook(_capture_hook)

        with torch.no_grad():
            out = layer(
                hidden_states,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden_states = out if isinstance(out, torch.Tensor) else out[0]

        hook.remove()

        moe_flat = moe_capture[0].reshape(-1, H)

        # Router
        with torch.no_grad():
            raw_logits = F.linear(moe_flat, layer.mlp.gate.weight)
            rw = F.softmax(raw_logits, dim=-1, dtype=torch.float)
            topk_weights, topk_ids = torch.topk(rw, top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(torch.bfloat16)

        # BF16 expert weights (on GPU with the layer)
        bf16_w13 = layer.mlp.experts.gate_up_proj.data  # [E, 2*I, H]
        bf16_w2 = layer.mlp.experts.down_proj.data  # [E, H, I]

        # INT4 → dequantized BF16
        logger.info("  Loading + dequantizing INT4 weights")
        int4_w13, int4_w2 = _load_int4_dequantized(
            args.int4_checkpoint, layer_idx, E, H, I, args.group_size, device
        )

        # Sweep
        logger.info(
            f"  Sweeping experts [{expert_start},{expert_end}) "
            f"({moe_flat.shape[0]} tokens)"
        )
        with torch.no_grad():
            expert_results, baseline_norm = _measure_expert_sensitivity(
                moe_flat, topk_weights, topk_ids,
                bf16_w13, bf16_w2, int4_w13, int4_w2,
                E, expert_start, expert_end,
            )

        del int4_w13, int4_w2, moe_capture, moe_flat, raw_logits
        del topk_weights, topk_ids
        layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        all_results["per_layer"][str(layer_idx)] = {
            "baseline_norm": baseline_norm,
            "experts": {str(e): r for e, r in expert_results.items()},
        }

        elapsed = time.time() - t0
        top5 = sorted(expert_results.items(), key=lambda x: -x[1]["sensitivity"])[:5]
        logger.info(
            f"  Done in {elapsed:.1f}s  |  baseline_norm={baseline_norm:.2f}  |  "
            f"top-5: "
            + ", ".join(f"E{e}={r['sensitivity']:.4f}" for e, r in top5)
        )

        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_t0
    logger.info(f"Done in {total_elapsed:.0f}s.  Results: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Per-expert INT4 sensitivity analysis")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--int4_checkpoint", default="")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--start_layer", type=int, default=0)
    ap.add_argument("--end_layer", type=int, default=-1)
    ap.add_argument("--expert_start", type=int, default=0)
    ap.add_argument("--expert_end", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rank_tag = f"[rank{args.rank}] " if args.rank >= 0 else ""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s {rank_tag}%(levelname)s %(message)s",
    )

    if args.merge:
        _merge(args)
    else:
        _run_sweep(args)


if __name__ == "__main__":
    main()

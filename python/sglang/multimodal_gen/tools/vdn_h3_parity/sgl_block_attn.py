"""SGLang side of the VDN-H3 block-level parity check.

Builds one DiT block's attention (MiniMaxH3Attention with the hybrid modules)
from the materialized VDN-H3 checkpoint, runs it on a synthetic packed input
with the hybrid_window_attn_h3 backend, and dumps inputs + outputs so the VDN
reference stack (other venv) can replay the same block on the same inputs.

    python sgl_block_attn.py --block 0 --out /scratch/parity/blk0.pt [--shape small]
"""
import argparse, json, math, os, sys, time
import torch

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel, model_parallel_is_initialized)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import ensure_distributed_env_defaults
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import AttentionRequirements
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import HybridWindowAttentionH3MetadataBuilder
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention, MiniMaxH3Rope, _rope_cos_sin_cache
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import vdn_h3_layout_from_packed
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from safetensors import safe_open

import os
M = os.environ.get("VDN_H3_MATERIALIZED_DIR", "/scratch/sgl_diffusion_cache/materialized_models/OpenVDN__vdn-minimax-h3-75289d93ebd09d34")
SHAPES = {
    "small": dict(text_len=200, latent_t=12, latent_h=48, latent_w=84, audio_t=100),
    "paper": dict(text_len=300, latent_t=102, latent_h=48, latent_w=84, audio_t=575),
}


def load_block_weights(attn: MiniMaxH3Attention, block: int, device):
    idx = json.load(open(f"{M}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    handles = {}
    def get(name):
        shard = idx[name]
        if shard not in handles:
            handles[shard] = safe_open(f"{M}/transformer/{shard}", "pt", device="cpu")
        return handles[shard].get_tensor(name)
    pre = f"transformer_blocks.{block}.attn."
    renames = {"out_proj.weight": "to_out.0.weight", "q_norm.weight": "norm_q.weight", "k_norm.weight": "norm_k.weight"}
    with torch.no_grad():
        for name, p in attn.named_parameters():
            if name == "qkv_proj.weight":
                w = torch.cat([get(pre + f"to_{c}.weight") for c in "qkv"], dim=0)
            else:
                w = get(pre + renames.get(name, name))
            assert w.shape == p.shape, (name, w.shape, p.shape)
            p.copy_(w.to(p.dtype))
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--shape", default="small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kernel", default="decomposed")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--fp8", action="store_true")
    args = ap.parse_args()
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults(); maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    device = torch.device("cuda")
    shape = SHAPES[args.shape]
    cfg = json.load(open(f"{M}/transformer/config.json"))
    dit = MiniMaxH3DiTConfig(); dit.update_model_arch(cfg); arch = dit.arch_config
    arch.checkpoint_uses_diffusers_layout = True
    hybrid = arch.hybrid_attention

    packed = minimax_h3_packed_sequence(**shape, include_keyframe_cond=False)
    layout = vdn_h3_layout_from_packed(packed, latent_t=shape["latent_t"], latent_h=shape["latent_h"], latent_w=shape["latent_w"])
    seq_len, used = layout.seq_len, layout.used
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    # block inputs are post-norm, post-AdaLN activations: unit-ish scale
    x = torch.randn(seq_len, arch.hidden_size, generator=g).to(torch.bfloat16)
    x[used:] = 0
    x = x.to(device)
    pos = packed["img_position_ids"][None].to(torch.float32).to(device)

    quant = None
    if args.fp8:
        from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
        quant = Fp8Config()
    attn = MiniMaxH3Attention(arch, quant, prefix=f"blocks.{args.block}.attn").to(device)
    load_block_weights(attn, args.block, device)
    if args.fp8:
        for lin in (attn.qkv_proj, attn.out_proj, attn.to_out_linear):
            lin.quant_method.process_weights_after_loading(lin)
        print("fp8:", attn.qkv_proj.weight.dtype, type(attn.qkv_proj.quant_method).__name__)
    rope = MiniMaxH3Rope(arch.rope_inv_freq_len).to(device)
    with torch.no_grad():
        exponents = torch.arange(0, 2 * 16, 2, dtype=torch.float32) / 32
        rope.inv_freq.copy_((1.0 / (10000.0 ** exponents)).to(device))
        rope_cache = (_rope_cos_sin_cache(rope(pos), dtype=torch.bfloat16), torch.arange(seq_len, device=device))
    cu = torch.tensor([0, used, seq_len], dtype=torch.int32, device=device)
    cu_host = (0, used, seq_len)

    results = {}
    with torch.inference_mode():
        # 1. hybrid path
        backend = get_attn_backend(arch.attention_head_dim, torch.bfloat16,
            selected_attention_backend=AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3,
            attention_requirements=AttentionRequirements(packed_varlen=True))
        attn._set_attention_backend(backend)
        meta = HybridWindowAttentionH3MetadataBuilder().build(layout=layout, hybrid=hybrid, device=device, kernel=args.kernel)
        with set_forward_context(current_timestep=0, attn_metadata=meta, forward_batch=None):
            for _ in range(2):
                torch.cuda.synchronize(); t = time.perf_counter()
                out_h = attn(x, rope_cache=rope_cache, cu_seqlens=cu, cu_seqlens_host=cu_host, max_seqlen=used)
                torch.cuda.synchronize(); dt = time.perf_counter() - t
            if args.profile:
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    for _ in range(3):
                        attn(x, rope_cache=rope_cache, cu_seqlens=cu, cu_seqlens_host=cu_host, max_seqlen=used)
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
                print(f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        results["hybrid"] = out_h.cpu(); results["hybrid_ms"] = dt * 1e3
        # 2. dense fa (base + LoRA, no gate, no branch): the M1 smoke
        backend = get_attn_backend(arch.attention_head_dim, torch.bfloat16,
            selected_attention_backend=AttentionBackendEnum.FA,
            attention_requirements=AttentionRequirements(packed_varlen=True))
        attn._set_attention_backend(backend)
        out_d = attn(x, rope_cache=rope_cache, cu_seqlens=cu, cu_seqlens_host=cu_host, max_seqlen=used)
        results["dense"] = out_d.cpu()
    torch.save({"x": x.cpu(), "position_ids": packed["img_position_ids"], "layout": dict(seq_len=seq_len, used=used, text_len=layout.text_len,
                video_start=layout.video_start, num_frames=layout.num_frames, tokens_per_frame=layout.tokens_per_frame,
                frame_height=layout.frame_height, frame_width=layout.frame_width), "block": args.block, "shape": shape,
                "hybrid_config": cfg["hybrid_attention"], **results}, args.out)
    print(f"wrote {args.out}; hybrid attention {results['hybrid_ms']:.1f} ms at used={used}")


if __name__ == "__main__":
    main()

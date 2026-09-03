"""VDN reference side of the block-level parity check (run in /scratch/venvs/vdn).

    python vdn_block_attn.py --inp /scratch/parity/blk0.pt --out /scratch/parity/blk0_vdn.pt
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, "/scratch/src/vdn-minimax-h3")
from safetensors import safe_open
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention, MiniMaxH3RotaryPosEmbed
from src.models.hybrid_attention import HybridAttention
from src.models.sequence_layout import SequenceLayout

import os
M = os.environ.get("VDN_H3_MATERIALIZED_DIR", "/scratch/sgl_diffusion_cache/materialized_models/OpenVDN__vdn-minimax-h3-75289d93ebd09d34")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--softmax-impl", default="ref")
    args = ap.parse_args()
    d = torch.load(args.inp, weights_only=False)
    block = d["block"]; L = d["layout"]; hc = d["hybrid_config"]
    device = torch.device("cuda")
    idx = json.load(open(f"{M}/transformer/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    handles = {}
    def get(name):
        shard = idx[name]
        if shard not in handles: handles[shard] = safe_open(f"{M}/transformer/{shard}", "pt", device="cpu")
        return handles[shard].get_tensor(name)
    orig = MiniMaxH3Attention(hidden_size=5376, heads=56, dim_head=128, qk_norm_eps=1e-5)
    pre = f"transformer_blocks.{block}.attn."
    sd = {k[len(pre):]: get(k) for k in idx if k.startswith(pre) and not any(t in k for t in ("linear_attention", "softmax_gate", "to_out_linear"))}
    missing, unexpected = orig.load_state_dict(sd, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    lin = hc["linear_attention"]; soft = hc["softmax_attention"]
    hybrid = HybridAttention(orig, hidden_size=5376, delta_rule=lin["delta_rule"], radius=soft["radius"], chunk=soft["chunk"],
                             enable_softmax_gate=hc["enable_softmax_gate"], linear_head_dim=lin["linear_head_dim"],
                             softmax_impl=args.softmax_impl, anchor_frames=hc["anchor_frames"], enable_text_state=lin["enable_text_state"],
                             bridge=lin["bridge"], a_fp32=lin["a_fp32"], short_conv=tuple(lin["short_conv"]["targets"]))
    bsd = {k[len(pre):]: get(k) for k in idx if k.startswith(pre) and any(t in k for t in ("linear_attention", "softmax_gate", "to_out_linear"))}
    missing, unexpected = hybrid.load_state_dict(bsd, strict=False)
    missing = [m for m in missing if not m.startswith("orig.")]
    assert not unexpected and not missing, (missing, unexpected)
    hybrid = hybrid.to(device, torch.bfloat16).eval()
    for p in hybrid.parameters():  # VDN's assemble casts every hybrid param to bf16 at inference
        p.data = p.data.to(torch.bfloat16)
    used = L["used"]
    x = d["x"][:used].to(device, torch.bfloat16)
    pos = d["position_ids"][:used].to(device)
    rope = MiniMaxH3RotaryPosEmbed(16).to(device)
    cos, sin = rope(pos)
    layout = SequenceLayout(seq_len=used, video_start=L["video_start"], num_frames=L["num_frames"], tokens_per_frame=L["tokens_per_frame"],
                            frame_height=L["frame_height"], frame_width=L["frame_width"], text_start=0, text_len=L["text_len"])
    hybrid.layout = layout
    out = {}
    with torch.no_grad():
        out["hybrid"] = hybrid(x[None], (cos, sin))[0].cpu()
        # ablation: dense, no gate, no branch == base + LoRAs
        hybrid.enable_softmax_gate = False; hybrid.linear_attention_enabled = False; hybrid.radius = 10_000
        out["dense"] = hybrid(x[None], (cos, sin))[0].cpu()
    torch.save(out, args.out); print("wrote", args.out)


if __name__ == "__main__":
    main()

"""Prove sglang's ATOM SoA packing == aiter's expected byte layout.

Pipeline:
  bf16 [R,512] --sglang quantizer--> NopeFp8RopeBf16Pack --pack_to_atom_soa-->
      (nope_scale[R,512]u8, rope[R,64]bf16)
Compared against aiter's reference _native_to_2buff_for_asm(bf16) and read back
with aiter's _quant_2buff_to_native. Runs on MI355X (gfx950, fp8_e4m3fn).
"""
import sys, torch

torch.manual_seed(0)
dev = "cuda"

# --- sglang side ---
from sglang.kernels.ops.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.kernels.ops.attention.dsv4.atom_soa_pack import (
    pack_to_atom_soa, atom_soa_to_bf16, store_atom_soa,
)

# --- aiter reference helpers (verbatim from op_tests/test_mla_v4_kargpreld.py) ---
from aiter import dtypes
_FP8_MAX = torch.finfo(dtypes.fp8).max
_QD, _QDN, _QDR, _TILE = 512, 448, 64, 64

def _cast_scale_inv_to_ue8m0(t):
    return torch.exp2(torch.ceil(torch.log2(t)))

def _native_to_2buff_for_asm(x):
    nope = x[..., :_QDN]; rope = x[..., _QDN:].contiguous()
    buf = torch.zeros(x.shape[:-1] + (_QD,), dtype=dtypes.fp8, device=x.device)
    npart = buf[..., :_QDN]
    spart = buf[..., _QDN:_QDN + 14].view(dtypes.fp8_e8m0)
    for t in range(_QDN // _TILE):
        s, e = t * _TILE, (t + 1) * _TILE
        tile = nope[..., s:e]
        sc = torch.abs(tile).max(dim=-1).values.float() / _FP8_MAX
        sc = _cast_scale_inv_to_ue8m0(sc)
        spart[..., 2 * t] = sc.to(dtypes.fp8_e8m0)
        spart[..., 2 * t + 1] = sc.to(dtypes.fp8_e8m0)
        npart[..., s:e] = (tile.float() / sc.unsqueeze(-1)).to(dtypes.fp8)
    return buf, rope

def _quant_2buff_to_native(buf, rope):
    npart = buf[..., :_QDN]
    spart = buf[..., _QDN:_QDN + 14].view(dtypes.fp8_e8m0)
    out = torch.empty(buf.shape[:-1] + (_QD,), dtype=dtypes.bf16, device=buf.device)
    for t in range(_QDN // _TILE):
        s, e = t * _TILE, (t + 1) * _TILE
        out[..., s:e] = npart[..., s:e].to(dtypes.bf16) * spart[..., 2 * t : 2 * t + 1].to(dtypes.bf16)
    out[..., _QDN:] = rope
    return out

# ---------------------------------------------------------------------------
R = 4096
x = (torch.randn(R, 512, device=dev) * 3.0).to(torch.bfloat16)

pack = quant_to_nope_fp8_rope_bf16_pack_triton(x)
mine_ns, mine_rope = pack_to_atom_soa(pack)               # [R,512]u8, [R,64]bf16
aiter_ns, aiter_rope = _native_to_2buff_for_asm(x)        # [R,512]fp8, [R,64]bf16
aiter_ns_u8 = aiter_ns.view(torch.uint8)

print(f"shapes: mine_ns={tuple(mine_ns.shape)}{mine_ns.dtype} "
      f"aiter_ns={tuple(aiter_ns.shape)}{aiter_ns.dtype}")

# 1) byte-exact NOPE region [0:448]
nope_eq = (mine_ns[:, :448] == aiter_ns_u8[:, :448]).float().mean().item()
# 2) byte-exact SCALE region [448:462] (dup)
sc_eq = (mine_ns[:, 448:462] == aiter_ns_u8[:, 448:462]).float().mean().item()
# 3) scale duplication correctness (even==odd)
dup_ok = (mine_ns[:, 448:462:2] == mine_ns[:, 449:462:2]).all().item()
# 4) pad zero
pad_ok = (mine_ns[:, 462:512] == 0).all().item()
# 5) rope byte-exact
rope_eq = (mine_rope.view(torch.uint8) == aiter_rope.view(torch.uint8)).float().mean().item()

print(f"[byte] nope_match={nope_eq*100:.4f}%  scale_match={sc_eq*100:.4f}%  "
      f"rope_match={rope_eq*100:.4f}%  dup_ok={dup_ok}  pad_zero={pad_ok}")

# 6) feed MY buffers into AITER's reader -> reconstruct, compare vs original
recon_by_aiter = _quant_2buff_to_native(mine_ns.view(dtypes.fp8), mine_rope)
recon_by_mine = atom_soa_to_bf16(mine_ns, mine_rope)
aiter_full = _quant_2buff_to_native(aiter_ns, aiter_rope)

def stats(a, b, name):
    a = a.float(); b = b.float()
    err = (a - b)
    rel = err.norm() / (b.norm() + 1e-9)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(f"[num] {name:32s} rel_l2={rel.item():.3e}  cos={cos.item():.7f}  "
          f"max_abs={err.abs().max().item():.3e}")

stats(recon_by_aiter, aiter_full, "aiter-reads-MINE vs aiter-full")   # ~0 if bytes match
stats(recon_by_mine, aiter_full, "MINE-dequant vs aiter-full")
stats(recon_by_aiter, x, "aiter-reads-MINE vs original(bf16)")        # fp8 quant err

# 7) paged store round-trip
page_size = 8
num_pages = (R + page_size - 1) // page_size + 1
ns_buf = torch.zeros(num_pages, page_size * 512, dtype=torch.uint8, device=dev)
rp_buf = torch.zeros(num_pages, page_size * 64, dtype=torch.bfloat16, device=dev)
loc = torch.arange(R, device=dev, dtype=torch.int64)
store_atom_soa(ns_buf, rp_buf, loc, pack, page_size)
ns_flat = ns_buf.view(num_pages, page_size, 512)[loc // page_size, loc % page_size]
rp_flat = rp_buf.view(num_pages, page_size, 64)[loc // page_size, loc % page_size]
paged_ok = (ns_flat == mine_ns).all().item() and (rp_flat.view(torch.uint8) == mine_rope.view(torch.uint8)).all().item()
print(f"[paged] store/gather round-trip exact = {paged_ok}")

ok = (nope_eq > 0.999 and sc_eq > 0.999 and rope_eq > 0.999 and dup_ok and pad_ok and paged_ok)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

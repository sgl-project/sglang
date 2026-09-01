"""Integration test: runtime_fp8 (aiter fp8 kernels) vs runtime (bf16 Triton/OPUS)
golden, driven through the SAME unified_kv ragged index streams.

Proves the two aiter kernels run correctly end-to-end through sglang's SoA store +
dispatch code: decode -> mla_decode_fwd_v4_nm (#3112), prefill ->
pa_sparse_prefill_fp8_opus (#3751). Difference vs the bf16 golden = fp8 quant noise
only (both q and kv are fp8 in the aiter path, matching deployment).

MI355X (gfx950). head_dim = 448 nope + 64 rope = 512; v_head_dim = 512.
"""
import sys, torch, torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
H, D, VD = 16, 512, 512

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime, runtime_fp8


def build_pool_from_bf16(src_kv):
    """src_kv [rows,512] bf16 -> (nope_buf[rows,512]u8, rope_buf[rows,64]bf16)."""
    ns, rp = runtime_fp8.quant_to_soa(src_kv)
    return ns.contiguous(), rp.contiguous()


def ragged(lengths):
    return F.pad(torch.cumsum(lengths, 0, dtype=torch.int32), (1, 0))


def stats(name, a, b):
    a = a.float(); b = b.float()
    rel = (a - b).norm() / (b.norm() + 1e-9)
    cos = F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
    snr = 20 * torch.log10(b.norm() / ((a - b).norm() + 1e-9)).item()
    print(f"[{name}] rel_l2={rel.item():.4e} cos={cos:.6f} SNR={snr:5.1f}dB "
          f"max_abs={(a-b).abs().max().item():.3e}")
    return cos, rel.item()


ok = True

# =========================== DECODE ===========================
print("########## DECODE: mla_decode_fwd_v4_nm vs bf16 golden ##########")
T = 4
rows = 4096
src_kv = (torch.randn(rows, D, device=dev) * 2.0).to(torch.bfloat16)
nope_buf, rope_buf = build_pool_from_bf16(src_kv)

q = (torch.randn(T, H, D, device=dev) * 1.5).to(torch.bfloat16)
sink = (torch.randn(H, device=dev) * 0.1).to(torch.float32)
scale = D ** -0.5

# each token attends to a distinct ragged set of unified rows
lens = torch.tensor([128, 256, 192, 64], device=dev, dtype=torch.int32)[:T]
kv_indptr = ragged(lens)
kv_indices = torch.cat([
    torch.randint(0, rows, (int(l),), device=dev, dtype=torch.int32) for l in lens
])

out_fp8 = runtime_fp8.decode(
    q=q, nope_buf=nope_buf, rope_buf=rope_buf,
    kv_indices=kv_indices, kv_indptr=kv_indptr,
    attn_sink=sink, softmax_scale=scale,
)
out_bf16 = runtime.decode(
    q=q, unified_kv=src_kv,
    kv_indices=kv_indices, kv_indptr=kv_indptr,
    attn_sink=sink, softmax_scale=scale,
)
print(f"shapes: fp8={tuple(out_fp8.shape)} bf16={tuple(out_bf16.shape)}")
c, r = stats("decode", out_fp8, out_bf16)
ok = ok and (c > 0.99 and r < 0.08)

# =========================== PREFILL ===========================
print("########## PREFILL: pa_sparse_prefill_fp8_opus vs bf16 golden ##########")
Tp = 4
src_pref = (torch.randn(rows, D, device=dev) * 2.0).to(torch.bfloat16)
nope_p, rope_p = build_pool_from_bf16(src_pref)
qp = (torch.randn(Tp, H, D, device=dev) * 1.5).to(torch.bfloat16)

# prefix (paged) region
plen = torch.tensor([128, 64, 192, 128], device=dev, dtype=torch.int32)[:Tp]
pidptr = ragged(plen)
pidx = torch.cat([
    torch.randint(0, rows, (int(l),), device=dev, dtype=torch.int32) for l in plen
])
# extend region (this fwd's fresh K)
ext_tokens = 64
kv_ext = (torch.randn(ext_tokens, D, device=dev) * 2.0).to(torch.bfloat16)
elen = torch.tensor([16, 16, 16, 16], device=dev, dtype=torch.int32)[:Tp]
eidptr = ragged(elen)
eidx = torch.cat([
    torch.randint(0, ext_tokens, (int(l),), device=dev, dtype=torch.int32) for l in elen
])

out_pf8 = runtime_fp8.prefill(
    q=qp, nope_buf=nope_p, rope_buf=rope_p,
    kv_indices_prefix=pidx, kv_indptr_prefix=pidptr,
    kv_extend=kv_ext, kv_indices_extend=eidx, kv_indptr_extend=eidptr,
    attn_sink=sink, softmax_scale=scale,
)
out_pbf = runtime.prefill(
    q=qp, unified_kv=src_pref,
    kv_indices_prefix=pidx, kv_indptr_prefix=pidptr,
    kv_extend=kv_ext, kv_indices_extend=eidx, kv_indptr_extend=eidptr,
    attn_sink=sink, softmax_scale=scale,
)
print(f"shapes: fp8={tuple(out_pf8.shape)} bf16={tuple(out_pbf.shape)}")
c2, r2 = stats("prefill", out_pf8, out_pbf)
ok = ok and (c2 > 0.99 and r2 < 0.08)

print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)

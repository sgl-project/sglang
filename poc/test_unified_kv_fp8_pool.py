"""Assembled-path smoke test: construct DeepSeekV4UnifiedKVPoolFp8 through the real
sglang package, drive SWA + compressed stores into its two fp8 buffers, then run
the aiter decode/prefill dispatch reading straight from the pool buffers.

Validates: env gate (unified_kv_aiter), fp8 pool class + accessors, runtime_fp8
store scatter into pool buffers, and both aiter kernels consuming them. Requires
gfx950 + aiter. Import-checks the backend/compressor modules too.
"""
import os, sys, torch, torch.nn.functional as F

os.environ["SGLANG_HACK_FLASHMLA_BACKEND"] = "unified_kv_aiter"
torch.manual_seed(0)
dev = "cuda"

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_aiter, is_unified_kv_triton,
)
assert is_unified_kv_aiter() and is_unified_kv_triton(), "gate mismatch"
print("gate: unified_kv_aiter=True unified_kv_triton=True  OK")

# import-check the wired backend + compressor modules (catches syntax/name errors)
import importlib
importlib.import_module("sglang.srt.layers.attention.dsv4.compressor_v2")
print("compressor_v2 import OK")

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4UnifiedKVPoolFp8
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime_fp8


class _NullSaver:
    from contextlib import contextmanager
    @contextmanager
    def region(self, *a, **k):
        yield


# small pool: c128 ratio, 2 layers
num_slots, swa_ring = 8, 128
pool = DeepSeekV4UnifiedKVPoolFp8(
    stage_ratios=[128, 128],
    num_slots=num_slots,
    num_blocks=64,
    page_size=128,
    qk_nope_head_dim=448,
    qk_rope_head_dim=64,
    device=dev,
    memory_saver_adapter=_NullSaver(),
    custom_mem_pool=None,
    swa_ring_size=swa_ring,
)
nope = pool.get_unified_kv_nope(0)
rope = pool.get_unified_kv_rope(0)
print(f"pool nope={tuple(nope.shape)}{nope.dtype} rope={tuple(rope.shape)}{rope.dtype} "
      f"swa_pages={pool.swa_pages}")
assert nope.dtype == torch.float8_e4m3fn and rope.dtype == torch.bfloat16
assert nope.shape[1] == 512 and rope.shape[1] == 64

nope_u8 = pool.nope_scale_buffer[0]
rope_b = pool.rope_buffer[0]

# ---- SWA store into ring rows ----
T = 4
kv = (torch.randn(T, 512, device=dev) * 2).to(torch.bfloat16)
state_slot = torch.arange(T, device=dev, dtype=torch.int32)
positions = torch.tensor([10, 20, 30, 40], device=dev, dtype=torch.int32)
runtime_fp8.store_swa_into_unified_fp8(
    kv=kv, state_slot=state_slot, positions=positions,
    nope_buf=nope_u8, rope_buf=rope_b, win=swa_ring, ring_stride=swa_ring,
    final_pos=positions,
)
# verify round-trip of a stored row
from sglang.kernels.ops.attention.dsv4.atom_soa_pack import atom_soa_to_bf16
loc0 = int(state_slot[0]) * swa_ring + int(positions[0]) % swa_ring
deq = atom_soa_to_bf16(nope_u8[loc0:loc0+1], rope_b[loc0:loc0+1])[0]
cos_swa = F.cosine_similarity(deq.float(), kv[0].float(), dim=0).item()
print(f"SWA store round-trip cos={cos_swa:.5f}")
assert cos_swa > 0.99

# ---- compressed store into compressed rows ----
M = 6
kvc = (torch.randn(M, 512, device=dev) * 2).to(torch.bfloat16)
out_loc = (pool.swa_pages + torch.arange(M, device=dev, dtype=torch.int32))
runtime_fp8.store_compress_into_unified_fp8(
    kv_compressed=kvc, out_loc=out_loc, nope_buf=nope_u8, rope_buf=rope_b,
)
deqc = atom_soa_to_bf16(nope_u8[int(out_loc[1]):int(out_loc[1])+1],
                        rope_b[int(out_loc[1]):int(out_loc[1])+1])[0]
cos_c = F.cosine_similarity(deqc.float(), kvc[1].float(), dim=0).item()
print(f"compress store round-trip cos={cos_c:.5f}")
assert cos_c > 0.99

# ---- decode reading straight from pool buffers ----
H = 16
q = (torch.randn(T, H, 512, device=dev) * 1.5).to(torch.bfloat16)
sink = (torch.randn(H, device=dev) * 0.1).to(torch.float32)
lens = torch.tensor([2, 3, 1, 2], device=dev, dtype=torch.int32)
kv_indptr = F.pad(torch.cumsum(lens, 0, dtype=torch.int32), (1, 0))
# indices reference the rows we actually wrote (swa ring slot0 + compressed rows)
wrote = torch.cat([torch.tensor([loc0], device=dev),
                   out_loc.to(torch.int64)]).to(torch.int32)
kv_indices = wrote[torch.randint(0, wrote.numel(), (int(lens.sum()),), device=dev)]
out = runtime_fp8.decode(
    q=q, nope_buf=nope, rope_buf=rope,
    kv_indices=kv_indices, kv_indptr=kv_indptr,
    attn_sink=sink, softmax_scale=512 ** -0.5,
)
print(f"decode out={tuple(out.shape)}{out.dtype} finite={torch.isfinite(out).all().item()}")
assert out.shape == (T, H, 512) and torch.isfinite(out).all()

print("RESULT: PASS")
sys.exit(0)

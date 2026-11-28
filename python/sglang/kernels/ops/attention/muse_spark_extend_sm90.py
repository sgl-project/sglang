"""SM90 extend / mixed attention (fused QK-norm, RoPE, KV write, FP8 stream-K, norm gate), NVRTC source
extend_attention.cuh; SGLang numerics except rare 1-ulp E4M3 differences in new K rows."""

import functools
import struct

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

_NS, _STG, _CW = 8, 16384, 3
_SMEM = {"pa_fwd": _NS * _STG + 1024 + 4096 * _CW, "pa_cmb": 2 * 9 * 128 * 16}
_SLOT = 128 * 36 * 4


class _Mod:
    def __init__(self, dev):
        self.dev = dev
        mod, self.G = cu.device_module(
            dev, "extend_attention.cu", cu.read_source("extend_attention.cuh")
        )
        self.ps = torch.empty(self.G * 2 * _CW * _SLOT, dtype=torch.uint8, device=dev)
        self.f = {
            n: cu.function(mod, n, _SMEM.get(n, 0))
            for n in ("pa_post", "pa_pack", "pa_fwd", "pa_cmb")
        }

    def launch(self, name, grid, block, args, cl=0):
        # PDL: pa_cmb reads the plan early (pa_fwd triggers after its wait); others wait first
        cu.launch(
            self.f[name],
            grid,
            block,
            _SMEM.get(name, 0),
            torch.cuda.current_stream(self.dev).cuda_stream,
            args,
            [cu.pdl()] + ([cu.cluster_dim(cl)] if cl else []),
        )


_mod = functools.cache(_Mod)


def extend_attention(
    qkv,
    cache_loc,
    k_cache,
    v_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    kv_bound,
    sm_scale,
    qk_scale_factor,
    eps,
    out,
    window,
    cos_sin,
    positions,
    push=None,
    last_only=False,
):
    """out [T, 1024] = norm-gated attention, new K / V -> cache_loc; kv_bound >= packed keys; window includes the query."""
    T, B, dev = qkv.shape[0], cache_seqlens.shape[0], qkv.device
    m = _mod(dev)
    G = m.G
    um = 2 * (T // 24 + B) + 2
    nkt = (kv_bound + T) // 128 + B
    tb = qkv.new_empty(
        4 + ((5 * B + 4) & ~3) + 4 * um + 2 * (G + 1) + 2 * um + 4 * G + 12,
        dtype=torch.int32,
    )
    q8 = qkv.new_empty(T * 1024, dtype=torch.uint8)
    # packed K | V^T: 32 KB per 128 keys read (~256 B per KV token)
    kv = qkv.new_empty(0 if last_only else nkt * 32768, dtype=torch.uint8)
    args = b"".join(
        struct.pack("Q", t.data_ptr())
        for t in (
            qkv,
            out,
            q8,
            k_cache,
            v_cache,
            kv,
            m.ps,
            cache_loc,
            page_table,
            cu_seqlens_q,
            cache_seqlens,
            tb,
        )
    )
    rope = cos_sin is not None
    args += struct.pack(
        "qiiiiffQQi",
        page_table.stride(0),
        B,
        T,
        G,
        um,
        cu.scale_log2(sm_scale, qk_scale_factor),
        eps,
        positions.data_ptr() if rope else 0,
        cos_sin.data_ptr() if rope else 0,
        window,
    )
    table, rows, ld, col, base, tz = push or (None, 1, 0, 0, 0, T)
    tz = max(tz, T)
    args += struct.pack(
        "=4xQqiiii", table.data_ptr() if push else 0, ld, rows, col, base, tz
    )
    m.launch("pa_post", (tz + 7) // 8 + 1, 256, args)
    if last_only:
        last = cu_seqlens_q[1:].long() - 1
        return extend_attention(
            qkv[last],
            cache_loc[last],
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            torch.arange(B + 1, dtype=cu_seqlens_q.dtype, device=dev),
            kv_bound,
            sm_scale,
            qk_scale_factor,
            eps,
            out,
            window,
            cos_sin,
            positions[last] if rope else None,
        )
    m.launch("pa_pack", min(2 * nkt, 8 * G), 256, args)
    m.launch("pa_fwd", G, 128 * (_CW + 1), args, 2)
    m.launch("pa_cmb", (G - 2) * _CW, 128, args)
    return out

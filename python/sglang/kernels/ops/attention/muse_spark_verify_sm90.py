"""SM90 DFlash verify attention (paged FP8 KV; fused QK-norm, RoPE, KV write, norm gate), NVRTC source
verify_attention.cuh. The verify tile follows https://github.com/gauravjain14/muse-okc-submissions (Apache-2.0)."""

import functools
import struct

import torch

from sglang.kernels.jit import muse_spark_cuda as cu

_XR = 272
# hard limit: one warp scans the whole batch
MAX_BATCH = 32
# (max batch = co-resident C-CTA clusters on H200, C, T 64-key stages per part)
_VS_CFG = ((15, 8, 5), (17, 6, 6), (22, 5, 7), (30, 4, 9), (32, 3, 11))


def _vs_smem(C, T):
    rcv = 4 * (C - 1) * (-(-32 // C)) * _XR if C != 3 else 0
    return (
        1024 + T * 16384 + 8 * 4608 + 256 * 8 + 8192 + 8 * T + 8 + 64 + 128 + rcv + 16
    )


_VF_SMEM = (
    6 * 16384
    + 6 * 8192
    + 9 * 256 * 16
    + 16384
    + 6 * 256
    + 34 * 16
    + 16 * 6
    + 16 * 6
    + 1024
)
_VF_SLOT = 128 * 64 * 4 + 128 * 8


class _Mod:
    def __init__(self, dev):
        self.dev = dev
        src = cu.read_source("verify_attention.cuh")
        (self.mod, self.nsm), (self.modp, _) = (
            cu.device_module(dev, "verify_attention.cu", f"#define PS {i}\n" + src)
            for i in (0, 1)
        )
        self.ws = torch.empty(
            (self.nsm + MAX_BATCH) * _VF_SLOT, dtype=torch.uint8, device=dev
        )
        for n, s, *c in (
            [("vf_main", _VF_SMEM)]
            + [(f"vf_comb{1 << i}", 120 * 1024) for i in range(6)]
            + [(f"vs_c{C}", _vs_smem(C, T), C) for _, C, T in _VS_CFG]
        ):
            (
                self.func(n, s, False),
                self.func(n, s, True),
                c and self.max_clusters(n, s, *c),
            )

    @functools.cache
    def func(self, name, smem, ps=False):
        return cu.function(
            self.modp if ps else self.mod, name, smem, nonportable_cluster=1
        )

    @functools.cache
    def max_clusters(self, name, smem, cl):
        err, n = cu.driver().cuOccupancyMaxActiveClusters(
            self.func(name, smem),
            cu.launch_config(cl, 512, smem, None, [cu.cluster_dim(cl)]),
        )
        return int(n) if int(err) == 0 else 0

    def launch(self, name, grid, block, smem, args, cl=1, ps=False):
        attrs = ([cu.cluster_dim(cl)] if cl > 1 else []) + [cu.pdl()]
        cu.launch(
            self.func(name, smem, ps),
            grid,
            block,
            smem,
            torch._C._cuda_getCurrentRawStream(self.dev.index),
            args,
            attrs,
        )


_mod = functools.cache(_Mod)
_FREQ = {}


@functools.cache
def _vs_cfg(dev, B):
    for _, C, T in (r for r in _VS_CFG if B <= r[0]):
        sel = f"vs_c{C}", C, _vs_smem(C, T)
        if _mod(dev).max_clusters(sel[0], sel[2], C) >= B:
            return sel
    return sel


def _args(qkv, kv, out, cfg, tree, fq, ws=None, G=0, push=None):
    a = b"".join(struct.pack("Q", t.data_ptr()) for t in (qkv, *kv, out))
    a += struct.pack(
        "qiif4xQii",
        qkv.stride(-2),
        kv[4].shape[1],
        cfg["window"],
        cu.scale_log2(cfg["sm_scale"], cfg["qk_scale_factor"]),
        0 if ws is None else ws.data_ptr(),
        qkv.shape[0],
        G,
    ) + (fq or bytes(128))
    a += b"".join(struct.pack("Q", 0 if t is None else t.data_ptr()) for t in tree)
    return a + (
        struct.pack("Qi4xqii", push[0].data_ptr(), *push[1:]) if push else bytes(32)
    )


def verify_attention(
    qkv,
    cache_seqlens,
    cache_loc,
    k_cache,
    v_cache,
    page_table,
    cfg,
    tree=(None, None),
    out=None,
    push=None,
):
    """[B, 8, 1024] bf16 out (``out`` if given), new K / V written at cache_loc."""
    B, dev, kv = (
        qkv.shape[0],
        qkv.device,
        (cache_seqlens, cache_loc, k_cache, v_cache, page_table),
    )
    m, ps = _mod(dev), push is not None
    out = (
        torch.empty((B, 8, 16 * 64), dtype=torch.bfloat16, device=dev)
        if out is None
        else out
    )
    if not cfg["window"]:
        G = m.nsm
        args = _args(qkv, kv, out, cfg, tree, b"", m.ws, G, push)
        m.launch("vf_main", G, 640, _VF_SMEM, args, ps=ps)
        npg = 1
        while npg < 32 and 8 * npg < G // B + 2:
            npg *= 2
        m.launch(
            f"vf_comb{32 // npg}", (128 * npg // 32, B), 512, 120 * 1024, args, ps=ps
        )
        return out
    name, C, smem = _vs_cfg(dev, B)
    fq = _FREQ.get(cfg["rope_theta"])
    if fq is None:
        inv = 1.0 / (
            cfg["rope_theta"]
            ** (torch.arange(0, 64, 2, dtype=torch.float, device=dev) / 64)
        )
        fq = _FREQ[cfg["rope_theta"]] = struct.pack("32f", *inv.tolist())
    m.launch(
        name,
        B * C,
        512,
        smem,
        _args(qkv, kv, out, cfg, tree, fq, push=push),
        cl=C,
        ps=ps,
    )
    return out

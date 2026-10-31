"""Kimi-K3 MLA extend-with-prefix K/V assembly.

The aiter backend's extend over a cached prefix rebuilds per-head K/V for all
prefix + new tokens of an MLA layer (index_select of the FP8 latent cache,
strided casts, kv_b_proj GEMM, torch.cat of [k_nope | broadcast k_pe]) before
the opus gqa_d192_v128 varlen attention. ``k3_mla_prefix_kv`` builds the same
K/V without the casts / concat. This checks, against a verbatim copy of the
original chain:

K / V and the attention output are bit-identical, both for the kernel alone
and through AiterAttnBackend.forward_extend.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

H = 12  # K3 heads per rank (TP8)
DN, DR, DV, L = 128, 64, 128, 512  # qk_nope, rope, v, kv_lora
SCALE = (DN + DR) ** -0.5


class _KvB(torch.nn.Module):
    """kv_b_proj stand-in that runs the production linear method."""

    def __init__(self, weight):
        super().__init__()
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, None), None


def make_case(prefix, extend, bs, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    seq = prefix + extend
    n = seq * bs
    pool = n + 4096
    cache = (torch.randn(pool, 1, L + DR, device=device, generator=g) * 0.5).to(
        torch.float8_e4m3fn
    )
    kv_indices = torch.randperm(pool, device=device, generator=g)[:n].to(torch.int32)
    qo_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32) * extend
    kv_indptr = torch.arange(bs + 1, device=device, dtype=torch.int32) * seq
    q = torch.randn(bs * extend, H, DN + DR, device=device, generator=g).to(
        torch.bfloat16
    )
    w = (torch.randn(H * (DN + DV), L, device=device, generator=g) * 0.05).to(
        torch.bfloat16
    )
    return dict(
        cache=cache,
        kv_indices=kv_indices,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        q=q,
        kv_b=_KvB(w),
        max_q=extend,
        max_kv=seq,
    )


def old_kv(c):
    """Verbatim copy of aiter_backend.forward_extend's prefix branch."""
    K_Buffer = torch.index_select(c["cache"], 0, c["kv_indices"])
    kvc, k_pe = torch.split(K_Buffer, [L, DR], dim=-1)
    kvc = kvc.to(torch.bfloat16)
    k_pe = k_pe.to(torch.bfloat16)
    kv = c["kv_b"](kvc.contiguous())[0]
    kv = kv.view(-1, H, DN + DV)
    k, v = torch.split(kv, [DN, DV], dim=-1)
    k = torch.cat(
        [k, torch.broadcast_to(k_pe, (k_pe.shape[0], H, k_pe.shape[2]))], dim=-1
    )
    return k, v


def new_kv(c):
    from sglang.kernels.ops.attention.k3_mla_prefix_nocat import k3_mla_prefix_kv

    return k3_mla_prefix_kv(c["cache"], c["kv_indices"], c["kv_b"], H, DN, DV, L, DR)


def attn(c, k, v):
    from aiter import flash_attn_varlen_func

    return flash_attn_varlen_func(
        c["q"],
        k,
        v,
        c["qo_indptr"],
        c["kv_indptr"],
        c["max_q"],
        c["max_kv"],
        softmax_scale=SCALE,
        causal=True,
    )


def old_layer(c):
    return attn(c, *old_kv(c))


def new_layer(c):
    return attn(c, *new_kv(c))


class TestK3MlaPrefixNocat(CustomTestCase):
    CASES = [(1000, 200, 1), (3000, 512, 3), (130, 70, 2)]

    def test_bitwise(self):
        for prefix, extend, bs in self.CASES:
            c = make_case(prefix, extend, bs)
            k0, v0 = old_kv(c)
            k1, v1 = new_kv(c)
            self.assertTrue(torch.equal(k0, k1), (prefix, extend, bs))
            self.assertTrue(torch.equal(v0, v1), (prefix, extend, bs))
            o0, o1 = attn(c, k0, v0), attn(c, k1, v1)
            self.assertTrue(torch.equal(o0, o1), (prefix, extend, bs))

    def test_gather_bitwise_bf16_cache(self):
        from sglang.kernels.ops.attention.k3_mla_prefix_nocat import _gather_dequant

        c = make_case(300, 50, 2)
        cache = c["cache"].to(torch.bfloat16)
        idx = c["kv_indices"].long()
        n = idx.shape[0]
        kvc = torch.empty(n, L, dtype=torch.bfloat16, device="cuda")
        kpe = torch.empty(n, H, DN + DR, dtype=torch.bfloat16, device="cuda")
        _gather_dequant(cache, idx, kvc, kpe[:, :, DN:], L, DR)
        ref = cache[idx, 0]
        self.assertTrue(torch.equal(kvc, ref[:, :L]))
        self.assertTrue(torch.equal(kpe[:, :, DN:], ref[:, None, L:].expand(n, H, DR)))


def backend_extend(c, nocat):
    """Drive AiterAttnBackend.forward_extend's MLA extend-with-prefix branch
    (flag off = original code, on = k3_mla_prefix_kv) on a minimal backend."""
    from types import SimpleNamespace

    import sglang.srt.layers.attention.aiter_backend as ab
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    be = object.__new__(ab.AiterAttnBackend)
    cache = c["cache"]
    bs = c["qo_indptr"].shape[0] - 1
    extend, seq = c["max_q"], c["max_kv"]
    be.kv_cache_dtype = cache.dtype
    be.k_scale = be.v_scale = None
    be.use_mla = True
    be.dcp_world_size = 1
    be.use_fp8_prefill_attn = False
    be.head_pad_mode = "zero"
    be.k3_mla_prefix_nocat = nocat
    be.forward_metadata = SimpleNamespace(
        k3_fp8_prefill_plan=None,
        max_q_len=extend,
        max_kv_len=seq,
        kv_indptr=c["kv_indptr"],
        kv_indices=c["kv_indices"],
        qo_indptr=c["qo_indptr"],
    )
    be.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda i: cache, get_value_buffer=lambda i: cache[..., :L]
    )
    layer = SimpleNamespace(
        logit_cap=0.0,
        is_cross_attention=False,
        k_scale=None,
        v_scale=None,
        layer_id=0,
        qk_head_dim=DN + DR,
        v_head_dim=DV,
        tp_k_head_num=H,
        tp_q_head_num=H,
        kv_b_proj=c["kv_b"],
        scaling=SCALE,
    )
    fb = SimpleNamespace(
        attn_attend_prefix_cache=False,
        out_cache_loc=None,
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens_cpu=[seq - extend] * bs,
        extend_prefix_lens=torch.full((bs,), seq - extend, device="cuda"),
        extend_seq_lens=torch.full((bs,), extend, device="cuda"),
        mha_return_lse=False,
    )
    q = c["q"]
    k = torch.zeros(q.shape[0], H, DN + DR, dtype=q.dtype, device=q.device)
    v = torch.zeros(q.shape[0], H, DV, dtype=q.dtype, device=q.device)
    return be.forward_extend(q, k, v, layer, fb, save_kv_cache=False)


class TestK3MlaPrefixNocatBackend(CustomTestCase):
    def test_backend_branch(self):
        for prefix, extend, bs in [(1000, 200, 1), (3000, 512, 3)]:
            c = make_case(prefix, extend, bs)
            o_ref = old_layer(c)
            o_off = backend_extend(c, False)
            o_on = backend_extend(c, True)
            self.assertTrue(torch.equal(o_ref, o_off))
            self.assertTrue(torch.equal(o_off, o_on))


if __name__ == "__main__":
    unittest.main()

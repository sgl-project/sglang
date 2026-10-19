"""CPU tests for the v0.5.22 CORAL hooks (coral/sitecustomize.py): which kernel each token count
routes to. Kernels are stubs (Triton has no macOS build); kernel numerics are checked on a GPU.

    python -m pytest coral/test_dispatch.py -q      (needs torch, CPU is fine)
"""
import importlib, os, sys, types
from types import SimpleNamespace as NS

import pytest
import torch

PATCHES = os.path.dirname(os.path.abspath(__file__))


class _Any:
    """Permissive stand-in for triton / triton.language: any attribute, call, index or decorator works."""

    def __getattr__(self, k):
        return _Any()

    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]  # bare decorator: @triton.jit
        return _Any()

    def __getitem__(self, k):
        return _Any()


def _stub_triton():
    t = types.ModuleType("triton")
    t.__getattr__ = lambda k: _Any()
    t.jit = lambda f=None, **k: f if f is not None else (lambda g: g)
    t.cdiv = lambda a, b: -(-a // b)
    t.next_power_of_2 = lambda n: 1 << (int(n) - 1).bit_length()
    tl = types.ModuleType("triton.language")
    tl.__getattr__ = lambda k: _Any()
    t.language = tl
    sys.modules["triton"], sys.modules["triton.language"] = t, tl


def load(monkeypatch, **env):
    """Fresh import of the fork's sitecustomize under the given CORAL_* env (it reads env at import time)."""
    for k in [k for k in os.environ if k.startswith("CORAL_")]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    _stub_triton()
    monkeypatch.syspath_prepend(PATCHES)
    for m in ("sitecustomize", "coral_kernels", "coral_mma"):
        sys.modules.pop(m, None)
    sc = importlib.import_module("sitecustomize")
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_Hook"]  # don't hook this process's imports
    assert sc.__file__.startswith(PATCHES), sc.__file__
    return sc


class Rec:
    """Records calls; returns `ret(*a, **k)`."""

    def __init__(self, ret=None):
        self.calls, self.ret = [], ret

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.ret(*a, **k) if self.ret else None


# ---------------------------------------------------------------------------------------------------------------
# GDN verify kernel relaunch (CORAL_GDN_VERIFY_BV): v0.5.21 grid layout
# ---------------------------------------------------------------------------------------------------------------
class _Kern:
    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *a, **kw: self.launches.append((grid, kw))


@pytest.mark.parametrize("split, grid_in, grid_out", [
    (True, (4, 1, 32), (32, 1, 32)),    # v0.5.21 CUDA: (NV, N, HV), N = 1 request
    (True, (4, 3, 32), (32, 3, 32)),    # 3 requests: N must survive
    (False, (1, 4, 32), (1, 32, 32)),   # v0.5.18 layout: (NK, NV, N*HV)
])
def test_gdn_verify_bv_grid(monkeypatch, split, grid_in, grid_out):
    sc = load(monkeypatch, CORAL_GDN_VERIFY_BV=4)
    k = _Kern()
    mod = NS(fused_sigmoid_gating_delta_rule_update_kernel=k)
    sc._install_gdn_verify_bv(mod)
    kw = dict(V=128, BV=32)
    if split:
        kw["SPLIT_N_HV_GRID"] = True
    mod.fused_sigmoid_gating_delta_rule_update_kernel[grid_in](**kw)
    (grid, out_kw), = k.launches
    assert grid == grid_out and out_kw["BV"] == 4


# ---------------------------------------------------------------------------------------------------------------
# Dense FP8 linears (CORAL_FP8GEMV + CORAL_FP8GEMV_MMA): route per token count
# ---------------------------------------------------------------------------------------------------------------
def _fp8_env(max_m):
    return dict(CORAL_FP8GEMV=1, CORAL_FP8GEMV_MMA=1, CORAL_FP8GEMV_NS=1, CORAL_MAX_M=max_m)


def _install_fp8(monkeypatch, sc):
    import coral_kernels as ck, coral_mma
    mma = Rec(lambda x, w, s, warps, rt: ("mma", x.shape[0], warps, rt))
    monkeypatch.setattr(coral_mma, "_load", lambda: None)
    monkeypatch.setattr(coral_mma, "w8a16_mma", mma)
    monkeypatch.setattr(ck, "fp8w_gemv_ns", lambda x, *a: ("ns", x.shape[0]))
    monkeypatch.setattr(ck, "fp8w_gemv_splitk", lambda x, *a: ("splitk", x.shape[0]))

    class Fp8LinearMethod:
        block_quant = True

        def apply(self, layer, x, bias=None, *a, **k):
            return ("stock", x.shape[0])

    sc._install_fp8gemv(NS(Fp8LinearMethod=Fp8LinearMethod))
    return Fp8LinearMethod()


def _layer(N, K):
    return NS(weight=torch.empty(N, K, dtype=torch.float8_e4m3fn),
              weight_scale_inv=torch.ones(N // 128, K // 128))


@pytest.mark.parametrize("M, want", [
    (1, ("ns", 1)), (4, ("mma", 4, 16, 1)), (5, ("mma", 5, 16, 0)), (8, ("mma", 8, 16, 0)),
    (9, ("mma", 9, 16, -1)), (16, ("mma", 16, 16, -1)), (17, ("stock", 17)),
])
def test_fp8gemv_routes_by_m(monkeypatch, M, want):
    sc = load(monkeypatch, **_fp8_env(16))
    meth = _install_fp8(monkeypatch, sc)
    x = torch.zeros(M, 4096, dtype=torch.bfloat16)
    assert meth.apply(_layer(2048, 4096), x) == want


def test_fp8gemv_cap_8_keeps_v0518_behaviour(monkeypatch):
    sc = load(monkeypatch, **_fp8_env(8))
    meth = _install_fp8(monkeypatch, sc)
    assert meth.apply(_layer(2048, 4096), torch.zeros(8, 4096, dtype=torch.bfloat16)) == ("mma", 8, 16, 0)
    assert meth.apply(_layer(2048, 4096), torch.zeros(9, 4096, dtype=torch.bfloat16)) == ("stock", 9)


def test_max_m_above_16_refused(monkeypatch):
    with pytest.raises(AssertionError):
        load(monkeypatch, CORAL_MAX_M=17)


# ---------------------------------------------------------------------------------------------------------------
# MoE block (CORAL_MOE): v0.5.21 defer_finalize kwarg, v9 / v10c routes, M cap
# ---------------------------------------------------------------------------------------------------------------
def _moe_block_cls():
    meta = dict(device="meta")

    class Qwen2MoeSparseMoeBlock:
        tp_size = 1

        def __init__(self):
            self.experts = NS(w13_weight=torch.empty(256, 1024, 2048, dtype=torch.float8_e4m3fn, **meta),
                              w13_weight_scale_inv=torch.empty(256, 8, 16, **meta),
                              w2_weight=torch.empty(256, 2048, 512, dtype=torch.float8_e4m3fn, **meta),
                              w2_weight_scale_inv=torch.empty(256, 16, 4, **meta))
            lin = lambda n, k: NS(weight=torch.empty(n, k, dtype=torch.float8_e4m3fn, **meta),
                                  weight_scale_inv=torch.empty(n // 128, k // 128, **meta))
            self.shared_expert = NS(gate_up_proj=lin(1024, 2048), down_proj=lin(2048, 512))
            self.shared_expert_gate = NS(weight=torch.empty(1, 2048, **meta))
            self.topk = NS(topk_config=NS(renormalize=True, top_k=8, use_grouped_topk=False, correction_bias=None))
            self.gate = lambda x: (torch.zeros(x.shape[0], 256), None)

        def forward(self, hidden_states, forward_batch=None, defer_finalize=False):  # v0.5.21 signature
            return ("stock", hidden_states.shape[0], defer_finalize)

    return Qwen2MoeSparseMoeBlock


def _install_moe(monkeypatch, **env):
    sc = load(monkeypatch, CORAL_MOE=1, **env)
    import coral_kernels as ck
    seen = []
    for name in ("coral_moe_decode_v9", "coral_moe_decode_v10c"):
        def f(x, *a, _n=name, **k):
            seen.append((_n.split("_")[-1], x.shape[0], k.get("max_m")))
            return torch.zeros(x.shape[0], 2048, dtype=torch.bfloat16)
        monkeypatch.setattr(ck, name, f)
    cls = _moe_block_cls()
    sc._install_moe(NS(Qwen2MoeSparseMoeBlock=cls))
    return cls(), seen


def test_moe_fires_with_v0521_defer_finalize_kwarg(monkeypatch):
    blk, seen = _install_moe(monkeypatch, CORAL_MOE_ROUTE="v9", CORAL_MOE_MAX_M=16)
    y = blk.forward(torch.zeros(8, 2048, dtype=torch.bfloat16), None, defer_finalize=False)
    assert isinstance(y, torch.Tensor) and seen == [("v9", 8, 16)]


def test_moe_defer_finalize_true_goes_stock(monkeypatch):
    blk, seen = _install_moe(monkeypatch, CORAL_MOE_ROUTE="v9", CORAL_MOE_MAX_M=16)
    assert blk.forward(torch.zeros(8, 2048, dtype=torch.bfloat16), None, defer_finalize=True) == ("stock", 8, True)
    assert seen == []


@pytest.mark.parametrize("M, want", [(1, [("v9", 1, 16)]), (8, [("v10c", 8, 16)]), (16, [("v10c", 16, 16)])])
def test_moe_v10c_route(monkeypatch, M, want):
    blk, seen = _install_moe(monkeypatch, CORAL_MOE_ROUTE="v10c", CORAL_MOE_MAX_M=16)
    blk.forward(torch.zeros(M, 2048, dtype=torch.bfloat16), None, defer_finalize=False)
    assert seen == want


def test_moe_above_cap_goes_stock(monkeypatch):
    blk, seen = _install_moe(monkeypatch, CORAL_MOE_ROUTE="v9", CORAL_MOE_MAX_M=8)
    assert blk.forward(torch.zeros(16, 2048, dtype=torch.bfloat16), None, defer_finalize=False) == ("stock", 16, False)
    assert seen == []


# ---------------------------------------------------------------------------------------------------------------
# Fused add+norm+router (CORAL_NORM_ROUTER): cap follows CORAL_MAX_M
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("max_m, M, fused", [(16, 16, True), (16, 17, False), (8, 9, False), (4, 8, True)])
def test_norm_router_cap(monkeypatch, max_m, M, fused):
    sc = load(monkeypatch, CORAL_NORM_ROUTER=1, CORAL_MAX_M=max_m)
    import coral_kernels as ck
    monkeypatch.setattr(ck, "add_norm_router", lambda x, r, w, eps, wr, bn, nw: (x, r, torch.zeros(x.shape[0], 256)))

    class GemmaRMSNorm:
        def _forward_impl(self, x, residual=None, post_residual_addition=None):
            return "stock"

    sc._install_norm_router(NS(GemmaRMSNorm=GemmaRMSNorm))
    n = GemmaRMSNorm()
    n._coral_router, n.weight, n.variance_epsilon = torch.zeros(256, 2048), torch.zeros(2048), 1e-6
    x = torch.zeros(M, 2048, dtype=torch.bfloat16)
    out = n._forward_impl(x, torch.zeros_like(x))
    assert (out != "stock") == fused


# CORAL_NORM_ROUTER_MT_MIN_M: from that many tokens up, the multi-token kernel replaces the serial one; above the
# cap both give way to stock. Unset keeps the v0.5.18 behaviour (serial at every M).
@pytest.mark.parametrize("min_m, M, want", [
    (None, 8, "serial"), (None, 16, "serial"), (0, 8, "serial"),
    (6, 1, "serial"), (6, 5, "serial"), (6, 6, ("mt", 8)), (6, 8, ("mt", 8)),
    (6, 9, ("mt", 16)), (6, 16, ("mt", 16)), (6, 17, "stock"),
])
def test_norm_router_mt_dispatch(monkeypatch, min_m, M, want):
    env = dict(CORAL_NORM_ROUTER=1, CORAL_MAX_M=16)
    if min_m is not None:
        env["CORAL_NORM_ROUTER_MT_MIN_M"] = min_m
    sc = load(monkeypatch, **env)
    import coral_kernels as ck
    seen = []
    monkeypatch.setattr(ck, "add_norm_router", lambda x, r, w, eps, wr, bn, nw: (
        seen.append("serial"), (x, r, torch.zeros(x.shape[0], 256)))[1])
    monkeypatch.setattr(ck, "add_norm_router_mt", lambda x, r, w, eps, wr, bn, kc, nw: (
        seen.append(("mt", (bn, kc, nw))), (x, r, torch.zeros(x.shape[0], 256)))[1])

    class GemmaRMSNorm:
        def _forward_impl(self, x, residual=None, post_residual_addition=None):
            seen.append("stock")
            return "stock"

    sc._install_norm_router(NS(GemmaRMSNorm=GemmaRMSNorm))
    n = GemmaRMSNorm()
    n._coral_router, n.weight, n.variance_epsilon = torch.zeros(256, 2048), torch.zeros(2048), 1e-6
    x = torch.zeros(M, 2048, dtype=torch.bfloat16)
    n._forward_impl(x, torch.zeros_like(x))
    if isinstance(want, tuple):
        assert seen == [("mt", sc._NORM_ROUTER_MT_CFG[want[1]])]
    else:
        assert seen == [want]


# ---------------------------------------------------------------------------------------------------------------
# GDN hooks: conv-fold turns off upstream's fused decode proj+conv; verify-views skipped on v0.5.21
# ---------------------------------------------------------------------------------------------------------------
def test_conv_fold_disables_upstream_fused_decode_conv(monkeypatch):
    sc = load(monkeypatch, CORAL_GDN_CONV_FOLD=1)
    mod = NS(Qwen3_5GatedDeltaNet=type("G", (), {"forward": lambda self, h, fb: None}),
             _gdn_decode_fused_proj_conv=True)
    sc._install_gdn_conv_fold(mod)
    assert mod._gdn_decode_fused_proj_conv is False


@pytest.mark.parametrize("v0521, installed", [(True, False), (False, True)])
def test_verify_views_only_on_v0518(monkeypatch, v0521, installed):
    sc = load(monkeypatch, CORAL_GDN_VERIFY_VIEWS=1)
    backend = type("GDNAttnBackend", (), {"_init_target_verify_qkv_routing": lambda self, fb: None} if v0521 else {})
    orig = lambda *a: "orig"
    mod = NS(GDNAttnBackend=backend, fused_qkv_split_gdn_prefill=orig, __name__="gdn_backend")
    sc._install_gdn_verify_views(mod)
    assert (mod.fused_qkv_split_gdn_prefill is not orig) == installed


# ---------------------------------------------------------------------------------------------------------------
# DFlash2 selector-confidence log (CORAL_DFLASH_LOG): what it records per step
# ---------------------------------------------------------------------------------------------------------------
def test_dflash_log_records_ranks_and_probs(monkeypatch, tmp_path):
    load(monkeypatch)
    sys.modules.pop("coral_dflash_log", None)
    monkeypatch.setenv("CORAL_DFLASH_LOG", str(tmp_path / "log"))
    import coral_dflash_log as dl
    bs, gamma, k = 1, 3, 4
    cand_out = torch.tensor([[[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]])
    q_out = torch.tensor([[[.1, .6, .2, .1], [.5, .3, .1, .1], [.25, .25, .4, .1]]])
    candidates = torch.tensor([[7, 11, 20, 99]])        # anchor + drafts (pos 3's draft not a candidate)
    target = torch.tensor([[11, 21, 33, 5]])            # target accepts pos 1, rejects pos 2 (21 != 20)

    class DFlashWorkerV2:
        _draft_sampler = NS(candidate_out=cand_out, q_out=q_out)

        def _accept_block(self, **kw):
            return torch.tensor([1]), None, None, None, None, target

    dl.install(NS(DFlashWorkerV2=DFlashWorkerV2), lambda m: None)
    DFlashWorkerV2()._accept_block(candidates=candidates, next_token_logits=None, sampling_info=None,
                                   draft_input=None, prefix_lens=torch.tensor([100]), bs=bs)
    (row,) = dl._ROWS
    assert row["accept"] == 1 and row["prefix"] == 100
    assert row["q_draft"] == [0.6, 0.5, -1.0] and row["stale"] is True
    assert row["q1"] == [0.6, 0.5, 0.4] and row["q2"] == [0.2, 0.3, 0.25]
    assert row["trank"] == [0, 1, 3]                    # 11 top; 21 second; 33 (q .1) last of 4
    dl._ROWS.clear()


def test_dflash_log_soft_path_probs(monkeypatch, tmp_path):
    load(monkeypatch)
    sys.modules.pop("coral_dflash_log", None)
    monkeypatch.setenv("CORAL_DFLASH_LOG", str(tmp_path / "log"))
    import coral_dflash_log as dl
    bs, gamma, K = 1, 2, 3
    ids = torch.tensor([[[5, 6, 7], [8, 9, 10]]])
    scores = torch.zeros(bs, gamma, K, K)
    scores[0, 0, 0] = torch.tensor([2.0, 0.0, 0.0])           # slot 0 (anchor row): picks 5
    scores[0, 1, 0] = torch.tensor([0.0, 0.0, 3.0])           # slot 1 given cand 0 (=5): picks 10

    class Sampler:
        def __init__(self):
            self.out = torch.zeros(4, dtype=torch.long); self.q_out = torch.zeros(1, gamma, K)

        def __call__(self, hs, ii):
            i, sc = mod._selector_lattice(None, None, None)
            self.out[:2] = torch.tensor([5, 10])

    mod = NS(_SelectorDraftSampler=Sampler, _selector_lattice=lambda *a: (ids, scores),
             DFlashWorkerV2=type("W", (), {"_accept_block": lambda self, **k: None}))
    dl.install(mod, lambda m: None)
    s = Sampler(); s(None, None)
    want0 = torch.softmax(torch.tensor([2.0, 0.0, 0.0]), 0)
    want1 = torch.softmax(torch.tensor([0.0, 0.0, 3.0]), 0)
    assert torch.allclose(s.q_soft[0, 0], want0) and torch.allclose(s.q_soft[0, 1], want1)


# ---------------------------------------------------------------------------------------------------------------
# FA3 target-verify split-KV (CORAL_FA_VERIFY_SPLITS): only TARGET_VERIFY sees the forced num_splits, and it is restored
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("verify, by_kw", [(True, False), (False, False), (True, True)])
def test_fa_verify_splits(monkeypatch, verify, by_kw):
    sc = load(monkeypatch, CORAL_FA_VERIFY_SPLITS=64)

    class FlashAttentionBackend:
        num_splits = 0

        def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
            return self.num_splits

    sc._install_fa_verify_splits(NS(FlashAttentionBackend=FlashAttentionBackend, __name__="fa"))
    b = FlashAttentionBackend()
    fb = NS(forward_mode=NS(is_target_verify=lambda: verify))
    seen = b.forward_extend(1, 2, 3, 4, forward_batch=fb) if by_kw else b.forward_extend(1, 2, 3, 4, fb, False)
    assert seen == (64 if verify else 0)
    assert b.num_splits == 0

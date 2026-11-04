"""bf16_dot holds ROCm DFLASH draft attention over an FP8 KV to BF16 accuracy and
leaves the default FP8 path and a BF16 cache unchanged."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.triton_draft_hip import init_draft_backend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_hip_version
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Rounding the probabilities, the split partials and the output to BF16 costs a
# few 1e-3; FP8 q or probabilities cost several 1e-2.
BF16_ERR = 1e-2
BF16_ATOL = 2e-2
BF16_RTOL = 1e-2
# Triton 3.4 on ROCm 7.0 aborts gfx950 fp8 KV tl.dot (triton-lang/triton#8278);
# only the default path's FP8 dots need it.
FP8_DOT_OK = get_hip_version()[:2] != (7, 0)

# K3.1 DFlash2 MLA draft at TP8: 12 local heads over a 576 / 512 latent, one
# 8-token block.
MLA = dict(h_q=12, h_kv=1, d=576, dv=512, l_ext=8, causal=True, mla=True)

# the MLA5 draft's sliding-window layers (window 1024), plus full attention
EXTEND_CASES = {
    "mla5_window": dict(MLA, prefix_lens=(3000, 1500, 5), window=1023),
    "mla_full": dict(MLA, prefix_lens=(3000, 300, 5)),
}


def _inputs(case, cache_dtype, seed=0):
    """Draft-block q / k / v and a shuffled cache holding each request's prefix.
    MLA reads V as the first dv columns of the latent K, as its pool does."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    h_q, h_kv, d, dv = case["h_q"], case["h_kv"], case["d"], case["dv"]
    prefix_lens, l_ext = case["prefix_lens"], case["l_ext"]
    bs, rows = len(prefix_lens), len(prefix_lens) * l_ext

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=g)

    pool = sum(prefix_lens) + 64
    q = randn(rows, h_q, d)
    k = randn(rows, h_kv, d)
    k_buffer = randn(pool, h_kv, d).to(cache_dtype)
    if case["mla"]:
        v = k[..., :dv]
        v_buffer = k_buffer[..., :dv]
    else:
        v = randn(rows, h_kv, dv)
        v_buffer = randn(pool, h_kv, dv).to(cache_dtype)
    kv_indices = torch.randperm(pool, device="cuda", generator=g)[: sum(prefix_lens)]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(torch.tensor(prefix_lens, device="cuda"), 0)
    qo_indptr = torch.arange(0, rows + 1, l_ext, dtype=torch.int32, device="cuda")
    return q, k, v.contiguous(), k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices


def _scale(case):
    # MLA scales by the 128 + 64 qk head of the target, not the latent width.
    return (128 + 64) ** -0.5 if case["mla"] else case["d"] ** -0.5


def _reference(case, q, k, v, k_buffer, v_buffer, kv_indptr, kv_indices, scale):
    """fp32 attention of each draft row over its request's prefix and block, the
    cache read exactly (FP8 and BF16 values are exact in fp32) and descaled.
    The block is causal or not; a window keeps the keys of [p - window, p]."""
    h_q, h_kv, l_ext = case["h_q"], case["h_kv"], case["l_ext"]
    window = case.get("window", -1)
    k_scale, v_scale = case.get("k_scale", 1.0), case.get("v_scale", 1.0)
    bounds = kv_indptr.tolist()
    out = []
    for b in range(len(bounds) - 1):
        locs = kv_indices[bounds[b] : bounds[b + 1]]
        prefix = locs.numel()
        rows = slice(b * l_ext, (b + 1) * l_ext)
        keys = torch.cat([k_buffer[locs].float() * k_scale, k[rows].float()])
        vals = torch.cat([v_buffer[locs].float() * v_scale, v[rows].float()])
        q_rows = q[rows].float().unflatten(1, (h_kv, h_q // h_kv))
        s = torch.einsum("lkgd,tkd->kglt", q_rows, keys) * scale
        q_pos = prefix + torch.arange(l_ext, device="cuda")
        k_pos = torch.arange(prefix + l_ext, device="cuda")
        seen = torch.ones(l_ext, prefix + l_ext, dtype=torch.bool, device="cuda")
        if case["causal"]:
            seen &= k_pos[None, :] <= q_pos[:, None]
        if window > 0:
            seen &= k_pos[None, :] >= q_pos[:, None] - window
        p = torch.softmax(s.masked_fill(~seen, float("-inf")), dim=-1)
        out.append(torch.einsum("kglt,tkd->lkgd", p, vals).flatten(1, 2))
    return torch.cat(out)


def _extend(case, inputs, bf16_dot):
    q, k, v, k_buffer, v_buffer, qo_indptr, kv_indptr, kv_indices = inputs
    out = torch.empty(*q.shape[:2], case["dv"], dtype=q.dtype, device="cuda")
    extend_attention_fwd(
        q,
        k,
        v,
        out,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        None,
        case["causal"],
        None,
        case["l_ext"],
        case.get("k_scale", 1.0),
        case.get("v_scale", 1.0),
        sm_scale=_scale(case),
        sliding_window_size=case.get("window", -1),
        **({"bf16_dot": True} if bf16_dot else {}),
    )
    return out


def _err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


KERNELS = {"extend": (_extend, EXTEND_CASES)}
PARAMS = [
    pytest.param(kernel, name, id=f"{kernel}-{name}")
    for kernel, (_, cases) in KERNELS.items()
    for name in cases
]
UNCHANGED_PARAMS = PARAMS


@pytest.mark.parametrize("kernel,name", PARAMS)
def test_fp8_cache_bf16_dot_matches_fp32(kernel, name):
    run, cases = KERNELS[kernel]
    case = cases[name]
    inputs = _inputs(case, fp8_dtype)
    q, k, v, k_buffer, v_buffer, _, kv_indptr, kv_indices = inputs
    ref = _reference(
        case, q, k, v, k_buffer, v_buffer, kv_indptr, kv_indices, _scale(case)
    )

    out = run(case, inputs, bf16_dot=True)
    err = _err(out, ref)
    assert err < BF16_ERR, err
    torch.testing.assert_close(out.float(), ref, atol=BF16_ATOL, rtol=BF16_RTOL)

    if FP8_DOT_OK:
        # The default still runs FP8 q and probabilities over an FP8 cache.
        default = run(case, inputs, bf16_dot=False)
        default_err = _err(default, ref)
        assert not torch.equal(default, out)
        assert default_err > 2 * err, (default_err, err)


@pytest.mark.parametrize("kernel,name", UNCHANGED_PARAMS)
def test_bf16_cache_unchanged_by_bf16_dot(kernel, name):
    run, cases = KERNELS[kernel]
    case = cases[name]
    inputs = _inputs(case, torch.bfloat16, seed=1)
    torch.testing.assert_close(
        run(case, inputs, bf16_dot=True),
        run(case, inputs, bf16_dot=False),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        (SpeculativeAlgorithm.DFLASH, True),
        (SpeculativeAlgorithm.DSPARK, True),
        (SpeculativeAlgorithm.EAGLE, False),
    ],
)
def test_draft_backend_gate(algorithm, expected):
    # TritonAttnBackend runs init_draft_backend for ROCm draft workers only.
    backend = SimpleNamespace(use_mla=False, dcp_size=1)
    init_draft_backend(backend, SimpleNamespace(spec_algorithm=algorithm))
    assert backend.draft_bf16_dot is expected


def _recording_backend(draft_bf16_dot):
    """A TritonAttnBackend whose verify and extend kernels record their kwargs;
    the verify kernel declines, as it does for a sliding window."""
    calls = {}

    def record(name):
        def kernel(*args, **kwargs):
            calls[name] = kwargs
            return False

        return kernel

    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.draft_bf16_dot = draft_bf16_dot
    backend.use_verify_shared_kv = False
    backend.use_verify_splitkv = True
    backend.verify_shared_kv_fwd = record("verify_shared_kv")
    backend.verify_splitkv_fwd = record("verify_splitkv")
    backend.extend_attention_fwd = record("extend")
    backend._can_run_dense_fp8_chunked_mha = lambda *args: False
    backend.dcp_size = 1
    backend.enable_deterministic = False
    backend.dllm_attention = None
    backend.page_size = 1
    backend.req_to_token_pool = SimpleNamespace(size=4)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: None, get_value_buffer=lambda _: None
    )
    window_indices = torch.zeros(1, dtype=torch.int64)
    backend.forward_metadata = SimpleNamespace(
        kv_indptr=None,
        kv_indices=None,
        window_kv_indptr=None,
        window_kv_indices=window_indices,
        window_kv_offsets=None,
        qo_indptr=None,
        custom_mask=None,
        mask_indptr=None,
        max_extend_len=MLA["l_ext"],
    )
    return backend, calls


@pytest.mark.parametrize("draft_bf16_dot", [False, True])
def test_forward_extend_hands_bf16_dot_to_kernels(draft_bf16_dot):
    backend, calls = _recording_backend(draft_bf16_dot)
    h_q, d, dv, l_ext = MLA["h_q"], MLA["d"], MLA["dv"], MLA["l_ext"]
    layer = SimpleNamespace(
        layer_id=0,
        qk_head_dim=d,
        v_head_dim=dv,
        tp_q_head_num=h_q,
        k_scale=None,
        v_scale=None,
        logit_capping_method="tanh",
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.DECODER,
        sliding_window_size=255,
        scaling=(128 + 64) ** -0.5,
        xai_temperature_len=-1,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        mha_one_shot=False,
        extend_seq_lens_cpu=None,
    )
    q = torch.zeros(l_ext, h_q * d)
    k = torch.zeros(l_ext, 1, d)
    backend.forward_extend(q, k, k[..., :dv], layer, forward_batch, save_kv_cache=False)

    assert calls["extend"]["bf16_dot"] is draft_bf16_dot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

import math
from types import SimpleNamespace

import pytest
import sgl_kernel
import torch

import sglang.srt.layers.attention.intel_amx_backend as intel_amx_backend_mod
from sglang.srt.layers.attention.intel_amx_backend import IntelAMXAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="extra-test-cpu-intel")


def _softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    if cap <= 0.0:
        return x
    return cap * torch.tanh(x / cap)


def _reference_extend_attention(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor | None,
    v_extend: torch.Tensor | None,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    sm_scale: float,
    out_dtype: torch.dtype,
    *,
    is_causal: bool = True,
    sliding_window_size: int = 0,
    window_anchor_at_start: bool = False,
    tree_mask: torch.Tensor | None = None,
    rel_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    logit_cap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_q_heads, head_dim = q_extend.shape
    num_kv_heads = k_buffer.shape[1]
    v_head_dim = v_buffer.shape[2]
    kv_group = num_q_heads // num_kv_heads
    o_ref = torch.empty((total_q, num_q_heads, v_head_dim), dtype=out_dtype)
    lse_ref = torch.empty((total_q, num_q_heads), dtype=torch.float32)

    max_len_extend = (
        int(extend_seq_lens.max().item()) if extend_seq_lens.numel() > 0 else 0
    )
    num_seqs = req_pool_indices.shape[0]
    tree_3d = (
        tree_mask.view(num_seqs, max_len_extend, max_len_extend)
        if tree_mask is not None
        else None
    )
    for req_idx in range(num_seqs):
        pool_idx = int(req_pool_indices[req_idx].item())
        seq_len = int(seq_lens[req_idx].item())
        ext_len = int(extend_seq_lens[req_idx].item())
        q_start = int(extend_start_loc[req_idx].item())
        prefix_len = seq_len - ext_len

        slot_ids = req_to_token[pool_idx, :seq_len].to(torch.long)
        if k_extend is not None and prefix_len < seq_len:
            k_full = torch.cat(
                [
                    k_buffer[slot_ids[:prefix_len]],
                    k_extend[q_start : q_start + ext_len],
                ],
                dim=0,
            ).float()
            v_full = torch.cat(
                [
                    v_buffer[slot_ids[:prefix_len]],
                    v_extend[q_start : q_start + ext_len],
                ],
                dim=0,
            ).float()
        else:
            k_full = k_buffer[slot_ids].float()
            v_full = v_buffer[slot_ids].float()

        k_exp = k_full.repeat_interleave(kv_group, dim=1)
        v_exp = v_full.repeat_interleave(kv_group, dim=1)

        if tree_3d is not None:
            req_tree = tree_3d[req_idx, :ext_len, :ext_len]
        else:
            req_tree = None

        q_req = q_extend[q_start : q_start + ext_len].float()
        logits = torch.einsum("qhd,khd->hqk", q_req, k_exp) * sm_scale

        q_lin = torch.arange(prefix_len, seq_len, dtype=torch.long)[:, None]
        k_lin = torch.arange(seq_len, dtype=torch.long)[None, :]
        if req_tree is not None:
            tri = torch.tril(req_tree.bool())
            vis = tri.sum(dim=-1).to(torch.long)
            depths = torch.where(
                vis > 0, vis - 1, torch.arange(ext_len, dtype=torch.long)
            )
            q_pos = (prefix_len + depths)[:, None]
            k_pos = torch.cat(
                [torch.arange(prefix_len, dtype=torch.long), prefix_len + depths],
                dim=0,
            )[None, :]
        else:
            q_pos = q_lin
            k_pos = k_lin

        if rel_bias is not None:
            rb = rel_bias[q_start : q_start + ext_len].float()
            rel_extent = rb.shape[-1]
            dist = q_pos - k_pos
            valid = (dist >= 0) & (dist < rel_extent)
            dist_clamped = dist.clamp(0, rel_extent - 1)
            rb_gathered = torch.gather(
                rb.permute(1, 0, 2),
                2,
                dist_clamped[None, :, :].expand(num_q_heads, -1, -1),
            )
            logits = logits + rb_gathered * valid[None, :, :].float()

        if logit_cap > 0.0:
            logits = logit_cap * torch.tanh(logits / logit_cap)

        mask = torch.ones((ext_len, seq_len), dtype=torch.bool)
        if req_tree is not None:
            mask[:, prefix_len:] = req_tree.bool()
        elif is_causal:
            mask &= k_lin <= q_lin

        if sliding_window_size > 0:
            swa_w = sliding_window_size - 1
            if window_anchor_at_start:
                kv_min = max(0, prefix_len - swa_w)
                mask &= k_lin >= kv_min
            else:
                mask &= k_pos >= (q_pos - swa_w)
                if not is_causal and req_tree is None:
                    mask &= k_pos <= (q_pos + swa_w)

        logits = logits.masked_fill(~mask[None, :, :], float("-inf"))
        if sinks is not None:
            sink_col = sinks.float().view(num_q_heads, 1, 1).expand(-1, ext_len, 1)
            logits_with_sink = torch.cat([logits, sink_col], dim=-1)
            lse = torch.logsumexp(logits_with_sink, dim=-1)
            probs = torch.softmax(logits_with_sink, dim=-1)[..., :-1]
        else:
            lse = torch.logsumexp(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

        out_req = torch.einsum("hqk,khd->qhd", probs, v_exp)
        o_ref[q_start : q_start + ext_len] = out_req.to(out_dtype)
        lse_ref[q_start : q_start + ext_len] = lse.transpose(0, 1)

    return o_ref, lse_ref


def _reference_decode_attention(
    query: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    out_dtype: torch.dtype,
    *,
    sliding_window_size: int = 0,
    rel_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
    logit_cap: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bs, num_q_heads, _ = query.shape
    num_kv_heads = k_buffer.shape[1]
    kv_group = num_q_heads // num_kv_heads
    v_head_dim = v_buffer.shape[2]
    o_ref = torch.empty((bs, num_q_heads, v_head_dim), dtype=out_dtype)
    lse_ref = torch.empty((bs, num_q_heads), dtype=torch.float32)

    for b in range(bs):
        pool_idx = int(req_pool_indices[b].item())
        seq_len = int(seq_lens[b].item())
        kv_start = (
            max(0, seq_len - sliding_window_size) if sliding_window_size > 0 else 0
        )
        slot_ids = req_to_token[pool_idx, kv_start:seq_len].to(torch.long)
        k_seq = k_buffer[slot_ids].float().repeat_interleave(kv_group, dim=1)
        v_seq = v_buffer[slot_ids].float().repeat_interleave(kv_group, dim=1)
        q_b = query[b].float()
        logits = torch.einsum("hd,khd->hk", q_b, k_seq) * sm_scale
        if rel_bias is not None:
            rb = rel_bias[b].float()
            rel_extent = rb.shape[-1]
            k_pos = torch.arange(kv_start, seq_len)
            dist = (seq_len - 1) - k_pos
            valid = (dist >= 0) & (dist < rel_extent)
            rb_vals = rb[:, dist.clamp(0, rel_extent - 1)] * valid[None, :].float()
            logits = logits + rb_vals
        if logit_cap > 0.0:
            logits = logit_cap * torch.tanh(logits / logit_cap)
        if sinks is not None:
            sink_col = sinks.float().view(num_q_heads, 1)
            logits_s = torch.cat([logits, sink_col], dim=-1)
            lse = torch.logsumexp(logits_s, dim=-1)
            probs = torch.softmax(logits_s, dim=-1)[..., :-1]
        else:
            lse = torch.logsumexp(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
        out_b = torch.einsum("hk,khd->hd", probs, v_seq)
        o_ref[b] = out_b.to(out_dtype)
        lse_ref[b] = lse

    return o_ref, lse_ref


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("window_anchor_at_start", [False, True])
@pytest.mark.parametrize("use_rel_bias", [False, True])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_extend_attention_cpu_inkling2_features(
    out_dtype: torch.dtype,
    is_causal: bool,
    window_anchor_at_start: bool,
    use_rel_bias: bool,
    use_sinks: bool,
):
    torch.manual_seed(42)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    prefixes = [40, 12, 0]
    extends = [17, 9, 25]
    seq_lens = torch.tensor(
        [p + e for p, e in zip(prefixes, extends)], dtype=torch.int64
    )
    extend_seq_lens = torch.tensor(extends, dtype=torch.int32)
    extend_start_loc = torch.tensor([0, 17, 26], dtype=torch.int32)
    total_q = sum(extends)
    max_seq = int(seq_lens.max().item())

    swa_window_plus_1 = 21
    sm_scale = 1.0 / math.sqrt(head_dim)

    q_extend = torch.randn((total_q, num_q_heads, head_dim), dtype=torch.bfloat16) * 0.5
    k_extend = (
        torch.randn((total_q, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )
    v_extend = (
        torch.randn((total_q, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )

    total_slots = 512
    k_buffer = (
        torch.randn((total_slots, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )
    v_buffer = (
        torch.randn((total_slots, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )
    req_pool_indices = torch.tensor([2, 0, 1], dtype=torch.int64)
    req_to_token = torch.arange(4 * max_seq, dtype=torch.int32).view(4, max_seq) + 7

    rel_bias = (
        torch.randn((total_q, num_q_heads, 32), dtype=torch.float32) * 0.4
        if use_rel_bias
        else None
    )
    sinks = (
        torch.randn((num_q_heads,), dtype=torch.bfloat16) * 0.5 if use_sinks else None
    )

    o_out = torch.empty((total_q, num_q_heads, head_dim), dtype=out_dtype)
    lse_out = torch.empty((total_q, num_q_heads), dtype=torch.float32)

    torch.ops.sgl_kernel.extend_attention_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_out,
        k_buffer,
        v_buffer,
        1.0,
        1.0,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        max(extends),
        sm_scale,
        0.0,
        False,
        swa_window_plus_1,
        None,
        sinks,
        None,
        is_causal,
        rel_bias,
        lse_out,
        window_anchor_at_start,
    )

    o_ref, lse_ref = _reference_extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        out_dtype,
        is_causal=is_causal,
        sliding_window_size=swa_window_plus_1,
        window_anchor_at_start=window_anchor_at_start,
        rel_bias=rel_bias,
        sinks=sinks,
    )

    atol = 2e-2 if out_dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if out_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(o_out, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse_out, lse_ref, atol=1e-2, rtol=1e-2)


def test_extend_attention_cpu_tree_mask_and_anchor():
    torch.manual_seed(7)
    num_q_heads, num_kv_heads, head_dim = 4, 4, 64
    prefixes = [30, 8]
    extends = [5, 5]
    seq_lens = torch.tensor(
        [p + e for p, e in zip(prefixes, extends)], dtype=torch.int64
    )
    extend_seq_lens = torch.tensor(extends, dtype=torch.int32)
    extend_start_loc = torch.tensor([0, 5], dtype=torch.int32)
    total_q = sum(extends)
    sm_scale = 1.0 / math.sqrt(head_dim)

    q_extend = torch.randn((total_q, num_q_heads, head_dim), dtype=torch.bfloat16)
    k_extend = torch.randn((total_q, num_kv_heads, head_dim), dtype=torch.bfloat16)
    v_extend = torch.randn((total_q, num_kv_heads, head_dim), dtype=torch.bfloat16)
    k_buffer = torch.randn((128, num_kv_heads, head_dim), dtype=torch.bfloat16)
    v_buffer = torch.randn((128, num_kv_heads, head_dim), dtype=torch.bfloat16)
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
    req_to_token = torch.arange(80, dtype=torch.int32).view(2, 40)

    m0 = torch.tril(torch.ones((5, 5), dtype=torch.bool))
    m0[3, 2] = False
    m0[4, 2] = False
    m1 = torch.tril(torch.ones((5, 5), dtype=torch.bool))
    m1[2, 1] = False
    m1[3, 1] = False
    m1[4, 1] = False
    m1[4, 3] = False
    tree_mask = torch.cat([m0.reshape(-1), m1.reshape(-1)], dim=0)
    rel_bias = torch.randn((total_q, num_q_heads, 24), dtype=torch.bfloat16) * 0.3

    o_out = torch.empty((total_q, num_q_heads, head_dim), dtype=torch.float32)
    lse_out = torch.empty((total_q, num_q_heads), dtype=torch.float32)

    torch.ops.sgl_kernel.extend_attention_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_out,
        k_buffer,
        v_buffer,
        1.0,
        1.0,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        5,
        sm_scale,
        0.0,
        False,
        16,
        None,
        None,
        tree_mask,
        True,
        rel_bias,
        lse_out,
        True,
    )

    o_ref, lse_ref = _reference_extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        torch.float32,
        is_causal=True,
        sliding_window_size=16,
        window_anchor_at_start=True,
        tree_mask=tree_mask,
        rel_bias=rel_bias,
    )
    torch.testing.assert_close(o_out, o_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_out, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 4), (8, 2)])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_kv_splits", [1, 8])
@pytest.mark.parametrize("sliding_window_size", [0, 33])
@pytest.mark.parametrize("use_rel_bias", [False, True])
@pytest.mark.parametrize("use_sinks", [False, True])
def test_decode_attention_cpu_inkling2_features(
    num_q_heads: int,
    num_kv_heads: int,
    out_dtype: torch.dtype,
    num_kv_splits: int,
    sliding_window_size: int,
    use_rel_bias: bool,
    use_sinks: bool,
):
    torch.manual_seed(13)
    head_dim = 64
    bs = 3
    seq_lens_vals = [18, 97, 260]
    seq_lens = torch.tensor(seq_lens_vals, dtype=torch.int64)
    max_seq = max(seq_lens_vals)
    sm_scale = 1.0 / math.sqrt(head_dim)

    query = torch.randn((bs, num_q_heads, head_dim), dtype=torch.bfloat16) * 0.5
    key = torch.randn((bs, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    value = torch.randn((bs, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5

    total_slots = bs * max_seq + 64
    k_buffer = (
        torch.randn((total_slots, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )
    v_buffer = (
        torch.randn((total_slots, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    )
    req_pool_indices = torch.tensor([1, 2, 0], dtype=torch.int64)
    req_to_token = torch.arange(3 * max_seq, dtype=torch.int32).view(3, max_seq)
    loc = torch.tensor(
        [
            req_to_token[int(req_pool_indices[b]), seq_lens_vals[b] - 1].item()
            for b in range(bs)
        ],
        dtype=torch.int64,
    )

    attn_logits = torch.empty(
        (bs, num_q_heads, num_kv_splits, head_dim + 1), dtype=torch.float32
    )
    output = torch.empty((bs, num_q_heads, head_dim), dtype=out_dtype)
    lse_out = torch.empty((bs, num_q_heads), dtype=torch.float32)
    rel_bias = (
        torch.randn((bs, num_q_heads, 64), dtype=torch.float32) * 0.4
        if use_rel_bias
        else None
    )
    sinks = (
        torch.randn((num_q_heads,), dtype=torch.bfloat16) * 0.5 if use_sinks else None
    )

    k_buf_ref = k_buffer.clone()
    v_buf_ref = v_buffer.clone()
    k_buf_ref[loc.long()] = key
    v_buf_ref[loc.long()] = value

    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        1.0,
        1.0,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        0.0,
        False,
        sliding_window_size,
        None,
        sinks,
        rel_bias,
        lse_out,
    )

    o_ref, lse_ref = _reference_decode_attention(
        query,
        k_buf_ref,
        v_buf_ref,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        out_dtype,
        sliding_window_size=sliding_window_size,
        rel_bias=rel_bias,
        sinks=sinks,
    )

    atol = 2e-2 if out_dtype == torch.bfloat16 else 1e-2
    rtol = 2e-2 if out_dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(output, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse_out, lse_ref, atol=1e-2, rtol=1e-2)


def _make_amx_backend_and_kv(
    monkeypatch,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_tokens: int = 256,
):
    k_buf = torch.zeros((max_tokens, num_kv_heads, head_dim), dtype=torch.bfloat16)
    v_buf = torch.zeros((max_tokens, num_kv_heads, head_dim), dtype=torch.bfloat16)

    class _DummyKVLayer:
        def store(self, layer, write_loc, k, v, **kwargs):
            loc = (
                write_loc.loc.long() if hasattr(write_loc, "loc") else write_loc.long()
            )
            k_buf[loc] = k.view(-1, num_kv_heads, head_dim)
            v_buf[loc] = v.view(-1, num_kv_heads, head_dim)

    class _DummyKVLayerView:
        def key(self):
            return k_buf

        def value(self):
            return v_buf

    class _DummyKVLayers:
        def read(self, layer_id):
            return _DummyKVLayerView()

        def layer(self, layer_id):
            return _DummyKVLayer()

    class _DummyLayerRouter:
        def routes_to(self, kind):
            return False

        def layer_ids(self):
            return [0]

        def read(self, layer_id):
            return _DummyKVLayerView()

        def layer(self, layer_id):
            return _DummyKVLayer()

    req_to_token = torch.arange(max_tokens, dtype=torch.int32).view(4, max_tokens // 4)
    token_to_kv_pool = SimpleNamespace(
        layer_router=_DummyLayerRouter(),
        get_value_buffer=lambda lid: v_buf,
        kv_layers=_DummyKVLayers(),
    )
    model_runner = SimpleNamespace(
        device=torch.device("cpu"),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=token_to_kv_pool,
        runner_ids=None,
        use_mla_backend=False,
        model_config=SimpleNamespace(context_len=1024, num_attention_heads=num_q_heads),
    )
    monkeypatch.setattr(
        intel_amx_backend_mod,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_size=1),
    )
    monkeypatch.setattr(
        intel_amx_backend_mod,
        "get_spec",
        lambda: SimpleNamespace(speculative_num_draft_tokens=1),
    )
    backend = IntelAMXAttnBackend(model_runner)
    return backend, k_buf, v_buf, req_to_token


def test_intel_amx_backend_inkling2_modes(monkeypatch):
    torch.manual_seed(99)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 64
    backend, k_buf, v_buf, req_to_token = _make_amx_backend_and_kv(
        monkeypatch, num_q_heads, num_kv_heads, head_dim, max_tokens=256
    )

    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=num_q_heads,
        tp_k_head_num=num_kv_heads,
        tp_v_head_num=num_kv_heads,
        qk_head_dim=head_dim,
        v_head_dim=head_dim,
        scaling=1.0 / math.sqrt(head_dim),
        logit_cap=0.0,
        is_cross_attention=False,
        attn_type=AttentionType.ENCODER_ONLY,
        sliding_window_size=12,
        window_anchor_at_start=True,
        out_dtype=torch.float32,
        k_scale_float=None,
        v_scale_float=None,
    )

    prefixes = [20, 4]
    extends = [6, 5]
    for b, p_len in enumerate(prefixes):
        slots = req_to_token[b, :p_len].long()
        k_buf[slots] = (
            torch.randn((p_len, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
        )
        v_buf[slots] = (
            torch.randn((p_len, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
        )

    seq_lens = torch.tensor(
        [p + e for p, e in zip(prefixes, extends)], dtype=torch.int32
    )
    ext_lens = torch.tensor(extends, dtype=torch.int32)
    ext_start = torch.tensor([0, 6], dtype=torch.int32)
    out_locs = torch.cat(
        [
            req_to_token[0, prefixes[0] : prefixes[0] + extends[0]],
            req_to_token[1, prefixes[1] : prefixes[1] + extends[1]],
        ]
    )
    fb_ext = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.EXTEND,
        seq_lens=seq_lens,
        extend_seq_lens=ext_lens,
        extend_start_loc=ext_start,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        out_cache_loc=out_locs,
        encoder_out_cache_loc=None,
        encoder_lens=None,
        mha_return_lse=False,
        spec_info=None,
    )
    backend.init_forward_metadata(fb_ext)
    q = torch.randn((11, num_q_heads * head_dim), dtype=torch.bfloat16) * 0.5
    k = torch.randn((11, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    v = torch.randn((11, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.5
    rb = torch.randn((11, num_q_heads, 16), dtype=torch.float32) * 0.3

    out, lse = backend.forward_extend(
        q, k, v, layer, fb_ext, rel_bias=rb, return_lse=True
    )
    assert out.dtype == torch.float32
    assert lse.dtype == torch.float32
    assert out.shape == (11, num_q_heads * head_dim)
    assert lse.shape == (11, num_q_heads)

    o_ref, lse_ref = _reference_extend_attention(
        q.view(11, num_q_heads, head_dim),
        k,
        v,
        k_buf,
        v_buf,
        req_to_token,
        fb_ext.req_pool_indices,
        seq_lens,
        ext_lens,
        ext_start,
        layer.scaling,
        torch.float32,
        is_causal=False,
        sliding_window_size=13,
        window_anchor_at_start=True,
        rel_bias=rb,
    )
    torch.testing.assert_close(out.view_as(o_ref), o_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, lse_ref, atol=1e-2, rtol=1e-2)

    layer.attn_type = AttentionType.DECODER
    layer.window_anchor_at_start = False
    layer.sliding_window_size = -1
    draft_num = 3
    backend.num_draft_tokens = 99
    ver_prefixes = [10, 8]
    ver_out_locs = torch.cat(
        [
            req_to_token[0, ver_prefixes[0] : ver_prefixes[0] + draft_num],
            req_to_token[1, ver_prefixes[1] : ver_prefixes[1] + draft_num],
        ]
    )
    fb_ver = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.tensor(ver_prefixes, dtype=torch.int32),
        extend_seq_lens=None,
        extend_start_loc=None,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        out_cache_loc=ver_out_locs,
        encoder_out_cache_loc=None,
        encoder_lens=None,
        mha_return_lse=True,
        spec_info=SimpleNamespace(
            draft_token_num=draft_num, tree_topk=1, custom_mask=None
        ),
    )
    backend.init_forward_metadata(fb_ver)
    q_v = (
        torch.randn((2 * draft_num, num_q_heads * head_dim), dtype=torch.bfloat16) * 0.4
    )
    k_v = (
        torch.randn((2 * draft_num, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
    )
    v_v = (
        torch.randn((2 * draft_num, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
    )
    out_v, lse_v = backend.forward_extend(q_v, k_v, v_v, layer, fb_ver)
    assert out_v.dtype == torch.float32
    assert lse_v.shape == (6, num_q_heads)

    # 2b) DRAFT_EXTEND_V2 fallback when extend_seq_lens is None
    fb_dext = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        seq_lens=torch.tensor([13, 11], dtype=torch.int32),
        extend_seq_lens=None,
        extend_start_loc=None,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        out_cache_loc=ver_out_locs,
        encoder_out_cache_loc=None,
        encoder_lens=None,
        mha_return_lse=True,
        spec_info=SimpleNamespace(
            draft_token_num=draft_num, tree_topk=1, custom_mask=None
        ),
    )
    backend.init_forward_metadata(fb_dext)
    out_de, lse_de = backend.forward_extend(q_v, k_v, v_v, layer, fb_dext)
    assert out_de.dtype == torch.float32
    assert lse_de.shape == (6, num_q_heads)

    fb_dec = SimpleNamespace(
        batch_size=2,
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.tensor([14, 12], dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        out_cache_loc=torch.tensor(
            [req_to_token[0, 13].item(), req_to_token[1, 11].item()], dtype=torch.int64
        ),
        encoder_out_cache_loc=None,
        encoder_lens=None,
        mha_return_lse=True,
        spec_info=None,
    )
    backend.init_forward_metadata(fb_dec)
    q_d = torch.randn((2, num_q_heads * head_dim), dtype=torch.bfloat16) * 0.4
    k_d = torch.randn((2, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
    v_d = torch.randn((2, num_kv_heads, head_dim), dtype=torch.bfloat16) * 0.4
    rb_d = torch.randn((2, num_q_heads, 16), dtype=torch.float32) * 0.3
    out_d, lse_d = backend.forward_decode(q_d, k_d, v_d, layer, fb_dec, rel_bias=rb_d)
    assert out_d.dtype == torch.float32
    assert lse_d.shape == (2, num_q_heads)

    layer.window_anchor_at_start = True
    with pytest.raises(NotImplementedError):
        backend.forward_decode(q_d, k_d, v_d, layer, fb_dec)


def _reference_inkling2_moe(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gate_softcap: float,
    up_softcap: float,
) -> torch.Tensor:
    M, _ = hidden_states.shape
    topk = topk_ids.shape[1]
    out = torch.zeros_like(hidden_states, dtype=torch.float32)
    x_f = hidden_states.float()
    w13_f = w13.float()
    w2_f = w2.float()
    g_f = topk_weights.float()

    for m in range(M):
        for k_idx in range(topk):
            e = int(topk_ids[m, k_idx].item())
            gamma = float(g_f[m, k_idx].item())
            proj = x_f[m] @ w13_f[e].T
            gate = proj[0::2]
            up = proj[1::2]
            act = (
                torch.sigmoid(gate)
                * _softcap(gate, gate_softcap)
                * _softcap(up, up_softcap)
                * gamma
            )
            act_bf16 = act.to(hidden_states.dtype).float()
            out[m] += act_bf16 @ w2_f[e].T
    return out.to(hidden_states.dtype)


@pytest.mark.parametrize("M", [0, 1, 2, 3, 4, 5, 19])
@pytest.mark.parametrize("I", [32, 64])
@pytest.mark.parametrize("gate_softcap,up_softcap", [(10.0, 12.0), (0.0, 0.0)])
@pytest.mark.parametrize("is_vnni", [False, True])
def test_inkling2_fused_experts_cpu(
    M: int, I: int, gate_softcap: float, up_softcap: float, is_vnni: bool
):
    torch.manual_seed(2025)
    H, E, topk = 128, 6, 2
    x = torch.randn((M, H), dtype=torch.bfloat16) * 0.4
    w13 = torch.randn((E, 2 * I, H), dtype=torch.bfloat16) * 0.3
    w2 = torch.randn((E, H, I), dtype=torch.bfloat16) * 0.3
    topk_ids = torch.zeros((M, topk), dtype=torch.int32)
    for m in range(M):
        topk_ids[m] = torch.randperm(E, dtype=torch.int32)[:topk]
    gammas = torch.randn((M, topk), dtype=torch.float32) * 0.7

    w13_arg = torch.ops.sgl_kernel.convert_weight_packed(w13) if is_vnni else w13
    w2_arg = torch.ops.sgl_kernel.convert_weight_packed(w2) if is_vnni else w2

    out = sgl_kernel.inkling2_fused_experts_cpu(
        x, w13_arg, w2_arg, gammas, topk_ids, gate_softcap, up_softcap, False, is_vnni
    )
    ref = _reference_inkling2_moe(
        x, w13, w2, gammas, topk_ids, gate_softcap, up_softcap
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    if M > 0:
        x_in = x.clone()
        out_in = sgl_kernel.inkling2_fused_experts_cpu(
            x_in,
            w13_arg,
            w2_arg,
            gammas,
            topk_ids,
            gate_softcap,
            up_softcap,
            True,
            is_vnni,
        )
        assert out_in.data_ptr() == x_in.data_ptr()
        torch.testing.assert_close(out_in, ref, atol=2e-2, rtol=2e-2)


def _reference_short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    cache: torch.Tensor,
    safe_idx: torch.Tensor,
    cache_mask: torch.Tensor,
    cu: torch.Tensor,
    si: torch.Tensor,
    *,
    dynamic_weights: torch.Tensor | None = None,
    decode: bool = False,
    use_residual: bool = True,
    track_mask: torch.Tensor | None = None,
    track_indices: torch.Tensor | None = None,
    window: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    T, D = x.shape
    cache_out = cache.clone()
    win_out = window.clone() if window is not None else None
    out = torch.empty_like(x)
    if T == 0:
        return out, cache_out, win_out

    dynamic = dynamic_weights is not None
    width = weight.shape[0] if dynamic else weight.shape[1]
    n_heads = dynamic_weights.shape[1] if dynamic else 1
    head_dim = D // n_heads if dynamic else D
    scale = (width * (2 if dynamic else 1) + (1 if use_residual else 0)) ** -0.5

    for row in range(T):
        seq = row if decode else int(si[row].item())
        bos = row if decode else int(cu[seq].item())
        slot = int(safe_idx[seq].item())
        in_pool = bool(cache_mask[seq].item() != 0)
        has_cache = in_pool and slot > 0

        acc = torch.zeros((D,), dtype=torch.float32)
        old_hist = {}
        for tap in range(width):
            lag = tap if dynamic else (width - 1 - tap)
            src = row - lag
            from_x = src >= bos
            if from_x:
                val = x[src].float()
                raw_prefix = val
            else:
                cpos = src - bos + width - 1
                raw_prefix = cache[slot, cpos].float() if in_pool else torch.zeros((D,))
                val = raw_prefix if has_cache else torch.zeros((D,))
                old_hist[cpos] = raw_prefix

            if win_out is not None and lag < width - 1 and in_pool:
                win_out[seq, row - bos, width - 2 - lag] = (
                    val if from_x else raw_prefix
                ).to(win_out.dtype)

            if dynamic:
                dw = dynamic_weights[row].float().repeat_interleave(head_dim, dim=0)
                w_val = weight[lag].float() + dw[:, lag]
            else:
                w_val = weight[:, tap].to(x.dtype).float()
            acc += val * w_val

        total = acc + x[row].float() if use_residual else acc
        out[row] = (total * scale).to(x.dtype)

        if decode and has_cache and width > 1:
            do_track = (
                track_mask is not None
                and track_indices is not None
                and bool(track_mask[seq].item() != 0)
            )
            t_slot = int(track_indices[seq].item()) if do_track else 0
            for tap in range(width - 1):
                c_val = (
                    x[row]
                    if tap == width - 2
                    else old_hist[tap + 1].to(cache_out.dtype)
                )
                cache_out[slot, tap] = c_val
                if do_track:
                    cache_out[t_slot, tap] = c_val

    return out, cache_out, win_out


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("use_residual", [True, False])
def test_inkling2_short_convolution_cpu(
    dynamic: bool, decode: bool, use_residual: bool
):
    torch.manual_seed(314)
    D, width, n_heads = 96, 4, 4
    num_slots = 10
    if decode:
        B, T = 4, 4
        cu = torch.arange(B + 1, dtype=torch.int32)
        si = torch.arange(T, dtype=torch.int32)
        max_q = 1
    else:
        lens = [5, 3, 7]
        B = len(lens)
        T = sum(lens)
        cu = torch.tensor([0, 5, 8, 15], dtype=torch.int32)
        si = torch.cat(
            [torch.full((l,), i, dtype=torch.int32) for i, l in enumerate(lens)]
        )
        max_q = max(lens)

    x = torch.randn((T, D), dtype=torch.bfloat16) * 0.5
    weight = (
        torch.randn((width, D), dtype=torch.bfloat16) * 0.4
        if dynamic
        else torch.randn((D, width), dtype=torch.bfloat16) * 0.4
    )
    dyn_w = (
        torch.randn((T, n_heads, width), dtype=torch.bfloat16) * 0.3
        if dynamic
        else None
    )
    cache = torch.randn((num_slots, width - 1, D), dtype=torch.bfloat16) * 0.5
    safe_idx = torch.tensor([2, 0, 5, 7][:B], dtype=torch.int32)
    cache_mask = torch.tensor([1, 1, 0, 1][:B], dtype=torch.uint8)
    track_mask = torch.tensor([1, 0, 1, 1][:B], dtype=torch.bool) if decode else None
    track_indices = (
        torch.tensor([6, 8, 9, 4][:B], dtype=torch.int32) if decode else None
    )
    window = (
        torch.zeros((B, max_q, width - 1, D), dtype=torch.bfloat16)
        if not decode
        else None
    )

    cache_cpu = cache.clone()
    win_cpu = window.clone() if window is not None else None
    out = sgl_kernel.inkling2_short_convolution_cpu(
        x,
        weight,
        cache_cpu,
        safe_idx,
        cache_mask,
        cu,
        si,
        dyn_w,
        decode,
        use_residual,
        track_mask,
        track_indices,
        win_cpu,
    )
    ref, cache_ref, win_ref = _reference_short_conv(
        x,
        weight,
        cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        dynamic_weights=dyn_w,
        decode=decode,
        use_residual=use_residual,
        track_mask=track_mask,
        track_indices=track_indices,
        window=window,
    )
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(cache_cpu, cache_ref, atol=0.0, rtol=0.0)
    if win_cpu is not None:
        torch.testing.assert_close(win_cpu, win_ref, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("top_k", [0, 2])
@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("has_partial", [False, True])
@pytest.mark.parametrize("round_bf16", [False, True])
def test_inkling2_routed_sconv_epilogue_cpu(
    top_k: int, decode: bool, has_partial: bool, round_bf16: bool
):
    torch.manual_seed(55)
    D, width = 96, 4
    num_slots = 10
    if decode:
        B, T = 3, 3
        cu = torch.arange(B + 1, dtype=torch.int32)
        si = torch.arange(T, dtype=torch.int32)
        window = None
    else:
        lens = [4, 2, 5]
        B, T = len(lens), sum(lens)
        cu = torch.tensor([0, 4, 6, 11], dtype=torch.int32)
        si = torch.cat(
            [torch.full((l,), i, dtype=torch.int32) for i, l in enumerate(lens)]
        )
        window = torch.zeros((B, max(lens), width - 1, D), dtype=torch.bfloat16)

    if top_k > 0:
        down = torch.randn((T * top_k + 3, D), dtype=torch.bfloat16) * 0.4
        src2dst = torch.randperm(T * top_k + 3, dtype=torch.int32)[: T * top_k].view(
            T, top_k
        )
        comb = down[src2dst.long()].float().sum(dim=1).to(down.dtype)
        if round_bf16:
            comb = comb.to(torch.bfloat16)
    else:
        down = torch.randn((T, D), dtype=torch.bfloat16) * 0.4
        src2dst = None
        comb = down

    weight = torch.randn((D, width), dtype=torch.bfloat16) * 0.3
    cache = torch.randn((num_slots, width - 1, D), dtype=torch.bfloat16) * 0.4
    safe_idx = torch.tensor([3, 0, 5], dtype=torch.int32)
    cache_mask = torch.tensor([1, 1, 0], dtype=torch.uint8)
    attention = torch.randn((T, D), dtype=torch.float32) * 0.5
    partial = torch.randn((T, D), dtype=torch.float32) * 0.3 if has_partial else None
    track_mask = torch.tensor([1, 0, 1], dtype=torch.bool) if decode else None
    track_indices = torch.tensor([7, 8, 9], dtype=torch.int32) if decode else None

    cache_cpu = cache.clone()
    win_cpu = window.clone() if window is not None else None
    out = sgl_kernel.inkling2_routed_sconv_epilogue_cpu(
        down,
        src2dst,
        top_k,
        weight,
        cache_cpu,
        safe_idx,
        cache_mask,
        cu,
        si,
        attention,
        partial,
        track_mask,
        track_indices,
        win_cpu,
        round_bf16,
    )

    conv_ref, cache_ref, win_ref = _reference_short_conv(
        comb,
        weight,
        cache,
        safe_idx,
        cache_mask,
        cu,
        si,
        decode=decode,
        use_residual=True,
        track_mask=track_mask,
        track_indices=track_indices,
        window=window,
    )
    y_ref = conv_ref.float()
    out_ref = (
        (partial.float() + attention.float() + y_ref)
        if has_partial
        else (attention.float() + y_ref)
    )
    out_ref = out_ref.to(out.dtype)
    torch.testing.assert_close(out, out_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(cache_cpu, cache_ref, atol=0.0, rtol=0.0)
    if win_cpu is not None:
        torch.testing.assert_close(win_cpu, win_ref, atol=0.0, rtol=0.0)


def test_inkling2_update_sconv_cache_cpu():
    torch.manual_seed(11)
    D, W1 = 64, 3
    q_lens = [5, 2, 1, 0, 4]
    query_start_loc = torch.tensor([0, 5, 7, 8, 8, 12], dtype=torch.int32)
    x = torch.randn((12, D), dtype=torch.bfloat16)
    cache_indices = torch.tensor([1, 3, 2, 4, -1], dtype=torch.int32)
    has_initial_state = torch.tensor([True, True, False, True, True], dtype=torch.bool)
    cache = torch.randn((6, W1, D), dtype=torch.bfloat16)
    cache_ref = cache.clone()

    sgl_kernel.inkling2_update_sconv_cache_cpu(
        x, cache, cache_indices, has_initial_state, query_start_loc
    )

    for b, qlen in enumerate(q_lens):
        ci = int(cache_indices[b].item())
        if ci == -1 or qlen <= 0:
            continue
        stop = int(query_start_loc[b + 1].item())
        has_init = bool(has_initial_state[b].item())
        orig = cache_ref[ci].clone()
        for w in range(W1):
            if qlen >= W1 - w:
                cache_ref[ci, w] = x[stop - W1 + w]
            elif has_init:
                cache_ref[ci, w] = orig[w + qlen]
            else:
                cache_ref[ci, w] = 0

    torch.testing.assert_close(cache, cache_ref, atol=0.0, rtol=0.0)


def test_inkling2_update_sconv_cache_cpu_strided():
    import sgl_kernel  # noqa: F401

    torch.manual_seed(7)
    pool, W1, D = 8, 3, 64
    x = torch.randn(5, D, dtype=torch.bfloat16)
    backing = torch.randn(pool, D, W1, dtype=torch.bfloat16)
    cache_cpp = backing.transpose(1, 2)
    cache_ref = cache_cpp.clone()
    cache_indices = torch.tensor([1, 3, -1], dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True], dtype=torch.bool)
    query_start_loc = torch.tensor([0, 2, 5, 5], dtype=torch.int32)
    torch.ops.sgl_kernel.inkling2_update_sconv_cache_cpu(
        x, cache_cpp, cache_indices, has_initial_state, query_start_loc
    )
    for b, (s, e) in enumerate([(0, 2), (2, 5), (5, 5)]):
        ci = int(cache_indices[b].item())
        if ci == -1 or s == e:
            continue
        qlen = e - s
        if qlen >= W1:
            cache_ref[ci] = x[e - W1 : e]
        else:
            tail = (
                cache_ref[ci, qlen:].clone()
                if bool(has_initial_state[b])
                else torch.zeros_like(cache_ref[ci, qlen:])
            )
            cache_ref[ci] = torch.cat((tail, x[s:e]), dim=0)
    torch.testing.assert_close(cache_cpp, cache_ref)


def test_extend_attention_cpu_tree_mask_with_swa_and_rel_bias():
    import sgl_kernel  # noqa: F401

    torch.manual_seed(42)
    bs = 1
    prefix_len = 6
    qlen = 4
    num_heads, num_kv_heads, head_dim = 4, 2, 64
    rel_extent = 16
    swa = 4
    sm_scale = head_dim**-0.5

    q_ext = torch.randn(qlen, num_heads, head_dim, dtype=torch.bfloat16)
    k_ext = torch.randn(qlen, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v_ext = torch.randn(qlen, num_kv_heads, head_dim, dtype=torch.bfloat16)
    k_buf = torch.randn(16, num_kv_heads, head_dim, dtype=torch.bfloat16)
    v_buf = torch.randn(16, num_kv_heads, head_dim, dtype=torch.bfloat16)
    req_to_token = torch.arange(16, dtype=torch.int32).unsqueeze(0)
    req_pool_indices = torch.zeros(1, dtype=torch.int64)
    seq_lens = torch.tensor([prefix_len + qlen], dtype=torch.int64)
    extend_seq_lens = torch.tensor([qlen], dtype=torch.int32)
    extend_start_loc = torch.tensor([0], dtype=torch.int32)
    rel_bias = torch.randn(qlen, num_heads, rel_extent, dtype=torch.float32) * 0.2

    # Tree: 0 (root, depth 0), 1->0 (depth 1), 2->0 (depth 1), 3->2 (depth 2)
    tree_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )
    depths = [0, 1, 1, 2]
    o_ext = torch.empty(qlen, num_heads, head_dim, dtype=torch.float32)

    torch.ops.sgl_kernel.extend_attention_cpu(
        q_ext,
        k_ext,
        v_ext,
        o_ext,
        k_buf,
        v_buf,
        1.0,
        1.0,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        qlen,
        sm_scale,
        0.0,
        False,
        swa,
        None,
        None,
        tree_mask.reshape(-1),
        True,
        rel_bias,
        None,
        False,
    )

    k_all = (
        torch.cat((k_buf[:prefix_len], k_ext), dim=0)
        .repeat_interleave(num_heads // num_kv_heads, dim=1)
        .float()
    )
    v_all = (
        torch.cat((v_buf[:prefix_len], v_ext), dim=0)
        .repeat_interleave(num_heads // num_kv_heads, dim=1)
        .float()
    )
    ref = torch.zeros_like(o_ext)
    for r in range(qlen):
        qd = prefix_len + depths[r]
        q_r = q_ext[r].float()
        scores = torch.einsum("hd,khd->hk", q_r, k_all) * sm_scale
        for c in range(prefix_len + qlen):
            if c < prefix_len:
                dist = qd - c
                vis = dist < swa
            else:
                ec = c - prefix_len
                vis = bool(tree_mask[r, ec])
                dist = depths[r] - depths[ec]
                if dist >= swa:
                    vis = False
            if not vis:
                scores[:, c] = float("-inf")
            elif 0 <= dist < rel_extent:
                scores[:, c] += rel_bias[r, :, dist]
        probs = torch.softmax(scores, dim=-1)
        ref[r] = torch.einsum("hk,khd->hd", probs, v_all)
    torch.testing.assert_close(o_ext, ref, atol=2e-2, rtol=2e-2)

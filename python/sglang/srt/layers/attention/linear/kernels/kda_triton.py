import os
from typing import Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu

if not is_cpu():
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.srt.layers.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.fla.kda import chunk_kda

if is_npu():
    from sgl_kernel_npu.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update_npu
    fused_sigmoid_gating_delta_rule_update = fused_sigmoid_gating_delta_rule_update_npu
    from sglang.srt.hardware_backend.npu.kernels.kda_target_verify import (
        kda_target_verify_npu,
    )

_KDA_USE_TORCH_NATIVE = os.getenv("SGLANG_KDA_TORCH_NATIVE_DECODE", "0") == "1"
_KDA_USE_TORCH_NATIVE_EXTEND = os.getenv("SGLANG_KDA_TORCH_NATIVE_EXTEND", "0") == "1"


def _kda_precompute_gate(g, A_log, dt_bias):
    """Pre-activate the raw gate in PyTorch so chunk_kda can skip the
    kda_gate_chunk_cumsum Triton kernel (which fails to compile on NPU).

    Computes: g_act = -exp(A_log) * softplus(g + dt_bias)
    Then chunk_kda with A_log=None will only do chunk_local_cumsum on g_act.
    """
    g_dtype = g.dtype
    K_dim = g.shape[-1]
    g = g.float()
    A_log = A_log.float().view(1, 1, -1, 1)       # broadcast over K_dim
    dt_bias = dt_bias.float().view(1, 1, -1, K_dim)
    gate_x = g + dt_bias
    gate_act = -torch.exp(A_log) * F.softplus(gate_x, beta=1.0, threshold=20.0)
    return gate_act.to(g_dtype)


def kda_decode_torch_native(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
) -> torch.Tensor:
    """Torch-native reference for KDA decode.

    Reproduces the byte-identical logic of
    ``fused_sigmoid_gating_delta_rule_update`` with IS_KDA=True, using pure
    PyTorch operations.  Useful as a ground-truth oracle on Ascend NPU hardware
    where the Triton kernel may have layout or precision issues.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    orig_type = q.dtype
    B = q.shape[1]
    q_head_num = q.shape[2]
    k_head_num = k.shape[2]
    K_dim = q.shape[3]
    HV = v.shape[2]
    V_dim = v.shape[3]

    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)
    b_flat = b.reshape(B, HV).float()

    if use_qk_l2norm_in_kernel:
        q = q / (q.float().norm(p=2, dim=-1, keepdim=True) + 1e-6)
        k = k / (k.float().norm(p=2, dim=-1, keepdim=True) + 1e-6)
    q = q.float() * scale
    k = k.float()
    v = v.float()

    q_gqa_ratio = HV // q_head_num
    k_gqa_ratio = HV // k_head_num

    a = a.float().view(B, k_head_num, K_dim)
    dt_bias = dt_bias.float().view(k_head_num, K_dim)
    A_log = A_log.float().view(k_head_num, 1)

    x = a + dt_bias
    softplus_x = torch.where(
        x <= softplus_threshold,
        torch.log1p(torch.exp(x.to(torch.float32))) / softplus_beta,
        x,
    )
    gate = -torch.exp(A_log) * softplus_x
    gate_exp = torch.exp(gate)

    beta = torch.sigmoid(b_flat)

    out = torch.empty(B, HV, V_dim, device=q.device, dtype=q.dtype)
    ssm_pool = initial_state_source

    for tok in range(B):
        idx = initial_state_indices[tok].item()
        if idx < 0:
            out[tok] = 0.0
            continue

        state = ssm_pool[idx].float()
        g_exp_tok = gate_exp[tok].repeat_interleave(k_gqa_ratio, dim=0)
        beta_tok = beta[tok]
        k_tok = k[tok].repeat_interleave(k_gqa_ratio, dim=0)
        q_tok = q[tok].repeat_interleave(q_gqa_ratio, dim=0)

        state = state * g_exp_tok.unsqueeze(1)

        v_upd = v[tok] - (state @ k_tok.unsqueeze(-1)).squeeze(-1)

        v_upd = v_upd * beta_tok.unsqueeze(-1)

        state = state + v_upd.unsqueeze(-1) * k_tok.unsqueeze(1)

        o_tok = (state @ q_tok.unsqueeze(-1)).squeeze(-1)
        out[tok] = o_tok.to(q.dtype)

        ssm_pool[idx] = state.to(ssm_pool.dtype)

    return out.unsqueeze(0).to(orig_type)


def kda_target_verify_torch_native(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """NPU fallback for fixed-width linear-chain KDA target verification.

    Target-verify graph capture cannot perform the device-to-host copies used by
    ``Tensor.item()`` or ``Tensor.cpu()``. DSpark static verify has a fixed
    ``cache_steps`` tokens per request, so run the recurrence batched across
    requests and keep only the small, shape-static token loop in Python.
    """
    del cu_seqlens  # Static DSpark verify is laid out as B contiguous chains.

    if scale is None:
        scale = k.shape[-1] ** -0.5
    if q.shape[1] % cache_steps != 0:
        raise ValueError(
            "KDA NPU target verify expects a fixed-width layout: "
            f"tokens={q.shape[1]}, cache_steps={cache_steps}"
        )

    batch_size = q.shape[1] // cache_steps
    q_head_num = q.shape[2]
    k_head_num = k.shape[2]
    key_dim = q.shape[3]
    value_head_num = v.shape[2]
    value_dim = v.shape[3]
    q_gqa_ratio = value_head_num // q_head_num
    k_gqa_ratio = value_head_num // k_head_num

    q = q.squeeze(0).view(
        batch_size, cache_steps, q_head_num, key_dim
    )
    k = k.squeeze(0).view(
        batch_size, cache_steps, k_head_num, key_dim
    )
    v = v.squeeze(0).view(
        batch_size, cache_steps, value_head_num, value_dim
    )
    a = a.reshape(batch_size, cache_steps, k_head_num, key_dim)
    b = b.reshape(batch_size, cache_steps, value_head_num)

    cache_indices = initial_state_indices[:batch_size]
    valid_cache = cache_indices >= 0
    safe_cache_indices = cache_indices.clamp_min(0).to(torch.int64)
    state = initial_state_source.index_select(0, safe_cache_indices).float()
    state = state * valid_cache.view(batch_size, 1, 1, 1)

    scratch_indices = intermediate_state_indices[:batch_size].to(torch.int64)
    dt_bias = dt_bias.float().view(1, k_head_num, key_dim)
    A_log = A_log.float().view(1, k_head_num, 1)
    outputs = []

    for step in range(cache_steps):
        q_step = q[:, step].float()
        k_step = k[:, step].float()
        q_step = q_step / (
            q_step.norm(p=2, dim=-1, keepdim=True) + 1e-6
        )
        k_step = k_step / (
            k_step.norm(p=2, dim=-1, keepdim=True) + 1e-6
        )
        q_step = q_step.repeat_interleave(q_gqa_ratio, dim=1) * scale
        k_step = k_step.repeat_interleave(k_gqa_ratio, dim=1)

        gate = -torch.exp(A_log) * F.softplus(
            a[:, step].float() + dt_bias,
            beta=1.0,
            threshold=20.0,
        )
        gate = torch.exp(gate).repeat_interleave(k_gqa_ratio, dim=1)
        beta = torch.sigmoid(b[:, step].float())

        state = state * gate.unsqueeze(2)
        value = v[:, step].float()
        value = value - torch.matmul(
            state, k_step.unsqueeze(-1)
        ).squeeze(-1)
        value = value * beta.unsqueeze(-1)
        state = state + value.unsqueeze(-1) * k_step.unsqueeze(2)

        output = torch.matmul(
            state, q_step.unsqueeze(-1)
        ).squeeze(-1)
        outputs.append(output.to(v.dtype).unsqueeze(1))
        intermediate_states_buffer[:, step].index_copy_(
            0,
            scratch_indices,
            state.to(intermediate_states_buffer.dtype),
        )

    return torch.cat(outputs, dim=1).reshape(
        1, batch_size * cache_steps, value_head_num, value_dim
    )


def kda_extend_torch_native(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float = None,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor = None,
    dt_bias: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: torch.Tensor = None,
) -> torch.Tensor:
    """Torch-native KDA prefill (sequential recurrence).

    Replaces ``chunk_kda`` / ``fused_recurrent_kda`` with a pure PyTorch
    per-token loop.  Correctness reference on Ascend where the Triton
    JIT may produce zero-valued or silently incorrect results.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5

    B = q.shape[0]
    T = q.shape[1]
    H = q.shape[2]
    K_dim = q.shape[3]
    HV = v.shape[2]
    V_dim = v.shape[3]

    out_dtype = v.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    beta = beta.float()
    A_log = A_log.float().view(1, 1, -1, 1)
    dt_bias = dt_bias.float().view(1, 1, -1, K_dim)

    if use_qk_l2norm_in_kernel:
        q = q / (q.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
    q = q * scale

    # Activate gate: g_act = -exp(A_log) * softplus(g + dt_bias)
    gate_x = g.float() + dt_bias
    gate_act = -torch.exp(A_log) * F.softplus(gate_x, beta=1.0, threshold=20.0)
    gate_exp = torch.exp(gate_act)  # [1, T, H, K]

    ssm_pool = initial_state
    out = torch.empty(B, T, HV, V_dim, device=q.device, dtype=q.dtype)

    if cu_seqlens is not None and cu_seqlens.shape[0] > 2:
        seq_starts = cu_seqlens[:-1]
    else:
        seq_starts = torch.tensor([0], dtype=torch.long, device=cu_seqlens.device if cu_seqlens is not None else "cpu")

    for seq_i, start in enumerate(seq_starts):
        if cu_seqlens is not None:
            end = cu_seqlens[seq_i + 1]
        else:
            end = B * T

        idx = initial_state_indices[seq_i].item()
        if idx >= 0:
            state = ssm_pool[idx].float()  # [HV, V, K]
        else:
            state = torch.zeros(HV, V_dim, K_dim, device=q.device)

        gqa_ratio = HV // H
        for t in range(start, end):
            qt = q[0, t]  # [H, K]
            kt = k[0, t]  # [H, K]
            vt = v[0, t]  # [HV, V]
            ge = gate_exp[0, t]  # [H, K]
            bt = beta[0, t]  # [HV]

            if gqa_ratio > 1:
                kt = kt.repeat_interleave(gqa_ratio, dim=0)
                qt = qt.repeat_interleave(gqa_ratio, dim=0)
                ge = ge.repeat_interleave(gqa_ratio, dim=0)

            state = state * ge.unsqueeze(1)
            v_upd = vt - (state @ kt.unsqueeze(-1)).squeeze(-1)
            v_upd = v_upd * bt.unsqueeze(-1)
            state = state + v_upd.unsqueeze(-1) * kt.unsqueeze(1)
            ot = (state @ qt.unsqueeze(-1)).squeeze(-1)
            out[0, t] = ot.to(out.dtype)

        if idx >= 0:
            ssm_pool[idx] = state.to(ssm_pool.dtype)

    return out.to(out_dtype)


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    supports_packed_decode: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # KDA ReplaySSM buffered decode: drop-in for the packed decode, same
        # args plus the three per-layer ring caches + the per-row write cursor
        # (and optional radix-track force-flush). Uses the gate-generic kernel
        # with is_kda=True (per-K gate); g_cache is [num_slots, HV, L, K].
        # When any ring tensor / cursor is None (flag off) we fall through to
        # the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            K = ssm_states.shape[-1]  # ssm_states: [num_slots, HV, V, K]
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(B, num_v_heads, K).contiguous(),
                b=b.reshape(B, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, K).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
            )
            return out.transpose(0, 1)

        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        import os as _os
        _diag = _os.getenv("SGLANG_KDA_DEBUG", "0") == "1"
        import sys as _sys

        if _KDA_USE_TORCH_NATIVE:
            if _diag:
                s_norm = ssm_states[cache_indices[0]].float().norm().item() if cache_indices[0] >= 0 else -1
                print(f"[KDA-torch decode] q_norm={q.float().norm().item():.2f} "
                      f"k_norm={k.float().norm().item():.2f} v_norm={v.float().norm().item():.2f} "
                      f"state_norm_before={s_norm:.2f} a_norm={a.float().norm().item():.2f}",
                      file=_sys.stderr, flush=True)
            out = kda_decode_torch_native(
                A_log=A_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
                is_kda=True,
            )
            if _diag and cache_indices[0] >= 0:
                s_norm_after = ssm_states[cache_indices[0]].float().norm().item()
                print(f"[KDA-torch decode] out_norm={out.float().norm().item():.2f} "
                      f"state_norm_after={s_norm_after:.2f}",
                      file=_sys.stderr, flush=True)
            return out

        # Triton path with optional native comparison
        if _diag:
            ssm_backup = ssm_states.clone()
            q_bak, k_bak, v_bak, a_bak, b_bak = q.clone(), k.clone(), v.clone(), a.clone(), b.clone()

        out_triton = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
        )

        if _diag:
            ssm_native = ssm_backup.clone()
            # Log input shapes for first decode step only
            if not hasattr(self, "_decode_shape_logged"):
                self._decode_shape_logged = True
                print(f"[KDA-decode-shape] q={list(q_bak.shape)} k={list(k_bak.shape)} v={list(v_bak.shape)} "
                      f"a={list(a_bak.shape)} b={list(b_bak.shape)} A_log={list(A_log.shape)} "
                      f"dt_bias={list(dt_bias.shape)} ssm={list(ssm_backup.shape)} "
                      f"cache_idx={list(cache_indices.shape)} qsl={list(query_start_loc.shape)}",
                      file=_sys.stderr, flush=True)
                print(f"[KDA-decode-shape] q_norm={q_bak.float().norm().item():.4f} k_norm={k_bak.float().norm().item():.4f} "
                      f"v_norm={v_bak.float().norm().item():.4f} a_norm={a_bak.float().norm().item():.4f} "
                      f"b_norm={b_bak.float().norm().item():.4f} A_log_range=[{A_log.float().min().item():.4f},{A_log.float().max().item():.4f}] "
                      f"dt_bias_range=[{dt_bias.float().min().item():.4f},{dt_bias.float().max().item():.4f}]",
                      file=_sys.stderr, flush=True)
            out_native = kda_decode_torch_native(
                A_log=A_log,
                a=a_bak,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q_bak,
                k=k_bak,
                v=v_bak,
                b=b_bak,
                initial_state_source=ssm_native,
                initial_state_indices=cache_indices,
                scale=None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
                is_kda=True,
            )
            t_f = out_triton.float()
            n_f = out_native.float()
            diff = (t_f - n_f).abs()
            rel = diff.max().item() / (n_f.abs().max().item() + 1e-9)
            ssm_t = ssm_states[cache_indices[0]].float() if cache_indices[0] >= 0 else None
            ssm_n = ssm_native[cache_indices[0]].float() if cache_indices[0] >= 0 else None
            ssm_diff = (ssm_t - ssm_n).abs().max().item() if ssm_t is not None else -1
            ssm_t_norm = ssm_t.norm().item() if ssm_t is not None else -1
            ssm_n_norm = ssm_n.norm().item() if ssm_n is not None else -1
            print(f"[KDA-decode-cmp] triton_norm={t_f.norm().item():.4f} native_norm={n_f.norm().item():.4f} "
                  f"max_diff={diff.max().item():.6f} rel_err={rel:.4f} ssm_diff={ssm_diff:.6f} "
                  f"ssm_t_norm={ssm_t_norm:.4f} ssm_n_norm={ssm_n_norm:.4f} "
                  f"triton_nan={torch.isnan(t_f).any().item()} native_nan={torch.isnan(n_f).any().item()}",
                  file=_sys.stderr, flush=True)

        return out_triton

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if is_npu():
            return kda_target_verify_npu(
                A_log=A_log,
                dt_bias=dt_bias,
                q=q,
                k=k,
                v=v,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                intermediate_states_buffer=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=cache_steps,
            )
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, Optional[torch.Tensor]]:
        # Early input check (before any kernel call)
        import os as _os_early
        if _os_early.getenv("SGLANG_KDA_DEBUG", "0") == "1" and A_log is not None:
            import sys as _sys_early
            print(f"[KDA-extend-input] q_nan={torch.isnan(q).any().item()} "
                  f"k_nan={torch.isnan(k).any().item()} "
                  f"v_nan={torch.isnan(v).any().item()} "
                  f"g_nan={torch.isnan(g).any().item()} "
                  f"beta_nan={torch.isnan(beta).any().item()} "
                  f"ssm_nan={torch.isnan(ssm_states).any().item()} "
                  f"q_norm={q.float().norm().item():.4f} "
                  f"k_norm={k.float().norm().item():.4f} "
                  f"v_norm={v.float().norm().item():.4f} "
                  f"g_norm={g.float().norm().item():.4f} "
                  f"beta_norm={beta.float().norm().item():.4f} "
                  f"ssm_norm={ssm_states.float().norm().item():.4f} "
                  f"q_shape={list(q.shape)}",
                  file=_sys_early.stderr, flush=True)
        if _KDA_USE_TORCH_NATIVE_EXTEND and A_log is not None:
            import os as _os
            if _os.getenv("SGLANG_KDA_DEBUG", "0") == "1":
                import sys as _sys
                print(f"[KDA-torch extend] q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} "
                      f"g={list(g.shape)} beta={list(beta.shape)} "
                      f"seqlen={query_start_loc[-1].item() if query_start_loc is not None else '?'}",
                      file=_sys.stderr, flush=True)
            out = kda_extend_torch_native(
                q=q, k=k, v=v, g=g, beta=beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
            return out, None, None

        # kimi_linear.py (212 version) already calls fused_kda_gate to activate
        # the gate before passing it here. So g is already the activated gate
        # (-exp(A_log) * softplus(g + dt_bias)). We pass A_log=None to chunk_kda
        # so it only does chunk_local_cumsum on the already-activated g.
        import os as _os
        _kda_debug = _os.getenv("SGLANG_KDA_DEBUG", "0") == "1"

        # Backup ssm_states BEFORE triton call to detect contamination
        ssm_pre_triton = ssm_states.clone() if _kda_debug else None
        ssm_pre_nan = torch.isnan(ssm_states).any().item() if _kda_debug else False

        out_triton, last_recurrent_state, h = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=query_start_loc,
            A_log=None,
            dt_bias=None,
            lower_bound=lower_bound,
        )

        # Debug: compare triton vs native on NPU
        if _kda_debug and A_log is not None:
            import sys as _sys
            ssm_post_nan = torch.isnan(ssm_states).any().item()
            # Restore pre-triton ssm_states for native comparison
            ssm_states.copy_(ssm_pre_triton)
            out_native = kda_extend_torch_native(
                q=q, k=k, v=v, g=g_raw, beta=beta,
                scale=None,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                A_log=A_log,
                dt_bias=dt_bias,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=query_start_loc,
            )
            ssm_post_native_nan = torch.isnan(ssm_states).any().item()
            # Restore triton-updated ssm_states
            ssm_states.copy_(ssm_pre_triton)
            diff = (out_triton.float() - out_native.float()).abs()
            # Gate statistics
            g_raw_f = g_raw.float()
            A_log_f = A_log.float()
            dt_bias_f = dt_bias.float()
            gate_x = g_raw_f + dt_bias_f.view(1, 1, -1, g_raw_f.shape[-1])
            gate_act = -torch.exp(A_log_f.view(1, 1, -1, 1)) * F.softplus(gate_x, beta=1.0, threshold=20.0)
            gate_exp = torch.exp(gate_act)
            print(f"[KDA-extend-cmp] triton_norm={out_triton.float().norm().item():.4f} "
                  f"native_norm={out_native.float().norm().item():.4f} "
                  f"triton_has_nan={torch.isnan(out_triton).any().item()} "
                  f"ssm_post_triton_nan={ssm_post_nan} "
                  f"ssm_post_native_nan={ssm_post_native_nan} "
                  f"q_norm={q.float().norm().item():.4f} "
                  f"g_raw_min={g_raw_f.min().item():.4f} g_raw_max={g_raw_f.max().item():.4f} "
                  f"A_log_min={A_log_f.min().item():.4f} A_log_max={A_log_f.max().item():.4f} "
                  f"gate_act_min={gate_act.min().item():.4f} gate_act_max={gate_act.max().item():.4f} "
                  f"gate_exp_min={gate_exp.min().item():.4e} gate_exp_max={gate_exp.max().item():.4e} "
                  f"beta_min={beta.float().min().item():.4f} beta_max={beta.float().max().item():.4f}",
                  file=_sys.stderr, flush=True)

        return out_triton, last_recurrent_state, h

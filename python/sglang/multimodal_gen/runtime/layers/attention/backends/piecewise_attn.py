# Copyright (c) 2025-2026, Haopeng Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import atexit
import json
import os
import threading
from contextlib import nullcontext

import torch
import triton
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from triton.tools.tensor_descriptor import TensorDescriptor

logger = init_logger(__name__)

_COMPILED_TAYLOR_ERROR_BLOCK_INDICES = None
_PIECEWISE_STATS_LOCK = threading.Lock()
_PIECEWISE_STATS_REGISTERED = False
_PIECEWISE_STATS: dict[str, object] = {
    "total_calls": 0,
    "sparse_calls": 0,
    "fallback_calls": 0,
    "by_stage": {},
    "by_prefix": {},
    "by_reason": {},
    "by_shape": {},
}


def _piecewise_stats_path() -> str:
    return os.environ.get("SGLANG_PIECEWISE_ATTN_STATS_PATH", "")


def _piecewise_stats_enabled() -> bool:
    return bool(_piecewise_stats_path())


def _piecewise_stats_flush_every() -> int:
    try:
        return max(0, int(os.environ.get("SGLANG_PIECEWISE_ATTN_STATS_FLUSH_EVERY", "100")))
    except ValueError:
        return 100


def _piecewise_dump_stats() -> None:
    path = _piecewise_stats_path()
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with _PIECEWISE_STATS_LOCK:
            payload = json.loads(json.dumps(_PIECEWISE_STATS))
        tmp_path = f"{abs_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp_path, abs_path)
    except Exception as exc:
        logger.warning("Failed to dump piecewise attention stats: %s", exc)


def _piecewise_register_stats_dump() -> None:
    global _PIECEWISE_STATS_REGISTERED
    if _PIECEWISE_STATS_REGISTERED:
        return
    atexit.register(_piecewise_dump_stats)
    _PIECEWISE_STATS_REGISTERED = True


def _piecewise_bump(mapping: dict, key: str, branch: str) -> None:
    item = mapping.setdefault(key, {"total": 0, "sparse": 0, "fallback": 0})
    item["total"] = int(item["total"]) + 1
    item[branch] = int(item[branch]) + 1


def _piecewise_record_stats(
    *,
    prefix: str,
    stage: object,
    branch: str,
    reason: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    density: float,
    block_size: int | None = None,
    exact_density: float | None = None,
) -> None:
    if not _piecewise_stats_enabled():
        return
    _piecewise_register_stats_dump()
    stage_key = str(stage if stage is not None else "unknown")
    prefix_key = prefix or "<unknown>"
    reason_key = reason or branch
    shape_key = (
        f"q={tuple(int(x) for x in query.shape)};"
        f"k={tuple(int(x) for x in key.shape)};"
        f"v={tuple(int(x) for x in value.shape)};"
        f"density={density:.6f};block={block_size}"
    )
    with _PIECEWISE_STATS_LOCK:
        _PIECEWISE_STATS["total_calls"] = int(_PIECEWISE_STATS["total_calls"]) + 1
        if branch == "sparse":
            _PIECEWISE_STATS["sparse_calls"] = int(_PIECEWISE_STATS["sparse_calls"]) + 1
        else:
            _PIECEWISE_STATS["fallback_calls"] = int(_PIECEWISE_STATS["fallback_calls"]) + 1
        _piecewise_bump(_PIECEWISE_STATS["by_stage"], stage_key, branch)
        _piecewise_bump(_PIECEWISE_STATS["by_prefix"], prefix_key, branch)
        _piecewise_bump(_PIECEWISE_STATS["by_reason"], reason_key, branch)
        _piecewise_bump(_PIECEWISE_STATS["by_shape"], shape_key, branch)
        if exact_density is not None:
            shape_item = _PIECEWISE_STATS["by_shape"][shape_key]
            shape_item["exact_density"] = exact_density
        total = int(_PIECEWISE_STATS["total_calls"])
    flush_every = _piecewise_stats_flush_every()
    if flush_every and total % flush_every == 0:
        _piecewise_dump_stats()


def _parse_int_set(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        out = set()
        for item in value:
            out.update(_parse_int_set(item))
        return out
    text = str(value).strip()
    if not text or text.lower() in ("none", "false", "no"):
        return set()
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            if end < start:
                start, end = end, start
            out.update(range(start, end + 1))
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _cfg_or_env_int_set(cfg: dict, cfg_key: str, env_key: str) -> set[int]:
    value = cfg.get(cfg_key, None)
    if value is None:
        value = os.environ.get(env_key)
    return _parse_int_set(value)


def _ltx2_layer_idx_from_prefix(prefix: str) -> int | None:
    marker = "transformer_blocks."
    pos = prefix.find(marker)
    if pos < 0:
        return None
    start = pos + len(marker)
    end = start
    while end < len(prefix) and prefix[end].isdigit():
        end += 1
    if end == start:
        return None
    try:
        return int(prefix[start:end])
    except ValueError:
        return None

_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _make_tma_allocator():
    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    return alloc_fn


def build_block_map(indices, nt_kv):
    block_map = torch.zeros(*indices.shape[:-1], nt_kv, device=indices.device, dtype=torch.int8)
    block_map.scatter_(-1, indices.to(torch.long), 1)
    return block_map.contiguous()


@triton.jit
def chunk_reduce_kv_kernel(
    k,
    v,
    kc,
    vc,
    k_var,   # [B*H, N]
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    p_v = tl.make_tensor_descriptor(v + i_bh * T * V, (T, V), (V, 1), (BT, BV))

    b_k = p_k.load([i_t * BT, 0])
    b_v = p_v.load([i_t * BT, 0])

    b_kc = tl.sum(b_k, axis=0) / block_size
    b_vc = tl.sum(b_v, axis=0)

    # Var(K_j) = E[||k||^2] - ||E[k]||^2
    mean_norm = tl.sum(b_k * b_k) / block_size

    kc_norm = tl.sum(b_kc * b_kc, axis=0)
    b_k_var = tl.maximum(mean_norm - kc_norm, 0.0)

    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    p_vc = tl.make_block_ptr(vc + i_bh * N * V + i_t * V, (V,), (1,), (0,), (BV,), (0,))

    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(p_vc, b_vc.to(p_vc.dtype.element_ty), boundary_check=(0,))
    tl.store(k_var + i_bh * N + i_t, b_k_var)


@triton.jit
def chunk_reduce_k_kernel(
    k,
    kc,
    k_var,   # [B*H, N]
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_k = tl.make_tensor_descriptor(k + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_k = p_k.load([i_t * BT, 0])

    b_kc = tl.sum(b_k, axis=0) / block_size

    # Var(K_j) = E[||k||^2] - ||E[k]||^2
    mean_norm = tl.sum(b_k * b_k) / block_size

    kc_norm = tl.sum(b_kc * b_kc, axis=0)
    b_k_var = tl.maximum(mean_norm - kc_norm, 0.0)

    p_kc = tl.make_block_ptr(kc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))

    tl.store(p_kc, b_kc.to(p_kc.dtype.element_ty), boundary_check=(0,))
    tl.store(k_var + i_bh * N + i_t, b_k_var)


@triton.jit
def chunk_reduce_q_kernel(
    q,
    qc,
    T,
    N: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    block_size = tl.minimum(BT, T - i_t * BT).to(tl.float32)

    p_q = tl.make_tensor_descriptor(q + i_bh * T * K, (T, K), (K, 1), (BT, BK))
    b_q = p_q.load([i_t * BT, 0])

    b_qc = tl.sum(b_q, axis=0) / block_size

    p_qc = tl.make_block_ptr(qc + i_bh * N * K + i_t * K, (K,), (1,), (0,), (BK,), (0,))
    tl.store(p_qc, b_qc.to(p_qc.dtype.element_ty), boundary_check=(0,))


def chunk_reduce_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    include_v_centroid: bool = True,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]

    N_Q = triton.cdiv(T_Q, block_size)
    N_KV = triton.cdiv(T_KV, block_size)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    qc = torch.empty(B, H, N_Q, K, device=q.device, dtype=q.dtype)
    kc = torch.empty(B, H, N_KV, K, device=k.device, dtype=k.dtype)
    vc = (
        torch.empty(B, H, N_KV, V, device=v.device, dtype=v.dtype)
        if include_v_centroid
        else None
    )

    # scalar variance proxy per KV block
    k_var = torch.empty(B, H, N_KV, device=k.device, dtype=k.dtype)

    chunk_reduce_q_kernel[(N_Q, B * H)](
        q=q,
        qc=qc,
        T=T_Q,
        N=N_Q,
        K=K,
        BT=block_size,
        BK=BK,
        num_warps=4,
        num_stages=2,
    )

    if include_v_centroid:
        chunk_reduce_kv_kernel[(N_KV, B * H)](
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            k_var=k_var,
            T=T_KV,
            N=N_KV,
            K=K,
            V=V,
            BT=block_size,
            BK=BK,
            BV=BV,
            num_warps=4,
            num_stages=3,
        )
    else:
        chunk_reduce_k_kernel[(N_KV, B * H)](
            k=k,
            kc=kc,
            k_var=k_var,
            T=T_KV,
            N=N_KV,
            K=K,
            BT=block_size,
            BK=BK,
            num_warps=4,
            num_stages=3,
        )

    return qc, kc, vc, k_var


@triton.jit
def piecewise_attn_fwd_kernel(
    q_desc,
    k_desc,
    v_desc,
    o_desc,
    kc,
    vc,
    lse,
    indices,
    scale,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    NS: tl.constexpr,
    B_NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    APPROX_REMAINDER: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    token_offsets = tl.arange(0, BT)
    token_offsets = tl.max_contiguous(token_offsets, BT)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])

    sm_scale = scale * 1.44269504
    acc = tl.zeros([BT, BV], dtype=tl.float32)
    l_i = tl.zeros((BT,), dtype=tl.float32)
    m_i = tl.zeros((BT,), dtype=tl.float32) - float("inf")

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        kv_start = i_n * BT
        tl.multiple_of(kv_start, BT)

        b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])

        b_s = tl.dot(b_q, b_k.T) * sm_scale
        b_s += tl.where((kv_start + token_offsets)[None, :] < T_KV, 0, float("-inf"))

        new_m = tl.maximum(m_i, tl.max(b_s,  axis=1))
        alpha = tl.math.exp2(m_i - new_m)
        score = tl.math.exp2(b_s - new_m[:, None])

        b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])

        l_i = l_i * alpha + tl.sum(score, axis=1)
        acc = acc * alpha[:, None] + tl.dot(score.to(b_v.dtype), b_v)
        m_i = new_m

    offs_n_idx = tl.arange(0, B_NS)
    selected = tl.load(
        indices + i_bh * NT_Q * NS + i_t * NS + offs_n_idx,
        mask=offs_n_idx < NS,
        other=-1,
    )

    if APPROX_REMAINDER:
        for start_n in range(0, NT_KV, GROUP_SIZE):
            p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (GROUP_SIZE, BK))
            b_kc = p_kc.load([start_n, 0])

            chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
            is_selected = chunk_indices[:, None] == selected[None, :]
            selected_mask = tl.max(is_selected, axis=1)
            valid_mask = (chunk_indices < NT_KV) & (selected_mask == 0)

            current_lens = tl.minimum(BT, tl.maximum(0, T_KV - chunk_indices * BT)).to(tl.float32)

            b_s_mean = tl.dot(b_q, b_kc.T) * sm_scale
            b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))

            new_m = tl.maximum(m_i, tl.max(b_s_mean, axis=1))
            alpha = tl.math.exp2(m_i - new_m)

            prob_chunk = tl.math.exp2(b_s_mean - new_m[:, None])

            p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (GROUP_SIZE, BV))
            b_vc = p_vc.load([start_n, i_v * BV])

            acc = acc * alpha[:, None] + tl.dot(prob_chunk.to(b_vc.dtype), b_vc)
            l_i = l_i * alpha + tl.sum(prob_chunk * current_lens[None, :], axis=1)
            m_i = new_m

    acc = acc / l_i[:, None]
    m_i += tl.math.log2(l_i)

    p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
    tl.store(p_lse, m_i, boundary_check=(0,))

    o_desc.store([i_bh, q_start, i_v * BV], acc[None, :, :])


@triton.jit
def attn_bwd_preprocess(
    o_desc,
    do_desc,
    delta,
    T,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)

    t_start = i_t * BT
    tl.multiple_of(t_start, BT)

    b_o = o_desc.load([i_bh, t_start, 0]).reshape([BT, BV])
    b_do = do_desc.load([i_bh, t_start, 0]).reshape([BT, BV])
    b_delta = tl.sum(b_o.to(tl.float32) * b_do.to(tl.float32), axis=1)

    p_delta = tl.make_block_ptr(delta + i_bh * T, (T,), (1,), (t_start,), (BT,), (0,))
    tl.store(p_delta, b_delta, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"GROUP_SIZE": GROUP_SIZE}, num_warps=num_warps, num_stages=num_stages)
        for GROUP_SIZE in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ],
    key=["T_Q", "T_KV", "K", "V", "BT", "NS"],
)
@triton.jit
def piecewise_attn_bwd_dq_kernel(
    do_desc,
    q_desc,
    k_desc,
    v_desc,
    kc,
    vc,
    lse,
    delta,
    dq,
    indices,
    block_map,
    scale,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q,
    NT_KV,
    NS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    q_start = i_t * BT
    tl.multiple_of(q_start, BT)

    b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
    b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])

    p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

    b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
    b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

    sm_scale = scale * 1.44269504
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    offs_bt = tl.arange(0, BT)
    offs_bt = tl.max_contiguous(offs_bt, BT)

    for i in range(NS):
        i_n = tl.load(indices + i_bh * NT_Q * NS + i_t * NS + i).to(tl.int32)
        bos = i_n * BT
        tl.multiple_of(bos, BT)
        offs_n = bos + offs_bt

        b_k = k_desc.load([i_bh, bos, 0]).reshape([BT, BK])
        b_v = v_desc.load([i_bh, bos, i_v * BV]).reshape([BT, BV])

        b_s = tl.dot(b_q, tl.trans(b_k)) * sm_scale
        b_s += tl.where(offs_n[None, :] < T_KV, 0, float("-inf"))
        b_p = tl.math.exp2(b_s - b_lse[:, None])

        b_term1 = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_term1 - b_D[:, None])

        b_dq += tl.dot(b_ds.to(b_k.dtype), b_k) * scale

    p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (GROUP_SIZE, BK))
    p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (GROUP_SIZE, BV))

    for start_n in range(0, NT_KV, GROUP_SIZE):
        tl.multiple_of(start_n, GROUP_SIZE)

        chunk_indices = start_n + tl.arange(0, GROUP_SIZE)
        b_kc = p_kc.load([start_n, 0])
        selected = tl.load(
            block_map + (i_bh * NT_Q + i_t) * NT_KV + chunk_indices,
            mask=chunk_indices < NT_KV,
            other=1,
        )
        valid_mask = (chunk_indices < NT_KV) & (selected == 0)

        current_lens = tl.minimum(BT, tl.maximum(0, T_KV - chunk_indices * BT)).to(tl.float32)
        b_s_mean = tl.dot(b_q, tl.trans(b_kc)) * sm_scale
        b_s_mean = tl.where(valid_mask[None, :], b_s_mean, float("-inf"))
        b_p = tl.math.exp2(b_s_mean - b_lse[:, None])

        b_vc = p_vc.load([start_n, i_v * BV])

        b_term1 = tl.dot(b_do, tl.trans(b_vc))
        b_ds = b_p * (b_term1 - b_D[:, None] * current_lens[None, :])

        b_dq += tl.dot(b_ds.to(b_kc.dtype), b_kc) * scale

    offs_q_n = tl.arange(0, BK)
    offs_q = q_start + tl.arange(0, BT)
    ptr_dq = dq + (i_bh * T_Q * K + offs_q[:, None] * K + offs_q_n[None, :])
    tl.store(ptr_dq, b_dq, mask=(offs_q[:, None] < T_Q) & (offs_q_n[None, :] < K))


@triton.autotune(
    configs=[
        triton.Config({"BN": BN}, num_warps=num_warps, num_stages=num_stages)
        for BN in [32, 64, 128]
        for num_warps in [4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["T_Q", "T_KV", "K", "V", "BT", "BN"],
)
@triton.jit
def piecewise_attn_bwd_approx_dkdv_kernel(
    do_desc,
    q_desc,
    kc,
    vc,
    lse,
    delta,
    dkc_grad,
    dvc_grad,
    block_map,
    scale: tl.constexpr,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
    BN: tl.constexpr,
):
    i_v, i_kv_group, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    offs_c = i_kv_group * BN + tl.arange(0, BN)

    p_kc = tl.make_tensor_descriptor(kc + i_bh * NT_KV * K, (NT_KV, K), (K, 1), (BN, BK))
    p_vc = tl.make_tensor_descriptor(vc + i_bh * NT_KV * V, (NT_KV, V), (V, 1), (BN, BV))

    b_kc = p_kc.load([i_kv_group * BN, 0])
    b_vc = p_vc.load([i_kv_group * BN, i_v * BV])

    sm_scale: tl.constexpr = scale * 1.44269504
    b_dkc = tl.zeros([BN, BK], dtype=tl.float32)
    b_dvc = tl.zeros([BN, BV], dtype=tl.float32)

    for i_q in tl.range(0, NT_Q, 1, num_stages=1):
        q_start = i_q * BT
        tl.multiple_of(q_start, BT)

        q_offs = q_start + tl.arange(0, BT)
        selected = tl.load(block_map + (i_bh * NT_Q + i_q) * NT_KV + offs_c, mask=offs_c < NT_KV, other=1)
        valid_chunk = (offs_c < NT_KV) & (selected == 0)

        b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
        b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])

        p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
        p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
        b_lse = tl.where(q_offs < T_Q, b_lse, float("inf"))
        b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

        current_lens = tl.minimum(BT, tl.maximum(0, T_KV - offs_c * BT)).to(tl.float32)
        safe_lens = tl.maximum(current_lens, 1.0)
        b_s_t = tl.dot(b_kc, tl.trans(b_q)) * sm_scale
        b_p_t = tl.math.exp2(b_s_t - b_lse[None, :])
        b_p_t = tl.where(valid_chunk[:, None] & (q_offs[None, :] < T_Q), b_p_t, 0.0)

        b_dvc += tl.dot(b_p_t.to(b_do.dtype), b_do)
        b_dp_t = tl.dot(b_vc, tl.trans(b_do))
        b_ds_t = b_p_t * (b_dp_t - current_lens[:, None] * b_D[None, :])
        b_dkc += tl.dot(b_ds_t.to(b_q.dtype), b_q) * scale / safe_lens[:, None]

    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)

    tl.store(
        dkc_grad + i_bh * NT_KV * K + offs_c[:, None] * K + offs_k[None, :],
        b_dkc,
        mask=(offs_c[:, None] < NT_KV) & (offs_k[None, :] < K),
    )
    tl.store(
        dvc_grad + i_bh * NT_KV * V + offs_c[:, None] * V + offs_v[None, :],
        b_dvc,
        mask=(offs_c[:, None] < NT_KV) & (offs_v[None, :] < V),
    )


@triton.jit
def piecewise_attn_bwd_exact_dkdv_kernel(
    do_desc,
    q_desc,
    k_desc,
    v_desc,
    lse,
    delta,
    dkc_grad,
    dvc_grad,
    dk,
    dv,
    block_map,
    scale: tl.constexpr,
    T_Q,
    T_KV,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_Q: tl.constexpr,
    NT_KV: tl.constexpr,
):
    i_v, i_kv, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    kv_start = i_kv * BT
    tl.multiple_of(kv_start, BT)

    offs_n = kv_start + tl.arange(0, BT)

    b_k = k_desc.load([i_bh, kv_start, 0]).reshape([BT, BK])
    b_v = v_desc.load([i_bh, kv_start, i_v * BV]).reshape([BT, BV])

    sm_scale = scale * 1.44269504
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    for i_q in tl.range(0, NT_Q, 1, num_stages=1):
        selected = tl.load(block_map + (i_bh * NT_Q + i_q) * NT_KV + i_kv)
        if selected == 1:
            q_start = i_q * BT
            tl.multiple_of(q_start, BT)

            q_offs = q_start + tl.arange(0, BT)
            b_q = q_desc.load([i_bh, q_start, 0]).reshape([BT, BK])
            b_do = do_desc.load([i_bh, q_start, i_v * BV]).reshape([BT, BV])

            p_lse = tl.make_block_ptr(lse + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))
            p_delta = tl.make_block_ptr(delta + i_bh * T_Q, (T_Q,), (1,), (q_start,), (BT,), (0,))

            b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
            b_lse = tl.where(q_offs < T_Q, b_lse, float("inf"))
            b_D = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

            b_s_t = tl.dot(b_k, tl.trans(b_q)) * sm_scale
            b_p_t = tl.math.exp2(b_s_t - b_lse[None, :])
            b_p_t = tl.where(offs_n[:, None] < T_KV, b_p_t, 0.0)

            b_dv += tl.dot(b_p_t.to(b_do.dtype), b_do)
            b_dp_t = tl.dot(b_v, tl.trans(b_do))
            b_ds_t = b_p_t * (b_dp_t - b_D[None, :])
            b_dk += tl.dot(b_ds_t.to(b_q.dtype), b_q)

    offs_k = tl.arange(0, BK)
    offs_v = i_v * BV + tl.arange(0, BV)

    b_dkc = tl.load(
        dkc_grad + i_bh * NT_KV * K + i_kv * K + offs_k,
        mask=offs_k < K,
        other=0.0,
    )
    tl.store(
        dk + i_bh * T_KV * K + offs_n[:, None] * K + offs_k[None, :],
        b_dk * scale + b_dkc[None, :],
        mask=(offs_n[:, None] < T_KV) & (offs_k[None, :] < K),
    )

    b_dvc = tl.load(
        dvc_grad + i_bh * NT_KV * V + i_kv * V + offs_v,
        mask=offs_v < V,
        other=0.0,
    )
    tl.store(
        dv + i_bh * T_KV * V + offs_n[:, None] * V + offs_v[None, :],
        b_dv + b_dvc[None, :],
        mask=(offs_n[:, None] < T_KV) & (offs_v[None, :] < V),
    )


def piecewise_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float,
    approx_remainder: bool = True,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]
    BT, NS = block_size, block_indices.shape[-1]

    o = torch.empty(B, H, T_Q, V, device=q.device, dtype=v.dtype)
    lse = torch.empty(B, H, T_Q, device=q.device, dtype=torch.float)

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    B_NS = triton.next_power_of_2(NS)

    NT_Q = triton.cdiv(T_Q, BT)
    NT_KV = triton.cdiv(T_KV, BT)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, block_size, BK])
    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, block_size, BV])

    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, block_size, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, block_size, BV])

    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    piecewise_attn_fwd_kernel[grid](
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        o_desc=o_desc,
        kc=kc,
        vc=vc,
        lse=lse,
        indices=block_indices,
        scale=scale,
        T_Q=T_Q,
        T_KV=T_KV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NS=NS,
        B_NS=B_NS,
        NT_Q=NT_Q,
        NT_KV=NT_KV,
        GROUP_SIZE=64,
        APPROX_REMAINDER=approx_remainder,
        num_warps=4,
        num_stages=2,
    )
    return o, lse


def piecewise_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    kc: torch.Tensor,
    vc: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float,
):
    B, H, T_Q, K, T_KV, V = *q.shape, *v.shape[-2:]
    BT, NS = block_size, block_indices.shape[-1]

    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)

    NT_Q = triton.cdiv(T_Q, BT)
    NT_KV = triton.cdiv(T_KV, BT)

    delta = torch.empty_like(lse)

    o_desc = TensorDescriptor.from_tensor(o.reshape(B * H, T_Q, V), [1, BT, BV])
    do_desc = TensorDescriptor.from_tensor(do.reshape(B * H, T_Q, V), [1, BT, BV])

    attn_bwd_preprocess[(NT_Q, B * H)](
        o_desc=o_desc,
        do_desc=do_desc,
        delta=delta,
        T=T_Q,
        BT=BT,
        BV=BV,
    )

    block_map = build_block_map(block_indices, NT_KV)

    dq = torch.empty_like(q)

    q_desc = TensorDescriptor.from_tensor(q.reshape(B * H, T_Q, K), [1, BT, BK])
    k_desc = TensorDescriptor.from_tensor(k.reshape(B * H, T_KV, K), [1, BT, BK])
    v_desc = TensorDescriptor.from_tensor(v.reshape(B * H, T_KV, V), [1, BT, BV])

    grid = (triton.cdiv(V, BV), NT_Q, B * H)
    piecewise_attn_bwd_dq_kernel[grid](
        do_desc=do_desc,
        q_desc=q_desc,
        k_desc=k_desc,
        v_desc=v_desc,
        kc=kc,
        vc=vc,
        lse=lse,
        delta=delta,
        dq=dq,
        indices=block_indices,
        block_map=block_map,
        scale=scale,
        T_Q=T_Q,
        T_KV=T_KV,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        NT_Q=NT_Q,
        NT_KV=NT_KV,
        NS=NS,
    )

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    dkc_grad = torch.empty(B, H, NT_KV, K, device=k.device, dtype=kc.dtype)
    dvc_grad = torch.empty(B, H, NT_KV, V, device=v.device, dtype=vc.dtype)

    grid = lambda META: (triton.cdiv(V, BV), triton.cdiv(NT_KV, META["BN"]), B * H)
    piecewise_attn_bwd_approx_dkdv_kernel[grid](
        do_desc,
        q_desc,
        kc,
        vc,
        lse,
        delta,
        dkc_grad,
        dvc_grad,
        block_map,
        scale,
        T_Q,
        T_KV,
        K,
        V,
        BT,
        BK,
        BV,
        NT_Q,
        NT_KV,
    )

    grid = (triton.cdiv(V, BV), NT_KV, B * H)
    piecewise_attn_bwd_exact_dkdv_kernel[grid](
        do_desc,
        q_desc,
        k_desc,
        v_desc,
        lse,
        delta,
        dkc_grad,
        dvc_grad,
        dk,
        dv,
        block_map,
        scale,
        T_Q,
        T_KV,
        K,
        V,
        BT,
        BK,
        BV,
        NT_Q,
        NT_KV,
    )
    return dq, dk, dv


@torch.no_grad()
def taylor_error_block_indices(
    qc: torch.Tensor,       # [B, H, NT_Q, K]
    kc: torch.Tensor,       # [B, H, NT_KV, K]
    k_var: torch.Tensor,    # [B, H, NT_KV]
    density: float,
    scale: float,
    eps: float = 1e-8,
):
    NT_KV = kc.shape[2]

    top_k = max(1, int(density * NT_KV))
    top_k = min(top_k, NT_KV)

    # [B, H, NT_Q, NT_KV]
    route_score = torch.einsum("bhik,bhjk->bhij", qc, kc)
    route_score.mul_(scale)

    # Use log-domain score to avoid exp overflow.
    # Equivalent to topk(exp(2*logit) * k_var).
    log_k_var = torch.log(k_var.clamp_min(eps)).unsqueeze(-2)  # [B,H,1,NT_KV]
    route_score.add_(log_k_var)

    block_indices = torch.topk(
        route_score,
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    return block_indices


def _should_compile_piecewise_route() -> bool:
    return os.environ.get("SGLANG_PIECEWISE_ATTN_COMPILE_ROUTE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _compiled_taylor_error_block_indices():
    global _COMPILED_TAYLOR_ERROR_BLOCK_INDICES
    if _COMPILED_TAYLOR_ERROR_BLOCK_INDICES is None:
        mode = os.environ.get(
            "SGLANG_PIECEWISE_ATTN_COMPILE_ROUTE_MODE",
            "max-autotune-no-cudagraphs",
        )
        _COMPILED_TAYLOR_ERROR_BLOCK_INDICES = torch.compile(
            taylor_error_block_indices,
            mode=mode,
            fullgraph=False,
        )
    return _COMPILED_TAYLOR_ERROR_BLOCK_INDICES


class PiecewiseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, density, block_size, scale):

        triton.set_allocator(_make_tma_allocator())

        qc, kc, vc, k_var = chunk_reduce_qkv(q=q, k=k, v=v, block_size=block_size)

        block_indices = taylor_error_block_indices(
            qc=qc,
            kc=kc,
            k_var=k_var,
            density=density,
            scale=scale,
        )
        o, lse = piecewise_attn_fwd(
            q=q,
            k=k,
            v=v,
            kc=kc,
            vc=vc,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale
        )

        ctx.save_for_backward(q, k, v, kc, vc, o, lse, block_indices)
        ctx.scale = scale
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, kc, vc, o, lse, block_indices = ctx.saved_tensors

        scale = ctx.scale
        block_size = ctx.block_size

        triton.set_allocator(_make_tma_allocator())

        dq, dk, dv = piecewise_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            kc=kc,
            vc=vc,
            do=do,
            lse=lse,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
        )

        return dq, dk, dv, None, None, None


def piecewise_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    density: float = 0.1,
    block_size: int = 64,
) -> torch.Tensor:
    """Public entry point for PISA piecewise sparse attention.

    Args:
        q: queries of shape [B, H, T, K].
        k: keys of shape [B, H, T, K].
        v: values of shape [B, H, T, V].
        scale: attention softmax scale (default 1/sqrt(K)).
        density: fraction of KV blocks kept exactly.
        block_size: tile size for the sparse routing.
    """
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    return PiecewiseAttentionFunction.apply(q, k, v, density, block_size, scale)


class PiecewiseAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.PIECEWISE_ATTN

    @staticmethod
    def get_impl_cls() -> type["PiecewiseAttentionImpl"]:
        return PiecewiseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls():
        return None


class PiecewiseAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = float(extra_impl_args.get("dropout_p", 0.0))
        self.allow_cudnn_sdp = bool(extra_impl_args.get("allow_cudnn_sdp", False))
        self.prefix = prefix
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_size = head_size
        try:
            cfg = get_global_server_args().attention_backend_config or {}
        except ValueError:
            cfg = {}
        dense_fallback = cfg.get("piecewise_dense_fallback", None)
        if dense_fallback is None:
            dense_fallback = os.environ.get("SGLANG_PIECEWISE_ATTN_DENSE_FALLBACK", "fa")
        self.dense_fallback = str(dense_fallback).strip().lower()
        self._dense_fa_impl = None
        if self.dropout == 0.0 and self.dense_fallback != "sdpa":
            try:
                from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
                    FlashAttentionBackend,
                )

                self._dense_fa_impl = FlashAttentionBackend.get_impl_cls()(
                    num_heads=num_heads,
                    head_size=head_size,
                    causal=causal,
                    softmax_scale=softmax_scale,
                    num_kv_heads=num_kv_heads,
                    prefix=prefix,
                    **extra_impl_args,
                )
            except Exception as exc:
                logger.info_once(
                    f"Piecewise attention dense fallback will use SDPA for "
                    f"{self.prefix or '<unknown>'}: {exc}"
                )

        sparsity = cfg.get("piecewise_sparsity", None)
        if sparsity is None:
            sparsity = os.environ.get("SGLANG_PIECEWISE_ATTN_SPARSITY", "0.9")
        density = cfg.get("piecewise_density", None)
        if density is None:
            density = 1.0 - float(sparsity)
        self.density = min(1.0, max(0.0, float(density)))

        stage1_schedule = cfg.get("piecewise_stage1_schedule", None)
        if stage1_schedule is None:
            stage1_schedule = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_STAGE1_SCHEDULE", "false"
            )
        self.stage1_schedule = str(stage1_schedule).lower() not in (
            "0",
            "false",
            "no",
        )

        dense_steps = cfg.get("piecewise_stage1_dense_steps", None)
        if dense_steps is None:
            dense_steps = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_STEPS", "5"
            )
        self.stage1_dense_steps = max(0, int(dense_steps))

        start_sparsity = cfg.get("piecewise_stage1_start_sparsity", None)
        if start_sparsity is None:
            start_sparsity = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_STAGE1_START_SPARSITY", "0.8"
            )
        end_sparsity = cfg.get("piecewise_stage1_end_sparsity", None)
        if end_sparsity is None:
            end_sparsity = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_STAGE1_END_SPARSITY", "0.9"
            )
        self.stage1_start_sparsity = min(1.0, max(0.0, float(start_sparsity)))
        self.stage1_end_sparsity = min(1.0, max(0.0, float(end_sparsity)))

        block_size = cfg.get("piecewise_block_size", None)
        if block_size is None:
            block_size = os.environ.get("SGLANG_PIECEWISE_ATTN_BLOCK_SIZE", "64")
        self.block_size = max(1, int(block_size))

        only_v2v_self = cfg.get("piecewise_only_video_self_attention", None)
        if only_v2v_self is None:
            only_v2v_self = cfg.get("piecewise_only_self_attention", None)
        if only_v2v_self is None:
            only_v2v_self = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_ONLY_VIDEO_SELF", "true"
            )
        self.only_video_self_attention = str(only_v2v_self).lower() not in (
            "0",
            "false",
            "no",
        )

        approx_remainder = cfg.get("piecewise_approx_remainder", None)
        if approx_remainder is None:
            approx_remainder = os.environ.get(
                "SGLANG_PIECEWISE_ATTN_APPROX_REMAINDER", "true"
            )
        self.approx_remainder = str(approx_remainder).lower() not in (
            "0",
            "false",
            "no",
        )

        route_mode = cfg.get("piecewise_route_mode", None)
        if route_mode is None:
            route_mode = os.environ.get("SGLANG_PIECEWISE_ATTN_ROUTE_MODE", "score")
        self.route_mode = str(route_mode).lower()
        self.layer_idx = _ltx2_layer_idx_from_prefix(self.prefix or "")
        self.dense_layers = _cfg_or_env_int_set(
            cfg, "piecewise_dense_layers", "SGLANG_PIECEWISE_ATTN_DENSE_LAYERS"
        )
        self.stage1_dense_layers = _cfg_or_env_int_set(
            cfg,
            "piecewise_stage1_dense_layers",
            "SGLANG_PIECEWISE_ATTN_STAGE1_DENSE_LAYERS",
        )
        self.stage2_dense_layers = _cfg_or_env_int_set(
            cfg,
            "piecewise_stage2_dense_layers",
            "SGLANG_PIECEWISE_ATTN_STAGE2_DENSE_LAYERS",
        )

        logger.info_once(
            "Piecewise attention configured for "
            f"{self.prefix or '<unknown>'}: density={self.density:.4f} "
            f"block_size={self.block_size} "
            f"only_video_self={self.only_video_self_attention} "
            f"dense_fallback={self.dense_fallback} "
            f"approx_remainder={self.approx_remainder} "
            f"route_mode={self.route_mode} "
            f"stage1_schedule={self.stage1_schedule} "
            f"layer_idx={self.layer_idx} "
            f"dense_layers={sorted(self.dense_layers)} "
            f"stage1_dense_layers={sorted(self.stage1_dense_layers)} "
            f"stage2_dense_layers={sorted(self.stage2_dense_layers)}"
        )

    def _layer_forces_dense(self, stage: str | None) -> bool:
        if self.layer_idx is None:
            return False
        if self.layer_idx in self.dense_layers:
            return True
        if stage == "stage1" and self.layer_idx in self.stage1_dense_layers:
            return True
        if stage == "stage2" and self.layer_idx in self.stage2_dense_layers:
            return True
        return False

    def _density_for_step(self, attn_metadata: AttentionMetadata | None) -> float:
        stage = getattr(attn_metadata, "ltx2_stage", None) if attn_metadata is not None else None
        if self._layer_forces_dense(stage):
            return 1.0
        if not self.stage1_schedule or attn_metadata is None:
            return self.density

        if stage != "stage1":
            return self.density

        step_value = getattr(attn_metadata, "current_timestep", 0)
        num_steps_value = getattr(attn_metadata, "ltx2_num_steps", 0)
        step = int(0 if step_value is None else step_value)
        num_steps = int(0 if num_steps_value is None else num_steps_value)
        if step < self.stage1_dense_steps:
            return 1.0

        ramp_steps = max(1, num_steps - self.stage1_dense_steps - 1)
        progress = min(1.0, max(0.0, (step - self.stage1_dense_steps) / ramp_steps))
        sparsity = self.stage1_start_sparsity + progress * (
            self.stage1_end_sparsity - self.stage1_start_sparsity
        )
        return min(1.0, max(0.0, 1.0 - sparsity))

    def _piecewise_fallback_reason(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        density: float,
    ) -> str:
        if self.dropout != 0.0:
            return "dropout"
        if self.causal:
            return "causal"
        if query.device.type != "cuda":
            return "non_cuda"
        if query.shape[1] != key.shape[1]:
            return "qk_sequence_mismatch"
        if query.shape[2] != key.shape[2] or key.shape[2] != value.shape[2]:
            return "head_mismatch"
        if self.only_video_self_attention and (
            ".attn1.attn" not in self.prefix or ".audio_attn1." in self.prefix
        ):
            return "not_video_self_attention"
        if density >= 1.0:
            return "density_ge_1"
        return ""

    def _should_use_piecewise(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        density: float,
    ) -> bool:
        return not self._piecewise_fallback_reason(query, key, value, density)

    def _dense_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self._dense_fa_impl is not None:
            return self._dense_fa_impl.forward(query, key, value, attn_metadata)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_kwargs = {
            "attn_mask": None,
            "dropout_p": self.dropout,
            "is_causal": self.causal,
            "scale": self.softmax_scale,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        sdpa_context = (
            sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
            if self.allow_cudnn_sdp and query.device.type == "cuda"
            else nullcontext()
        )
        with sdpa_context:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, **attn_kwargs
            )
        return output.transpose(1, 2).contiguous()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        density = self._density_for_step(attn_metadata)
        fallback_reason = self._piecewise_fallback_reason(query, key, value, density)
        if fallback_reason:
            _piecewise_record_stats(
                prefix=self.prefix,
                stage=getattr(attn_metadata, "ltx2_stage", None) if attn_metadata is not None else None,
                branch="fallback",
                reason=fallback_reason,
                query=query,
                key=key,
                value=value,
                density=density,
            )
            return self._dense_sdpa(query, key, value, attn_metadata)

        logger.debug(
            "Piecewise attention active for %s density=%.6f",
            self.prefix or "<unknown>",
            density,
        )
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        block_size = min(self.block_size, q.shape[-2], k.shape[-2])
        exact_density = None
        with torch.no_grad():
            triton.set_allocator(_make_tma_allocator())
            if self.route_mode == "local" and not self.approx_remainder:
                nt_q = triton.cdiv(q.shape[-2], block_size)
                nt_kv = triton.cdiv(k.shape[-2], block_size)
                block_indices = (
                    torch.arange(nt_q, device=q.device, dtype=torch.int32)
                    .clamp_max(nt_kv - 1)
                    .view(1, 1, nt_q, 1)
                    .expand(q.shape[0], q.shape[1], nt_q, 1)
                    .contiguous()
                )
                kc = k
                vc = v
            else:
                qc, kc, vc, k_var = chunk_reduce_qkv(
                    q=q,
                    k=k,
                    v=v,
                    block_size=block_size,
                    include_v_centroid=self.approx_remainder,
                )
                route_fn = (
                    _compiled_taylor_error_block_indices()
                    if _should_compile_piecewise_route()
                    else taylor_error_block_indices
                )
                block_indices = route_fn(
                    qc=qc,
                    kc=kc,
                    k_var=k_var,
                    density=density,
                    scale=self.softmax_scale,
                )
                nt_kv = kc.shape[2]
                ns = block_indices.shape[-1]
                exact_density = float(ns) / float(nt_kv)
                logger.debug(
                    "Piecewise attention routing for %s: T_Q=%s T_KV=%s "
                    "BT=%s NT_Q=%s NT_KV=%s NS=%s exact_density=%.6f "
                    "actual_sparsity=%.6f",
                    self.prefix or "<unknown>",
                    q.shape[-2],
                    k.shape[-2],
                    block_size,
                    qc.shape[2],
                    nt_kv,
                    ns,
                    exact_density,
                    1.0 - exact_density,
                )
            output, _ = piecewise_attn_fwd(
                q=q,
                k=k,
                v=v,
                kc=kc,
                vc=vc if vc is not None else kc,
                block_indices=block_indices,
                block_size=block_size,
                scale=self.softmax_scale,
                approx_remainder=self.approx_remainder,
            )
        _piecewise_record_stats(
            prefix=self.prefix,
            stage=getattr(attn_metadata, "ltx2_stage", None) if attn_metadata is not None else None,
            branch="sparse",
            reason="",
            query=query,
            key=key,
            value=value,
            density=density,
            block_size=block_size,
            exact_density=exact_density,
        )
        return output.transpose(1, 2).contiguous()

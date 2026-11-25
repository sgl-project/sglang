from __future__ import annotations

import functools
import inspect
import logging
from typing import Optional, Tuple, Union

import torch


logger = logging.getLogger(__name__)

MegaWeights = Tuple[torch.Tensor, torch.Tensor]

# K3 SiTU constants the fp8 x fp4 MegaMoE kernel is built for
_SITU_BETA = 4.0
_SITU_LINEAR_BETA = 25.0

# trtllm-gen's finalize (flashinfer 0.7.x) changes its top-k combine order above this
_TRTLLM_SPLIT4_COMBINE_MAX_TOKENS = 84

# Largest M tile of the MegaMoE buffer
_MAX_BLOCK_M = 64

_CALL_KEYWORDS = (
    "weight_layout",
    "numerics",
    "split4_combine_max_tokens",
    "topk_format",
    "topk_first_expert",
)
_BUFFER_KEYWORDS = ("exact_num_max_tokens", "max_block_m")


def _deep_gemm():
    try:
        import deep_gemm
    except Exception:  # not installed, or its native library does not load here
        return None
    return deep_gemm


@functools.lru_cache(maxsize=1)
def missing_capability() -> Optional[str]:
    """What the installed DeepGEMM lacks for this path, or None."""
    deep_gemm = _deep_gemm()
    if deep_gemm is None:
        return "deep_gemm cannot be imported"
    try:
        call = deep_gemm.fp8_fp4_mega_moe
        buffer_cls = deep_gemm.mega.SymmBuffer
        call_keywords = inspect.signature(call).parameters
        buffer_keywords = inspect.signature(buffer_cls.__init__).parameters
    except Exception as e:
        return f"this deep_gemm has no inspectable fp8_fp4_mega_moe / mega.SymmBuffer ({type(e).__name__}: {e})"
    missing = [name for name in _CALL_KEYWORDS if name not in call_keywords] + [
        name for name in _BUFFER_KEYWORDS if name not in buffer_keywords
    ]
    if missing:
        try:
            version = str(deep_gemm.__version__)
        except Exception:
            version = "unknown"
        return (
            f"deep_gemm {version} lacks the one-rank MegaMoE API ({', '.join(missing)})"
        )
    return None


def mismatch(layer: torch.nn.Module, with_bias: bool, precision: str) -> Optional[str]:
    """The first requirement of this path the layer or DeepGEMM misses, or None."""
    cfg = layer.moe_runner_config
    if (
        cfg.activation != "situ"
        or cfg.gemm1_alpha != _SITU_BETA
        or cfg.gemm1_clamp_limit != _SITU_LINEAR_BETA
    ):
        return f"activation {cfg.activation}, SiTU constants ({cfg.gemm1_alpha}, {cfg.gemm1_clamp_limit})"
    if getattr(cfg, "gate_up_interleaved", True):
        return "gate / up rows interleaved in the checkpoint"
    if with_bias:
        return "expert bias"
    if precision != "default":
        return f"{precision} activations, not MXFP8"
    reason = missing_capability()
    if reason is not None:
        return reason
    if not torch.distributed.is_initialized():
        return "torch.distributed is not initialized"
    try:
        num_domains = int(_deep_gemm().get_num_locality_domains())
    except Exception as e:
        return f"deep_gemm cannot report locality domains ({type(e).__name__}: {e})"
    if num_domains < 2:
        return "the device has no second locality domain"
    return None


def handles(layer: torch.nn.Module, num_tokens: int) -> bool:
    """Whether a routed call of `num_tokens` rows on this layer runs on MegaMoE."""
    if getattr(layer, "mega_l1_weights", None) is None:
        return False
    from sglang.srt.layers.moe.k3_die_local_moe import takes

    return takes(num_tokens)


class _Group:
    """One-rank MegaMoE group of this process: buffer and counters."""

    buf = None
    stats = None
    shape = None
    packed = None  # int32 view of the start of buf.topk_idx: packed top-k words
    first_expert = 0  # global id of this rank's first expert


_group = _Group()


def reserve_buffer(
    num_experts: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    device,
    first_expert: int,
):
    """The process's MegaMoE buffer, created on first use (shared by its layers)."""
    shape = (num_experts, top_k, hidden, intermediate)
    if _group.buf is None:
        import deep_gemm
        import torch.distributed as dist

        from sglang.srt.layers.moe.k3_die_local_moe import MAX_VERIFY_ROWS

        # new_group is a collective: every rank creates all groups in the same order
        backend = "gloo" if dist.is_gloo_available() else None
        groups = [
            dist.new_group([r], backend=backend) for r in range(dist.get_world_size())
        ]
        _group.buf = deep_gemm.mega.SymmBuffer(
            groups[dist.get_rank()],
            num_experts,
            MAX_VERIFY_ROWS,
            top_k,
            hidden,
            intermediate,
            mma_type="fp8xfp4",
            activation="situ",
            exact_num_max_tokens=True,
            max_block_m=_MAX_BLOCK_M,
        )
        _group.stats = torch.zeros(num_experts, dtype=torch.int, device=device)
        _group.shape = shape
        # topk_format='packed': int32 words at the start of the int64 topk_idx region
        rows = _group.buf.topk_idx.shape[0]
        words = _group.buf.topk_idx.view(torch.int32).view(-1)
        _group.packed = words[: rows * top_k].view(rows, top_k)
        _group.first_expert = int(first_expert)
    assert _group.shape == shape, (
        f"K3 routed MoE layers differ in shape: {_group.shape} vs {shape}"
    )
    return _group.buf


def activation_dest(num_tokens: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """(x, x_sf) rows of the MegaMoE buffer to quantize into, or None."""
    buf = _group.buf
    if buf is None or num_tokens > buf.x.shape[0]:
        return None
    from sglang.srt.layers.moe.k3_die_local_moe import takes

    if not takes(num_tokens):
        return None
    return buf.x[:num_tokens], buf.x_sf[:num_tokens]


def route_quant_into_buffer(
    scores: torch.Tensor,
    bias: torch.Tensor,
    x: torch.Tensor,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_scale: bool,
    num_token_non_padded: Optional[torch.Tensor],
):
    """Route and quantize into the MegaMoE buffer in one launch, or None."""
    buf = _group.buf
    num_tokens = x.shape[0]
    if buf is None or num_tokens > buf.x.shape[0] or top_k != _group.packed.shape[1]:
        return None
    from sglang.kernels.ops.moe import (
        k3_route_quant_packed,
        kimi_k3_fused_routing,
        moe_route_quant_fused,
    )

    # same bf16 / fp32 routing-weight split as trtllm-gen, to stay bit-identical
    bf16_weights = moe_route_quant_fused.covered(scores, bias, top_k, x)
    if (
        not (bf16_weights or kimi_k3_fused_routing.covered(scores, bias, top_k, x))
        or not k3_route_quant_packed.available()
    ):
        return None
    packed, x_q, x_s = (
        _group.packed[:num_tokens],
        buf.x[:num_tokens],
        buf.x_sf[:num_tokens],
    )
    weights, ids = k3_route_quant_packed.route_quant(
        scores,
        bias,
        x,
        top_k,
        renormalize,
        routed_scaling_factor,
        apply_scale,
        packed,
        x_q,
        x_s,
        num_token_non_padded,
        None if bf16_weights else buf.topk_weights,
    )
    return weights, ids, packed, x_q, x_s


_pack_kernel = None


def _get_pack_kernel():
    """Triton launch that stages packed top-k words (and x, x_sf) into the buffer."""
    global _pack_kernel
    if _pack_kernel is not None:
        return _pack_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _k3_deepgemm_routed_pack(
        ids_ptr,
        w_ptr,
        out_words_ptr,
        out_w_ptr,
        x_ptr,
        xs_ptr,
        buf_x_ptr,
        buf_xs_ptr,
        TOPK: tl.constexpr,
        PACKED: tl.constexpr,
        W_F32: tl.constexpr,
        COPY_X: tl.constexpr,
        X_BYTES: tl.constexpr,
        XS_WORDS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, TOPK)
        if PACKED:
            p = tl.load(ids_ptr + t * TOPK + k)
        else:
            gid = tl.load(ids_ptr + t * TOPK + k).to(tl.int32)
            if W_F32:
                wf = tl.load(w_ptr + t * TOPK + k).to(tl.float32)
                tl.store(out_w_ptr + t * TOPK + k, wf)
                p = tl.where(gid >= 0, ((gid & 0x3FFF) << 16) | 0x40000000, -1)
            else:
                wb = (
                    tl.load(w_ptr + t * TOPK + k)
                    .to(tl.bfloat16)
                    .to(tl.uint16, bitcast=True)
                    .to(tl.int32)
                )
                p = (gid << 16) | (wb & 0xFFFF)
        tl.store(out_words_ptr + t * TOPK + k, p)
        if COPY_X:
            for c in range(0, X_BYTES, BLOCK):
                o = c + tl.arange(0, BLOCK)
                m = o < X_BYTES
                tl.store(
                    buf_x_ptr + t * X_BYTES + o,
                    tl.load(x_ptr + t * X_BYTES + o, mask=m),
                    mask=m,
                )
            o2 = tl.arange(0, 32)
            m2 = o2 < XS_WORDS
            tl.store(
                buf_xs_ptr + t * XS_WORDS + o2,
                tl.load(xs_ptr + t * XS_WORDS + o2, mask=m2),
                mask=m2,
            )

    _pack_kernel = _k3_deepgemm_routed_pack
    return _pack_kernel


def _call_keywords(num_call_tokens: int) -> dict:
    """Keywords of every MegaMoE call of this path."""
    limit = _TRTLLM_SPLIT4_COMBINE_MAX_TOKENS
    return dict(
        cumulative_local_expert_recv_stats=_group.stats,
        recipe=(1, 1, 32),
        activation="situ",
        fast_math=True,
        weight_layout="trtllm",
        numerics="trtllm",
        split4_combine_max_tokens=0 if num_call_tokens > limit else limit,
        topk_format="packed",
        topk_first_expert=_group.first_expert,
    )


def select_routing(prepared_ids, prepared_routing, standard_routing):
    """The `routing` argument of `run`: packed words, or (ids, weights)."""
    if prepared_routing is not None:
        return prepared_ids, prepared_routing[0].expert_weights
    if prepared_ids is not None:
        return prepared_ids
    return standard_routing()


def run(
    l1_weights: MegaWeights,
    l2_weights: MegaWeights,
    x_quant: torch.Tensor,
    x_scale: torch.Tensor,
    routing: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    local_expert_offset: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Writes the combined expert output (bf16 [T, H]) to `out`."""
    import deep_gemm

    num_tokens = x_quant.shape[0]
    if num_tokens == 0:
        return out
    buf = _group.buf
    assert buf is not None and num_tokens <= buf.x.shape[0], (
        f"k3_deepgemm_routed: {num_tokens} rows do not fit the MegaMoE buffer"
    )
    assert int(local_expert_offset) == _group.first_expert, (
        f"first expert {_group.first_expert} of the buffer != {local_expert_offset}"
    )
    packed = not isinstance(routing, tuple)
    ids, weights = (routing, routing) if packed else routing
    ids, weights = ids.contiguous(), weights.contiguous()
    copy_x = x_quant.data_ptr() != buf.x.data_ptr()
    if not (packed and not copy_x and ids.data_ptr() == _group.packed.data_ptr()):
        hidden = x_quant.shape[1]
        xs_words = x_scale.view(torch.uint8).reshape(num_tokens, -1).view(torch.int32)
        buf_xs = (
            buf.x_sf.view(torch.uint8).reshape(buf.x_sf.shape[0], -1).view(torch.int32)
        )
        assert xs_words.shape[1] == buf_xs.shape[1], (xs_words.shape, buf_xs.shape)
        _get_pack_kernel()[(num_tokens,)](
            ids,
            weights,
            _group.packed,
            buf.topk_weights,
            x_quant.view(torch.uint8),
            xs_words,
            buf.x.view(torch.uint8),
            buf_xs,
            TOPK=ids.shape[1],
            PACKED=packed,
            W_F32=(not packed and weights.dtype == torch.float32),
            COPY_X=copy_x,
            X_BYTES=hidden,
            XS_WORDS=xs_words.shape[1],
            BLOCK=1024,
            num_warps=4,
        )
    deep_gemm.fp8_fp4_mega_moe(
        out, l1_weights, l2_weights, buf, **_call_keywords(num_tokens)
    )
    return out

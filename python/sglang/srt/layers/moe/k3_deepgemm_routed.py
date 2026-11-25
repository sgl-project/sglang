"""Kimi K3 routed experts on DeepGEMM's one-rank MegaMoE kernel, for the calls
SGLANG_K3_ROUTED_MOE_DEEPGEMM dispatches to it (layers/moe/k3_die_local_moe.py
decides which and prepares the weights).

Each EP rank is a one-rank MegaMoE group over its local experts. The kernel
reads the layer's trtllm-gen weight tensors directly (`weight_layout='trtllm'`)
and is asked for trtllm-gen's rounding and combine order (`numerics='trtllm'`),
so a call it takes is bit-identical to the same call on trtllm-gen:

  route  routing + activation quant in one launch that writes the packed top-k
         words, x and x_sf into the MegaMoE buffer (`route_quant_into_buffer`)
  call   one `deep_gemm.fp8_fp4_mega_moe` launch into the caller's output; when
         the inputs were routed elsewhere, one staging launch first (`run`)

Needs a DeepGEMM with the one-rank API listed in `missing_capability`; without
it the layer is never prepared and every call stays on trtllm-gen.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Optional, Tuple, Union

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

MegaWeights = Tuple[torch.Tensor, torch.Tensor]

# K3 SiTU constants the fp8 x fp4 MegaMoE kernel is built for
_SITU_BETA = 4.0
_SITU_LINEAR_BETA = 25.0

# trtllm-gen's finalize (flashinfer 0.7.x) sums a token's top-k rows in four
# interleaved accumulators for calls of at most this many tokens and in one
# chain above. Only used to ask DeepGEMM for the same order.
_TRTLLM_SPLIT4_COMBINE_MAX_TOKENS = 84

# Largest M tile of the MegaMoE buffer. With the 256 rows of
# k3_die_local_moe.MAX_VERIFY_ROWS the kernel never picks a larger one.
_MAX_BLOCK_M = 64

# Keywords of the one-rank, shared-weight MegaMoE API this path calls
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
    """What the installed DeepGEMM lacks for this path, or None. The path needs
    `fp8_fp4_mega_moe(weight_layout='trtllm', numerics='trtllm',
    split4_combine_max_tokens=, topk_format='packed', topk_first_expert=)` on
    FP8 x FP4 SiTU experts and `SymmBuffer(exact_num_max_tokens=,
    max_block_m=)`; the buffer keywords arrived together with the FP8 x FP4
    `weight_layout='trtllm'`, which the call's signature alone does not show."""
    deep_gemm = _deep_gemm()
    if deep_gemm is None:
        return "deep_gemm cannot be imported"
    try:
        call = deep_gemm.fp8_fp4_mega_moe
        buffer_cls = deep_gemm.mega.SymmBuffer
        call_keywords = inspect.signature(call).parameters
        buffer_keywords = inspect.signature(buffer_cls.__init__).parameters
    except Exception as e:
        # no such entry point, or one whose signature Python cannot read (a
        # build that exposes it as a bound native op): not the API this path calls
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
    """The first requirement of this path that the layer or the installed
    DeepGEMM misses, or None."""
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
        # the kernel splits every expert's rows between two domains' SMs
        return "the device has no second locality domain"
    return None


def handles(layer: torch.nn.Module, num_tokens: int) -> bool:
    """Whether a routed call of `num_tokens` rows on this layer runs on the
    MegaMoE kernel. That kernel finalizes (top-k weighted combine) in-kernel, so
    a caller that would defer the finalize into its all-reduce
    (SGLANG_K3_AR_FUSION) must not defer such a call."""
    if getattr(layer, "mega_l1_weights", None) is None:
        return False
    from sglang.srt.layers.moe.k3_die_local_moe import takes

    return takes(num_tokens)


_SHARED_AR_ORDERS = ("fused", "after", "before", "overlap")
_FUSED_PULL_UNROLLS = (2, 4, 8, 16)  # the widths compiled into all_reduce_pull_norm


def _fused_pull_env(field) -> Optional[int]:
    if not field.is_set():
        return None
    number = field.get()
    if number < 1:
        raise ValueError(f"{field.name} is a positive integer, got {number}")
    return number


@functools.lru_cache(maxsize=1)
def fused_pull_tuning() -> dict:
    """Launch geometry of the one pull over the [latent | shared] pair, as
    keyword arguments of k3_ar_fusion.all_reduce_norm. Empty (the default) =
    the kernel's own tuned table, which is keyed by the byte size of what it
    reduces and so already sees the whole pair. The environment is the same on
    every rank, which the pull's barrier needs (one block count per call)."""
    blocks = _fused_pull_env(envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_FUSED_PULL_BLOCKS)
    unroll = _fused_pull_env(envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_FUSED_PULL_UNROLL)
    if unroll is not None and unroll not in _FUSED_PULL_UNROLLS:
        raise ValueError(
            f"SGLANG_K3_ROUTED_MOE_DEEPGEMM_FUSED_PULL_UNROLL is one of {_FUSED_PULL_UNROLLS}, got {unroll}"
        )
    tuning = {}
    if blocks is not None:
        tuning["num_blocks"] = blocks
    if unroll is not None:
        tuning["unroll"] = unroll
    return tuning


@functools.lru_cache(maxsize=1)
def configured_shared_ar_order() -> str:
    order = envs.SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER.get()
    if order not in _SHARED_AR_ORDERS:
        raise ValueError(
            f"SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER is one of {_SHARED_AR_ORDERS}, got {order!r}"
        )
    if order == "fused":
        fused_pull_tuning()  # a bad geometry fails with the order, not mid-forward
    return order


def shared_ar_order(layer: torch.nn.Module, num_tokens: int) -> str:
    """Where KimiK3MoE._forward_fused puts the shared experts' pull all-reduce
    under SGLANG_K3_AR_FUSION, relative to this routed call.

    Stock queues it on the side stream right before the routed call ("overlap"):
    trtllm-gen's blocks leave room for its 16 blocks. The MegaMoE kernel does not:
    it is persistent, one CTA per SM with the whole register file and shared
    memory, and it starts with a grid-wide sync. A pull block that reaches an SM
    first holds the kernel back until every rank has arrived at the pull's
    barriers; a rank whose kernel got there first only starts its pull when the
    kernel exits. The two orders mix across ranks and layers, and each mixed
    layer costs a second kernel time on every rank (measured on Kimi K3.1, EP8:
    +7.0 ms per verify round at 80 rows, +4.8 ms at 160).

    For a call the kernel takes (:func:`handles`) the order is therefore fixed,
    the same on every rank
    (SGLANG_K3_ROUTED_MOE_DEEPGEMM_SHARED_AR_ORDER):
      after   kernel, latent all-reduce, then the shared all-reduce next to the
              tail's local up_proj GEMM (A/B control: level with stock, and the
              two pulls were seen to overlap by about 8 us in the engine)
      before  shared all-reduce, then the kernel
      overlap the stock order (the measured regression; A/B control only)
      fused   (default) kernel, then ONE pull over the whole [latent | shared]
              pair with the RMSNorm on the latent rows: the collective of
              _forward_fused's single-stream branch (all_reduce_pull_norm takes the pair
              with num_norm_rows = the latent rows). One barrier entry and
              exit instead of two, nothing left on the side stream after the
              kernel, and no second pull that could meet the first one on the
              pull semaphores.
    The same kernels read the same inputs in after / before / overlap (the two
    all-reduces work on disjoint parts of the [latent | shared] buffer), so the
    result does not depend on the order. Under fused every element is still
    reduced once by the same multicast load and the latent rows get the same
    epilogue; only the rank that issues the load for a given element changes."""
    if not handles(layer, num_tokens):
        return "overlap"
    return configured_shared_ar_order()


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
    """The process's MegaMoE buffer (`MAX_VERIFY_ROWS` rows, shared by its
    layers), created on first use. Called at weight load so the buffer is
    counted before the engine sizes its memory pools."""
    shape = (num_experts, top_k, hidden, intermediate)
    if _group.buf is None:
        import deep_gemm
        import torch.distributed as dist

        from sglang.srt.layers.moe.k3_die_local_moe import MAX_VERIFY_ROWS

        # new_group is a collective over the default group: every rank creates
        # all of them in the same order and keeps its own. DeepGEMM takes only
        # size(), rank() and barrier() from a one-rank group, so the group needs
        # no device communicator (an NCCL one costs ~0.4 GB of device memory).
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
        # deep_gemm `topk_format='packed'`: int32 words at the start of the int64
        # topk_idx region
        rows = _group.buf.topk_idx.shape[0]
        words = _group.buf.topk_idx.view(torch.int32).view(-1)
        _group.packed = words[: rows * top_k].view(rows, top_k)
        _group.first_expert = int(first_expert)
    assert _group.shape == shape, (
        f"K3 routed MoE layers differ in shape: {_group.shape} vs {shape}"
    )
    return _group.buf


def activation_dest(num_tokens: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Rows of the MegaMoE buffer a quantizer may write `num_tokens` MXFP8
    activations and their packed scales to, so that `run` need not copy them;
    None when the MegaMoE kernel does not take this call."""
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
    """Routing + MXFP8 quant of the activations in ONE launch that writes the packed
    top-k words, x and x_sf into the MegaMoE buffer, so that `run` issues the
    MegaMoE launch alone. Returns `(weights, ids, packed, x_q, x_s)` or None
    (the caller then routes as before and `run` stages the inputs)."""
    buf = _group.buf
    num_tokens = x.shape[0]
    if buf is None or num_tokens > buf.x.shape[0] or top_k != _group.packed.shape[1]:
        return None
    from sglang.kernels.ops.moe import (
        k3_route_quant_packed,
        kimi_k3_fused_routing,
        moe_route_quant_fused,
    )

    # trtllm-gen takes bf16 routing weights where the fused route + quant kernel
    # applies and fp32 above: the same split keeps the outputs bit-identical
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
    """Staging launch for `topk_format='packed'`: packed words into the buffer (from
    packed words, or from ids + weights: bf16 weights go into the word, fp32
    weights into the buffer's topk_weights with bit 30 set in the word), and the
    activations when they are not in place."""
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
    """Keywords of every MegaMoE call of this path. trtllm-gen picks its combine
    order from the engine call's size; the same threshold keeps the outputs
    bit-identical."""
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
    """What `run` gets as `routing`, from what the K3 routing handoff published:

    - `kimi_k3_fused_routing` ran (`prepared_routing` = its (metadata, body)):
      `prepared_ids` are PLAIN global expert ids and the weights are in the
      trtllm-gen routing metadata -> `(ids, weights)`;
    - `route_quant_fused` (or this module's one-launch route) ran: `prepared_ids`
      are packed words `(id << 16) | bf16(weight)` -> the words;
    - nothing was prepared -> `standard_routing()` (ids, weights of the top-k).
    """
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
    """`routing`: packed int32 `[T, k]` words `(id << 16) | bf16(weight)`, or
    `(ids, weights)`. Writes the combined expert output (bf16 `[T, H]`) to `out`."""
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
    # activation_dest: the rows are already in place
    copy_x = x_quant.data_ptr() != buf.x.data_ptr()
    if not (packed and not copy_x and ids.data_ptr() == _group.packed.data_ptr()):
        # the inputs were not all written in place by route_quant_into_buffer
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

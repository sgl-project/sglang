"""
LFM3 (Liquid Foundation Model 3) implementation for SGLang.

Hybrid architecture with three layer flavors:
  - `full_attention`: GQA with partial-RoPE (global rope_theta, e.g. 1e6)
  - `sliding_attention`: GQA with full-RoPE (SWA rope_theta, e.g. 1e4)
                        + RadixAttention sliding_window_size
  - `conv`: gated short-conv (B*x -> CuTe DSL causal-conv1d -> C*conv_out)

Adds over LFM2:
  - Per-layer SWA via `RadixAttention(sliding_window_size=...)`
  - Two RoPE tables (global partial-RoPE + local full-RoPE) routed by layer_type
  - Per-head sigmoid gate on the attention output (`gate_proj`, always on)
  - Optional sandwich norm (`use_sandwich_norm`): extra RMSNorm AFTER the
    operator and AFTER the FFN

Attention layers use RadixAttention; conv layers use the MambaPool conv-state
cache (via HybridReqToTokenPool) with a width-generic CuTe DSL causal-conv1d
kernel (`sglang.kernels.ops.mamba.cutedsl_gated_conv`) for both prefill and decode.
"""

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.lfm3 import Lfm3Config
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    track_mamba_states_if_needed,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import Mamba2Metadata
from sglang.srt.layers.layer_boundary import append_stages, declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_parallel, get_platform
from sglang.srt.utils import (
    add_prefix,
    get_compiler_backend,
    make_layers,
    set_weight_attrs,
)

logger = logging.getLogger(__name__)

_is_cuda = get_platform().is_cuda

# The width-generic K=7 conv is a CuTe DSL kernel. It was first tuned for Blackwell
# datacenter (sm100/sm103) but is pure-SIMT (no MMA/TMA/tensor-core ops); its only
# arch-specific feature is PDL (griddepcontrol + use_pdl), which Hopper (sm90) and
# consumer Blackwell (sm120) also support. Validated numerically against the torch
# reference on sm90 (H100), sm100, and sm120 (RTX PRO 6000 Blackwell — full
# test_cutedsl_gated_conv suite), so enable on all three. Gate on cuda first
# (short-circuit so the sm-probe never runs on ROCm/CPU) and import lazily — on any
# other platform fall back to the torch F.conv1d path so ROCm / non-CUDA builds are
# unaffected.
_use_cute_conv = _is_cuda and (
    get_platform().is_sm100 or get_platform().is_sm90 or get_platform().is_sm120
)
if _use_cute_conv:
    from sglang.kernels.ops.mamba.cutedsl_gated_conv import (
        gated_causal_conv1d_decode,
        gated_causal_conv1d_prefill,
    )

# Layer types that run attention (consume KV cache + RoPE); the conv layer type does
# neither. Used to flag attention layers and to count KV-cache layers.
_ATTENTION_TYPES = frozenset({"full_attention", "sliding_attention"})

# rope_parameters keys consumed directly by the get_rope wrapper signature
# (everything else is forwarded into the scaling dict).
_ROPE_RESERVED_KEYS = frozenset({"rope_theta", "rope_type", "partial_rotary_factor"})


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _lfm3_apply_headwise_gate(
    gate_logits: torch.Tensor,
    attn_out: torch.Tensor,
    T: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Inductor-fused sigmoid + view + mul + reshape for headwise gated attention.

    Same shape as llama4's `_mul_attn_scale` and Qwen3-Next's
    `_apply_q_scale_and_softmax_scale` -- a tiny pure-math elementwise function
    that benefits from being compiled into a single kernel under
    `dynamic=True`.
    """
    gate = torch.sigmoid(gate_logits).view(T, num_heads, 1)
    return (gate * attn_out.view(T, num_heads, head_dim)).reshape(T, -1)


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _lfm3_conv_input_gate(
    proj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed in_proj [B_gate | C_gate | x] and form the conv input Bx=B_gate*x.

    Returns (Bx, C_gate) for the torch fallback. The CuTe kernel fuses
    this split and both gates internally.
    """
    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    return B_gate * x, C_gate


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _lfm3_conv_decode_torch(
    proj: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    mamba_indices: torch.Tensor,
) -> torch.Tensor:
    """Compiled gated convolution update when the CuTe kernel is unavailable.

    Form B*x, convolve its rolling K-token window, apply C, and advance
    the per-request state in place.
    """
    B_gate, C_gate, x = proj.chunk(3, dim=-1)
    Bx = B_gate * x
    sel = conv_state[mamba_indices].to(Bx.dtype)
    window = torch.cat([sel, Bx.unsqueeze(-1)], dim=-1)
    w = weight.to(Bx.dtype)
    out = (window * w.unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        out = out + bias.to(out.dtype)
    conv_state[mamba_indices] = window[:, :, 1:].to(conv_state.dtype)
    return C_gate * out


# Cached all-true has_initial_state flags for _conv_prefill: the value is constant
# (see the comment at the call site), so allocate once per device and slice a view
# instead of re-filling a tiny tensor per conv layer per prefill step. Mirrors the
# kernel module's `_zero_bias` caching.
_ONES_FLAGS_CACHE: dict = {}


def _ones_flags(n: int, device: torch.device) -> torch.Tensor:
    key = str(device)
    t = _ONES_FLAGS_CACHE.get(key)
    if t is None or t.numel() < n:
        t = torch.ones(max(n, 256), dtype=torch.int32, device=device)
        _ONES_FLAGS_CACHE[key] = t
    return t[:n]


class Lfm3MLP(nn.Module):
    """SwiGLU MLP with fused gate_up_proj (w1+w3) and w2 down-proj.

    The stage boundary completes the TP output sum before the sandwich norm
    and residual update.
    """

    def __init__(
        self,
        config: Lfm3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("w2", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        out, _ = self.w2(
            self.act_fn(gate_up),
        )
        return out


def _resolve_rope_kwargs(
    rope_parameters: dict,
    partial_rotary_factor: float,
) -> tuple[dict, float, float]:
    """Build `(rope_scaling, base, partial_rotary_factor)` for get_rope.

    For partial RoPE, use rope_type="proportional" (SGLang's Gemma4RotaryEmbedding),
    which matches HF LFM3's Gemma-4-style rotation pairing; the plain RotaryEmbedding
    pairs the rotated dims differently and is the wrong rotation for these checkpoints.
    """
    # rope_theta is always present in each per-layer rope dict (full=1e6, sliding=1e4),
    # so index it directly -- a default here would silently inject a wrong theta.
    base = rope_parameters["rope_theta"]
    rope_type = rope_parameters.get("rope_type", "default")
    if partial_rotary_factor < 1.0 and rope_type == "default":
        rope_type = "proportional"
    rope_scaling: dict = {"rope_type": rope_type}
    # Carry through anything that get_rope's scaling branch knows how to consume.
    rope_scaling |= {
        k: v for k, v in rope_parameters.items() if k not in _ROPE_RESERVED_KEYS
    }
    return rope_scaling, base, partial_rotary_factor


class Lfm3Attention(nn.Module):
    """GQA with Q/K RMSNorm, partial or full RoPE, per-head sigmoid output gate.

    The sliding-window-ness of the layer is signalled by `sliding_window`:
    `None` => full attention, `int` => SWA. Both flavors share the config's
    head counts and gating (per the HF-canonical config there are no per-type
    overrides); they differ only in the RoPE table and the window.
    """

    def __init__(
        self,
        config: Lfm3Config,
        layer_id: int,
        sliding_window: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.sliding_window = sliding_window
        self.alt_stream = alt_stream

        self.total_num_kv_heads = config.num_key_value_heads
        # head_dim is always set: the HF config's __post_init__ computes it
        # from hidden_size // num_attention_heads when absent.
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5

        # rope_parameters is nested by layer type per the HF-canonical config:
        # full_attention carries partial RoPE, sliding_attention is full RoPE at
        # its own theta. SWA never carries partial_rotary_factor (defaults to 1.0).
        layer_key = (
            "sliding_attention" if sliding_window is not None else "full_attention"
        )
        rope_parameters = config.rope_parameters[layer_key]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

        rope_scaling, rope_base, partial_factor = _resolve_rope_kwargs(
            rope_parameters, partial_rotary_factor
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=rope_base,
            is_neox_style=True,
            partial_rotary_factor=partial_factor,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)

        self.num_local_q_heads = self.qkv_proj.num_heads
        self.num_local_kv_heads = self.qkv_proj.num_kv_heads

        # Per-head sigmoid output gate (one logit per head), always present in
        # the HF-canonical architecture.
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.total_num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )

        self.attn = RadixAttention(
            num_heads=self.num_local_q_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_local_kv_heads,
            layer_id=layer_id,
            # HF `sliding_window` is a token COUNT; RadixAttention wants a max q-kv
            # DISTANCE, so pass sliding_window - 1 (matches gemma2 / exaone4 / afmoe and
            # this model's get_attention_sliding_window_size). Without the -1 the SWA
            # layers would attend to one extra token vs HF.
            sliding_window_size=(
                sliding_window - 1 if sliding_window is not None else -1
            ),
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        T = hidden_states.shape[0]

        # Overlap qkv_proj and gate_proj (independent matmuls on the same input)
        # on alt_stream under cuda-graph capture below a token threshold; above
        # it the GEMMs saturate the GPU and wait_stream is pure overhead. The
        # win mostly shows on larger (>4B) models; piecewise CUDA graph already
        # overlaps them, so the manual path is off then. Same gate as
        # qwen3_next/qwen3_5 `_forward_input_proj`.
        if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        gate_logits: torch.Tensor | None = None
        overlap_with_gate = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and T < DUAL_STREAM_TOKEN_THRESHOLD
        )
        if overlap_with_gate:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            qkv, _ = self.qkv_proj(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                gate_logits, _ = self.gate_proj(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            qkv, _ = self.qkv_proj(hidden_states)

        q_size = self.num_local_q_heads * self.head_dim
        kv_size = self.num_local_kv_heads * self.head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Per-head Q/K RMSNorm. Prefers sgl-kernel's fused inplace JIT QK-norm
        # (faster than two separate RMSNorms) when eligible (CUDA, eps_q==eps_k,
        # supported head_dim/dtype); else falls back to plain RMSNorm reshape.
        q, k = apply_qk_norm(
            q,
            k,
            self.q_norm,
            self.k_norm,
            head_dim=self.head_dim,
            alt_stream=self.alt_stream,
        )

        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q.reshape(T, -1), k.reshape(T, -1), v, forward_batch)

        if gate_logits is None:
            gate_logits, _ = self.gate_proj(hidden_states)
        # Compiled helper (Inductor) fuses sigmoid + view + mul + reshape
        # into one kernel; same pattern as llama4's `_mul_attn_scale` /
        # Qwen3-Next's `_apply_q_scale_and_softmax_scale`.
        attn_out = _lfm3_apply_headwise_gate(
            gate_logits, attn_out, T, self.num_local_q_heads, self.head_dim
        )

        out, _ = self.o_proj(
            attn_out,
        )
        return out


class Lfm3ShortConv(nn.Module):
    """Gated short conv: in_proj -> split(B, C, x) -> Bx -> conv1d -> C*conv_out.

    Prefill and decode run a width-generic CuTe DSL causal-conv1d kernel
    (`sglang.kernels.ops.mamba.cutedsl_gated_conv`) that matches the HF `slow_forward`
    semantics exactly. LFM3 uses K=7, which the in-tree sgl_kernel C++ (widths
    2-4) and shared Triton (2-5) conv kernels cannot serve; the CuTe DSL kernel
    is width-generic and memory-bandwidth bound (SIMT, sm90/sm100). The hidden
    dimension is sharded across TP ranks.

    Algorithm (per request):
      * The conv state holds the last `K-1` Bx tokens (zero-padded on first
        prefill). For a new chunk of `T_i` tokens the full causal window is
        `concat(state, Bx_i)`; depthwise conv with `padding=0` produces `T_i`
        outputs. The new state is the last `K-1` columns of that window.

    Prefill is varlen over the token axis (one block per channel-tile x token-chunk x
    request; each thread owns one channel and streams its chunk, holding the K-1
    causal window in registers -- no shared memory, no barrier). Decode is one block
    per channel-tile across the request-batch, since each request contributes
    exactly one token.
    """

    def __init__(
        self,
        config: Lfm3Config,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_kernel = int(config.conv_L_cache)
        self.use_bias = bool(config.conv_bias)
        self.hidden_size = config.hidden_size

        tp_size = get_parallel().attn_tp_size
        self.hidden_size_per_partition = self.hidden_size // tp_size

        self.in_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.hidden_size] * 3,
            bias=self.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )
        self.out_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=self.use_bias,
            input_is_parallel=True,
            quant_config=quant_config,
            reduce_results=False,
            prefix=f"{prefix}.out_proj",
        )

        self.conv_weight = nn.Parameter(
            torch.empty(self.hidden_size_per_partition, self.conv_kernel)
        )
        set_weight_attrs(self.conv_weight, {"weight_loader": sharded_weight_loader(0)})
        if self.use_bias:
            self.conv_bias = nn.Parameter(torch.empty(self.hidden_size_per_partition))
            set_weight_attrs(
                self.conv_bias, {"weight_loader": sharded_weight_loader(0)}
            )
        else:
            self.register_parameter("conv_bias", None)

    def _conv_prefill(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        mamba_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        """Per-request causal depthwise conv1d with **fused B/C gating** via the CuTe
        DSL kernel.

        Inputs:
          proj:           (T_total, 3*D_local) packed in_proj output
                          [B_gate | C_gate | x] (sglang-native token-major)
          conv_state:     (max_batch, D_local, K-1) per-request rolling state, mutated
          mamba_indices:  (num_reqs,) state-slot index per request
          query_start_loc:(num_reqs+1,) cumulative token offsets [0, t0, t0+t1, ...]
        Returns:
          (T_total, D_local) **C_gate-multiplied** conv output, token-major.

        The cached state is read for every request (zero-padded for fresh ones), so
        `has_initial_state` is uniformly true — chunked prefill carries state across
        chunks through `conv_state`, which the kernel reads and then advances.
        """
        D = self.hidden_size_per_partition
        if proj.shape[0] == 0:
            return proj.new_zeros(0, D)
        if not _use_cute_conv:
            # Torch fallback: form Bx, run the channel-major F.conv1d
            # path on the transpose, then apply the C gate and return token-major.
            Bx, C_gate = _lfm3_conv_input_gate(proj)
            Bx_t = Bx.transpose(0, 1).contiguous()  # (D, T)
            y_t = self._conv_prefill_torch(
                Bx_t, conv_state, mamba_indices, query_start_loc
            )  # (D, T)
            return C_gate * y_t.transpose(0, 1)  # (T, D)
        bias = self.conv_bias.to(proj.dtype) if self.conv_bias is not None else None
        # All-true is exact, not an approximation: MambaPool.alloc pre-zeroes fresh
        # slots ("clear at alloc time"), so reading them == zero-pad; continuation /
        # chunked-prefill requests must read their carried state. Don't derive this
        # from forward_batch.extend_prefix_lens (the LFM2 pattern): extend_* fields
        # stay None on TARGET_VERIFY (ForwardBatch.init_new skips them for
        # decode/verify modes), which routes here and needs all-true regardless.
        # The flags are constant, so they come from a per-device cache instead of a
        # torch.ones fill per conv layer per prefill step.
        has_initial_state = _ones_flags(mamba_indices.shape[0], proj.device)
        return gated_causal_conv1d_prefill(
            proj,
            conv_state,
            self.conv_weight.to(proj.dtype),
            bias,
            query_start_loc,
            mamba_indices,
            has_initial_state,
            max_seq_len=max_seq_len,
        )

    def _conv_prefill_torch(
        self,
        Bx_t: torch.Tensor,
        conv_state: torch.Tensor,
        mamba_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Per-request causal depthwise conv1d via `F.conv1d` (channel-major (D, T)).

        Matches the HF reference on devices without the CuTe kernel.
        The caller pre-computes Bx = B_gate*x (channel-major, in a compiled helper) and
        post-multiplies by C_gate; this routine only does the depthwise conv + state
        update. Its per-request loop has data-dependent control flow (varlen slicing +
        .tolist()), so it stays eager -- only the elementwise gates are compiled.
        """
        K = self.conv_kernel
        D = self.hidden_size_per_partition
        w = self.conv_weight.to(Bx_t.dtype).unsqueeze(1)  # (D, 1, K)
        bias = self.conv_bias.to(Bx_t.dtype) if self.conv_bias is not None else None

        starts = query_start_loc.tolist()
        idxs = mamba_indices.tolist()
        out_chunks: list[torch.Tensor] = []
        for i, slot in enumerate(idxs):
            s, e = starts[i], starts[i + 1]
            if s == e:
                continue
            seg = Bx_t[:, s:e]  # (D, T_i)
            state_i = conv_state[slot].to(Bx_t.dtype)  # (D, K-1)
            cat = torch.cat([state_i, seg], dim=-1)  # (D, K-1 + T_i)
            y = F.conv1d(cat.unsqueeze(0), w, bias=bias, groups=D)[0]  # (D, T_i)
            out_chunks.append(y)
            new_state = cat[:, -(K - 1) :]
            if new_state.shape[-1] < K - 1:
                new_state = F.pad(new_state, (K - 1 - new_state.shape[-1], 0))
            conv_state[slot] = new_state.to(conv_state.dtype)

        if not out_chunks:
            return Bx_t.new_zeros(D, 0)
        return torch.cat(out_chunks, dim=-1)

    def _maybe_track_conv_prefill(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        metadata: Mamba2Metadata,
    ) -> None:
        """Snapshot B*x at the prefix-cache boundaries planned by the backend."""
        if metadata.track_conv_indices is None:
            return
        D = self.hidden_size_per_partition
        sel = proj[metadata.track_conv_indices]  # [n_track, K-1, 3D]
        bx = sel[..., :D] * sel[..., 2 * D : 3 * D]
        conv_state[metadata.conv_states_mask_indices] = bx.transpose(1, 2).to(
            conv_state.dtype
        )

    def _conv_decode(
        self,
        proj: torch.Tensor,
        conv_state: torch.Tensor,
        mamba_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Single-token-per-request rolling conv update with **fused B/C gating**.

        Inputs:
          proj: (B, 3*D_local) packed in_proj output [B_gate | C_gate | x] (one token/req).
        Returns:
          (B, D_local) **C_gate-multiplied** conv output.

        The cached state for each request holds the last K-1 Bx values; the full
        causal K-window for the new token is `[state || B_gate*x]`. The kernel
        computes the per-channel dot with `conv_weight` (+bias), multiplies by
        `C_gate`, writes the output, and advances `conv_state` to the last K-1 Bx
        values of that window in place. Other devices use the compiled torch fallback.
        """
        if not _use_cute_conv:
            return _lfm3_conv_decode_torch(
                proj, conv_state, self.conv_weight, self.conv_bias, mamba_indices
            )
        bias = self.conv_bias.to(proj.dtype) if self.conv_bias is not None else None
        return gated_causal_conv1d_decode(
            proj, conv_state, self.conv_weight.to(proj.dtype), bias, mamba_indices
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return hidden_states

        req_to_token_pool = get_req_to_token_pool()
        layer_cache = req_to_token_pool.mamba2_layer_cache(self.layer_idx)
        conv_state = layer_cache.conv[0]
        metadata = get_attn_backend().linear_attn_backend.forward_metadata
        mamba_indices = metadata.mamba_cache_indices

        proj, _ = self.in_proj(hidden_states)
        # proj cols are [0:D)=B_gate, [D:2D)=C_gate, [2D:3D)=x (native in_proj order). The
        # gated conv reads these slices directly and fuses Bx=B*x and the C gate.
        if forward_batch.forward_mode.is_decode():
            conv_out = self._conv_decode(proj, conv_state, mamba_indices)
            if forward_batch.mamba_track_mask is not None:
                # extra_buffer prefix caching: when a request crosses a
                # mamba_track_interval boundary during decode, duplicate its
                # (just-advanced) working conv state into the checkpoint slot.
                # Same graph-safe triton helper Mamba2AttnBackend.forward uses;
                # `temporal` is the zero-sized SSM tensor for ShortConv, the
                # kernel copies 0 elements for it.
                track_mamba_states_if_needed(
                    conv_state,
                    layer_cache.temporal,
                    mamba_indices,
                    forward_batch.mamba_track_mask,
                    metadata.mamba_track_indices,
                    forward_batch.batch_size,
                )
        else:
            self._maybe_track_conv_prefill(proj, conv_state, metadata)
            conv_out = self._conv_prefill(
                proj,
                conv_state,
                mamba_indices,
                metadata.query_start_loc,
                max_seq_len=(
                    max(forward_batch.extend_seq_lens_cpu)
                    if forward_batch.extend_seq_lens_cpu
                    else None
                ),
            )

        output, _ = self.out_proj(
            conv_out,  # already C_gate-multiplied inside the conv kernel / fallback
        )
        return output


class Lfm3DecoderLayer(nn.Module):
    """Hybrid decoder layer with sandwich norms and upstream stage boundaries."""

    def __init__(
        self,
        config: Lfm3Config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.layer_type = config.layer_types[layer_id]
        self.is_attention_layer = self.layer_type in _ATTENTION_TYPES
        self.use_sandwich_norm = bool(config.use_sandwich_norm)

        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if self.use_sandwich_norm:
            self.post_operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        match self.layer_type:
            case "full_attention":
                self.self_attn = Lfm3Attention(
                    config=config,
                    layer_id=layer_id,
                    sliding_window=None,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                    alt_stream=alt_stream,
                )
            case "sliding_attention":
                if config.sliding_window is None:
                    raise ValueError(
                        "sliding_attention layer requires config.sliding_window"
                    )
                self.self_attn = Lfm3Attention(
                    config=config,
                    layer_id=layer_id,
                    sliding_window=config.sliding_window,
                    quant_config=quant_config,
                    prefix=add_prefix("self_attn", prefix),
                    alt_stream=alt_stream,
                )
            case "conv":
                self.conv = Lfm3ShortConv(
                    config=config,
                    layer_idx=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("conv", prefix),
                )
            case other:
                raise ValueError(
                    f"Unsupported layer_type {other!r} at layer {layer_id}. "
                    "LFM3 supports {full_attention, sliding_attention, conv}."
                )

        self.feed_forward = Lfm3MLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("feed_forward", prefix),
        )

        self.operator_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(
                    output_transform=OutputTransform(self.post_operator_norm)
                    if self.use_sandwich_norm
                    else None,
                ),
                self.operator_norm,
            ),
            (
                declare_ffn(
                    sparse=False,
                    next_layer_sparse=False,
                    output_transform=OutputTransform(self.post_ffn_norm)
                    if self.use_sandwich_norm
                    else None,
                ),
                self.ffn_norm,
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.operator_boundary.prepare(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            if self.is_attention_layer:
                hidden_states = self.self_attn(positions, hidden_states, forward_batch)
            else:
                hidden_states = self.conv(hidden_states, forward_batch)
        hidden_states = self.operator_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.feed_forward(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class Lfm3Model(nn.Module):
    def __init__(
        self,
        config: Lfm3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.num_attention_layers = sum(
            1 for lt in config.layer_types if lt in _ATTENTION_TYPES
        )

        # One side-channel CUDA stream shared by every gated-attention layer so
        # that qkv_proj and gate_proj can be issued concurrently under cuda
        # graph capture. Outside capture we fall through to a single-stream
        # schedule.
        alt_stream = torch.cuda.Stream() if _is_cuda else None

        def get_layer(idx: int, prefix: str, **_kwargs) -> Lfm3DecoderLayer:
            return Lfm3DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            input_embeds if input_embeds is not None else self.embed_tokens(input_ids)
        )

        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        return residual_batch.final_norm(
            hidden_states, forward_batch, self.embedding_norm, skip_empty=True
        )


class Lfm3ForCausalLM(nn.Module):
    """LFM3 for causal LM with hybrid attention/SWA/conv architecture."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Lfm3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_parallel().pp_group
        assert self.pp_group.is_first_rank and self.pp_group.is_last_rank

        self.quant_config = quant_config
        self.model = Lfm3Model(config, quant_config, prefix=add_prefix("model", prefix))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                org_num_embeddings=config.vocab_size,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(config)
        self.num_attention_layers = self.model.num_attention_layers

    def get_num_kv_cache_layers(self) -> int:
        return self.num_attention_layers

    def get_attention_sliding_window_size(self) -> int | None:
        # HF's `sliding_window` is a token COUNT (its mask allows
        # kv > q - sliding_window, i.e. `sliding_window` visible tokens incl. q).
        # SGLang's sliding_window_size is a max q-kv DISTANCE (kernel keeps
        # kv >= q - size, i.e. size+1 visible tokens). So size = sliding_window-1
        # gives identical visibility. Same convention as afmoe / exaone4.
        sw = self.config.sliding_window
        return sw - 1 if sw is not None else None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        is_mtp: bool = False,
    ) -> set[str]:
        # (param_name_in_module, weight_name_in_checkpoint, shard_id)
        # HF ships QKV as q_proj/k_proj/v_proj and SwiGLU MLP as w1 (gate) / w3
        # (up). We fuse both into the merged params used internally so the QKV
        # matmul and the gate/up matmul each run as a single TP-collective gemm
        # plus a silu_and_mul kernel call.
        stacked_params_mapping: list[tuple[str, str, str | int]] = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        unexpected_weights: list[str] = []

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.tie_word_embeddings and name.startswith("lm_head."):
                continue

            # HF stores the depthwise conv weight as `.conv.conv.weight` with
            # shape (D, 1, K); we hold a (D, K) parameter named `.conv.conv_weight`.
            if ".conv.conv.weight" in name:
                name = name.replace(".conv.conv.weight", ".conv.conv_weight")
                loaded_weight = loaded_weight.squeeze(1)
            elif ".conv.conv.bias" in name:
                name = name.replace(".conv.conv.bias", ".conv.conv_bias")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    unexpected_weights.append(name)
                    break
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    unexpected_weights.append(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        if unexpected_weights:
            raise ValueError(f"Unexpected LFM3 weights: {unexpected_weights}")
        return loaded_params


EntryClass = [Lfm3ForCausalLM]

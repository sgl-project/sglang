# Copyright 2025-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""RWKV-7 (Goose) model for sglang.

All elementwise math (token-shift lerp, projections, LoRAs, gating, GroupNorm,
gate-correction) is plain torch and matches the RWKV-LM numpy reference exactly; only
the WKV recurrence runs in a dedicated Triton kernel (via Rwkv7AttnBackend). Module /
parameter names mirror the fla-format checkpoint so `load_weights` uses
`default_weight_loader` with no remapping.

Quantization: the linear projections (r/k/v/o_proj, ffn key/value) and the
LoRA down/up projections are sglang quant-aware `ReplicatedLinear` (tp=1) threaded
with `quant_config`. With `quant_config=None` they are unquantized `F.linear`
(bit-identical to the previous `nn.Linear`, so greedy stays EXACT); quantized
configs carry their quantized weights through the same modules. The WKV
recurrence/state and the small per-channel params (x_*, k_k, k_a, r_k, g_norm)
are never quantized — they stay bf16/fp32.

Tensor parallelism is head-parallel: head_dim stays whole and whole heads are
split across ranks (r/k/v + LoRA-up column-parallel with no gather, per-channel
params / g_norm / WKV state on the local head slice, o_proj and ffn.value
row-parallel with a single allreduce each). The token-shift mix vectors and the
conv (prev-token) state stay full-width — they act on the replicated hidden
before the column-parallel projections. tp=1 keeps the exact original path.

Pipeline parallelism partitions the layer stack into contiguous per-rank slices
(llama-style make_layers + PPMissingLayer): the first rank owns the embeddings
(+ ln0 inside layer 0), the last rank owns the final norm + lm_head, and stages
hand off {hidden_states, v_first} as PPProxyTensors — v_first (layer 0's value
projection, under tp>1 the LOCAL head slice) must ride along because every later
layer's v-residual mix consumes it. Backend state stays indexed by GLOBAL
layer_id; the mamba/linear-state pool allocates only this rank's layer slice
(the runner filters by model.start_layer/end_layer). pp=1 keeps the exact
original path.

Per-layer time-mix (att):
  shifted = prev_token(x);  x* = x + x_*·(shifted - x)
  r = r_proj(xr); k = k_proj(xk); v = v_proj(xv)
  w_log = -e^-0.5 * sigmoid( w_up(tanh(w_down(xw))) + w_bias )       # log decay
  a = sigmoid( a_up(a_down(xa)) + a_bias )
  g = g_up( sigmoid(g_down(xg)) )                                    # no bias
  v-residual (layer>0): v += (v_first - v) * sigmoid( v_up(v_down(xv)) + v_bias )
  kk = k * k_k ; k = k + k*(a-1)*k_a ; kk = L2norm(kk) over head_dim
  y = WKV(r, w_log, k, v, kk, a)                                     # backend kernel
  y = g_norm(y) + (r*k*r_k).sum * v ; out = o_proj(y * g)
Channel-mix (ffn): shifted=prev(x); xk = x + x_k·(shifted-x); out = value(relu(key(xk))**2)
"""

import logging
from typing import Iterable, Optional, Set, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.attention.rwkv7_kernels.fused import (
    fused_gate_corr,
    fused_kk_kmix,
    fused_lerp6,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


def _tp_size() -> int:
    """TP world size, tolerating uninitialized distributed state (standalone
    tools may build layers without an engine)."""
    try:
        return get_tensor_model_parallel_world_size()
    except (AssertionError, ValueError):
        return 1


def _tp_rank() -> int:
    try:
        return get_tensor_model_parallel_rank()
    except (AssertionError, ValueError):
        return 0


# e^-0.5 = 1/sqrt(e); w_log = -this * sigmoid(w_raw)  =>  decay = exp(w_log).
_INV_SQRT_E = 0.6065306597126334


def _make_proj(
    in_f: int, out_f: int, quant_config, prefix: str, parallel: str = "column"
):
    """A bias-free projection: the quant-aware ReplicatedLinear at tp=1. Under tp>1
    the projection is head-parallel instead: ColumnParallelLinear (output = this
    rank's head slice, no gather) or RowParallelLinear (local-slice input,
    allreduce inside)."""
    if _tp_size() > 1:
        if parallel == "row":
            m = RowParallelLinear(
                in_f,
                out_f,
                bias=False,
                input_is_parallel=True,
                reduce_results=True,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            m = ColumnParallelLinear(
                in_f,
                out_f,
                bias=False,
                gather_output=False,
                quant_config=quant_config,
                prefix=prefix,
            )
    else:
        m = ReplicatedLinear(
            in_f, out_f, bias=False, quant_config=quant_config, prefix=prefix
        )
    return m


def _linear_backend(forward_batch: ForwardBatch):
    """The RWKV-7 linear-attention backend (HybridLinearAttnBackend's linear half).

    Normally read from the global forward context; batches that carry their own
    attn_backend (e.g. cuda-graph capture paths) take precedence."""
    ab = getattr(forward_batch, "attn_backend", None)
    if ab is None:
        ab = get_attn_backend()
    return ab.linear_attn_backend


class Rwkv7LoRA(nn.Module):
    """fla low-rank block: up(act(down(x))) [+ bias].

    Keys: lora.0.weight (down), lora.2.weight (up), lora.2.bias (up bias).

    The down/up projections are sglang ``ReplicatedLinear`` (tp=1) so they are
    quant-aware: with ``quant_config=None`` they fall through to an
    unquantized ``F.linear`` (bit-identical to ``nn.Linear``); with a quant
    config they carry int8/4-bit weights. The ``nn.Sequential`` is kept purely as
    a name container so checkpoint keys stay ``lora.0`` / ``lora.2`` (we drive the
    forward manually because ReplicatedLinear returns a ``(out, bias)`` tuple).

    Under tp>1 the down proj stays replicated (its input is the full replicated
    hidden and the rank-dim output is tiny, so every rank computes it locally,
    no comm) while the up proj is ColumnParallelLinear (no gather): its output —
    and its bias, sharded by the ColumnParallelLinear bias loader — is exactly
    this rank's head slice, matching the head-parallel r/k/v projections.
    """

    def __init__(
        self,
        hidden_size: int,
        low_rank: int,
        activation: str,
        bias: bool,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        if activation == "tanh":
            act = nn.Tanh()
        elif activation == "sigmoid":
            act = nn.Sigmoid()
        else:
            act = nn.Identity()
        if _tp_size() > 1:
            up = ColumnParallelLinear(
                low_rank,
                hidden_size,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        else:
            up = ReplicatedLinear(
                low_rank,
                hidden_size,
                bias=bias,
                quant_config=quant_config,
                prefix=add_prefix("lora.2", prefix),
            )
        self.lora = nn.Sequential(
            ReplicatedLinear(
                hidden_size,
                low_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("lora.0", prefix),
            ),
            act,
            up,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lora[0](x)
        h = self.lora[1](h)
        out, _ = self.lora[2](h)
        return out


class Rwkv7Attention(nn.Module):
    """RWKV-7 time-mixing block."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        # WKV heads tile the channel dim exactly; g_norm(num_groups=num_heads,
        # num_channels=H) and every [T, nh, hd] reshape below silently corrupt if
        # this is violated, so fail loudly at construction instead.
        assert self.num_heads * self.head_dim == self.hidden_size, (
            f"RWKV-7 head geometry mismatch: num_heads({self.num_heads}) * "
            f"head_dim({self.head_dim}) != hidden_size({self.hidden_size})"
        )
        # Head-parallel TP: head_dim stays whole, whole heads are split across
        # ranks. Everything downstream of the r/k/v/LoRA-up projections (per-
        # channel params, g_norm, the WKV recurrence and its state) lives on
        # this rank's head slice; o_proj (row-parallel) restores the full H.
        tp_size = _tp_size()
        assert self.num_heads % tp_size == 0, (
            f"RWKV-7 TP requires num_heads({self.num_heads}) divisible by "
            f"tp_size({tp_size})"
        )
        self.local_num_heads = self.num_heads // tp_size
        self.local_hidden_size = self.local_num_heads * self.head_dim

        H = self.hidden_size
        Hl = self.local_hidden_size
        # token-shift mix vectors (lerp coefficients)
        self.x_r = nn.Parameter(torch.zeros(1, 1, H))
        self.x_w = nn.Parameter(torch.zeros(1, 1, H))
        self.x_k = nn.Parameter(torch.zeros(1, 1, H))
        self.x_v = nn.Parameter(torch.zeros(1, 1, H))
        self.x_a = nn.Parameter(torch.zeros(1, 1, H))
        self.x_g = nn.Parameter(torch.zeros(1, 1, H))

        # Projections are quant-aware ReplicatedLinear (tp=1) / parallel linears (tp>1).
        self.r_proj = _make_proj(H, H, quant_config, add_prefix("r_proj", prefix))
        self.k_proj = _make_proj(H, H, quant_config, add_prefix("k_proj", prefix))
        self.v_proj = _make_proj(H, H, quant_config, add_prefix("v_proj", prefix))
        self.o_proj = _make_proj(
            H, H, quant_config, add_prefix("o_proj", prefix), parallel="row"
        )

        self.w_lora = Rwkv7LoRA(
            H,
            config.decay_low_rank_dim,
            "tanh",
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("w_lora", prefix),
        )
        self.a_lora = Rwkv7LoRA(
            H,
            config.a_low_rank_dim,
            "identity",
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("a_lora", prefix),
        )
        self.g_lora = Rwkv7LoRA(
            H,
            config.gate_low_rank_dim,
            "sigmoid",
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("g_lora", prefix),
        )
        if layer_id > 0:
            self.v_lora = Rwkv7LoRA(
                H,
                config.v_low_rank_dim,
                "identity",
                bias=True,
                quant_config=quant_config,
                prefix=add_prefix("v_lora", prefix),
            )

        self.k_k = nn.Parameter(torch.zeros(Hl))
        self.k_a = nn.Parameter(torch.zeros(Hl))
        self.r_k = nn.Parameter(torch.zeros(self.local_num_heads, self.head_dim))

        self.g_norm = nn.GroupNorm(
            num_groups=self.local_num_heads,
            num_channels=Hl,
            eps=self.head_dim * config.norm_eps,
            affine=True,
        )

        # Stacked token-shift mix vectors, lazily built (post weight-load)
        # on first forward and cached. Order [x_r, x_k, x_w, x_a, x_g, x_v].
        self._mix6 = None

    def _mix6_buf(self) -> torch.Tensor:
        if self._mix6 is None:
            self._mix6 = torch.stack(
                [
                    self.x_r.reshape(-1),
                    self.x_k.reshape(-1),
                    self.x_w.reshape(-1),
                    self.x_a.reshape(-1),
                    self.x_g.reshape(-1),
                    self.x_v.reshape(-1),
                ],
                dim=0,
            ).contiguous()
        return self._mix6

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T = x.shape[0]
        if T == 0:
            return x, v_first

        be = _linear_backend(forward_batch)
        # Local (per-rank) head slice; == the full width at tp=1.
        H, hd, nh = self.local_hidden_size, self.head_dim, self.local_num_heads

        # Fused triton elementwise path: bit-identical to the torch reference at
        # bf16/fp16 (verified), so it stacks with cuda-graph + int8. fp32 keeps the
        # original torch path (1-ULP reduction-order drift would risk the fp32 gate).
        fused = x.dtype != torch.float32

        if fused:
            shifted = be.token_shift(x, self.layer_id, 0, forward_batch)
            lp = fused_lerp6(x, shifted, self._mix6_buf())
            xr, xk, xw, xa, xg, xv = lp[0], lp[1], lp[2], lp[3], lp[4], lp[5]
        else:
            shifted = be.token_shift(x, self.layer_id, 0, forward_batch)
            d = shifted - x
            xr = x + self.x_r.view(-1) * d
            xw = x + self.x_w.view(-1) * d
            xk = x + self.x_k.view(-1) * d
            xv = x + self.x_v.view(-1) * d
            xa = x + self.x_a.view(-1) * d
            xg = x + self.x_g.view(-1) * d

        r = self.r_proj(xr)[0]
        k = self.k_proj(xk)[0]
        v = self.v_proj(xv)[0]

        if self.layer_id == 0:
            v_first = v

        # LoRA gates: w=decay, a=in-context-lr, g=output-gate, v=v-residual (layer>0).
        w_log = -torch.sigmoid(self.w_lora(xw)) * _INV_SQRT_E
        a = torch.sigmoid(self.a_lora(xa))
        g = self.g_lora(xg)
        if self.layer_id != 0:
            v = v + (v_first - v) * torch.sigmoid(self.v_lora(xv))

        if fused:
            # kk = L2norm(k·k_k) over hd; k <- k + k·(a-1)·k_a  (one launch)
            kk, k = fused_kk_kmix(k, a, self.k_k, self.k_a, nh)
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
        else:
            kk = k * self.k_k
            k = k + k * (a - 1.0) * self.k_a
            r = r.view(T, nh, hd)
            w_log = w_log.view(T, nh, hd)
            k = k.view(T, nh, hd)
            v = v.view(T, nh, hd)
            a = a.view(T, nh, hd)
            kk = kk.view(T, nh, hd)
            kk = kk / kk.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        o = be.recurrence(r, w_log, k, v, kk, a, self.layer_id, forward_batch)
        # o: [T, nh, hd]
        o = self.g_norm(o.reshape(T, H))
        if fused:
            # o = (g_norm(o) + (r*k*r_k).sum(-1)*v) * g   (one launch)
            o = fused_gate_corr(o, r, k, self.r_k, v, g, nh)
        else:
            gate_corr = ((r * k * self.r_k).sum(dim=-1, keepdim=True) * v).reshape(T, H)
            o = o + gate_corr
            o = o * g
        out = self.o_proj(o)[0]
        return out, v_first


class Rwkv7FeedForward(nn.Module):
    """RWKV-7 channel-mixing block (sqrelu)."""

    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        self.hidden_size = H
        inter = config.intermediate_size
        self.x_k = nn.Parameter(torch.zeros(H))
        # tp>1: key is column-parallel (local inter slice; sqrelu is elementwise so
        # it acts per-slice), value is row-parallel (allreduce restores the full H).
        self.key = _make_proj(H, inter, quant_config, add_prefix("key", prefix))
        self.value = _make_proj(
            inter, H, quant_config, add_prefix("value", prefix), parallel="row"
        )

    def forward(self, forward_batch: ForwardBatch, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        be = _linear_backend(forward_batch)
        shifted = be.token_shift(x, self.layer_id, 1, forward_batch)
        xk = x + self.x_k * (shifted - x)
        k = self.key(xk)[0]
        act = torch.relu(k) ** 2
        out = self.value(act)[0]
        return out


class Rwkv7DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        H = config.hidden_size
        eps = config.norm_eps
        bias = config.norm_bias
        if layer_id == 0:
            # ln0: applied ONCE to the embeddings (driven from Rwkv7Model.forward).
            self.pre_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.ffn_norm = nn.LayerNorm(H, eps=eps, bias=bias)
        self.attn = Rwkv7Attention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ffn = Rwkv7FeedForward(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("ffn", prefix),
        )

    def forward(
        self,
        forward_batch: ForwardBatch,
        x: torch.Tensor,
        v_first: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, v_first = self.attn(forward_batch, self.attn_norm(x), v_first)
        x = x + attn_out
        x = x + self.ffn(forward_batch, self.ffn_norm(x))
        return x, v_first


class Rwkv7Model(nn.Module):
    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        # PP: the first rank owns the embeddings (ln0 lives inside layer 0, which
        # make_layers also puts on the first rank), the last rank owns the final
        # norm; every other position is a PPMissingLayer placeholder. pp=1 (all
        # ranks first AND last) constructs exactly the original module tree.
        if self.pp_group.is_first_rank:
            self.embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
            )
        else:
            self.embeddings = PPMissingLayer()
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Rwkv7DecoderLayer(
                config, idx, quant_config=quant_config, prefix=prefix
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias
            )
        else:
            self.norm = PPMissingLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                x = inputs_embeds
            else:
                x = self.embeddings(input_ids)

            if x.shape[0] > 0:
                # ln0 on the embeddings (once), then the recurrent stack.
                x = self.layers[0].pre_norm(x)
            v_first = None
        else:
            assert pp_proxy_tensors is not None
            x = pp_proxy_tensors["hidden_states"]
            v_first = pp_proxy_tensors["v_first"]
            # v_first crosses the stage boundary FULL-WIDTH (see the send side:
            # sglang's pp tensor-dict transfer chunk-sends over the tp group and
            # all-gathers on receive, which is only lossless for tp-replicated
            # tensors) — slice back to this rank's head slice.
            tp_size = _tp_size()
            if tp_size > 1 and v_first.shape[0] > 0:
                Hl = v_first.shape[-1] // tp_size
                r = _tp_rank()
                v_first = v_first[:, r * Hl : (r + 1) * Hl].contiguous()

        for i in range(self.start_layer, self.end_layer):
            x, v_first = self.layers[i](forward_batch, x, v_first)

        if not self.pp_group.is_last_rank:
            # v_first (layer 0's value projection — under tp>1 the LOCAL head
            # slice, same layout on the matching tp rank of the next stage) rides
            # along with the hidden state: every later layer's v-residual mix
            # consumes it. It is None only for empty batches (T==0 skips every
            # layer); send a same-width empty placeholder so the p2p tensor dict
            # stays uniform.
            if v_first is None:
                v_first = x.new_zeros(
                    x.shape[0], self.layers[self.start_layer].attn.local_hidden_size
                )
            # sglang's pp transfer chunk-sends each tensor across the tp group and
            # reassembles rank-by-rank on receive — lossless ONLY for tp-replicated
            # tensors (#30015). v_first is the LOCAL head slice under tp>1, so
            # gather it to full width here (the receiver slices its head range
            # back out). Once send_tensor_dict honors the model-declared
            # `pp_proxy_tensors_all_gather_exclude` (#30095), this transit can send
            # the per-rank slice whole instead — kept for now so PP×TP is correct
            # regardless of which PR lands first.
            if _tp_size() > 1 and v_first.shape[0] > 0:
                v_first = tensor_model_parallel_all_gather(v_first.contiguous())
            return PPProxyTensors({"hidden_states": x, "v_first": v_first})

        x = self.norm(x)
        return x


class Rwkv7ForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    # PP proxy tensors that are NOT replicated across the attention-TP group:
    # v_first is a per-rank head slice. send_tensor_dict's slice/all-gather
    # optimization must send these whole (#30015 / #30095). Today the model
    # additionally all-gathers v_first to full width before the stage boundary
    # (see Rwkv7Model.forward), so correctness does not depend on #30095;
    # declaring the key here lets that transit be dropped once it lands.
    pp_proxy_tensors_all_gather_exclude = frozenset({"v_first"})

    # ---- BitsAndBytes (4-bit nf4 / 8-bit) support metadata ----
    # RWKV-7 has no fused/stacked projections (r/k/v/o are separate linears), so
    # the stacked-params mapping is empty. The target modules list the linear
    # sub-modules the bnb loader should quantize on the fly (substring match on
    # the checkpoint weight name); it mirrors the ReplicatedLinear layers above.
    bitsandbytes_stacked_params_mapping = {}
    default_bitsandbytes_target_modules = [
        ".r_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".key.",
        ".value.",
        ".lora.0.",
        ".lora.2.",
    ]

    def __init__(
        self,
        config: Rwkv7Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_pp_group()
        self.model = Rwkv7Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        # lm_head exists on every pp rank (llama pattern; only the last rank uses
        # it — the logits_processor runs there).
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            inputs_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        # Non-last pp rank: hand the PPProxyTensors (hidden_states + v_first) to
        # the next stage; logits only exist on the last rank.
        return hidden_states

    def get_embed_and_head(self):
        return self.model.embeddings.weight, self.lm_head.weight

    # The runner reads model.start_layer/end_layer (llama pattern) to size the
    # per-rank mamba/linear-state pool: under pp>1 only this rank's layer slice
    # is allocated and mamba2_layer_cache maps the GLOBAL layer_id to it.
    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        tp_size = _tp_size()
        tp_rank = _tp_rank()
        # Head-sharded per-channel params (tp>1): the checkpoint stores the full
        # tensor; narrow dim 0 (channels resp. heads) to this rank's head slice
        # before the plain copy. Parallel linears shard via their own weight_loader.
        _head_sharded = (".k_k", ".k_a", ".r_k", ".g_norm.weight", ".g_norm.bias")
        loaded_params: Set[str] = set()
        pp_skipped = 0
        for name, loaded_weight in weights:
            if name not in params_dict:
                # pp>1: keys for another stage's slice (layers outside
                # [start_layer, end_layer), the embeddings off the first rank,
                # the final norm off the last rank) are PPMissingLayer here —
                # skip them. Anything else is still a hard error, and at pp=1
                # every miss raises exactly as before.
                if self.pp_group.world_size > 1 and self._on_other_pp_rank(name):
                    pp_skipped += 1
                    continue
                raise KeyError(
                    f"[rwkv7.load_weights] unexpected checkpoint key: {name}"
                )
            param = params_dict[name]
            if tp_size > 1 and name.endswith(_head_sharded):
                shard = param.shape[0]
                loaded_weight = loaded_weight.narrow(0, tp_rank * shard, shard)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if pp_skipped:
            logger.info(
                "[rwkv7.load_weights] pp rank %s: skipped %d checkpoint keys "
                "owned by other pp ranks",
                self.pp_group.rank_in_group,
                pp_skipped,
            )

        # Assert every model parameter was loaded (catches naming mismatches).
        missing = set(params_dict.keys()) - loaded_params
        if missing:
            raise RuntimeError(
                f"[rwkv7.load_weights] {len(missing)} params not loaded, e.g. "
                f"{sorted(missing)[:8]}"
            )
        # Weight updates (update_weights_from_disk / RL sync) copy into x_* in
        # place; drop the stacked mix buffers so the next forward rebuilds them.
        for module in self.modules():
            if hasattr(module, "_mix6"):
                module._mix6 = None
        return loaded_params

    def _on_other_pp_rank(self, name: str) -> bool:
        """True iff this checkpoint key belongs to a module another pp rank owns
        (so this rank holds a PPMissingLayer for it and must skip the key)."""
        layer_id = get_layer_id(name)
        if layer_id is not None:
            return not (self.model.start_layer <= layer_id < self.model.end_layer)
        if name.startswith("model.embeddings."):
            return not self.pp_group.is_first_rank
        if name.startswith("model.norm."):
            return not self.pp_group.is_last_rank
        return False


# config.json architectures = ["RWKV7ForCausalLM"]; the registry keys by class
# __name__, so expose that spelling too (thin subclass).
class RWKV7ForCausalLM(Rwkv7ForCausalLM):
    pass


EntryClass = [Rwkv7ForCausalLM, RWKV7ForCausalLM]

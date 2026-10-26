# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""Kimi-K3 mono decode for ROCm TP8: the SGLang side of the FlyDSL kernels.

One persistent launch runs the whole MoE at a small decode shape -- the router
and its top-16, the latent down and shared gate_up projections, the MXFP4
experts, the shared down and latent up projections, and the two TP all-reduces,
which go over a HIP-IPC peer buffer rather than leaving the kernel.

Two properties are worth knowing before reading the rest:

* The router and the latent down projection are **TP sharded** here. Each rank
  computes its eighth of the rows and the result is all-gathered in-kernel, so
  the two widest weights on the decode critical path are read 8x less. The
  unfused path replicates both.
* Nothing is repacked. The dense weights are read as BF16 in their checkpoint
  layout, and the experts are read in whichever AITER A16W4 layout
  ``Mxfp4MoEMethod`` left behind. So there is no prepared-weight copy and no
  cost to the KV cache budget.

``SGLANG_K3_MONO_LAYER`` widens that launch to the whole layer (``MonoK2``):
it then starts from the attention's TP-local output, so o_proj, the
attention-TP all-reduce and the MLP attention-residual seam join the MoE in
the same launch. The attention itself and the final residual add stay outside.

Every rank must agree on whether to take this path: the kernels spin in-kernel
on each peer's mailbox, so a rank that opted out alone would leave the other
seven waiting for ever. ``_tp_agree`` makes the decision collective.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.attn_residual import AttnResidual
    from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer, KimiK3MoE

logger = logging.getLogger(__name__)

# The kernels are built for at most this many rows, because the rows have to
# fit one CTA's LDS. At concurrency 1 a DSPARK-7 verify step is exactly 8.
MONO_MAX_ROWS = 8
MONO_TP_SIZE = 8
HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
# o_proj's input width per rank: 12 heads of 128.
CORE_SIZE = 1536
# The attention-residual bank rows the layer launch is built for.
MONO_MAX_BLOCKS = 8

_logged: set[str] = set()


class MonoUnsupported(Exception):
    """This rank cannot serve the mono path."""


def _log_once(message: str) -> None:
    """Report a per-layer decision once. Every layer answers the same way."""
    if message not in _logged:
        _logged.add(message)
        logger.info("%s", message)


def _tp_agree(ok: bool) -> bool:
    """Return True only when every TP rank passed ``ok``.

    Every rank has to reach this at the same point. A MIN all-reduce over the
    gloo group is enough and keeps the decision off the compute stream.
    """
    tp_group = get_parallel().tp_group
    if tp_group.world_size == 1:
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=tp_group.cpu_group)
    return bool(flag.item())


def _bind_agreed(build) -> None:
    """Run ``build`` on every rank, and fail on all of them if any one fails."""
    try:
        build()
    except Exception:
        _tp_agree(False)
        raise
    if not _tp_agree(True):
        raise MonoUnsupported("another TP rank refused the mono path")


def step_begin() -> None:
    """Advance the mailbox epoch. Call once per forward, before the first layer.

    The epoch is a device counter, so a captured graph replays the bump. It is
    what lets a tag identify one launch of one step without anything being
    zeroed in between.
    """
    from sglang.kernels.ops.moe.k3_mono_flydsl import runner

    runner.step_begin()


def _is_decode_step(forward_batch) -> bool:
    """Whether this forward is a decode-side step the launches are meant for.

    Shape alone does not say: a prefill of eight tokens looks exactly like a
    DSPARK-7 verify step. Two reasons to ask anyway. The launches are a
    latency optimization for the decode path and buy nothing on a short
    prefill, and prefill is the phase that can run under a breakable CUDA
    graph, whose segmenting these launches have not been validated against.

    Target-verify counts: with DSPARK that *is* the decode step.
    """
    if forward_batch is None:
        return False
    mode = forward_batch.forward_mode
    return mode.is_decode_or_idle() or mode.is_target_verify()


def _experts_are_gu_interleaved(experts) -> bool:
    """Whether the experts carry the gate/up interleaved MXFP4 layout.

    ``is_shuffled`` alone cannot answer this: Mxfp4MoEMethod sets it on both of
    its branches, and only the SiTU A8W4 branch interleaves. Reading those
    bytes with the wrong layout does not fail, it returns wrong numbers, so ask
    the same question Mxfp4MoEMethod asked.
    """
    if getattr(experts.moe_runner_config, "activation", None) != "situ":
        return False
    from sglang.srt.layers.quantization.mxfp4 import (
        _aiter_situ_uses_gu_interleaved_weights,
    )

    return _aiter_situ_uses_gu_interleaved_weights()


def mono_supported(moe: KimiK3MoE) -> bool:
    """Static eligibility, evaluated once per layer.

    Only conditions that are the same on every TP rank belong here: this is
    read without a collective, so a per-rank answer would deadlock the peers.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.utils import is_hip

    if not envs.SGLANG_K3_MONO_DECODE.get() or not is_hip():
        return False

    experts = getattr(moe, "experts", None)
    shared = getattr(moe, "shared_experts", None)
    up_proj = getattr(moe, "routed_expert_up_proj", None)
    checks = (
        (moe.use_latent_moe, "not a latent MoE layer"),
        (moe.tp_size == MONO_TP_SIZE, f"needs TP{MONO_TP_SIZE}"),
        (get_moe_a2a_backend().is_none(), "needs --moe-a2a-backend none"),
        (not moe._dp_attention, "incompatible with DP attention"),
        (shared is not None, "needs shared experts"),
        (moe.routed_expert_norm is not None, "needs the latent RMSNorm"),
        (up_proj is not None, "needs the latent up projection"),
        (moe.moe_hidden_size == LATENT_SIZE, f"latent width must be {LATENT_SIZE}"),
        (moe.routed_scaling_factor == 1.0, "routed scaling must be 1.0"),
        (experts is not None and experts.top_k == 16, "top-k must be 16"),
        (
            experts is not None and getattr(experts, "num_local_experts", None) == 896,
            "expert count per rank must be 896",
        ),
        (
            experts is not None
            and getattr(experts, "intermediate_size_per_partition", None) == 384,
            "expert width per rank must be 384",
        ),
        (
            experts is not None
            and getattr(experts.w13_weight, "is_shuffled", False)
            and getattr(experts.w2_weight, "is_shuffled", False),
            "experts must be in AITER's preshuffled MXFP4 layout",
        ),
        (
            shared is not None and shared.down_proj.weight.dtype == torch.bfloat16,
            "shared down projection must be BF16",
        ),
        (
            up_proj is not None
            and up_proj.weight.shape[0] == HIDDEN_SIZE
            and up_proj.weight.is_contiguous(),
            "latent up projection must be a contiguous [hidden, latent]",
        ),
    )
    for ok, why in checks:
        if not ok:
            # The operator asked for this explicitly, so say why it did not
            # happen rather than falling back in silence.
            _log_once(f"Kimi-K3 mono decode off: {why}")
            return False
    return True


class MonoMoe:
    """Per-layer binding of one ``KimiK3MoE`` to the fused MoE launch.

    Everything is set up here, eagerly, before graph capture. That ordering is
    required, not an optimization: the decode path replays a captured graph, so
    whichever branch ``KimiK3MoE.forward`` takes at capture time is the branch
    every replay runs. It also has to be before capture because the HIP-IPC
    handshake allocates and runs a collective, neither of which is legal under
    capture.
    """

    def __init__(self, moe: KimiK3MoE) -> None:
        from sglang.kernels.ops.moe.k3_mono_flydsl import runner

        self._moe = moe
        self._layer_idx = moe.layer_idx
        self._rank = get_parallel().tp_rank
        self._guint = _experts_are_gu_interleaved(moe.experts)
        self._w_up: Optional[torch.Tensor] = None
        self._ready = False
        device = moe.gate.weight.device

        def build():
            runner.alloc_scratch(device)
            runner.peers(device)

        try:
            _bind_agreed(build)
            self._ready = True
        except Exception as error:
            logger.warning(
                "Kimi-K3 mono decode disabled on layer %d: %s", self._layer_idx, error
            )

    def w_up(self) -> torch.Tensor:
        """This rank's rows of the replicated latent up projection.

        Slicing the first dimension of a contiguous weight keeps it contiguous,
        so this is a view, not a copy.
        """
        if self._w_up is None:
            from sglang.kernels.ops.moe.k3_mono_flydsl.stages.moe import UP_N

            self._w_up = self._moe.routed_expert_up_proj.weight.narrow(
                0, self._rank * UP_N, UP_N
            )
        return self._w_up

    def covered(self, x: torch.Tensor, forward_batch) -> bool:
        """Whether this batch can take the fused launch.

        Fail closed. The kernel reads these tensors as raw addresses, so an
        unexpected dtype or stride would be read as if it were the expected one
        rather than raising.
        """
        return (
            self._ready
            and _is_decode_step(forward_batch)
            and 0 < x.size(0) <= MONO_MAX_ROWS
            and x.size(1) == HIDDEN_SIZE
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the whole MoE for one decode step. Returns the all-reduced sum
        of the routed and shared experts; the caller still owns the residual."""
        from sglang.kernels.ops.moe.k3_mono_flydsl import runner
        from sglang.kernels.ops.moe.k3_mono_flydsl.stages import moe as moe_stage

        _log_once("Kimi-K3 mono decode engaged on a decode step")
        m = self._moe
        experts = m.experts
        out = torch.empty_like(x)
        moe_stage.moe(
            moe_stage.MoeBuild(
                tokens=x.size(0),
                eps=m.routed_expert_norm.variance_epsilon,
                guint=self._guint,
            ),
            x=x,
            w_gate=m.gate.weight,
            bias=m.gate.e_score_correction_bias,
            w_ld=m.routed_expert_down_proj.weight,
            w_sgu=m.shared_experts.gate_up_proj.weight,
            w_sd=m.shared_experts.down_proj.weight,
            w13=experts.w13_weight,
            w13s=experts.w13_weight_scale,
            w2=experts.w2_weight,
            w2s=experts.w2_weight_scale,
            ln_w=m.routed_expert_norm.weight,
            w_up=self.w_up(),
            out=out,
            queue=runner.queue(),
            **runner.launch_args(),
            # K1 owns tags below 128; the MoE launches take the upper half.
            layer=128 + self._layer_idx,
        )
        return out


def maybe_build_mono(moe: KimiK3MoE) -> Optional[MonoMoe]:
    """Return this layer's mono binding, or None when it does not apply."""
    if not mono_supported(moe):
        return None
    mono = MonoMoe(moe)
    if not mono._ready:
        return None
    _log_once("Kimi-K3 mono decode armed (TP8, decode steps of at most 8 rows)")
    return mono


def k2_supported(layer: KimiK3DecoderLayer, mono: MonoMoe) -> bool:
    """Static eligibility of one decoder layer for the whole-layer launch.

    The MoE binding is already a precondition, so only what the wider launch
    adds is asked here: the attention seam it absorbs and the o_proj it runs
    itself. Same rule as ``mono_supported`` -- every condition has to answer
    the same on every TP rank.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_K3_MONO_LAYER.get():
        return False
    if not layer.use_attn_residuals:
        # prev_valid_blocks and the seam itself only exist with the stream.
        _log_once("Kimi-K3 mono layer off: needs the attention-residual stream")
        return False

    attn = layer.self_attn
    o_proj = getattr(attn, "o_proj", None)
    weight = getattr(o_proj, "weight", None)
    checks = (
        # The seam the launch replaces is the plain one. SP-MoE reduce-scatters
        # o_proj's output and the fused all-reduce completes it out of a
        # symmetric buffer; the launch does neither.
        (not layer._sp_moe, "incompatible with SP-MoE"),
        (not layer.all_reduce_fusion, "incompatible with the fused all-reduce"),
        (not layer._trim_padded_attn, "incompatible with mlp-sync padding"),
        (
            hasattr(attn, "forward_gated_core"),
            "attention cannot stop before o_proj",
        ),
        (
            weight is not None
            and weight.dtype == torch.bfloat16
            and tuple(weight.shape) == (HIDDEN_SIZE, CORE_SIZE)
            and weight.is_contiguous(),
            f"o_proj must be a contiguous BF16 [{HIDDEN_SIZE}, {CORE_SIZE}]",
        ),
        (
            o_proj is not None and getattr(o_proj, "bias", None) is None,
            "o_proj must have no bias",
        ),
        (
            _k2_blocks(layer) <= MONO_MAX_BLOCKS,
            f"at most {MONO_MAX_BLOCKS} attention-residual blocks",
        ),
    )
    for ok, why in checks:
        if not ok:
            _log_once(f"Kimi-K3 mono layer off: {why}")
            return False
    return mono is not None


def _k2_blocks(layer: KimiK3DecoderLayer) -> int:
    """Bank rows readable at this layer's MLP seam.

    A block-write layer snapshots its own pre-attention prefix on the
    attention side, so by the MLP seam it has one row more than it inherited.
    """
    return layer.prev_valid_blocks + int(layer.is_block_write_layer)


class MonoK2:
    """Per-layer binding of one ``KimiK3DecoderLayer`` to the layer launch.

    The launch starts from the attention's TP-local output (o_proj's input)
    and ends at the layer's MoE output, so it owns o_proj, the attention-TP
    all-reduce, the MLP attention-residual seam and the MoE. What stays
    outside is the attention itself and the final residual add.
    """

    def __init__(self, layer: KimiK3DecoderLayer, mono: MonoMoe) -> None:
        self._layer = layer
        self._mono = mono
        self._nblocks = _k2_blocks(layer)
        # A block-write layer restarts the prefix at the attention output.
        # The kernel then writes `prefix` instead of accumulating into it, so
        # this class owns that buffer. Allocate it before graph capture.
        self._prefix: Optional[torch.Tensor] = None
        if layer.is_block_write_layer:
            weight = layer.self_attn.o_proj.weight
            self._prefix = torch.empty(
                MONO_MAX_ROWS,
                HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=weight.device,
            )

    def covered(
        self,
        normed: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: AttnResidual,
        forward_batch,
        nvb: Optional[int] = None,
    ) -> bool:
        """Whether this batch can take the layer launch.

        ``nvb`` overrides the bank count to check against, for a caller that
        knows how many rows will be live by the time the launch runs.

        Fail closed, for the reason ``MonoMoe.covered`` fails closed: the
        kernel reads every tensor as a raw address, so a wrong dtype or stride
        is read as if it were the expected one.
        """
        rows = normed.size(0)
        if not (
            self._mono._ready
            and _is_decode_step(forward_batch)
            and 0 < rows <= MONO_MAX_ROWS
            and normed.size(1) == HIDDEN_SIZE
            and normed.dtype == torch.bfloat16
            and normed.is_contiguous()
        ):
            return False
        # The kernel's two prefix modes are chosen at build time, so the
        # caller's mode has to be the one this layer was built for.
        if (prefix_sum is None) != self._layer.is_block_write_layer:
            return False
        if prefix_sum is not None and not (
            prefix_sum.shape == normed.shape
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.is_contiguous()
        ):
            return False
        bank = attn_res.block_residual
        if nvb is None:
            nvb = attn_res.num_valid_blocks
        return (
            nvb == self._nblocks
            and bank.dtype == torch.bfloat16
            and bank.size(0) >= rows
            and bank.size(2) == HIDDEN_SIZE
            and bank.stride(2) == 1
        )

    def forward(
        self,
        normed: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        attn_res: AttnResidual,
        positions: torch.Tensor,
        forward_batch,
        zero_allocator,
        core: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run attention, then the whole layer tail in one launch.

        Returns the layer's output: the updated prefix plus the MoE output.
        ``core`` lets K1 hand over the attention output it already produced.
        """
        from sglang.kernels.ops.moe.k3_mono_flydsl import layer as k2
        from sglang.kernels.ops.moe.k3_mono_flydsl import runner

        _log_once("Kimi-K3 mono layer engaged on a decode step")
        layer = self._layer
        moe = layer.mlp
        experts = moe.experts
        if core is None:
            core = layer._run_self_attn_inner(
                normed, positions, forward_batch, zero_allocator, core=True
            )
        rows = core.size(0)
        assert core.shape == (rows, CORE_SIZE) and core.dtype == torch.bfloat16
        core = core.contiguous()

        if prefix_sum is None:
            # The prefix restarts at the attention output. The kernel writes
            # this buffer. It does not read the caller's.
            prefix = self._prefix[:rows]
        else:
            # Updated in place to prefix + attn_out, which is what the layer
            # output and the next layer's bank row both need.
            prefix = prefix_sum
        out = torch.empty_like(prefix)
        k2.k2_launch(
            k2.K2Build(
                tokens=rows,
                nblocks=self._nblocks,
                eps=layer.mlp_res_norm.variance_epsilon,
                out_eps=layer.post_attention_layernorm.variance_epsilon,
                ln_eps=moe.routed_expert_norm.variance_epsilon,
                reset=layer.is_block_write_layer,
                guint=self._mono._guint,
            ),
            core=core,
            w_o=layer.self_attn.o_proj.weight,
            prefix=prefix,
            blocks=attn_res.block_residual,
            ares_nw=layer.mlp_res_norm.weight,
            ares_qk=layer.mlp_res_proj.weight.view(HIDDEN_SIZE),
            in_nw=layer.post_attention_layernorm.weight,
            w_gate=moe.gate.weight,
            bias=moe.gate.e_score_correction_bias,
            w_ld=moe.routed_expert_down_proj.weight,
            w_sgu=moe.shared_experts.gate_up_proj.weight,
            w_sd=moe.shared_experts.down_proj.weight,
            w13=experts.w13_weight,
            w13s=experts.w13_weight_scale,
            w2=experts.w2_weight,
            w2s=experts.w2_weight_scale,
            ln_w=moe.routed_expert_norm.weight,
            w_up=self._mono.w_up(),
            out=out,
            **runner.launch_args(),
            # Same tag as this layer's MoE-only launch: only one of them runs
            # for a given layer in a given step.
            layer=128 + layer.layer_idx,
        )
        return out + prefix


def maybe_build_k2(layer: KimiK3DecoderLayer) -> Optional[MonoK2]:
    """Return this layer's whole-layer binding, or None when it does not apply."""
    mono = getattr(layer.mlp, "_mono", None)
    if mono is None or not k2_supported(layer, mono):
        return None
    _log_once("Kimi-K3 mono layer armed (o_proj + AttnRes seam + MoE in one launch)")
    return MonoK2(layer, mono)


# K1 shapes, from kernels/ops/moe/k3_mono_flydsl/attention/kda.py.
K1_CONV_W = 4
K1_WIN = K1_CONV_W - 1
K1_NPROJ = 6288
K1_HEADS = 12
K1_HEAD_DIM = 128
K1_QKV_DIM = 3 * CORE_SIZE  # the conv channels: q, k and v
# The launch rebases its conv buffer every call, writing qlen entries after the
# carried window. We own the buffer, so size it for the widest step.
K1_STATE_LEN = MONO_MAX_ROWS + K1_WIN


def k1_supported(layer: KimiK3DecoderLayer) -> bool:
    """Static eligibility of one KDA layer for the K1 launch.

    Every weight below is a view of something SGLang already built -- in
    particular ``_qkvgbfa_layer.weight`` is exactly K1's in-projection layout
    ([q,k,v,g | f_a | b | pad], 6288 rows), so nothing is repacked.
    """
    from sglang.srt.environ import envs

    if not (envs.SGLANG_K3_MONO_LAYER.get() and envs.SGLANG_K3_MONO_K1.get()):
        return False
    attn = layer.self_attn
    # K1 is the KDA half of the model; MLA layers keep their own attention and
    # only take K2. Check before anything else -- the conditions below read
    # KDA-only attributes, and a tuple of them is built eagerly.
    inner = getattr(attn, "attn", None)
    if inner is None or not hasattr(attn, "qkv_conv1d"):
        return False
    inproj = getattr(attn, "_qkvgbfa_layer", None)
    conv = attn.qkv_conv1d
    checks = (
        (getattr(attn, "use_full_rank_gate", False), "needs the full-rank gate"),
        (
            inproj is not None
            and tuple(inproj.weight.shape) == (K1_NPROJ, HIDDEN_SIZE)
            and inproj.weight.dtype == torch.bfloat16,
            "needs the merged ROCm in-projection (SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1)",
        ),
        (
            getattr(attn, "local_num_heads", None) == K1_HEADS
            and getattr(attn, "head_dim", None) == K1_HEAD_DIM,
            f"needs {K1_HEADS} local heads of {K1_HEAD_DIM}",
        ),
        (
            conv.bias is None and conv.weight.dtype == torch.float32,
            "conv1d must be bias-free FP32 (K1 has no bias term)",
        ),
        (
            getattr(inner, "lower_bound", None) is not None,
            "needs the KDA gate lower bound",
        ),
        (layer.prev_valid_blocks <= MONO_MAX_BLOCKS, "too many residual blocks"),
    )
    for ok, why in checks:
        if not ok:
            _log_once(f"Kimi-K3 mono K1 off: {why}")
            return False
    return True


class MonoK1:
    """Per-layer binding of one KDA layer to the K1 launch.

    K1 runs from the attention-residual seam to the gated core output, so it
    replaces aggregation 1, input_layernorm and the whole attention. Its output
    is what ``forward_gated_core`` returns, which is what the layer launch (K2)
    consumes -- the two chain with no glue.

    This class owns the conv buffer. SGLang's speculative window is not used.
    The buffer is reseeded from the committed conv state on every launch, which
    keeps ``num_acc`` at 1. That drops the deepest read index from
    ``qlen + CONV_W - 2`` to ``CONV_W - 1``. The launch then depends on no
    buffer that SGLang leaves stale. See test_kimi_k3_mono_kda_pre.py.
    """

    def __init__(self, layer: KimiK3DecoderLayer) -> None:
        attn = layer.self_attn
        self._layer = layer
        self._nblocks = layer.prev_valid_blocks
        self._write_idx = layer.prev_valid_blocks if layer.is_block_write_layer else -1
        self._w_in = attn._qkvgbfa_layer.weight
        self._w_fb = attn.f_b_proj.weight
        self._conv_w = attn.qkv_conv1d.weight.squeeze(1)
        self._a_log = attn.A_log.detach().reshape(-1)
        self._dt_bias = attn.dt_bias
        self._on_w = attn.o_norm.weight.data.to(torch.bfloat16).contiguous()
        self._lower_bound = float(attn.attn.lower_bound)
        self._eps = layer.self_attention_res_norm.variance_epsilon
        self._out_eps = layer.input_layernorm.variance_epsilon
        self._onorm_eps = float(attn.o_norm.eps)
        assert self._conv_w.shape == (K1_QKV_DIM, K1_CONV_W), self._conv_w.shape
        # Ours, so it never goes stale: reseeded from the committed state every
        # launch. One request's worth -- see covered().
        self._conv_buf = torch.empty(
            1,
            K1_QKV_DIM,
            K1_STATE_LEN,
            dtype=torch.bfloat16,
            device=self._w_in.device,
        )

    def covered(self, prefix: torch.Tensor, attn_res: AttnResidual) -> bool:
        """Fail closed, like the other two launches: the kernel reads raw
        addresses, so a wrong dtype or stride is read, not rejected."""
        bank = attn_res.block_residual
        return (
            0 < prefix.size(0) <= MONO_MAX_ROWS
            and prefix.size(1) == HIDDEN_SIZE
            and prefix.dtype == torch.bfloat16
            and prefix.is_contiguous()
            and attn_res.num_valid_blocks == self._nblocks
            and bank.dtype == torch.bfloat16
            and bank.size(2) == HIDDEN_SIZE
            and bank.stride(2) == 1
        )

    def forward(self, prefix, attn_res, forward_batch) -> Optional[torch.Tensor]:
        """Returns the gated core output, or None when the batch is not served.

        None is a clean fallback: nothing has been written yet when it is
        returned, so the caller just takes the normal path.
        """
        from sglang.kernels.ops.moe.k3_mono_flydsl import runner
        from sglang.kernels.ops.moe.k3_mono_flydsl.attention import kda
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = getattr(get_attn_backend(), "linear_attn_backend", None)
        state = (
            backend.kimi_k3_mono_state(self._layer.self_attn.attn, forward_batch)
            if hasattr(backend, "kimi_k3_mono_state")
            else None
        )
        if state is None:
            _log_once("Kimi-K3 mono K1 skipped: no speculative state")
            return None
        qlen = state["num_draft"]
        rows = state["rows"]
        slots = state["slots"]
        # Take the request count from the token rows, not from rows.numel().
        # A captured graph pads req_pool_indices out to the captured batch
        # size, so that table can be longer than the batch.
        #
        # One request per launch. K1 indexes the conv buffer and the recurrent
        # state with the same slot number. A private conv buffer therefore
        # serves one request only. covered() caps the rows at MONO_MAX_ROWS,
        # so a second request cannot fit anyway.
        if qlen <= 0 or prefix.size(0) != qlen or rows.numel() < 1:
            _log_once(
                f"Kimi-K3 mono K1 skipped: {prefix.size(0)} rows for a "
                f"{qlen}-token step, {rows.numel()} state rows"
            )
            return None
        inter_ssm = state["inter_ssm"]
        if inter_ssm.dtype != torch.float32 or inter_ssm.shape[1] != qlen:
            _log_once(
                f"Kimi-K3 mono K1 skipped: intermediate_ssm is "
                f"{inter_ssm.dtype} {tuple(inter_ssm.shape)}"
            )
            return None

        # Every index below stays on the device. Reading one out with int()
        # is a host sync. A CUDA graph capture rejects a host sync, and this
        # path only runs inside a captured decode graph.
        row = rows[:1].long()
        slot = slots[:1].long()
        st_idx = (row * qlen).view(1, 1) + torch.arange(
            qlen, device=prefix.device, dtype=torch.long
        ).view(1, qlen)
        # acc_idx = 0: the window sits at offset 0 and the initial recurrent
        # state is the one at st_idx[0]. Seed both from the committed state.
        # No buffer of this class then has to stay live across steps.
        num_acc = torch.ones(1, dtype=torch.int32, device=prefix.device)

        window = state["committed_conv"].index_select(0, slot)
        assert window.shape[-1] == K1_QKV_DIM and window.shape[-2] == K1_WIN, (
            f"unexpected committed conv layout {tuple(window.shape)}"
        )
        self._conv_buf[:, :, :K1_WIN] = window.transpose(-1, -2)
        flat_ssm = inter_ssm.reshape(-1, *inter_ssm.shape[2:])
        flat_ssm.index_copy_(
            0,
            st_idx[:, 0],
            state["committed_ssm"].index_select(0, slot).to(flat_ssm.dtype),
        )

        core = torch.empty(
            prefix.size(0), CORE_SIZE, dtype=torch.bfloat16, device=prefix.device
        )
        kda.kda_pre(
            kda.KdaPreBuild(
                tokens=prefix.size(0),
                qlen=qlen,
                nblocks=self._nblocks,
                delta=False,
                state_len=K1_STATE_LEN,
                eps=self._eps,
                out_eps=self._out_eps,
                onorm_eps=self._onorm_eps,
                lower_bound=self._lower_bound,
                write_idx=self._write_idx,
            ),
            prefix=prefix,
            delta=None,
            blocks=attn_res.block_residual,
            ares_nw=self._layer.self_attention_res_norm.weight,
            ares_qk=self._layer.self_attention_res_proj.weight.view(HIDDEN_SIZE),
            in_nw=self._layer.input_layernorm.weight,
            w_in=self._w_in,
            w_fb=self._w_fb,
            conv_w=self._conv_w,
            # Expanded so that its slot stride is zero. K1 indexes the conv
            # buffer with the number it indexes the recurrent state with.
            # That number is a row of SGLang's scratch, not of this buffer.
            # A zero stride folds every such number onto the single row here,
            # so the expanded size does not matter.
            conv_state=self._conv_buf.expand(2, -1, -1),
            a_log=self._a_log,
            dt_bias=self._dt_bias,
            on_w=self._on_w,
            rstate=flat_ssm,
            st_idx=st_idx.to(torch.int32),
            num_acc=num_acc,
            core_out=core,
            scratch=runner.scratch(),
            layer=self._layer.layer_idx,
            epoch=runner.epoch(),
        )
        # Hand the step's conv inputs to SGLang's speculative window so its own
        # accept-time commit (fused_conv_window_scatter_with_mask) advances the
        # committed state. view[s, j, k] aliases phys[s, j + k], so writing
        # column CONV_W - 2 of steps 1.. lays down one entry each.
        # Indexed with a tensor, not an int, for the reason above. The two
        # writes are disjoint in the physical buffer -- the first lands on
        # entries 0 .. CONV_W - 2, the second on the rest -- so the view's
        # overlapping windows cannot alias between them.
        view = state["inter_window"]
        view[row, 0] = window
        view[row, 1:, K1_WIN - 1] = self._conv_buf[
            :, :, K1_WIN : K1_STATE_LEN - 1
        ].transpose(-1, -2)
        _log_once("Kimi-K3 mono K1 engaged on a decode step")
        return core


def maybe_build_k1(layer: KimiK3DecoderLayer) -> Optional[MonoK1]:
    """Return this layer's K1 binding, or None when it does not apply."""
    if not k1_supported(layer):
        return None
    _log_once("Kimi-K3 mono K1 armed (KDA layers, seam + attention in one launch)")
    return MonoK1(layer)

"""ROCm FlyDSL mono decode for Qwen3.8 (``SGLANG_QWEN3_5_MONO_DECODE=1``, MI355X, TP8).

The kernels are vllm-project/vllm#60833's FlyDSL K1 / K2 (``sglang.kernels.ops.moe.qwen3_5_mono_flydsl``).
A pure decode step of at most ``MAX_TOKENS`` rows runs each decoder layer as
two persistent launches:

- K1 (``gdn_pre``, GDN layers): the residual add + input_layernorm ->
  in_proj_qkvz / in_proj_ba -> conv1d -> the gated delta rule -> the gated
  norm, writing out_proj's input.
- K2 (``layer_post``, every layer): out_proj / o_proj + its TP all-reduce + the
  residual add + post_attention_layernorm + the MoE (router, top-10, MXFP4
  experts, the gated shared expert) + the MoE all-reduce. A full-attention
  layer runs SGLang's attention up to o_proj's input, then K2.

The decode CUDA graph captures the step per batch size; any other step takes
the stock layers. Needs the shared expert unfused
(``--disable-shared-experts-fusion``). K1 also needs an fp32 SSM state (no
``--mamba-ssm-dtype bfloat16``) and no ReplaySSM ring.

With speculative decoding, a target-verify step of at most ``MAX_TOKENS``
rows (batch x draft tokens) runs K2 only: K1 writes no per-draft-token state
for the rollback, so GDN layers run SGLang's ``linear_attn.core`` there.

The all-reduces run in-kernel over a peer buffer every TP rank maps. Mailbox
tags carry a device epoch bumped once a step (captured with it), so no buffer
is zeroed between steps. Each width and kernel has its own scratch.
"""

import logging

import torch

from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layout as L
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_memory, get_parallel, get_spec
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.common import get_device_core_count

logger = logging.getLogger(__name__)


def enabled() -> bool:
    if not envs.SGLANG_QWEN3_5_MONO_DECODE.get() or not is_hip():
        return False
    if not is_gfx95_supported():
        logger.warning("SGLANG_QWEN3_5_MONO_DECODE needs gfx950 (MI355X); ignored")
        return False
    return True


def _is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def _unsupported(model) -> str | None:
    """Why the kernels cannot serve this model as built, else None."""
    from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

    c = model.config
    if get_tp_group().world_size != L.TP or get_parallel().attn_tp_size != L.TP:
        return "needs TP8 without DP attention"
    if model.pp_group.world_size != 1:
        return "pipeline parallel"
    # Every CTA spin-waits on the others, so all of them must be resident.
    if get_device_core_count() < L.BLOCKS:
        return f"fewer than {L.BLOCKS} compute units (a partitioned GPU)"
    # Their copy kernels on another stream can hold CUs the spinning CTAs need.
    if get_memory().enable_hierarchical_cache or get_memory().enable_lmcache:
        return "hierarchical cache or LMCache"
    shape = (
        c.hidden_size == L.HIDDEN
        and c.linear_num_key_heads == L.NK * L.TP
        and c.linear_num_value_heads == L.NV * L.TP
        and c.linear_key_head_dim == L.HD
        and c.linear_value_head_dim == L.HD
        and c.linear_conv_kernel_dim == L.CONV_W
        and c.num_experts == L.E
        and c.num_experts_per_tok == L.TOPK
        and c.moe_intermediate_size == L.RI * L.TP
        and c.shared_expert_intermediate_size == L.SI * L.TP
        and c.hidden_act == "silu"
        and getattr(c, "norm_topk_prob", True)
    )
    if not shape:
        return "not the Qwen3.8 shape"
    for layer in model.layers:
        mlp = layer.mlp
        if not isinstance(mlp, Qwen2MoeSparseMoeBlock):
            return "dense MLP"
        if mlp.num_fused_shared_experts or mlp.shared_expert is None:
            return "fused shared expert (use --disable-shared-experts-fusion)"
        if _is_gdn(layer):
            a = layer.linear_attn
            gdn = (
                a.activation == "silu"
                and a.conv1d.bias is None
                and a.norm.activation in ("silu", "swish")
                and a.norm.group_size is None
                and a.norm.norm_before_gate
            )
            if not gdn:
                return "GDN layer variant"
        elif not layer.attn_output_gate or layer.q_size != L.CORE:
            return "attention width or no output gate"
    return None


class MonoDecode:
    """Bound to one Qwen3.5 text model; built at model init on every TP rank."""

    def __init__(self, model):
        from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import PeerBuffer
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn, layer

        self.model = model
        why = _unsupported(model)
        self.ok = why is None
        self._runtime_ok: bool | None = None
        self._logged: set = set()
        # The target model runs no plain decode steps under speculative decoding.
        self.k1 = get_spec().speculative_algorithm is None
        if not self.ok:
            logger.info("Qwen3.8 mono decode off: %s", why)
            return
        dev = model.embed_tokens.weight.device

        def buf(n: int) -> torch.Tensor:
            return torch.zeros((n + 255) // 256 * 256, dtype=torch.uint8, device=dev)

        widths = range(1, L.MAX_TOKENS + 1)
        self.k1_scratch = {
            s: buf(
                max(
                    gdn.scratch_bytes(gdn.K1Build(tokens=s, first=f))
                    for f in (False, True)
                )
            )
            for s in widths
        }
        self.k2_scratch = {
            s: buf(layer.scratch_bytes(layer.K2Build(tokens=s))) for s in widths
        }
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        tp = get_tp_group()
        self.rank = tp.rank_in_group
        # collective: every rank builds the model, so every rank gets here
        self.peers = PeerBuffer(
            layer.peer_bytes(L.MAX_TOKENS),
            tp.cpu_group,
            tp.rank_in_group,
            tp.world_size,
            dev,
        )
        self.peers.bytes.zero_()
        logger.info(
            "Qwen3.8 mono decode on for %s steps of <= %d rows",
            "decode" if self.k1 else "target-verify (K2 only)",
            L.MAX_TOKENS,
        )

    def _runtime_unsupported(self, forward_batch) -> str | None:
        """After loading, on the first decode step: weights and mamba state."""
        bf = torch.bfloat16
        pool = get_attn_backend().linear_attn_backend.req_to_token_pool
        for layer in self.model.layers:
            mlp = layer.mlp
            ex = mlp.experts
            if not getattr(ex.w13_weight, "is_shuffled", False):
                return "routed experts not shuffled (needs aiter on gfx950)"
            if ex.w13_weight.shape != (L.E, 2 * L.RI, L.HIDDEN // 2):
                return f"w13 shape {tuple(ex.w13_weight.shape)} (EP or padding)"
            if ex.w13_weight.dtype != torch.uint8 or ex.w2_weight.dtype != torch.uint8:
                return "routed experts not MXFP4"
            if _is_gdn(layer) and not self.k1:
                w_o = layer.linear_attn.out_proj.weight
                dense = (w_o,)
            elif _is_gdn(layer):
                a = layer.linear_attn
                dense = (
                    a.in_proj_qkvz.weight,
                    a.in_proj_ba.weight,
                    a.conv1d.weight,
                    a.dt_bias,
                    a.norm.weight,
                    a.out_proj.weight,
                )
                if a.A_log.dtype != torch.float32:
                    return "A_log not fp32"
                st = pool.mamba2_layer_cache(a.layer_id)
                if st.temporal.dtype != torch.float32:
                    return "SSM state not fp32 (drop --mamba-ssm-dtype bfloat16)"
                if getattr(st, "replayssm_d", None) is not None:
                    return "ReplaySSM ring"
                if st.conv[0].dtype != bf or st.conv[0].shape[1:] != (
                    L.CONV,
                    L.CONV_W - 1,
                ):
                    return f"conv state {tuple(st.conv[0].shape)} {st.conv[0].dtype}"
                if not st.temporal[0].is_contiguous():
                    return "SSM state slot not dense"
                w_o = a.out_proj.weight
            else:
                dense = (layer.o_proj.weight,)
                w_o = layer.o_proj.weight
            dense += (
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
                mlp.gate.weight,
                mlp.shared_expert_gate.weight,
                mlp.shared_expert.gate_up_proj.weight,
                mlp.shared_expert.down_proj.weight,
            )
            if any(w.dtype != bf for w in dense):
                return f"layer {layer.layer_id}: a dense weight is not bf16"
            if w_o.shape != (L.HIDDEN, L.CORE):
                return f"layer {layer.layer_id}: o_proj shape {tuple(w_o.shape)}"
        return None

    def eligible(self, input_ids, forward_batch, input_embeds, pp_proxy_tensors):
        """A decode step (K1 on) or a target-verify step (K1 off) of at most
        ``MAX_TOKENS`` rows."""
        if not self.ok or input_embeds is not None or pp_proxy_tensors is not None:
            return False
        if input_ids is None or not 1 <= input_ids.size(0) <= L.MAX_TOKENS:
            return False
        mode = forward_batch.forward_mode
        if not (mode.is_decode() if self.k1 else mode.is_target_verify()):
            return False
        if self.k1 and forward_batch.spec_info is not None:
            return self._skip("spec_info set")
        if self.model.layers_to_capture:
            return self._skip("layers_to_capture set")
        lab = getattr(get_attn_backend(), "linear_attn_backend", None)
        if lab is None:
            return self._skip("no linear_attn_backend")
        if self.k1:
            idx = lab.forward_metadata.mamba_cache_indices
            if (
                idx is None
                or idx.dtype != torch.int32
                or idx.numel() < input_ids.size(0)
            ):
                return self._skip(
                    f"mamba_cache_indices {None if idx is None else (idx.dtype, idx.numel())}"
                )
            if not idx.is_contiguous():
                return self._skip("mamba_cache_indices not contiguous")
        if self._runtime_ok is None:
            why = self._runtime_unsupported(forward_batch)
            if why is not None:
                logger.warning("Qwen3.8 mono decode off: %s", why)
            self._runtime_ok = why is None
        return self._runtime_ok

    def _skip(self, why: str) -> bool:
        if why not in self._logged:
            self._logged.add(why)
            logger.warning("Qwen3.8 mono decode skips a decode step: %s", why)
        return False

    def forward(self, input_ids, positions, forward_batch) -> torch.Tensor:
        model = self.model
        s = input_ids.size(0)
        if s not in self._logged:
            self._logged.add(s)
            logger.info("Qwen3.8 mono decode step at width %d", s)
        self.epoch.add_(1)
        lab = get_attn_backend().linear_attn_backend
        idx = lab.forward_metadata.mamba_cache_indices
        pool = lab.req_to_token_pool
        h = model.embed_tokens(input_ids)
        residual: torch.Tensor | None = None
        for i, layer in enumerate(model.layers):
            if _is_gdn(layer) and self.k1:
                core = torch.empty(s, L.CORE, dtype=h.dtype, device=h.device)
                res_attn = torch.empty_like(h)
                self._k1(i, layer, pool, idx, h, residual, res_attn, core)
            else:
                if residual is None:
                    res_attn = h
                    x = layer.input_layernorm(h)
                else:
                    x, res_attn = layer.input_layernorm(h, residual)
                if _is_gdn(layer):
                    core = layer.linear_attn.core(x, forward_batch).contiguous()
                else:
                    core = layer.attention_core(positions, x, forward_batch)
                    core = core.contiguous()
            h, residual = self._k2(i, layer, core, res_attn)
        h, _ = model.norm(h, residual)
        return h

    def _k1(self, i, layer, pool, idx, h, residual, res_out, core) -> None:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn

        a = layer.linear_attn
        st = pool.mamba2_layer_cache(a.layer_id)
        gdn.gdn_pre(
            gdn.K1Build(
                tokens=h.size(0),
                first=residual is None,
                eps=layer.input_layernorm.variance_epsilon,
                norm_eps=a.norm.eps,
            ),
            hidden=h,
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=a.in_proj_qkvz.weight,
            w_ba=a.in_proj_ba.weight,
            conv_w=a.conv1d.weight,
            conv_state=st.conv[0],
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=st.temporal,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch[h.size(0)],
            epoch=self.epoch,
            layer=i,
        )

    def _k2(self, i, layer, core, residual) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layer as k2

        mlp = layer.mlp
        ex = mlp.experts
        w_o = (
            layer.linear_attn.out_proj.weight if _is_gdn(layer) else layer.o_proj.weight
        )
        s = core.size(0)
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        k2.layer_post(
            k2.K2Build(tokens=s, eps=layer.post_attention_layernorm.variance_epsilon),
            core=core,
            residual=residual,
            w_o=w_o,
            ln_w=layer.post_attention_layernorm.weight,
            w_gate=mlp.gate.weight,
            w_sg=mlp.shared_expert_gate.weight,
            w_sgu=mlp.shared_expert.gate_up_proj.weight,
            w_sd=mlp.shared_expert.down_proj.weight,
            w13=ex.w13_weight,
            w13s=ex.w13_weight_scale,
            w2=ex.w2_weight,
            w2s=ex.w2_weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.k2_scratch[s],
            peers=self.peers.addresses,
            rank=self.rank,
            epoch=self.epoch,
            layer=k2.TAG_SLOTS // 2 + i,
        )
        return out, res_out

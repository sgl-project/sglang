"""ROCm FlyDSL dense mono decode for Qwen3.8-27B (``SGLANG_QWEN3_8_DENSE_MONO=1``,
gfx950, TP1).

A pure decode step of at most ``MAX_TOKENS`` rows runs each decoder layer as at
most two persistent launches (``sglang.kernels.ops.moe.qwen3_5_mono_flydsl``):

- K1 (``dense_gdn``, GDN layers, ``SGLANG_QWEN3_8_DENSE_MONO_K1``): the residual
  add + input_layernorm -> in_proj_qkvz / in_proj_ba -> conv1d -> the gated
  delta rule -> the gated norm, writing out_proj's input.
- K2 (``dense_ffn``, every layer): out_proj / o_proj
  (``SGLANG_QWEN3_8_DENSE_MONO_OPROJ``) + the residual add +
  post_attention_layernorm + the MXFP4 gate_up, SiLU-mul and down.

Attention layers keep the stock input norm and attention core; lm_head stays
stock. Both kernels are bit-exact with the stock decode path (Triton
``gemm_afp4wfp4``, GemmaRMSNorm's ROCm Triton kernel, the Triton conv update,
the packed gated delta rule, RMSNormGated); ``prepare`` checks that with
``torch.equal`` on the loaded weights and turns the path off on any mismatch.
The kernels read (16, 16)-shuffled copies of the MXFP4 weights (about 11 GiB),
made before decode graph capture; prefill keeps the stock GEMMs on the
checkpoint layout.

The decode CUDA graph captures the step per batch size; any other step takes
the stock layers. Mailbox tags carry a device epoch bumped once a step
(captured with it), so the scratch is never zeroed between steps.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_memory
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.common import get_device_core_count

logger = logging.getLogger(__name__)

# The K1 self-check's pool: slot 0 unused, a pad row at width > 1.
_CHECK_SLOTS = 4


def enabled() -> bool:
    if not envs.SGLANG_QWEN3_8_DENSE_MONO.get() or not is_hip():
        return False
    if not is_gfx95_supported():
        logger.warning("SGLANG_QWEN3_8_DENSE_MONO needs gfx950; ignored")
        return False
    return True


def _is_gdn(layer) -> bool:
    return hasattr(layer, "linear_attn")


def _o_proj(layer):
    return layer.linear_attn.out_proj if _is_gdn(layer) else layer.o_proj


def _unsupported(model) -> str | None:
    """Why the kernels cannot serve this model as built, else None."""
    from sglang.kernels.ops.moe.k3_mono_flydsl.common.plan import BLOCKS
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.layers import layernorm
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    c = model.config
    if get_tp_group().world_size != 1:
        return "needs TP1"
    # The kernels follow the ROCm Triton Gemma norm's reduction; without aiter
    # GemmaRMSNorm takes torch ops.
    if not (layernorm._use_aiter and layernorm._has_rocm_triton_gemma_rms_norm):
        return "GemmaRMSNorm not on the ROCm Triton kernel (needs SGLANG_USE_AITER=1)"
    if model.pp_group.world_size != 1:
        return "pipeline parallel"
    # Every CTA spin-waits on the others, so all of them must be resident.
    if get_device_core_count() < BLOCKS:
        return f"fewer than {BLOCKS} compute units (a partitioned GPU)"
    # Their copy kernels on another stream can hold CUs the spinning CTAs need.
    if get_memory().enable_hierarchical_cache or get_memory().enable_lmcache:
        return "hierarchical cache or LMCache"
    if (
        c.hidden_size != D.HIDDEN
        or c.intermediate_size != D.INTER
        or c.hidden_act != "silu"
    ):
        return "not the Qwen3.8-27B shape"
    if any(not isinstance(layer.mlp, Qwen2MoeMLP) for layer in model.layers):
        return "not a dense MLP"
    return None


def _mxfp4_unsupported(proj, n: int, k: int) -> str | None:
    """Why ``proj`` is not a plain MXFP4 [n, k] linear on the Triton
    ``gemm_afp4wfp4`` path (the kernels' reference), else None."""
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
    from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
        QuarkW4A4MXFP4,
    )

    if not isinstance(proj.quant_method, QuarkLinearMethod) or not isinstance(
        proj.scheme, QuarkW4A4MXFP4
    ):
        return "not Quark MXFP4"
    w, ws = proj.weight, proj.weight_scale
    # The kernels shuffle their own copy; a weight some other path already
    # preshuffled would be shuffled twice.
    for marker in ("is_shuffled", "aiter_bpreshuffled"):
        if getattr(w, marker, False) or getattr(ws, marker, False):
            return f"weights already preshuffled ({marker})"
    if w.dtype != torch.uint8 or w.shape != (n, k // 2):
        return f"weight {tuple(w.shape)} {w.dtype}, want MXFP4 {n} x {k}"
    if ws is None or ws.shape != (n, k // 32) or not ws.is_contiguous():
        return "weight scales not plain e8m0 [N, K / 32]"
    return None


def _weights_unsupported(layer, k1: bool, oproj: bool) -> str | None:
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

    projs = [
        ("gate_up_proj", layer.mlp.gate_up_proj, 2 * D.INTER, D.HIDDEN),
        ("down_proj", layer.mlp.down_proj, D.HIDDEN, D.INTER),
    ]
    if oproj:
        projs.append(("o_proj", _o_proj(layer), D.HIDDEN, D.CORE))
    for name, proj, n, k in projs:
        why = _mxfp4_unsupported(proj, n, k)
        if why is not None:
            return f"{name}: {why}"
    return None


def _k1_weights_unsupported(a) -> str | None:
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

    for name, proj, n in (
        ("in_proj_qkvz", a.in_proj_qkvz, K.QKVZ),
        ("in_proj_ba", a.in_proj_ba, K.BA),
    ):
        why = _mxfp4_unsupported(proj, n, K.HIDDEN)
        if why is not None:
            return f"{name}: {why}"
    if (
        a.conv1d.weight.dtype != torch.bfloat16
        or a.conv1d.weight.numel() != K.CONV * K.CONV_W
    ):
        return "conv1d weight"
    if not a.conv1d.weight.is_contiguous() or a.conv1d.bias is not None:
        return "conv1d weight not contiguous or with bias"
    if a.A_log.dtype != torch.float32 or a.dt_bias.dtype != torch.bfloat16:
        return f"A_log {a.A_log.dtype} / dt_bias {a.dt_bias.dtype}"
    gated_norm = (
        a.norm.weight.dtype == torch.bfloat16
        and a.norm.activation in ("silu", "swish")
        and a.norm.group_size is None
        and a.norm.norm_before_gate
    )
    if not gated_norm:
        return "gated norm variant"
    if (a.num_k_heads, a.num_v_heads, a.head_k_dim, a.head_v_dim) != (
        K.NK,
        K.NV,
        K.HD,
        K.HD,
    ):
        return "head shape"
    return None


def _k1_state_unsupported(model, pool) -> str | None:
    """On the first decode step: the mamba pool's layout."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

    for layer in model.layers:
        if not _is_gdn(layer):
            continue
        st = pool.mamba2_layer_cache(layer.linear_attn.layer_id)
        if st.temporal.dtype != torch.float32 or st.temporal.shape[1:] != (
            K.NV,
            K.HD,
            K.HD,
        ):
            return f"SSM state {tuple(st.temporal.shape)} {st.temporal.dtype}"
        if not st.temporal[0].is_contiguous():
            return "SSM state slot not dense"
        if st.replayssm_d is not None:
            return "ReplaySSM ring"
        c = st.conv[0]
        if c.dtype != torch.bfloat16 or c.shape[1:] != (K.CONV, K.SL):
            return f"conv state {tuple(c.shape)} {c.dtype}"
    return None


def stock_gdn_decode(layer, h, residual, conv_state, ssm_state, idx):
    """The stock decode path of a GDN layer from its input to out_proj's input
    (as ``Qwen3_5GatedDeltaNet.forward`` + the Triton GDN backend run it on
    ROCm), on the given states. -> core, residual out."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

    a = layer.linear_attn
    if residual is None:
        x, res = layer.input_layernorm(h), h
    else:
        x, res = layer.input_layernorm(h, residual)
    qkvz, ba = a._forward_input_proj(x)
    q, k, v, z, b, ga = a.fix_query_key_value_ordering(qkvz, ba)
    mixed = torch.cat([q, k, v.reshape(v.size(0), -1)], dim=-1)
    mixed = causal_conv1d_update(
        mixed,
        conv_state,
        a.conv1d.weight.view(a.conv1d.weight.size(0), -1),
        a.conv1d.bias,
        a.activation,
        conv_state_indices=idx,
    )
    o = mixed.new_empty(h.size(0), 1, a.num_v_heads, a.head_v_dim)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed,
        a=ga.contiguous(),
        b=b.contiguous(),
        A_log=a.A_log,
        dt_bias=a.dt_bias,
        scale=a.head_k_dim**-0.5,
        initial_state=ssm_state,
        out=o,
        ssm_state_indices=idx,
        use_qk_l2norm_in_kernel=True,
    )
    core = a.norm(o.view(-1, a.head_v_dim), z.reshape(-1, a.head_v_dim))
    return core.view(h.size(0), -1), res


class DenseMono:
    """Bound to one Qwen3.5 text model; built at model init."""

    def __init__(self, model):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        self.model = model
        why = _unsupported(model)
        self.ok = why is None
        self._logged: set = set()
        self.w13: list[torch.Tensor] = []
        self.w2: list[torch.Tensor] = []
        self.w_o: list[torch.Tensor] = []
        # per layer: the shuffled (in_proj_qkvz, in_proj_ba) of a GDN layer, else None
        self.w_in: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.oproj = envs.SGLANG_QWEN3_8_DENSE_MONO_OPROJ.get()
        self.k1 = self.oproj and envs.SGLANG_QWEN3_8_DENSE_MONO_K1.get()
        # out_proj / o_proj return their input while a mono step traces the layers
        self._skip_o = False
        self._runtime_ok: bool | None = None
        if not self.ok:
            logger.info("Qwen3.8 dense mono off: %s", why)
            return
        dev = model.embed_tokens.weight.device
        self.scratch = torch.zeros(
            max(
                D.scratch_bytes(D.FfnBuild(tokens=s))
                for s in range(1, D.MAX_TOKENS + 1)
            ),
            dtype=torch.uint8,
            device=dev,
        )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.k1_scratch = None
        if self.k1:
            from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn

            self.k1_scratch = torch.zeros(
                max(
                    dense_gdn.scratch_bytes(dense_gdn.K1Build(tokens=s, first=f))
                    for s in range(1, D.MAX_TOKENS + 1)
                    for f in (False, True)
                ),
                dtype=torch.uint8,
                device=dev,
            )

    def prepare(self) -> None:
        """Before decode graph capture: the weight checks, the shuffled copies,
        then the kernels checked bit-exact against the stock layers."""
        if not self.ok:
            return
        for layer in self.model.layers:
            why = _weights_unsupported(layer, self.k1, self.oproj)
            if why is not None:
                self._off(f"layer {layer.layer_id} {why}")
                return
            if self.k1 and _is_gdn(layer):
                why = _k1_weights_unsupported(layer.linear_attn)
                if why is not None:
                    self.k1 = False
                    logger.warning(
                        "Qwen3.8 dense mono K1 off: layer %d %s", layer.layer_id, why
                    )
        self._shuffle_weights()
        why = self._self_check()
        if why is not None:
            self._off(f"self-check failed: {why}")
            return
        logger.info(
            "Qwen3.8 dense mono on for decode steps of <= %d rows, o_proj %s, "
            "GDN K1 %s (%.1f GiB of shuffled weights)",
            self._max_tokens(),
            "folded" if self.oproj else "stock",
            "on" if self.k1 else "off",
            sum(w.numel() for w in self.w13 + self.w2 + self.w_o) / 2**30
            + sum(a.numel() + b.numel() for a, b in filter(None, self.w_in)) / 2**30,
        )

    @staticmethod
    def _max_tokens() -> int:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        return D.MAX_TOKENS

    def _off(self, why: str) -> None:
        self.ok = False
        self.w13, self.w2, self.w_o, self.w_in = [], [], [], []
        logger.warning("Qwen3.8 dense mono off: %s", why)

    def _shuffle_weights(self) -> None:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        for layer in self.model.layers:
            self.w13.append(D.shuffle_w(layer.mlp.gate_up_proj.weight.data))
            self.w2.append(D.shuffle_w(layer.mlp.down_proj.weight.data))
            if self.oproj:
                o = _o_proj(layer)
                self.w_o.append(D.shuffle_w(o.weight.data))
                self._wrap_o(o)
            self.w_in.append(None)
            if self.k1 and _is_gdn(layer):
                a = layer.linear_attn
                self.w_in[-1] = (
                    D.shuffle_w(a.in_proj_qkvz.weight.data.contiguous()),
                    D.shuffle_w(a.in_proj_ba.weight.data.contiguous()),
                )

    def _wrap_o(self, o) -> None:
        stock = o.forward

        def forward(x, *args, **kwargs):
            if self._skip_o:
                return x, None
            return stock(x, *args, **kwargs)

        o.forward = forward

    def _self_check(self) -> str | None:
        why = self._ffn_check()
        if why is None and self.k1:
            why = self._k1_check()
        return why

    def _ffn_check(self) -> str | None:
        layer = self.model.layers[0]
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(0)
        for s in (1, 8):
            h = torch.randn(s, 5120, generator=g, device=dev).mul_(0.5).bfloat16()
            r = torch.randn(s, 5120, generator=g, device=dev).mul_(2).bfloat16()
            core = torch.randn(s, 6144, generator=g, device=dev).mul_(0.5).bfloat16()
            if self.oproj:
                h, _ = _o_proj(layer)(core)
            x, res_ref = layer.post_attention_layernorm(h.clone(), r.clone())
            ref = layer.mlp(x)
            self.epoch.add_(1)
            out, res = self._ffn(0, layer, core if self.oproj else h, r)
            if not torch.equal(res, res_ref):
                return f"K2 res_out at width {s}"
            if not torch.equal(out, ref):
                d = (out.float() - ref.float()).abs().max().item()
                return f"K2 out at width {s}: max abs err {d}"
        return None

    def _k1_check(self) -> str | None:
        """The first GDN layer at widths 1 and 8 (with and without the residual,
        a pad row) on synthetic conv / SSM states: core, residual and both
        states."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

        i, layer = next((i, l) for i, l in enumerate(self.model.layers) if _is_gdn(l))
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(1)
        for s, first in ((1, True), (8, False)):
            h = torch.randn(s, K.HIDDEN, generator=g, device=dev).mul_(0.5).bfloat16()
            r = (
                None
                if first
                else torch.randn(s, K.HIDDEN, generator=g, device=dev)
                .mul_(2)
                .bfloat16()
            )
            conv = torch.randn(1 + s, K.CONV, K.SL, generator=g, device=dev).bfloat16()
            ssm = torch.randn(1 + s, K.NV, K.HD, K.HD, generator=g, device=dev).mul_(
                0.05
            )
            idx = torch.arange(1, 1 + s, dtype=torch.int32, device=dev)
            if s > 1:
                idx[1] = -1
            conv_ref, ssm_ref = conv.clone(), ssm.clone()
            core_ref, res_ref = stock_gdn_decode(layer, h, r, conv_ref, ssm_ref, idx)
            self.epoch.add_(1)
            core, res = self._k1(i, layer, conv, ssm, idx, h, r)
            for name, got, ref in (
                ("core", core, core_ref),
                ("res_out", res, res_ref),
                ("conv state", conv, conv_ref),
                ("SSM state", ssm, ssm_ref),
            ):
                if not torch.equal(got, ref):
                    return f"K1 {name} at width {s}"
        return None

    def eligible(
        self, input_ids, forward_batch, input_embeds, pp_proxy_tensors, deepstack
    ):
        """A pure decode step of at most ``MAX_TOKENS`` rows. The VL wrapper hands
        a text decode step in as ``input_embeds``."""
        if not self.ok or not self.w13 or pp_proxy_tensors is not None:
            return False
        if deepstack is not None and deepstack.numel() > 0:
            return False
        rows = input_ids if input_embeds is None else input_embeds
        if rows is None or not 1 <= rows.size(0) <= self._max_tokens():
            return False
        if not forward_batch.forward_mode.is_decode():
            return False
        if forward_batch.spec_info is not None:
            return self._skip("spec_info set")
        if self.model.layers_to_capture:
            return self._skip("layers_to_capture set")
        if self.k1:
            return self._k1_eligible(rows.size(0))
        return True

    def _k1_eligible(self, s) -> bool:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        lab = get_attn_backend().linear_attn_backend
        idx = lab.forward_metadata.mamba_cache_indices
        if idx is None or idx.dtype != torch.int32 or idx.numel() < s:
            return self._skip(
                f"mamba_cache_indices {None if idx is None else (idx.dtype, idx.numel())}"
            )
        if not idx.is_contiguous():
            return self._skip("mamba_cache_indices not contiguous")
        if self._runtime_ok is None:
            why = _k1_state_unsupported(self.model, lab.req_to_token_pool)
            if why is not None:
                logger.warning("Qwen3.8 dense mono K1 off: %s", why)
                self.k1 = False
            self._runtime_ok = why is None
        return True

    def _skip(self, why: str) -> bool:
        if why not in self._logged:
            self._logged.add(why)
            logger.warning("Qwen3.8 dense mono skips a decode step: %s", why)
        return False

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        model = self.model
        h = model.embed_tokens(input_ids) if input_embeds is None else input_embeds
        s = h.size(0)
        if s not in self._logged:
            self._logged.add(s)
            logger.info("Qwen3.8 dense mono step at width %d", s)
        self.epoch.add_(1)
        self._skip_o = self.oproj
        try:
            h, residual = self._layers(h, positions, forward_batch)
        finally:
            self._skip_o = False
        h, _ = model.norm(h, residual)
        return h

    def _layers(self, h, positions, forward_batch):
        residual = None
        if self.k1:
            from sglang.srt.model_executor.forward_context import get_attn_backend

            lab = get_attn_backend().linear_attn_backend
            idx = lab.forward_metadata.mamba_cache_indices
        for i, layer in enumerate(self.model.layers):
            if self.k1 and _is_gdn(layer):
                a = layer.linear_attn
                st = lab.req_to_token_pool.mamba2_layer_cache(a.layer_id)
                core, res = self._k1(
                    i, layer, st.conv[0], st.temporal, idx, h, residual
                )
                # The stock decode's copies to prefix-cache track slots.
                lab._track_mamba_state_decode(
                    forward_batch, st.conv[0], st.temporal, idx, a.layer_id
                )
                h, residual = self._ffn(i, layer, core, res)
                continue
            if residual is None:
                res = h
                x = layer.input_layernorm(h)
            else:
                x, res = layer.input_layernorm(h, residual)
            if _is_gdn(layer):
                a = layer.linear_attn(x, forward_batch)
            else:
                a = layer.self_attention(
                    positions=positions, hidden_states=x, forward_batch=forward_batch
                )
            h, residual = self._ffn(i, layer, a.contiguous(), res.contiguous())
        return h, residual

    def _k1(self, i, layer, conv_state, ssm_state, idx, h, residual):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn

        a = layer.linear_attn
        s = h.size(0)
        core = torch.empty(s, dense_gdn.CORE, dtype=h.dtype, device=h.device)
        res_out = torch.empty_like(h)
        w_qkvz, w_ba = self.w_in[i]
        dense_gdn.dense_gdn_pre(
            dense_gdn.K1Build(
                tokens=s,
                first=residual is None,
                eps=layer.input_layernorm.variance_epsilon,
                norm_eps=a.norm.eps,
            ),
            hidden=h.contiguous(),
            residual=residual,
            res_out=res_out,
            ln_w=layer.input_layernorm.weight,
            w_qkvz=w_qkvz,
            w_qkvzs=a.in_proj_qkvz.weight_scale,
            w_ba=w_ba,
            w_bas=a.in_proj_ba.weight_scale,
            conv_w=a.conv1d.weight,
            conv_state=conv_state,
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=ssm_state,
            st_idx=idx,
            core=core,
            scratch=self.k1_scratch,
            epoch=self.epoch,
            layer=i,
        )
        return core, res_out

    def _ffn(self, i, layer, h, residual) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        mlp = layer.mlp
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        D.dense_ffn(
            D.FfnBuild(
                tokens=h.size(0),
                eps=layer.post_attention_layernorm.variance_epsilon,
                shuffled=True,
                oproj=self.oproj,
            ),
            hidden=h,
            w_o=self.w_o[i] if self.oproj else None,
            w_os=_o_proj(layer).weight_scale if self.oproj else None,
            residual=residual,
            ln_w=layer.post_attention_layernorm.weight,
            w13=self.w13[i],
            w13s=mlp.gate_up_proj.weight_scale,
            w2=self.w2[i],
            w2s=mlp.down_proj.weight_scale,
            out=out,
            res_out=res_out,
            scratch=self.scratch,
            epoch=self.epoch,
            layer=i,
        )
        return out, res_out

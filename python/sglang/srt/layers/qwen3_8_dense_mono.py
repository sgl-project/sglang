"""ROCm FlyDSL dense FFN stage for Qwen3.8-27B decode (``SGLANG_QWEN3_8_DENSE_MONO=1``,
gfx950, TP1). Study wiring.

A pure decode step of at most ``MAX_TOKENS`` rows runs each decoder layer's
out_proj / o_proj (``SGLANG_QWEN3_8_DENSE_MONO_OPROJ``, default on) + residual
add + post_attention_layernorm + MLP (MXFP4 gate_up, SiLU-mul, MXFP4 down) as
one persistent launch (``qwen3_5_mono_flydsl.dense_ffn``, K2). With
``SGLANG_QWEN3_8_DENSE_MONO_K1`` (default on, needs the o_proj fold) each GDN
layer's input norm + in_proj + conv + gated delta rule + gated norm is a second
launch (``qwen3_5_mono_flydsl.gdn27``, K1) that hands K2 its core. Attention
layers keep the stock input norm and attention core; lm_head stays stock.
The kernels read (16, 16)-shuffled copies of the MXFP4 weights, made before
decode graph capture (11 GB for 64 layers); prefill keeps the stock GEMMs on the
checkpoint layout.

The decode CUDA graph captures the step per batch size; any other step takes
the stock layers. Mailbox tags carry a device epoch bumped once a step
(captured with it), so the scratch is never zeroed between steps.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)


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
    """Why the kernel cannot serve this model as built, else None."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D
    from sglang.srt.distributed.parallel_state import get_tp_group
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    c = model.config
    if get_tp_group().world_size != 1:
        return "needs TP1"
    if model.pp_group.world_size != 1:
        return "pipeline parallel"
    if (
        c.hidden_size != D.HIDDEN
        or c.intermediate_size != D.INTER
        or c.hidden_act != "silu"
    ):
        return "not the Qwen3.8-27B shape"
    if any(not isinstance(layer.mlp, Qwen2MoeMLP) for layer in model.layers):
        return "not a dense MLP"
    return None


def _k1_weights_unsupported(a) -> str | None:
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn27 as K

    h = K.HIDDEN
    for name, proj, n in (("in_proj_qkvz", a.in_proj_qkvz, K.QKVZ), ("in_proj_ba", a.in_proj_ba, K.BA)):
        if proj.weight.dtype != torch.uint8 or proj.weight.shape != (n, h // 2):
            return f"{name} is not MXFP4 {n} x {h}"
        if proj.weight_scale.shape != (n, h // 32) or not proj.weight_scale.is_contiguous():
            return f"{name} scales {tuple(proj.weight_scale.shape)}"
    if a.conv1d.weight.dtype != torch.bfloat16 or a.conv1d.weight.numel() != K.CONV * K.CONV_W:
        return "conv1d weight"
    if not a.conv1d.weight.is_contiguous():
        return "conv1d weight not contiguous"
    if a.A_log.dtype != torch.float32 or a.dt_bias.dtype != torch.bfloat16:
        return f"A_log {a.A_log.dtype} / dt_bias {a.dt_bias.dtype}"
    if a.norm.weight.dtype != torch.bfloat16 or a.norm.activation != "swish":
        return "gated norm"
    if (a.num_k_heads, a.num_v_heads, a.head_k_dim, a.head_v_dim) != (K.NK, K.NV, K.HD, K.HD):
        return "head shape"
    return None


def _k1_state_unsupported(model, pool) -> str | None:
    """On the first decode step: the mamba pool's layout."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn27 as K

    for layer in model.layers:
        if not _is_gdn(layer):
            continue
        st = pool.mamba2_layer_cache(layer.linear_attn.layer_id)
        if st.temporal.dtype != torch.float32 or st.temporal.shape[1:] != (K.NV, K.HD, K.HD):
            return f"SSM state {tuple(st.temporal.shape)} {st.temporal.dtype}"
        if not st.temporal[0].is_contiguous():
            return "SSM state slot not dense"
        if getattr(st, "replayssm_d", None) is not None:
            return "ReplaySSM ring"
        c = st.conv[0]
        if c.dtype != torch.bfloat16 or c.shape[1:] != (K.CONV, K.SL):
            return f"conv state {tuple(c.shape)} {c.dtype}"
    return None


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
        if not self.ok:
            logger.info("Qwen3.8 dense mono off: %s", why)
            return
        dev = model.embed_tokens.weight.device
        self.scratch = torch.zeros(
            max(D.scratch_bytes(D.FfnBuild(tokens=s)) for s in range(1, D.MAX_TOKENS + 1)),
            dtype=torch.uint8,
            device=dev,
        )
        self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.k1_scratch = None
        if self.k1:
            from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn27

            self.k1_scratch = torch.zeros(
                max(
                    gdn27.scratch_bytes(gdn27.K1Build(tokens=s))
                    for s in range(1, D.MAX_TOKENS + 1)
                ),
                dtype=torch.uint8,
                device=dev,
            )
        self._runtime_ok: bool | None = None

    def prepare(self) -> None:
        """Before decode graph capture: the shuffled weight copies, then one layer
        checked against the stock norm + MLP."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        if not self.ok:
            return
        for layer in self.model.layers:
            gu, dn = layer.mlp.gate_up_proj, layer.mlp.down_proj
            plain = (
                gu.weight.dtype == torch.uint8
                and gu.weight.shape == (2 * D.INTER, D.HIDDEN // 2)
                and gu.weight_scale.shape == (2 * D.INTER, D.HIDDEN // 32)
                and dn.weight.dtype == torch.uint8
                and dn.weight.shape == (D.HIDDEN, D.INTER // 2)
                and dn.weight_scale.shape == (D.HIDDEN, D.INTER // 32)
            )
            if not plain:
                self.ok = False
                logger.warning(
                    "Qwen3.8 dense mono off: layer %d MLP is not plain MXFP4",
                    layer.layer_id,
                )
                return
            self.w13.append(D.shuffle_w(gu.weight.data))
            self.w2.append(D.shuffle_w(dn.weight.data))
            if self.oproj:
                o = _o_proj(layer)
                if o.weight.dtype != torch.uint8 or o.weight.shape != (
                    D.HIDDEN,
                    D.CORE // 2,
                ):
                    self.ok = False
                    logger.warning(
                        "Qwen3.8 dense mono off: layer %d o_proj is not MXFP4 %d x %d",
                        layer.layer_id,
                        D.HIDDEN,
                        D.CORE,
                    )
                    return
                self.w_o.append(D.shuffle_w(o.weight.data))
                self._wrap_o(o)
            self.w_in.append(None)
            if self.k1 and _is_gdn(layer):
                why = _k1_weights_unsupported(layer.linear_attn)
                if why is not None:
                    self.k1 = False
                    self.w_in = []
                    logger.warning(
                        "Qwen3.8 dense mono K1 off: layer %d %s", layer.layer_id, why
                    )
                else:
                    a = layer.linear_attn
                    self.w_in[-1] = (
                        D.shuffle_w(a.in_proj_qkvz.weight.data.contiguous()),
                        D.shuffle_w(a.in_proj_ba.weight.data.contiguous()),
                    )
        err = self._self_check()
        if err is not None:
            self.ok = False
            logger.warning("Qwen3.8 dense mono off: self-check failed (%s)", err)
            return
        logger.info(
            "Qwen3.8 dense mono on for decode steps of <= %d rows, o_proj %s, "
            "GDN K1 %s (%.1f GB of shuffled weights)",
            D.MAX_TOKENS,
            "folded" if self.oproj else "stock",
            "on" if self.k1 else "off",
            sum(w.numel() for w in self.w13 + self.w2 + self.w_o) / 2**30
            + sum(a.numel() + b.numel() for a, b in filter(None, self.w_in)) / 2**30,
        )

    def _wrap_o(self, o) -> None:
        stock = o.forward

        def forward(x, *args, **kwargs):
            if self._skip_o:
                return x, None
            return stock(x, *args, **kwargs)

        o.forward = forward

    def _self_check(self) -> str | None:
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
                return f"res_out at width {s}"
            if not torch.equal(out, ref):
                d = (out.float() - ref.float()).abs().max().item()
                return f"out at width {s}: max abs err {d}"
        return None

    def eligible(
        self, input_ids, forward_batch, input_embeds, pp_proxy_tensors, deepstack
    ):
        """A pure decode step of at most ``MAX_TOKENS`` rows. The VL wrapper hands
        a text decode step in as ``input_embeds``."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        if not self.ok or not self.w13 or pp_proxy_tensors is not None:
            return False
        if deepstack is not None and deepstack.numel() > 0:
            return False
        rows = input_ids if input_embeds is None else input_embeds
        if rows is None or not 1 <= rows.size(0) <= D.MAX_TOKENS:
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
            pool = lab.req_to_token_pool
        for i, layer in enumerate(self.model.layers):
            if self.k1 and _is_gdn(layer):
                core, res = self._k1(i, layer, pool, idx, h, residual)
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

    def _k1(self, i, layer, pool, idx, h, residual):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn27

        a = layer.linear_attn
        st = pool.mamba2_layer_cache(a.layer_id)
        s = h.size(0)
        core = torch.empty(s, gdn27.CORE, dtype=h.dtype, device=h.device)
        res_out = torch.empty_like(h)
        w_qkvz, w_ba = self.w_in[i]
        gdn27.gdn27_pre(
            gdn27.K1Build(
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
            conv_state=st.conv[0],
            a_log=a.A_log,
            dt_bias=a.dt_bias,
            norm_w=a.norm.weight,
            rstate=st.temporal,
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

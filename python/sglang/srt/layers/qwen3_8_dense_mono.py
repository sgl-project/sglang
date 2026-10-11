"""ROCm FlyDSL dense mono decode for Qwen3.8-27B (``SGLANG_QWEN3_8_DENSE_MONO=1``,
gfx950, TP1 or TP2).

A pure decode step of at most 8 rows (16 at TP2, see below) runs each decoder
layer as at most two persistent launches
(``sglang.kernels.ops.moe.qwen3_5_mono_flydsl``):

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

At TP2 the stock layers run the step and the kernels take the rank's share
between the all-reduces: each GDN layer's ``linear_attn`` takes K1 (its half of
the heads, from the stock input_layernorm's output) up to out_proj's partial
(out_proj folded in with ``SGLANG_QWEN3_8_DENSE_MONO_OPROJ``, else stock), and
each layer's ``mlp`` takes K2 (``SGLANG_QWEN3_8_DENSE_MONO_K2``, its half of the
intermediate, from the stock post_attention_layernorm's output) up to down's
partial. The all-reduces, the norms, attention and its o_proj stay stock.
K1 and K2 each take steps of up to 8 rows by default, up to 16 with
``SGLANG_QWEN3_8_DENSE_MONO_K1_MAX_ROWS`` / ``_K2_MAX_ROWS``; a step wider than
one kernel's cap runs that part stock.

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
    from sglang.srt.layers.layernorm import gemma_rms_norm_uses_rocm_triton
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP

    c = model.config
    if get_tp_group().world_size not in (1, 2):
        return "needs TP1 or TP2"
    # The kernels follow the ROCm Triton Gemma norm's reduction; without aiter
    # GemmaRMSNorm takes torch ops.
    if not gemma_rms_norm_uses_rocm_triton():
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
    if getattr(proj, "dequantized_bf16", False):
        return "MXFP4 weights dequantized to bf16 (AITER_FORCE_A8W4)"
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


def _weights_unsupported(layer, oproj: bool, tp: int) -> str | None:
    """Why K2 cannot run this layer's FFN share at TP ``tp``, else None."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

    inter = D.FGeo(tp).inter
    projs = [
        ("gate_up_proj", layer.mlp.gate_up_proj, 2 * inter, D.HIDDEN),
        ("down_proj", layer.mlp.down_proj, D.HIDDEN, inter),
    ]
    if oproj:
        projs.append(("o_proj", _o_proj(layer), D.HIDDEN, D.CORE))
    for name, proj, n, k in projs:
        why = _mxfp4_unsupported(proj, n, k)
        if why is not None:
            return f"{name}: {why}"
    return None


def _k1_weights_unsupported(a, tp: int) -> str | None:
    """Why K1 cannot run this GDN layer's share at TP ``tp``, else None."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

    geo = K.Geo(tp)
    if a.attn_tp_size != tp:
        return f"GDN heads split {a.attn_tp_size} ways at TP{tp}"
    for name, proj, n in (
        ("in_proj_qkvz", a.in_proj_qkvz, geo.qkvz),
        ("in_proj_ba", a.in_proj_ba, geo.ba),
    ):
        why = _mxfp4_unsupported(proj, n, K.HIDDEN)
        if why is not None:
            return f"{name}: {why}"
    if (
        a.conv1d.weight.dtype != torch.bfloat16
        or a.conv1d.weight.numel() != geo.conv * K.CONV_W
    ):
        return "conv1d weight"
    if not a.conv1d.weight.is_contiguous() or a.conv1d.bias is not None:
        return "conv1d weight not contiguous or with bias"
    if a.A_log.dtype != torch.float32 or a.dt_bias.dtype != torch.bfloat16:
        return f"A_log {a.A_log.dtype} / dt_bias {a.dt_bias.dtype}"
    if a.A_log.numel() != geo.nv or a.dt_bias.numel() != geo.nv:
        return f"A_log / dt_bias of {a.A_log.numel()} heads, want {geo.nv}"
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


def _k1_state_unsupported(model, pool, tp: int) -> str | None:
    """On the first decode step: the mamba pool's layout."""
    from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

    geo = K.Geo(tp)
    for layer in model.layers:
        if not _is_gdn(layer):
            continue
        st = pool.mamba2_layer_cache(layer.linear_attn.layer_id)
        if st.temporal.dtype != torch.float32 or st.temporal.shape[1:] != (
            geo.nv,
            K.HD,
            K.HD,
        ):
            return f"SSM state {tuple(st.temporal.shape)} {st.temporal.dtype}"
        if not st.temporal[0].is_contiguous():
            return "SSM state slot not dense"
        if st.replayssm_d is not None:
            return "ReplaySSM ring"
        c = st.conv[0]
        if c.dtype != torch.bfloat16 or c.shape[1:] != (geo.conv, K.SL):
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
    o = mixed.new_empty(h.size(0), 1, a.num_v_heads // a.attn_tp_size, a.head_v_dim)
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
        from sglang.srt.distributed.parallel_state import get_tp_group

        self.model = model
        why = _unsupported(model)
        self.ok = why is None
        self._logged: set = set()
        self.w13: list[torch.Tensor] = []
        self.w2: list[torch.Tensor] = []
        self.w_o: list[torch.Tensor] = []
        # per layer: the shuffled (in_proj_qkvz, in_proj_ba) of a GDN layer, else None
        self.w_in: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.tp = get_tp_group().world_size
        # TP1 runs the whole step (K1 + K2); TP2 K1 / K2 inside the stock layers.
        self.whole_step = self.tp == 1
        self.oproj = self.whole_step and envs.SGLANG_QWEN3_8_DENSE_MONO_OPROJ.get()
        k1_flag = envs.SGLANG_QWEN3_8_DENSE_MONO_K1.get()
        self.k1 = k1_flag and (self.oproj or not self.whole_step)
        # TP2: out_proj folded into K1
        self.k1_oproj = (
            self.k1
            and not self.whole_step
            and envs.SGLANG_QWEN3_8_DENSE_MONO_OPROJ.get()
        )
        # TP2: K2 in each layer's mlp
        self.k2 = not self.whole_step and envs.SGLANG_QWEN3_8_DENSE_MONO_K2.get()
        # the widest step K1 / K2 take
        self.k1_rows, self.k2_rows = self._row_caps()
        # out_proj / o_proj return their input while a mono step traces the layers
        self._skip_o = False
        # TP2: whether the step being run (or captured) takes K1 in linear_attn
        # and K2 in mlp
        self._gdn_step = False
        self._ffn_step = False
        self._prepared = False
        self._runtime_ok: bool | None = None
        if self.ok and not self.whole_step and not (self.k1 or self.k2):
            self.ok, why = False, "TP2 with neither K1 nor K2"
        if not self.ok:
            logger.info("Qwen3.8 dense mono off: %s", why)
            return
        dev = model.embed_tokens.weight.device
        self.scratch = None
        if self.whole_step or self.k2:
            self.scratch = torch.zeros(
                max(
                    D.scratch_bytes(self._k2_key(D, s))
                    for s in range(1, self.k2_rows + 1)
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
                    dense_gdn.scratch_bytes(self._k1_key(dense_gdn, s, first=f))
                    for s in range(1, self.k1_rows + 1)
                    for f in (False, True)
                ),
                dtype=torch.uint8,
                device=dev,
            )

    def _row_caps(self) -> tuple[int, int]:
        """K1's and K2's widest step: 8 at TP1, the env caps (1..16) at TP2."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        if self.whole_step:
            return D.MAX_TOKENS, D.MAX_TOKENS
        top = D.max_tokens(self.tp)
        caps = []
        for env in (
            envs.SGLANG_QWEN3_8_DENSE_MONO_K1_MAX_ROWS,
            envs.SGLANG_QWEN3_8_DENSE_MONO_K2_MAX_ROWS,
        ):
            cap = env.get()
            if not 1 <= cap <= top:
                cap = min(max(cap, 1), top)
                logger.warning(
                    "%s=%d: not in 1..%d, using %d", env.name, env.get(), top, cap
                )
            caps.append(cap)
        return caps[0], caps[1]

    def _k1_key(self, dense_gdn, s: int, *, first: bool, eps=1e-6, norm_eps=1e-6):
        """K1's build at width ``s``: at TP2 from the stock norm's output."""
        if self.whole_step:
            return dense_gdn.K1Build(tokens=s, first=first, eps=eps, norm_eps=norm_eps)
        return dense_gdn.K1Build(
            tokens=s, norm_eps=norm_eps, tp=self.tp, normed=True, oproj=self.k1_oproj
        )

    def _k2_key(self, D, s: int, eps=1e-6):
        """K2's build at width ``s``: at TP2 from the stock norm's output."""
        if self.whole_step:
            return D.FfnBuild(tokens=s, eps=eps, shuffled=True, oproj=self.oproj)
        return D.FfnBuild(tokens=s, shuffled=True, tp=self.tp, normed=True)

    def prepare(self) -> None:
        """Before decode graph capture: the weight checks, the shuffled copies,
        then the kernels checked bit-exact against the stock layers. At TP2 the
        ranks agree on the outcome."""
        why = self._prepare() if self.ok else None
        if self.tp > 1:
            why = self._agree(why)
        if why is not None:
            self._off(why)
            return
        if not self.ok:
            return
        self._prepared = True
        logger.info(
            "Qwen3.8 dense mono on for decode steps of <= %d rows at TP%d, o_proj "
            "%s, FFN %s, GDN K1 %s (%.1f GiB of shuffled weights)",
            self._max_tokens(),
            self.tp,
            "folded" if self.oproj else "in K1" if self.k1_oproj else "stock",
            f"K2 (<= {self.k2_rows} rows)" if self.whole_step or self.k2 else "stock",
            f"on (<= {self.k1_rows} rows)" if self.k1 else "off",
            sum(w.numel() for w in self.w13 + self.w2 + self.w_o if w is not None)
            / 2**30
            + sum(a.numel() + b.numel() for a, b in filter(None, self.w_in)) / 2**30,
        )

    def _prepare(self) -> str | None:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        for layer in self.model.layers:
            if self.whole_step or self.k2:
                why = _weights_unsupported(layer, self.oproj, self.tp)
                if why is not None:
                    return f"layer {layer.layer_id} {why}"
            if self.k1 and _is_gdn(layer):
                why = _k1_weights_unsupported(layer.linear_attn, self.tp)
                if why is None and self.k1_oproj:
                    why = _mxfp4_unsupported(
                        layer.linear_attn.out_proj,
                        D.HIDDEN,
                        D.CORE // self.tp,
                    )
                    why = None if why is None else f"out_proj: {why}"
                if why is not None:
                    if not self.whole_step:
                        return f"K1 needed at TP2: layer {layer.layer_id} {why}"
                    self.k1 = False
                    logger.warning(
                        "Qwen3.8 dense mono K1 off: layer %d %s", layer.layer_id, why
                    )
        self._shuffle_weights()
        why = self._self_check()
        if why is not None:
            return f"self-check failed: {why}"
        return None

    def _agree(self, why: str | None) -> str | None:
        """Off on every rank when off on any: at TP2 the ranks must all run
        either K1 or the stock GDN path."""
        from sglang.srt.distributed.parallel_state import get_tp_group

        ok = torch.tensor([int(self.ok and why is None)], dtype=torch.int32)
        torch.distributed.all_reduce(
            ok, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
        )
        if why is None and self.ok and not ok.item():
            return "off on another TP rank"
        return why

    def _max_tokens(self) -> int:
        """The widest step a kernel takes (K1 may turn off on the first step)."""
        return max(
            self.k1_rows if self.k1 else 0,
            self.k2_rows if self.whole_step or self.k2 else 0,
        )

    @staticmethod
    def _check_widths(cap: int) -> list[int]:
        """The self-check's widths: 1, 8 and the cap."""
        return sorted({1, min(8, cap), cap})

    def _off(self, why: str) -> None:
        self.ok = False
        self.w13, self.w2, self.w_o, self.w_in = [], [], [], []
        logger.warning("Qwen3.8 dense mono off: %s", why)

    def _shuffle_weights(self) -> None:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        for i, layer in enumerate(self.model.layers):
            if self.whole_step or self.k2:
                self.w13.append(D.shuffle_w(layer.mlp.gate_up_proj.weight.data))
                self.w2.append(D.shuffle_w(layer.mlp.down_proj.weight.data))
            if self.k2:
                self._wrap_mlp(i, layer)
            if self.oproj:
                o = _o_proj(layer)
                self.w_o.append(D.shuffle_w(o.weight.data))
                self._wrap_o(o)
            if self.k1_oproj:
                o = layer.linear_attn.out_proj if _is_gdn(layer) else None
                self.w_o.append(None if o is None else D.shuffle_w(o.weight.data))
            self.w_in.append(None)
            if self.k1 and _is_gdn(layer):
                a = layer.linear_attn
                self.w_in[-1] = (
                    D.shuffle_w(a.in_proj_qkvz.weight.data.contiguous()),
                    D.shuffle_w(a.in_proj_ba.weight.data.contiguous()),
                )
                if not self.whole_step:
                    self._wrap_linear_attn(i, layer)

    def _wrap_o(self, o) -> None:
        stock = o.forward

        def forward(x, *args, **kwargs):
            if self._skip_o:
                return x, None
            return stock(x, *args, **kwargs)

        o.forward = forward

    def _wrap_linear_attn(self, i, layer) -> None:
        """TP2: on a mono step the GDN mixer is K1 on the stock norm's output,
        with out_proj folded in or then the stock out_proj (its partial left
        to the stock all-reduce)."""
        a = layer.linear_attn
        stock = a.forward

        def forward(hidden_states, forward_batch):
            # A fused AR + norm + quant read hands a tuple; K1 reads bf16 rows.
            if not self._gdn_step or not isinstance(hidden_states, torch.Tensor):
                return stock(hidden_states, forward_batch)
            core, out = self._k1_from_normed(i, layer, hidden_states, forward_batch)
            if out is None:
                out, _ = a.out_proj(core)
            return out

        a.forward = forward

    def _wrap_mlp(self, i, layer) -> None:
        """TP2: on a mono step the FFN is K2 on the stock norm's output, its
        partial left to the stock all-reduce."""
        mlp = layer.mlp
        stock = mlp.forward

        def forward(x):
            if not self._ffn_step or not isinstance(x, torch.Tensor):
                return stock(x)
            return self._ffn_partial(i, layer, x)

        mlp.forward = forward

    def _k1_from_normed(self, i, layer, x, forward_batch):
        from sglang.srt.model_executor.forward_context import get_attn_backend

        lab = get_attn_backend().linear_attn_backend
        idx = lab.forward_metadata.mamba_cache_indices
        a = layer.linear_attn
        st = lab.req_to_token_pool.mamba2_layer_cache(a.layer_id)
        core, _, out = self._k1(i, layer, st.conv[0], st.temporal, idx, x, None)
        # The stock decode's copies to prefix-cache track slots.
        lab._track_mamba_state_decode(
            forward_batch, st.conv[0], st.temporal, idx, a.layer_id
        )
        return core, out

    def _self_check(self) -> str | None:
        why = self._ffn_check() if self.whole_step or self.k2 else None
        if why is None and self.k1:
            why = self._k1_check()
        return why

    def _ffn_check(self) -> str | None:
        """The first layer at widths 1, 8 and K2's cap: out and res_out (TP1), or
        this rank's down partial (TP2)."""
        layer = self.model.layers[0]
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(0)
        for s in self._check_widths(self.k2_rows):
            h = torch.randn(s, 5120, generator=g, device=dev).mul_(0.5).bfloat16()
            r = torch.randn(s, 5120, generator=g, device=dev).mul_(2).bfloat16()
            core = torch.randn(s, 6144, generator=g, device=dev).mul_(0.5).bfloat16()
            if self.oproj:
                h, _ = _o_proj(layer)(core)
            x, res_ref = layer.post_attention_layernorm(h.clone(), r.clone())
            ref = layer.mlp(x)
            self.epoch.add_(1)
            if not self.whole_step:
                out = self._ffn_partial(0, layer, x)
                if not torch.equal(out, ref):
                    d = (out.float() - ref.float()).abs().max().item()
                    return f"K2 partial at width {s}: max abs err {d}"
                continue
            out, res = self._ffn(0, layer, core if self.oproj else h, r)
            if not torch.equal(res, res_ref):
                return f"K2 res_out at width {s}"
            if not torch.equal(out, ref):
                d = (out.float() - ref.float()).abs().max().item()
                return f"K2 out at width {s}: max abs err {d}"
        return None

    def _k1_check(self) -> str | None:
        """The first GDN layer at widths 1, 8 and K1's cap (with and without the
        residual, a pad row) on synthetic conv / SSM states: core, residual
        (TP1), both states of this rank's heads, and with the fold out_proj's
        partial."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn as K

        geo = K.Geo(self.tp)
        i, layer = next((i, l) for i, l in enumerate(self.model.layers) if _is_gdn(l))
        dev = self.epoch.device
        g = torch.Generator(device=dev).manual_seed(1)
        for s in self._check_widths(self.k1_rows):
            first = s == 1
            h = torch.randn(s, K.HIDDEN, generator=g, device=dev).mul_(0.5).bfloat16()
            r = (
                None
                if first
                else torch.randn(s, K.HIDDEN, generator=g, device=dev)
                .mul_(2)
                .bfloat16()
            )
            conv = torch.randn(
                1 + s, geo.conv, K.SL, generator=g, device=dev
            ).bfloat16()
            ssm = torch.randn(1 + s, geo.nv, K.HD, K.HD, generator=g, device=dev).mul_(
                0.05
            )
            idx = torch.arange(1, 1 + s, dtype=torch.int32, device=dev)
            if s > 1:
                idx[1] = -1
            conv_ref, ssm_ref = conv.clone(), ssm.clone()
            core_ref, res_ref = stock_gdn_decode(layer, h, r, conv_ref, ssm_ref, idx)
            self.epoch.add_(1)
            out = out_ref = None
            if self.whole_step:
                core, res, _ = self._k1(i, layer, conv, ssm, idx, h, r)
            else:
                x = (
                    layer.input_layernorm(h)
                    if r is None
                    else layer.input_layernorm(h, r)[0]
                )
                core, _, out = self._k1(i, layer, conv, ssm, idx, x, None)
                res = res_ref
                if out is not None:
                    out_ref, _ = layer.linear_attn.out_proj(core_ref)
            for name, got, ref in (
                ("core", core, core_ref),
                ("res_out", res, res_ref),
                ("conv state", conv, conv_ref),
                ("SSM state", ssm, ssm_ref),
                ("out_proj partial", out, out_ref),
            ):
                if got is not None and not torch.equal(got, ref):
                    return f"K1 {name} at width {s}"
        return None

    def eligible(
        self, input_ids, forward_batch, input_embeds, pp_proxy_tensors, deepstack
    ):
        """A pure decode step no wider than a kernel's cap. The VL wrapper hands
        a text decode step in as ``input_embeds``."""
        if not self.ok or not self._prepared or pp_proxy_tensors is not None:
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
        s = rows.size(0)
        if self.k1 and s <= self.k1_rows:
            # K1 may turn off here; then only K2's cap counts
            return self._k1_eligible(s) and s <= self._max_tokens()
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
            why = _k1_state_unsupported(self.model, lab.req_to_token_pool, self.tp)
            if why is not None:
                logger.warning("Qwen3.8 dense mono K1 off: %s", why)
                self.k1 = False
            self._runtime_ok = why is None
        return self.k1 or self.whole_step or self.k2

    def set_step(self, on: bool, rows: int) -> None:
        """TP2, before the stock layers run a step of ``rows``: whether its GDN
        mixers take K1 and its FFNs K2 (``eligible``, then each kernel's cap)."""
        self._gdn_step = on and self.k1 and rows <= self.k1_rows
        self._ffn_step = on and self.k2 and rows <= self.k2_rows
        if self._gdn_step or self._ffn_step:
            self.epoch.add_(1)

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
                core, res, _ = self._k1(
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
        """-> core, residual out, out_proj's output; at TP2 ``h`` is
        input_layernorm's output and the residual out None, the last None
        unless out_proj is folded in."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_gdn

        a = layer.linear_attn
        s = h.size(0)
        key = self._k1_key(
            dense_gdn,
            s,
            first=residual is None,
            eps=layer.input_layernorm.variance_epsilon,
            norm_eps=a.norm.eps,
        )
        core = torch.empty(
            s, dense_gdn.Geo(self.tp).val, dtype=h.dtype, device=h.device
        )
        front = {"hidden": h.contiguous()}
        res_out = out = None
        if not key.normed:
            res_out = torch.empty_like(h)
            front.update(
                residual=residual, res_out=res_out, ln_w=layer.input_layernorm.weight
            )
        if key.oproj:
            out = torch.empty_like(h)
            front.update(w_o=self.w_o[i], w_os=a.out_proj.weight_scale, out=out)
        w_qkvz, w_ba = self.w_in[i]
        dense_gdn.dense_gdn_pre(
            key,
            **front,
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
        return core, res_out, out

    def _ffn(self, i, layer, h, residual) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        mlp = layer.mlp
        out = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        D.dense_ffn(
            self._k2_key(
                D, h.size(0), eps=layer.post_attention_layernorm.variance_epsilon
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

    def _ffn_partial(self, i, layer, x) -> torch.Tensor:
        """TP2: K2 on post_attention_layernorm's output -> this rank's down
        partial (before the all-reduce)."""
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D

        mlp = layer.mlp
        out = torch.empty_like(x)
        D.dense_ffn(
            self._k2_key(D, x.size(0)),
            hidden=x.contiguous(),
            residual=None,
            ln_w=None,
            w13=self.w13[i],
            w13s=mlp.gate_up_proj.weight_scale,
            w2=self.w2[i],
            w2s=mlp.down_proj.weight_scale,
            out=out,
            res_out=None,
            scratch=self.scratch,
            epoch=self.epoch,
            layer=i,
        )
        return out

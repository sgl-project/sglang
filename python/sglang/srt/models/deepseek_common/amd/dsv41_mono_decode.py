"""SGLang glue for the DeepSeek-V4.1 gfx950 mono decode launches (FFN launch and K1)."""

import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import get_bool_env_var, is_gfx95_supported

logger = logging.getLogger(__name__)

MAX_ROWS = 48
# DeepSeek-V4.1-Flash's shapes, routing and activation, which the kernels build in
_MODEL = dict(
    hidden_size=5120,
    moe_intermediate_size=2304,
    n_routed_experts=384,
    num_experts_per_tok=6,
    n_shared_experts=1,
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    topk_method="noaux_tc",
    hidden_act="silu",
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-20,
)
# n_group / topk_group (8 / 4 in SGLang's DeepseekV41Config) are not read: DS4.1 routes ungrouped
_SHARED = ("gate_proj", "up_proj", "down_proj")

_runner = None


def _get_runner(device: torch.device):
    """Return this TP rank's runner; its peer-memory setup is collective, so every rank builds it at the same layer."""
    global _runner
    p = get_parallel()
    if _runner is None:
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
            MAX_TOKENS,
            DSV41MonoLayer,
            no_capture,
        )

        no_capture("SGLANG_ROCM_MONO_DECODE")
        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        _runner = DSV41MonoLayer(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
        logger.info(
            "DeepSeek-V4.1 mono decode: the FFN launch runs decode and verify steps of <= %d rows",
            MAX_ROWS,
        )
    if (_runner.device, _runner.tp, _runner.rank) != (device, p.tp_size, p.tp_rank):
        raise RuntimeError(
            f"DSv4.1 mono decode: one runner a process, built for {_runner.device} TP{_runner.tp} rank {_runner.rank}"
        )
    return _runner


_normal_path_warm = False


def _warm_normal_path(moe, device: torch.device) -> None:
    """Warm the normal MoE kernels for widths 1 to MAX_ROWS, so no later extend step compiles them."""
    global _normal_path_warm
    if _normal_path_warm:
        return
    _normal_path_warm = True
    t0 = time.perf_counter()
    x = torch.randn(
        MAX_ROWS, _MODEL["hidden_size"], dtype=torch.bfloat16, device=device
    )
    for m in range(1, MAX_ROWS + 1):
        # per-rank, no collective
        moe.forward_normal(x[:m], return_moe_output=True)
    torch.cuda.synchronize(device)
    logger.info(
        "DeepSeek-V4.1 mono decode: normal MoE path warmed for 1-%d tokens in %.1f s",
        MAX_ROWS,
        time.perf_counter() - t0,
    )


def unsupported_model(config) -> Optional[str]:
    """Why the kernels cannot serve this model config, or None."""
    for key, want in _MODEL.items():
        if getattr(config, key, None) != want:
            return f"needs DeepSeek-V4.1-Flash's {key} = {want!r}"
    return None


def _unsupported(config) -> Optional[str]:
    """Why the kernels cannot serve this deployment, or None."""
    # speculative target steps take the launches as DSpark's uniform verify steps only
    algo = (get_spec().speculative_algorithm or "none").upper()
    if algo not in ("NONE", "DSPARK"):
        return "supports speculative decoding with DSpark only"
    if algo == "DSPARK" and envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        return "needs SGLANG_RAGGED_VERIFY_MODE=static with DSpark"
    if not is_gfx95_supported():
        return "needs gfx950"
    p = get_parallel()
    if p.tp_size not in (2, 4):
        return "needs tensor parallel size 2 or 4"
    if p.attn_tp_size != p.tp_size or p.moe_ep_size != 1:
        return "needs tensor parallelism only: no attention data parallelism, no expert parallelism"
    why = unsupported_model(config)
    if why is not None:
        return why
    # the MoE's MXFP4 experts in aiter's gate / up interleaved (16, 16) shuffle (Fp8MoEMethod)
    if (
        not envs.SGLANG_DSV4_FP4_EXPERTS.get()
        or envs.SGLANG_DSV4_FP4_DEQUANT.get()
        or not envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
        or get_bool_env_var("AITER_FORCE_A8W4")
    ):
        return "needs the MXFP4 experts in aiter's gate / up interleaved layout"
    try:
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import BLOCKS
    except ImportError as err:
        return f"needs FlyDSL and AITER's FlyDSL helpers ({err})"
    # every CTA of a launch stays resident: a partitioned GPU deadlocks
    cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if cus < BLOCKS:
        return f"needs {BLOCKS} compute units, the GPU has {cus}"
    return None


def _unsupported_k1(attn) -> Optional[str]:
    """Why K1 cannot serve this deployment's attention front, or None."""
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    if is_unified_kv_triton():
        return "needs the paged SWA cache, not unified_kv"
    if envs.SGLANG_DSV4_KV_LAYOUT.get().lower() != "v4":
        return "needs the V4 KV layout (fp8 nope, bf16 rope, 584 B a token)"
    if attn.q_head_norm or not attn.fuse_wqa_wkv:
        return "needs the fused wqkv_a and no q head norm"
    if attn.head_dim != 512 or attn.q_lora_rank != 1280 or attn.rope_head_dim != 64:
        return "needs DeepSeek-V4.1-Flash's attention shapes"
    return None


def init_layer(layer) -> None:
    """Attach the layer's FFN launch and K1 launch (or None); an opt-in the deployment cannot serve raises."""
    layer.mono_ffn = layer.mono_k1 = None
    if not envs.SGLANG_ROCM_MONO_DECODE.get():
        if envs.SGLANG_ROCM_MONO_K1.get():
            raise ValueError("SGLANG_ROCM_MONO_K1 needs SGLANG_ROCM_MONO_DECODE=1.")
        return
    # draft layers (DSpark stages, next-n layers) keep the normal path
    if layer.mlp.is_nextn:
        return
    why = _unsupported(layer.config)
    if why is None and envs.SGLANG_ROCM_MONO_K1.get():
        why = _unsupported_k1(layer.self_attn)
    if why is not None:
        raise ValueError(f"SGLANG_ROCM_MONO_DECODE {why}.")
    if layer.hc_boundary_fused and not layer.mlp.is_hash:
        layer.mono_ffn = MonoFfn(layer)
        a = layer.self_attn
        if (
            envs.SGLANG_ROCM_MONO_K1.get()
            and a.compress_ratio in (1, 2)
            and a.compressor is None
            and a.indexer is None
        ):
            layer.mono_k1 = MonoK1(layer)


def stash_shared_expert(layers, name: str, loaded_weight: torch.Tensor) -> None:
    """Keep this rank's shard of a checkpoint FP8 shared-expert tensor for the FFN launch."""
    if not envs.SGLANG_ROCM_MONO_DECODE.get() or "mlp.shared_experts" not in name:
        return
    i = get_layer_id(name)
    # skip tensors outside the decoder layers (the checkpoint's mtp.* layers)
    if i is None or i >= len(layers):
        return
    mono = getattr(layers[i], "mono_ffn", None)
    if mono is None:
        return
    proj = next(p for p in _SHARED if f".{p}." in name)
    kind = "weight" if name.endswith(".weight") else "scale"
    # the TP-sharded intermediate dimension
    dim = 1 if proj == "down_proj" else 0
    p = get_parallel()
    t = loaded_weight.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    mono.shared[proj, kind] = (
        t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device()).contiguous()
    )


class MonoFfn:
    """One layer's FFN launch. Its weights are taken from the layer at the first call, after loading."""

    def __init__(self, layer):
        self.layer = layer
        # (projection, "weight" / "scale") -> this rank's checkpoint shard
        self.shared: dict = {}
        self._weights = None

    def weights(self):
        if self._weights is not None:
            return self._weights
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
            MonoLayerWeights,
        )

        layer, moe = self.layer, self.layer.mlp
        # n: the routed experts, before a fused shared one
        e, n = moe.experts, moe.gate.weight.shape[0]
        if not getattr(e.w13_weight, "is_shuffled", False):
            raise RuntimeError(
                "SGLANG_ROCM_MONO_DECODE: MoE weights not in aiter's shuffle"
            )
        if len(self.shared) != 2 * len(_SHARED):
            raise RuntimeError(
                f"SGLANG_ROCM_MONO_DECODE: shared-expert shards {sorted(self.shared)}"
            )
        sh = self.shared

        def u8(t):
            return t.view(torch.uint8)

        self._weights = MonoLayerWeights(
            hc_ffn_fn=layer.hc_ffn_fn,
            hc_ffn_scale=layer.hc_ffn_scale,
            hc_ffn_base=layer.hc_ffn_base,
            ffn_norm=layer.post_attention_layernorm.weight,
            gate_w=moe.gate.weight,
            # SGLang routes with a bf16 copy of the f32 checkpoint bias; the kernels read f32
            bias=moe.gate.e_score_correction_bias.float(),
            w13=e.w13_weight[:n],
            w13_s=u8(e.w13_weight_scale_inv)[: n * e.w13_weight.shape[1]],
            w2=e.w2_weight[:n],
            w2_s=u8(e.w2_weight_scale_inv)[: n * e.w2_weight.shape[1]],
            sgu=torch.cat([sh["gate_proj", "weight"], sh["up_proj", "weight"]]).view(
                torch.float8_e4m3fn
            ),
            sgu_s=torch.cat([sh["gate_proj", "scale"], sh["up_proj", "scale"]]),
            sw2=sh["down_proj", "weight"].view(torch.float8_e4m3fn),
            sw2_s=sh["down_proj", "scale"],
        )
        self.shared = {}
        self._weights.check(get_parallel().tp_size)
        # first called eagerly (capture warms every graph up first), after weight loading
        _warm_normal_path(moe, layer.input_layernorm.weight.device)
        return self._weights

    def __call__(self, part, residual, post, comb, pre):
        """Return (MoE output, residual, post, comb, pre) from wo_b's unreduced output and the seam state."""
        out, residual, post, comb, pre = _get_runner(part.device).ffn(
            self.weights(), part, residual, post, comb, pre
        )
        return out, residual, post.view(post.shape[0], -1), comb, pre


def ffn_launch(layer, forward_batch, rows: int) -> Optional[MonoFfn]:
    """The layer's FFN launch when this step takes it: decode or target verify, at most ``MAX_ROWS`` rows."""
    mono = getattr(layer, "mono_ffn", None)
    mode = forward_batch.forward_mode
    if mono is None or not (mode.is_decode() or mode.is_target_verify()):
        return None
    return mono if 1 <= rows <= MAX_ROWS else None


_k1_kernels: dict = {}
_k1_tables: list = []


def _k1_kernel(rows: int, tp: int):
    """K1 for ``rows`` rows at TP ``tp``."""
    k = _k1_kernels.get((rows, tp))
    if k is None:
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
            MonoBuild,
            build_mono_k1,
        )
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import no_capture

        no_capture(f"K1 width {rows}")
        k = _k1_kernels[rows, tp] = build_mono_k1(MonoBuild(rows, tp))
        logger.info("DeepSeek-V4.1 mono decode: K1 runs steps of %d rows", rows)
    return k


def _cos_sin(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Return K1's rope table [positions, 64] f32 (32 cos, then 32 sin); equal tables share one copy."""
    for f, t in _k1_tables:
        if (
            f.shape == freqs_cis.shape
            and f.device == freqs_cis.device
            and torch.equal(f, freqs_cis)
        ):
            return t
    t = torch.cat([freqs_cis.real, freqs_cis.imag], -1).float().contiguous()
    _k1_tables.append((freqs_cis, t))
    return t


def _e8m0(scale: torch.Tensor) -> torch.Tensor:
    """A linear's f32 32 x 32 block scales as the E8M0 codes K1 reads: a power of two's exponent bits."""
    if scale.dtype != torch.float32:
        raise RuntimeError(f"K1 needs f32 block scales, got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32)
    if not bool(((bits & 0x807FFFFF) == 0).all()):
        raise RuntimeError("K1 needs power-of-two (ue8m0) block scales")
    return (bits >> 23).to(torch.uint8)


class MonoK1:
    """One standard layer's K1 launch: the attention mHC seam and the attention front."""

    def __init__(self, layer):
        self.layer = layer
        self._weights = None
        # (wqkv_a, wq_b) as loaded, from stash_k1_weights
        self.plain = None
        self.cos_sin = _cos_sin(layer.self_attn.freqs_cis)

    def weights(self):
        if self._weights is not None:
            return self._weights
        layer, a = self.layer, self.layer.self_attn
        heads = a.n_local_heads
        if self.plain is None:
            raise RuntimeError("stash_k1_weights did not run after loading")

        def fp8(lin, plain, shape):
            # the FP8 linear method may rewrite the weight in place; K1 reads the row-major copy
            shuffled = getattr(lin, "aiter_bpreshuffled", False) or getattr(
                lin, "mxfp8_native_ready", False
            )
            w = plain if shuffled else lin.weight
            s = _e8m0(lin.weight_scale_inv)
            if not (
                tuple(w.shape) == shape
                and w.dtype == torch.float8_e4m3fn
                and w.is_contiguous()
                and tuple(s.shape) == (shape[0] // 32, shape[1] // 32)
            ):
                raise RuntimeError(
                    f"K1: weight {tuple(w.shape)} {w.dtype}, scales {tuple(s.shape)}, want {shape}"
                )
            return w, s

        wqkv, wqkv_s = fp8(a.wqkv_a, self.plain[0], (1792, 5120))
        wqb, wqb_s = fp8(a.wq_b, self.plain[1], (heads * 512, 1280))
        # free the copies K1 does not read
        self.plain = None
        self._weights = dict(
            hc_fn=layer.hc_attn_fn,
            hc_scale=layer.hc_attn_scale,
            hc_base=layer.hc_attn_base,
            norm=layer.input_layernorm.weight,
            wqkv=wqkv,
            wqkv_s=wqkv_s,
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wqb=wqb,
            wqb_s=wqb_s,
            cos_sin=self.cos_sin,
        )
        w = self._weights
        bad = [name for name, t in w.items() if not (t.is_contiguous() and t.is_cuda)]
        if (
            bad
            or w["hc_fn"].dtype != torch.float32
            or not (w["q_norm"].dtype == w["kv_norm"].dtype == torch.bfloat16)
        ):
            raise RuntimeError(f"K1: unexpected weights {bad or 'dtypes'}")
        cs = w["cos_sin"]
        if not (cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == 64):
            raise RuntimeError(
                f"K1: rope table {tuple(cs.shape)} {cs.dtype}, want [positions, 64] f32"
            )
        return w

    def __call__(self, positions, forward_batch, x, residual, post, comb, pre):
        """Run K1, store the step's KV rows, and return (q, residual, post, comb, pre)."""
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
            HC,
            HIDDEN,
            check_rows,
        )

        M = x.shape[0]
        r = _get_runner(x.device)
        w = self.weights()
        if r.q.shape[1] != self.layer.self_attn.n_local_heads:
            raise RuntimeError(
                f"K1: q buffer {tuple(r.q.shape)}, {self.layer.self_attn.n_local_heads} heads"
            )
        slot = get_attn_backend().get_swa_out_cache_loc(forward_batch)
        # K1 writes through raw pointers: a layout it does not expect would corrupt memory, not fault
        if not (1 <= M <= MAX_ROWS and slot.shape[0] >= M and positions.shape[0] >= M):
            raise RuntimeError(
                f"K1: {M} rows, {tuple(slot.shape)} SWA slots, {tuple(positions.shape)} positions"
            )
        bf, f32 = torch.bfloat16, torch.float32
        check_rows(
            "K1",
            M,
            r.device,
            {
                "x": (x, HIDDEN, bf),
                "residual": (residual, HC * HIDDEN, bf),
                "post": (post, HC, f32),
                "comb": (comb, HC * HC, f32),
                "pre": (pre, HC, f32),
            },
        )
        lid = self.layer.self_attn.layer_id
        meta = getattr(forward_batch, "_mono_k1_meta", None)
        if meta is None or lid <= meta[0]:
            # the first K1 layer of every forward pass copies them, so graph capture records the copy
            meta = forward_batch._mono_k1_meta = (
                lid,
                positions[:M].to(torch.int64).contiguous(),
                slot[:M].to(torch.int64).contiguous(),
            )
        _, pos, slot = meta
        pool = get_token_to_kv_pool()
        if pool.request_window is not None or pool.uniform_fp8:
            raise RuntimeError("K1 needs the paged V4 SWA pool")
        swa = pool.get_swa_raw_buffer(lid)
        block = pool.get_swa_key_page_size()
        if not (
            swa.dtype == torch.uint8
            and swa.dim() == 2
            and swa.stride(0) >= block * 584
            and swa.stride(1) == 1
        ):
            raise RuntimeError(
                f"K1: SWA cache {swa.dtype} {tuple(swa.shape)} {swa.stride()}, page {block} x 584 B"
            )
        _k1_kernel(M, r.tp)(
            residual.data_ptr(),
            x.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            pre.data_ptr(),
            w["hc_fn"].data_ptr(),
            w["hc_scale"].data_ptr(),
            w["hc_base"].data_ptr(),
            w["norm"].data_ptr(),
            r.res_mid.data_ptr(),
            r.post_a.data_ptr(),
            r.comb_a.data_ptr(),
            r.pre_a.data_ptr(),
            w["wqkv"].data_ptr(),
            w["wqkv_s"].data_ptr(),
            w["q_norm"].data_ptr(),
            w["kv_norm"].data_ptr(),
            w["wqb"].data_ptr(),
            w["wqb_s"].data_ptr(),
            w["cos_sin"].data_ptr(),
            pos.data_ptr(),
            slot.data_ptr(),
            swa.data_ptr(),
            swa.stride(0),
            block,
            r.q.data_ptr(),
            r.scratch(M).data_ptr(),
            r.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return r.q[:M], r.res_mid[:M], r.post_a[:M], r.comb_a[:M], r.pre_a[:M]


def stash_k1_weights(layers) -> None:
    """Copy the K1 layers' row-major wqkv_a and wq_b weights before the FP8 linear method rewrites them."""
    for layer in layers:
        k1 = getattr(layer, "mono_k1", None)
        if k1 is not None:
            a = layer.self_attn
            k1.plain = (
                a.wqkv_a.weight.detach().clone(),
                a.wq_b.weight.detach().clone(),
            )


def k1_launch(layer, forward_batch, pending_post, prev_pre) -> Optional[MonoK1]:
    """Return the layer's K1 launch when this step takes it, else None."""
    k1 = getattr(layer, "mono_k1", None)
    if k1 is None or pending_post is None or prev_pre is None:
        return None
    x, residual, post, _ = pending_post
    if x is None or post is None:
        return None
    return (
        k1 if ffn_launch(layer, forward_batch, residual.shape[0]) is not None else None
    )

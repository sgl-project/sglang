"""DeepSeek-V4.1 mono decode on gfx950 (``SGLANG_ROCM_MONO_DECODE=1``): the FFN launch.

On a decode step of at most ``MAX_ROWS`` rows, each layer's attention TP all-reduce, FFN mHC seam, MoE
(router, top-6, shared expert, MXFP4 experts) and MoE TP all-reduce run as one persistent FlyDSL launch
(``dsv41_mono``, from vllm-project/vllm#60397) on the attention's unreduced wo_b output. The launch
returns what the fused mHC boundary returns: the MoE output with its post still pending, the residual
and the FFN seam's mixes. Every TP rank takes the same path at every layer, since the kernels wait on
each other's pushes: the conditions read only what every rank holds alike.
"""

import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.runtime_context import get_parallel
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
    """The process's runner (one per TP rank, shared by every layer). Its peer-memory handshake is
    collective over the TP group: every rank builds it at the same layer of the same step."""
    global _runner
    if _runner is None:
        if torch.cuda.is_current_stream_capturing():
            # its peer memory is allocated and exchanged eagerly; SGLang warms every graph up eagerly
            raise RuntimeError(
                "SGLANG_ROCM_MONO_DECODE was first reached inside a CUDA graph capture"
            )
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
            MAX_TOKENS,
            DSV41MonoLayer,
        )

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        p = get_parallel()
        _runner = DSV41MonoLayer(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
        logger.info(
            "DeepSeek-V4.1 mono decode: the FFN launch runs decode steps of <= %d rows",
            MAX_ROWS,
        )
    return _runner


_normal_path_warm = False


def _warm_normal_path(moe, device: torch.device) -> None:
    """Run the normal MoE path once at every width up to ``MAX_ROWS`` tokens, before serving. Its
    small-token Triton kernels (the MoE sort's, keyed by the token count's power-of-2 bucket) are
    otherwise compiled by decode graph capture, which the launch now takes: a short extend (a chunk
    tail) after serving starts would stall every rank on the compile."""
    global _normal_path_warm
    if _normal_path_warm:
        return
    _normal_path_warm = True
    t0 = time.perf_counter()
    x = torch.randn(
        MAX_ROWS, _MODEL["hidden_size"], dtype=torch.bfloat16, device=device
    )
    for m in range(1, MAX_ROWS + 1):
        moe.forward_normal(x[:m], return_moe_output=True)  # per-rank, no collective
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


def init_layer(layer) -> None:
    """``layer.mono_ffn``: the decoder layer's FFN launch, or None (mono decode off, a hash-routed MoE).
    An opt-in the deployment cannot serve raises."""
    layer.mono_ffn = None
    if not envs.SGLANG_ROCM_MONO_DECODE.get():
        return
    why = _unsupported(layer.config)
    if why is not None:
        raise ValueError(f"SGLANG_ROCM_MONO_DECODE {why}.")
    if layer.hc_boundary_fused and not layer.mlp.is_hash:
        layer.mono_ffn = MonoFfn(layer)


def stash_shared_expert(layers, name: str, loaded_weight: torch.Tensor) -> None:
    """Keep this rank's shard of a checkpoint shared-expert tensor (FP8 e4m3, 32 x 32 E8M0 block
    scales) for the layer's FFN launch: shared-expert fusion requantizes SGLang's own copy to MXFP4."""
    i = get_layer_id(name)
    if i is None or i >= len(
        layers
    ):  # not a decoder layer's (the checkpoint's mtp.* draft layers)
        return
    mono = getattr(layers[i], "mono_ffn", None)
    if mono is None:
        return
    proj = next(p for p in _SHARED if f".{p}." in name)
    kind = "weight" if name.endswith(".weight") else "scale"
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate
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
        self.shared: dict = {}  # (projection, "weight" / "scale") -> this rank's checkpoint shard
        self._weights = None

    def weights(self):
        if self._weights is not None:
            return self._weights
        from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
            MonoLayerWeights,
        )

        layer, moe = self.layer, self.layer.mlp
        e, n = (
            moe.experts,
            moe.gate.weight.shape[0],
        )  # the routed experts, before a fused shared one
        assert getattr(e.w13_weight, "is_shuffled", False), (
            "MoE weights not in aiter's shuffle"
        )
        assert len(self.shared) == 2 * len(_SHARED), sorted(self.shared)
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
        """The layer's (MoE output, residual, FFN post, comb, pre) from wo_b's unreduced output ``part``
        and the attention seam's residual and mixes."""
        out, residual, post, comb, pre = _get_runner(part.device).ffn(
            self.weights(), part, residual, post, comb, pre
        )
        return out, residual, post.view(post.shape[0], -1), comb, pre


def ffn_launch(layer, forward_batch, rows: int) -> Optional[MonoFfn]:
    """The layer's FFN launch when this step takes it: decode only, at most ``MAX_ROWS`` rows."""
    mono = getattr(layer, "mono_ffn", None)
    if mono is None or not forward_batch.forward_mode.is_decode():
        return None
    return mono if 1 <= rows <= MAX_ROWS else None

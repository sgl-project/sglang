import logging
import weakref
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.hardware_backend.npu.utils import NPUACLFormat, npu_format_cast
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.npu_mxfp4_utils import MXFP4_BLOCK_SIZE

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.dtype_utils import (
    _get_float4_e2m1fn_x2_dtype,
    _get_float8_e8m0fnu_dtype,
)
from sglang.srt.hardware_backend.npu.quantization.linear_utils import (
    cache_linear_bias_fp32,
    create_unquantized_linear_weight,
    get_linear_bias_fp32,
    prepare_w8a8_int8_linear_weights,
)

logger = logging.getLogger(__name__)

MXFP8_BLOCK_SIZE = 32


def _empty_quant_matmul_output(
    x: torch.Tensor, layer: torch.nn.Module, dtype: torch.dtype
) -> torch.Tensor:
    # npu_quant_matmul raises "QuantMatmul not support to process empty tensor
    # currently" on 0-row inputs, which attn-DP idle ranks legitimately produce
    # (0-token keep-alive batches). Mirror GEMM semantics ([M, K] x [K, N] ->
    # [M, N]) using the pre-transposed [K, N] NPU weight layout.
    return x.new_empty((*x.shape[:-1], layer.weight.shape[1]), dtype=dtype)


def _prepare_mxfp4_w4a8_bias(
    bias: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """A8W4 matmul requires BF16 bias with shape [1, N]."""
    if bias is None:
        return None
    bias = bias.to(torch.bfloat16)
    return bias.unsqueeze(0) if bias.ndim == 1 else bias


class _NPULinearMethodBase(LinearMethodBase):
    def __init__(
        self,
        quant_config: Optional["QuantizationConfig"] = None,
    ):
        self.quant_config = quant_config


class NPUW8A8Int8LinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        prepare_w8a8_int8_linear_weights(layer, weight_scale_required=False)

        expanding_factor = layer.weight.data.shape[0]
        layer.aclnn_input_scale = torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_scale_reciprocal = 1 / torch.nn.Parameter(
            layer.input_scale.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )
        layer.aclnn_input_offset = torch.nn.Parameter(
            layer.input_offset.data.repeat(expanding_factor).to(device="npu"),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.linear import RowParallelLinear

        original_dtype = x.dtype
        if x.numel() == 0:
            return _empty_quant_matmul_output(x, layer, original_dtype)
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]) if x.ndim > 2 else x
        if original_dtype != torch.int8:
            x_2d = torch.ops.npu.npu_quantize(
                x_2d,
                layer.aclnn_input_scale_reciprocal,
                layer.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in Attention TP>1 case)
        if (
            isinstance(layer, RowParallelLinear)
            and layer.tp_group is not None
            and layer.tp_group.rank_in_group > 0
        ):
            quant_bias = None
        else:
            quant_bias = layer.quant_bias
        out = torch.ops.npu.npu_quant_matmul(
            x_2d,
            layer.weight,
            layer.deq_scale,
            bias=quant_bias,
            output_dtype=original_dtype,
        )
        if x.ndim > 2:
            out = out.reshape(*original_shape[:-1], out.shape[-1])
        return out


class NPUW8A8Int8DynamicLinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer: torch.nn.Module):
        prepare_w8a8_int8_linear_weights(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if isinstance(x, tuple):
            """dynamic_scale is calculated in malprolog kernel"""
            original_dtype = torch.bfloat16
            quant_out, dynamic_scale = x
            if quant_out.numel() == 0:
                return _empty_quant_matmul_output(quant_out, layer, original_dtype)
            return torch.ops.npu.npu_quant_matmul(
                quant_out,
                layer.weight,
                layer.weight_scale,
                pertoken_scale=dynamic_scale.flatten(),
                bias=bias,
                output_dtype=original_dtype,
            )

        # The vision encoder feeds linears a 3-D (seq, 1, hidden) layout that
        # the NPU quant ops only accept flattened to 2-D.  Fold the leading
        # batch dims into m for the quantized GEMM, then restore the layout.
        original_dtype = x.dtype
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]) if x.ndim > 2 else x
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(x_2d)
        if quant_out.numel() == 0:
            return _empty_quant_matmul_output(quant_out, layer, original_dtype)
        out = torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )
        if x.ndim > 2:
            out = out.reshape(*original_shape[:-1], out.shape[-1])
        return out


class NPUMXFP8LinearMethod(_NPULinearMethodBase):
    """NPU MXFP8 linear method for LLM (SRT) models.

    Shared kernel for both the online config path (``--quantization mxfp8``) and
    the offline ModelSlimMXFP8Scheme (which delegates to this as ``self.kernel``).
    process_weights_after_loading branches on weight dtype: FP16/BF16 weights are
    quantised to MXFP8 at load time (online); pre-quantised float8_e4m3fn weights
    are only re-laid-out (offline). Inference: dynamic MXFP8 activation quant +
    MXFP8 matmul (block_size=32).
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        create_unquantized_linear_weight(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            hasattr(layer, "weight_scale_inv")
            and layer.weight.dtype == torch.float8_e4m3fn
        ):
            return
        weight = layer.weight.data
        if weight.dtype == torch.float8_e4m3fn:
            # Offline (ModelSlim) path: weight is already MXFP8-quantised and
            # layer.weight_scale holds one uint8 scale per 32 input elements.
            # Pair the scales for the NPU kernel, padding an odd final count
            # (e.g. K=4304 -> 135 scales -> 136 -> 68 pairs).
            n_dim, k_dim = layer.weight_scale.data.shape
            scale_data = layer.weight_scale.data
            if k_dim % 2 != 0:
                scale_data = F.pad(scale_data, (0, 1), mode="constant", value=0)
                k_dim += 1
            scale = scale_data.reshape(n_dim, k_dim // 2, 2)
            layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                scale.transpose(0, 1), requires_grad=False
            )
            # weight_scale is now folded into weight_scale_inv (which keeps the
            # underlying storage alive via its view); drop the stale parameter so
            # it doesn't linger in named_parameters() / state_dict().
            del layer.weight_scale
        else:
            # Online path: quantise FP16/BF16 weights to MXFP8 at load time.
            if weight.dtype not in (torch.float16, torch.bfloat16):
                logger.warning(
                    "NPUMXFP8LinearMethod: weight dtype %s is not float16/bfloat16; "
                    "casting to bfloat16 before MXFP8 quantisation.",
                    weight.dtype,
                )
                weight = weight.to(torch.bfloat16)
            # Move weight to NPU if needed (cpu offload may move it back to CPU).
            if not weight.is_npu:
                weight = weight.to(f"npu:{torch.npu.current_device()}")
            # Online MXFP8 quantisation of weights (block_size=32).
            # qw: [out, in] float8_e4m3fn, w_scale: [out, in//64, 2] uint8.
            qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
                weight, dst_type=torch.float8_e4m3fn
            )
            layer.weight = Parameter(qw.transpose(0, 1), requires_grad=False)
            layer.weight_scale_inv = Parameter(
                w_scale.transpose(0, 1), requires_grad=False
            )

        # Both paths produce weight [in, out] and weight_scale_inv [in//64, out,
        # 2] as strided transpose views — DO NOT call .contiguous(). The matmul
        # reduction loop scans the in-dim per output column; the [out, in]
        # row-major source gives stride-1 access for that scan via the transpose
        # view, matching msmodelslim's offline layout. Calling .contiguous()
        # physically reorders to [in, out] row-major, making the inner-loop stride
        # equal to out and tanking HBM bandwidth.

        # Cache FP32 bias once to avoid a per-forward dtype conversion + alloc.
        cache_linear_bias_fp32(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | Tuple[torch.Tensor, torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            # Pre-quantized MXFP8 activation from fused norm/swiglu+mx_quant or MLAProlog.
            qx, input_scale = x
            input_shape = qx.shape
            e8m0_dt = _get_float8_e8m0fnu_dtype()
            if input_scale.dtype == torch.uint8 and isinstance(e8m0_dt, torch.dtype):
                input_scale = input_scale.view(e8m0_dt)
            if input_scale.ndim != 3 or not input_scale.is_contiguous():
                input_scale = input_scale.reshape(
                    qx.shape[0], qx.shape[1] // (2 * MXFP8_BLOCK_SIZE), 2
                ).contiguous()
            original_dtype = torch.bfloat16
        else:
            original_dtype = x.dtype
            if original_dtype not in (torch.float16, torch.bfloat16):
                x = x.to(torch.bfloat16)
                original_dtype = torch.bfloat16

            # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant
            input_shape = x.shape
            x_2d = x.reshape(-1, x.shape[-1])

            # Dynamic MXFP8 activation quantisation
            qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
                x_2d, dst_type=torch.float8_e4m3fn
            )

        # MXFP8 matmul (weight & scale already transposed at load time)
        # Use the cached FP32 bias from process_weights_after_loading; fall back
        # to per-call conversion if the cache was bypassed (e.g. dynamic bias).
        quant_bias = get_linear_bias_fp32(layer, bias)

        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale_inv,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features)
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


# Tensor-keyed online MXFP8 weight cache:
# data_ptr -> (weakref(w), w_q_t, w_scale_t, shape, stride, dtype, device, version).
# Sharing by underlying weight tensor guarantees zero duplicate copies across target & draft
# lm_heads (including set_embed_and_head weight sharing), ABA safety, and GC cleanup.
_NPU_ARCH35_MXFP8_WEIGHT_CACHE: dict[
    int,
    Tuple[
        weakref.ReferenceType,
        torch.Tensor,
        torch.Tensor,
        torch.Size,
        Tuple[int, ...],
        torch.dtype,
        torch.device,
        int,
    ],
] = {}


def _is_valid_online_mxfp8_weight(w: Optional[torch.Tensor]) -> bool:
    return (
        w is not None
        and w.is_floating_point()
        and w.ndim == 2
        and w.dtype in (torch.bfloat16, torch.float16)
        and w.shape[1] % (2 * MXFP8_BLOCK_SIZE) == 0
    )


def _get_or_quantize_mxfp8_weight(
    w: torch.Tensor, *, allow_create: bool = True
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    key = w.data_ptr()
    entry = _NPU_ARCH35_MXFP8_WEIGHT_CACHE.get(key)
    if entry is not None:
        w_live = entry[0]()
        if w_live is not None and (
            (w_live is w and entry[7] == w._version)
            or (
                w_live.data_ptr() == key
                and entry[3] == w.shape
                and entry[4] == w.stride()
                and entry[5] == w.dtype
                and entry[6] == w.device
            )
        ):
            return entry[1], entry[2]
    if not allow_create or torch.get_device_module().is_current_stream_capturing():
        return None
    w_src = w.detach()
    w_in = (
        w_src
        if w_src.dtype in (torch.bfloat16, torch.float16)
        else w_src.to(torch.bfloat16)
    )
    qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(w_in, dst_type=torch.float8_e4m3fn)
    # Keep non-contiguous transpose views (stride-1 reduction along K).
    w_q_t, w_scale_t = qw.transpose(0, 1), w_scale.transpose(0, 1)
    w_ref = weakref.ref(w)
    _NPU_ARCH35_MXFP8_WEIGHT_CACHE[key] = (
        w_ref,
        w_q_t,
        w_scale_t,
        w.shape,
        w.stride(),
        w.dtype,
        w.device,
        w._version,
    )

    def _evict(_k=key, _ref=w_ref):
        cur = _NPU_ARCH35_MXFP8_WEIGHT_CACHE.get(_k)
        if cur is not None and cur[0] is _ref:
            _NPU_ARCH35_MXFP8_WEIGHT_CACHE.pop(_k, None)

    weakref.finalize(w, _evict)
    return w_q_t, w_scale_t


def npu_arch35_prepare_online_mxfp8_cache(
    weight_holder: torch.nn.Module,
    *,
    weight: Optional[torch.Tensor] = None,
    cache_attr: str = "_npu_arch35_mxfp8_cache",
) -> bool:
    """Eagerly populate the online MXFP8 weight cache before ACLGraph capture."""
    w = weight if weight is not None else getattr(weight_holder, "weight", None)
    if not _is_valid_online_mxfp8_weight(w):
        return False
    pair = _get_or_quantize_mxfp8_weight(w, allow_create=True)
    if pair is None:
        return False
    setattr(weight_holder, cache_attr, (pair[0], pair[1], w.shape, w.data_ptr()))
    return True


def npu_arch35_online_mxfp8_matmul(
    x: torch.Tensor,
    weight_holder: torch.nn.Module,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cache_attr: str = "_npu_arch35_mxfp8_cache",
) -> torch.Tensor:
    """Online BF16/FP16 -> MXFP8 matmul on Ascend 950 (arch35) without mutating weight."""
    w = weight if weight is not None else weight_holder.weight
    pair = _get_or_quantize_mxfp8_weight(w, allow_create=True)
    if pair is None:
        return F.linear(x.to(w.dtype), w, bias)
    w_q_t, w_scale_t = pair
    orig_shape, orig_dtype = x.shape, x.dtype
    x_2d = x.reshape(-1, orig_shape[-1])
    if x_2d.dtype not in (torch.float16, torch.bfloat16):
        x_2d = x_2d.to(torch.bfloat16)
    qx, x_scale = torch.ops.npu.npu_dynamic_mx_quant(x_2d, dst_type=torch.float8_e4m3fn)
    e8m0_dtype = _get_float8_e8m0fnu_dtype()
    out_dtype = (
        orig_dtype if orig_dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
    )
    out = torch.ops.npu.npu_quant_matmul(
        qx,
        w_q_t,
        w_scale_t,
        scale_dtype=e8m0_dtype,
        pertoken_scale=x_scale,
        pertoken_scale_dtype=e8m0_dtype,
        bias=bias.to(torch.float32) if bias is not None else None,
        output_dtype=out_dtype,
        group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
    )
    return out.reshape(*orig_shape[:-1], out.shape[-1])


def npu_arch35_mxfp8_lm_head_enabled(*, is_draft: bool) -> bool:
    """Return whether Ascend 950 online MXFP8 LM-head is enabled for draft/target."""
    mode = envs.SGLANG_NPU_ARCH35_MXFP8_LM_HEAD.get().strip().lower()
    return mode in ("draft", "all", "1", "true") if is_draft else mode == "all"


def npu_arch35_maybe_project_mxfp8_lm_head(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    is_draft: bool = True,
    force: bool = False,
) -> Optional[torch.Tensor]:
    """Unified Ascend 950 online MXFP8 LM-head hook for LogitsProcessor, DSpark, and DFlash."""
    if hidden_states.shape[0] <= 0:
        return None
    if not force and not npu_arch35_mxfp8_lm_head_enabled(is_draft=is_draft):
        return None
    weight = getattr(lm_head, "weight", None)
    if not _is_valid_online_mxfp8_weight(weight):
        return None
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    if not is_npu_arch35():
        return None
    return npu_arch35_online_mxfp8_matmul(hidden_states, lm_head)


def _find_unquantized_lm_head(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    head = getattr(model, "lm_head", None)
    if head is None:
        lm = getattr(model, "language_model", None)
        head = getattr(lm, "lm_head", None) if lm is not None else None
    if head is None:
        return None
    from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method

    if should_apply_lm_head_quant_method(head, getattr(head, "quant_method", None)):
        return None
    return head


def npu_arch35_post_load_model(model: torch.nn.Module, *, is_draft: bool) -> None:
    """Centralized Ascend 950 post-load hook invoked by NPUSRTPlatform.

    Handles all speculative draft & target models in one place:
      - Marks draft ``logits_processor._is_draft_lm_head = True`` (MTP/EAGLE/Standalone).
      - Pre-warms tensor-keyed online MXFP8 cache on target/draft ``lm_head`` before graph capture.
      - Converts dense BF16 DSpark/DFlash draft backbones (MLP/o_proj/fc) to online MXFP8.
    """
    from sglang.srt.hardware_backend.npu.utils import is_npu_arch35

    if is_draft:
        lp = getattr(model, "logits_processor", None)
        if lp is not None:
            lp._is_draft_lm_head = True

    if not is_npu_arch35():
        return

    if is_draft:
        if npu_arch35_mxfp8_lm_head_enabled(is_draft=True):
            head = _find_unquantized_lm_head(model)
            if head is not None:
                npu_arch35_prepare_online_mxfp8_cache(head)
        if envs.SGLANG_NPU_ARCH35_MXFP8_DSPARK_DRAFT.get():
            from sglang.srt.models.dflash import DFlashDraftModel

            if isinstance(model, DFlashDraftModel):
                npu_arch35_quantize_dspark_draft_model(model)
    else:
        from sglang.srt.runtime_context import get_spec

        try:
            has_spec = bool(get_spec().speculative_algorithm)
        except Exception:
            has_spec = False
        if npu_arch35_mxfp8_lm_head_enabled(is_draft=False) or (
            has_spec and npu_arch35_mxfp8_lm_head_enabled(is_draft=True)
        ):
            head = _find_unquantized_lm_head(model)
            if head is not None:
                npu_arch35_prepare_online_mxfp8_cache(head)


def npu_arch35_quantize_dspark_draft_model(model: torch.nn.Module) -> int:
    """Convert dense BF16 DSpark/DFlash draft MLP, o_proj, and aux fc layers to online MXFP8.

    Leaves ``qkv_proj`` and stacked target-context KV projections in BF16 to preserve
    full attention K/V cache precision (maximizing speculative acceptance length),
    while quantizing MLP (``gate_up_proj``/``down_proj``), ``o_proj``, and ``fc``
    (~80% of draft weight bandwidth) to MXFP8 with fused
    ``npu_add_rms_norm_dynamic_mx_quant`` and ``npu_swiglu_mx_quant``. 100% lossless
    because all draft tokens are verified by the target model. Idempotent if invoked
    more than once.
    """
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    converted = 0
    for layer in getattr(model, "layers", ()):
        candidates = []
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            inner = getattr(layer, "decoder_layer", None)
            mlp = getattr(inner, "mlp", None) if inner is not None else None
        if mlp is not None:
            candidates.extend(
                [getattr(mlp, "gate_up_proj", None), getattr(mlp, "down_proj", None)]
            )
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            candidates.append(getattr(attn, "o_proj", None))
        for mod in candidates:
            if (
                mod is None
                or not isinstance(mod, LinearBase)
                or type(mod.quant_method) is not UnquantizedLinearMethod
                or not hasattr(mod, "weight")
                or mod.weight.dtype not in (torch.bfloat16, torch.float16)
                or mod.weight.ndim != 2
                or mod.weight.shape[1] % (2 * MXFP8_BLOCK_SIZE) != 0
            ):
                continue
            mxfp8_method = NPUMXFP8LinearMethod()
            mod.quant_method = mxfp8_method
            mxfp8_method.process_weights_after_loading(mod)
            converted += 1

    fc = getattr(model, "fc", None)
    if (
        isinstance(fc, torch.nn.Linear)
        and fc.bias is None
        and fc.weight.dtype in (torch.bfloat16, torch.float16)
        and fc.weight.ndim == 2
        and fc.weight.shape[1] % (2 * MXFP8_BLOCK_SIZE) == 0
        and getattr(fc, "_npu_arch35_mxfp8_cache", None) is None
    ):
        if npu_arch35_prepare_online_mxfp8_cache(fc):
            converted += 1

    if converted > 0:
        logger.info(
            "Ascend 950: quantized %d draft linear layers to online MXFP8.",
            converted,
        )
    return converted


def npu_w8a8_mxfp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Block-FP8 linear on Ascend 950, used as the ``w8a8_block_fp8_linear``
    backend on NPU (see ``fp8_utils._dispatch_auto_backend``).

    The loading path requantizes block-FP8 weights into the Ascend 950 MXFP8 layout;
    activations are quantized per call. ``block_size`` is retained for the shared
    block-FP8 backend interface and ``input_scale`` is unused because activation
    scales are always dynamic here.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"npu_w8a8_mxfp8_linear expects float8_e4m3fn weights, got {weight.dtype}"
        )

    original_dtype = input.dtype
    if original_dtype not in (torch.float16, torch.bfloat16):
        input = input.to(torch.bfloat16)
        original_dtype = torch.bfloat16

    orig_shape = input.shape
    input_2d = input.view(-1, orig_shape[-1]).contiguous()

    x_fp8, x_scale = torch.ops.npu.npu_dynamic_mx_quant(
        input_2d, dst_type=torch.float8_e4m3fn
    )
    e8m0_dtype = _get_float8_e8m0fnu_dtype()
    quant_bias = (
        bias.to(torch.float32)
        if bias is not None and bias.dtype != torch.float32
        else bias
    )
    output_2d = torch.ops.npu.npu_quant_matmul(
        x_fp8,
        weight,
        scale=weight_scale,
        scale_dtype=e8m0_dtype,
        pertoken_scale=x_scale,
        pertoken_scale_dtype=e8m0_dtype,
        bias=quant_bias,
        output_dtype=original_dtype,
        group_sizes=(1, 1, MXFP8_BLOCK_SIZE),
    )

    return output_2d.reshape(*orig_shape[:-1], output_2d.shape[-1])


class NPU_W4A4DynamicLinearMethod(_NPULinearMethodBase):
    def process_weights_after_loading(self, layer):
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        layer.weight_offset.data = layer.weight_offset.data.flatten()
        if envs.SGLANG_NPU_W4A4_NEW_PACKING.get():
            layer.weight.data = layer.weight.data.view(torch.int32).contiguous()
        else:
            layer.weight.data = torch.ops.npu.npu_convert_weight_to_int4pack(
                layer.weight.data.to(torch.int32)
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        quant_out, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
            x, dst_type=torch.quint4x2
        )
        return torch.ops.npu.npu_quant_matmul(
            quant_out,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=original_dtype,
        )


class NPUMXFP4W4A8LinearMethod(_NPULinearMethodBase):
    """Quantize complete FP16/BF16 Linear weights to MXFP4 during loading.

    Inference uses the offline packed-weight layout with dynamic MXFP8 inputs.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Allocate lazily and convert as soon as the logical weight is loaded."""
        from sglang.srt.hardware_backend.npu.quantization.online_weight_loader import (
            NPUOnlineDenseWeightLoader,
        )

        loader = NPUOnlineDenseWeightLoader(
            layer,
            params_dtype,
            extra_weight_attrs.get("weight_loader"),
            lambda: self._quantize_weights(layer),
        )
        with torch.device("meta"):
            create_unquantized_linear_weight(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                params_dtype,
                **{**extra_weight_attrs, "weight_loader": loader.weight_loader},
            )
        loader.register_source(layer.weight)
        layer._npu_online_dense_loader = loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer._npu_online_dense_loader.finish_post_load()

    def _quantize_weights(self, layer: torch.nn.Module) -> None:
        # Online single-level MXFP4 weight quant, then lay the weight out exactly
        # like the offline W4A8 path so the same npu_quant_matmul(x2_dtype=fp4)
        # kernel accepts it. All NPU ops go through torch.ops.npu.* (no torch_npu).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        # npu_dynamic_mx_quant returns the scale as [out, in//64, 2] (3D); older
        # builds may return [out, in//32] (2D) — handle both before the transpose.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp,
            dst_type=fp4_dtype,
            round_mode="rint",
            scale_alg=2,
            dst_type_max=7.25,
        )

        # weight: packed FP4 -> FRACTAL_NZ (float8_e4m3fn view) -> transpose
        # [in//2, out]. Mirror the offline path (no .contiguous() on the NZ view);
        # view as uint8 first because npu_format_cast only accepts int-dtype tensors.
        qw_nz = npu_format_cast(
            qw.view(torch.uint8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight = Parameter(qw_nz.transpose(-1, -2), requires_grad=False)

        # weight_scale -> [in//64, out, 2] to match npu_quant_matmul.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8 — FP8, not FP4).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        # True W4(weight)A8(activation) matmul, identical to the offline path.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUMXFP4W4A8OfflineLinearMethod(_NPULinearMethodBase):
    """NPU offline W4A8 (ModelSlim ``W4A8_MXFP``): packed-FP4 weights + MXFP8 activations.

    Kernel for the offline ModelSlimMXFP4W4A8Scheme (delegated as ``self.kernel``).
    The msmodelslim ``W4A8_MXFP`` checkpoint stores weights as *packed FP4*
    (``pack_fp4_to_uint8`` → ``uint8`` shape ``[out, in//2]``) plus UE8M0 block
    scales (``uint8`` shape ``[out, in//group_size]``):

      process_weights_after_loading:
        weight (uint8 packed FP4 [out, in//2]) → npu_format_cast(29,
            customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2) → FRACTAL_NZ
            → transpose [in//2, out]
        weight_scale [out, in/32] → reshape [out, in/64, 2] → transpose → [in/64, out, 2]

      apply:
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float8_e4m3fn)  (A8, MXFP8)
        → npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0, 0, block])

    The weight is cast to FRACTAL_NZ then transposed; ``npu_dynamic_mx_quant`` already
    returns a 3D ``[tokens, in//64, 2]`` block scale so the matmul needs no extra
    scale-layout normalization.

    ⚠️ REQUIRES a recent torch_npu build for the FP4 ``npu_quant_matmul``. On the
    Ascend 950 this device forces ``allow_internal_format=False`` (the NZ cast still produces
    a ``FRACTAL_NZ_C0_16`` tensor, which is fine). Older torch_npu (e.g.
    ``2.10.0.dev20260320``) had a broken FP4 matmul that rejected the NZ weight in
    *prefill* with ``x2 should be in ... nz format, but it is 2``;
    ``2.10.0.post1.dev20260624`` (and later) runs the FRACTAL_NZ path correctly.
    If you hit ``it is 2``, update torch_npu — do NOT "fix" it by
    switching the weight to ND.

    ⚠️ A ``atb::OperationSetup`` *segfault during decode* (not prefill) is a
    DIFFERENT, unrelated issue: it is the eager-decode ``ascend`` attention
    backend, NOT this matmul (verified by stage-sync bisection — qkv's matmul
    syncs clean, the fault surfaces at the entry-sync of the next layer, i.e. the
    decode attention between qkv and o_proj). Run with the NPU decode graph (do
    NOT pass ``--disable-decode-cuda-graph --disable-prefill-cuda-graph``); graph mode is the NPU default. This
    attention issue is model-agnostic and out of scope for W4A8.

    This is a true W4(weight) A8(activation) single-level matmul. The *online*
    ``NPUMXFP4W4A8LinearMethod`` now uses this exact apply path — the only
    difference is that it quantizes BF16/FP16 weights to FP4 at load time instead
    of loading them from a msmodelslim checkpoint. ``group_size`` is fixed at 32
    by the ``W4A8_MXFP`` export format.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Cast the packed-FP4 weight to FRACTAL_NZ then transpose. All NPU ops go
        # through torch.ops.npu.* (no torch_npu). A recent torch_npu build is
        # required (see class docstring): older builds reject the NZ weight
        # ("x2 ... it is 2").
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        # weight: packed-FP4 uint8 [out, in//2] -> FRACTAL_NZ (float8_e4m3fn view)
        # -> transpose to [in//2, out].
        layer.weight.data = npu_format_cast(
            layer.weight.data,
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.float8_e4m3fn,
            input_dtype=fp4_dtype,
        )
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        # weight_scale: [out, in/32] uint8 -> [in/64, out, 2].
        n, k = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(
            n, k // 2, 2
        ).transpose(-3, -2)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation (A8).
        quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch.float8_e4m3fn
        )

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        # W4(weight)A8(activation) matmul.
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=quant_bias,
            output_dtype=original_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[0, 0, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUSingleLevelMXFP4LinearMethod(_NPULinearMethodBase):
    """NPU W4A4 online quantization: single-level MXFP4.

    True W4(weight) A4(activation): both weights and activations are quantised to
    single-level MXFP4 (``float4_e2m1fn_x2``), unlike the W4A8 path which keeps FP8
    activations. All NPU ops go through ``torch.ops.npu.*`` (no top-level
    ``torch_npu``) and the fp4 dtype comes from ``_get_float4_e2m1fn_x2_dtype()``.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)
        → (packed FP4 [out, in//2], UE8M0 block scale) → transpose [in//2, out]

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_mx_quant(dst=float4_e2m1fn_x2)  (A4)
        → npu_quant_matmul(x1_dtype = x2_dtype = float4_e2m1fn_x2,
                           group_sizes=[1, 1, MXFP4_BLOCK_SIZE])

    Triggered by ``--quantization mxfp4`` on NPU. Hardware: Ascend 950 NPU with a recent
    torch_npu exposing ``float4_e2m1fn_x2``.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Register an unquantized (``params_dtype``) weight placeholder.

        The checkpoint still holds full-precision BF16/FP16 weights: the loader
        fills this buffer, then ``process_weights_after_loading`` quantizes it to
        MXFP4 in place.
        """
        create_unquantized_linear_weight(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Online single-level MXFP4 weight quant. All NPU ops go through
        # torch.ops.npu.* (no torch_npu); the fp4 dtype comes from the shared
        # _get_float4_e2m1fn_x2_dtype() helper (the torch_npu int enum).
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # BF16 -> packed FP4 (float4_e2m1fn_x2, [out, in//2]) + UE8M0 block scale.
        qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(
            weight_fp, dst_type=fp4_dtype, round_mode="round"
        )
        # Pre-transpose the weight to [in//2, out] for npu_quant_matmul; use
        # .data= to preserve the non-contiguous transpose view (npu_quant_matmul
        # reads strides directly — .contiguous() would reorder data and break
        # block-scale alignment).
        layer.weight = Parameter(qw, requires_grad=False)
        layer.weight.data = layer.weight.data.transpose(0, 1)

        # weight_scale -> [in//64, out, 2] (3D), matching the offline W4A4 and
        # W4A8 paths. npu_dynamic_mx_quant
        # already returns the scale as [out, in//64, 2] (3D) on current builds;
        # older builds may return [out, in//32] (2D) — reshape those first so the
        # transpose always yields the 3D layout npu_quant_matmul requires.
        if w_scale.dim() == 2:
            n, k = w_scale.shape
            w_scale = w_scale.reshape(n, k // 2, 2)
        layer.weight_scale = Parameter(w_scale.transpose(-3, -2), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e8m0_dtype = _get_float8_e8m0fnu_dtype()
        fp4_dtype = _get_float4_e2m1fn_x2_dtype()

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for npu_dynamic_mx_quant.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic single-level MXFP4 activation quantisation (A4 — FP4).
        qx, input_scale = torch.ops.npu.npu_dynamic_mx_quant(
            x_2d, dst_type=fp4_dtype, round_mode="round"
        )

        # Single-level MXFP4 matmul (weight & scale already transposed at load
        # time): x1_dtype = x2_dtype = fp4, group_sizes=[1, 1, block].
        output = torch.ops.npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=e8m0_dtype,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=e8m0_dtype,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            x1_dtype=fp4_dtype,
            x2_dtype=fp4_dtype,
            group_sizes=[1, 1, MXFP4_BLOCK_SIZE],
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)


class NPUSingleLevelMXFP4OfflineLinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """NPU offline W4A4 (ModelSlim ``W4A4_MXFP4``): packed FP4 weights.

    Kernel for the offline ``ModelSlimMXFP4Scheme`` (delegated as ``self.kernel``).
    The msmodelslim ``W4A4_MXFP4`` checkpoint stores weights as packed ``uint8``
    [out, in//2] (two FP4 values per byte) plus UE8M0 block scales (``uint8``
    [out, in//32]). The weight is transposed and the scale reshaped to 3D; it then
    shares the online :class:`NPUSingleLevelMXFP4LinearMethod` matmul (``apply``)
    exactly — only the weight source differs (msmodelslim checkpoint vs online RTN).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        # The checkpoint is already packed two-FP4-per-byte. Preserve the strided
        # transpose used by the online path.
        layer.weight = Parameter(weight.transpose(0, 1), requires_grad=False)

        weight_scale = layer.weight_scale.data
        if not weight_scale.is_npu:
            weight_scale = weight_scale.to(f"npu:{torch.npu.current_device()}")
        # npu_quant_matmul with float4_e2m1fn_x2 requires x2Scale to be 3D:
        # [out, in/32] -> [out, in/64, 2] -> transpose to [in/64, out, 2].
        n_dim, k_dim = weight_scale.shape
        layer.weight_scale = Parameter(
            weight_scale.reshape(n_dim, k_dim // 2, 2).transpose(0, 1),
            requires_grad=False,
        )


class NPUDualLevelMXFP4LinearMethod(NPUSingleLevelMXFP4LinearMethod):
    """NPU W4A4 online quantization: dual-level MXFP4 (higher accuracy).

    This is the sole online ``--quantization mxfp4`` linear path. Instead of a single
    UE8M0 (power-of-2) block scale, dual-level MX quant produces a finer L0 (FP8 E4M3)
    block scale plus a coarser L1 scale, so per-block dynamic range is captured far
    more accurately — this fixed the online-RTN degradation that made single-level
    decoding loop (never emitting EOS) under greedy sampling. (The single-level
    :class:`NPUSingleLevelMXFP4LinearMethod` is retained only as the offline path's
    base — msmodelslim checkpoints ship single-level UE8M0 scales.)

    All NPU ops go through ``torch.ops.npu.*`` (no top-level ``torch_npu``). Only
    ``create_weights`` (the BF16/FP16 placeholder) is shared with the single-level
    base; weight post-processing and the matmul are fully dual-level.

    Weight quantization (process_weights_after_loading):
        BF16/FP16 weight → npu_dynamic_dual_level_mx_quant
        → (packed FP4 weight, L0 scale, L1 scale); weight cast to FRACTAL_NZ,
          L0 scale transposed to [in//l0_block, out].

    Inference (apply):
        BF16/FP16 activation → npu_dynamic_dual_level_mx_quant  (A4, dual-level)
        → npu_dual_level_quant_matmul(act, weight, act_l0, w_l0, act_l1, w_l1)

    Reference: Diffusion ``NPUMXFP4DiffusionLinearMethod``. Hardware: Ascend 950 NPU
    only — the ``DualLevelQuantBatchMatmul`` op is unavailable on A2/A3.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_fp = layer.weight.data
        if weight_fp.dtype not in (torch.float16, torch.bfloat16):
            weight_fp = weight_fp.to(torch.bfloat16)
        # Move to NPU if needed (cpu offload may have put it on CPU).
        if not weight_fp.is_npu:
            weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

        # Dual-level MXFP4 weight quant: packed FP4 weight + L0 (fine, FP8 E4M3)
        # and L1 (coarse) block scales.
        qw, w_l0_scale, w_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            weight_fp, smooth_scale=None
        )

        # npu_dual_level_quant_matmul requires the weight (x2) in FRACTAL_NZ.
        # View the packed FP4 as int8 first (npu_format_cast takes int dtypes).
        qw_nz = npu_format_cast(
            qw.view(torch.int8),
            NPUACLFormat.ACL_FORMAT_FRACTAL_NZ,
            customize_dtype=torch.int8,
        )

        # L0 scale -> [in//l0_block, out] (op returns [out, in//l0_block, 1]).
        w_l0_scale = w_l0_scale.squeeze(-1).transpose(0, 1).contiguous()

        layer.weight = Parameter(qw_nz, requires_grad=False)
        layer.weight_l0_scale = Parameter(w_l0_scale, requires_grad=False)
        layer.weight_l1_scale = Parameter(w_l1_scale, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # Flatten to 2D [tokens, hidden] for the quant operators.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic dual-level MXFP4 activation quant (A4): packed FP4 + L0/L1 scales.
        qx, act_l0_scale, act_l1_scale = torch.ops.npu.npu_dynamic_dual_level_mx_quant(
            x_2d, smooth_scale=None
        )

        # Dual-level matmul. Arg order (act, weight, act_l0, w_l0, act_l1, w_l1);
        # the weight is NOT transposed here (unlike the single-level path).
        output = torch.ops.npu.npu_dual_level_quant_matmul(
            qx,
            layer.weight,
            act_l0_scale,
            layer.weight_l0_scale,
            act_l1_scale,
            layer.weight_l1_scale,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
        )

        # Restore original shape (replace last dim with output features).
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)

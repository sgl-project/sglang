from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from itertools import pairwise
from math import isfinite
from numbers import Integral, Real
from typing import Any, ClassVar

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def validate_attention_config(config: dict) -> dict:
    """Copy and validate an explicit dense configuration, independent of ServerArgs.

    Layout and softmax scale belong to the attention layer. Configuration values
    are immutable scalars or top-level tensors so callers can cache delegates
    without comparing tensors or nested mutable structures.
    """
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("attention_config must be a dictionary with string keys")
    options = dict(config)
    precision = options.setdefault("precision", "bf16")
    if not isinstance(precision, str) or precision not in (
        "bf16",
        "fp8",
        "mxfp8",
        "mxfp4",
    ):
        raise ValueError("Dense attention precision must be bf16, fp8, mxfp8, or mxfp4")
    common = {"q_rot", "k_rot", "pre_tokens", "next_tokens"}
    mx_options = {
        "mxfp4_scale_alg",
        "mxfp4_dst_type_max",
        "metadata",
        "num_key_value_heads",
        "max_seqlen_q",
        "max_seqlen_kv",
        "mask_mode",
        "win_left",
        "win_right",
        "block_table",
        "sinks",
        "attn_mask",
        "return_softmax_lse",
    }
    allowed = {"precision"}
    if precision == "bf16":
        allowed.add("enable_laser")
    if precision != "bf16":
        allowed |= common
    if precision == "fp8":
        allowed.add("fp8_fa_mode")
    elif precision == "mxfp4":
        allowed |= mx_options
    unknown = options.keys() - allowed
    if unknown:
        raise ValueError(
            f"Unsupported dense attention options for {precision}: {', '.join(sorted(unknown))}"
        )
    if "enable_laser" in options and not isinstance(options["enable_laser"], bool):
        raise ValueError("enable_laser must be a boolean")
    if precision == "fp8":
        mode = options.get("fp8_fa_mode")
        mode = "HIGH_PRECISION" if mode is None else mode
        if not isinstance(mode, str) or mode not in (
            "HIGH_PRECISION",
            "C8V16_TILING512",
        ):
            raise ValueError("fp8_fa_mode must be HIGH_PRECISION or C8V16_TILING512")
        options["fp8_fa_mode"] = mode
    tensor_options = {"q_rot", "k_rot", "metadata", "block_table", "sinks", "attn_mask"}
    for name in tensor_options & options.keys():
        if options[name] is not None and not isinstance(options[name], torch.Tensor):
            raise ValueError(f"{name} must be a tensor or None")
    for name in (common & options.keys()) - tensor_options:
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        options[name] = int(value)
    for name in {
        "mxfp4_scale_alg",
        "num_key_value_heads",
        "max_seqlen_q",
        "max_seqlen_kv",
        "mask_mode",
        "win_left",
        "win_right",
    } & options.keys():
        value = options[name]
        if value is None and name in (
            "mxfp4_scale_alg",
            "num_key_value_heads",
            "win_left",
            "win_right",
        ):
            continue
        minimum = -1 if name.startswith("max_seqlen_") else 0
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        if name == "num_key_value_heads" and value == 0:
            raise ValueError("num_key_value_heads must be positive")
        options[name] = int(value)
    if "mxfp4_dst_type_max" in options and options["mxfp4_dst_type_max"] is not None:
        value = options["mxfp4_dst_type_max"]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            or value <= 0
        ):
            raise ValueError("mxfp4_dst_type_max must be positive and finite")
        options["mxfp4_dst_type_max"] = float(value)
    if "return_softmax_lse" in options:
        value = options["return_softmax_lse"]
        if not isinstance(value, (Integral, bool)) or value != 0:
            raise ValueError("Explicit quantized attention does not return softmax LSE")
        options["return_softmax_lse"] = 0
    return options


def validate_attention_batch_config(
    config: dict, *, batch_size=1, bounds=None, total_tokens=None
) -> None:
    """Reject reuse of batch-specific MXFP4 tensors across samples/documents.

    Call before splitting packed inputs, including when a caller delegates each
    resulting document to ``forward`` separately. Rotation matrices are shared
    head-dimension transforms and do not carry batch/document coordinates.
    """
    if not any(
        config.get(name) is not None
        for name in ("attn_mask", "metadata", "block_table", "sinks")
    ):
        return
    if batch_size != 1:
        raise ValueError("Batch-specific MXFP4 tensors require a single sample")
    if bounds is not None and (
        total_tokens is None or tuple(bounds) != (0, total_tokens)
    ):
        raise ValueError(
            "Batch-specific MXFP4 tensors require one unpadded packed document"
        )


def _supports_laser_shape(
    query_shape, key_shape, value_shape, *, causal, return_softmax_lse
):
    return (
        not causal
        and not return_softmax_lse
        and len(query_shape) == 4
        and query_shape == key_shape == value_shape
        and query_shape[1] >= 2048
        and query_shape[-1] in (32, 64, 96, 128)
    )


def resolve_mx_fa_scheme(quant_config) -> str | None:
    """Resolve the opt-in MXFP8 attention scheme for an NPU quant config."""
    if (
        quant_config is None
        or not current_platform.is_npu()
        or not envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION
    ):
        return None
    if type(quant_config).__name__ not in ("MXFP8Config", "ModelSlimConfig"):
        return None

    if torch.npu.get_soc_version() < 260:
        logger.warning_once(
            "MXFP8 attention is disabled because MXFP8 quantization is only "
            "supported on Ascend 950 (A5) devices."
        )
        return None

    required_ops = (
        "npu_dynamic_mx_quant",
        "npu_fused_infer_attention_score_v2",
    )
    missing_ops = [name for name in required_ops if not hasattr(torch.ops.npu, name)]
    required_dtypes = ("float8_e4m3fn", "float8_e8m0fnu")
    missing_dtypes = [name for name in required_dtypes if not hasattr(torch, name)]
    if missing_ops or missing_dtypes:
        missing_features = missing_ops + missing_dtypes
        logger.warning_once(
            "MXFP8 attention is disabled because the installed torch_npu does not "
            f"provide the required APIs: {', '.join(missing_features)}. "
            "Please install torch==2.10.0, torch_npu>=2.10.0.post4, and CANN>=9.1.1."
        )
        return None
    return "MXFP8"


def _packed_boundaries(
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: Sequence[int] | None,
    total_tokens: int,
    name: str,
) -> tuple[int, ...]:
    if cu_seqlens is None:
        raise ValueError(f"{name} is required for NPU packed attention")
    if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError(f"{name} must be a 1D int32 or int64 tensor")
    if cu_seqlens_host is not None and len(cu_seqlens_host) != cu_seqlens.numel():
        raise ValueError(f"{name} and its host copy must have the same length")

    boundaries = tuple(
        int(value)
        for value in (
            cu_seqlens.tolist() if cu_seqlens_host is None else cu_seqlens_host
        )
    )
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError(f"{name} must start with 0 and contain at least one sequence")
    if boundaries[-1] != total_tokens:
        raise ValueError(
            f"{name} must end at the packed token count {total_tokens}, "
            f"got {boundaries[-1]}"
        )
    if any(stop < start for start, stop in pairwise(boundaries)):
        raise ValueError(f"{name} must be non-decreasing")
    return boundaries


def fused_infer_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    cu_seqlens_q_host: Sequence[int] | None = None,
    cu_seqlens_k_host: Sequence[int] | None = None,
    softmax_scale: float | None = None,
    return_softmax_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    tensors = {"q": q, "k": k, "v": v}
    invalid_layouts = [name for name, tensor in tensors.items() if tensor.ndim != 3]
    if invalid_layouts:
        raise ValueError(
            "NPU packed attention requires q, k, and v in [T, N, D] layout; "
            f"invalid tensors: {', '.join(invalid_layouts)}"
        )
    invalid_devices = [
        name
        for name, tensor in tensors.items()
        if tensor.device.type != "npu" or tensor.device != q.device
    ]
    if invalid_devices:
        raise ValueError(
            "NPU packed attention requires q, k, and v on the same NPU; "
            f"invalid tensors: {', '.join(invalid_devices)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            "NPU packed attention requires q, k, and v with the same dtype"
        )
    if k.shape[:2] != v.shape[:2]:
        raise ValueError(
            "NPU packed attention requires matching K/V token and head counts"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("NPU packed attention requires matching Q/K head dimensions")

    q_boundaries = _packed_boundaries(
        cu_seqlens_q, cu_seqlens_q_host, q.shape[0], "cu_seqlens_q"
    )
    k_boundaries = _packed_boundaries(
        cu_seqlens_k, cu_seqlens_k_host, k.shape[0], "cu_seqlens_k"
    )
    if len(q_boundaries) != len(k_boundaries):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must describe the same batch")

    q_nonempty = [stop > start for start, stop in pairwise(q_boundaries)]
    k_nonempty = [stop > start for start, stop in pairwise(k_boundaries)]
    if q_nonempty != k_nonempty:
        raise NotImplementedError(
            "NPU packed attention does not support a sequence that is empty only "
            "on the query or key/value side"
        )
    actual_seq_lengths = [
        stop for stop, nonempty in zip(q_boundaries[1:], q_nonempty) if nonempty
    ]
    actual_seq_lengths_kv = [
        stop for stop, nonempty in zip(k_boundaries[1:], k_nonempty) if nonempty
    ]
    if not actual_seq_lengths:
        output = torch.empty_like(q)
        if return_softmax_lse:
            lse = torch.empty(
                (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
            )
            return output, lse
        return output

    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        if q.shape == k.shape == v.shape:
            q, k, v = torch.stack((q, k, v), dim=0).unbind(0)
        else:
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    output, lse = torch.ops.npu.npu_fused_infer_attention_score(
        q,
        k,
        v,
        num_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        scale=q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
        input_layout="TND",
        actual_seq_lengths=actual_seq_lengths,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        softmax_lse_flag=return_softmax_lse,
    )
    if not return_softmax_lse:
        return output
    if lse.shape != (q.shape[0], q.shape[1], 1):
        raise RuntimeError(
            "Unexpected Ascend TND softmax LSE shape: "
            f"expected {(q.shape[0], q.shape[1], 1)}, got {tuple(lse.shape)}"
        )
    return output, lse.squeeze(-1).transpose(0, 1).contiguous()


@dataclass
class AscendFAMetadata:
    pass


class AscendFAMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(
        self,
        **kwargs: dict[str, Any],
    ) -> AttentionMetadata:
        return AscendFAMetadata()


class AscendFABackend(AttentionBackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.FA

    @staticmethod
    def get_impl_cls() -> type["AscendFAImpl"]:
        return AscendFAImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        return AscendFAMetadataBuilder

    @classmethod
    def supports_ring_rotation(cls) -> bool:
        """Whether this backend can serve as the ring-attention kernel; the
        per-hop online-softmax merge needs the kernel's softmax LSE."""
        return True


class AscendFAImpl(AttentionImpl):
    """Dense NPU attention with an optional explicit, caller-owned configuration.

    Omitting ``attention_config`` preserves checkpoint-driven native MXFP8.
    An explicit dictionary defaults to native BF16. ``enable_laser=True`` opts
    BF16 into Laser's FP16 intermediates on supported 910 hardware.
    """

    # FA v2 uses per-token-group quantization (mode 6) for Q/K and
    # per-channel-group quantization (mode 8) for V in the packed TND path.
    _MXFP8_LAYOUT = "TND"
    _MXFP8_QK_QUANT_AXIS = -1
    _MXFP8_V_QUANT_AXIS = 0
    _MXFP8_QK_QUANT_MODE = 6
    _MXFP8_V_QUANT_MODE = 8

    # Online Q/K rotations are deterministic CPU FP32 tensors shared
    # by all backend instances and keyed by head size. Applying the same
    # orthogonal matrix R preserves scores:
    # (Q @ R) @ (K @ R).T = Q @ R @ R.T @ K.T = Q @ K.T.
    # Offline checkpoint rotations do not use this generated-matrix cache.
    _rot_matrices: ClassVar[dict[int, torch.Tensor]] = {}

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        attention_config: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.attention_config = (
            None
            if attention_config is None
            else validate_attention_config(attention_config)
        )
        self._quant_attention = None
        self._laser_impl = None
        self._laser_unavailable = False
        self._configured_heads = num_heads
        self._configured_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self._configured_head_size = head_size
        self._packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
        if self.attention_config is not None:
            if (
                isinstance(num_heads, bool)
                or not isinstance(num_heads, Integral)
                or num_heads <= 0
                or isinstance(self._configured_kv_heads, bool)
                or not isinstance(self._configured_kv_heads, Integral)
                or self._configured_kv_heads <= 0
                or num_heads % self._configured_kv_heads
            ):
                raise ValueError(
                    "Explicit Ascend attention requires valid Q/KV head counts"
                )
            if not isinstance(self._packed_trailing_padding, bool):
                raise ValueError("packed_trailing_padding must be a boolean")
            if any(
                extra_impl_args.get(name, 0.0)
                for name in ("dropout_p", "dropout_rate", "dropout")
            ):
                raise NotImplementedError(
                    "Ascend attention does not support attention dropout"
                )
            if self.attention_config["precision"] != "bf16":
                if causal:
                    raise NotImplementedError(
                        "MindIE-SD quantized attention does not support causal attention"
                    )
                if extra_impl_args.get("ring_degree", 1) > 1:
                    raise NotImplementedError(
                        "MindIE-SD quantized attention does not support Ring/LSE"
                    )
                if self.attention_config["precision"] == "fp8" and (
                    head_size not in (64, 128)
                    or (
                        self.attention_config["fp8_fa_mode"] == "C8V16_TILING512"
                        and head_size != 128
                    )
                ):
                    raise ValueError(
                        "FP8 HIGH_PRECISION requires D64/128; C8V16_TILING512 requires D128"
                    )
        # An explicit configuration takes precedence over checkpoint MXFP8.
        # Omitting it leaves the existing native/offline-rotation path intact.
        quant_config = (
            extra_impl_args.get("quant_config") if attention_config is None else None
        )
        self._quant_scheme = (
            resolve_mx_fa_scheme(quant_config) if attention_config is None else None
        )
        self.use_offline_qk_rotation = (
            quant_config.use_offline_qk_rotation
            if hasattr(quant_config, "use_offline_qk_rotation")
            else False
        )
        self._is_cross_attention = bool(
            extra_impl_args.get("is_cross_attention", False)
        )
        if self._quant_scheme is not None:
            self._head_size = head_size
            self._mxfp8_head_chunk_size = envs.SGLANG_DIFFUSION_MXFP8_FA_HEAD_CHUNK_SIZE
            self._rot_device: torch.Tensor | None = None
            if not self.use_offline_qk_rotation:
                self._ensure_rot_matrix(head_size)

    @classmethod
    def _ensure_rot_matrix(cls, head_size: int) -> None:
        if head_size in cls._rot_matrices:
            return
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        rotation, _ = torch.linalg.qr(
            torch.randn(
                head_size,
                head_size,
                generator=generator,
                device="cpu",
                dtype=torch.float32,
            )
        )
        cls._rot_matrices[head_size] = rotation

    def _get_rotation(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if (
            self._rot_device is None
            or self._rot_device.device != device
            or self._rot_device.dtype != dtype
        ):
            self._rot_device = self._rot_matrices[self._head_size].to(
                device=device, dtype=dtype
            )
        return self._rot_device

    def _validate_explicit_qkv(self, query, key, value, *, packed=False):
        ndim = 3 if packed else 4
        tensors = (query, key, value)
        if any(tensor.ndim != ndim for tensor in tensors):
            raise ValueError("Ascend attention requires TND packed or BSND dense Q/K/V")
        if (
            key.shape != value.shape
            or query.shape[0] != key.shape[0]
            or any(tensor.shape[-1] != self._configured_head_size for tensor in tensors)
        ):
            raise ValueError(
                "Ascend attention requires matching batch/head dimensions and K/V shapes"
            )
        heads, kv_heads = query.shape[-2], key.shape[-2]
        if (
            heads <= 0
            or kv_heads <= 0
            or heads % kv_heads
            or self._configured_heads % heads
            or self._configured_kv_heads % kv_heads
        ):
            raise ValueError(
                "Runtime Q/KV heads must form valid partitions of the configured heads"
            )
        if any(
            tensor.device != query.device or tensor.dtype != query.dtype
            for tensor in tensors
        ):
            raise ValueError(
                "Ascend attention requires Q/K/V with the same device and dtype"
            )
        allowed = (
            (torch.bfloat16,)
            if self.attention_config["precision"] == "bf16"
            else (torch.float16, torch.bfloat16, torch.float32)
        )
        if query.dtype not in allowed:
            raise ValueError(
                "Explicit BF16 attention requires BF16; quantized attention requires floating-point Q/K/V"
            )

    def _get_laser_impl(self, query, key, value, return_softmax_lse):
        """Optionally accelerate supported 910 BF16 inputs without a new dependency."""
        if (
            not self.attention_config.get("enable_laser", False)
            or self._laser_unavailable
            or query.device.type != "npu"
            or not _supports_laser_shape(
                query.shape,
                key.shape,
                value.shape,
                causal=self.causal,
                return_softmax_lse=return_softmax_lse,
            )
        ):
            return None
        if not torch.npu.get_device_name(query.device).startswith("Ascend910"):
            return None
        if self._laser_impl is None:
            try:
                if find_spec("attentions") is None:
                    self._laser_unavailable = True
                    return None
                from sglang.multimodal_gen.runtime.layers.attention.backends.laser_attn import (
                    LaserAttentionImpl,
                )
            except (ImportError, OSError, ValueError):
                self._laser_unavailable = True
                return None
            if not hasattr(torch.ops.attentions, "la"):
                self._laser_unavailable = True
                return None
            self._laser_impl = LaserAttentionImpl(
                query.shape[2],
                query.shape[3],
                False,
                self.softmax_scale,
                num_kv_heads=key.shape[2],
            )
        return self._laser_impl

    def _forward_quantized(self, query, key, value, return_softmax_lse):
        if return_softmax_lse:
            raise NotImplementedError(
                "MindIE-SD quantized attention does not support softmax LSE"
            )
        self._validate_explicit_qkv(query, key, value)
        if query.shape[0] == 0 or query.shape[1] == 0:
            return torch.empty_like(query)
        if key.shape[1] == 0:
            raise ValueError("Nonempty queries require nonempty K/V")
        config = self.attention_config
        validate_attention_batch_config(config, batch_size=query.shape[0])
        if config.get("num_key_value_heads") not in (None, key.shape[2]):
            raise ValueError(
                "num_key_value_heads must match the runtime K/V head count"
            )
        if self._quant_attention is None:
            try:
                from mindiesd import quant_attention
            except (ImportError, OSError) as error:
                raise ImportError(
                    "Quantized Ascend FA requires MindIE-SD quant_attention "
                    "with the requested precision installed"
                ) from error
            self._quant_attention = quant_attention
        options = dict(config)
        options.update(layout="BSND", scale=self.softmax_scale)
        options.setdefault("pre_tokens", 2147483647)
        options.setdefault("next_tokens", 2147483647)
        if config["precision"] == "mxfp4":
            options.update(layout_kv="BSND", layout_out="BSND")
        logger.info_once(
            f"Ascend FA: operator=mindiesd.quant_attention; precision={config['precision']}; "
            f"layout=BSND; scale={self.softmax_scale}"
        )
        output = torch.empty_like(query)
        # FP8 block and MX quantization scales must not mix independent samples.
        for index in range(query.shape[0]):
            result = self._quant_attention(
                query[index : index + 1].contiguous(),
                key[index : index + 1].contiguous(),
                value[index : index + 1].contiguous(),
                **options,
            )
            result = result[0] if isinstance(result, tuple) else result
            if (
                not isinstance(result, torch.Tensor)
                or result.shape != query[index : index + 1].shape
                or result.dtype != query.dtype
                or result.device != query.device
            ):
                raise RuntimeError(
                    "MindIE-SD quant_attention must return the input BSND shape, device and dtype"
                )
            output[index : index + 1].copy_(result)
        return output

    def _forward_explicit_varlen(
        self, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host
    ):
        self._validate_explicit_qkv(query, key, value, packed=True)
        if (
            isinstance(max_seqlen, bool)
            or not isinstance(max_seqlen, Integral)
            or max_seqlen < 0
        ):
            raise ValueError("max_seqlen must be a non-negative integer")
        if cu_seqlens_host is not None and (
            not isinstance(cu_seqlens_host, Sequence)
            or any(
                isinstance(item, bool) or not isinstance(item, Integral)
                for item in cu_seqlens_host
            )
        ):
            raise ValueError("cu_seqlens_host must contain integers")
        if not isinstance(cu_seqlens, torch.Tensor):
            raise ValueError("cu_seqlens must be an integer tensor")
        bounds = _packed_boundaries(
            cu_seqlens, cu_seqlens_host, query.shape[0], "cu_seqlens"
        )
        validate_attention_batch_config(
            self.attention_config, bounds=bounds, total_tokens=query.shape[0]
        )
        if self._packed_trailing_padding:
            used = trailing_padding_used_len(query.shape[0], max_seqlen, bounds)
            if used is not None:
                bounds = (0, used)
        if any(stop - start > max_seqlen for start, stop in pairwise(bounds)):
            raise ValueError("max_seqlen is smaller than a packed sequence")
        output = torch.zeros_like(query)
        for start, stop in pairwise(bounds):
            if start == stop:
                continue
            result = self.forward(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
            )
            output[start:stop].copy_(result[0])
        return output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if self.attention_config is not None:
            if self.attention_config["precision"] != "bf16":
                return self._forward_quantized(query, key, value, return_softmax_lse)
            self._validate_explicit_qkv(query, key, value)
            if query.shape[0] == 0 or query.shape[1] == 0:
                output = torch.empty_like(query)
                if return_softmax_lse:
                    return output, torch.empty(
                        (query.shape[0], query.shape[2], query.shape[1]),
                        device=query.device,
                        dtype=torch.float32,
                    )
                return output
            if key.shape[1] == 0 or (self.causal and query.shape[1] != key.shape[1]):
                raise ValueError(
                    "BF16 causal attention requires matching nonempty Q/KV lengths"
                )
            laser = (
                self._get_laser_impl(query, key, value, return_softmax_lse)
                if self.attention_config.get("enable_laser", False)
                else None
            )
            if laser is not None:
                logger.info_once(
                    "Ascend FA: operator=sgl_kernel_npu.laser; precision=bf16 inputs; intermediates=fp16"
                )
                return laser._forward_dense(
                    query, key, value, attn_metadata, preserve_bf16_range=True
                )
            logger.info_once(
                "Ascend FA: operator=torch_npu.npu_fused_infer_attention_score; precision=bf16"
            )
        if (
            self._quant_scheme == "MXFP8"
            and not self.causal
            and not self._is_cross_attention
            and query.shape[1:3] == key.shape[1:3]
            and key.shape == value.shape
            and (query.shape[0] * query.shape[1]) % 64 == 0
        ):
            batch_size, query_length, num_heads, head_size = query.shape
            key_length = key.shape[1]
            actual_seq_qlen = [
                query_length * batch_index for batch_index in range(1, batch_size + 1)
            ]
            actual_seq_kvlen = [
                key_length * batch_index for batch_index in range(1, batch_size + 1)
            ]
            output = self._forward_mxfp8_tnd(
                query.reshape(-1, num_heads, head_size),
                key.reshape(-1, key.shape[2], head_size),
                value.reshape(-1, value.shape[2], head_size),
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                return_softmax_lse=return_softmax_lse,
            )
            return output.reshape(batch_size, query_length, num_heads, head_size)

        mask = None
        num_heads, num_key_value_heads = query.shape[2], key.shape[2]
        if self.causal:
            seq_len = query.shape[1]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device), diagonal=1
            ).bool()[None]
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        output, lse = torch.ops.npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            scale=self.softmax_scale,
            input_layout="BNSD",
            softmax_lse_flag=return_softmax_lse,
            atten_mask=mask,
        )
        output = output.transpose(1, 2)
        if return_softmax_lse:
            return output, lse.squeeze(-1)
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if self.attention_config is not None:
            return self._forward_explicit_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        del max_seqlen
        if (
            self._quant_scheme == "MXFP8"
            and not self.causal
            and not self._is_cross_attention
            and query.shape == key.shape
            and key.shape == value.shape
            and query.shape[0] % 64 == 0
        ):
            boundaries = _packed_boundaries(
                cu_seqlens, cu_seqlens_host, query.shape[0], "cu_seqlens"
            )
            actual_seq_lengths = [
                stop for start, stop in pairwise(boundaries) if stop > start
            ]
            if not actual_seq_lengths:
                return torch.empty_like(query)
            return self._forward_mxfp8_tnd(
                query,
                key,
                value,
                actual_seq_qlen=actual_seq_lengths,
                actual_seq_kvlen=actual_seq_lengths,
            )

        if self.causal:
            bounds = (
                cu_seqlens_host
                if cu_seqlens_host is not None
                else tuple(int(item) for item in cu_seqlens.tolist())
            )
            output = torch.empty_like(query)
            for start, stop in pairwise(bounds):
                if start == stop:
                    continue
                segment = self.forward(
                    query[start:stop].unsqueeze(0),
                    key[start:stop].unsqueeze(0),
                    value[start:stop].unsqueeze(0),
                    None,
                )
                output[start:stop].copy_(segment[0])
            return output

        return fused_infer_attention_varlen(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            cu_seqlens_q_host=cu_seqlens_host,
            cu_seqlens_k_host=cu_seqlens_host,
            softmax_scale=self.softmax_scale,
        )

    def _forward_mxfp8_tnd(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        actual_seq_qlen: Sequence[int],
        actual_seq_kvlen: Sequence[int],
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if return_softmax_lse:
            raise NotImplementedError(
                "MXFP8 attention does not support returning softmax LSE"
            )

        logger.info_once("Using MXFP8 quantized Ascend Flash Attention.")
        if not self.use_offline_qk_rotation:
            rotation = self._get_rotation(query.device, query.dtype)
            query = torch.matmul(query, rotation)
            key = torch.matmul(key, rotation)

        num_heads = query.shape[1]
        num_kv_heads = key.shape[1]
        if num_heads != num_kv_heads:
            raise NotImplementedError("MXFP8 attention currently requires MHA")

        head_chunk_size = self._mxfp8_head_chunk_size
        if head_chunk_size > 0 and num_heads > head_chunk_size:
            num_groups, remainder = divmod(num_heads, head_chunk_size)
            head_groups = [head_chunk_size] * num_groups
            if remainder:
                head_groups.append(remainder)
            outputs = [
                self._run_mxfp8_attention(
                    query_chunk,
                    key_chunk,
                    value_chunk,
                    actual_seq_qlen=actual_seq_qlen,
                    actual_seq_kvlen=actual_seq_kvlen,
                )
                for query_chunk, key_chunk, value_chunk in zip(
                    query.split(head_groups, dim=1),
                    key.split(head_groups, dim=1),
                    value.split(head_groups, dim=1),
                )
            ]
            return torch.cat(outputs, dim=1)

        return self._run_mxfp8_attention(
            query,
            key,
            value,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )

    def _run_mxfp8_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        actual_seq_qlen: Sequence[int],
        actual_seq_kvlen: Sequence[int],
    ) -> torch.Tensor:
        quant_dtype = torch.float8_e4m3fn
        scale_dtype = torch.float8_e8m0fnu
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        query_fp8, query_scale = torch.ops.npu.npu_dynamic_mx_quant(
            query, dst_type=quant_dtype, axis=self._MXFP8_QK_QUANT_AXIS
        )
        key_fp8, key_scale = torch.ops.npu.npu_dynamic_mx_quant(
            key, dst_type=quant_dtype, axis=self._MXFP8_QK_QUANT_AXIS
        )
        value_fp8, value_scale = torch.ops.npu.npu_dynamic_mx_quant(
            value, dst_type=quant_dtype, axis=self._MXFP8_V_QUANT_AXIS
        )
        return torch.ops.npu.npu_fused_infer_attention_score_v2(
            query_fp8,
            key_fp8,
            value_fp8,
            input_layout=self._MXFP8_LAYOUT,
            num_query_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            softmax_scale=self.softmax_scale,
            dequant_scale_query=query_scale,
            dequant_scale_key=key_scale,
            dequant_scale_value=value_scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=0,
            query_quant_mode=self._MXFP8_QK_QUANT_MODE,
            key_quant_mode=self._MXFP8_QK_QUANT_MODE,
            value_quant_mode=self._MXFP8_V_QUANT_MODE,
            query_dtype=quant_dtype,
            key_dtype=quant_dtype,
            value_dtype=quant_dtype,
            dequant_scale_query_dtype=scale_dtype,
            dequant_scale_key_dtype=scale_dtype,
            dequant_scale_value_dtype=scale_dtype,
            out_dtype=query.dtype,
        )[0]

    def forward_ring_kv_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one Ascend TND ring chunk and return LSE in ``[H, Tq]``."""
        if (
            self.attention_config is not None
            and self.attention_config["precision"] != "bf16"
        ):
            raise NotImplementedError(
                "MindIE-SD quantized attention does not support Ring/LSE"
            )
        cu_seqlens_q = torch.tensor(
            [0, query.shape[0]], dtype=torch.int32, device=query.device
        )
        cu_seqlens_k = torch.tensor(
            [0, key.shape[0]], dtype=torch.int32, device=key.device
        )
        return fused_infer_attention_varlen(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_host=(0, query.shape[0]),
            cu_seqlens_k_host=(0, key.shape[0]),
            softmax_scale=self.softmax_scale,
            return_softmax_lse=True,
        )

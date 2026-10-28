from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Integral
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
    """Ascend FA with an optional MindIE-SD quantized attention delegate."""

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
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        config, server_args = self._get_attention_config()
        self._sd_executor = None
        if "precision" in config:
            self._sd_executor = AscendQuantAttention(
                config,
                head_size=head_size,
                softmax_scale=softmax_scale,
                causal=causal,
                ring_degree=getattr(server_args, "ring_degree", 1),
                **extra_impl_args,
            )
            return

        quant_config = extra_impl_args.get("quant_config")
        self._quant_scheme = resolve_mx_fa_scheme(quant_config)
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

    @staticmethod
    def _get_attention_config() -> tuple[dict, Any]:
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            # Direct native callers do not have to initialize ServerArgs.
            return {}, None
        return server_args.attention_backend_config or {}, server_args

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if self._sd_executor is not None:
            return self._sd_executor.forward(
                query, key, value, attn_metadata, return_softmax_lse=return_softmax_lse
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
        if self._sd_executor is not None:
            return self._sd_executor.forward_varlen(
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
        if self._sd_executor is not None:
            raise NotImplementedError("MindIE-SD attention does not support Ring/LSE")
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


class AscendQuantAttention:
    """Internal MindIE-SD FP8/MXFP8/MXFP4 implementation for Ascend FA."""

    _rot_matrices: ClassVar[dict[int, torch.Tensor]] = {}

    def __init__(
        self,
        config: dict,
        *,
        head_size: int,
        causal: bool,
        softmax_scale: float | None,
        ring_degree: int = 1,
        **extra_impl_args,
    ) -> None:
        from sglang.multimodal_gen.runtime.platforms.npu import (
            load_mindiesd_quant_attention,
        )

        if causal:
            raise ValueError("Quantized Ascend FA does not support causal attention")
        if extra_impl_args.get("dropout_p", 0.0):
            raise ValueError("Quantized Ascend FA does not support attention dropout")
        if ring_degree > 1:
            raise ValueError("Quantized Ascend FA does not support Ring/LSE")
        self.precision = config["precision"]
        if not isinstance(self.precision, str) or not self.precision:
            raise ValueError("precision must be a non-empty string")
        # SD owns supported precisions, mode normalization and mode/head-size
        # constraints. Only reject options that this adapter would not forward.
        self.fp8_fa_mode = None
        if self.precision == "fp8":
            self.fp8_fa_mode = config.get("fp8_fa_mode", "HIGH_PRECISION")
        elif "fp8_fa_mode" in config:
            raise ValueError("fp8_fa_mode is only supported with precision=fp8")
        self.head_chunk_size = config.get("head_chunk_size", 4)
        if (
            isinstance(self.head_chunk_size, bool)
            or not isinstance(self.head_chunk_size, Integral)
            or self.head_chunk_size < 0
        ):
            raise ValueError("head_chunk_size must be a non-negative integer")
        offline_rotation = getattr(
            extra_impl_args.get("quant_config"), "use_offline_qk_rotation", False
        )
        self.qk_rotation = config.get("qk_rotation", not offline_rotation)
        if not isinstance(self.qk_rotation, bool):
            raise ValueError("qk_rotation must be a boolean")
        if offline_rotation and self.qk_rotation:
            raise ValueError("qk_rotation=true conflicts with offline Q/K rotation")

        self.head_size = head_size
        self.softmax_scale = head_size**-0.5 if softmax_scale is None else softmax_scale
        self.packed_trailing_padding = bool(
            extra_impl_args.get("packed_trailing_padding", False)
        )
        self._rot_device: torch.Tensor | None = None
        if self.qk_rotation:
            self._ensure_rot_matrix(head_size)
        self._quant_attention = load_mindiesd_quant_attention()
        logger.info_once(
            f"Using Ascend FA: provider=mindiesd, precision={self.precision}, "
            f"fp8_fa_mode={self.fp8_fa_mode}, qk_rotation={self.qk_rotation}, "
            f"head_chunk_size={self.head_chunk_size}"
        )

    @classmethod
    def _ensure_rot_matrix(cls, head_size: int) -> None:
        if head_size not in cls._rot_matrices:
            # Match AscendFAImpl without changing its implementation or global RNG.
            generator = torch.Generator(device="cpu").manual_seed(42)
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

    def _get_rotation(self, query: torch.Tensor) -> torch.Tensor | None:
        if not self.qk_rotation:
            return None
        if (
            self._rot_device is None
            or self._rot_device.device != query.device
            or self._rot_device.dtype != query.dtype
        ):
            self._rot_device = self._rot_matrices[self.head_size].to(
                device=query.device, dtype=query.dtype
            )
        return self._rot_device

    def _validate_adapter_inputs(self, query, key, value, *, ndim: int) -> None:
        # Validate only the adapter's layout/configuration and slicing contract.
        # SD owns dtype, device, GQA and K/V shape validation for each native call.
        layout = "BSND" if ndim == 4 else "TND"
        if any(t.ndim != ndim for t in (query, key, value)):
            raise ValueError(f"MindIE-SD attention requires {layout} Q/K/V")
        if query.shape[-1] != self.head_size:
            raise ValueError("Query head dimension must match configured head_size")
        # Check before slicing: extra K/V batches or tokens would otherwise be
        # discarded, hiding the mismatch from the public SD entry point.
        if any(t.shape[0] != query.shape[0] for t in (key, value)):
            raise ValueError("Q/K/V batch sizes or packed token counts must match")

    def _forward_dense(self, query, key, value) -> torch.Tensor:
        # Match native Ascend FA's MHA head chunking. Preserve SDK handling of
        # GQA and invalid K/V head counts instead of slicing away mismatches.
        if (
            0 < self.head_chunk_size < query.shape[2]
            and query.shape[2] == key.shape[2] == value.shape[2]
        ):
            return torch.cat(
                [
                    self._run_attention(q, k, v)
                    for q, k, v in zip(
                        query.split(self.head_chunk_size, dim=2),
                        key.split(self.head_chunk_size, dim=2),
                        value.split(self.head_chunk_size, dim=2),
                    )
                ],
                dim=2,
            )
        return self._run_attention(query, key, value)

    def _run_attention(self, query, key, value) -> torch.Tensor:
        rotation = self._get_rotation(query)
        options = {}
        if self.precision == "fp8":
            options["fp8_fa_mode"] = self.fp8_fa_mode
        # SD applies rotation exactly once, before quantization. Preserve its
        # output dtype, including the current MXFP4 kernel's BF16 output.
        return self._quant_attention(
            query,
            key,
            value,
            precision=self.precision,
            layout="BSND",
            scale=self.softmax_scale,
            q_rot=rotation,
            k_rot=rotation,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            **options,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        *,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        if return_softmax_lse:
            raise NotImplementedError("MindIE-SD attention does not return softmax LSE")
        self._validate_adapter_inputs(query, key, value, ndim=4)
        # FP8 requires B=1. MXFP8 groups V over flattened B*S, so isolate
        # batches unless their sequence boundaries are aligned to 64 tokens.
        split_batch = self.precision == "fp8" or (
            self.precision == "mxfp8" and key.shape[1] % 64 != 0
        )
        # Send zero-batch inputs to SD validation instead of torch.cat([]).
        if not split_batch or query.shape[0] <= 1:
            return self._forward_dense(query, key, value)
        return torch.cat(
            [
                self._forward_dense(query[i : i + 1], key[i : i + 1], value[i : i + 1])
                for i in range(query.shape[0])
            ],
            dim=0,
        )

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
        self._validate_adapter_inputs(query, key, value, ndim=3)
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a 1D int32 or int64 tensor")
        bounds = (
            tuple(cu_seqlens.tolist()) if cu_seqlens_host is None else cu_seqlens_host
        )
        if (
            len(bounds) < 2
            or len(bounds) != cu_seqlens.numel()
            or any(isinstance(x, bool) or not isinstance(x, Integral) for x in bounds)
            or bounds[0] != 0
            or bounds[-1] != query.shape[0]
            or any(start > stop for start, stop in zip(bounds[:-1], bounds[1:]))
        ):
            raise ValueError("cu_seqlens must monotonically cover all packed tokens")
        if (
            isinstance(max_seqlen, bool)
            or not isinstance(max_seqlen, Integral)
            or max_seqlen < 0
        ):
            raise ValueError("max_seqlen must be a non-negative integer")
        if self.packed_trailing_padding:
            used = trailing_padding_used_len(query.shape[0], max_seqlen, bounds)
            if used is not None:
                bounds = (0, used)
        if any(
            stop - start > max_seqlen for start, stop in zip(bounds[:-1], bounds[1:])
        ):
            raise ValueError("Packed sequence length exceeds max_seqlen")

        output = None
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            segment = self._forward_dense(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
            )
            if output is None:
                output = segment.new_zeros(query.shape)
            output[start:stop].copy_(segment[0])
        return torch.zeros_like(query) if output is None else output

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_utils import (
    mxfp4_quant_kwargs,
    resolve_precision,
    uses_mxfp4_attention,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _can_use_mxfp8_attention(quant_config) -> bool:
    """Check the opt-in MXFP8 attention requirements."""
    if (
        quant_config is None
        or not current_platform.is_npu()
        or not envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION
    ):
        return False
    if type(quant_config).__name__ not in ("MXFP8Config", "ModelSlimConfig"):
        return False

    if torch.npu.get_soc_version() < 260:
        logger.warning_once(
            "MXFP8 attention is disabled because MXFP8 quantization is only "
            "supported on Ascend 950 (A5) devices."
        )
        return False

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
        return False
    return True


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


def _attention_quant_kwargs(description: dict, quant_op) -> dict:
    """Use C7 defaults and rint rounding when the operator supports it."""
    kwargs = mxfp4_quant_kwargs(description)
    try:
        has_round_mode = "round_mode" in inspect.signature(quant_op).parameters
    except (TypeError, ValueError):
        has_round_mode = True
    if has_round_mode:
        kwargs["round_mode"] = "rint"
    else:
        logger.warning_once(
            "torch_npu.npu_dynamic_mx_quant exposes no round_mode; MXFP4 "
            "attention falls back to operator-default rounding"
        )
    return kwargs


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
        quant_config = extra_impl_args.get("quant_config")
        self.use_mxfp4_attention = uses_mxfp4_attention(quant_config, prefix)
        self._attention_description = (
            getattr(quant_config, "quant_description", {}) or {}
        )
        self._use_mxfp8_attention = (
            not self.use_mxfp4_attention and _can_use_mxfp8_attention(quant_config)
        )
        attention_prefix = prefix.removesuffix(".impl")
        self.use_offline_qk_rotation = all(
            any(
                self._attention_description.get(f"{attention_prefix}.{name}") == "FLOAT"
                for name in names
            )
            for names in (("to_q_rot", "q_rot"), ("to_k_rot", "k_rot"))
        ) or (
            not attention_prefix
            and getattr(quant_config, "use_offline_qk_rotation", False)
        )
        self._is_cross_attention = bool(
            extra_impl_args.get("is_cross_attention", False)
        )
        if self.use_mxfp4_attention:
            if self._is_cross_attention:
                raise NotImplementedError(
                    "Quantized MXFP4 attention is supported only for self-attention"
                )
            self._init_mxfp4_attention(head_size, causal)
        if self._use_mxfp8_attention:
            self._head_size = head_size
            self._mxfp8_head_chunk_size = envs.SGLANG_DIFFUSION_MXFP8_FA_HEAD_CHUNK_SIZE
            self._rot_device: torch.Tensor | None = None
            if not self.use_offline_qk_rotation:
                self._ensure_rot_matrix(head_size)

    # MXFP4 attention adapted from:
    # https://gitcode.com/Ascend/MindIE-SD/blob/dev/mindiesd/quantization/layer.py
    def _init_mxfp4_attention(self, head_size: int, causal: bool) -> None:
        if causal or head_size != 128 or get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "MXFP4 attention requires unmasked D=128 attention without "
                "ring parallelism"
            )
        if torch.npu.get_soc_version() < 260:
            raise RuntimeError("MXFP4 attention requires Ascend 950")
        if not hasattr(torch, "float8_e8m0fnu"):
            raise RuntimeError(
                "MXFP4 attention requires PyTorch with float8_e8m0fnu support"
            )
        self.pad_base = self._attention_description.get("mxfp4_attention_pad_size", 512)
        if self.pad_base not in (256, 512):
            raise ValueError("mxfp4_attention_pad_size must be 256 or 512")
        import torch_npu

        if not hasattr(torch_npu, "npu_dynamic_mx_quant") or not hasattr(
            torch_npu, "float4_e2m1fn_x2"
        ):
            raise RuntimeError(
                "MXFP4 attention requires torch_npu dynamic MXFP4 quantization"
            )
        try:
            from cann_ops_transformer.ops import (
                quant_flash_attn,
                quant_flash_attn_metadata,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MXFP4 attention requires the CANN experimental quant_flash_attn "
                "and quant_flash_attn_metadata extension"
            ) from exc
        self.quant_flash_attn = quant_flash_attn
        self.metadata_op = quant_flash_attn_metadata
        self.metadata_parameters = inspect.signature(self.metadata_op).parameters
        if not {"quant_mode", "layout_q_descale"}.issubset(self.metadata_parameters):
            raise RuntimeError(
                "Unsupported CANN quant_flash_attn_metadata ABI; install the "
                "experimental MXFP4 extension"
            )
        self._quant_kwargs = _attention_quant_kwargs(
            self._attention_description, torch_npu.npu_dynamic_mx_quant
        )

    def _forward_mxfp4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        """Dense attention over BSND inputs; every batch row has length S."""
        if return_softmax_lse:
            raise NotImplementedError("MXFP4 attention does not support softmax LSE")
        if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
            raise ValueError("MXFP4 attention requires matching BSND Q/K/V shapes")
        self._validate_mxfp4(query, key, value)
        batch, length = query.shape[0], query.shape[1]
        q, k, v = (t.transpose(1, 2).contiguous() for t in (query, key, value))
        q, k, v = self._pad_mxfp4(q, k, v)
        seqused = torch.full((batch,), length, dtype=torch.int32, device=q.device)
        output = self._run_mxfp4(q, k, v, scale, seqused)
        return output[:, :, :length].transpose(1, 2).to(query.dtype)

    def _forward_mxfp4_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        boundaries: tuple[int, ...],
    ) -> torch.Tensor:
        """Bucket packed [T, N, D] sequences by padded length; seqused excludes padding."""
        if query.ndim != 3 or query.shape != key.shape or key.shape != value.shape:
            raise ValueError(
                "Packed MXFP4 attention requires matching [T, N, D] Q/K/V shapes"
            )
        self._validate_mxfp4(query, key, value)
        output = torch.empty_like(query)
        buckets: dict[int, list[int]] = {}
        for index, (start, stop) in enumerate(pairwise(boundaries)):
            if stop > start:
                buckets.setdefault(self._mxfp4_padded_length(stop - start), []).append(
                    index
                )
        for padded_length, indices in buckets.items():
            segments = [boundaries[index : index + 2] for index in indices]
            q, k, v = (
                torch.stack(
                    [
                        F.pad(
                            tensor[start:stop].transpose(0, 1),
                            (0, 0, 0, padded_length - stop + start),
                        )
                        for start, stop in segments
                    ]
                )
                for tensor in (query, key, value)
            )
            seqused = torch.tensor(
                [stop - start for start, stop in segments],
                dtype=torch.int32,
                device=q.device,
            )
            bucket_output = self._run_mxfp4(q, k, v, scale, seqused)
            for position, index in enumerate(indices):
                start, stop = boundaries[index], boundaries[index + 1]
                output[start:stop] = bucket_output[
                    position, :, : stop - start
                ].transpose(0, 1)
        return output

    def _validate_mxfp4(self, query, key, value) -> None:
        if query.device.type != "npu" or not (
            query.device == key.device == value.device
        ):
            raise ValueError("MXFP4 attention Q/K/V must be on the same NPU")
        if query.dtype not in (torch.bfloat16, torch.float16) or not (
            query.dtype == key.dtype == value.dtype
        ):
            raise ValueError("MXFP4 attention Q/K/V must have matching FP16/BF16 dtype")
        if query.shape[-1] != 128 or query.shape[-2] == 0 or query.shape[0] == 0:
            raise ValueError("MXFP4 attention requires nonempty Q/K/V with D=128")

    def _mxfp4_padded_length(self, length: int) -> int:
        return -(-length // self.pad_base) * self.pad_base

    def _pad_mxfp4(self, q, k, v):
        padding = (-q.shape[2]) % self.pad_base
        if padding:
            q, k, v = (F.pad(t, (0, 0, 0, padding)) for t in (q, k, v))
        return q, k, v

    def _run_mxfp4(self, q, k, v, scale, seqused) -> torch.Tensor:
        """Quantize padded BNSD Q/K/V and invoke the quantized-attention kernel."""
        import torch_npu

        batch, heads, length, dim = q.shape
        quantized, descales = [], []
        for tensor, axis in ((q, -1), (k, -1), (v, 2)):
            packed, descale = torch_npu.npu_dynamic_mx_quant(
                tensor.contiguous(),
                dst_type=torch_npu.float4_e2m1fn_x2,
                axis=axis,
                **self._quant_kwargs,
            )
            quantized.append(packed.view(torch.uint8).contiguous())
            descales.append(descale)
        q_scale, k_scale, v_scale = descales
        q_scale = q_scale.reshape(batch, heads, length, dim // 64, 2)
        k_scale = k_scale.reshape(batch, heads, length, dim // 64, 2)
        if v_scale.ndim == 4:
            v_scale = v_scale.reshape(batch, heads, length // 64, 2, dim).transpose(
                -1, -2
            )
        descales = [
            s.contiguous().view(torch.float8_e8m0fnu)
            for s in (q_scale, k_scale, v_scale)
        ]
        expected = (batch, heads, length, dim // 2)
        if any(t.shape != expected for t in quantized) or descales[2].shape != (
            batch,
            heads,
            length // 64,
            dim,
            2,
        ):
            raise RuntimeError(
                "Unsupported torch_npu MXFP4 attention packing/scale layout"
            )
        common = {
            "quant_mode": 5,
            "seqused_q": seqused,
            "seqused_kv": seqused,
            "max_seqlen_q": length,
            "max_seqlen_kv": length,
            "mask_mode": 0,
            "win_left": -1,
            "win_right": -1,
            "layout_q": "BNSD",
            "layout_q_descale": "BNSD",
            "layout_kv": "BNSD",
            "layout_out": "BNSD",
        }
        metadata_options = {
            "num_heads_q": heads,
            "num_heads_kv": heads,
            "head_dim": dim,
            "batch_size": batch,
        }
        if "v_descale" in self.metadata_parameters:
            metadata_options["v_descale"] = descales[2]
        elif "head_dim_v" in self.metadata_parameters:
            metadata_options["head_dim_v"] = dim
        metadata = self.metadata_op(**metadata_options, **common)
        return self.quant_flash_attn(
            *quantized,
            *descales,
            metadata=metadata,
            softmax_scale=scale,
            return_softmax_lse=False,
            **common,
        )[0]

    def should_apply_qk_rot(self, query, key, value) -> bool:
        if not self.use_offline_qk_rotation:
            return False
        if self.use_mxfp4_attention:
            return (
                resolve_precision(self._attention_description, "fa", "MXFP4") == "MXFP4"
            )
        return (
            self._use_mxfp8_attention
            and not self.causal
            and not self._is_cross_attention
            and query.shape[1:3] == key.shape[1:3]
            and key.shape == value.shape
            and (query.shape[0] * query.shape[1]) % 64 == 0
        )

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
        if (
            self.use_mxfp4_attention
            and resolve_precision(self._attention_description, "fa", "MXFP4") == "MXFP4"
        ):
            return self._forward_mxfp4(
                query, key, value, self.softmax_scale, return_softmax_lse
            )
        if (
            self._use_mxfp8_attention
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
        del max_seqlen
        if (
            self.use_mxfp4_attention
            and resolve_precision(self._attention_description, "fa", "MXFP4") == "MXFP4"
        ):
            boundaries = _packed_boundaries(
                cu_seqlens, cu_seqlens_host, query.shape[0], "cu_seqlens"
            )
            return self._forward_mxfp4_varlen(
                query, key, value, self.softmax_scale, boundaries
            )
        if (
            self._use_mxfp8_attention
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

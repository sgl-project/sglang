import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_npu = current_platform.is_npu()
if _is_npu:
    import torch_npu


def resolve_mx_fa_scheme(quant_config) -> str | None:
    """Resolve the opt-in MXFP8 attention scheme for an NPU quant config."""
    if (
        quant_config is None
        or not _is_npu
        or not envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION
    ):
        return None
    if type(quant_config).__name__ not in ("MXFP8Config", "ModelSlimConfig"):
        return None
    required_apis = (
        "npu_dynamic_mx_quant",
        "npu_fused_infer_attention_score_v2",
    )
    missing_apis = [name for name in required_apis if not hasattr(torch_npu, name)]
    if missing_apis:
        logger.warning_once(
            "MXFP8 attention is disabled because torch_npu lacks: "
            + ", ".join(missing_apis)
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
    if any(stop < start for start, stop in zip(boundaries[:-1], boundaries[1:])):
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

    q_nonempty = [
        stop > start for start, stop in zip(q_boundaries[:-1], q_boundaries[1:])
    ]
    k_nonempty = [
        stop > start for start, stop in zip(k_boundaries[:-1], k_boundaries[1:])
    ]
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

    _MXFP8_FA_PARAMS = {
        "q_quant_mode": 6,
        "k_quant_mode": 6,
        "v_quant_mode": 8,
        "qk_quant_axis": -1,
        "v_quant_axis": 0,
        "layout": "TND",
    }
    _rot_matrices: dict[int, torch.Tensor] = {}
    _use_sub_head = int(os.getenv("USE_SUB_HEAD", "5"))

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
            self._ensure_rot_matrix(head_size)
            self._rot_device: torch.Tensor | None = None

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

    def _get_rotation(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
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
            self._quant_scheme == "MXFP8"
            and not self.causal
            and not self._is_cross_attention
            and query.shape[1:3] == key.shape[1:3]
            and key.shape == value.shape
            and (query.shape[0] * query.shape[1]) % 64 == 0
        ):
            batch_size, query_length, num_heads, head_size = query.shape
            key_length = key.shape[1]
            actual_seq_qlen = torch.arange(
                query_length,
                query_length * (batch_size + 1),
                query_length,
                dtype=torch.int64,
                device=query.device,
            )
            actual_seq_kvlen = torch.arange(
                key_length,
                key_length * (batch_size + 1),
                key_length,
                dtype=torch.int64,
                device=key.device,
            )
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
                stop
                for start, stop in zip(boundaries[:-1], boundaries[1:])
                if stop > start
            ]
            if not actual_seq_lengths:
                return torch.empty_like(query)
            actual_seq_lengths_tensor = torch.tensor(
                actual_seq_lengths,
                dtype=torch.int64,
                device=query.device,
            )
            return self._forward_mxfp8_tnd(
                query,
                key,
                value,
                actual_seq_qlen=actual_seq_lengths_tensor,
                actual_seq_kvlen=actual_seq_lengths_tensor,
            )

        if self.causal:
            bounds = (
                cu_seqlens_host
                if cu_seqlens_host is not None
                else tuple(int(item) for item in cu_seqlens.tolist())
            )
            output = torch.empty_like(query)
            for start, stop in zip(bounds[:-1], bounds[1:]):
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
        actual_seq_qlen: torch.Tensor,
        actual_seq_kvlen: torch.Tensor,
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

        sub_heads = self._use_sub_head
        if sub_heads > 0 and num_heads > sub_heads:
            head_groups = [sub_heads] * (num_heads // sub_heads)
            if remainder := num_heads % sub_heads:
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
        actual_seq_qlen: torch.Tensor,
        actual_seq_kvlen: torch.Tensor,
    ) -> torch.Tensor:
        params = self._MXFP8_FA_PARAMS
        quant_dtype = torch.float8_e4m3fn
        scale_dtype = torch_npu.float8_e8m0fnu
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        query_fp8, query_scale = torch_npu.npu_dynamic_mx_quant(
            query, dst_type=quant_dtype, axis=params["qk_quant_axis"]
        )
        key_fp8, key_scale = torch_npu.npu_dynamic_mx_quant(
            key, dst_type=quant_dtype, axis=params["qk_quant_axis"]
        )
        value_fp8, value_scale = torch_npu.npu_dynamic_mx_quant(
            value, dst_type=quant_dtype, axis=params["v_quant_axis"]
        )
        return torch_npu.npu_fused_infer_attention_score_v2(
            query_fp8,
            key_fp8,
            value_fp8,
            input_layout=params["layout"],
            num_query_heads=query.shape[1],
            num_key_value_heads=key.shape[1],
            softmax_scale=self.softmax_scale,
            dequant_scale_query=query_scale,
            dequant_scale_key=key_scale,
            dequant_scale_value=value_scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            sparse_mode=0,
            query_quant_mode=params["q_quant_mode"],
            key_quant_mode=params["k_quant_mode"],
            value_quant_mode=params["v_quant_mode"],
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

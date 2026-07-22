import math
import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_is_npu = current_platform.is_npu()
if _is_npu:
    import torch_npu


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


class AscendFAImpl(AttentionImpl):
    """Flash Attention implementation for Ascend NPU.

    Supports two paths:
    - Standard FA (default): BNSD layout, torch.ops.npu.npu_fused_infer_attention_score
    - MX quantized FA: online rotation + dynamic MX quantization,
      activated when a supported ``quant_config`` is passed via ``extra_impl_args``.

    Supported quant configs (checked by class name):
    - ``MXFP8Config`` → MXFP8 (fp8_e4m3fn, quant_mode 6/6/8, TND layout)
    - ``MXFP4Config`` (future) → MXFP4 (fp4_e2m1fn, quant_mode 3/3/3)

    To add a new MX quant scheme, add an entry to :attr:`_MX_QUANT_PARAMS`
    and a corresponding ``_forward_mx_*`` method if the FA call differs
    significantly from the default TND path.
    """

    # ------------------------------------------------------------------
    # Per-scheme static quantization parameters.
    # Key = QuantConfig class name.  Each value is a dict of kwargs
    # forwarded to the MX quant + FA v2 calls in :meth:`_forward_mx_quant`.
    # ------------------------------------------------------------------
    _MX_QUANT_PARAMS: dict[str, dict] = {
        "MXFP8Config": {
            "quant_dtype": None,  # resolved at forward time → torch.float8_e4m3fn
            "scale_dtype": None,  # resolved at forward time → torch_npu.float8_e8m0fnu
            "q_quant_mode": 6,
            "k_quant_mode": 6,
            "v_quant_mode": 8,
            "qk_quant_axis": -1,  # per-token along head_dim
            "v_quant_axis": 0,  # per-channel along token dim (TND layout)
            "layout": "TND",
        },
        # "MXFP4Config": { ... }  # add when MXFP4 FA support lands
    }

    # Cache of orthogonal rotation matrices keyed by head_dim.
    # Q and K share the same matrix so that attention scores are preserved:
    #   (Q @ R) @ (K @ R)^T = Q @ R @ R^T @ K^T = Q @ K^T
    # Generated once with fixed seed (42) for deterministic results.
    _rot_matrices: dict[int, torch.Tensor] = {}

    # Sub-head splitting: when > 0, split Q/K/V into chunks of this many
    # heads along the heads dimension and call FA v2 per-chunk.  This
    # improves NPU kernel utilisation when the total head count (e.g. 40)
    # is too large for an efficient single dispatch.
    # Set via env ``USE_SUB_HEAD`` (matching the Wan2.2 reference).
    _USE_SUB_HEAD: int = int(os.getenv("USE_SUB_HEAD", "5"))

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
        self._quant_scheme = self._resolve_quant_scheme(quant_config)

        if self._quant_scheme is not None:
            self._head_dim = head_size
            self._ensure_rot_matrix(head_size)
            # Device-cached rotation matrix (lazily moved from CPU on first forward)
            self._rot_device: torch.Tensor | None = None

    @staticmethod
    def _resolve_quant_scheme(quant_config) -> str | None:
        """Return the quant scheme name if *quant_config* is a supported MX FA config.

        Returns ``None`` when *quant_config* is ``None``, not recognised,
        or running on a non-NPU platform.
        """
        if quant_config is None or not _is_npu:
            return None
        name = type(quant_config).__name__
        return name if name in AscendFAImpl._MX_QUANT_PARAMS else None

    # ------------------------------------------------------------------
    # Rotation matrix helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_rot_matrix(cls, head_dim: int) -> None:
        """Generate and cache an orthogonal rotation matrix for *head_dim*.

        The matrix is stored on CPU in fp32 and moved to the target device/dtype
        on each forward pass.  Seed 42 ensures deterministic results across runs.
        """
        if head_dim not in cls._rot_matrices:
            gen = torch.Generator()
            gen.manual_seed(42)
            rot, _ = torch.linalg.qr(
                torch.randn(head_dim, head_dim, generator=gen, device="cpu").to(dtype=torch.float32)
            )
            cls._rot_matrices[head_dim] = rot  # CPU fp32

    def _get_rot(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(q_rot, k_rot)`` on *device* with *dtype*.

        The rotation matrix is copied from CPU to device on first call and
        cached for subsequent forwards (avoids per-step CPU→NPU transfer).
        Q and K share the same rotation matrix.
        """
        if self._rot_device is None or self._rot_device.device != device:
            rot = self._rot_matrices[self._head_dim]
            self._rot_device = rot.to(device=device, dtype=dtype)
        return self._rot_device, self._rot_device

    # ------------------------------------------------------------------
    # Forward dispatcher
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        # MX quantization only applies to self-attention (Q and K/V share the
        # same sequence length and head count).  Cross-attention and GQA fall
        # back to the standard FA path.
        if self._quant_scheme is not None:
            if query.shape[1] == key.shape[1] and query.shape[2] == key.shape[2]:
                return self._forward_mx_quant(
                    query, key, value, attn_metadata,
                    self._quant_scheme, return_softmax_lse,
                )
        return self._forward_standard(
            query, key, value, attn_metadata, return_softmax_lse
        )

    def _forward_standard(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        mask = None
        num_heads, num_key_value_heads = query.shape[2], key.shape[2]
        if self.causal:
            seq_len = query.shape[1]
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device), diagonal=1
            ).bool()
        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
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
            return output, lse
        return output

    def _forward_mx_quant(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        scheme: str,
        return_softmax_lse: bool = False,
    ) -> torch.Tensor:
        """Generic MX-quantized attention path.

        Looks up quantization parameters from :attr:`_MX_QUANT_PARAMS` for
        *scheme*, then:

        1. Rotate Q/K with orthogonal matrices (smooths per-block distribution)
        2. Reshape BSND → TND (merge batch + seq dims)
        3. (optional) Split along heads dim when ``USE_SUB_HEAD > 0``
        4. Dynamic MX quantize Q, K, V
        5. Call ``npu_fused_infer_attention_score_v2`` (per-chunk when splitting)
        6. Reshape TND → BSND

        Input / output: ``[B, S, N, D]`` (BSND, sglang convention).

        To support a quant scheme that needs a different FA API or layout,
        add a dedicated ``_forward_mx_{scheme}`` method and dispatch to it
        from here.
        """
        if return_softmax_lse:
            raise NotImplementedError(
                f"return_softmax_lse is not supported for {scheme} attention"
            )

        params = self._MX_QUANT_PARAMS[scheme]
        B, S_q, N, D = query.shape
        S_kv = key.shape[1]
        N_kv = key.shape[2]
        device, dtype = query.device, query.dtype

        # Resolve dtype references that can't be stored as class-level constants
        quant_dtype = params.get("quant_dtype") or torch.float8_e4m3fn
        scale_dtype = params.get("scale_dtype")
        if scale_dtype is None and _is_npu:
            scale_dtype = torch_npu.float8_e8m0fnu

        qk_axis = params["qk_quant_axis"]
        v_axis = params["v_quant_axis"]
        layout = params["layout"]

        # 1. Rotate Q/K — done once on the full tensor regardless of sub-head
        #    splitting, since rotation is O(B·S·N·D²) independent of the FA
        #    kernel's head-count sensitivity.
        q_rot, k_rot = self._get_rot(device, dtype)
        query = torch.matmul(query, q_rot)
        key = torch.matmul(key, k_rot)

        # 2. Reshape BSND → TND (Q and K/V may have different seq lens)
        query = query.reshape(B * S_q, N, D)
        key = key.reshape(B * S_kv, N_kv, D)
        value = value.reshape(B * S_kv, N_kv, D)

        # 3. Build cu_seqlens — shared across sub-head chunks (seq dim unaffected)
        actual_seq_qlen = torch.arange(
            S_q, S_q * (B + 1), S_q, dtype=torch.int64, device=device
        )
        actual_seq_kvlen = torch.arange(
            S_kv, S_kv * (B + 1), S_kv, dtype=torch.int64, device=device
        )

        # 4. Sub-head splitting (optional): when N is large (e.g. 40) the NPU
        #    FA kernel can be inefficient.  Splitting into smaller head groups
        #    improves kernel utilisation while rotation + cu_seqlens are shared.
        #    When N is not evenly divisible by USE_SUB_HEAD, the last chunk
        #    processes the remainder heads (e.g. N=36, USE_SUB_HEAD=5 → 7×5 + 1).
        sub_heads = self._USE_SUB_HEAD
        if sub_heads > 0 and N > sub_heads:
            # Build head-group sizes: N full groups of sub_heads, + optional remainder
            n_full = N // sub_heads
            head_groups = [sub_heads] * n_full
            if N % sub_heads != 0:
                head_groups.append(N % sub_heads)

            # For K/V the heads may differ from Q (GQA/MQA); compute independently.
            n_kv_full = N_kv // sub_heads
            kv_head_groups = [sub_heads] * n_kv_full
            if N_kv % sub_heads != 0:
                kv_head_groups.append(N_kv % sub_heads)

            # Split along the heads dimension (dim=1 in TND)
            q_chunks = query.split(head_groups, dim=1)
            k_chunks = key.split(kv_head_groups, dim=1)
            v_chunks = value.split(kv_head_groups, dim=1)

            outputs = []
            for i, (q_c, k_c, v_c) in enumerate(
                zip(q_chunks, k_chunks, v_chunks)
            ):
                n_heads_chunk = head_groups[i]
                n_kv_heads_chunk = kv_head_groups[i]

                # MX quantize per chunk
                q_fp8, q_scale = torch_npu.npu_dynamic_mx_quant(
                    q_c, dst_type=quant_dtype, axis=qk_axis
                )
                k_fp8, k_scale = torch_npu.npu_dynamic_mx_quant(
                    k_c, dst_type=quant_dtype, axis=qk_axis
                )
                v_fp8, v_scale = torch_npu.npu_dynamic_mx_quant(
                    v_c, dst_type=quant_dtype, axis=v_axis
                )

                # FA v2 per chunk
                output_chunk = torch_npu.npu_fused_infer_attention_score_v2(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    input_layout=layout,
                    num_query_heads=n_heads_chunk,
                    num_key_value_heads=n_kv_heads_chunk,
                    softmax_scale=1.0 / math.sqrt(D),
                    dequant_scale_query=q_scale,
                    dequant_scale_key=k_scale,
                    dequant_scale_value=v_scale,
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
                    out_dtype=dtype,
                )[0]
                outputs.append(output_chunk)

            # Concat along heads dim → [B*S_q, N, D]
            output = torch.cat(outputs, dim=1)
        else:
            # 4. MX dynamic quantization (full heads)
            q_fp8, q_scale = torch_npu.npu_dynamic_mx_quant(
                query, dst_type=quant_dtype, axis=qk_axis
            )
            k_fp8, k_scale = torch_npu.npu_dynamic_mx_quant(
                key, dst_type=quant_dtype, axis=qk_axis
            )
            v_fp8, v_scale = torch_npu.npu_dynamic_mx_quant(
                value, dst_type=quant_dtype, axis=v_axis
            )

            # 5. Flash Attention v2 (full heads)
            output = torch_npu.npu_fused_infer_attention_score_v2(
                q_fp8,
                k_fp8,
                v_fp8,
                input_layout=layout,
                num_query_heads=N,
                num_key_value_heads=N_kv,
                softmax_scale=1.0 / math.sqrt(D),
                dequant_scale_query=q_scale,
                dequant_scale_key=k_scale,
                dequant_scale_value=v_scale,
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
                out_dtype=dtype,
            )[0]

        # 5/6. Reshape output: [B*S_q, N, D] → [B, S_q, N, D]
        output = output.reshape(B, S_q, N, D)

        return output

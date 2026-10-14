"""Descriptor-selected Wan attention using ModelSlim offline calibration."""

import inspect

import torch
import torch.nn.functional as F
import torch_npu

from sglang.multimodal_gen.runtime.distributed import get_ring_parallel_world_size
from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp_utils import (
    mxfp4_quant_kwargs,
    resolve_precision,
)


class ModelSlimAttention:
    def __init__(self, scheme, description, head_size, causal):
        if causal or head_size != 128 or get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "ModelSlim Wan attention requires unmasked D=128 attention without ring parallelism"
            )
        if torch.npu.get_soc_version() < 260:
            raise RuntimeError("ModelSlim FP8/MXFP4 attention requires Ascend 950")
        if not hasattr(torch, "float8_e8m0fnu"):
            raise RuntimeError(
                "ModelSlim attention requires PyTorch with float8_e8m0fnu support"
            )
        self.scheme = scheme
        self.description = description
        self.pad_base = description.get("mxfp4_attention_pad_size", 512)
        if self.pad_base not in (256, 512):
            raise ValueError("mxfp4_attention_pad_size must be 256 or 512")
        self.quant_flash_attn = self.metadata_op = None
        policy = description.get("timestep_policy", {}).get("fa", {})
        precisions = {scheme, *policy.values()}
        if "FP8" in precisions:
            for name in (
                "npu_dynamic_block_quant",
                "npu_fused_infer_attention_score_v2",
            ):
                if not hasattr(torch_npu, name):
                    raise RuntimeError(
                        f"ModelSlim FP8 attention requires torch_npu.{name}"
                    )
        if "MXFP4" in precisions:
            if not hasattr(torch_npu, "npu_dynamic_mx_quant") or not hasattr(
                torch_npu, "float4_e2m1fn_x2"
            ):
                raise RuntimeError(
                    "ModelSlim MXFP4 attention requires torch_npu dynamic MXFP4 quantization"
                )
            try:
                from cann_ops_transformer.ops import (
                    quant_flash_attn,
                    quant_flash_attn_metadata,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "ModelSlim MXFP4 attention requires the CANN experimental "
                    "quant_flash_attn and quant_flash_attn_metadata extension"
                ) from exc
            self.quant_flash_attn = quant_flash_attn
            self.metadata_op = quant_flash_attn_metadata
            self.metadata_parameters = inspect.signature(self.metadata_op).parameters
            if not {"quant_mode", "layout_q_descale"}.issubset(
                self.metadata_parameters
            ):
                raise RuntimeError(
                    "Unsupported CANN quant_flash_attn_metadata ABI; install the experimental MXFP4 extension"
                )

    def forward(self, query, key, value, scale, return_softmax_lse=False):
        if return_softmax_lse:
            raise NotImplementedError(
                "ModelSlim Wan attention does not support softmax LSE"
            )
        if query.ndim != 4 or query.shape != key.shape or key.shape != value.shape:
            raise ValueError(
                "ModelSlim Wan attention requires matching BSND Q/K/V shapes"
            )
        if query.shape[-1] != 128 or min(query.shape[:3]) == 0:
            raise ValueError(
                "ModelSlim Wan attention requires nonempty Q/K/V with D=128"
            )
        if query.device.type != "npu" or not (
            query.device == key.device == value.device
        ):
            raise ValueError("ModelSlim Wan Q/K/V must be on the same NPU")
        if query.dtype not in (torch.bfloat16, torch.float16) or not (
            query.dtype == key.dtype == value.dtype
        ):
            raise ValueError("ModelSlim Wan Q/K/V must have matching FP16/BF16 dtype")
        precision = resolve_precision(self.description, "fa", self.scheme)
        length = query.shape[1]
        q, k, v = (t.transpose(1, 2).contiguous() for t in (query, key, value))
        if precision == "FLOAT":
            output = torch_npu.npu_fused_infer_attention_score_v2(
                q,
                k,
                v,
                input_layout="BNSD",
                num_query_heads=q.shape[1],
                softmax_scale=scale,
                pre_tokens=2147483647,
                next_tokens=2147483647,
                out_dtype=query.dtype,
            )[0]
        elif precision == "FP8":
            output = self._forward_fp8(q, k, v, scale, query.dtype)
        else:
            output = self._forward_mxfp4(q, k, v, scale)
        return output[:, :, :length].transpose(1, 2).to(query.dtype)

    def _forward_fp8(self, q, k, v, scale, dtype):
        quantized, descales = [], []
        for tensor, block_size in ((q, 128), (k, 256), (v, 256)):
            # MindIE's block quantizer takes one [N,S,D] sample at a time.
            samples = [
                torch_npu.npu_dynamic_block_quant(
                    sample.contiguous(),
                    dst_type=torch_npu.float8_e4m3fn,
                    row_block_size=block_size,
                    col_block_size=128,
                )
                for sample in tensor
            ]
            quantized.append(torch.stack([sample[0] for sample in samples]))
            descales.append(torch.stack([sample[1] for sample in samples]))
        return torch_npu.npu_fused_infer_attention_score_v2(
            *quantized,
            input_layout="BNSD",
            num_query_heads=q.shape[1],
            softmax_scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            query_quant_mode=7,
            key_quant_mode=7,
            value_quant_mode=7,
            dequant_scale_query=descales[0],
            dequant_scale_key=descales[1],
            dequant_scale_value=descales[2],
            out_dtype=dtype,
        )[0]

    def _forward_mxfp4(self, q, k, v, scale):
        # The pinned reference includes zero-padded KV positions in softmax.
        q, k, v = (
            F.pad(t, (0, 0, 0, (-t.shape[2]) % self.pad_base)) for t in (q, k, v)
        )
        batch, heads, length, dim = q.shape
        options = mxfp4_quant_kwargs(self.description)
        quantized, descales = [], []
        for tensor, axis in ((q, -1), (k, -1), (v, 2)):
            packed, descale = torch_npu.npu_dynamic_mx_quant(
                tensor.contiguous(),
                dst_type=torch_npu.float4_e2m1fn_x2,
                axis=axis,
                **options,
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
        seqused = torch.full((batch,), length, dtype=torch.int32, device=q.device)
        common = dict(
            quant_mode=5,
            seqused_q=seqused,
            seqused_kv=seqused,
            max_seqlen_q=length,
            max_seqlen_kv=length,
            mask_mode=0,
            win_left=-1,
            win_right=-1,
            layout_q="BNSD",
            layout_q_descale="BNSD",
            layout_kv="BNSD",
            layout_out="BNSD",
        )
        metadata_options = dict(
            num_heads_q=heads, num_heads_kv=heads, head_dim=dim, batch_size=batch
        )
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

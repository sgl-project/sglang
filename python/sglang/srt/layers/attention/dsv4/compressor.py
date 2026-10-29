from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn as nn

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.deepseek_v2 import _is_hip, _use_aiter
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import add_prefix, is_npu, set_weight_attrs

_is_npu = is_npu()

if _use_aiter:
    from aiter.tuned_gemm import tgemm

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.quantization.base_config import QuantizationConfig
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class Compressor(BaseFusedOp):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        set_weight_attrs(self.ape, {"weight_loader": self.load_ape_weight})
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        if (
            is_in_indexer
            and _is_hip
            and get_exec().kernel.enable_deepseek_v4_fp4_indexer
        ):
            self._init_fp4_norm_weight()
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def _init_fp4_norm_weight(self) -> None:
        """Mirror the FP32 norm weight in BF16 for the AITER FP4 K writer.

        The FP8 path feeds the FP32 weight straight to its kernel; AITER wants
        BF16, and converting at the call site costs one copy per C4 layer per
        forward. A buffer keeps the conversion out of the forward and survives
        module ``_apply``, and the loader below re-derives it so online weight
        updates propagate -- they land as ``param.data.copy_``, which leaves the
        parameter's identity and ``_version`` untouched and would silently
        defeat any cache keyed on those. Same reach as ``load_ape_weight``:
        ``update_weights_from_tensor(load_format="direct")`` calls
        ``default_weight_loader`` itself and so skips both hooks.
        """
        self.norm.register_buffer(
            "fp4_weight_bf16",
            self.norm.weight.detach().to(torch.bfloat16).contiguous(),
            persistent=False,
        )
        set_weight_attrs(self.norm.weight, {"weight_loader": self.load_norm_weight})

    def load_norm_weight(
        self, param: torch.Tensor, loaded_weight: torch.Tensor
    ) -> None:
        assert param is self.norm.weight
        param.data.copy_(loaded_weight)
        self.norm.fp4_weight_bf16.copy_(param.data)

    def _apply_ape_hotfix(self):
        self.ape_converted = True

        if _is_npu:
            return

        if self.overlap:
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[0], ape[1]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self._apply_ape_hotfix()

    def load_ape_weight(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        assert param is self.ape
        assert loaded_weight.shape == param.shape
        param.data.copy_(loaded_weight)
        self._apply_ape_hotfix()

    def get_state_pool(self, attn_backend: AttentionBackend) -> CompressStatePool:
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)
        assert isinstance(ret, CompressStatePool)
        return ret

    def compute_kv_score(self, x: torch.Tensor, forward_batch: ForwardBatch):
        kv_score = self._compute_wkv_gate(x)

        # CUDA path: delegate to backend
        if dsa_use_prefill_cp(forward_batch):
            kv_score = cp_materialize_global_token_order(
                kv_score,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return kv_score

    def _compute_wkv_gate(self, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(self.wkv_gate, "weight", None)
        if weight is not None:
            if _use_aiter and weight.dtype == torch.bfloat16:
                # aiter's tuned GEMM for these shapes; kv_score is bf16-rounded
                return tgemm.mm(x, weight, otype=x.dtype).float()
            return linear_bf16_fp32(x, weight)

        from sglang.srt.layers.quantization.gguf import fused_mul_mat_gguf

        return fused_mul_mat_gguf(
            x,
            self.wkv_gate.qweight,
            self.wkv_gate.qweight_type.weight_type,
        )

    def forward_native(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: Optional[AttentionBackend] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        kv_score = self.compute_kv_score(x, forward_batch)

        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)
        kv_score_buffer = self.get_state_pool(attn_backend).kv_score_buffer.kv_score
        return attn_backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def forward_npu(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: Optional[AttentionBackend] = None,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        if dsa_use_prefill_cp(forward_batch):
            x = cp_materialize_global_token_order(
                x,
                forward_batch,
                torch.cuda.current_stream(),
            )

        return get_attn_backend().forward_compress(self, x, forward_batch)

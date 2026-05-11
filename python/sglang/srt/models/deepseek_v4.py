from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import TYPE_CHECKING, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import fused_rope, rmsnorm_self
from sglang.srt.configs.deepseek_v4 import (
    DeepSeekV4Config,
    set_fp4_experts,
)
from sglang.srt.debug_utils.deepseek_v4_debug_utils import (
    deepseek_v4_moe_code_path_checker,
)
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsv4.compressor import (
    Compressor,
    rms_normalize_triton,
)
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    can_nsa_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_round_robin_split,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.communicator import LayerScatterModes, get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import (
    apply_rotary_emb_triton,
)
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    dp_gather_partial,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    get_is_capture_mode,
)
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu

if not _is_hip:
    from sglang.srt.layers.utils.cp_utils import (
        prepare_context_parallel_metadata,
    )

from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    log_info_on_rank0,
    make_layers,
    maybe_torch_compile,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config

logger = logging.getLogger(__name__)

from sglang.srt.environ import envs

MOE_BIT_WISE_EQUAL_MODE = False
ATTN_BIT_WISE_EQUAL_MODE = False
COMPRESSOR_BIT_WISE_EQUAL_MODE = False
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend_hip import DeepseekV4Backend
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = (
                get_attention_tp_size() if _is_hip else get_attention_cp_size()
            )
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        else:
            self.qk_nope_head_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio  # type: ignore

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert self.head_dim == config.head_dim
        else:
            assert self.head_dim == config.v_head_dim
        assert config.num_key_value_heads == 1

        # need a indexer for compress ratio = 4
        rope_scaling = config.rope_scaling
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # Please keep this assertion and not remove it
        # NOTE:
        # 1. 2601
        #    The `260119-updated` code changed compress_rope_theta
        # 2. 2604
        #    `official_code_0409/code/config.json` is 160000
        #    while `official_code_0409/config.json` is 40000
        #    maybe the latter is buggy? b/c dpsk's official generate.py uses `code/config.json`
        expected_compress_rope_theta = os.environ.get(
            "SGLANG_HACK_ASSERT_COMPRESS_ROPE_THETA"
        )
        if expected_compress_rope_theta is None:
            expected_compress_rope_theta = "160000"
        expected_compress_rope_theta = int(expected_compress_rope_theta)
        assert (
            config.compress_rope_theta == expected_compress_rope_theta
        ), f"{config.compress_rope_theta=} {expected_compress_rope_theta=}"

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        # naive impl: copy from reference code
        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert rope_scaling["factor"] == 16
        elif envs.SGLANG_DSV4_MODE.get() == "2601":
            assert rope_scaling["factor"] == 4
        else:
            raise NotImplementedError

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert self.compress_ratio in {0, 4, 128}
            if self.compress_ratio:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
                assert original_seq_len == 65536
            else:
                original_seq_len = 0
        else:
            original_seq_len = rope_scaling["original_max_position_embeddings"]

        rope_scaling = config.rope_scaling
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]  # use first 3 streams for mqa layer
            self.alt_streams_indexer = alt_streams[
                -2:
            ]  # use last 2 streams for indexer
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
                rotary_emb=getattr(self, "rotary_emb", None),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                    rotary_emb=getattr(self, "rotary_emb", None),
                )

        # Note: attention sink should be replicated
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))

        self.fuse_wqa_wkv = not _is_hip and envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        if not self.fuse_wqa_wkv:
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            # fp8_einsum handles scale transform internally — skip UE8M0 conversion
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.overlap_store_cache = envs.SGLANG_OPT_USE_OVERLAP_STORE_CACHE.get()
        self.use_jit_norm = not _is_hip and envs.SGLANG_OPT_USE_JIT_NORM.get()

    def _compute_q_a(
        self,
        x: torch.Tensor,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            q = qkv_a[..., : self.q_lora_rank]
        else:
            q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q_lora = q
        return q_lora

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)

        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return q

    def _compute_kv(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        qkv_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qkv_a is not None:
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            kv, _ = self.wkv(x)
        # [bs, head_dim]
        kv = self.kv_norm(kv)
        if positions is not None:
            fused_rope(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(kv[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4Backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        qkv_a: Optional[torch.Tensor] = None
        qkv_a_ready: Optional[torch.cuda.Event] = None
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            qkv_a_ready = current_stream.record_event()

        q_lora = self._compute_q_a(x, qkv_a=qkv_a)
        q_lora_ready = current_stream.record_event()

        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        with torch.cuda.stream(stream_kv):
            if qkv_a_ready is not None:
                stream_kv.wait_event(qkv_a_ready)
            kv = self._compute_kv(x, positions, freqs_cis, qkv_a=qkv_a)
            if self.overlap_store_cache:
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )

        del qkv_a

        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        q = self._compute_q_b(q_lora, positions, freqs_cis)
        if q_out is not None:
            q_out.copy_(q)

        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q, kv

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            q = qkv_a[..., : self.q_lora_rank]
            kv = qkv_a[..., self.q_lora_rank :]
            del qkv_a
        else:
            q, _ = self.wq_a(x)
            kv = None
        q = self.q_norm(q)
        q_lora = q
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_jit_norm:
            q = rmsnorm_self(q, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)

        if kv is None:
            kv, _ = self.wkv(x)
        # [bs, head_dim]
        kv = self.kv_norm(kv)

        fused_rope(
            q[..., -self.qk_rope_head_dim :],
            kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
            self.freqs_cis,
            positions=positions,
        )

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        if _use_cp:
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            x_for_compressor = (
                cp_all_gather_rerange_output(
                    x.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                if self.compressor is not None
                else x
            )
        else:
            x_for_compressor = x

        if self.overlap_store_cache:
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                x_for_compressor=x_for_compressor if _use_cp else None,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x_for_compressor, forward_batch, self.layer_id, self.compressor
            )

        if q_out is not None:
            q_out.copy_(q)
        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        debug_return_kv: bool = False,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4Backend)

        freqs_cis = None

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            # pad the q to [batch_size, n_heads]
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            q, kv = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )

        # for TP attention, use the padded q, since q_out is set to the correct slice
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=not self.overlap_store_cache,
        )
        # NOTE: no-op for pure DP-attention
        o = o[:, tp_slice, :]
        fused_rope(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D).contiguous(),
                group_size=128,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
            compress_ratio_override=compress_ratio_override,
        )
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        # TODO: check whether the implementation matches
        # TODO: make necessary changes if possible
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # self.layer_communicator = LayerCommunicator(
        #     layer_scatter_modes=self.layer_scatter_modes,
        #     input_layernorm=self.input_layernorm,
        #     post_attention_layernorm=self.post_attention_layernorm,
        #     allow_reduce_scatter=True,
        #     is_last_layer=(
        #         is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        #     ),
        # )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            first_k_dense_replace = 0
            moe_layer_freq = 1
        else:
            first_k_dense_replace = self.config.first_k_dense_replace
            moe_layer_freq = self.config.moe_layer_freq
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        # x: [n,hc,d] -> y: [n,d], where n=b*s
        shape, dtype = x.size(), x.dtype

        # Handle empty batch
        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            # returned post should be [n, hc_mult]
            return y, post.squeeze(-1), comb, norm is not None

        if _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_PRE.get():
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            # returned post should be [n, hc_mult]
            return y, post.squeeze(-1), comb, False

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            # DeepGEMM implementation
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            # TODO: maybe remove the contiguity requirement?
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            # Naive Torch implementation
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        # x: [n,d], residual: [n,hc,d] -> y: [n,hc,d]
        # post: [n,hc], comb: [n,hc,hc]

        # Handle empty batch
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            result = mhc_post(x, residual, post, comb)
            return result

        elif _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_POST.get():
            from aiter.ops.mhc import mhc_post

            result = torch.empty_like(residual)
            mhc_post(result, x, residual, post, comb)

            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @maybe_torch_compile
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        result = hc_post_torch_impl(x, residual, post, comb)
        return result

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            pass

        residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            norm=self.input_layernorm,
        )
        if not norm_fused:
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states
        hidden_states, post, comb, norm_fused = self.hc_pre(
            hidden_states,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            norm=self.post_attention_layernorm,
        )
        if not norm_fused:
            hidden_states = self.post_attention_layernorm(hidden_states)

        # Communication logic (equivalent to LayerCommunicator):
        #
        # ======================== i. TP MoE ========================
        # DP attn + TP moe (moe_a2a_backend=none):
        # * mlp_mode = FULL (each-rank-has-whole-world-tokens)
        # * prepare_mlp -> _gather_hidden_states_and_residual -> dp_gather_partial
        # * postprocess_layer -> _scatter_hidden_states -> dp_scatter
        # Need Gather before MoE and Scatter after MoE.
        #
        # ======================== ii. DeepEP MoE ========================
        # DP attn + DeepEP moe (moe_a2a_backend=deepep/flashinfer/etc):
        # * mlp_mode = SCATTERED (each-rank-only-has-this-rank-tokens)
        # * prepare_mlp -> _simple (just layernorm, no gather)
        # * postprocess_layer -> _trivial (no scatter)
        # Because attn_tp_size==1 when tp==dp==ep, SCATTERED and TP_ATTN_FULL
        # have the same group_size. Token dispatch/combine is handled by
        # DeepEP inside MoE forward. No Gather/Scatter around MoE.
        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        # ----------------------------------- CP: fix input_ids to LOCAL ----------------
        if _use_cp:
            # CP requires DeepEP — TP MoE's all-reduce assumes identical tokens
            # across ranks, which CP violates. (Analogous to NSACPLayerCommunicator's
            # assert mlp_mode==SCATTERED when dp_size>1.)
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            # DeepEP handles cross-rank MoE dispatch/combine internally.
            # No gather/scatter needed — tokens stay LOCAL (SCATTERED).
            # This matches DSV3.2's mlp_mode=SCATTERED behavior with DeepEP + CP.
            #
            # Hash gating (n_hash_layers=3) needs input_ids[i] to correspond to
            # hidden_states[i]. hidden_states is LOCAL [N/cp_size] (round-robin).
            # input_ids is ORIGINAL [N] on every rank (never CP-split).
            # Slice to LOCAL to match hidden_states.
            cp_rank = get_attention_tp_rank()
            cp_size = get_attention_tp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            # TODO: improve the name - it is indeed local in CP, but is only used by e.g. Hash gating
            input_ids_global = input_ids
        # ----------------------------------- DP: gather for TP MoE --------------------
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = get_global_dp_buffer(), hidden_states
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        # ----------------------------------- MoE ------------------------------------
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        # ----------------------------------- Scatter (DP only, not CP) ----------------
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        hidden_states = self.hc_post(
            hidden_states, residual, post, comb
        )  # [n, d] -> [n, hc, d]

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            deepseek_v4_moe_code_path_checker.observed = 0

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.first_k_dense_replace = config.first_k_dense_replace
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gemm_output_zero_allocator_size = 0
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        if x.numel() > 0:
            from sglang.srt.layers.mhc_head import fused_hc_head

            return fused_hc_head(
                x.contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
            )
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    # TODO
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> torch.Tensor:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )
        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        # Reset Compressor's per-step freqs_cis cache from any previous step.
        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)

        for i in range(self.start_layer, self.end_layer):
            # TODO: ctx?
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                # zero_allocator,
                # gemm_output_zero_allocator,
            )

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        pre_hc_head = (
            hidden_states.flatten(1)
            if envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
            else None
        )

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        if pre_hc_head is not None:
            return hidden_states, pre_hc_head
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        set_fp4_experts(getattr(config, "expert_dtype", None) == "fp4")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        # TODO: is this true that compress is kind of NSA
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)
            }
        )

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            if _is_hip:
                self.cp_rank = get_attention_tp_rank()
                self.cp_size = get_attention_tp_size()
            else:
                self.cp_rank = get_attention_cp_rank()
                self.cp_size = get_attention_cp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if self.config.n_routed_experts != 256 or self.config.n_shared_experts != 1:
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."
        elif getattr(self.config, "expert_dtype", None) == "fp4":
            disable_reason = "Routed experts use FP4 while shared experts remain FP8; fusion would incorrectly apply FP4 to shared experts."

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            disable_reason = "2604B checkpoint requires different clamping for shared and routed experts"

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:

        if self.nsa_enable_prefill_cp:
            if _is_hip:
                if can_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                    forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                        len(input_ids),
                        self.cp_rank,
                        self.cp_size,
                        forward_batch.seq_lens_cpu.tolist(),
                    )
            else:
                if can_nsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                    forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                        len(input_ids),
                        self.cp_rank,
                        self.cp_size,
                        forward_batch.seq_lens_cpu.tolist(),
                    )
                    if is_nsa_prefill_cp_round_robin_split():
                        metadata = forward_batch.attn_backend.forward_metadata
                        core_meta = metadata.core_attn_metadata
                        core_meta.apply_cp_reindex()
                        core_meta.init_flashmla_related()
                        if metadata.indexer_metadata is not None:
                            metadata.indexer_metadata = forward_batch.attn_backend.init_forward_metadata_indexer(
                                core_meta
                            )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        pre_hc_head = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if (
            envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            hidden_states, pre_hc_head = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            # TODO: indeed ours is "hidden_states_before_hc_head" instead of "before norm"
            #       abuse the existing field temporarily to minimize code diff
            #       should rename and generalize later, e.g. "hidden_states_for_spec"
            hidden_states_before_norm=pre_hc_head,
        )

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        layers = self.model.layers
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            # Pre-transform weight scale to DeepGEMM required layout (TMA-aligned / UE8M0 packed)
            # fp8_einsum('bhr,hdr->bhd') maps B=[h,d,r]=[G,R,D], so N=R, K=D for the B-side scale
            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        # ================ apply_ape_hotfix, should not be needed for final ckpt ================
        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    # This is used externally, please try to keep the API mostly unchanged
    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str, is_nextn: bool = False, num_hidden_layers: Optional[int] = None
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if not ATTN_BIT_WISE_EQUAL_MODE:
            if "self_attn" in name and (
                "compressor" not in name or not COMPRESSOR_BIT_WISE_EQUAL_MODE
            ):
                name = name.replace(".scale", ".weight_scale_inv")

        if not MOE_BIT_WISE_EQUAL_MODE:
            name = name.replace(".gate.tid2eid", ".topk.tid2eid")
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            if "mlp" in name:
                name = name.replace(".scale", ".weight_scale_inv")

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        assert envs.SGLANG_DSV4_MODE.get() in ["2601", "2604"]
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() in ["2604A", "2604B"]
        else:
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() == ""

        if MOE_BIT_WISE_EQUAL_MODE:
            assert (
                self.num_fused_shared_experts == 0
            ), "use --disable-shared-experts-fusion for MoE bit-wise equal mode"

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        # Ignore this, b/c it is for nvfp4 ckpt
        # weights = self._maybe_quant_weights_to_fp8_ue8m0(
        #     weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, is_nextn
        # )

        if (
            envs.SGLANG_DSV4_MODE.get() == "2604"
            and not envs.SGLANG_OPT_FP8_WO_A_GEMM.get()
        ):
            if getattr(self.config, "expert_dtype", None) == "fp4":
                weights = _dequant_fp8_wo_a(weights)
            else:
                # Converted FP8 checkpoint: wo_a is already bf16; drop stale wo_a.scale if present
                weights = ((n, t) for n, t in weights if not n.endswith(".wo_a.scale"))

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # fuse compressor wkv and wgate weights into wkv_gate
        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"  # match wkv and wgate, skip ape

        fuse_wqa_wkv = not _is_hip and envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        # use default weight loader if module has no custom weight_loader
        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",  # Note that we need a . here to avoid confusion with gate_proj
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    # remap reference's temp ckpt weight -> deepseek hf format
                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        # Use shared head and embed weights from target model
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        # Skip target model weights
                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        # For nextn specific weights (out of layer)
                        # The nextn layer prefix of these weights has been removed
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        # For decoder layer weights
                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        # Skip non-stacked layers and experts (experts handled below).
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        # We have mlp.experts[0].gate_proj in the checkpoint.
                        # Since we handle the experts below in expert_params_mapping,
                        # we need to skip here BEFORE we update the name, otherwise
                        # name will be updated to mlp.experts[0].gate_up_proj, which
                        # will then be updated below in expert_params_mapping
                        # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):  # TODO
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            if MOE_BIT_WISE_EQUAL_MODE:
                                continue
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            # Skip loading extra bias for GPTQ models.
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            # Skip loading embed_tokens if not first rank in pipeline parallelism
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            # Skip loading norm if not last rank in pipeline parallelism
                            if ".norm." in name and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate  # exactly one is true
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                bucket = cache_wqkv_a_weight.setdefault(param_name, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = loaded_weight
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(param_name)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    # modelopt attn kv scale is named differently
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    # modelopt ckpt contains not needed weights for MTP module:
                                    # model.decoder.self_attn.attn_mqa.v_scale and
                                    # model.decoder.self_attn.attn_mqa.k_scale
                                    if not name.startswith("mtp"):  # TODO: mtp
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                # if "attn_sink" in name:
                                #     attn_tp_rank = get_attention_tp_rank()
                                #     start = attn_tp_rank * param.numel()
                                #     param.data.copy_(
                                #         loaded_weight[start : start + param.numel()]
                                #     )
                                #     loaded_params.add(name)
                                #     continue

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            # hack to skip checking these in default ckpt. should have more rigorous check.
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if os.environ.get("SGLANG_SKIP_CHECKPOINT_LOAD_CHECK", "0") == "0":
            if unloaded_params:
                raise RuntimeError(
                    f"Some weights are not initialized from checkpoints: {unloaded_params}"
                )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequant fp8 block-quantized wo_a weight: bf16 = fp8_weight * e8m0_scale.

    The weight and scale use 128x128 block quantization:
      weight: [N, K] fp8_e4m3fn   (N and K must be divisible by 128)
      scale:  [N//128, K//128]    fp8_e8m0fnu  (per 128x128 block)
    """
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert (
        scale.dtype == torch.float8_e8m0fnu
    ), f"expected fp8_e8m0fnu, got {scale.dtype}"
    N, K = weight.shape
    assert (
        N % 128 == 0 and K % 128 == 0
    ), f"weight dims must be divisible by 128, got {weight.shape}"
    assert scale.shape == (
        N // 128,
        K // 128,
    ), f"scale shape {scale.shape} doesn't match weight shape {weight.shape}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    return result.to(torch.bfloat16)


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Dequant fp8 wo_a weights inline: pair (wo_a.scale, wo_a.weight) -> bf16 wo_a.weight.

    Streaming version: buffers only wo_a.scale tensors (tiny) until the
    corresponding wo_a.weight arrives.  All other weights pass through
    immediately so we never materialise the full checkpoint in memory.
    """
    pending_scales: dict[str, torch.Tensor] = {}

    for name, tensor in weights:
        if name.endswith(".wo_a.scale"):
            pending_scales[name] = tensor
            continue

        if name.endswith(".wo_a.weight") and tensor.dtype == torch.float8_e4m3fn:
            scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
            assert scale_name in pending_scales, (
                f"wo_a.scale must appear before wo_a.weight in checkpoint, "
                f"missing {scale_name}"
            )
            scale = pending_scales.pop(scale_name)
            yield name, _dequant_fp8(tensor, scale)
            continue

        yield name, tensor

    for scale_name, scale_tensor in pending_scales.items():
        yield scale_name, scale_tensor

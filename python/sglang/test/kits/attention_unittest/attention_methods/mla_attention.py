from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_token_to_kv_pool,
)
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import set_global_server_args_for_scheduler

from ..mock_server_args import make_mock_server_args

_parallel_override = get_parallel().override(attn_tp_size=1)
_parallel_override.__enter__()

DEFAULT_HIDDEN_SIZE = 64
DEFAULT_KV_LORA_RANK = 32
DEFAULT_QK_ROPE_HEAD_DIM = 0
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda"
MLA_ATOL = 3e-2
MLA_RTOL = 3e-2


@dataclass(frozen=True)
class MLAAttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...] = ()

    @property
    def batch_size(self) -> int:
        return len(self.prefix_lens)

    @property
    def input_lens(self) -> tuple[int, ...]:
        if self.forward_mode.is_decode():
            return (1,) * self.batch_size
        return self.extend_lens

    @property
    def seq_lens(self) -> tuple[int, ...]:
        return tuple(p + q for p, q in zip(self.prefix_lens, self.input_lens))

    @property
    def num_input_tokens(self) -> int:
        return sum(self.input_lens)


def make_mla_cases(backend: str) -> tuple[MLAAttentionCase, ...]:
    common = dict(backend=backend, num_heads=4)
    return (
        MLAAttentionCase(
            name="mla_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        MLAAttentionCase(
            name="mla_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        MLAAttentionCase(
            name="mla_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        MLAAttentionCase(
            name="mla_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


class TinyMLAModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        hidden_size: int,
        context_len: int,
    ):
        self.attention_arch = AttentionArch.MLA
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_key_value_heads = 1
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.head_dim = kv_lora_rank + qk_rope_head_dim
        self.v_head_dim = kv_lora_rank
        self.swa_v_head_dim = kv_lora_rank
        self.scaling = self.head_dim**-0.5
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.sliding_window_size = None
        self.hf_config = SimpleNamespace(
            architectures=["TinyMLAForCausalLM"],
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=kv_lora_rank,
        )
        self.hf_text_config = self.hf_config

    def get_num_attention_heads(self, tp_size: int) -> int:
        assert self.num_attention_heads % tp_size == 0
        return self.num_attention_heads // tp_size

    def get_num_kv_heads(self, tp_size: int) -> int:
        return 1


class MockMLAModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: MLAAttentionCase,
        model_config: TinyMLAModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        disable_cuda_graph: bool = True,
        disable_piecewise_cuda_graph: bool = True,
        runner_batch_size: int | None = None,
        fp8_kv_cache: bool = False,
    ):
        pool_batch_size = runner_batch_size or case.batch_size
        self.device = device
        self.dtype = dtype
        # `kv_cache_dtype` is the dtype the *storage* uses. For FP8 KV
        # cache (the production deployment dtype for tokenspeed_mla and
        # some trtllm_mla configs), the pool stores quantized bytes
        # while the model still projects K/V in bf16; `set_mla_kv_buffer`
        # does the BF16->FP8 cast on the way in.
        self.kv_cache_dtype = torch.float8_e4m3fn if fp8_kv_cache else dtype
        self.gpu_id = 0
        self.canary_manager = None
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        speculative_num_draft_tokens = (
            max(case.input_lens)
            if case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend_v2()
            else 0
        )
        self.server_args = make_mock_server_args(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            cuda_graph_config=CudaGraphConfig(
                decode=PhaseConfig(
                    backend=Backend.DISABLED if disable_cuda_graph else Backend.FULL,
                ),
                prefill=PhaseConfig(
                    backend=(
                        Backend.DISABLED
                        if (disable_cuda_graph or disable_piecewise_cuda_graph)
                        else Backend.TC_PIECEWISE
                    ),
                ),
            ),
            disable_chunked_prefix_cache=True,
            disable_radix_cache=False,
            disaggregation_mode=None,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            dp_size=1,
            enable_dp_attention=False,
            enable_deterministic_inference=False,
            enable_mis=False,
            flashinfer_mla_disable_ragged=True,
            is_embedding=False,
            kv_cache_dtype="fp8_e4m3" if fp8_kv_cache else "auto",
            max_running_requests=None,
            model_path=None,
            pp_size=1,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=0,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_num_steps=max(0, speculative_num_draft_tokens - 1),
            tp_size=1,
            triton_attention_num_kv_splits=8,
            triton_attention_split_tile_size=None,
        )
        set_global_server_args_for_scheduler(self.server_args)
        self.req_to_token_pool = ReqToTokenPool(
            size=pool_batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        max_token_loc = case.page_size + pool_batch_size * max_context_len
        self.token_to_kv_pool = MLATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=self.kv_cache_dtype,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=case.page_size)
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.sliding_window_size = None
        self.use_mla_backend = True
        self.is_draft_worker = False
        self._kernel_warmed_up = True
        # Runner-mode helpers mutate speculative graph sizes after construction.
        self.graph_shared_output = GraphSharedOutput(
            device=self.device,
            max_rows=pool_batch_size * max_context_len,
        )

    @property
    def hybrid_gdn_config(self):
        return None

    @property
    def hybrid_lightning_config(self):
        return None

    @property
    def kimi_linear_config(self):
        return None

    @property
    def linear_attn_model_spec(self):
        return None

    @property
    def mamba2_config(self):
        return None

    @property
    def mambaish_config(self):
        return None


class TinyDeepseekMLAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = kv_lora_rank
        self.rms_norm_eps = 1e-6
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.qk_nope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_proj = nn.Linear(
            hidden_size,
            kv_lora_rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=dtype, device=device)
        )
        self.q_rope_proj = (
            nn.Linear(
                hidden_size,
                num_heads * qk_rope_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if qk_rope_head_dim
            else None
        )
        self.k_rope_proj = (
            nn.Linear(
                hidden_size,
                qk_rope_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if qk_rope_head_dim
            else None
        )
        self.w_kc = nn.Parameter(
            torch.randn(
                num_heads,
                self.qk_nope_head_dim,
                kv_lora_rank,
                dtype=dtype,
                device=device,
            )
            * 0.1
        )
        self.w_vc = nn.Parameter(
            torch.randn(
                num_heads,
                kv_lora_rank,
                self.v_head_dim,
                dtype=dtype,
                device=device,
            )
            * 0.1
        )
        self.o_proj = nn.Linear(
            num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn_mqa = RadixAttention(
            num_heads=num_heads,
            head_dim=kv_lora_rank + qk_rope_head_dim,
            scaling=(kv_lora_rank + qk_rope_head_dim) ** -0.5,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=kv_lora_rank,
        )

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.rms_norm_eps)
        return (x.to(self.kv_a_layernorm_weight.dtype) * self.kv_a_layernorm_weight).to(
            dtype=self.kv_a_layernorm_weight.dtype
        )

    def forward_absorb_prepare(self, hidden_states: torch.Tensor):
        q_nope = self.q_proj(hidden_states).view(
            -1, self.num_heads, self.qk_nope_head_dim
        )
        k_nope = self._rms_norm(self.kv_a_proj(hidden_states)).unsqueeze(1)
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)
        if self.qk_rope_head_dim:
            assert self.q_rope_proj is not None
            assert self.k_rope_proj is not None
            q_rope = self.q_rope_proj(hidden_states).view(
                -1, self.num_heads, self.qk_rope_head_dim
            )
            k_rope = self.k_rope_proj(hidden_states).view(-1, 1, self.qk_rope_head_dim)
        else:
            q_rope = k_nope.new_empty(
                k_nope.shape[0], self.num_heads, self.qk_rope_head_dim
            )
            k_rope = k_nope.new_empty(k_nope.shape[:-1] + (self.qk_rope_head_dim,))
        return q_nope_out, k_nope, q_rope, k_rope

    def write_kv_cache(
        self,
        cache_locs: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
    ):
        token_to_kv_pool = get_token_to_kv_pool()
        if self.qk_rope_head_dim:
            token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mqa,
                cache_locs,
                k_nope,
                k_rope,
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                self.attn_mqa,
                cache_locs,
                k_nope,
                k_nope,
            )

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q_nope_out, k_nope, q_rope, k_rope = self.forward_absorb_prepare(hidden_states)
        self.write_kv_cache(forward_batch.out_cache_loc, k_nope, k_rope)
        q = q_nope_out
        if self.qk_rope_head_dim:
            q = torch.cat([q_nope_out, q_rope], dim=-1)
        attn_output = self.attn_mqa(
            q.flatten(1, 2),
            None,
            None,
            forward_batch,
            save_kv_cache=False,
        )
        attn_output = attn_output.view(-1, self.num_heads, self.kv_lora_rank)
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc).transpose(
            0, 1
        )
        return self.o_proj(attn_bmm_output.flatten(1, 2))


class ReferenceDeepseekMLAAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = kv_lora_rank
        self.scaling = (kv_lora_rank + qk_rope_head_dim) ** -0.5
        self.rms_norm_eps = 1e-6
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * self.qk_nope_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_proj = nn.Linear(
            hidden_size,
            kv_lora_rank,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.kv_a_layernorm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=dtype, device=device)
        )
        self.q_rope_proj = (
            nn.Linear(
                hidden_size,
                num_heads * qk_rope_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if qk_rope_head_dim
            else None
        )
        self.k_rope_proj = (
            nn.Linear(
                hidden_size,
                qk_rope_head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            if qk_rope_head_dim
            else None
        )
        self.w_kc = nn.Parameter(
            torch.empty(
                num_heads,
                self.qk_nope_head_dim,
                kv_lora_rank,
                dtype=dtype,
                device=device,
            )
        )
        self.w_vc = nn.Parameter(
            torch.empty(
                num_heads,
                kv_lora_rank,
                self.v_head_dim,
                dtype=dtype,
                device=device,
            )
        )
        self.o_proj = nn.Linear(
            num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.rms_norm_eps)
        return (x.to(self.kv_a_layernorm_weight.dtype) * self.kv_a_layernorm_weight).to(
            dtype=self.kv_a_layernorm_weight.dtype
        )

    def project_latent_qk(self, hidden_states: torch.Tensor):
        q_nope = self.q_proj(hidden_states).view(
            -1, self.num_heads, self.qk_nope_head_dim
        )
        k_nope = self._rms_norm(self.kv_a_proj(hidden_states)).unsqueeze(1)
        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)
        if self.qk_rope_head_dim:
            assert self.q_rope_proj is not None
            assert self.k_rope_proj is not None
            q_rope = self.q_rope_proj(hidden_states).view(
                -1, self.num_heads, self.qk_rope_head_dim
            )
            k_rope = self.k_rope_proj(hidden_states).view(-1, 1, self.qk_rope_head_dim)
        else:
            q_rope = k_nope.new_empty(
                k_nope.shape[0], self.num_heads, self.qk_rope_head_dim
            )
            k_rope = k_nope.new_empty(k_nope.shape[:-1] + (self.qk_rope_head_dim,))
        return q_nope_out, k_nope, q_rope, k_rope

    def reconstruct_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc).transpose(
            0, 1
        )
        return F.linear(attn_bmm_output.flatten(1, 2), self.o_proj.weight)


@dataclass
class MLAAttentionFixture:
    case: MLAAttentionCase
    runner: MockMLAModelRunner
    backend: object
    actual_module: TinyDeepseekMLAAttention
    reference_module: ReferenceDeepseekMLAAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    return page_size + req_idx * max_context_len + pos


def _make_forward_batch(
    case: MLAAttentionCase,
    runner: MockMLAModelRunner,
    *,
    max_context_len: int,
    device: str,
    loc_fn=None,
) -> ForwardBatch:
    seq_lens = case.seq_lens
    input_lens = case.input_lens
    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    out_cache_locs: List[int] = []
    positions: List[int] = []

    if loc_fn is None:

        def loc_fn(req_idx: int, pos: int) -> int:
            return _token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )

    for req_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = loc_fn(req_idx, pos)

        if case.forward_mode.is_decode():
            positions.append(seq_len - 1)
            out_cache_locs.append(loc_fn(req_idx, seq_len - 1))
        else:
            prefix_len = case.prefix_lens[req_idx]
            for offset in range(input_lens[req_idx]):
                positions.append(prefix_len + offset)
                out_cache_locs.append(loc_fn(req_idx, prefix_len + offset))

    batch = ForwardBatch(
        forward_mode=case.forward_mode,
        batch_size=case.batch_size,
        input_ids=torch.arange(case.num_input_tokens, dtype=torch.int64, device=device),
        req_pool_indices=req_pool_indices,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32, device="cpu"),
        out_cache_loc=torch.tensor(out_cache_locs, dtype=torch.int64, device=device),
        seq_lens_sum=sum(seq_lens),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
    )

    if case.forward_mode.is_extend(include_draft_extend_v2=True):
        extend_seq_lens = torch.tensor(input_lens, dtype=torch.int32, device=device)
        batch.extend_prefix_lens = torch.tensor(
            case.prefix_lens, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = list(case.prefix_lens)
        batch.extend_seq_lens = extend_seq_lens
        batch.extend_seq_lens_cpu = list(input_lens)
        batch.extend_start_loc = torch.zeros_like(extend_seq_lens)
        if case.batch_size > 1:
            batch.extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0)
        batch.extend_num_tokens = case.num_input_tokens

    return batch


def _split_by_lens(tensor: torch.Tensor, lens: tuple[int, ...]):
    parts = []
    start = 0
    for length in lens:
        parts.append(tensor[start : start + length])
        start += length
    return parts


def _mla_attention_reference(
    module: ReferenceDeepseekMLAAttention,
    case: MLAAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k, q_rope, k_rope = module.project_latent_qk(input_hidden)
    q_parts = _split_by_lens(q, case.input_lens)
    k_parts = _split_by_lens(k, case.input_lens)
    q_rope_parts = _split_by_lens(q_rope, case.input_lens)
    k_rope_parts = _split_by_lens(k_rope, case.input_lens)
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k, _, prefix_k_rope = module.project_latent_qk(prefix)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0).squeeze(1)
        req_k_rope = torch.cat([prefix_k_rope, k_rope_parts[req_idx]], dim=0).squeeze(1)

        for offset, query in enumerate(q_parts[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            keys = req_k[: query_pos + 1].movedim(0, 1)
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,dk->hk", query, keys) * module.scaling
            if module.qk_rope_head_dim:
                query_rope = q_rope_parts[req_idx][offset].float()
                keys_rope = req_k_rope[: query_pos + 1].movedim(0, 1).float()
                scores = scores + (
                    torch.einsum("hd,dk->hk", query_rope, keys_rope) * module.scaling
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,kd->hd", probs, req_k[: query_pos + 1].float())
            outputs.append(out)

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def mla_attention_reference_with_custom_mask(
    module: ReferenceDeepseekMLAAttention,
    case: MLAAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
    custom_mask_by_req: list[torch.Tensor],
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k, q_rope, k_rope = module.project_latent_qk(input_hidden)
    q_parts = _split_by_lens(q, case.input_lens)
    k_parts = _split_by_lens(k, case.input_lens)
    q_rope_parts = _split_by_lens(q_rope, case.input_lens)
    k_rope_parts = _split_by_lens(k_rope, case.input_lens)
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k, _, prefix_k_rope = module.project_latent_qk(prefix)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0).squeeze(1)
        req_k_rope = torch.cat([prefix_k_rope, k_rope_parts[req_idx]], dim=0).squeeze(1)
        req_mask = custom_mask_by_req[req_idx].to(torch.bool)

        for offset, query in enumerate(q_parts[req_idx]):
            allowed = req_mask[offset, : req_k.shape[0]]
            keys = req_k[allowed].movedim(0, 1)
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,dk->hk", query, keys) * module.scaling
            if module.qk_rope_head_dim:
                query_rope = q_rope_parts[req_idx][offset].float()
                keys_rope = req_k_rope[allowed].movedim(0, 1).float()
                scores = scores + (
                    torch.einsum("hd,dk->hk", query_rope, keys_rope) * module.scaling
                )
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,kd->hd", probs, req_k[allowed].float())
            outputs.append(out)

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def _populate_prefix_kv(
    module: TinyDeepseekMLAAttention,
    case: MLAAttentionCase,
    runner: MockMLAModelRunner,
    backend: object,
    prefix_hidden: list[torch.Tensor],
    *,
    max_context_len: int,
    loc_fn=None,
):
    if loc_fn is None:

        def loc_fn(req_idx: int, pos: int) -> int:
            return _token_loc(
                req_idx,
                pos,
                page_size=case.page_size,
                max_context_len=max_context_len,
            )

    locs = []
    keys = []
    ropes = []
    for req_idx, prefix in enumerate(prefix_hidden):
        if prefix.shape[0] == 0:
            continue
        _, k, _, k_rope = module.forward_absorb_prepare(prefix)
        keys.append(k)
        ropes.append(k_rope)
        for pos in range(prefix.shape[0]):
            locs.append(loc_fn(req_idx, pos))

    if not locs:
        return

    loc_tensor = torch.tensor(locs, dtype=torch.int64, device=runner.device)
    cache_k = torch.cat(keys, dim=0)
    cache_k_rope = torch.cat(ropes, dim=0)
    with forward_context(ForwardContext(attn_backend=backend)):
        module.write_kv_cache(loc_tensor, cache_k, cache_k_rope)


def _copy_mla_weights(
    actual: TinyDeepseekMLAAttention,
    reference: ReferenceDeepseekMLAAttention,
):
    with torch.no_grad():
        reference.q_proj.weight.copy_(actual.q_proj.weight)
        reference.kv_a_proj.weight.copy_(actual.kv_a_proj.weight)
        reference.kv_a_layernorm_weight.copy_(actual.kv_a_layernorm_weight)
        if actual.q_rope_proj is not None:
            assert reference.q_rope_proj is not None
            reference.q_rope_proj.weight.copy_(actual.q_rope_proj.weight)
        if actual.k_rope_proj is not None:
            assert reference.k_rope_proj is not None
            reference.k_rope_proj.weight.copy_(actual.k_rope_proj.weight)
        reference.w_kc.copy_(actual.w_kc)
        reference.w_vc.copy_(actual.w_vc)
        reference.o_proj.weight.copy_(actual.o_proj.weight)


def build_mla_attention_fixture(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    fp8_kv_cache: bool = False,
    loc_layout: str = "shuffled_pages",
) -> MLAAttentionFixture:
    seed = 3090 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyMLAModelConfig(
        num_heads=case.num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
    )
    runner = MockMLAModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
        fp8_kv_cache=fp8_kv_cache,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    actual_module = TinyDeepseekMLAAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )
    reference_module = ReferenceDeepseekMLAAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dtype=dtype,
        device=device,
    )
    _copy_mla_weights(actual_module, reference_module)
    prefix_hidden = [
        torch.randn(length, hidden_size, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    from .dense_attention import make_loc_fn as _dense_make_loc_fn

    loc_fn = _dense_make_loc_fn(
        loc_layout,
        batch_size=case.batch_size,
        seq_lens=case.seq_lens,
        prefix_lens=case.prefix_lens,
        page_size=case.page_size,
        max_context_len=max_context_len,
        seed=seed,
    )
    forward_batch = _make_forward_batch(
        case,
        runner,
        max_context_len=max_context_len,
        device=device,
        loc_fn=loc_fn,
    )
    _populate_prefix_kv(
        actual_module,
        case,
        runner,
        backend,
        prefix_hidden,
        max_context_len=max_context_len,
        loc_fn=loc_fn,
    )

    return MLAAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def run_mla_fixture_eager(fixture: MLAAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(fixture.input_hidden, fixture.forward_batch)


def expected_mla_fixture_output(fixture: MLAAttentionFixture) -> torch.Tensor:
    return _mla_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def make_mla_case_with_prefix_lens(
    case: MLAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> MLAAttentionCase:
    extend_lens = ()
    if not case.forward_mode.is_decode():
        if not case.input_lens:
            raise ValueError("Non-decode cases require input lengths.")
        if len(prefix_lens) <= len(case.input_lens):
            extend_lens = case.input_lens[: len(prefix_lens)]
        else:
            extend_lens = case.input_lens + (case.input_lens[-1],) * (
                len(prefix_lens) - len(case.input_lens)
            )

    return MLAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
    )


def make_mla_case_with_lens(
    case: MLAAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
    input_lens: tuple[int, ...],
) -> MLAAttentionCase:
    return MLAAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=input_lens,
    )


def mla_fixture_inputs(fixture: MLAAttentionFixture) -> dict[str, Any]:
    return {
        "prefix_hidden": fixture.prefix_hidden,
        "input_hidden": fixture.input_hidden,
    }


def _random_hidden_by_lens(
    lens: tuple[int, ...],
    *,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
) -> list[torch.Tensor]:
    return [
        torch.randn(length, hidden_size, dtype=dtype, device=device) for length in lens
    ]


def make_mla_random_inputs(
    case: MLAAttentionCase,
    fixture: MLAAttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    hidden_size = fixture.actual_module.hidden_size
    return {
        "prefix_hidden": _random_hidden_by_lens(
            case.prefix_lens,
            hidden_size=hidden_size,
            dtype=dtype,
            device=device,
        ),
        "input_hidden": torch.randn(
            case.num_input_tokens,
            hidden_size,
            dtype=dtype,
            device=device,
        ),
    }


def make_mla_padded_replay_inputs(
    case: MLAAttentionCase,
    fixture: MLAAttentionFixture,
    pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    hidden_size = fixture.actual_module.hidden_size
    pad_prefix_hidden = _random_hidden_by_lens(
        pad_prefix_lens,
        hidden_size=hidden_size,
        dtype=dtype,
        device=device,
    )
    pad_input_hidden = torch.randn(
        case.num_input_tokens - base_inputs["input_hidden"].shape[0],
        hidden_size,
        dtype=dtype,
        device=device,
    )
    return {
        "prefix_hidden": base_inputs["prefix_hidden"] + pad_prefix_hidden,
        "input_hidden": torch.cat(
            [base_inputs["input_hidden"], pad_input_hidden],
            dim=0,
        ),
    }


def make_mla_token_padded_inputs(
    _case: MLAAttentionCase,
    fixture: MLAAttentionFixture,
    static_num_tokens: int,
    base_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    hidden_size = fixture.actual_module.hidden_size
    raw_num_tokens = base_inputs["input_hidden"].shape[0]
    if static_num_tokens < raw_num_tokens:
        raise ValueError("static_num_tokens must cover the live input token count.")

    pad_input_hidden = torch.randn(
        static_num_tokens - raw_num_tokens,
        hidden_size,
        dtype=dtype,
        device=device,
    )
    return {
        "prefix_hidden": base_inputs["prefix_hidden"],
        "input_hidden": torch.cat(
            [base_inputs["input_hidden"], pad_input_hidden],
            dim=0,
        ),
    }


def prepare_mla_runner_inputs(
    fixture: MLAAttentionFixture,
    case: MLAAttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, Any],
    *,
    max_context_len: int,
) -> None:
    del batch
    _populate_prefix_kv(
        fixture.actual_module,
        case,
        fixture.runner,
        fixture.backend,
        inputs["prefix_hidden"],
        max_context_len=max_context_len,
    )


def run_mla_forward(
    fixture: MLAAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, Any],
) -> torch.Tensor:
    return fixture.actual_module(inputs["input_hidden"], batch)


def mla_attention_layers(fixture: MLAAttentionFixture) -> list[RadixAttention]:
    return [fixture.actual_module.attn_mqa]


def expected_mla_output_from_inputs(
    fixture: MLAAttentionFixture,
    case: MLAAttentionCase,
    inputs: dict[str, Any],
    _state,
) -> torch.Tensor:
    return _mla_attention_reference(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
    )


def run_mla_attention_case(
    testcase,
    case: MLAAttentionCase,
    *,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
    qk_rope_head_dim: int = DEFAULT_QK_ROPE_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    fp8_kv_cache: bool = False,
    atol: float = MLA_ATOL,
    rtol: float = MLA_RTOL,
    loc_layout: str = "shuffled_pages",
):
    fixture = build_mla_attention_fixture(
        testcase,
        case,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        fp8_kv_cache=fp8_kv_cache,
        loc_layout=loc_layout,
    )
    actual = run_mla_fixture_eager(fixture)
    expected = expected_mla_fixture_output(fixture)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

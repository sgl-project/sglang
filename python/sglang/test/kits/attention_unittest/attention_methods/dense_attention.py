from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from ..mock_server_args import make_mock_server_args

# Unit tests run without distributed initialization. Backends that size buffers by
# attention tensor-parallel degree should see the single-rank default.
_parallel_override = get_parallel().override(attn_tp_size=1)
_parallel_override.__enter__()

DEFAULT_HEAD_DIM = 16
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_MAX_CONTEXT_LEN = 64
DEFAULT_DTYPE = torch.float16
DEFAULT_DEVICE = "cuda"
DENSE_ATOL = 3e-2
DENSE_RTOL = 3e-2

# SWA decode rule classification — production metadata builders differ:
#   - `min_seq_len_window` rule: `window_kv_lens = min(seq_lens, window)` (the
#     extra current-token slot is NOT included; total = `window` keys).
#   - `extend_window` rule: keys at `[query_pos - window, query_pos]` are
#     allowed by the extend kernel mask (the current token IS included; total
#     = `window + 1` keys). FlashInfer's SWA decode metadata uses
#     `clamp(seq_lens, max=window + 1)` (`flashinfer_backend.py:1031`) which
#     gives `window + 1` keys when `seq_len > window`, matching this rule.
#     Within-window seqs collapse to `seq_len` in both rules, so cases that
#     stay below the window can't distinguish them.
# Each known backend must be classified into exactly one set; an unclassified
# backend trips `_swa_decode_uses_min_seq_len_rule` so a future SWA backend
# can't silently inherit the wrong rule via a fallback.
_SWA_DECODE_MIN_SEQ_LEN_WINDOW: frozenset[str] = frozenset({"triton"})
_SWA_DECODE_EXTEND_WINDOW: frozenset[str] = frozenset(
    {"torch_native", "fa3", "fa4", "flex_attention", "trtllm_mha", "flashinfer"}
)


def _swa_decode_uses_min_seq_len_rule(case: "DenseAttentionCase") -> bool:
    if case.backend in _SWA_DECODE_MIN_SEQ_LEN_WINDOW:
        return True
    if case.backend in _SWA_DECODE_EXTEND_WINDOW:
        return False
    raise ValueError(
        f"Unknown SWA decode rule for backend {case.backend!r}. Add it to "
        f"either `_SWA_DECODE_MIN_SEQ_LEN_WINDOW` or `_SWA_DECODE_EXTEND_WINDOW` "
        f"in common/attention_methods/dense_attention.py, depending on what its "
        f"`init_forward_metadata_decode` metadata builder produces."
    )


@dataclass(frozen=True)
class DenseAttentionCase:
    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    num_kv_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...] = ()
    sliding_window_size: int | None = None
    # Tree spec decoding: backends read this at construction time (e.g. the
    # trtllm_mha verify path keys its tree-mask handling on topk > 1).
    speculative_eagle_topk: int = 0

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


def make_dense_input_config_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    """MHA cases that focus on input-layout coverage, not head-layout coverage."""
    common = dict(backend=backend, num_heads=4, num_kv_heads=4)
    return (
        DenseAttentionCase(
            name="mha_extend_page_size_1",
            forward_mode=ForwardMode.EXTEND,
            page_size=1,
            prefix_lens=(2, 4),
            extend_lens=(3, 1),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_zero_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0,),
            extend_lens=(16,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_zero_prefix_input_page_edges",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(15, 16, 17),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_prefix_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(16,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_total_exact_page",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_cross_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(15,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_ragged_page_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=16,
            prefix_lens=(0, 8, 16),
            extend_lens=(15, 8, 1),
            **common,
        ),
        DenseAttentionCase(
            name="mha_extend_page32_cross_boundary",
            forward_mode=ForwardMode.EXTEND,
            page_size=32,
            prefix_lens=(31,),
            extend_lens=(2,),
            **common,
        ),
        DenseAttentionCase(
            name="mha_decode_page_boundary",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(14, 15, 16),
            **common,
        ),
        DenseAttentionCase(
            name="mha_decode_bsz1_nonzero_prefix",
            forward_mode=ForwardMode.DECODE,
            page_size=16,
            prefix_lens=(7,),
            **common,
        ),
    )


def make_dense_attention_config_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    """Head-layout variants. Keep these separate from input-layout coverage."""
    return (
        DenseAttentionCase(
            name="gqa_decode_page_boundary",
            backend=backend,
            forward_mode=ForwardMode.DECODE,
            num_heads=4,
            num_kv_heads=2,
            page_size=16,
            prefix_lens=(14, 15, 16),
        ),
        DenseAttentionCase(
            name="mqa_extend_total_exact_page",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=1,
            page_size=16,
            prefix_lens=(8,),
            extend_lens=(8,),
        ),
    )


def make_dense_cases(backend: str) -> tuple[DenseAttentionCase, ...]:
    return make_dense_input_config_cases(backend) + make_dense_attention_config_cases(
        backend
    )


def make_swa_no_prefix_input_config_cases(
    backend: str,
) -> tuple[DenseAttentionCase, ...]:
    """SWA no-prefix cases with lengths below, exactly at, and above the window."""
    return (
        DenseAttentionCase(
            name="swa_extend_no_prefix_window_edges",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(0, 0, 0),
            extend_lens=(3, 4, 5),
            sliding_window_size=4,
        ),
    )


def make_swa_prefix_input_config_cases(
    backend: str,
) -> tuple[DenseAttentionCase, ...]:
    """SWA prefix cases with prefix lengths below, at, and above the window."""
    return (
        DenseAttentionCase(
            name="swa_extend_prefix_window_edges",
            backend=backend,
            forward_mode=ForwardMode.EXTEND,
            num_heads=4,
            num_kv_heads=4,
            page_size=16,
            prefix_lens=(3, 4, 5),
            extend_lens=(2, 2, 2),
            sliding_window_size=4,
        ),
    )


class TinyModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        context_len: int,
        sliding_window_size: int | None = None,
    ):
        self.attention_arch = AttentionArch.MHA
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.v_head_dim = head_dim
        self.swa_v_head_dim = head_dim
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.is_hybrid_swa = sliding_window_size is not None
        self.is_local_attention_model = sliding_window_size is not None
        self.attention_chunk_size = None
        self.sliding_window_size = sliding_window_size
        self.hf_config = SimpleNamespace(
            architectures=["TinyForCausalLM"],
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.hf_text_config = self.hf_config

    def get_num_attention_heads(self, tp_size: int) -> int:
        assert self.num_attention_heads % tp_size == 0
        return self.num_attention_heads // tp_size

    def get_num_kv_heads(self, tp_size: int) -> int:
        assert self.num_key_value_heads % tp_size == 0
        return self.num_key_value_heads // tp_size


class MockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        case: DenseAttentionCase,
        model_config: TinyModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        head_dim: int,
        disable_cuda_graph: bool = True,
        disable_piecewise_cuda_graph: bool = True,
        runner_batch_size: int | None = None,
    ):
        pool_batch_size = runner_batch_size or case.batch_size
        self.device = device
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.gpu_id = 0
        self.canary_manager = None
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.is_draft_worker = False
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        # The runner lifecycle warms up kernels in capture() / first execute()
        # via BaseRunner.warmup(); this mock never calls init_backends and has no
        # real kernels to warm up, so mark it done (warmup becomes a no-op for
        # the runner-mode attention tests that drive capture directly).
        self._kernel_warmed_up = True
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
            disable_radix_cache=False,
            dllm_algorithm=None,
            dllm_algorithm_config=None,
            dp_size=1,
            enable_dp_attention=False,
            enable_deterministic_inference=False,
            enable_mis=False,
            is_embedding=False,
            kv_cache_dtype="auto",
            max_running_requests=None,
            model_path=None,
            pp_size=1,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=case.speculative_eagle_topk,
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
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_token_loc + case.page_size,
            page_size=case.page_size,
            dtype=dtype,
            head_num=case.num_kv_heads,
            head_dim=head_dim,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=case.page_size,
            get_kvcache=lambda: self.token_to_kv_pool,
        )
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = case.sliding_window_size is not None
        self.sliding_window_size = case.sliding_window_size
        self.use_mla_backend = False

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


class ProjectedDenseAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        sliding_window_size: int | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=head_dim,
            scaling=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            layer_id=0,
            sliding_window_size=(
                sliding_window_size if sliding_window_size is not None else -1
            ),
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        return q, k, v

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        q, k, v = self.project_qkv(hidden_states)
        attn_output = self.attn(q, k, v, forward_batch)
        return self.o_proj(attn_output)


class ReferenceDenseAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.v_proj = nn.Linear(
            hidden_size,
            num_kv_heads * head_dim,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.o_proj = nn.Linear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

    def project_qkv(self, hidden_states: torch.Tensor):
        return (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )

    def reconstruct_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        return F.linear(attn_output, self.o_proj.weight)


@dataclass
class DenseAttentionFixture:
    case: DenseAttentionCase
    runner: MockModelRunner
    backend: object
    actual_module: ProjectedDenseAttention
    reference_module: ReferenceDenseAttention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, page_size: int, max_context_len: int) -> int:
    """Default contiguous loc mapping: each request gets a contiguous block of
    `max_context_len` slots starting at slot `page_size + req_idx * max_context_len`.

    See `make_loc_fn(layout=...)` for non-tidy variants that catch backend
    bugs in page-table derivation from non-contiguous `out_cache_loc` /
    `req_to_token`.
    """
    return page_size + req_idx * max_context_len + pos


def make_loc_fn(
    layout: str,
    *,
    batch_size: int,
    seq_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    page_size: int,
    max_context_len: int,
    seed: int = 0,
):
    """Build a `(req_idx, pos) -> physical_cache_loc` callable for non-tidy
    layouts that stress the backend's `(req_to_token, out_cache_loc)`
    interpretation.

    Layouts:
    - ``contiguous``: the original tidy mapping (`_token_loc`).
      Each request's pages occupy a contiguous physical-slot range.
      Kept as a baseline for regression tests; production rarely
      produces this exact layout.
    - ``shuffled_pages`` (DEFAULT): within each request, page order is
      randomly permuted. The set of physical pages is unchanged; only
      the mapping (logical_page -> physical_page) is. Catches backends
      that assume `req_to_token[req_idx, pos]` increases monotonically
      with `pos`. Picked as the default because (a) it's
      production-realistic — allocator fragmentation can produce
      non-monotonic per-request page assignments — and (b) all backends
      currently pass it, so no existing tests break by enabling it.
    - ``interleaved_pages``: pages from different requests are interleaved
      in physical-slot order. With `bs=2`, req 0's pages land on physical
      pages [0, 2, 4, ...] and req 1's on [1, 3, 5, ...]. Catches
      backends assuming a request's pages occupy a contiguous physical
      range.
    - ``non_monotonic_extend``: prefix uses contiguous layout; the
      extend tokens for a request scatter to slots in a non-monotonic
      order, which is what fragmented allocators can produce in
      production. Catches backends assuming `out_cache_loc[i+1] ==
      out_cache_loc[i] + 1` within an extend.

    All non-contiguous layouts produce a bijection over the same set
    of physical slots used by the contiguous baseline, so the test
    pool size stays unchanged.
    """
    if layout == "contiguous":

        def loc_fn(req_idx: int, pos: int) -> int:
            return _token_loc(
                req_idx, pos, page_size=page_size, max_context_len=max_context_len
            )

        return loc_fn

    import random

    rng = random.Random(seed)

    # Each layout precomputes a per-request mapping `pos -> slot_offset_within_request`.
    # Final loc = page_size + req_idx * max_context_len + slot_offset.
    # For interleaved we additionally rewrite the per-request base.
    per_req_mapping: list[dict[int, int]] = []
    per_req_base: list[int] = []

    pages_per_req = max(1, max_context_len // page_size)

    if layout == "shuffled_pages":
        for req_idx in range(batch_size):
            page_perm = list(range(pages_per_req))
            random.Random(seed + 17 * (req_idx + 1)).shuffle(page_perm)
            mapping = {}
            for pos in range(seq_lens[req_idx]):
                logical_page = pos // page_size
                pos_within = pos % page_size
                physical_page = page_perm[logical_page % pages_per_req]
                mapping[pos] = physical_page * page_size + pos_within
            per_req_mapping.append(mapping)
            per_req_base.append(req_idx * max_context_len)

    elif layout == "interleaved_pages":
        # Global physical page assignment: with bs=B, request r's logical
        # page p maps to physical page (p * B + r). All requests share the
        # same global pool [0, total_pages * page_size).
        # Total pages allocated = max(pages_per_req * batch_size,
        #                            sum(ceil(seq_len/page_size)))
        for req_idx in range(batch_size):
            mapping = {}
            for pos in range(seq_lens[req_idx]):
                logical_page = pos // page_size
                pos_within = pos % page_size
                physical_page = logical_page * batch_size + req_idx
                mapping[pos] = physical_page * page_size + pos_within
            per_req_mapping.append(mapping)
            per_req_base.append(0)  # no per-request offset; pages are global

    elif layout == "non_monotonic_extend":
        # Prefix tokens stay contiguous; extend tokens (positions
        # >= prefix_lens[req_idx]) are scattered within the request's
        # block via a fixed permutation. The set of slots is unchanged.
        for req_idx in range(batch_size):
            prefix_len = prefix_lens[req_idx]
            extend_len = seq_lens[req_idx] - prefix_len
            mapping = {}
            for pos in range(prefix_len):
                mapping[pos] = pos
            extend_perm = list(range(extend_len))
            random.Random(seed + 31 * (req_idx + 1)).shuffle(extend_perm)
            for offset in range(extend_len):
                # original extend position is prefix_len + offset
                # remapped position within the request's block:
                # prefix_len + extend_perm[offset]
                mapping[prefix_len + offset] = prefix_len + extend_perm[offset]
            per_req_mapping.append(mapping)
            per_req_base.append(req_idx * max_context_len)

    else:
        raise ValueError(f"unknown loc layout: {layout!r}")

    def loc_fn(req_idx: int, pos: int) -> int:
        within = per_req_mapping[req_idx][pos]
        return page_size + per_req_base[req_idx] + within

    return loc_fn


def _make_forward_batch(
    case: DenseAttentionCase,
    runner: MockModelRunner,
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
        batch.extend_prefix_lens = torch.tensor(
            case.prefix_lens, dtype=torch.int32, device=device
        )
        batch.extend_prefix_lens_cpu = list(case.prefix_lens)
        batch.extend_seq_lens = torch.tensor(
            input_lens, dtype=torch.int32, device=device
        )
        batch.extend_seq_lens_cpu = list(input_lens)
        batch.extend_num_tokens = case.num_input_tokens

    return batch


def _split_by_lens(tensor: torch.Tensor, lens: tuple[int, ...]):
    parts = []
    start = 0
    for length in lens:
        parts.append(tensor[start : start + length])
        start += length
    return parts


def _expand_gqa(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[0]
    if num_kv_heads == num_heads:
        return x
    assert num_heads % num_kv_heads == 0
    return x.repeat_interleave(num_heads // num_kv_heads, dim=0)


def _dense_attention_reference(
    module: ReferenceDenseAttention,
    case: DenseAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k, v = module.project_qkv(input_hidden)
    q_parts = _split_by_lens(
        q.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    k_parts = _split_by_lens(
        k.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    v_parts = _split_by_lens(
        v.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k, prefix_v = module.project_qkv(prefix)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, module.head_dim)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, module.head_dim)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0)
        req_v = torch.cat([prefix_v, v_parts[req_idx]], dim=0)

        for offset, query in enumerate(q_parts[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            key_start = 0
            if case.sliding_window_size is not None:
                # Two SWA mask rules in production:
                #   - extend kernel: `kv_id >= q_id - window` (window + 1 keys).
                #   - SWA-aware decode metadata: `min(seq_lens, window)` keys.
                if case.forward_mode.is_decode() and _swa_decode_uses_min_seq_len_rule(
                    case
                ):
                    key_start = max(0, query_pos + 1 - case.sliding_window_size)
                else:
                    key_start = max(0, query_pos - case.sliding_window_size)
            keys = _expand_gqa(
                req_k[key_start : query_pos + 1].movedim(0, 1), case.num_heads
            )
            values = _expand_gqa(
                req_v[key_start : query_pos + 1].movedim(0, 1), case.num_heads
            )
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,hkd->hk", query, keys) * module.scaling
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, values.float())
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def dense_attention_reference_with_custom_mask(
    module: ReferenceDenseAttention,
    case: DenseAttentionCase,
    prefix_hidden: list[torch.Tensor],
    input_hidden: torch.Tensor,
    custom_mask_by_req: list[torch.Tensor],
) -> torch.Tensor:
    dtype = input_hidden.dtype
    q, k, v = module.project_qkv(input_hidden)
    q_parts = _split_by_lens(
        q.view(-1, case.num_heads, module.head_dim), case.input_lens
    )
    k_parts = _split_by_lens(
        k.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    v_parts = _split_by_lens(
        v.view(-1, case.num_kv_heads, module.head_dim), case.input_lens
    )
    outputs = []

    for req_idx, prefix in enumerate(prefix_hidden):
        _, prefix_k, prefix_v = module.project_qkv(prefix)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, module.head_dim)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, module.head_dim)
        req_k = torch.cat([prefix_k, k_parts[req_idx]], dim=0)
        req_v = torch.cat([prefix_v, v_parts[req_idx]], dim=0)
        req_mask = custom_mask_by_req[req_idx].to(torch.bool)

        for offset, query in enumerate(q_parts[req_idx]):
            allowed = req_mask[offset, : req_k.shape[0]]
            if case.sliding_window_size is not None:
                query_pos = case.prefix_lens[req_idx] + offset
                # Target-verify draft tokens are appended through the extend
                # kernel, which applies `kv_id >= q_id - sliding_window_size`
                # (see `dense_attention_reference` note). The custom-mask
                # reference must use the same rule.
                window_allowed = torch.arange(
                    req_k.shape[0], device=req_k.device
                ) >= max(0, query_pos - case.sliding_window_size)
                allowed = allowed & window_allowed
            keys = _expand_gqa(req_k[allowed].movedim(0, 1), case.num_heads)
            values = _expand_gqa(req_v[allowed].movedim(0, 1), case.num_heads)
            query = query.float()
            keys = keys.float()
            scores = torch.einsum("hd,hkd->hk", query, keys) * module.scaling
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, values.float())
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def _copy_dense_weights(
    actual: ProjectedDenseAttention,
    reference: ReferenceDenseAttention,
):
    with torch.no_grad():
        reference.q_proj.weight.copy_(actual.q_proj.weight)
        reference.k_proj.weight.copy_(actual.k_proj.weight)
        reference.v_proj.weight.copy_(actual.v_proj.weight)
        reference.o_proj.weight.copy_(actual.o_proj.weight)


def _populate_prefix_kv(
    module: ProjectedDenseAttention,
    case: DenseAttentionCase,
    runner: MockModelRunner,
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
    values = []
    for req_idx, prefix in enumerate(prefix_hidden):
        if prefix.shape[0] == 0:
            continue
        _, k, v = module.project_qkv(prefix)
        keys.append(k.view(-1, case.num_kv_heads, module.head_dim))
        values.append(v.view(-1, case.num_kv_heads, module.head_dim))
        for pos in range(prefix.shape[0]):
            locs.append(loc_fn(req_idx, pos))

    if not locs:
        return

    loc_tensor = torch.tensor(locs, dtype=torch.int64, device=runner.device)
    runner.token_to_kv_pool.set_kv_buffer(
        module.attn,
        loc_tensor,
        torch.cat(keys, dim=0),
        torch.cat(values, dim=0),
    )


def build_dense_attention_fixture(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    loc_layout: str = "shuffled_pages",
) -> DenseAttentionFixture:
    seed = 2026 + len(case.name) + case.num_kv_heads
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyModelConfig(
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        context_len=max_context_len,
        sliding_window_size=case.sliding_window_size,
    )
    runner = MockModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        head_dim=head_dim,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    actual_module = ProjectedDenseAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        sliding_window_size=case.sliding_window_size,
    )
    reference_module = ReferenceDenseAttention(
        hidden_size=hidden_size,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    _copy_dense_weights(actual_module, reference_module)
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
    loc_fn = make_loc_fn(
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
        prefix_hidden,
        max_context_len=max_context_len,
        loc_fn=loc_fn,
    )

    return DenseAttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        reference_module=reference_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def run_dense_fixture_eager(fixture: DenseAttentionFixture) -> torch.Tensor:
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture.actual_module(fixture.input_hidden, fixture.forward_batch)


def replace_backend(fixture: DenseAttentionFixture, backend) -> DenseAttentionFixture:
    """Swap the backend on a built fixture (used to wire wrapper backends)."""
    fixture.backend = backend
    return fixture


def expected_dense_fixture_output(fixture: DenseAttentionFixture) -> torch.Tensor:
    return _dense_attention_reference(
        fixture.reference_module,
        fixture.case,
        fixture.prefix_hidden,
        fixture.input_hidden,
    )


def make_dense_case_with_prefix_lens(
    case: DenseAttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DenseAttentionCase:
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

    return DenseAttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        num_kv_heads=case.num_kv_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        sliding_window_size=case.sliding_window_size,
    )


def dense_fixture_inputs(fixture: DenseAttentionFixture) -> dict[str, Any]:
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


def make_dense_random_inputs(
    case: DenseAttentionCase,
    fixture: DenseAttentionFixture,
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


def make_dense_padded_replay_inputs(
    case: DenseAttentionCase,
    fixture: DenseAttentionFixture,
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


def make_dense_token_padded_inputs(
    _case: DenseAttentionCase,
    fixture: DenseAttentionFixture,
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


def prepare_dense_runner_inputs(
    fixture: DenseAttentionFixture,
    case: DenseAttentionCase,
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
        inputs["prefix_hidden"],
        max_context_len=max_context_len,
    )


def run_dense_forward(
    fixture: DenseAttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, Any],
) -> torch.Tensor:
    return fixture.actual_module(inputs["input_hidden"], batch)


def dense_attention_layers(fixture: DenseAttentionFixture) -> list[RadixAttention]:
    return [fixture.actual_module.attn]


def expected_dense_output_from_inputs(
    fixture: DenseAttentionFixture,
    case: DenseAttentionCase,
    inputs: dict[str, Any],
    _state,
) -> torch.Tensor:
    return _dense_attention_reference(
        fixture.reference_module,
        case,
        inputs["prefix_hidden"],
        inputs["input_hidden"],
    )


def run_dense_attention_case(
    testcase,
    case: DenseAttentionCase,
    *,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    loc_layout: str = "shuffled_pages",
):
    fixture = build_dense_attention_fixture(
        testcase,
        case,
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        loc_layout=loc_layout,
    )
    actual = run_dense_fixture_eager(fixture)
    expected = expected_dense_fixture_output(fixture)

    torch.testing.assert_close(actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL)

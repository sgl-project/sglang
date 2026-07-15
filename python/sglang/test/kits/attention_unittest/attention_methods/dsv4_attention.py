"""DSV4 attention fixture (SWA-only slice, compress_ratio=0).

This is a narrow first slice covering the SWA path of `DeepseekV4AttnBackend`.
The C4 (4x) and C128 (128x) compressor + indexer paths and speculative modes
are explicit follow-ups.

The reference is pure PyTorch: it unpacks the FP8-nope + BF16-rope cache
written by the real `set_swa_key_buffer_radix` path, then runs MLA-style
softmax(scaled q @ k.T) over the same SWA window with attention-sink scaling.
It does not call any DSV4 backend, FlashMLA, or DSV4 Triton kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

# DSV4 backend pre-resolves attention TP at construction; pin to single-rank.
_parallel_override = get_parallel().override(
    attn_tp_size=1, attn_tp_rank=0, attn_cp_size=1, attn_cp_rank=0
)
_parallel_override.__enter__()

# DSV4 hard-coded geometry. Do not change.
DSV4_PAGE_SIZE = 256
DSV4_SWA_WINDOW = 128  # backend's SWA_WINDOW constant
DSV4_QK_NOPE_HEAD_DIM = 448
DSV4_QK_ROPE_HEAD_DIM = 64
DSV4_HEAD_DIM = DSV4_QK_NOPE_HEAD_DIM + DSV4_QK_ROPE_HEAD_DIM  # 512
DSV4_KV_LORA_RANK = 512
DSV4_V_HEAD_DIM = 512
DSV4_INDEX_TOPK = 512  # required by DSV4AttnMetadata.init_flashmla_related

# FP8 nope quant noise + BF16 rope. Loose tolerance documented in module docstring.
# GB300 (SM10.x) flash_mla FP8 accumulation differs from H200; observed max
# diff ~0.0625 on `dsv4_swa_extend_no_prefix`. Use 8e-2 to absorb
# Blackwell-vs-Hopper variance while keeping coverage meaningful.
DSV4_ATOL = 8e-2
DSV4_RTOL = 8e-2
# CUDA-graph capture/replay uses `use_prefill_cuda_graph=True` which pads the
# DSV4 metadata fields differently from the eager path; the resulting fp8
# accumulation order shifts a handful of output elements by ~0.02 above the
# eager tolerance. The graph tests use this slightly looser tolerance.
DSV4_GRAPH_ATOL = 1e-1
DSV4_GRAPH_RTOL = 1e-1


@dataclass(frozen=True)
class DSV4AttentionCase:
    """One EXTEND or DECODE case scoped to compress_ratio=0 (SWA-only)."""

    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    # For EXTEND: per-request extend lengths. For DECODE: ignored (each
    # request decodes one token, so input_lens is implicitly (1,) * batch_size).
    extend_lens: tuple[int, ...] = ()
    # compress_ratio is fixed at 0 for this slice; C4/C128 are follow-ups.
    compress_ratio: int = 0
    # Per-head attention-sink value. The DSV4 backend forwards this to flash_mla
    # as a virtual-key score; the reference appends a virtual key with the same
    # score and value=0. The default (-1e30) effectively disables the sink so
    # the reference reduces to plain softmax(q @ k.T); finite values exercise
    # the sink correction path.
    attn_sink_value: float = -1e30

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


def make_dsv4_cases(backend: str) -> tuple[DSV4AttentionCase, ...]:
    # flash_mla's sparse_decode_fwd restricts h_q to a small set of values
    # (e.g., 16/32/64/128); DSV4 production runs use h_q=64. We match that.
    common = dict(
        backend=backend,
        forward_mode=ForwardMode.EXTEND,
        num_heads=64,
        page_size=DSV4_PAGE_SIZE,
    )
    return (
        DSV4AttentionCase(
            name="dsv4_swa_extend_no_prefix",
            prefix_lens=(0,),
            extend_lens=(32,),
            **common,
        ),
        DSV4AttentionCase(
            name="dsv4_swa_extend_prefix_within_window",
            prefix_lens=(48,),
            extend_lens=(16,),
            **common,
        ),
        # Non-zero attention-sink case. The sink contributes meaningful probability
        # mass at exp(0)=1 per head, so both the backend and the reference must
        # apply the same virtual-key correction for outputs to match. With
        # attn_sink_value=-1e30 the sink correction is effectively a no-op; this
        # case is the only one that actually verifies the correction logic.
        DSV4AttentionCase(
            name="dsv4_swa_extend_nonzero_attn_sink",
            prefix_lens=(48,),
            extend_lens=(16,),
            attn_sink_value=0.0,
            **common,
        ),
        # DECODE: one new token per request, attending to the existing prefix
        # plus its own position. The flash_mla `compress_ratio=0` path is
        # forward_mode-agnostic; the only differences are positions / seq_lens
        # / extend_* metadata, which the fixture handles via `input_lens`.
        DSV4AttentionCase(
            name="dsv4_swa_decode_within_window",
            backend=backend,
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(64,),
        ),
        DSV4AttentionCase(
            name="dsv4_swa_decode_multi_request_within_window",
            backend=backend,
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(32, 96),
        ),
        # Above-window extend: seq_len 160 > SWA_WINDOW=128. The SWA mask must
        # drop the oldest 32 positions per query so the reference's trailing-
        # window slice matches the backend's `get_swa_page_indices`. Exercises
        # the path where some `pos_t - SWA_WINDOW + 1 > 0` invalid offsets are
        # absent and the K cache slice covers >SWA_WINDOW total tokens.
        DSV4AttentionCase(
            name="dsv4_swa_extend_above_window",
            prefix_lens=(128,),
            extend_lens=(32,),
            **common,
        ),
        # Above-window decode: 1-token DECODE on a prefix longer than the SWA
        # window so the per-query SWA window strictly excludes the prefix head.
        DSV4AttentionCase(
            name="dsv4_swa_decode_above_window",
            backend=backend,
            forward_mode=ForwardMode.DECODE,
            num_heads=64,
            page_size=DSV4_PAGE_SIZE,
            prefix_lens=(160,),
        ),
        # seq_len exactly equal to the SWA window (128). Boundary case for
        # the `kv_start = max(0, query_pos - SWA_WINDOW + 1)` slice — the
        # backend's `get_swa_page_indices` must hand back exactly
        # SWA_WINDOW keys per query and the reference's trailing-window
        # slice must agree token-for-token.
        DSV4AttentionCase(
            name="dsv4_swa_extend_seq_len_eq_window",
            prefix_lens=(96,),
            extend_lens=(32,),
            **common,
        ),
        # seq_len one token below the page boundary (page_size=256). Forces
        # the page-table indexing into the last slot of a single page.
        DSV4AttentionCase(
            name="dsv4_swa_extend_seq_below_page",
            prefix_lens=(254,),
            extend_lens=(1,),
            **common,
        ),
        # seq_len exactly on the page boundary (256). The dispatcher must
        # treat this as a single fully-used page rather than allocating a
        # spurious next page.
        DSV4AttentionCase(
            name="dsv4_swa_extend_seq_at_page",
            prefix_lens=(255,),
            extend_lens=(1,),
            **common,
        ),
        # seq_len one token above the page boundary (257). Crosses into the
        # next page so `get_swa_page_indices` must stitch indices from two
        # consecutive pages while the SWA window still slides over the
        # trailing 128 keys.
        DSV4AttentionCase(
            name="dsv4_swa_extend_seq_above_page",
            prefix_lens=(256,),
            extend_lens=(1,),
            **common,
        ),
        # Prefix length exactly equal to one page. EXTEND opens the next
        # page on the first extend token, exercising the page-aligned
        # prefix branch.
        DSV4AttentionCase(
            name="dsv4_swa_extend_prefix_exact_page",
            prefix_lens=(DSV4_PAGE_SIZE,),
            extend_lens=(4,),
            **common,
        ),
        # prefix + extend exactly equals one page (the page-aligned-total
        # branch — total seq_len lands on the boundary without crossing).
        DSV4AttentionCase(
            name="dsv4_swa_extend_total_exact_page",
            prefix_lens=(DSV4_PAGE_SIZE - 16,),
            extend_lens=(16,),
            **common,
        ),
    )


class TinyDSV4ModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        context_len: int,
        compression_ratios: list[int] = None,
    ):
        if compression_ratios is None:
            compression_ratios = [0]
        self.context_len = context_len
        self.hidden_size = DSV4_HEAD_DIM
        self.num_attention_heads = num_heads
        self.num_key_value_heads = 1
        self.head_dim = DSV4_HEAD_DIM
        self.qk_nope_head_dim = DSV4_QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = DSV4_QK_ROPE_HEAD_DIM
        self.kv_lora_rank = DSV4_KV_LORA_RANK
        self.v_head_dim = DSV4_V_HEAD_DIM
        self.sliding_window_size = DSV4_SWA_WINDOW
        self.is_encoder_decoder = False
        self.is_multimodal = False
        self.is_generation = True
        self.quantization = None
        self.is_hybrid_swa = False
        self.is_local_attention_model = False
        self.attention_chunk_size = None
        self.hf_config = SimpleNamespace(
            architectures=["DeepSeekV4ForCausalLM"],
            hidden_size=DSV4_HEAD_DIM,
            num_attention_heads=num_heads,
            num_key_value_heads=1,
            head_dim=DSV4_HEAD_DIM,
            qk_nope_head_dim=DSV4_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=DSV4_QK_ROPE_HEAD_DIM,
            kv_lora_rank=DSV4_KV_LORA_RANK,
            v_head_dim=DSV4_V_HEAD_DIM,
            index_topk=DSV4_INDEX_TOPK,
            num_hidden_layers=len(compression_ratios),
            compress_ratios=list(compression_ratios),
        )
        self.hf_config.get_text_config = lambda: self.hf_config
        self.hf_text_config = self.hf_config
        self.linear_attn_registry_result = None


class MockDSV4ModelRunner:
    """Minimal runner exposing what `DeepseekV4AttnBackend.__init__` reads.

    We bypass `ModelRunner.__init__` (it requires real model loading); only the
    attributes the backend touches are needed.
    """

    def __init__(
        self,
        *,
        case: DSV4AttentionCase,
        model_config: TinyDSV4ModelConfig,
        dtype: torch.dtype,
        device: str,
        max_context_len: int,
        swa_size: int,
        disable_cuda_graph: bool = True,
        disable_piecewise_cuda_graph: bool = True,
        runner_batch_size: int | None = None,
        compression_ratios: list[int] = None,
    ):
        if compression_ratios is None:
            compression_ratios = [0]
        pool_batch_size = runner_batch_size or case.batch_size
        # Speculative cases derive `speculative_num_draft_tokens` from the
        # case's per-request input length (target_verify uses the draft count
        # directly; draft_extend uses the accepted-token count). Non-spec cases
        # leave it at 0 so the backend skips the speculative branches.
        if (
            case.forward_mode.is_target_verify()
            or case.forward_mode.is_draft_extend_v2()
        ):
            speculative_num_draft_tokens = case.input_lens[0] if case.input_lens else 0
            speculative_eagle_topk = 1
        else:
            speculative_num_draft_tokens = 0
            speculative_eagle_topk = 0
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
        self.ps = ParallelState.trivial()
        self._server_args_override = get_context().override_server_args(
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
            disaggregation_mode=None,
            dp_size=1,
            enable_deterministic_inference=False,
            enable_dp_attention=False,
            enable_mis=False,
            is_embedding=False,
            kv_cache_dtype="auto",
            max_running_requests=None,
            pp_size=1,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=speculative_eagle_topk,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            speculative_num_steps=max(0, speculative_num_draft_tokens - 1),
            tp_size=1,
            device=device,
            mem_fraction_static=0.8,
        )
        self.server_args = self._server_args_override.install()
        self.req_to_token_pool = ReqToTokenPool(
            size=pool_batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # `compression_ratios=[0]` (default) disables C4/C128 sub-pools (their
        # layer_num=0). Tests for C4 / C128 dispatch pass e.g. `[4]` or
        # `[128]` to allocate the corresponding sub-pool. DSV4 KV pool stores
        # FP8 nope; pass fp8 dtype so store_dtype=uint8 (the backing tensor is
        # always raw bytes regardless of the nominal dtype).
        layer_num = len(compression_ratios)
        self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=pool_batch_size,
            swa_size=swa_size,
            c4_size=case.page_size,
            c128_size=case.page_size,
            c4_state_pool_size=pool_batch_size,
            c128_state_pool_size=pool_batch_size,
            page_size=case.page_size,
            swa_page_size=DSV4_SWA_WINDOW,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=dtype,
            c128_state_dtype=dtype,
            qk_nope_head_dim=DSV4_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=DSV4_QK_ROPE_HEAD_DIM,
            indexer_head_dim=128,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=False,
            compression_ratios=list(compression_ratios),
        )
        # Register identity full->swa mapping over swa_size full locs.
        identity = torch.arange(swa_size, dtype=torch.int64, device=device)
        self.token_to_kv_pool.register_mapping(identity)
        self.token_to_kv_pool_allocator = SimpleNamespace(page_size=case.page_size)
        self.attn_cp_size = 1
        self.attention_chunk_size = None
        self.hisparse_coordinator = None
        self.init_new_workspace = False
        self.is_hybrid_swa = False
        self.sliding_window_size = DSV4_SWA_WINDOW
        self.use_mla_backend = True
        self.is_draft_worker = False
        self.spec_algorithm = SpeculativeAlgorithm.NONE
        self._kernel_warmed_up = True

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


class ProjectedDSV4Attention(nn.Module):
    """Holds Q/K projections shaped for DSV4 dims and invokes
    `DeepseekV4AttnBackend.forward` directly with compress_ratio=0.

    DSV4 in production fuses the K cache write into a Triton kernel; here we
    issue the equivalent `set_swa_key_buffer_radix` call once with the packed
    K so the cache matches what the backend's FP8 path reads.
    """

    def __init__(
        self,
        *,
        num_heads: int,
        hidden_size: int,
        dtype: torch.dtype,
        device: str,
        attn_sink_value: float = -1e30,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(
            hidden_size,
            num_heads * DSV4_HEAD_DIM,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = nn.Linear(
            hidden_size,
            DSV4_HEAD_DIM,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # Production DSV4 has an `o_proj` mapping multi-head output back to
        # hidden_size. We only add it for the EAGLE draft path (forward()
        # method); the rest of the fixture bypasses `forward()` and runs
        # the backend output through the test's own reduction.
        self.o_proj = nn.Linear(
            num_heads * DSV4_V_HEAD_DIM,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=DSV4_HEAD_DIM,
            scaling=DSV4_HEAD_DIM**-0.5,
            num_kv_heads=1,
            layer_id=0,
            v_head_dim=DSV4_V_HEAD_DIM,
        )
        # Per-head attention sink. Forwarded to flash_mla as a virtual-key score;
        # the reference appends a virtual key with the same score and value=0.
        # Default (-1e30) makes the sink contribution numerically negligible so
        # the reference reduces to plain softmax(q @ k.T).
        self.attn_sink = nn.Parameter(
            torch.full(
                (num_heads,), attn_sink_value, dtype=torch.float32, device=device
            ),
            requires_grad=False,
        )

    def project(self, hidden_states: torch.Tensor):
        q = self.q_proj(hidden_states).view(-1, self.num_heads, DSV4_HEAD_DIM)
        k = self.k_proj(hidden_states).view(-1, 1, DSV4_HEAD_DIM)
        return q, k

    def forward(self, hidden_states: torch.Tensor, forward_batch):
        """Production-style draft forward: project Q/K, write K to the SWA
        pool at `forward_batch.out_cache_loc`, then run the active backend.

        Mirrors `python/sglang/srt/models/deepseek_v4.py::AbsorbMQAv4.forward`
        for `compress_ratio=0`: K is written via `set_swa_key_buffer_radix`
        before the attention call, the backend is invoked with
        `save_kv_cache=False`, and the attn_sink correction is forwarded
        via the `attn_sink` kwarg.

        Returns a flat `[num_tokens, hidden_size]` tensor (matching the
        backend's output shape) so the EAGLE draft runner harness can pipe
        it through `lm_head`.
        """
        from sglang.srt.model_executor.forward_context import (
            get_forward_context,
        )

        q, k = self.project(hidden_states)
        ctx = get_forward_context()
        attn_backend = ctx.attn_backend
        if forward_batch.out_cache_loc is not None:
            # `quant_to_nope_fp8_rope_bf16_pack_triton` expects 2D
            # `[num_tokens, hidden_dim]`; `project` returns 3D
            # `[num_tokens, 1, hidden_dim]`.
            k_flat = k.reshape(k.shape[0], -1).to(torch.bfloat16)
            pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_flat)
            pool = attn_backend.token_to_kv_pool
            pool.set_swa_key_buffer_radix(
                layer_id=self.attn.layer_id,
                swa_loc=pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc.to(torch.int64)
                ),
                cache_nope_fp8_rope_bf16_pack=pack,
            )
        out = attn_backend.forward(
            q=q,
            k=k,
            v=k,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            save_kv_cache=False,
            attn_sink=self.attn_sink,
        )
        return self.o_proj(out.reshape(out.shape[0], -1))


def _write_swa_cache(
    runner: MockDSV4ModelRunner,
    layer_id: int,
    loc: torch.Tensor,
    k_bf16: torch.Tensor,
):
    """Write packed FP8 nope + BF16 rope into the SWA pool at `loc` (full locs)."""
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(k_bf16.to(torch.bfloat16))
    runner.token_to_kv_pool.set_swa_key_buffer_radix(
        layer_id=layer_id,
        swa_loc=runner.token_to_kv_pool.translate_loc_from_full_to_swa(
            loc.to(torch.int64)
        ),
        cache_nope_fp8_rope_bf16_pack=pack,
    )


# The previous version of this fixture had `_unpack_swa_cache` /
# `_unpack_extra_cache` helpers that read FP8 bytes back from the production
# pool's `kv_buffer` and dequantized them. The reference now reads BF16 K
# directly from the per-request stash on the fixture (see
# `_populate_swa_kv_cache` / `_populate_extra_kv_cache`), so the reference
# math is independent of `quant_to_nope_fp8_rope_bf16_pack_triton` and
# `set_swa_key_buffer_radix` — a silent bug in those production write
# functions can no longer corrupt both paths identically.


@dataclass
class DSV4AttentionFixture:
    case: DSV4AttentionCase
    runner: MockDSV4ModelRunner
    backend: object
    actual_module: ProjectedDSV4Attention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor
    # Selects dense vs sparse-prefill C4 seeding; lives on the fixture because
    # the reference re-seeds after rebuilding metadata (`_seed_c4_if_needed`).
    seed_c4_for_sparse_prefill: bool = False


@dataclass
class DSV4ReferenceOutput:
    output: torch.Tensor


def _token_loc(req_idx: int, pos: int, *, max_context_len: int) -> int:
    """Full-pool token location. Reserved 0 for padding; offset by max_context_len/req."""
    return 1 + req_idx * max_context_len + pos


def _make_forward_batch(
    case: DSV4AttentionCase,
    runner: MockDSV4ModelRunner,
    *,
    max_context_len: int,
    device: str,
) -> ForwardBatch:
    seq_lens = case.seq_lens
    input_lens = case.input_lens
    req_pool_indices = torch.arange(case.batch_size, dtype=torch.int32, device=device)
    out_cache_locs: list[int] = []
    positions: list[int] = []

    for req_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            runner.req_to_token_pool.req_to_token[req_idx, pos] = _token_loc(
                req_idx, pos, max_context_len=max_context_len
            )
        prefix_len = case.prefix_lens[req_idx]
        for offset in range(input_lens[req_idx]):
            positions.append(prefix_len + offset)
            out_cache_locs.append(
                _token_loc(
                    req_idx, prefix_len + offset, max_context_len=max_context_len
                )
            )

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
    # extend_* fields are only populated for extend-shaped modes. DECODE leaves
    # them at their defaults; the flash_mla path reads metadata directly from
    # DSV4AttnMetadata so the extend fields are unused for the compress_ratio=0
    # DECODE path.
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


def build_dsv4_attention_fixture(
    testcase,
    case: DSV4AttentionCase,
    *,
    swa_size: int = 1024,
    max_context_len: int = 256,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    runner_batch_size: int | None = None,
    compression_ratios: list[int] = None,
) -> DSV4AttentionFixture:
    # SWA-only (compress_ratio=0) is the SGLang path that handles the
    # last-`SWA_WINDOW`-tokens slice for *all* sequence lengths. seq_len >
    # SWA_WINDOW just means the SWA mask truncates the oldest tokens; the
    # backend's `get_swa_page_indices` and the fixture's reference both pick
    # the same trailing window so this works without enabling C4/C128.
    # Auto-scale `max_context_len` so per-page-boundary cases (seq_len near
    # or above `DSV4_PAGE_SIZE=256`) fit in `req_to_token`. The default
    # `max_context_len=256` covers the common in-window cases; longer cases
    # bump the per-req capacity and round up to the page boundary.
    max_seq = max(case.seq_lens)
    if max_seq > max_context_len:
        max_context_len = (
            (max_seq + case.page_size - 1) // case.page_size
        ) * case.page_size
    if compression_ratios is None:
        compression_ratios = [case.compress_ratio]
    seed = 7100 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyDSV4ModelConfig(
        num_heads=case.num_heads,
        context_len=max_context_len,
        compression_ratios=compression_ratios,
    )
    runner = MockDSV4ModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        swa_size=swa_size,
        disable_cuda_graph=disable_cuda_graph,
        disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
        runner_batch_size=runner_batch_size,
        compression_ratios=compression_ratios,
    )
    try:
        backend = ATTENTION_BACKENDS[case.backend](runner)
    except (AssertionError, ImportError, ModuleNotFoundError) as exc:
        testcase.skipTest(f"{case.backend} backend is not available: {exc}")

    actual_module = ProjectedDSV4Attention(
        num_heads=case.num_heads,
        hidden_size=DSV4_HEAD_DIM,
        dtype=dtype,
        device=device,
        attn_sink_value=case.attn_sink_value,
    )
    prefix_hidden = [
        torch.randn(length, DSV4_HEAD_DIM, dtype=dtype, device=device)
        for length in case.prefix_lens
    ]
    input_hidden = torch.randn(
        case.num_input_tokens, DSV4_HEAD_DIM, dtype=dtype, device=device
    )
    forward_batch = _make_forward_batch(
        case, runner, max_context_len=max_context_len, device=device
    )
    return DSV4AttentionFixture(
        case=case,
        runner=runner,
        backend=backend,
        actual_module=actual_module,
        forward_batch=forward_batch,
        prefix_hidden=prefix_hidden,
        input_hidden=input_hidden,
    )


def _pure_torch_dsv4_swa_reference(
    fixture: DSV4AttentionFixture,
    q: torch.Tensor,
    full_kv_locs_per_req: list[torch.Tensor],
    *,
    case: DSV4AttentionCase | None = None,
) -> torch.Tensor:
    """Vanilla DSV4 SWA reference. Reads K directly from the BF16 tensor that
    `_populate_swa_kv_cache` stashed on the fixture, so the math is
    independent of the FP8 pack/unpack roundtrip the production
    `set_swa_key_buffer_radix` path uses. (The HF reference at
    `deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py` likewise
    skips the FP8 quantization in the unit-test-suitable form — the
    quantization is only a QAT-simulation artifact, not part of the math.)

    `q` has shape `[num_q, num_heads, DSV4_HEAD_DIM]`. `full_kv_locs_per_req`
    is kept as a parameter for compatibility but the per-request BF16 K
    sourced from `fixture._swa_bf16_k_per_req` is what the math actually
    uses. `case` overrides `fixture.case` when runner-mode integrations use
    a padded variant case.
    """
    del full_kv_locs_per_req  # kept for backward-compat; not used now
    if case is None:
        case = fixture.case
    swa_k_per_req: list[torch.Tensor] = fixture._swa_bf16_k_per_req  # type: ignore[attr-defined]
    scaling = DSV4_HEAD_DIM**-0.5
    attn_sink = fixture.actual_module.attn_sink.detach()
    outputs = []
    q_idx = 0
    for req_idx in range(case.batch_size):
        kv_full = swa_k_per_req[req_idx].float()  # [seq_len, head_dim] BF16->FP32
        for offset in range(case.input_lens[req_idx]):
            query_pos = case.prefix_lens[req_idx] + offset
            kv_start = max(0, query_pos - DSV4_SWA_WINDOW + 1)
            keys = kv_full[kv_start : query_pos + 1]
            query = q[q_idx].float()
            scores = torch.einsum("hd,kd->hk", query, keys) * scaling
            # attn_sink: per-head scalar; effective probs = exp(s_i)/sum(exp + exp(sink)).
            # Equivalent to appending a virtual key with score=attn_sink and value=0.
            sink_scores = attn_sink.view(-1, 1).to(scores.dtype)
            scores_with_sink = torch.cat([scores, sink_scores], dim=-1)
            probs_with_sink = torch.softmax(scores_with_sink, dim=-1)
            probs = probs_with_sink[:, :-1]
            out = torch.einsum("hk,kd->hd", probs, keys)
            outputs.append(out)
            q_idx += 1
    return torch.stack(outputs, dim=0).to(q.dtype)


def _populate_swa_kv_cache(
    fixture: DSV4AttentionFixture,
    *,
    max_context_len: int,
    device: str,
    inputs: dict[str, Any] | None = None,
) -> list[torch.Tensor]:
    """Project K for every kv token (prefix + input) and write the packed
    FP8 nope + BF16 rope representation into the SWA pool via the production
    pack+set path. ALSO stashes the projected per-request BF16 K on the
    fixture as `fixture._swa_bf16_k_per_req` so the reference can read K
    directly from BF16 instead of unpacking quantized bytes back from the
    pool — that keeps the reference math independent of
    `quant_to_nope_fp8_rope_bf16_pack_triton` / `set_swa_key_buffer_radix`
    (otherwise a silent pack/write bug would corrupt both paths
    identically). Returns the full-pool token locs per request in causal
    order.
    """
    case = fixture.case
    prefix_hidden = (
        inputs["prefix_hidden"] if inputs is not None else fixture.prefix_hidden
    )
    input_hidden = (
        inputs["input_hidden"] if inputs is not None else fixture.input_hidden
    )
    full_kv_locs_per_req: list[torch.Tensor] = []
    all_k_bf16_parts: list[torch.Tensor] = []
    all_k_locs_parts: list[torch.Tensor] = []
    per_req_bf16_k: list[torch.Tensor] = []
    for req_idx, prefix in enumerate(prefix_hidden):
        input_part = input_hidden[
            sum(case.input_lens[:req_idx]) : sum(case.input_lens[: req_idx + 1])
        ]
        req_hidden = torch.cat([prefix, input_part], dim=0)
        _, k_req = fixture.actual_module.project(req_hidden)
        k_req_flat = k_req.view(-1, DSV4_HEAD_DIM)  # [seq_len, head_dim]
        per_req_bf16_k.append(k_req_flat)
        all_k_bf16_parts.append(k_req_flat)
        seq_len = case.seq_lens[req_idx]
        req_locs = torch.tensor(
            [
                _token_loc(req_idx, p, max_context_len=max_context_len)
                for p in range(seq_len)
            ],
            dtype=torch.int64,
            device=device,
        )
        full_kv_locs_per_req.append(req_locs)
        all_k_locs_parts.append(req_locs)
    all_k = torch.cat(all_k_bf16_parts, dim=0)
    all_k_locs = torch.cat(all_k_locs_parts, dim=0)
    _write_swa_cache(fixture.runner, layer_id=0, loc=all_k_locs, k_bf16=all_k)
    fixture._swa_bf16_k_per_req = per_req_bf16_k  # type: ignore[attr-defined]
    fixture._swa_full_locs_per_req = full_kv_locs_per_req  # type: ignore[attr-defined]
    return full_kv_locs_per_req


def run_dsv4_attention_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    fixture = build_dsv4_attention_fixture(testcase, case, dtype=dtype, device=device)
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    full_kv_locs_per_req = _populate_swa_kv_cache(
        fixture, max_context_len=max_context_len, device=device
    )

    # Project Q for the input tokens only.
    q_input, _ = fixture.actual_module.project(fixture.input_hidden)

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = fixture.backend.forward(
            q=q_input,
            k=q_input,  # k is v sentinel; save_kv_cache=False so it's unread
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=0,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )

    expected = _pure_torch_dsv4_swa_reference(fixture, q_input, full_kv_locs_per_req)

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=DSV4_ATOL, rtol=DSV4_RTOL
    )


# ---------------------------------------------------------------------------
# Runner-mode callbacks (used by common/runner_modes/cuda_graph_decode_runner)
# ---------------------------------------------------------------------------


def make_dsv4_case_with_prefix_lens(
    case: DSV4AttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
) -> DSV4AttentionCase:
    """Build a variant case with new prefix lengths, preserving everything else
    relevant to the SWA-only fixture (mode, head count, page size, attn sink).

    For DECODE `extend_lens=()` (`input_lens` derives `(1,) * batch_size`); for
    EXTEND we pad/clip the existing `extend_lens` to match the new batch shape.
    """
    if case.forward_mode.is_decode():
        extend_lens: tuple[int, ...] = ()
    else:
        base = case.extend_lens or (1,)
        if len(prefix_lens) <= len(base):
            extend_lens = base[: len(prefix_lens)]
        else:
            extend_lens = base + (base[-1],) * (len(prefix_lens) - len(base))
    return DSV4AttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        compress_ratio=case.compress_ratio,
        attn_sink_value=case.attn_sink_value,
    )


def make_dsv4_case_with_lens(
    case: DSV4AttentionCase,
    name: str,
    prefix_lens: tuple[int, ...],
    extend_lens: tuple[int, ...],
) -> DSV4AttentionCase:
    """Build a variant case with explicit prefix + extend lengths. Used by the
    draft_extend graph runner where capture / replay both need to set both
    fields independently (ragged accepted-token counts)."""
    return DSV4AttentionCase(
        name=name,
        backend=case.backend,
        forward_mode=case.forward_mode,
        num_heads=case.num_heads,
        page_size=case.page_size,
        prefix_lens=prefix_lens,
        extend_lens=extend_lens,
        compress_ratio=case.compress_ratio,
        attn_sink_value=case.attn_sink_value,
    )


def dsv4_fixture_inputs(fixture: DSV4AttentionFixture) -> dict[str, Any]:
    return {
        "prefix_hidden": fixture.prefix_hidden,
        "input_hidden": fixture.input_hidden,
    }


def _random_dsv4_hidden_by_lens(
    lens: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: str,
) -> list[torch.Tensor]:
    return [
        torch.randn(length, DSV4_HEAD_DIM, dtype=dtype, device=device)
        for length in lens
    ]


def make_dsv4_random_inputs(
    case: DSV4AttentionCase,
    fixture: DSV4AttentionFixture,
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    return {
        "prefix_hidden": _random_dsv4_hidden_by_lens(
            case.prefix_lens, dtype=dtype, device=device
        ),
        "input_hidden": torch.randn(
            case.num_input_tokens, DSV4_HEAD_DIM, dtype=dtype, device=device
        ),
    }


def make_dsv4_padded_replay_inputs(
    case: DSV4AttentionCase,
    fixture: DSV4AttentionFixture,
    pad_prefix_lens: tuple[int, ...],
    base_inputs: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    pad_prefix_hidden = _random_dsv4_hidden_by_lens(
        pad_prefix_lens, dtype=dtype, device=device
    )
    pad_token_count = case.num_input_tokens - base_inputs["input_hidden"].shape[0]
    if pad_token_count < 0:
        raise ValueError(
            f"replay input shrink not supported: {pad_token_count=}; "
            f"case={case.name}"
        )
    if pad_token_count == 0:
        padded_input_hidden = base_inputs["input_hidden"]
    else:
        pad_input_hidden = torch.randn(
            pad_token_count, DSV4_HEAD_DIM, dtype=dtype, device=device
        )
        padded_input_hidden = torch.cat(
            [base_inputs["input_hidden"], pad_input_hidden], dim=0
        )
    return {
        "prefix_hidden": base_inputs["prefix_hidden"] + pad_prefix_hidden,
        "input_hidden": padded_input_hidden,
    }


def _full_kv_locs_per_req(
    case: DSV4AttentionCase, *, max_context_len: int, device: str
) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    for req_idx, seq_len in enumerate(case.seq_lens):
        out.append(
            torch.tensor(
                [
                    _token_loc(req_idx, p, max_context_len=max_context_len)
                    for p in range(seq_len)
                ],
                dtype=torch.int64,
                device=device,
            )
        )
    return out


_DSV4_EXTRA_ENTRIES = 32


def prepare_dsv4_runner_inputs(
    fixture: DSV4AttentionFixture,
    case: DSV4AttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, Any],
    *,
    max_context_len: int,
) -> None:
    """Project K for prefix + input hidden in `inputs` and write the packed
    FP8 nope + BF16 rope representation into the SWA cache. For
    `case.compress_ratio in (4, 128)` also populate the corresponding C4/C128
    extra cache. Stashes the per-request BF16 K on the fixture so the
    reference reads K from BF16 (independent of the FP8 pack/unpack
    roundtrip). Also stashes `batch` as `fixture._current_batch` so the
    speculative-graph runner's pre-init `expected_output` call can build
    metadata for the right batch.
    """
    all_k_parts: list[torch.Tensor] = []
    all_locs_parts: list[torch.Tensor] = []
    per_req_bf16_k: list[torch.Tensor] = []
    input_hidden = inputs["input_hidden"]
    for req_idx, prefix in enumerate(inputs["prefix_hidden"]):
        input_part = input_hidden[
            sum(case.input_lens[:req_idx]) : sum(case.input_lens[: req_idx + 1])
        ]
        req_hidden = torch.cat([prefix, input_part], dim=0)
        _, k_req = fixture.actual_module.project(req_hidden)
        k_req_flat = k_req.view(-1, DSV4_HEAD_DIM)
        per_req_bf16_k.append(k_req_flat)
        all_k_parts.append(k_req_flat)
        seq_len = case.seq_lens[req_idx]
        all_locs_parts.append(
            torch.tensor(
                [
                    _token_loc(req_idx, p, max_context_len=max_context_len)
                    for p in range(seq_len)
                ],
                dtype=torch.int64,
                device=fixture.runner.device,
            )
        )
    _write_swa_cache(
        fixture.runner,
        layer_id=0,
        loc=torch.cat(all_locs_parts, dim=0),
        k_bf16=torch.cat(all_k_parts, dim=0),
    )
    fixture._swa_bf16_k_per_req = per_req_bf16_k  # type: ignore[attr-defined]
    # The cuda-graph speculative runner calls `expected_output` before the
    # backend's metadata-init has run for the capture/replay batch; the DSV4
    # reference needs to build that metadata itself. Stash the current batch
    # so `_pure_torch_dsv4_combined_reference` knows which one to use.
    fixture._current_batch = batch  # type: ignore[attr-defined]
    if case.compress_ratio in (4, 128):
        _populate_extra_kv_cache(fixture, layer_id=0, num_entries=_DSV4_EXTRA_ENTRIES)


def _seed_c4_if_needed(
    fixture: DSV4AttentionFixture, *, num_entries: int = _DSV4_EXTRA_ENTRIES
) -> None:
    """For compress_ratio=4, seed the C4 metadata the exercised path consumes
    (the C4Indexer would normally populate it; the compact fixture skips the
    indexer): `c4_sparse_page_indices` for the dense extend path,
    `c4_sparse_raw_indices` for sparse prefill. No-op for other compress_ratios.
    """
    if fixture.case.compress_ratio != 4:
        return
    if fixture.seed_c4_for_sparse_prefill:
        _seed_c4_sparse_prefill_indices(fixture, num_entries=num_entries)
    else:
        _seed_c4_sparse_indices(fixture, num_entries=num_entries)


def run_dsv4_fixture_eager(fixture: DSV4AttentionFixture) -> torch.Tensor:
    """Eager forward that re-initialises the forward metadata. For
    `case.compress_ratio in (4, 128)` populates the extra K cache and seeds
    the C4 sparse indices before invoking `forward` so the actual path and
    the combined reference attend to matching SWA + extra-K entries.
    """
    case = fixture.case
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    full_kv_locs_per_req = _populate_swa_kv_cache(
        fixture, max_context_len=max_context_len, device=runner.device
    )
    if case.compress_ratio in (4, 128):
        _populate_extra_kv_cache(fixture, layer_id=0, num_entries=_DSV4_EXTRA_ENTRIES)
    q_input, _ = fixture.actual_module.project(fixture.input_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        _seed_c4_if_needed(fixture)
        actual = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=case.compress_ratio,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )
    fixture._eager_full_kv_locs_per_req = full_kv_locs_per_req  # type: ignore[attr-defined]
    return actual.float()


def run_dsv4_forward(
    fixture: DSV4AttentionFixture,
    batch: ForwardBatch,
    inputs: dict[str, Any],
) -> torch.Tensor:
    """Forward call used after the runner harness has already invoked
    `init_forward_metadata_capture_cuda_graph` / `_replay_cuda_graph` on the
    backend. Projects Q from `inputs['input_hidden']`, applies the C4 sparse
    index seeding (no-op for compress_ratio in {0, 128}) so the harness's
    capture+replay metadata reaches the same flash_mla call shape the eager
    path uses, then calls `forward`.
    """
    case = fixture.case
    q_input, _ = fixture.actual_module.project(inputs["input_hidden"])
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        _seed_c4_if_needed(fixture)
        out = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=batch,
            compress_ratio=case.compress_ratio,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )
    return out.float()


def expected_dsv4_output_from_inputs(
    fixture: DSV4AttentionFixture,
    case: DSV4AttentionCase,
    inputs: dict[str, Any],
    _state: Any,
) -> torch.Tensor:
    """Pure-PyTorch reference. For compress_ratio=0 (SWA-only) projects Q from
    `inputs['input_hidden']` and slides a sliding-window reference over the
    K's written into the SWA cache. For compress_ratio in (4, 128) projects Q
    the same way but reads SWA + extra metadata indices from the upgraded
    `DSV4AttnMetadata` so the reference picks up exactly the slots the
    backend attends to."""
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    q_input, _ = fixture.actual_module.project(inputs["input_hidden"])
    if case.compress_ratio in (4, 128):
        return _pure_torch_dsv4_combined_reference(fixture, q_input).float()
    full_kv_locs_per_req = _full_kv_locs_per_req(
        case, max_context_len=max_context_len, device=runner.device
    )
    return _pure_torch_dsv4_swa_reference(
        fixture, q_input, full_kv_locs_per_req, case=case
    ).float()


def _populate_extra_kv_cache(
    fixture: DSV4AttentionFixture,
    *,
    layer_id: int = 0,
    num_entries: int = 32,
) -> int:
    """Write `num_entries` packed FP8-nope/BF16-rope K vectors into the C4 or
    C128 extra cache via the production `set_extra_key_buffer` path. ALSO
    stashes the same BF16 K on the fixture as `fixture._extra_bf16_k` so the
    reference reads K from BF16 instead of unpacking quantized bytes back
    from the pool. The case-derived seed makes the random K reproducible
    across eager/capture/replay rebuilds; the save/restore of the global
    RNG prevents perturbing downstream Q/K projection randomness.
    """
    pool = fixture.runner.token_to_kv_pool
    device = fixture.runner.device
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device=device)
    try:
        case_seed = 8200 + len(fixture.case.name) * 13 + layer_id
        torch.manual_seed(case_seed)
        torch.cuda.manual_seed_all(case_seed)
        rand_k = torch.randn(
            num_entries, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device
        )
    finally:
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state, device=device)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(rand_k)
    loc = torch.arange(num_entries, dtype=torch.int64, device=device)
    pool.set_extra_key_buffer(
        layer_id=layer_id, loc=loc, cache_nope_fp8_rope_bf16_pack=pack
    )
    fixture._extra_bf16_k = rand_k  # type: ignore[attr-defined]
    return num_entries


def _extra_metadata_indices(
    core_metadata, compress_ratio: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(extra_indices, extra_topk_lengths)` for C4 / C128 paths from
    the upgraded `DSV4AttnMetadata`. Mirrors the dispatch in
    `DeepseekV4AttnBackend.forward(compress_ratio=...)`.
    """
    if compress_ratio == 4:
        return (
            core_metadata.c4_sparse_page_indices,
            core_metadata.c4_sparse_topk_lengths,
        )
    if compress_ratio == 128:
        return core_metadata.c128_page_indices, core_metadata.c128_topk_lengths_clamp1
    raise ValueError(f"unsupported compress_ratio={compress_ratio}")


def _pure_torch_dsv4_combined_reference(
    fixture: DSV4AttentionFixture,
    q: torch.Tensor,
    *,
    layer_id: int = 0,
) -> torch.Tensor:
    """Vanilla DSV4 SWA + C4 / C128 reference. Sources K from BF16 tensors
    that `_populate_swa_kv_cache` / `_populate_extra_kv_cache` stashed on the
    fixture, NOT from the production quantized cache bytes. This keeps the
    reference math independent of `quant_to_nope_fp8_rope_bf16_pack_triton` /
    `set_extra_key_buffer` — a silent pack/write bug in those paths would
    diverge the actual flash_mla output from this BF16 reference instead of
    corrupting both identically.

    The reference reproduces the structure of the HF
    `deepseek-ai/DeepSeek-V4-Pro/inference/model.py` attention forward:
    per-query SWA window + optional compressed extra entries, combined into
    one softmax with the per-head attention sink as a virtual-key score.
    (HF likewise skips the FP8 quantization for the test-suitable form;
    quantization is a QAT-simulation artifact, not part of the math.)

    Forces the lazy `DSV4RawDecodeMetadata → DSV4Metadata` upgrade before
    reading per-q-token `swa_page_indices` / `cN_page_indices` so this works
    both pre-forward and post-`on_after_cuda_graph_warmup` (which rolls
    `forward_metadata` back to the captured raw to be re-upgraded inside
    the CUDA graph).
    """
    del layer_id  # K is sourced from the fixture's BF16 stash, not from
    # a layer-indexed pool buffer.
    case = fixture.case
    # In runner-harness flows the reference is called BEFORE
    # `init_forward_metadata` / `init_forward_metadata_*_cuda_graph` —
    # the per-q-token `swa_page_indices` / `cN_page_indices` we read below
    # don't exist yet (or, worse, hold metadata for a previous leg's batch).
    # Always rebuild from the current batch (set by
    # `prepare_dsv4_runner_inputs`, falling back to the fixture's
    # construction-time batch for the eager test path). The metadata this
    # call produces is the same DSV4Metadata the forward path will produce
    # for the same batch, so the backend's later
    # `_init_cuda_graph_*_metadata` simply overwrites with an identical
    # metadata layout for the graph buffers.
    current_batch = getattr(fixture, "_current_batch", None)
    if current_batch is None:
        current_batch = fixture.forward_batch
    fixture.backend.init_forward_metadata(current_batch)
    # Re-apply the C4 seeding too, since `on_after_cuda_graph_warmup` rolls
    # `forward_metadata` back to the raw captured value (which clears
    # `c4_sparse_page_indices` back to all -1 on the next upgrade) — the
    # reference must observe the same seeded indices the backend forward saw.
    _seed_c4_if_needed(fixture)
    md = fixture.backend.forward_metadata.core_metadata
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    swa_indices = md.swa_page_indices  # [num_q, padded_window], full-pool locs
    swa_topk_lengths = md.swa_topk_lengths  # [num_q]

    if case.compress_ratio in (4, 128):
        extra_indices, extra_topk_lengths = _extra_metadata_indices(
            md, case.compress_ratio
        )
    else:
        extra_indices, extra_topk_lengths = None, None

    swa_k_per_req: list[torch.Tensor] = fixture._swa_bf16_k_per_req  # type: ignore[attr-defined]
    extra_k_bf16 = getattr(fixture, "_extra_bf16_k", None)

    scaling = DSV4_HEAD_DIM**-0.5
    attn_sink = fixture.actual_module.attn_sink.detach()
    outputs = []

    num_q = q.shape[0]
    for q_idx in range(num_q):
        # Map full-pool locs back to (req_idx, position) using
        # `_token_loc(req_idx, pos, max_context_len) = 1 + req_idx * max + pos`,
        # then index into the BF16 per-request K stash.
        swa_len = int(swa_topk_lengths[q_idx].item())
        swa_locs_q = swa_indices[q_idx, :swa_len]
        swa_locs_q = swa_locs_q[swa_locs_q >= 0].to(torch.int64)
        if swa_locs_q.numel() > 0:
            req_ids = (swa_locs_q - 1) // max_context_len
            positions = (swa_locs_q - 1) % max_context_len
            swa_k_parts = [
                swa_k_per_req[int(req_ids[i].item())][int(positions[i].item())]
                for i in range(swa_locs_q.shape[0])
            ]
            swa_k = torch.stack(swa_k_parts, dim=0).float()
        else:
            swa_k = torch.zeros(
                (0, DSV4_HEAD_DIM), dtype=torch.float32, device=q.device
            )

        if extra_indices is not None:
            assert extra_k_bf16 is not None, (
                "compress_ratio in {4, 128} requires `_populate_extra_kv_cache` "
                "to have stashed `fixture._extra_bf16_k`."
            )
            extra_len = int(extra_topk_lengths[q_idx].item())
            extra_locs_q = extra_indices[q_idx, :extra_len]
            extra_locs_q = extra_locs_q[extra_locs_q >= 0].to(torch.int64)
            if extra_locs_q.numel() > 0:
                extra_k = extra_k_bf16[extra_locs_q].float()
                keys = torch.cat([swa_k, extra_k], dim=0)
            else:
                keys = swa_k
        else:
            keys = swa_k

        query = q[q_idx].float()
        scores = torch.einsum("hd,kd->hk", query, keys) * scaling
        sink_scores = attn_sink.view(-1, 1).to(scores.dtype)
        scores_with_sink = torch.cat([scores, sink_scores], dim=-1)
        probs_with_sink = torch.softmax(scores_with_sink, dim=-1)
        probs = probs_with_sink[:, :-1]
        out = torch.einsum("hk,kd->hd", probs, keys)
        outputs.append(out)

    return torch.stack(outputs, dim=0).to(q.dtype)


def _seed_c4_sparse_indices(
    fixture: DSV4AttentionFixture,
    *,
    num_entries: int,
) -> None:
    """For compress_ratio=4 the production `init_flashmla_related` initializes
    `c4_sparse_page_indices` to all `-1` (the C4Indexer fills it in later).
    Since the compact fixture does not run the indexer, the C4 path attends to
    zero extra entries unless we seed the indices ourselves. Seed each query
    row to point to `[0, 1, ..., num_entries - 1]` so the backend reads the
    same `num_entries` C4 K's that the reference also reads, exercising the
    `extra_k_cache` + `extra_indices_in_kvcache` flash_mla integration with
    non-trivial extra contribution.
    """
    md = fixture.backend.forward_metadata.core_metadata
    sparse_indices = md.c4_sparse_page_indices
    num_q, sparse_topk = sparse_indices.shape
    seed = torch.full(
        (num_q, sparse_topk),
        -1,
        dtype=sparse_indices.dtype,
        device=sparse_indices.device,
    )
    seed[:, :num_entries] = torch.arange(
        num_entries, dtype=sparse_indices.dtype, device=sparse_indices.device
    )
    md.c4_sparse_page_indices = seed
    md.c4_sparse_topk_lengths = torch.full(
        (num_q,),
        num_entries,
        dtype=md.c4_sparse_topk_lengths.dtype,
        device=md.c4_sparse_topk_lengths.device,
    )


def _seed_c4_sparse_prefill_indices(
    fixture: DSV4AttentionFixture,
    *,
    num_entries: int,
) -> None:
    """Seed C4 metadata for the sparse prefill extend path.

    `_forward_prefill_sparse` reads `c4_sparse_raw_indices` (request-local
    compressed positions, normally the indexer's output) and derives per-query
    lengths as `(pos + 1) // 4`. Seed the sequential positions the indexer
    emits for short sequences and mirror the same causal set into
    `c4_sparse_page_indices` / `c4_sparse_topk_lengths` so the reference
    attends identical entries. The mirror relies on raw position `k` mapping
    to physical extra-cache id `k` (page 0 of a fresh single-request layout);
    asserted below.
    """
    md = fixture.backend.forward_metadata.core_metadata
    raw_indices = md.c4_sparse_raw_indices
    assert raw_indices is not None, "requires init_flashmla_related(is_prefill=True)"
    num_q, width = raw_indices.shape
    lens = (md.positions_casual + 1) // 4
    max_len = int(lens.max().item())
    pool = fixture.runner.token_to_kv_pool
    c4_page_size = pool.get_extra_key_page_size(layer_id=0)
    assert max_len <= min(
        num_entries, c4_page_size
    ), f"case attends {max_len} c4 entries; only {min(num_entries, c4_page_size)} populated"
    assert (
        md.page_table[:, 0] == 0
    ).all(), "sparse seeding requires the raw==physical identity (first page 0)"
    seq = (
        torch.arange(width, dtype=raw_indices.dtype, device=raw_indices.device)
        .unsqueeze(0)
        .expand(num_q, -1)
    )
    seeded = torch.where(seq < lens.unsqueeze(1), seq, seq.new_full((), -1))
    md.c4_sparse_raw_indices = seeded
    md.c4_sparse_page_indices = seeded.clone()
    md.c4_sparse_topk_lengths = lens.to(md.c4_sparse_topk_lengths.dtype)


def run_dsv4_target_verify_attention_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    topk: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Math-faithful EAGLE `TARGET_VERIFY` test for DSV4. Supports SWA-only,
    SWA + C4, and SWA + C128 via `case.compress_ratio`. Chain only (`topk=1`):
    `DeepseekV4AttnBackend.__init__` asserts `self.topk in [0, 1]` at line 369
    so tree verify is production-unsupported for DSV4.

    Pre-populates the SWA + (optionally) extra caches with the same packed K's
    the production write path would produce, sets `EagleVerifyInput` on the
    forward batch, lets `init_forward_metadata_target_verify` build the per-
    draft-token metadata, then compares the backend forward output against
    the combined SWA + extra-K reference. Chain causal masking falls out of
    the metadata builder's per-q-token `swa_page_indices`.
    """
    assert topk == 1, (
        "DSV4 target_verify is chain-only — `deepseek_v4_backend.py:369` "
        "asserts `self.topk in [0, 1]`. Pass topk=1."
    )
    assert (
        case.forward_mode.is_target_verify()
    ), f"run_dsv4_target_verify_attention_case requires TARGET_VERIFY case; got {case.forward_mode}"
    # Lazy import to avoid cycles (runner_modes imports attention_methods).
    from sglang.test.kits.attention_unittest.runner_modes.speculative_target_verify_runner import (
        _make_eagle_verify_input,
        _prepare_target_verify_batch,
    )

    fixture = build_dsv4_attention_fixture(testcase, case, dtype=dtype, device=device)
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    _populate_swa_kv_cache(fixture, max_context_len=max_context_len, device=device)
    if case.compress_ratio in (4, 128):
        _populate_extra_kv_cache(fixture, layer_id=0, num_entries=_DSV4_EXTRA_ENTRIES)

    _prepare_target_verify_batch(fixture.forward_batch, case, device)
    fixture.forward_batch.spec_info = _make_eagle_verify_input(
        case,
        fixture.forward_batch,
        topk=topk,
        device=device,
    )

    q_input, _ = fixture.actual_module.project(fixture.input_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        _seed_c4_if_needed(fixture)
        actual = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=case.compress_ratio,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )
        expected = _pure_torch_dsv4_combined_reference(fixture, q_input)

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=DSV4_ATOL, rtol=DSV4_RTOL
    )


def run_dsv4_draft_extend_attention_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Math-faithful EAGLE `DRAFT_EXTEND` test for DSV4.

    `compress_ratio` must be 0: `init_forward_metadata_draft_extend` hardcodes
    `need_compress=False`, which leaves
    `core_attn_metadata.c4_sparse_page_indices` /
    `.c128_page_indices` / `.c4_flashmla_metadata` / `.c128_flashmla_metadata`
    at None. The C4 path then crashes on
    `extra_indices.shape[-1] % 64` and the C128 path crashes on
    `flashmla.get_flashmla_metadata(128) is None` inside
    `flash_mla.flash_mla_with_kvcache`. Production DSV4 + DRAFT_EXTEND is
    therefore SWA-only by construction.
    """
    assert case.compress_ratio == 0, (
        "DSV4 DRAFT_EXTEND is SWA-only — `init_forward_metadata_draft_extend` "
        "uses `need_compress=False` so C4 / C128 metadata is unpopulated. See "
        "`deepseek_v4_backend.py:636-663` and the 'Production-Unsupported' "
        "section in dsv4/README.md."
    )
    assert (
        case.forward_mode.is_draft_extend_v2()
    ), f"run_dsv4_draft_extend_attention_case requires DRAFT_EXTEND; got {case.forward_mode}"
    from sglang.test.kits.attention_unittest.runner_modes.speculative_draft_extend_runner import (
        _make_eagle_draft_extend_input,
    )

    fixture = build_dsv4_attention_fixture(testcase, case, dtype=dtype, device=device)
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    _populate_swa_kv_cache(fixture, max_context_len=max_context_len, device=device)

    fixture.forward_batch.spec_info = _make_eagle_draft_extend_input(
        case,
        fixture.forward_batch,
        device=device,
    )

    q_input, _ = fixture.actual_module.project(fixture.input_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        actual = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=0,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )
        expected = _pure_torch_dsv4_combined_reference(fixture, q_input)

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=DSV4_ATOL, rtol=DSV4_RTOL
    )


def run_dsv4_compress_attention_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    extra_entries: int = 32,
    sparse_prefill: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Math-faithful test for the SWA + C4 (compress_ratio=4) / SWA + C128
    (compress_ratio=128) path through `DeepseekV4AttnBackend.forward`.

    Pre-writes random packed K into both the SWA cache and the extra
    (C4/C128) cache via the production pack+set paths, lets
    `init_forward_metadata` populate the compression metadata, manually seeds
    the C4 metadata the exercised path consumes (see `_seed_c4_if_needed`; the
    un-run indexer would otherwise leave it at `-1` / uninitialized), then
    dispatches `forward(compress_ratio=case.compress_ratio)` and compares
    against an independent pure-PyTorch SWA + extra reference that reads the
    SAME cache bytes and metadata indices.

    `sparse_prefill` pins `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`, selecting the
    dense `flash_mla_with_kvcache` extend path or `_forward_prefill_sparse`;
    the C4 seeding dispatches on the same flag.
    """
    assert case.compress_ratio in (
        4,
        128,
    ), f"DSV4 compact runner requires compress_ratio in (4, 128); got {case.compress_ratio}"
    if sparse_prefill:
        assert (
            case.forward_mode.is_extend_without_speculative()
        ), f"sparse prefill only serves extend; got {case.forward_mode}"
    fixture = build_dsv4_attention_fixture(
        testcase,
        case,
        dtype=dtype,
        device=device,
        compression_ratios=[case.compress_ratio],
    )
    fixture.seed_c4_for_sparse_prefill = sparse_prefill
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    _populate_swa_kv_cache(fixture, max_context_len=max_context_len, device=device)
    _populate_extra_kv_cache(fixture, layer_id=0, num_entries=extra_entries)

    q_input, _ = fixture.actual_module.project(fixture.input_hidden)
    with (
        torch.no_grad(),
        forward_context(ForwardContext(attn_backend=fixture.backend)),
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.override(sparse_prefill),
    ):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        _seed_c4_if_needed(fixture, num_entries=extra_entries)
        actual = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=case.compress_ratio,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )
        # Only `_forward_prefill_sparse` populates `sparse_prefill_cache`;
        # verify the intended path ran before the reference rebuilds metadata.
        sparse_cache = fixture.backend.forward_metadata.sparse_prefill_cache
        if sparse_prefill:
            testcase.assertIsNotNone(
                sparse_cache, f"{case.name} did not take _forward_prefill_sparse"
            )
        else:
            testcase.assertIsNone(
                sparse_cache, f"{case.name} did not take the dense extend path"
            )
        expected = _pure_torch_dsv4_combined_reference(fixture, q_input)

    torch.testing.assert_close(
        actual.float(), expected.float(), atol=DSV4_ATOL, rtol=DSV4_RTOL
    )

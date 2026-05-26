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

from sglang.srt.layers import dp_attention as _dp_attention
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.server_args import set_global_server_args_for_scheduler

# DSV4 backend pre-resolves attention TP at construction; pin to single-rank.
_dp_attention.get_attention_tp_size = lambda: 1
_dp_attention.get_attention_tp_rank = lambda: 0
_dp_attention.get_attention_cp_size = lambda: 1
_dp_attention.get_attention_cp_rank = lambda: 0

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
DSV4_ATOL = 5e-2
DSV4_RTOL = 5e-2


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
        self.hf_text_config = self.hf_config


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
        self.device = device
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.gpu_id = 0
        self.page_size = case.page_size
        self.model_config = model_config
        self.tp_size = 1
        self.dp_size = 1
        self.pp_size = 1
        self.server_args = SimpleNamespace(
            attention_backend=case.backend,
            chunked_prefill_size=-1,
            disable_cuda_graph=disable_cuda_graph,
            disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
            disable_radix_cache=False,
            disaggregation_mode=None,
            dp_size=1,
            enable_deterministic_inference=False,
            enable_dp_attention=False,
            enable_mis=False,
            is_embedding=False,
            kv_cache_dtype="auto",
            max_running_requests=None,
            model_path=None,
            pp_size=1,
            revision=None,
            speculative_algorithm=None,
            speculative_eagle_topk=0,
            speculative_num_draft_tokens=0,
            speculative_num_steps=0,
            tp_size=1,
            device=device,
            mem_fraction_static=0.8,
        )
        set_global_server_args_for_scheduler(self.server_args)
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
            state_dtype=dtype,
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
        raw_loc=loc.to(torch.int64),
        cache_nope_fp8_rope_bf16_pack=pack,
    )


def _unpack_swa_cache(
    runner: MockDSV4ModelRunner,
    layer_id: int,
    full_locs: torch.Tensor,
) -> torch.Tensor:
    """Read back FP8 nope + BF16 rope from the SWA buffer at `full_locs` and
    dequantize to bfloat16. Returns `[num_tokens, DSV4_HEAD_DIM]`.

    Mirrors the page layout written by `_set_k_and_s_triton` in
    `sglang/srt/layers/attention/dsv4/index_buf_accessor.py`.
    """
    pool = runner.token_to_kv_pool
    swa_locs = pool.translate_loc_from_full_to_swa(full_locs).to(torch.int64)
    swa_kv_pool = pool.swa_kv_pool
    page_size = swa_kv_pool.page_size  # 128
    buf = swa_kv_pool.kv_buffer[layer_id]  # [num_pages, bytes_per_page]
    num_pages, bytes_per_page = buf.shape

    nope_dim = DSV4_QK_NOPE_HEAD_DIM
    rope_dim = DSV4_QK_ROPE_HEAD_DIM
    nope_rope_bytes = nope_dim + rope_dim * 2  # 448 + 128 = 576
    s_offset_nbytes_in_page = page_size * nope_rope_bytes  # 128 * 576
    scale_dim = nope_dim // 64  # 7 tiles of 64 elems
    padded_scale = scale_dim + 1  # 8 bytes/token incl. pad

    fp8_dtype = torch.float8_e4m3fn
    buf_fp8 = buf.view(fp8_dtype)
    buf_bf16 = buf.view(torch.bfloat16)
    buf_u8 = buf.view(torch.uint8)

    num_tokens = full_locs.shape[0]
    device = buf.device

    page_idx = swa_locs // page_size
    tok_in_page = swa_locs % page_size

    # nope: [num_tokens, nope_dim] FP8
    nope_byte_base = page_idx * bytes_per_page + tok_in_page * nope_rope_bytes
    nope_byte_offsets = nope_byte_base.unsqueeze(1) + torch.arange(
        nope_dim, device=device, dtype=torch.int64
    ).unsqueeze(0)
    nope_fp8 = buf_fp8.flatten()[nope_byte_offsets.flatten()].view(num_tokens, nope_dim)

    # rope: [num_tokens, rope_dim] BF16 (BF16 strides are 2-byte; view is via bf16)
    rope_bf16_base = (
        page_idx * (bytes_per_page // 2)
        + tok_in_page * (nope_rope_bytes // 2)
        + (nope_dim // 2)
    )
    rope_bf16_offsets = rope_bf16_base.unsqueeze(1) + torch.arange(
        rope_dim, device=device, dtype=torch.int64
    ).unsqueeze(0)
    rope_bf16 = buf_bf16.flatten()[rope_bf16_offsets.flatten()].view(
        num_tokens, rope_dim
    )

    # scales: ue8m0 stored as uint8 = exponent + 127; dequant factor = 2^(byte-127).
    scale_byte_base = (
        page_idx * bytes_per_page + s_offset_nbytes_in_page + tok_in_page * padded_scale
    )
    scale_byte_offsets = scale_byte_base.unsqueeze(1) + torch.arange(
        scale_dim, device=device, dtype=torch.int64
    ).unsqueeze(0)
    scale_u8 = buf_u8.flatten()[scale_byte_offsets.flatten()].view(
        num_tokens, scale_dim
    )
    scale = torch.pow(
        2.0, (scale_u8.to(torch.float32) - 127.0)
    )  # [num_tokens, scale_dim]

    # nope_fp8 -> float32 -> per-tile scale broadcast.
    nope_f32 = nope_fp8.float().view(num_tokens, scale_dim, 64)
    nope_dequant = nope_f32 * scale.unsqueeze(-1)
    nope_dequant = nope_dequant.view(num_tokens, nope_dim).to(torch.bfloat16)

    return torch.cat([nope_dequant, rope_bf16], dim=-1)


@dataclass
class DSV4AttentionFixture:
    case: DSV4AttentionCase
    runner: MockDSV4ModelRunner
    backend: object
    actual_module: ProjectedDSV4Attention
    forward_batch: ForwardBatch
    prefix_hidden: list[torch.Tensor]
    input_hidden: torch.Tensor


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
    max_seq = max(case.seq_lens)
    # SWA-only (compress_ratio=0) is the SGLang path that handles the
    # last-`SWA_WINDOW`-tokens slice for *all* sequence lengths. seq_len >
    # SWA_WINDOW just means the SWA mask truncates the oldest tokens; the
    # backend's `get_swa_page_indices` and the fixture's reference both pick
    # the same trailing window so this works without enabling C4/C128.
    # The pool capacity (`swa_size`) and `max_context_len` are sized in
    # `build_dsv4_attention_fixture` so the full KV fits.
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
    """Independent reference: unpack same-quantized K from the SWA cache, run
    standard softmax(q @ k.T) with sliding-window-causal mask + attention sink.

    `q` has shape `[num_q, num_heads, DSV4_HEAD_DIM]`. `full_kv_locs_per_req[r]`
    are the **full-pool** locs of all KV tokens for request r in causal order.
    `case` overrides `fixture.case` — needed by runner-mode integrations where
    the replay/capture case has a different batch shape than the fixture's
    construction-time case.
    """
    if case is None:
        case = fixture.case
    scaling = DSV4_HEAD_DIM**-0.5
    attn_sink = fixture.actual_module.attn_sink.detach()
    outputs = []
    q_idx = 0
    for req_idx in range(case.batch_size):
        kv_locs = full_kv_locs_per_req[req_idx]
        kv_full = _unpack_swa_cache(fixture.runner, 0, kv_locs).float()
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
) -> list[torch.Tensor]:
    """Project K for every kv token (prefix + input) and write the packed
    FP8 nope + BF16 rope representation into the SWA pool via the production
    pack+set path. Returns the full-pool token locs per request in causal order.
    """
    case = fixture.case
    full_kv_locs_per_req: list[torch.Tensor] = []
    all_k_bf16_parts: list[torch.Tensor] = []
    all_k_locs_parts: list[torch.Tensor] = []
    for req_idx, prefix in enumerate(fixture.prefix_hidden):
        input_part = fixture.input_hidden[
            sum(case.input_lens[:req_idx]) : sum(case.input_lens[: req_idx + 1])
        ]
        req_hidden = torch.cat([prefix, input_part], dim=0)
        _, k_req = fixture.actual_module.project(req_hidden)
        all_k_bf16_parts.append(k_req.view(-1, DSV4_HEAD_DIM))
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
        torch.randn(length, DSV4_HEAD_DIM, dtype=dtype, device=device) for length in lens
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


def prepare_dsv4_runner_inputs(
    fixture: DSV4AttentionFixture,
    case: DSV4AttentionCase,
    batch: ForwardBatch,
    inputs: dict[str, Any],
    *,
    max_context_len: int,
) -> None:
    """Project K for prefix + input hidden in `inputs` and write the packed
    FP8 nope + BF16 rope representation into the SWA cache so the backend's
    SWA read path sees the matching values."""
    del batch
    all_k_parts: list[torch.Tensor] = []
    all_locs_parts: list[torch.Tensor] = []
    input_hidden = inputs["input_hidden"]
    for req_idx, prefix in enumerate(inputs["prefix_hidden"]):
        input_part = input_hidden[
            sum(case.input_lens[:req_idx]) : sum(case.input_lens[: req_idx + 1])
        ]
        req_hidden = torch.cat([prefix, input_part], dim=0)
        _, k_req = fixture.actual_module.project(req_hidden)
        all_k_parts.append(k_req.view(-1, DSV4_HEAD_DIM))
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


def run_dsv4_fixture_eager(fixture: DSV4AttentionFixture) -> torch.Tensor:
    """Eager forward that also re-initializes the forward metadata. Used as
    the eager baseline before graph capture/replay; the existing
    `run_dsv4_attention_case` already exercises this end-to-end including
    the reference comparison.
    """
    case = fixture.case
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    full_kv_locs_per_req = _populate_swa_kv_cache(
        fixture, max_context_len=max_context_len, device=runner.device
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
    backend. Projects Q from `inputs['input_hidden']` and calls `forward`.
    """
    case = fixture.case
    q_input, _ = fixture.actual_module.project(inputs["input_hidden"])
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
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
    """Pure PyTorch SWA reference built from `inputs`. The actual path is
    expected to have already written the corresponding K into the SWA cache
    via `prepare_dsv4_runner_inputs`; the reference unpacks back from the
    cache to absorb FP8 quant noise consistently."""
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
    q_input, _ = fixture.actual_module.project(inputs["input_hidden"])
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
) -> None:
    """Write `num_entries` packed FP8-nope/BF16-rope K vectors into the C4 or
    C128 extra cache via the production `set_extra_key_buffer` path. The
    backend's `forward(compress_ratio=4 or 128)` path reads from this buffer."""
    pool = fixture.runner.token_to_kv_pool
    device = fixture.runner.device
    rand_k = torch.randn(num_entries, DSV4_HEAD_DIM, dtype=torch.bfloat16, device=device)
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(rand_k)
    loc = torch.arange(num_entries, dtype=torch.int64, device=device)
    pool.set_extra_key_buffer(layer_id=layer_id, loc=loc, cache_nope_fp8_rope_bf16_pack=pack)


def run_dsv4_compress_smoke_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    extra_entries: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Smoke test for the C4 (compress_ratio=4) or C128 (compress_ratio=128)
    dispatch through `DeepseekV4AttnBackend.forward`.

    Pre-writes random packed K into both the SWA cache and the C4/C128 extra
    cache via the production pack+set paths, lets `init_forward_metadata`
    populate the compression metadata, then dispatches `forward(compress_ratio=
    case.compress_ratio)`. Asserts the output has the right shape and is
    finite; does not verify Compressor math (no independent compressor
    reference yet — that is a deferred follow-up).
    """
    assert case.compress_ratio in (4, 128), (
        f"smoke runner requires compress_ratio in (4, 128); got {case.compress_ratio}"
    )
    fixture = build_dsv4_attention_fixture(
        testcase,
        case,
        dtype=dtype,
        device=device,
        compression_ratios=[case.compress_ratio],
    )
    runner = fixture.runner
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]

    _populate_swa_kv_cache(fixture, max_context_len=max_context_len, device=device)
    _populate_extra_kv_cache(fixture, layer_id=0, num_entries=extra_entries)

    q_input, _ = fixture.actual_module.project(fixture.input_hidden)
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        out = fixture.backend.forward(
            q=q_input,
            k=q_input,
            v=q_input,
            layer=fixture.actual_module.attn,
            forward_batch=fixture.forward_batch,
            compress_ratio=case.compress_ratio,
            save_kv_cache=False,
            attn_sink=fixture.actual_module.attn_sink,
        )

    expected_shape = (case.num_input_tokens, case.num_heads, DSV4_V_HEAD_DIM)
    testcase.assertEqual(
        tuple(out.shape),
        expected_shape,
        f"compress_ratio={case.compress_ratio} forward must return shape "
        f"[num_input_tokens, num_heads, v_head_dim]",
    )
    testcase.assertTrue(
        torch.isfinite(out).all().item(),
        f"compress_ratio={case.compress_ratio} forward must return finite values "
        f"(no NaN/Inf); mean abs = {out.abs().mean().item()}",
    )

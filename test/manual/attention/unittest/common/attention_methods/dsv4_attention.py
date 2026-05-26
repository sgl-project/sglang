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
    """One EXTEND case scoped to compress_ratio=0 (SWA-only)."""

    name: str
    backend: str
    forward_mode: ForwardMode
    num_heads: int
    page_size: int
    prefix_lens: tuple[int, ...]
    extend_lens: tuple[int, ...]
    # compress_ratio is fixed at 0 for this slice; C4/C128 are follow-ups.
    compress_ratio: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.prefix_lens)

    @property
    def input_lens(self) -> tuple[int, ...]:
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
    )


class TinyDSV4ModelConfig:
    def __init__(
        self,
        *,
        num_heads: int,
        context_len: int,
    ):
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
            num_hidden_layers=1,
            compress_ratios=[0],
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
    ):
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
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
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
            size=case.batch_size,
            max_context_len=max_context_len,
            device=device,
            enable_memory_saver=False,
        )
        # DeepSeekV4TokenToKVPool requires page_size % swa_page_size == 0 and
        # the SWA window currently equals SWA_WINDOW (=128) in the backend.
        # `compression_ratios=[0]` disables C4/C128 sub-pools (their layer_num=0).
        # DSV4 KV pool stores FP8 nope; pass fp8 dtype so store_dtype=uint8 (the
        # backing tensor is always raw bytes regardless of the nominal dtype).
        self.token_to_kv_pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=case.batch_size,
            swa_size=swa_size,
            c4_size=case.page_size,  # minimum: one page worth; unused for layer_num=0
            c128_size=case.page_size,
            c4_state_pool_size=case.batch_size,
            c128_state_pool_size=case.batch_size,
            page_size=case.page_size,
            swa_page_size=DSV4_SWA_WINDOW,
            dtype=torch.float8_e4m3fn,
            state_dtype=dtype,
            qk_nope_head_dim=DSV4_QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=DSV4_QK_ROPE_HEAD_DIM,
            indexer_head_dim=128,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
            compression_ratios=[0],
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
        # Zero attention sink == identity scaling (exp(lse)/(exp(lse)+exp(0))
        # is still a meaningful sink, but for ease of reference we keep it at
        # -inf-equivalent by passing a fixed value the reference also applies).
        self.attn_sink = nn.Parameter(
            torch.full((num_heads,), -1e30, dtype=torch.float32, device=device),
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


def _build_dsv4_metadata(
    backend,
    runner: MockDSV4ModelRunner,
    case: DSV4AttentionCase,
    out_cache_loc: torch.Tensor,
):
    """Construct a DSV4Metadata that only exercises the compress_ratio=0 path."""
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        DSV4AttnMetadata,
        DSV4Metadata,
        _create_flashmla_metadata,
        _pad_last_dim,
    )

    device = runner.device
    num_q = case.num_input_tokens

    # seq_lens_casual: per-q-token, the kv-position+1 it attends to.
    seq_lens_casual = []
    req_idx_per_q = []
    positions = []
    for req_idx, (prefix_len, ext_len) in enumerate(
        zip(case.prefix_lens, case.input_lens)
    ):
        for offset in range(ext_len):
            pos = prefix_len + offset
            seq_lens_casual.append(pos + 1)
            req_idx_per_q.append(req_idx)
            positions.append(pos)
    seq_lens_casual_t = torch.tensor(seq_lens_casual, dtype=torch.int32, device=device)
    positions_casual = torch.tensor(positions, dtype=torch.int32, device=device)
    req_pool_indices_repeated = torch.tensor(
        req_idx_per_q, dtype=torch.int32, device=device
    )

    # SWA window: for each query, the last DSV4_SWA_WINDOW full locs (-1 where invalid).
    pos_t = positions_casual.unsqueeze(1) - torch.arange(
        DSV4_SWA_WINDOW, dtype=torch.int32, device=device
    ).unsqueeze(0)
    invalid = pos_t < 0
    pos_t = pos_t.masked_fill(invalid, 0)
    raw_full_locs = runner.req_to_token_pool.req_to_token[
        req_pool_indices_repeated[:, None], pos_t
    ]
    raw_full_locs = raw_full_locs.masked_fill(invalid, -1)
    swa_page_indices = runner.token_to_kv_pool.translate_loc_from_full_to_swa(
        raw_full_locs
    )
    swa_page_indices = _pad_last_dim(swa_page_indices, multiples_of=64)
    swa_topk_lengths = torch.clamp(seq_lens_casual_t, max=DSV4_SWA_WINDOW)

    # page_table: full-token-loc page indices, one row per q token.
    max_seq_len = max(case.seq_lens)
    page_table = runner.req_to_token_pool.req_to_token[
        req_pool_indices_repeated, : max_seq_len : case.page_size
    ]
    page_table = (page_table // case.page_size).to(torch.int32)

    metadata = DSV4AttnMetadata(
        page_size=case.page_size,
        page_table=page_table,
        raw_out_loc=out_cache_loc.to(torch.int32),
        cuda_int32_kwargs={"device": device, "dtype": torch.int32},
        seq_lens_casual=seq_lens_casual_t,
        positions_casual=positions_casual,
        swa_page_indices=swa_page_indices,
        swa_topk_lengths=swa_topk_lengths,
        c4_sparse_topk=DSV4_INDEX_TOPK,
    )
    # Only c1_flashmla_metadata is needed for compress_ratio=0; the C4/C128
    # metadata slots stay unused.
    metadata.c1_flashmla_metadata = _create_flashmla_metadata()
    metadata.c4_flashmla_metadata = None
    metadata.c128_flashmla_metadata = None
    metadata.c4_sparse_topk_lengths = None
    metadata.c4_sparse_page_indices = None

    backend.forward_metadata = DSV4Metadata(
        core_attn_metadata=metadata,
        indexer_metadata=None,
        c4_compress_metadata=None,
        c128_compress_metadata=None,
    )


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
) -> DSV4AttentionFixture:
    max_seq = max(case.seq_lens)
    # Each query must see only kv positions in its own SWA window. With page_size
    # 256 and one prefix-free batch, this fits comfortably.
    if max_seq > DSV4_SWA_WINDOW:
        testcase.skipTest(
            "DSV4 fixture currently restricts seq_len <= SWA_WINDOW; "
            "longer sequences require the C4/C128 paths."
        )
    seed = 7100 + len(case.name)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_config = TinyDSV4ModelConfig(
        num_heads=case.num_heads,
        context_len=max_context_len,
    )
    runner = MockDSV4ModelRunner(
        case=case,
        model_config=model_config,
        dtype=dtype,
        device=device,
        max_context_len=max_context_len,
        swa_size=swa_size,
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
) -> torch.Tensor:
    """Independent reference: unpack same-quantized K from the SWA cache, run
    standard softmax(q @ k.T) with sliding-window-causal mask + attention sink.

    `q` has shape `[num_q, num_heads, DSV4_HEAD_DIM]`. `full_kv_locs_per_req[r]`
    are the **full-pool** locs of all KV tokens for request r in causal order.
    """
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


def run_dsv4_attention_case(
    testcase,
    case: DSV4AttentionCase,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    fixture = build_dsv4_attention_fixture(testcase, case, dtype=dtype, device=device)
    runner = fixture.runner

    # Project Q and K for every kv token (prefix + input).
    full_kv_locs_per_req: list[torch.Tensor] = []
    all_k_bf16_parts = []
    all_k_locs_parts = []
    max_context_len = runner.req_to_token_pool.req_to_token.shape[1]
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

    # Write ALL kv (prefix + input) into the cache via the real pack+set path.
    _write_swa_cache(runner, layer_id=0, loc=all_k_locs, k_bf16=all_k)

    # Project Q for the input tokens only.
    q_input, _ = fixture.actual_module.project(fixture.input_hidden)

    # Build DSV4Metadata directly (compress_ratio=0 path bypasses
    # init_forward_metadata, which depends on compressor/indexer state we don't
    # exercise here).
    _build_dsv4_metadata(
        fixture.backend, runner, case, fixture.forward_batch.out_cache_loc
    )

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
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

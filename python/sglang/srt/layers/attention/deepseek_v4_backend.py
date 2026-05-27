from __future__ import annotations

import enum
import functools
import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if envs.SGLANG_OPT_USE_COMPRESSOR_V2.get():
    # NOTE: should eventually be the only compressor backend
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )
else:
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
        FusedCompressMetadata,
        create_paged_compressor_data,
    )

from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.layers.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from flash_mla.flash_mla_interface import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


# #region debug-point online-c128:report
def _debug_online_c128_event(
    hypothesis_id: str,
    location: str,
    msg: str,
    data: Optional[Dict] = None,
) -> None:
    if os.environ.get("SGLANG_DEBUG_ONLINE_C128_MTP") != "1":
        return
    url = os.environ.get("DEBUG_SERVER_URL", "http://127.0.0.1:7777/event")
    session_id = os.environ.get("DEBUG_SESSION_ID", "cuda-illegal-address")
    env_path = os.environ.get(
        "SGLANG_DEBUG_ONLINE_C128_ENV", ".dbg/cuda-illegal-address.env"
    )
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("DEBUG_SERVER_URL="):
                    url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
    except Exception:
        pass
    payload = {
        "sessionId": session_id,
        "runId": os.environ.get("SGLANG_DEBUG_RUN_ID", "pre-fix"),
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    if os.environ.get("SGLANG_DEBUG_ONLINE_C128_LOGGER") == "1":
        logger.warning(
            "[DEBUG][online-c128][%s] %s %s",
            hypothesis_id,
            location,
            json.dumps(payload, default=str, ensure_ascii=False),
        )
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload, default=str).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.2,
        ).read()
    except Exception:
        pass


def _debug_online_c128_sync(location: str, data: Optional[Dict] = None) -> None:
    if os.environ.get("SGLANG_DEBUG_ONLINE_C128_SYNC") != "1":
        return
    try:
        if torch.cuda.is_current_stream_capturing():
            _debug_online_c128_event(
                "D",
                location,
                "skip cuda synchronize during cuda graph capture",
                data,
            )
            return
        torch.cuda.synchronize()
        _debug_online_c128_event("D", location, "cuda synchronize ok", data)
    except Exception as e:
        _debug_online_c128_event(
            "D",
            location,
            "cuda synchronize failed",
            {"error": repr(e), **(data or {})},
        )
        raise


# #endregion


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    import flash_mla

    return flash_mla.get_mla_metadata()[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    c1_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 4, 128]):
        if compress_ratio == 0:
            return self.c1_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_page_indices",
            ],
            assign_fields=[
                "c1_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )

    def init_compression_metadata(self):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        (
            self.c4_out_loc,
            _,
            self.c4_topk_lengths_raw,
            self.c4_topk_lengths_clamp1,
            self.c128_out_loc,
            _,
            self.c128_topk_lengths_clamp1,
            self.c128_page_indices,
        ) = _init_compression_metadata_triton(
            self.seq_lens_casual,
            self.positions_casual,
            self.raw_out_loc,
            self.page_table,
            self.page_size,
            compute_page_indices=True,
        )

        self.c128_page_indices = _pad_last_dim(self.c128_page_indices)
        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_attention_cp_rank()
        cp_size = get_attention_cp_size()
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        assert self.c4_topk_lengths_clamp1 is not None
        self.c4_sparse_topk_lengths = torch.clamp(
            self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
        )
        self.c4_sparse_page_indices = torch.full(
            (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
            -1,
            dtype=torch.int32,
            device=self.c4_topk_lengths_clamp1.device,
        )
        self.c4_sparse_page_indices = _pad_last_dim(self.c4_sparse_page_indices)
        self.c1_flashmla_metadata = _create_flashmla_metadata()
        self.c4_flashmla_metadata = _create_flashmla_metadata()
        self.c128_flashmla_metadata = _create_flashmla_metadata()


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None
    seq_lens_host: Optional[torch.Tensor] = None
    extend_seq_lens_host: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens
        self.seq_lens_host = other.seq_lens_host
        self.extend_seq_lens_host = other.extend_seq_lens_host


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend(include_v2=True):
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


@dataclass
class _OnlineC128VerifyLayerState:
    pre_state: torch.Tensor
    kv_score_input: Optional[torch.Tensor] = None


@dataclass
class _OnlineC128VerifyContext:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    layer_states: Dict[int, _OnlineC128VerifyLayerState] = field(default_factory=dict)


class DeepseekV4AttnBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )

        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None
        self._replay_forward_batch: Optional[ForwardBatch] = None  # FIXME: out-of-band
        self._online_c128_verify_ctx: Optional[_OnlineC128VerifyContext] = None
        self._online_c128_kv_score_refs: Dict[int, torch.Tensor] = {}

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def _to_pinned_cpu_i64(self, x: torch.Tensor) -> torch.Tensor:
        pinned = torch.empty(
            x.shape, dtype=torch.int64, device="cpu", pin_memory=True
        )
        pinned.copy_(x.to(torch.int64), non_blocking=False)
        return pinned

    def _online_c128_mtp_enabled(self) -> bool:
        return (
            envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get()
            and envs.SGLANG_EXPERIMENTAL_ONLINE_C128_MTP.get()
            and self.mtp_enabled
        )

    def _clear_online_c128_verify_context(self) -> None:
        self._online_c128_verify_ctx = None

    def _set_online_c128_verify_context(
        self, req_pool_indices: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        if not self._online_c128_mtp_enabled():
            self._online_c128_verify_ctx = None
            return
        self._online_c128_verify_ctx = _OnlineC128VerifyContext(
            req_pool_indices=req_pool_indices.detach(),
            seq_lens=seq_lens.detach(),
        )
        _debug_online_c128_event(
            "C",
            "deepseek_v4_backend:_set_online_c128_verify_context",
            "set verify context",
            {
                "req_pool_indices_shape": list(req_pool_indices.shape),
                "seq_lens_shape": list(seq_lens.shape),
                "seq_lens_head": seq_lens[:8].detach().cpu().tolist(),
            },
        )
        self._prepare_online_c128_verify_layers()

    def _get_chunk_slot_indices(
        self, req_pool_indices: torch.Tensor, chunk_starts: torch.Tensor
    ) -> torch.Tensor:
        req_pool_indices_i64 = req_pool_indices.to(torch.int64)
        chunk_starts_i64 = chunk_starts.to(torch.int64)
        full_loc = self.req_to_token[req_pool_indices_i64, chunk_starts_i64]
        full_to_swa = self.token_to_kv_pool.full_to_swa_index_mapping.detach()
        swa_loc = full_to_swa[full_loc].to(torch.int64)
        return swa_loc // self.token_to_kv_pool.swa_page_size

    def _get_online_c128_layer_state(
        self, layer_id: int
    ) -> Optional[_OnlineC128VerifyLayerState]:
        ctx = self._online_c128_verify_ctx
        if ctx is None:
            return None
        return ctx.layer_states.get(layer_id)

    def _iter_online_c128_layers(self):
        for layer in self.model_runner.model.model.layers:
            attn = getattr(layer, "self_attn", None)
            compressor = getattr(attn, "compressor", None)
            if compressor is not None and compressor.ratio == 128:
                yield compressor.layer_id, compressor

    def _prepare_online_c128_verify_layers(self) -> None:
        ctx = self._online_c128_verify_ctx
        if ctx is None:
            return

        bs = ctx.seq_lens.shape[0]
        has_partial = (ctx.seq_lens > 0) & ((ctx.seq_lens % 128) != 0)
        slots = None
        if has_partial.any():
            chunk_starts = ((ctx.seq_lens[has_partial].to(torch.int64) - 1) // 128) * 128
            slots = self._get_chunk_slot_indices(
                req_pool_indices=ctx.req_pool_indices[has_partial],
                chunk_starts=chunk_starts,
            )
            _debug_online_c128_event(
                "A",
                "deepseek_v4_backend:_prepare_online_c128_verify_layers",
                "prepare partial state slots",
                {
                    "bs": int(bs),
                    "num_partial": int(has_partial.sum().item()),
                    "chunk_starts_head": chunk_starts[:8].detach().cpu().tolist(),
                    "slots_head": slots[:8].detach().cpu().tolist(),
                },
            )

        for layer_id, compressor in self._iter_online_c128_layers():
            temp_pool = self.token_to_kv_pool.get_temp_attention_compress_states(layer_id)
            pre_state = torch.zeros(
                (bs, compressor.head_dim * 3),
                dtype=temp_pool.kv_score_buffer.kv_score.dtype,
                device=temp_pool.kv_score_buffer.kv_score.device,
            )
            if has_partial.any():
                main_pool = self.token_to_kv_pool.get_attention_compress_states(layer_id)
                assert slots is not None
                main_state = main_pool.kv_score_buffer.kv_score[slots].clone()
                pre_state[has_partial] = main_state
                temp_pool.kv_score_buffer.kv_score[slots] = main_state
            ctx.layer_states[layer_id] = _OnlineC128VerifyLayerState(
                pre_state=pre_state,
                kv_score_input=self._online_c128_kv_score_refs.get(layer_id),
            )

    def get_override_compress_state_pool(
        self,
        compressor,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        forward_batch: ForwardBatch,
    ):
        if (
            not self._online_c128_mtp_enabled()
            or forward_batch.forward_mode != ForwardMode.TARGET_VERIFY
            or compressor.is_in_indexer
            or compressor.ratio != 128
        ):
            return None

        ctx = self._online_c128_verify_ctx
        if ctx is None:
            self._set_online_c128_verify_context(
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens.to(torch.int32),
            )
            ctx = self._online_c128_verify_ctx
            assert ctx is not None

        layer_state = ctx.layer_states.get(compressor.layer_id)
        temp_pool = token_to_kv_pool.get_temp_attention_compress_states(compressor.layer_id)
        if layer_state is None:
            self._prepare_online_c128_verify_layers()
        return temp_pool

    def capture_online_c128_verify_inputs(
        self,
        layer_id: int,
        compressor,
        kv_score_input: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        if (
            not self._online_c128_mtp_enabled()
            or forward_batch.forward_mode != ForwardMode.TARGET_VERIFY
            or compressor.is_in_indexer
            or compressor.ratio != 128
        ):
            return
        layer_state = self._get_online_c128_layer_state(layer_id)
        if layer_state is None:
            return
        if (
            layer_state.kv_score_input is None
            or layer_state.kv_score_input.shape != kv_score_input.shape
        ):
            layer_state.kv_score_input = kv_score_input.detach().clone()
        else:
            layer_state.kv_score_input.copy_(kv_score_input)
        self._online_c128_kv_score_refs[layer_id] = layer_state.kv_score_input
        _debug_online_c128_event(
            "C",
            "deepseek_v4_backend:capture_online_c128_verify_inputs",
            "captured verify kv_score_input",
            {
                "layer_id": int(layer_id),
                "kv_score_shape": list(kv_score_input.shape),
                "cached_shape": list(layer_state.kv_score_input.shape),
                "forward_mode": str(forward_batch.forward_mode),
            },
        )
        _debug_online_c128_sync(
            "deepseek_v4_backend:capture_online_c128_verify_inputs",
            {"layer_id": int(layer_id)},
        )

    def update_online_c128_state_after_mtp_verify(
        self,
        accept_lens: torch.Tensor,
        model,
    ) -> None:
        if not self._online_c128_mtp_enabled():
            return

        ctx = self._online_c128_verify_ctx
        if ctx is None or accept_lens.numel() == 0:
            self._clear_online_c128_verify_context()
            return

        accept_lens_i64 = accept_lens.to(torch.int64)
        actual_bs = accept_lens_i64.shape[0]
        num_verify_tokens = int(self.speculative_num_draft_tokens)
        accepted_kv_lens_all = torch.minimum(
            accept_lens_i64,
            torch.full_like(accept_lens_i64, num_verify_tokens),
        )
        _debug_online_c128_event(
            "E",
            "deepseek_v4_backend:update_online_c128_state_after_mtp_verify",
            "commit start",
            {
                "accept_lens": accept_lens_i64.detach().cpu().tolist(),
                "accepted_kv_lens": accepted_kv_lens_all.detach().cpu().tolist(),
                "actual_bs": int(actual_bs),
                "ctx_seq_lens_shape": list(ctx.seq_lens.shape),
                "ctx_req_pool_shape": list(ctx.req_pool_indices.shape),
                "num_verify_tokens": int(num_verify_tokens),
            },
        )

        for layer in model.model.layers:
            attn = getattr(layer, "self_attn", None)
            compressor = getattr(attn, "compressor", None)
            if compressor is None or compressor.ratio != 128:
                continue

            layer_state = ctx.layer_states.get(compressor.layer_id)
            if layer_state is None or layer_state.kv_score_input is None:
                continue

            kv_score_input = layer_state.kv_score_input
            head_dim = compressor.head_dim
            if kv_score_input.numel() == 0:
                continue

            total_bs = kv_score_input.numel() // (num_verify_tokens * head_dim * 2)
            layer_bs = min(
                actual_bs,
                ctx.seq_lens.shape[0],
                ctx.req_pool_indices.shape[0],
                layer_state.pre_state.shape[0],
                total_bs,
            )
            if layer_bs == 0:
                continue

            accept_lens_layer = accept_lens_i64[:layer_bs]
            seq_lens_before = ctx.seq_lens[:layer_bs].to(torch.int64)
            req_pool_indices = ctx.req_pool_indices[:layer_bs]
            kv_score_steps = kv_score_input.view(total_bs, num_verify_tokens, head_dim * 2)[
                :layer_bs
            ]
            bias = compressor.ape.view(128, head_dim).to(kv_score_steps.dtype)

            pre_state = layer_state.pre_state[:layer_bs]
            run_max = pre_state[:, :head_dim].to(kv_score_steps.dtype).clone()
            run_sum = pre_state[:, head_dim : 2 * head_dim].to(
                kv_score_steps.dtype
            ).clone()
            run_kv = pre_state[:, 2 * head_dim :].to(kv_score_steps.dtype).clone()
            if layer_bs != actual_bs or total_bs != actual_bs:
                _debug_online_c128_event(
                    "A",
                    "deepseek_v4_backend:update_online_c128_state_after_mtp_verify",
                    "batch size mismatch before commit",
                    {
                        "layer_id": int(compressor.layer_id),
                        "actual_bs": int(actual_bs),
                        "layer_bs": int(layer_bs),
                        "total_bs": int(total_bs),
                        "ctx_seq_lens_bs": int(ctx.seq_lens.shape[0]),
                        "pre_state_bs": int(layer_state.pre_state.shape[0]),
                        "kv_score_shape": list(kv_score_input.shape),
                    },
                )

            start_positions = (seq_lens_before % 128).to(torch.int64)
            for step in range(num_verify_tokens):
                active = step < accept_lens_layer
                if not active.any():
                    break

                pos = (start_positions + step) % 128
                kv_step = kv_score_steps[:, step, :head_dim]
                score_step = kv_score_steps[:, step, head_dim:] + bias[pos]

                start_new = active & (pos == 0)
                if start_new.any():
                    run_kv[start_new] = kv_step[start_new]
                    run_max[start_new] = score_step[start_new]
                    run_sum[start_new] = 1.0

                cont = active & ~start_new
                if cont.any():
                    old_max = run_max[cont]
                    old_sum = run_sum[cont]
                    old_kv = run_kv[cont]
                    new_score = score_step[cont]
                    new_kv = kv_step[cont]
                    new_max = torch.maximum(old_max, new_score)
                    old_sum_scaled = old_sum * torch.exp(old_max - new_max)
                    new_exp = torch.exp(new_score - new_max)
                    new_sum = old_sum_scaled + new_exp
                    run_kv[cont] = (
                        old_kv * old_sum_scaled + new_kv * new_exp
                    ) / new_sum
                    run_max[cont] = new_max
                    run_sum[cont] = new_sum

                closed = active & (pos == 127)
                if closed.any():
                    run_max[closed] = 0
                    run_sum[closed] = 0
                    run_kv[closed] = 0

            final_seq_lens = seq_lens_before + accept_lens_layer
            has_partial = (accept_lens_layer > 0) & ((final_seq_lens % 128) != 0)
            if not has_partial.any():
                continue

            chunk_starts = ((final_seq_lens[has_partial] - 1) // 128) * 128
            slots = self._get_chunk_slot_indices(
                req_pool_indices=req_pool_indices[has_partial],
                chunk_starts=chunk_starts,
            )
            state_to_commit = torch.cat(
                [
                    run_max[has_partial],
                    run_sum[has_partial],
                    run_kv[has_partial],
                ],
                dim=-1,
            ).to(layer_state.pre_state.dtype)
            _debug_online_c128_event(
                "A",
                "deepseek_v4_backend:update_online_c128_state_after_mtp_verify",
                "commit layer states",
                {
                    "layer_id": int(compressor.layer_id),
                    "layer_bs": int(layer_bs),
                    "num_commit": int(has_partial.sum().item()),
                    "seq_before_head": seq_lens_before[:8].detach().cpu().tolist(),
                    "accept_lens_head": accept_lens_layer[:8].detach().cpu().tolist(),
                    "final_seq_head": final_seq_lens[:8].detach().cpu().tolist(),
                    "chunk_starts_head": chunk_starts[:8].detach().cpu().tolist(),
                    "slots_head": slots[:8].detach().cpu().tolist(),
                },
            )
            main_pool = self.token_to_kv_pool.get_attention_compress_states(compressor.layer_id)
            main_pool.kv_score_buffer.kv_score[slots] = state_to_commit
            _debug_online_c128_sync(
                "deepseek_v4_backend:update_online_c128_state_after_mtp_verify",
                {"layer_id": int(compressor.layer_id), "num_commit": int(has_partial.sum().item())},
            )

        self._clear_online_c128_verify_context()
        _debug_online_c128_event(
            "A",
            "deepseek_v4_backend:update_online_c128_state_after_mtp_verify",
            "commit done",
            {"actual_bs": int(actual_bs)},
        )

    def init_forward_metadata_indexer(self, core_attn_metadata: DSV4AttnMetadata):
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=core_attn_metadata.page_table,
            c4_seq_lens=core_attn_metadata.c4_topk_lengths_raw,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            return DSV4RawDecodeMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=True,
        )

        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
            num_tokens=num_tokens,
            seq_lens=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            padded_num_tokens=out_cache_loc.shape[0],
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        if envs.SGLANG_PREP_IN_CUDA_GRAPH.get():
            assert out_cache_loc is not None
            if not hasattr(self, "extend_seq_lens_buffer"):
                self.extend_seq_lens_buffer = torch.tensor(
                    [self.speculative_num_draft_tokens] * 1025, device=self.device
                )
            extend_seq_lens = self.extend_seq_lens_buffer[: len(seq_lens)]
            seq_lens_host = self._to_pinned_cpu_i64(seq_lens)
            extend_seq_lens_host = torch.full(
                (len(seq_lens),),
                self.speculative_num_draft_tokens,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )

            return DSV4RawVerifyMetadata(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                extend_seq_lens=extend_seq_lens,
                seq_lens_host=seq_lens_host,
                extend_seq_lens_host=extend_seq_lens_host,
            )
        else:
            seq_lens_cpu = seq_lens.tolist()
            return self.init_forward_metadata_target_verify_old(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        seq_lens_cpu = [x + self.speculative_num_draft_tokens for x in seq_lens_cpu]
        extend_seq_lens_cpu = [self.speculative_num_draft_tokens] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = self.speculative_num_draft_tokens * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.speculative_num_draft_tokens
        seq_lens = seq_lens + self.speculative_num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens
        seq_lens_host = raw_metadata.seq_lens_host
        extend_seq_lens_host = raw_metadata.extend_seq_lens_host
        assert seq_lens_host is not None
        assert extend_seq_lens_host is not None
        seq_lens_planner = seq_lens_host + self.speculative_num_draft_tokens

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_planner,
            extend_lens=extend_seq_lens_host,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_draft_tokens * bs,
        )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = self.init_forward_metadata_indexer(core_attn_metadata)

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c4_compress_metadata=create(compress_ratio=4),
            c128_compress_metadata=create(compress_ratio=128),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_bs: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_bs] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_bs * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            self._clear_online_c128_verify_context()
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
       assert self.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            # DSv4 bakes this step's KV write target (c4/c128) into metadata,
            # so slice the shared multi-step out_cache_loc now rather than at
            # forward time.
            out_cache_loc = forward_batch.out_cache_loc
            if self.topk > 0 and self.speculative_num_steps > 1:
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            self._clear_online_c128_verify_context()
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            self._set_online_c128_verify_context(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
            )
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            self._clear_online_c128_verify_context()
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = forward_batch.forward_mode.is_draft_extend(include_v2=True)
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[DSV4Metadata, DSV4RawDecodeMetadata, DSV4RawVerifyMetadata],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_bs = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        assert req_pool_indices.size(0) == bs
        assert seq_lens.size(0) == bs

        bucket = _GraphBucket.of(forward_mode)
        raw_type: Optional[type] = None
        if bucket == _GraphBucket.DECODE_OR_IDLE:
            metadata = self.init_forward_metadata_decode(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=torch.zeros_like(seq_lens),
            )
            raw_type = DSV4RawDecodeMetadata
        elif bucket == _GraphBucket.TARGET_VERIFY:
            out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
            raw_type = DSV4RawVerifyMetadata
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_bs = num_tokens // bs
            metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError(f"{forward_mode=} not supported yet")

        self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = metadata
        self.forward_metadata = metadata
        if raw_type is not None:
            self._current_capture_raw = (
                metadata if isinstance(metadata, raw_type) else None
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        bucket = _GraphBucket.of(forward_mode)

        # FIXME: see cuda_graph_runner — this attribute is set out-of-band.
        fb = self._replay_forward_batch
        out_cache_loc = fb.out_cache_loc
        actual_forward_mode = fb.forward_mode

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            self._set_online_c128_verify_context(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
            )
            assert out_cache_loc is not None
            num_tokens = self.speculative_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            self._clear_online_c128_verify_context()
            num_tokens_per_bs = self.draft_extend_num_tokens_per_bs
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_bs=num_tokens_per_bs,
                use_prefill_cuda_graph=True,
            )
        else:
            self._clear_online_c128_verify_context()
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=bs, temp_metadata=temp_metadata, bucket=bucket
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c1_flashmla_metadata = _create_flashmla_metadata()
            core.c4_flashmla_metadata = _create_flashmla_metadata()
            core.c128_flashmla_metadata = _create_flashmla_metadata()

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        raw_loc = forward_batch.out_cache_loc
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_k=swa_k,
            )
        else:
            swa_k_pack = quant_to_nope_fp8_rope_bf16_pack_triton(swa_k)
            self.token_to_kv_pool.set_swa_key_buffer_radix(
                layer_id=layer_id,
                raw_loc=raw_loc,
                cache_nope_fp8_rope_bf16_pack=swa_k_pack,
            )

    def _maybe_upgrade_forward_metadata(self) -> None:
        # With SGLANG_PREP_IN_CUDA_GRAPH=1, init_forward_metadata_*
        # returns a Raw metadata that only carries a few tensors. The
        # full DSV4Metadata (including c4/c128 compress + core_attn +
        # indexer metadata) must be materialized before any caller that
        # touches those fields. For 1.6T the first two layers have
        # compress_ratio=128, so forward_core_compressor / forward_c4_indexer
        # can fire before attn_backend.forward(), and must trigger the
        # upgrade themselves.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        self._maybe_upgrade_forward_metadata()

        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                page_sizes = {
                    4: token_to_kv_pool.page_size // 4,
                    128: token_to_kv_pool.page_size // 128,
                }
                extra_k_cache = extra_k_cache[
                    :, : page_sizes[compress_ratio] * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    page_sizes[compress_ratio],
                    1,
                    k_cache_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if self.mtp_enabled:
                if swa_page_indices.shape[0] != q.shape[0]:
                    swa_page_indices = _pad_tensor_to_size(
                        swa_page_indices, q.shape[0], value=0
                    )

                if swa_topk_lengths.shape[0] != q.shape[0]:
                    swa_topk_lengths = _pad_tensor_to_size(
                        swa_topk_lengths, q.shape[0], value=1
                    )

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            import flash_mla

            o = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=flashmla_metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata()
            core_attn_metadata.init_flashmla_related()
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c1_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        return swa_indices


class DeepseekV4MultiStepBackend(DeepseekV4AttnBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4AttnBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4AttnBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )
    
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        if self.speculative_num_steps == 1:
            return
        self.attn_backends[0]._replay_forward_batch = forward_batch
        self.attn_backends[0].init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=ForwardMode.DECODE,
            spec_info=forward_batch.spec_info,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )
        self.attn_backends[0]._replay_forward_batch = None
        temp_metadata = self.attn_backends[0].forward_metadata

        for i in range(1, self.speculative_num_steps - 1):
            self.attn_backends[i].replay_cuda_graph_metadata_from(
                bs=bs,
                temp_metadata=temp_metadata,
                bucket=_GraphBucket.DECODE_OR_IDLE,
            )


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )

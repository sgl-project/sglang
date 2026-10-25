"""Runtime plans adapting the external inference.msa_v1 wrappers.

These adapters manage indexer/main-attention buffers, metadata and native or
FP8 fallback execution. The existing fmha_sm100 adapter remains in msa.py.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os

import torch

from sglang.kernels.ops.attention.msa_v1.compact_dequant import (
    dequant_nvfp4_page_table,
    gather_dequant_nvfp4_topk_pages_pair,
)
from sglang.kernels.ops.attention.msa_v1.decode_plan import update_msa_v1_decode_plan
from sglang.kernels.ops.attention.msa_v1.flashinfer_decode import FlashInferQ8K8Decode
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# MSA raises on a kernel without an AOT object unless this is set (any value);
# sglang ships no AOT objects, so JIT stays required until it does.
os.environ.setdefault("FMHA_SM100_ALLOW_JIT", "1")

# minimax-msa installs generic top-level packages (inference, datas, msa_v1);
# name any other distribution that also provides one of them.
_MSA_DISTRIBUTION = "minimax-msa"
_MSA_TOP_LEVEL_PACKAGES = ("inference", "datas", "msa_v1")


def _find_msa_package_conflicts() -> dict[str, list[str]]:
    providers = importlib.metadata.packages_distributions()
    return {
        name: providers[name]
        for name in _MSA_TOP_LEVEL_PACKAGES
        if providers.get(name, [_MSA_DISTRIBUTION]) != [_MSA_DISTRIBUTION]
    }


_MSA_PACKAGE_CONFLICTS = _find_msa_package_conflicts()
if _MSA_PACKAGE_CONFLICTS:
    logger.error(
        "MiniMax MSA top-level packages are also provided by other distributions: %s. "
        "MSA kernels may fail to import or load the wrong modules; install %s last "
        "or remove the conflicting distributions.",
        _MSA_PACKAGE_CONFLICTS,
        _MSA_DISTRIBUTION,
    )


def _describe_import_failure(error: Exception | None) -> str:
    if not _MSA_PACKAGE_CONFLICTS:
        return str(error)
    return f"{error} (conflicting top-level packages: {_MSA_PACKAGE_CONFLICTS})"


try:
    from inference.msa_v1.indexer.decode.q8kv4 import (
        BatchDecodeIndexerWithPagedKVCacheWrapper as _DecodeIndexerWrapper,
    )

    MSA_V1_DECODE_INDEXER_AVAILABLE = True
    _DECODE_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _DecodeIndexerWrapper = None
    MSA_V1_DECODE_INDEXER_AVAILABLE = False
    _DECODE_IMPORT_ERROR = error

try:
    from inference.msa_v1.indexer.decode.q8kv8 import (
        BatchDecodeIndexerWithPagedKVCacheWrapper as _DecodeIndexerFP8Wrapper,
    )

    MSA_V1_DECODE_INDEXER_FP8_AVAILABLE = True
    _DECODE_FP8_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _DecodeIndexerFP8Wrapper = None
    MSA_V1_DECODE_INDEXER_FP8_AVAILABLE = False
    _DECODE_FP8_IMPORT_ERROR = error

# Both decode indexer wrappers import it, so it is present whenever either one is.
try:
    from inference.msa_v1.indexer.decode import (
        BatchDecodeIndexerPlan as _SharedDecodeIndexerPlan,
    )
except Exception:
    _SharedDecodeIndexerPlan = None

try:
    from inference.msa_v1.indexer.prefill.q8kv8 import (
        BatchPrefillIndexerWithPagedKVCacheWrapper as _PrefillIndexerWrapper,
    )

    MSA_V1_PREFILL_INDEXER_AVAILABLE = True
    _PREFILL_INDEXER_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _PrefillIndexerWrapper = None
    MSA_V1_PREFILL_INDEXER_AVAILABLE = False
    _PREFILL_INDEXER_IMPORT_ERROR = error

try:
    from inference.dequant import dequantize_nvfp4_to_fp8 as _dequantize_nvfp4_to_fp8

    MSA_V1_PREFILL_DEQUANT_AVAILABLE = True
    _PREFILL_DEQUANT_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _dequantize_nvfp4_to_fp8 = None
    MSA_V1_PREFILL_DEQUANT_AVAILABLE = False
    _PREFILL_DEQUANT_IMPORT_ERROR = error

try:
    from inference.msa_v1.attention.prefill.q8kv8 import (
        BatchPrefillWithPagedKVCacheWrapper as _PrefillAttentionWrapper,
    )

    MSA_V1_PREFILL_ATTENTION_AVAILABLE = True
    _PREFILL_ATTENTION_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _PrefillAttentionWrapper = None
    MSA_V1_PREFILL_ATTENTION_AVAILABLE = False
    _PREFILL_ATTENTION_IMPORT_ERROR = error

try:
    from inference.msa_v1.attention.prefill.q8kv4 import (
        BatchPrefillWithPagedKVCacheWrapper as _PrefillAttentionQ4Wrapper,
    )

    MSA_V1_PREFILL_ATTENTION_Q4_AVAILABLE = True
    _PREFILL_ATTENTION_Q4_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _PrefillAttentionQ4Wrapper = None
    MSA_V1_PREFILL_ATTENTION_Q4_AVAILABLE = False
    _PREFILL_ATTENTION_Q4_IMPORT_ERROR = error

try:
    from inference.msa_v1.attention.decode.q8kv4 import (
        BatchDecodeWithPagedKVCacheWrapper as _DecodeAttentionWrapper,
    )
    from inference.msa_v1.attention.decode.q8kv4.jit import (
        _cuda_version as _qmul4_cuda_version,
    )
    from inference.msa_v1.attention.decode.q8kv4.jit import (
        _cutlass_root as _qmul4_cutlass_root,
    )
    from inference.msa_v1.attention.decode.q8kv4.jit import (
        _target_arch as _qmul4_target_arch,
    )

    MSA_V1_DECODE_ATTENTION_AVAILABLE = True
    _DECODE_ATTENTION_IMPORT_ERROR: Exception | None = None
except Exception as error:
    _DecodeAttentionWrapper = None
    _qmul4_cuda_version = None
    _qmul4_cutlass_root = None
    _qmul4_target_arch = None
    MSA_V1_DECODE_ATTENTION_AVAILABLE = False
    _DECODE_ATTENTION_IMPORT_ERROR = error


_PAGE_SIZE = 128
# MSA-v1 decode indexers accept 1..16 queries and 1, 2 or 4 index heads per launch.
_MAX_DECODE_QUERY_LENGTH = 16
_INDEX_HEADS = (1, 2, 4)
_TOPK = 16


def msa_v1_qmul4_decode_capability() -> tuple[bool, str | None]:
    """Probe prerequisites without compiling the Q8KV4 decode extension."""
    if envs.SGLANG_DISABLE_MSA_V1_Q8KV4.get():
        return False, "disabled by SGLANG_DISABLE_MSA_V1_Q8KV4=1"
    if not MSA_V1_DECODE_ATTENTION_AVAILABLE:
        return False, (
            "Q8KV4 wrapper import failed: "
            f"{_describe_import_failure(_DECODE_ATTENTION_IMPORT_ERROR)}"
        )
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    try:
        major, minor = torch.cuda.get_device_capability()
        if major not in (10,):
            return False, f"QMUL4 kernel requires SM100/SM103, got sm{major}{minor}"
        version = _qmul4_cuda_version()
        if version < (13, 4):
            return (
                False,
                f"QMUL4 requires CUDA compiler >=13.4, got {version[0]}.{version[1]}",
            )
        _qmul4_target_arch()
        _qmul4_cutlass_root()
    except Exception as error:
        return False, str(error)
    return True, None


def _require_available(
    available: bool, component: str, error: Exception | None
) -> None:
    if not available:
        raise RuntimeError(
            f"MiniMax MSA MSA-v1 {component} is unavailable: "
            f"{_describe_import_failure(error)}"
        ) from error


def build_msa_v1_page_table(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    *,
    max_seqlen_k: int,
) -> torch.Tensor:
    """Build the logical-to-physical 128-token page table used by MSA-v1."""
    max_pages = max(1, math.ceil(int(max_seqlen_k) / _PAGE_SIZE))
    if max_pages > 8192:
        raise ValueError(
            f"MSA-v1 supports at most {8192 * _PAGE_SIZE} tokens, got {max_seqlen_k}."
        )
    last_column = (max_pages - 1) * _PAGE_SIZE
    if last_column >= req_to_token.shape[1]:
        raise ValueError(
            "MSA-v1 page table exceeds req_to_token capacity: "
            f"last column={last_column}, capacity={req_to_token.shape[1]}."
        )
    columns = torch.arange(
        max_pages, device=req_to_token.device, dtype=torch.long
    ).mul_(_PAGE_SIZE)
    slots = req_to_token[req_pool_indices.long()][:, columns]
    return (
        torch.div(slots, _PAGE_SIZE, rounding_mode="floor").to(torch.int32).contiguous()
    )


def _check_index_heads(num_index_heads: int) -> int:
    if num_index_heads not in _INDEX_HEADS:
        raise ValueError(
            f"MSA-v1 indexers support {_INDEX_HEADS} local index heads, "
            f"got {num_index_heads}."
        )
    return int(num_index_heads)


class MSAV1DecodeIndexerPlan:
    """Graph-safe adapter from SGLang decode rows to MSA-v1 multi-head indexers."""

    def __init__(
        self,
        *,
        batch_size: int,
        query_length: int,
        max_pages: int,
        device: torch.device,
        use_cuda_graph: bool,
        num_index_heads: int = 1,
        prefer_qmul4: bool = True,
        fp8_fallback_scratch: torch.Tensor | None = None,
    ) -> None:
        self.num_index_heads = _check_index_heads(num_index_heads)
        if prefer_qmul4:
            _require_available(
                MSA_V1_DECODE_INDEXER_AVAILABLE,
                "Q8K4 decode indexer",
                _DECODE_IMPORT_ERROR,
            )
        elif fp8_fallback_scratch is None:
            raise ValueError("Q8K8 indexer fallback requires fixed FP8 scratch")
        if fp8_fallback_scratch is not None:
            _require_available(
                MSA_V1_DECODE_INDEXER_FP8_AVAILABLE,
                "Q8K8 decode indexer fallback",
                _DECODE_FP8_IMPORT_ERROR,
            )
        if min(batch_size, query_length, max_pages) <= 0:
            raise ValueError("batch_size, query_length, and max_pages must be positive")
        self.batch_size = int(batch_size)
        self.query_length = int(query_length)
        self.max_pages = int(max_pages)
        self.device = torch.device(device)
        self.use_cuda_graph = bool(use_cuda_graph)
        self.use_qmul4 = bool(prefer_qmul4)
        self.fp8_fallback_scratch = fp8_fallback_scratch
        self.qmul4_fallback_reason: str | None = None
        # The attention plan and the MSA wrappers read these; refresh() rewrites them.
        self.page_table = torch.empty(
            self.batch_size,
            self.max_pages,
            dtype=torch.int32,
            device=self.device,
        )
        self.final_seq_lens = torch.empty(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        self.chunks = [
            (start, min(start + _MAX_DECODE_QUERY_LENGTH, self.query_length))
            for start in range(0, self.query_length, _MAX_DECODE_QUERY_LENGTH)
        ]
        self._chunk_seq_lens_buffer = torch.empty(
            len(self.chunks), self.batch_size, dtype=torch.int32, device=self.device
        )
        # Wrappers cache plans by tensor identity, so keep the row views alive.
        self.chunk_seq_lens = list(self._chunk_seq_lens_buffer.unbind(0))
        # A graph plan binds each chunk's wrappers once to one MSA schedule, which
        # plan() and refresh() then update in place.
        self.shared_plans = (
            [
                _SharedDecodeIndexerPlan(
                    seq_lens,
                    max_pages=self.max_pages,
                    num_index_heads=self.num_index_heads,
                    query_length=end - start,
                )
                for (start, end), seq_lens in zip(self.chunks, self.chunk_seq_lens)
            ]
            if self.use_cuda_graph
            else None
        )
        self.wrappers = []
        self.fp8_wrappers = []
        # A single chunk reads Q in place; longer queries stage each chunk.
        self.query_buffers: list[torch.Tensor | None] = []
        for chunk_idx, ((start, end), seq_lens) in enumerate(
            zip(self.chunks, self.chunk_seq_lens)
        ):
            buffers = {"use_cuda_graph": self.use_cuda_graph}
            if self.use_cuda_graph:
                buffers.update(
                    page_table_buffer=self.page_table, seq_lens_buffer=seq_lens
                )
            wrapper = _DecodeIndexerWrapper(**buffers) if self.use_qmul4 else None
            fp8_wrapper = (
                _DecodeIndexerFP8Wrapper(**buffers)
                if fp8_fallback_scratch is not None
                else None
            )
            if self.use_cuda_graph:
                for bound in (wrapper, fp8_wrapper):
                    if bound is not None:
                        bound.plan(
                            self.page_table,
                            seq_lens,
                            num_index_heads=self.num_index_heads,
                            query_length=end - start,
                            shared_plan=self.shared_plans[chunk_idx],
                        )
            self.wrappers.append(wrapper)
            if fp8_wrapper is not None:
                self.fp8_wrappers.append(fp8_wrapper)
            self.query_buffers.append(
                None
                if len(self.chunks) == 1
                else torch.empty(
                    self.batch_size,
                    end - start,
                    self.num_index_heads,
                    128,
                    dtype=torch.float8_e4m3fn,
                    device=self.device,
                )
            )
        self.result = torch.empty(
            self.num_index_heads,
            self.batch_size * self.query_length,
            _TOPK,
            dtype=torch.int32,
            device=self.device,
        )

    def plan(self, page_table: torch.Tensor, final_seq_lens: torch.Tensor) -> None:
        if tuple(page_table.shape) != (self.batch_size, self.max_pages):
            raise ValueError(
                "MSA-v1 decode page table shape changed from the graph bucket: "
                f"expected {(self.batch_size, self.max_pages)}, got {tuple(page_table.shape)}."
            )
        self.page_table.copy_(page_table)
        self.final_seq_lens.copy_(final_seq_lens.to(torch.int32))
        for chunk_idx, ((start, end), seq_lens) in enumerate(
            zip(self.chunks, self.chunk_seq_lens)
        ):
            # Each chunk sees the KV length through its own last query.
            torch.sub(self.final_seq_lens, self.query_length - end, out=seq_lens)
            if self.shared_plans is not None:
                self.shared_plans[chunk_idx].update()
                continue
            kwargs = {
                "num_index_heads": self.num_index_heads,
                "query_length": end - start,
            }
            if self.wrappers[chunk_idx] is not None:
                self.wrappers[chunk_idx].plan(self.page_table, seq_lens, **kwargs)
            if self.fp8_wrappers:
                self.fp8_wrappers[chunk_idx].plan(self.page_table, seq_lens, **kwargs)

    def refresh(
        self,
        *,
        attention_plan: MSAV1DecodeAttentionPlan,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        is_target_verify: bool,
    ) -> None:
        if not self.use_cuda_graph:
            raise ValueError("refresh() needs a use_cuda_graph=True plan")
        if (
            attention_plan.page_table is not self.page_table
            or attention_plan.final_seq_lens is not self.final_seq_lens
        ):
            raise ValueError(
                "the attention plan must share the indexer's page_table and "
                "final_seq_lens"
            )
        fp8_attention = attention_plan.fp8_attention
        update_msa_v1_decode_plan(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            query_length=self.query_length,
            # A verify forward's seq_lens exclude its query_length draft tokens.
            seq_len_offset=self.query_length if is_target_verify else 0,
            page_table=self.page_table,
            final_seq_lens=self.final_seq_lens,
            chunk_seq_lens=self._chunk_seq_lens_buffer,
            query_seq_lens=attention_plan.query_seq_lens,
            flashinfer_seq_lens=(
                None if fp8_attention is None else fp8_attention.seq_lens
            ),
        )
        # Each MSA schedule reads the chunk lengths just written.
        for shared_plan in self.shared_plans:
            shared_plan.update()

    def run(
        self,
        q: torch.Tensor,
        packed_k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        global_scale: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.ndim != 3 or q.shape[2] != 128:
            raise ValueError("q must have shape [batch * query_length, heads, 128]")
        if q.shape[0] != self.batch_size * self.query_length:
            raise ValueError(
                f"expected {self.batch_size * self.query_length} query rows, got {q.shape[0]}"
            )
        if q.dtype != torch.float8_e4m3fn:
            raise TypeError("MSA-v1 decode index Q must be torch.float8_e4m3fn")
        if q.shape[1] != self.num_index_heads:
            raise ValueError(
                f"expected {self.num_index_heads} index heads, got {q.shape[1]}"
            )
        result = self.result
        if out is not None:
            if (
                tuple(out.shape) != tuple(result.shape)
                or out.dtype != torch.int32
                or out.device != q.device
            ):
                raise ValueError(
                    "MSA-v1 indexer out must match the topk shape, device and int32 dtype"
                )
            result = out
        q_bqhd = q.view(self.batch_size, self.query_length, self.num_index_heads, 128)
        scale_fp8 = k_scale.view(torch.float8_e4m3fn)
        if self.use_qmul4:
            try:
                self._run_chunks(
                    q_bqhd,
                    packed_k_cache,
                    result,
                    wrappers=self.wrappers,
                    k_scale=scale_fp8,
                )
            except Exception as error:
                if not self.fp8_wrappers:
                    raise
                self.use_qmul4 = False
                self.qmul4_fallback_reason = str(error)
                logger.warning(
                    "MSA-v1 Q8KV4 decode indexer failed (%s); "
                    "using the slower FP8 fallback for this plan.",
                    error,
                )
        if not self.use_qmul4:
            dequant_nvfp4_page_table(
                packed_k_cache,
                scale_fp8,
                global_scale,
                self.page_table,
                self.final_seq_lens,
                self.fp8_fallback_scratch,
            )
            self._run_chunks(
                q_bqhd,
                self.fp8_fallback_scratch,
                result,
                wrappers=self.fp8_wrappers,
            )
        # Preserve the caller's strides for attention's alias-aware copy_.
        return result

    def _run_chunks(
        self,
        q_bqhd: torch.Tensor,
        k_cache: torch.Tensor,
        result: torch.Tensor,
        *,
        wrappers: list,
        k_scale: torch.Tensor | None = None,
    ) -> None:
        kwargs = {} if k_scale is None else {"k_scale": k_scale}
        if len(self.chunks) == 1:
            wrapper = wrappers[0]
            if result.is_contiguous():
                wrapper.run(q_bqhd.contiguous(), k_cache, out=result, **kwargs)
            else:
                result.copy_(wrapper.run(q_bqhd.contiguous(), k_cache, **kwargs))
            return
        result_hbqk = result.unflatten(1, (self.batch_size, self.query_length))
        for (start, end), wrapper, query_buffer in zip(
            self.chunks, wrappers, self.query_buffers
        ):
            query_buffer.copy_(q_bqhd[:, start:end])
            chunk_topk = wrapper.run(query_buffer, k_cache, **kwargs)
            result_hbqk[:, :, start:end].copy_(
                chunk_topk.view(
                    self.num_index_heads, self.batch_size, end - start, _TOPK
                )
            )


def _shared_or_new_int32(
    *,
    name: str,
    shared: torch.Tensor | None,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if shared is None:
        return torch.empty(shape, dtype=torch.int32, device=device)
    if (
        shared.dtype != torch.int32
        or tuple(shared.shape) != shape
        or not shared.is_contiguous()
    ):
        raise ValueError(
            f"MSA-v1 decode {name} must be a contiguous int32 {shape} tensor"
        )
    return shared


class MSAV1DecodeAttentionPlan:
    """Q8KV4 decode with a graph-safe Q8K8 gather/dequant fallback."""

    def __init__(
        self,
        *,
        batch_size: int,
        query_length: int = 1,
        max_pages: int,
        num_q_heads: int,
        num_kv_heads: int,
        device: torch.device,
        sm_scale: float,
        prefer_qmul4: bool,
        workspace_buffer: torch.Tensor | None = None,
        page_table_buffer: torch.Tensor | None = None,
        final_seq_lens_buffer: torch.Tensor | None = None,
    ) -> None:
        if (
            min(
                batch_size,
                query_length,
                max_pages,
                num_q_heads,
                num_kv_heads,
            )
            <= 0
        ):
            raise ValueError("MSA-v1 decode plan dimensions must be positive")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("MSA-v1 Q heads must be divisible by KV heads")
        self.batch_size = int(batch_size)
        self.query_length = int(query_length)
        self.num_query_rows = self.batch_size * self.query_length
        self.max_pages = int(max_pages)
        self.num_q_heads = int(num_q_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.device = torch.device(device)
        self.sm_scale = float(sm_scale)
        self.gqa_group_size = self.num_q_heads // self.num_kv_heads

        # May alias the indexer plan's buffers, which the fused update then writes once.
        self.page_table = _shared_or_new_int32(
            name="page_table",
            shared=page_table_buffer,
            shape=(self.batch_size, self.max_pages),
            device=self.device,
        )
        self.final_seq_lens = _shared_or_new_int32(
            name="final_seq_lens",
            shared=final_seq_lens_buffer,
            shape=(self.batch_size,),
            device=self.device,
        )
        self.query_seq_lens = torch.empty(
            self.num_query_rows, dtype=torch.int32, device=self.device
        )
        query_rows = torch.arange(
            self.num_query_rows, dtype=torch.int64, device=self.device
        )
        self.query_to_request = torch.div(
            query_rows, self.query_length, rounding_mode="floor"
        )
        self.query_from_end = (
            self.query_length - 1 - query_rows.remainder(self.query_length)
        ).to(torch.int32)
        self.topk_qht = torch.empty(
            self.num_query_rows,
            self.num_kv_heads,
            _TOPK,
            dtype=torch.int32,
            device=self.device,
        )

        self.qmul4_wrapper = None
        self.qmul4_fallback_reason = None
        if prefer_qmul4 and self.gqa_group_size in (8, 16):
            try:
                wrapper = _DecodeAttentionWrapper()
                wrapper.plan(
                    self.topk_qht,
                    self.page_table,
                    self.final_seq_lens,
                    q_len_per_req=self.query_length,
                    num_q_heads=self.num_q_heads,
                    num_kv_heads=self.num_kv_heads,
                    sm_scale=self.sm_scale,
                )
                self.qmul4_wrapper = wrapper
            except Exception as error:
                self.qmul4_fallback_reason = str(error)
        elif prefer_qmul4:
            self.qmul4_fallback_reason = (
                "Q8KV4 decode requires 8 or 16 local Q heads per KV head, got "
                f"{self.gqa_group_size}."
            )

        self.compact_k = None
        self.compact_v = None
        self.fp8_attention = None
        if self.qmul4_wrapper is None:
            if self.qmul4_fallback_reason is not None:
                logger.warning(
                    "MSA-v1 attention uses external FlashInfer Q8K8 decode: %s",
                    self.qmul4_fallback_reason,
                )
            if self.gqa_group_size not in (1, 2, 4, 8, 16):
                raise ValueError(
                    "MSA-v1 Q8K8 decode fallback supports 1, 2, 4, 8, or 16 local Q heads "
                    f"per KV head, got {self.gqa_group_size}."
                )
            compact_shape = (
                self.num_query_rows,
                self.num_kv_heads,
                _TOPK,
                _PAGE_SIZE,
                128,
            )
            self.compact_k = torch.empty(
                compact_shape, dtype=torch.float8_e4m3fn, device=self.device
            )
            self.compact_v = torch.empty_like(self.compact_k)
            self.fp8_attention = FlashInferQ8K8Decode(
                self.compact_k,
                self.compact_v,
                num_q_heads=self.num_q_heads,
                sm_scale=self.sm_scale,
                workspace_buffer=workspace_buffer,
            )

    @property
    def uses_qmul4(self) -> bool:
        return self.qmul4_wrapper is not None

    def plan(self, page_table: torch.Tensor, final_seq_lens: torch.Tensor) -> None:
        if tuple(page_table.shape) != tuple(self.page_table.shape):
            raise ValueError(
                "MSA-v1 decode attention page-table shape changed from the plan: "
                f"expected {tuple(self.page_table.shape)}, got {tuple(page_table.shape)}."
            )
        if tuple(final_seq_lens.shape) != (self.batch_size,):
            raise ValueError("MSA-v1 final_seq_lens must have shape [batch]")
        self.page_table.copy_(page_table)
        self.final_seq_lens.copy_(final_seq_lens.to(torch.int32))
        torch.index_select(
            self.final_seq_lens,
            0,
            self.query_to_request,
            out=self.query_seq_lens,
        )
        self.query_seq_lens.sub_(self.query_from_end)
        self.query_seq_lens.clamp_min_(0)
        if self.fp8_attention is not None:
            self.fp8_attention.update_lengths(self.query_seq_lens)

    def run(
        self,
        q: torch.Tensor,
        topk_indices: torch.Tensor,
        packed_k_cache: torch.Tensor,
        packed_v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        expected_q = (self.num_query_rows, self.num_q_heads, 128)
        expected_topk = (self.num_kv_heads, self.num_query_rows, _TOPK)
        if tuple(q.shape) != expected_q or q.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"MSA-v1 decode Q must be FP8 with shape {expected_q}, got "
                f"{tuple(q.shape)} {q.dtype}."
            )
        if not q.is_contiguous():
            raise ValueError("MSA-v1 decode Q must be contiguous")
        if tuple(topk_indices.shape) != expected_topk:
            raise ValueError(
                f"MSA-v1 topk must have shape {expected_topk}, got "
                f"{tuple(topk_indices.shape)}."
            )
        self.topk_qht.copy_(topk_indices.permute(1, 0, 2))
        if self.qmul4_wrapper is not None:
            return self.qmul4_wrapper.run(
                q,
                (packed_k_cache, packed_v_cache),
                kv_cache_sf=(k_scale, v_scale),
            )

        assert self.compact_k is not None
        assert self.compact_v is not None
        assert self.fp8_attention is not None
        gather_dequant_nvfp4_topk_pages_pair(
            packed_k_cache,
            packed_v_cache,
            k_scale,
            v_scale,
            global_scale,
            self.page_table,
            self.topk_qht,
            self.query_seq_lens,
            self.compact_k,
            self.compact_v,
            q_len_per_req=self.query_length,
        )
        return self.fp8_attention.run(q)


class MSAV1PrefillPlan:
    """Q8K8 prefill indexer and native Q8KV4 attention with a Q8KV8 fallback."""

    def __init__(
        self,
        *,
        cu_seqlens_q: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        total_q: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        prefer_qmul4: bool = False,
        num_index_heads: int = 1,
    ) -> None:
        self.num_index_heads = _check_index_heads(num_index_heads)
        _require_available(
            MSA_V1_PREFILL_INDEXER_AVAILABLE,
            "prefill indexer",
            _PREFILL_INDEXER_IMPORT_ERROR,
        )
        self.cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
        self.source_page_table = page_table
        self.prefer_qmul4 = bool(prefer_qmul4)
        self.seq_lens = seq_lens.to(torch.int32).contiguous()
        self.cu_seqlens_k = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=seq_lens.device),
                self.seq_lens.cumsum(0),
            )
        ).to(torch.int32)
        physical_pages, inverse = torch.unique(
            page_table.reshape(-1).long(), sorted=True, return_inverse=True
        )
        self.physical_pages = physical_pages
        self.page_table = inverse.view_as(page_table).to(torch.int32).contiguous()
        self.total_q = int(total_q)
        self.total_k = int(self.seq_lens.sum().item())
        self.total_rows = int(
            torch.div(
                self.seq_lens + _PAGE_SIZE - 1,
                _PAGE_SIZE,
                rounding_mode="floor",
            )
            .sum()
            .item()
        )
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        self.indexer = _PrefillIndexerWrapper()
        self.indexer.plan(
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.page_table,
            total_q=self.total_q,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            num_index_heads=self.num_index_heads,
        )
        self.attention = None

    def run_indexer(
        self,
        idx_q: torch.Tensor,
        packed_k_cache: torch.Tensor,
        k_scale: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(idx_q.shape) != (self.total_q, self.num_index_heads, 128):
            raise ValueError(
                "MSA-v1 prefill index Q must match the planned token and head counts"
            )
        packed_pages = packed_k_cache.index_select(0, self.physical_pages)
        scale_pages = k_scale.index_select(0, self.physical_pages).view(
            torch.float8_e4m3fn
        )
        dequantized_pages = self._dequantize_prefill_nvfp4(
            packed_pages, scale_pages, global_scale
        )
        paged_k = dequantized_pages.unsqueeze(1).contiguous()
        # The indexer's default output is shared across layers; give each call its own.
        topk = torch.empty(
            (self.num_index_heads, self.total_q, _TOPK),
            dtype=torch.int32,
            device=idx_q.device,
        )
        return self.indexer.run(idx_q.contiguous(), paged_k, out=topk)

    @staticmethod
    def _validate_global_scale_contract(global_scale: torch.Tensor) -> None:
        # MiniMaxM3Nvfp4MainKVPool owns an immutable device-side unit scale.
        # Avoid reading it back to the host on every sparse layer.
        if global_scale.numel() != 1 or global_scale.dtype != torch.float32:
            raise ValueError("MSA-v1 NVFP4 requires one FP32 global scale")

    @classmethod
    def _dequantize_prefill_nvfp4(
        cls,
        packed: torch.Tensor,
        scales: torch.Tensor,
        global_scale: torch.Tensor,
    ) -> torch.Tensor:
        if MSA_V1_PREFILL_DEQUANT_AVAILABLE:
            cls._validate_global_scale_contract(global_scale)
            return _dequantize_nvfp4_to_fp8(packed, scales)

        from sglang.srt.layers.quantization.kvfp4_tensor import NVFP4KVQuantizeUtil

        return NVFP4KVQuantizeUtil.dequantize(
            packed,
            scales,
            global_scale,
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)

    def run_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        *,
        sm_scale: float,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _require_available(
            MSA_V1_PREFILL_ATTENTION_AVAILABLE,
            "Q8K8 prefill attention",
            _PREFILL_ATTENTION_IMPORT_ERROR,
        )
        if q.ndim != 3 or q.shape[2] != 128:
            raise ValueError(
                "native MSA-v1 Q8K8 prefill requires Q shape "
                "[total_q, num_q_heads, 128]"
            )
        num_q_heads = q.shape[1]
        is_nvfp4 = k_cache.ndim == 4 and k_cache.dtype == torch.uint8
        if is_nvfp4:
            if k_cache.shape != v_cache.shape or tuple(k_cache.shape[2:]) != (
                _PAGE_SIZE,
                64,
            ):
                raise ValueError(
                    "MSA-v1 NVFP4 prefill requires equal packed K/V caches with "
                    "shape [pages, num_kv_heads, 128, 64]."
                )
            if k_scale is None or v_scale is None or global_scale is None:
                raise ValueError("MSA-v1 NVFP4 prefill requires K/V and global scales")
            num_kv_heads = k_cache.shape[1]
            if (
                self.prefer_qmul4
                and MSA_V1_PREFILL_ATTENTION_Q4_AVAILABLE
                and num_q_heads == num_kv_heads * 16
            ):
                self._validate_global_scale_contract(global_scale)
                self.attention = _PrefillAttentionQ4Wrapper()
                self.attention.plan(
                    topk_indices,
                    self.cu_seqlens_q,
                    self.cu_seqlens_k,
                    self.source_page_table,
                    total_k=self.total_k,
                    total_rows=self.total_rows,
                    max_seqlen_q=self.max_seqlen_q,
                    max_seqlen_k=self.max_seqlen_k,
                    sm_scale=sm_scale,
                )
                return self.attention.run(
                    q.contiguous(),
                    (k_cache, v_cache),
                    kv_cache_sf=(k_scale, v_scale),
                )
            packed_k = k_cache.index_select(0, self.physical_pages)
            packed_v = v_cache.index_select(0, self.physical_pages)
            selected_k_scale = k_scale.index_select(0, self.physical_pages)
            selected_v_scale = v_scale.index_select(0, self.physical_pages)
            selected_pages = self.physical_pages.numel()
            paged_k = self._dequantize_prefill_nvfp4(
                packed_k.reshape(-1, _PAGE_SIZE, 64),
                selected_k_scale.reshape(-1, _PAGE_SIZE, 8),
                global_scale,
            ).view(selected_pages, num_kv_heads, _PAGE_SIZE, 128)
            paged_v = self._dequantize_prefill_nvfp4(
                packed_v.reshape(-1, _PAGE_SIZE, 64),
                selected_v_scale.reshape(-1, _PAGE_SIZE, 8),
                global_scale,
            ).view(selected_pages, num_kv_heads, _PAGE_SIZE, 128)
        else:
            if (
                k_cache.ndim != 3
                or k_cache.shape[2] != 128
                or k_cache.shape != v_cache.shape
                or k_cache.shape[0] % _PAGE_SIZE != 0
            ):
                raise ValueError(
                    "native MSA-v1 Q8K8 prefill requires equal, page-aligned "
                    "[slots, num_kv_heads, 128] K/V caches"
                )
            num_kv_heads = k_cache.shape[1]
            num_pages = k_cache.shape[0] // _PAGE_SIZE
            paged_k = (
                k_cache.view(num_pages, _PAGE_SIZE, num_kv_heads, 128)
                .index_select(0, self.physical_pages)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            paged_v = (
                v_cache.view(num_pages, _PAGE_SIZE, num_kv_heads, 128)
                .index_select(0, self.physical_pages)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        if num_q_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive")
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.attention = _PrefillAttentionWrapper()
        self.attention.plan(
            topk_indices,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.page_table,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            total_k=self.total_k,
            total_rows=self.total_rows,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            sm_scale=sm_scale,
        )
        return self.attention.run(q.contiguous(), (paged_k, paged_v))

"""Opt-in top-512 for the decode and target-verify rows of the ratio-1/2 index
layers (SGLANG_OPT_DSV41_LITETOPK_DECODE).

DeepGEMM's paged MXFP4 logits also count a coarse histogram of each row's scores,
from which the experimental ``litetopk_decode`` selector picks the exact BF16
top-512 pool slots in place of ``topk_transform_paged_v2``, with its output
contract: unordered, -1 padded, every slot of a row of at most 512 scores. Ties
prefer the lower physical slot. Exactness is relative to the BF16 logits; BF16
rounding can change selection from the default FP32 producer. The BF16 logits
are the same with or without the histogram. On a GPU or a DeepGEMM that
cannot count the histogram on MXFP4 inputs the default top-k stays.

The hook serves every ``FullTopKIndexer.topk_decode`` call and the top-512 of
``SparseTableBackend.publish_decode``. With candidate filtering (eager decode and
the ``candidate_filtered`` graphs) these are the ratio-2 layers and the candidate
source; the graphs that skip filtering (``candidate_c2_all``,
``candidate_unfiltered``) also send the candidate consumers through
``FullTopKIndexer.topk_decode``, so the hook serves them there too."""

from __future__ import annotations

import functools
import json
import logging
import os
import socket
import time
from typing import TYPE_CHECKING, Dict, Optional, Set

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import topk_transform_paged_from_metadata
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.kernels.experimental.litetopk_decode.bf16 import (
        Bf16Dsv41DecodePlan as Dsv41DecodePlan,
    )
    from sglang.kernels.experimental.litetopk_decode.bf16 import (
        Bf16Dsv41DecodeStorage as Dsv41DecodeStorage,
    )
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata

    from .scoring import DeepGEMMDecodeData
    from .types import Selection

logger = logging.getLogger(__name__)


@functools.cache
def get_litetopk_decode() -> Optional[LiteTopKDecode]:
    """The process-wide selector when the env asks for it and this GPU and
    DeepGEMM support it; None keeps the default top-k."""
    if not envs.SGLANG_OPT_DSV41_LITETOPK_DECODE.get():
        return None
    reason = _unsupported_reason()
    if reason is not None:
        logger.warning(
            "SGLANG_OPT_DSV41_LITETOPK_DECODE is set but unsupported here (%s); "
            "the V4.1 decode index top-k stays on topk_transform_paged_v2",
            reason,
        )
        return None
    logger.info(
        "V4.1 decode index top-k: BF16 exact histogram + LiteTopK, automatic next-n routing"
    )
    return LiteTopKDecode(
        device=torch.device("cuda", torch.cuda.current_device()),
        check=envs.SGLANG_DEBUG_DSV41_LITETOPK_CHECK.get(),
        check_file=envs.SGLANG_DEBUG_DSV41_LITETOPK_CHECK_FILE.get(),
    )


def _unsupported_reason() -> Optional[str]:
    try:
        from sglang.kernels.experimental.litetopk_decode import bf16
    except ImportError as e:
        return f"{type(e).__name__}: {e}"
    return bf16.dsv41_decode_unsupported_reason()


class LiteTopKDecode:
    """``Dsv41DecodePlan`` per row count, shared by every index layer: the layers
    of a forward run in order on their stream and each call leaves its buffers at
    rest. Every plan is a view of one ``Dsv41DecodeStorage`` sized for the largest
    row count so far (about 72 KB per row), so the buffers cost memory once rather
    than per row count. A storage that captured CUDA graphs use stays for their
    replays (SGLang captures the largest batch first, so one storage serves them
    all); a row count beyond the storage, outside capture, gets a new one of at
    least twice the rows, which bounds the storages kept alive to a few times the
    rows of the largest batch. None of this is counted in mem_fraction_static."""

    # Check mode: eager decode calls log the counters at most this often.
    _REPORT_INTERVAL_S = 10.0

    def __init__(self, *, device: torch.device, check: bool, check_file: Optional[str]):
        from sglang.kernels.experimental.litetopk_decode import bf16

        self._fused = bf16
        self._device = torch.device(device)
        self._topk = bf16.TOPK
        self._page_size = bf16.PAGE_SIZE
        self._heads = 32
        self._head_bytes = 64  # packed e2m1
        # The storage new plans are views of, and the streams of its eager calls.
        self._storage: Optional[Dsv41DecodeStorage] = None
        self._storage_streams: Set[torch.cuda.Stream] = set()
        self._plans: Dict[int, Dsv41DecodePlan] = {}
        # Row counts whose plans captured graphs use: kept with their storage.
        self._graph_rows: Set[int] = set()
        self._warned: Set[str] = set()
        self._check = check
        self._check_file = check_file
        # Check mode: device counters of [rows checked, rows whose slot multiset
        # differs from the default top-k, calls], and an event after the latest
        # eager update.
        self._check_counts = (
            torch.zeros(3, dtype=torch.int64, device=self._device) if check else None
        )
        self._check_event = torch.cuda.Event() if check else None
        self._reported = (0, 0, 0)
        self._report_time = 0.0

    def prepare_metadata(
        self,
        metadata: PagedIndexerMetadata,
        request_indices: torch.Tensor,
        tokens_per_request: int,
    ) -> None:
        """Once per forward, recorded inside CUDA graphs before index layers.

        Never reuse a warmup schedule: replay must rebuild from the live causal
        lengths and request IDs. Q4 serves n=1..4, Q6 serves n=5/6.
        """
        metadata.bf16_schedule = None
        metadata.bf16_indices = None
        metadata.bf16_tokens_per_request = tokens_per_request
        if (
            not 1 <= tokens_per_request <= 6
            or metadata.compressed_page_size != self._page_size
            or metadata.row_chunk > 0
            or isinstance(metadata.deep_gemm_metadata, list)
            or request_indices.numel() != metadata.compressed_seq_lens.numel()
            # The single-CTA schedule uses shared memory proportional to rows.
            or request_indices.numel() > 16384
        ):
            return
        import deep_gemm

        indices = request_indices.to(torch.int32).contiguous()
        metadata.bf16_indices = indices
        metadata.bf16_schedule = deep_gemm.get_paged_mqa_logits_bf16_metadata(
            _context_lens(metadata),
            self._page_size,
            deep_gemm.get_num_sms(),
            indices=indices,
            tokens_per_request=tokens_per_request,
        )

    def scores(
        self,
        *,
        data: DeepGEMMDecodeData,
        metadata: PagedIndexerMetadata,
        out: Selection,
    ) -> Optional[torch.Tensor]:
        """The layer's logits with their histogram counted for ``select``, or
        None to keep the default top-k for this call."""
        if self._check and time.monotonic() - self._report_time >= (
            self._REPORT_INTERVAL_S
        ):
            self.report_check()
        plan = self._get_plan(data=data, metadata=metadata, out=out)
        if plan is None:
            return None
        return plan.scores(
            q=(data.q_fp4, data.q_sf),
            kv_cache=data.k_cache,
            # The fused Q packer already rounds these weights to BF16 before
            # widening to FP32. This conversion is lossless for its output.
            weights=data.weights.to(torch.bfloat16),
            context_lens=_context_lens(metadata),
            block_table=metadata.page_table,
            schedule_metadata=metadata.bf16_schedule,
            max_context_len=metadata.max_compressed_seq_len,
            indices=metadata.bf16_indices,
            tokens_per_request=metadata.bf16_tokens_per_request,
        )

    def select(
        self,
        *,
        logits: torch.Tensor,
        metadata: PagedIndexerMetadata,
        out: Selection,
    ) -> None:
        """The top-512 pool slots of ``logits``, which the preceding ``scores``
        call returned, into ``out.page_indices``."""
        plan = self._plans[out.page_indices.shape[0]]
        plan.select(
            scores=logits,
            context_lens=_context_lens(metadata),
            block_table=metadata.page_table,
            out=out.page_indices,
        )
        if self._check:
            self._compare(plan=plan, logits=logits, metadata=metadata, out=out)

    def report_check(self) -> None:
        """Logs the check counters when they changed. Reads them back to the
        host, so it does nothing while a CUDA graph is captured.

        Called at eager prefills and, at most every ``_REPORT_INTERVAL_S``, at
        eager decode calls: counts of graph replays since then are read at the
        next eager forward of this process (the replays precede it on the
        forward stream)."""
        if not self._check or _in_graph_capture():
            return
        self._report_time = time.monotonic()
        # Eager updates may have run on another stream (multi-stream overlap).
        torch.cuda.current_stream(self._device).wait_event(self._check_event)
        counts = tuple(self._check_counts.tolist())
        if counts == self._reported:
            return
        self._reported = counts
        rows, differ, calls = counts
        log = logger.warning if differ else logger.info
        log(
            "LiteTopK decode check: %d of %d rows (%d calls) differ from the "
            "default top-k",
            differ,
            rows,
            calls,
        )
        if self._check_file is not None:
            ranks = _ranks()
            record = dict(
                time=time.time(),
                host=socket.gethostname(),
                pid=os.getpid(),
                **ranks,
                rows=rows,
                rows_differ=differ,
                calls=calls,
                plans=sorted(self._plans),
            )
            path = (
                f"{self._check_file}.TP{ranks['tp_rank']}_PP{ranks['pp_rank']}"
                f"_DP{ranks['dp_rank']}_pid{os.getpid()}.jsonl"
            )
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")

    def _get_plan(
        self,
        *,
        data: DeepGEMMDecodeData,
        metadata: PagedIndexerMetadata,
        out: Selection,
    ) -> Optional[Dsv41DecodePlan]:
        rows = out.page_indices.shape[0]
        if rows == 0:
            return None
        reason = self._unsupported_call(data=data, metadata=metadata, out=out)
        if reason is not None:
            self._warn_once(f"{reason}; the default top-k serves these calls")
            return None
        capturing = torch.cuda.is_current_stream_capturing()
        plan = self._plans.get(rows)
        if plan is None:
            storage = self._storage
            if storage is None or rows > storage.max_rows:
                if capturing:
                    # Buffers allocated now would come from the graph's private pool.
                    self._warn_once(
                        f"no buffers for {rows} rows when a CUDA graph is captured; "
                        "that graph keeps the default top-k"
                    )
                    return None
                storage = self._grow(rows)
            plan = storage.plan(rows)  # views of the storage: nothing is allocated
            self._plans[rows] = plan
        if capturing or get_is_capture_mode():
            self._graph_rows.add(rows)
        elif plan.storage is self._storage and self._device.type == "cuda":
            self._storage_streams.add(torch.cuda.current_stream(self._device))
        return plan

    def _grow(self, rows: int) -> Dsv41DecodeStorage:
        """A storage for ``rows`` rows or more, replacing the current one for new
        plans. Plans of captured graphs keep theirs; the other plans of the old
        storage are dropped and rebuilt on the new one when next used."""
        old = self._storage
        if old is not None and not any(
            self._plans[r].storage is old for r in self._graph_rows
        ):
            # Freed with its plans below: its eager calls may still run on the
            # streams they were issued on.
            for stream in self._storage_streams:
                for buffer in (old.histogram, old.workspace, old.output):
                    buffer.record_stream(stream)
        self._plans = {r: p for r, p in self._plans.items() if r in self._graph_rows}
        max_rows = rows if old is None else max(rows, 2 * old.max_rows)
        self._storage = self._fused.Bf16Dsv41DecodeStorage(max_rows, self._device)
        self._storage_streams = set()
        logger.info(
            "LiteTopK decode: %.1f MiB of selector buffers for up to %d rows",
            self._storage.nbytes / (1 << 20),
            max_rows,
        )
        return self._storage

    def _unsupported_call(
        self,
        *,
        data: DeepGEMMDecodeData,
        metadata: PagedIndexerMetadata,
        out: Selection,
    ) -> Optional[str]:
        rows = out.page_indices.shape[0]
        if getattr(metadata, "bf16_schedule", None) is None:
            return "no matching BF16 schedule for this forward"
        if out.raw_indices is not None:
            return "raw indices are requested"
        if metadata.compressed_page_size != self._page_size:
            return (
                f"index-K pages hold {metadata.compressed_page_size} slots, "
                f"not {self._page_size}"
            )
        if tuple(out.page_indices.shape[1:]) != (self._topk,):
            return (
                f"the selection is {tuple(out.page_indices.shape)}, "
                f"not [rows, {self._topk}]"
            )
        if not out.page_indices.is_contiguous():
            return "the selection buffer is not contiguous"
        if out.page_indices.device != self._device:
            return (
                f"the selection is on {out.page_indices.device}, the selector's "
                f"buffers on {self._device}"
            )
        expected = (rows, 1, self._heads, self._head_bytes)
        q_shapes = (tuple(data.q_fp4.shape), tuple(data.q_sf.shape))
        if q_shapes != (expected, expected[:3]):
            return (
                f"the index queries are {q_shapes[0]}, not {expected}: DeepGEMM's "
                f"histogram needs flattened rows and {self._heads} heads of "
                f"{2 * self._head_bytes} dimensions"
            )
        if data.weights.dtype not in (torch.float32, torch.bfloat16) or tuple(
            data.weights.shape
        ) != (
            rows,
            self._heads,
        ):
            return (
                f"the head weights are {data.weights.dtype} "
                f"{tuple(data.weights.shape)}, expected FP32/BF16 [rows, {self._heads}]"
            )
        return None

    def _warn_once(self, message: str) -> None:
        if message not in self._warned:
            self._warned.add(message)
            logger.warning("LiteTopK decode: %s", message)

    def _compare(
        self,
        *,
        plan: Dsv41DecodePlan,
        logits: torch.Tensor,
        metadata: PagedIndexerMetadata,
        out: Selection,
    ) -> None:
        rows = out.page_indices.shape[0]
        # The plan's own output is free: serving calls select into ``out``.
        reference = plan.output
        topk_transform_paged_from_metadata(logits.float(), metadata, reference)
        # Both are unordered: compare the sorted slot lists.
        differ = out.page_indices.sort(dim=1).values != reference.sort(dim=1).values
        self._check_counts[0] += rows
        self._check_counts[1] += differ.any(dim=1).sum()
        self._check_counts[2] += 1
        if not torch.cuda.is_current_stream_capturing():
            self._check_event.record()


def _context_lens(metadata: PagedIndexerMetadata) -> torch.Tensor:
    """int32 [rows, 1]: DeepGEMM takes next_n = 1 rows' lengths as a column."""
    lens = metadata.compressed_seq_lens.to(torch.int32)
    return lens.unsqueeze(-1) if lens.dim() == 1 else lens


def _ranks() -> Dict[str, int]:
    """This process's tensor, pipeline and data parallel ranks (dp_rank 0
    without a data parallel controller, i.e. one replica)."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    dp_rank = parallel.dp_rank
    return dict(
        tp_rank=parallel.tp_rank,
        pp_rank=parallel.pp_rank,
        dp_rank=dp_rank if dp_rank is not None else 0,
    )


def _in_graph_capture() -> bool:
    return (
        get_is_capture_mode()
        or torch.cuda.is_current_stream_capturing()
        or is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
    )

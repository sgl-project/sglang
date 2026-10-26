# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Profile-trace step-span naming (kept dependency-light for CPU unit tests).

The step span wraps each ``ModelRunner.forward`` in the torch/Perfetto trace.
Its name carries the forward mode and batch shape; when detailed annotations
are enabled it also folds in the per-iteration aggregates (for roofline-style
analysis) so a single label describes both timing and the analytical work of
that forward.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, List, Tuple

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# Process-wide toggle for detailed step-span annotations. Set by the scheduler's
# profiler manager on profile start/stop and read by build_step_span_name, so no
# per-runner flag or scheduler bridge is needed. Covers all model runners in the
# process (e.g. both the EAGLE draft and target runners).
_DETAILED_ANNOTATIONS_ENABLED = False


def set_detailed_annotations_enabled(enabled: bool) -> None:
    global _DETAILED_ANNOTATIONS_ENABLED
    _DETAILED_ANNOTATIONS_ENABLED = bool(enabled)


def detailed_annotations_enabled() -> bool:
    return _DETAILED_ANNOTATIONS_ENABLED


def _agg(nqs: List[int], nkvs: List[int]) -> Tuple[int, int, int, int]:
    """Return (Σ N_Q, Σ N_KV, Σ N_Q², Σ N_Q·N_KV) for one request group."""
    sq = sum(nqs)
    sk = sum(nkvs)
    sqsq = sum(nq * nq for nq in nqs)
    sqsk = sum(nq * nkv for nq, nkv in zip(nqs, nkvs))
    return sq, sk, sqsq, sqsk


def _decode_query_width(forward_batch: ForwardBatch) -> int:
    """Per-request query-token count (N_Q) for a decode-family forward.

    Vanilla decode emits one token per request, but speculative decoding does
    not: EAGLE/MTP draft-decode and target-verify process a uniform
    ``num_tokens_per_req`` tokens per request (draft top-k for draft-decode,
    the draft-token count for verify), so N_Q per request is that width, and
    the step's total Σ N_Q is ``bs * num_tokens_per_req`` rather than ``bs``.

    The width is read from ``forward_batch.spec_info.num_tokens_per_req`` and
    falls back to 1 when there is no spec input or the width is unset (-1).
    """
    spec = getattr(forward_batch, "spec_info", None)
    width = getattr(spec, "num_tokens_per_req", -1) if spec is not None else -1
    return width if isinstance(width, int) and width > 0 else 1


def build_detailed_annotation_suffix(forward_batch: ForwardBatch) -> str:
    """Compute the detailed-annotation aggregates from the batch's CPU-side length mirrors.

    All aggregates are emitted, prefixed by the roofline compute-shape bucket:
    ``c_`` for context/extend-shaped work (EXTEND and DRAFT_EXTEND_V2) and ``g_`` for
    single-query generation (DECODE, TARGET_VERIFY), with MIXED emitting both
    groups.
    """
    mode = forward_batch.forward_mode
    seq_lens_cpu = forward_batch.seq_lens_cpu

    # DECODE (vanilla or spec draft-decode) and TARGET_VERIFY both key off
    # ``seq_lens_cpu`` for N_KV and a uniform per-request query width N_Q, and
    # are both classified as generation (``g_``) by request phase
    #   * DECODE        -> N_Q is 1 (vanilla) or the spec draft-decode width.
    #   * TARGET_VERIFY -> N_Q is ``num_tokens_per_req`` (the draft-token count);
    #                      the request is past its prompt (generation phase)
    if mode == ForwardMode.DECODE or mode == ForwardMode.TARGET_VERIFY:
        if seq_lens_cpu is None:
            return ""
        nq = _decode_query_width(forward_batch)
        nkvs = [int(x) for x in seq_lens_cpu.tolist()]
        nqs = [nq] * len(nkvs)
        sq, sk, sqsq, sqsk = _agg(nqs, nkvs)
        # ``sq`` is always emitted (self-contained suffix): it equals ``bs``
        # (vanilla decode) or ``bs * num_tokens_per_req`` (spec draft-decode /
        # target-verify).
        return f"g_sq={sq} g_sqsq={sqsq} g_sqsk={sqsk} g_sk={sk}"

    ext_seq = forward_batch.extend_seq_lens_cpu
    ext_prefix = forward_batch.extend_prefix_lens_cpu
    if ext_seq is None or ext_prefix is None:
        return ""

    if mode == ForwardMode.EXTEND or mode == ForwardMode.DRAFT_EXTEND_V2:
        # Both are extend-shaped, multi-query context
        nqs = [int(q) for q in ext_seq]
        nkvs = [int(p) + int(q) for p, q in zip(ext_prefix, ext_seq)]
        sq, sk, sqsq, sqsk = _agg(nqs, nkvs)
        return f"c_sq={sq} c_sqsq={sqsq} c_sqsk={sqsk} c_sk={sk}"

    if mode == ForwardMode.MIXED:
        # A running-decode request appears as a length-1 extend; everything
        # else is a context (prefill) chunk.
        c_nqs: List[int] = []
        c_nkvs: List[int] = []
        g_nqs: List[int] = []
        g_nkvs: List[int] = []
        for p, q in zip(ext_prefix, ext_seq):
            nq, nkv = int(q), int(p) + int(q)
            if nq == 1:
                g_nqs.append(nq)
                g_nkvs.append(nkv)
            else:
                c_nqs.append(nq)
                c_nkvs.append(nkv)
        c_sq, c_sk, c_sqsq, c_sqsk = _agg(c_nqs, c_nkvs)
        g_sq, g_sk, g_sqsq, g_sqsk = _agg(g_nqs, g_nkvs)
        return (
            f"c={len(c_nqs)} g={len(g_nqs)} "
            f"c_sq={c_sq} c_sk={c_sk} c_sqsq={c_sqsq} c_sqsk={c_sqsk} "
            f"g_sq={g_sq} g_sk={g_sk} g_sqsq={g_sqsq} g_sqsk={g_sqsk}"
        )

    return ""


def _format_step_shape_stats(
    prefix: str, nqs: List[int], nkvs: List[int], include_vectors: bool
) -> str:
    """Return sums plus a compact per-request shape summary for an opt-in log."""
    sq, sk, sqsq, sqsk = _agg(nqs, nkvs)
    if not nkvs:
        return f"{prefix}_sq={sq} {prefix}_sqsq={sqsq} {prefix}_sqsk={sqsk} {prefix}_sk={sk}"

    fields = [
        f"{prefix}_sq={sq}",
        f"{prefix}_sqsq={sqsq}",
        f"{prefix}_sqsk={sqsk}",
        f"{prefix}_sk={sk}",
        f"{prefix}_qmin={min(nqs)}",
        f"{prefix}_qmed={statistics.median(nqs):g}",
        f"{prefix}_qmax={max(nqs)}",
        f"{prefix}_kmin={min(nkvs)}",
        f"{prefix}_kmed={statistics.median(nkvs):g}",
        f"{prefix}_kmax={max(nkvs)}",
    ]
    if include_vectors:
        fields.extend(
            (
                f"{prefix}_q=[{','.join(map(str, nqs))}]",
                f"{prefix}_k=[{','.join(map(str, nkvs))}]",
            )
        )
    return " ".join(fields)


def build_step_shape_log(
    forward_batch: ForwardBatch,
    *,
    is_draft_worker: bool = False,
    forward_id: int | None = None,
    include_vectors: bool = False,
) -> str:
    """Build one compact, parseable description of an execution step.

    This is intentionally separate from the profiler span. It is designed for
    opt-in server logging over a full workload: summary mode exposes the
    min/median/max needed to bucket ragged batches, while vector mode keeps the
    exact per-request N_Q and KV lengths for offline analysis.
    """
    mode = forward_batch.forward_mode
    stage = "DRAFT" if is_draft_worker else mode.name
    fields = [f"stage={stage}", f"bs={forward_batch.batch_size}"]
    if forward_id is not None:
        fields.append(f"forward_id={forward_id}")

    if mode == ForwardMode.DECODE or mode == ForwardMode.TARGET_VERIFY:
        if forward_batch.seq_lens_cpu is None:
            fields.append("shape=unavailable")
        else:
            nq = _decode_query_width(forward_batch)
            nkvs = [int(x) for x in forward_batch.seq_lens_cpu.tolist()]
            fields.append(
                _format_step_shape_stats("g", [nq] * len(nkvs), nkvs, include_vectors)
            )
    elif mode == ForwardMode.EXTEND or mode == ForwardMode.DRAFT_EXTEND_V2:
        if (
            forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_prefix_lens_cpu is None
        ):
            fields.append("shape=unavailable")
        else:
            nqs = [int(x) for x in forward_batch.extend_seq_lens_cpu]
            nkvs = [
                int(prefix) + int(query)
                for prefix, query in zip(
                    forward_batch.extend_prefix_lens_cpu, nqs, strict=True
                )
            ]
            fields.append(_format_step_shape_stats("c", nqs, nkvs, include_vectors))
    elif mode == ForwardMode.MIXED:
        if (
            forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_prefix_lens_cpu is None
        ):
            fields.append("shape=unavailable")
        else:
            c_nqs, c_nkvs, g_nqs, g_nkvs = [], [], [], []
            for prefix, query in zip(
                forward_batch.extend_prefix_lens_cpu,
                forward_batch.extend_seq_lens_cpu,
                strict=True,
            ):
                nq, nkv = int(query), int(prefix) + int(query)
                if nq == 1:
                    g_nqs.append(nq)
                    g_nkvs.append(nkv)
                else:
                    c_nqs.append(nq)
                    c_nkvs.append(nkv)
            fields.extend(
                (
                    f"c={len(c_nqs)}",
                    f"g={len(g_nqs)}",
                    _format_step_shape_stats("c", c_nqs, c_nkvs, include_vectors),
                    _format_step_shape_stats("g", g_nqs, g_nkvs, include_vectors),
                )
            )
    return " ".join(fields)

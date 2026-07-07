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
Its name carries the forward mode and batch shape; when roofline annotations
are enabled it also folds in the per-iteration roofline aggregates so a single
label describes both timing and the analytical work of that forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _agg(nqs: List[int], nkvs: List[int]) -> Tuple[int, int, int, int]:
    """Return (Σ N_Q, Σ N_KV, Σ N_Q², Σ N_Q·N_KV) for one request group."""
    sq = sum(nqs)
    sk = sum(nkvs)
    sqsq = sum(nq * nq for nq in nqs)
    sqsk = sum(nq * nkv for nq, nkv in zip(nqs, nkvs))
    return sq, sk, sqsq, sqsk


def build_roofline_suffix(forward_batch: "ForwardBatch") -> str:
    """Compute the roofline aggregates from the batch's CPU-side length mirrors.

    Only the terms SGLang's base ``step[...]`` label does *not* already carry
    are emitted, always prefixed by phase: ``c_`` for context (EXTEND) and
    ``g_`` for generation (DECODE), with MIXED emitting both groups. ``bs`` and
    ``toks`` are intentionally left to the base label to avoid duplication.

    Returns an empty string when the required host mirrors are unavailable
    (e.g. some overlap-schedule paths drop ``seq_lens_cpu``), so the base label
    is emitted unchanged rather than risking a device sync or a crash.
    """
    mode = forward_batch.forward_mode
    seq_lens_cpu = forward_batch.seq_lens_cpu

    if mode == ForwardMode.DECODE:
        if seq_lens_cpu is None:
            return ""
        nkvs = [int(x) for x in seq_lens_cpu.tolist()]
        nqs = [1] * len(nkvs)
        _, sk, sqsq, sqsk = _agg(nqs, nkvs)
        # Pure generation phase -> ``g_`` prefix, matching the MIXED split.
        return f"g_sqsq={sqsq} g_sqsk={sqsk} g_sk={sk}"

    ext_seq = forward_batch.extend_seq_lens_cpu
    ext_prefix = forward_batch.extend_prefix_lens_cpu
    if ext_seq is None or ext_prefix is None:
        return ""

    if mode == ForwardMode.EXTEND:
        nqs = [int(q) for q in ext_seq]
        nkvs = [int(p) + int(q) for p, q in zip(ext_prefix, ext_seq)]
        _, sk, sqsq, sqsk = _agg(nqs, nkvs)
        # Pure context phase -> ``c_`` prefix, matching the MIXED split.
        return f"c_sqsq={sqsq} c_sqsk={sqsk} c_sk={sk}"

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


def build_step_span_name(
    forward_batch: "ForwardBatch", roofline_annotations: bool = False
) -> str:
    """Build the profile-trace span name for one forward step.

    When ``roofline_annotations`` is set (only while a roofline-annotated
    profile is active), the roofline aggregates are folded into the label via
    :func:`build_roofline_suffix`.
    """
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        base = f"step[EXTEND bs={bs} toks={ext_toks}"
    else:
        base = f"step[{mode.name} bs={bs}"

    if roofline_annotations:
        suffix = build_roofline_suffix(forward_batch)
        if suffix:
            base = f"{base} {suffix}"
    return f"{base}]"

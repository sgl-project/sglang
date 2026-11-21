# SPDX-License-Identifier: Apache-2.0
"""Reuse VA 24487f4f48 request routing with SG compact512/residual-mode1.

The request/plan classes and builder below are copied from the fixed VA source;
the SG adapter prepares common tensors once and keeps GBSA spans coalesced.
"""

from dataclasses import dataclass

import torch

from .qsa_fia_va import prepare_qsa_fia_paged_kv, qsa_fia_attention
from .qsa_gbsa_prefill import qsa_gbsa_prefill

QSA_FIA_MAX_LOGICAL_KV_LENGTH = 24 * 1024


@dataclass(frozen=True)
class QSARequest:
    request_index: int
    query_start: int
    query_end: int
    kv_length: int
    is_prefill: bool

    @property
    def use_paged_kv(self) -> bool:
        return self.kv_length != self.query_end - self.query_start

    @property
    def use_fia(self) -> bool:
        """Whether this Prefill chunk starts below the FIA KV limit."""
        return self.fia_query_end > self.query_start

    @property
    def fia_query_end(self) -> int:
        """Keep a chunk crossing 24K on FIA; switch subsequent chunks."""
        if not self.is_prefill:
            return self.query_start
        query_length = self.query_end - self.query_start
        # Scheduler seq_lens already includes the current query chunk.
        total_logical_kv_length = self.kv_length
        fits_fia_limit = total_logical_kv_length <= QSA_FIA_MAX_LOGICAL_KV_LENGTH
        crosses_fia_limit = (
            total_logical_kv_length > QSA_FIA_MAX_LOGICAL_KV_LENGTH
            and total_logical_kv_length - query_length < QSA_FIA_MAX_LOGICAL_KV_LENGTH
        )
        if fits_fia_limit or crosses_fia_limit:
            return self.query_end
        return self.query_start

    @property
    def gbsa_query_start(self) -> int:
        """First row routed to GBSA; Decode starts at the request boundary."""
        return self.fia_query_end


@dataclass(frozen=True)
class QSARequestPlan:
    """Precomputed operator routing shared by every QSA layer in one step."""

    requests: tuple[QSARequest, ...]
    fia_requests: tuple[QSARequest, ...]
    gbsa_spans: tuple[tuple[int, int], ...]

    def __getitem__(self, index: int) -> QSARequest:
        return self.requests[index]

    def __iter__(self):
        return iter(self.requests)

    def __len__(self) -> int:
        return len(self.requests)


def build_qsa_request_plan(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_actual_tokens: int,
    is_prefilling_cpu: torch.Tensor,
) -> QSARequestPlan:
    """Classify mixed-batch rows once for reuse by every QSA layer."""
    if any(t.device.type != "cpu" for t in (query_start_loc_cpu, seq_lens_cpu, is_prefilling_cpu)):
        raise ValueError("QSA dispatch requires scheduler CPU metadata")
    starts = query_start_loc_cpu.tolist()
    lengths = seq_lens_cpu.tolist()
    phases = is_prefilling_cpu.tolist()
    if len(lengths) < len(starts) - 1 or len(phases) < len(starts) - 1:
        raise ValueError("QSA phase and KV metadata must cover every request")
    requests: list[QSARequest] = []
    for request_index, (start, end) in enumerate(zip(starts, starts[1:])):
        start = min(max(start, 0), num_actual_tokens)
        end = min(max(end, start), num_actual_tokens)
        if end > start:
            requests.append(
                QSARequest(request_index, start, end, int(lengths[request_index]), bool(phases[request_index]))
            )
    requests_tuple = tuple(requests)
    fia_requests = tuple(request for request in requests_tuple if request.use_fia)
    gbsa_spans: list[tuple[int, int]] = []
    for request in requests_tuple:
        start = request.gbsa_query_start
        if start >= request.query_end:
            continue
        if gbsa_spans and gbsa_spans[-1][1] == start:
            gbsa_spans[-1] = (gbsa_spans[-1][0], request.query_end)
        else:
            gbsa_spans.append((start, request.query_end))
    return QSARequestPlan(requests_tuple, fia_requests, tuple(gbsa_spans))


@dataclass(frozen=True)
class SGPrefillInputs:
    plan: QSARequestPlan
    page_table: torch.Tensor
    visible_lens: torch.Tensor
    sequence_lengths: torch.Tensor
    extend_lengths: torch.Tensor
    # start/end query rows, first/last request indices, request offsets.
    gbsa_calls: tuple


def prepare_sg_prefill_inputs(
    sequence_lengths_cpu,
    extend_lengths_cpu,
    sequence_lengths,
    extend_lengths,
    token_table,
    logical_positions,
):
    """Build shared routing/mask inputs without reading device values."""
    lengths = [int(n) for n in sequence_lengths_cpu]
    extends = [int(n) for n in extend_lengths_cpu]
    if len(lengths) != len(extends) or any(e < 0 or e > n for e, n in zip(extends, lengths)):
        raise ValueError("QSA prefill request lengths are inconsistent")
    starts = [0]
    for count in extends:
        starts.append(starts[-1] + count)
    if logical_positions.shape != (starts[-1],):
        raise ValueError("QSA prefill positions must cover exactly the live query rows")
    plan = build_qsa_request_plan(
        torch.tensor(starts, dtype=torch.int32, device="cpu"),
        torch.tensor(lengths, dtype=torch.int32, device="cpu"),
        starts[-1],
        torch.ones(len(lengths), dtype=torch.bool, device="cpu"),
    )
    if not plan.fia_requests:
        return None
    # These views are shared by every QSA layer in this forward batch.
    page_table = (token_table[:, ::128] // 128).to(torch.int32).contiguous()
    visible_lens = (logical_positions + 1).to(torch.int32)
    calls = []
    for start, end in plan.gbsa_spans:
        requests = [r for r in plan if start <= r.query_start < end]
        first, last = requests[0].request_index, requests[-1].request_index + 1
        ext = extend_lengths[first:last].to(torch.int64)
        cu_q = torch.cat((ext.new_zeros(1), ext.cumsum(0)))
        calls.append((start, end, first, last, cu_q))
    return SGPrefillInputs(
        plan, page_table, visible_lens, sequence_lengths, extend_lengths, tuple(calls)
    )


def dispatch_sg_prefill(q, k, v, key_cache, value_cache, groups, inputs, scale):
    """Use current-chunk KV or zero-copy paged KV with the VA request plan."""
    if groups.shape != (q.shape[0], 512):
        raise ValueError("SG prefill dispatch requires compact512 group indices")
    output = torch.empty_like(q)
    key_pages = key_cache.view(-1, 128, 1, 256)
    value_pages = value_cache.view(-1, 128, 1, 256)
    for request in inputs.plan.fia_requests:
        rows = slice(request.query_start, request.query_end)
        if request.use_paged_kv or k is None or v is None:
            keys, values, table, _ = prepare_qsa_fia_paged_kv(
                key_pages,
                value_pages,
                inputs.page_table[request.request_index : request.request_index + 1],
                request.kv_length,
            )
        else:
            keys, values, table = k[rows], v[rows], None
        qsa_fia_attention(
            q[rows], keys, values, groups[rows], inputs.visible_lens[rows],
            request.kv_length, output[rows], block_table=table, softmax_scale=scale,
        )
    for start, end, first, last, cu_q in inputs.gbsa_calls:
        output[start:end].copy_(qsa_gbsa_prefill(
            q[start:end], key_cache, value_cache, groups[start:end], None,
            inputs.sequence_lengths[first:last], inputs.extend_lengths[first:last],
            end - start, scale, indices_are_blocks=True,
            page_table=inputs.page_table[first:last], cu_seqlens_q=cu_q,
        ))
    return output

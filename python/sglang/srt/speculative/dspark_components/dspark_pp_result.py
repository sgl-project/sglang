"""DSpark result state carried by the scheduler's existing PP output channel."""

from typing import Literal

import msgspec
import torch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

_PREFIX = "dspark_"
_HEADER = "dspark_result"


class DSparkPPResultHeader(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    version: Literal[1]
    phase: Literal["prefill", "decode"]
    requests: tuple[tuple[str, int, int], ...]
    stride: int
    has_cap_lens: bool


def _request_keys(batch):
    return tuple(
        (req.rid, req.kv_committed_len, len(req.output_ids)) for req in batch.reqs
    )


def _phase(batch):
    if batch.forward_mode.is_extend():
        return "prefill"
    if batch.forward_mode.is_decode():
        return "decode"
    raise ValueError("PP DSpark results require a prefill or decode batch")


def _read_header(packet, batch):
    if not batch.spec_algorithm.is_dspark() or batch.return_logprob:
        raise ValueError("PP DSpark result does not match the batch algorithm")
    try:
        header = msgspec.convert(packet[_HEADER], type=DSparkPPResultHeader)
    except (KeyError, msgspec.ValidationError) as error:
        raise ValueError("invalid PP DSpark result header") from error
    if header.requests != _request_keys(batch) or header.phase != _phase(batch):
        raise ValueError("PP DSpark result belongs to a different request/step")
    if (
        header.stride < 1
        or (header.phase == "prefill" and header.stride != 1)
        or (header.phase == "decode" and header.stride < 2)
        or (header.phase == "prefill" and header.has_cap_lens)
    ):
        raise ValueError("invalid PP DSpark result stride or cap metadata")
    return header


def _read_tensors(packet, header):
    bs = len(header.requests)
    sizes = {
        "next_token_ids": bs * header.stride,
        "dspark_bonus_tokens": bs,
        "dspark_new_seq_lens": bs,
    }
    if header.phase == "decode":
        sizes.update(dspark_accept_lens=bs, dspark_block_accept_lens=bs)
    if header.has_cap_lens:
        sizes["dspark_cap_lens"] = bs
    expected = {_HEADER} | {name for name in sizes if name.startswith(_PREFIX)}
    if {name for name in packet if name.startswith(_PREFIX)} != expected:
        raise ValueError("PP DSpark result has missing or unexpected fields")
    for name, size in sizes.items():
        value = packet.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (size,)
            or value.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"invalid PP DSpark result tensor: {name}")
    bonus, lengths = packet["dspark_bonus_tokens"], packet["dspark_new_seq_lens"]
    if bonus.device != lengths.device:
        raise ValueError("PP DSpark draft state must share one device")


def pack_dspark_pp_result(result, batch):
    # A CUDA stream wait cannot make pinned D2H destinations readable by the
    # CPU/Gloo sender. The non-overlap worker may have already issued this copy.
    if result.copy_done is not None:
        result.copy_done.synchronize()
    phase = _phase(batch)
    if result.next_draft_input is None or result.new_seq_lens is None:
        raise ValueError("PP DSpark result is missing next-draft state")
    header = DSparkPPResultHeader(
        version=1,
        phase=phase,
        requests=_request_keys(batch),
        stride=result.speculative_num_draft_tokens if phase == "decode" else 1,
        has_cap_lens=result.cap_lens is not None,
    )
    packet = {
        _HEADER: msgspec.to_builtins(header),
        "next_token_ids": result.next_token_ids,
        "dspark_bonus_tokens": result.next_draft_input.bonus_tokens,
        "dspark_new_seq_lens": result.new_seq_lens,
    }
    if phase == "decode":
        packet["dspark_accept_lens"] = result.accept_lens
        packet["dspark_block_accept_lens"] = result.block_accept_lens
    elif result.accept_lens is not None or result.block_accept_lens is not None:
        raise ValueError("PP DSpark prefill cannot carry accepted-block lengths")
    if header.has_cap_lens:
        packet["dspark_cap_lens"] = result.cap_lens
    _read_tensors(packet, _read_header(packet, batch))
    return packet


def unpack_dspark_pp_result(packet, batch, *, can_run_cuda_graph):
    header = _read_header(packet, batch)
    _read_tensors(packet, header)
    lengths = packet["dspark_new_seq_lens"]
    return GenerationBatchResult(
        next_token_ids=packet["next_token_ids"],
        accept_lens=packet.get("dspark_accept_lens"),
        block_accept_lens=packet.get("dspark_block_accept_lens"),
        cap_lens=packet.get("dspark_cap_lens"),
        new_seq_lens=lengths,
        next_draft_input=make_draft_input_v2(
            bonus_tokens=packet["dspark_bonus_tokens"], new_seq_lens=lengths
        ),
        speculative_num_draft_tokens=(
            header.stride if header.phase == "decode" else None
        ),
        can_run_cuda_graph=can_run_cuda_graph,
    )


def install_dspark_pp_result(result, batch):
    """Validate a received commit after D2H, before advancing local batch state."""
    lengths_cpu = result.new_seq_lens.cpu()
    bonus_cpu = result.next_draft_input.bonus_tokens.cpu()
    tokens = result.next_token_ids
    if not tokens.is_cpu or (
        result.accept_lens is not None and not result.accept_lens.is_cpu
    ):
        raise ValueError("PP DSpark result must finish D2H before installation")
    if result.accept_lens is None:
        if not torch.equal(tokens, bonus_cpu) or not torch.equal(
            lengths_cpu, batch.seq_lens_cpu
        ):
            raise ValueError("PP DSpark prefill state does not match the batch")
    else:
        stride = result.speculative_num_draft_tokens
        accepts = result.accept_lens.long()
        if bool(((accepts < 1) | (accepts > stride)).any()):
            raise ValueError("PP DSpark accepted length exceeds its token block")
        rows = tokens.view(-1, stride)
        if not torch.equal(rows.gather(1, (accepts - 1)[:, None])[:, 0], bonus_cpu):
            raise ValueError("PP DSpark bonus differs from the accepted block")
        prefixes = torch.tensor([req.kv_committed_len for req in batch.reqs])
        if not torch.equal(lengths_cpu, prefixes + accepts):
            raise ValueError(
                "PP DSpark committed lengths disagree with accepted tokens"
            )
        block_accepts = result.block_accept_lens
        if bool(((block_accepts < accepts) | (block_accepts > stride)).any()):
            raise ValueError("PP DSpark block acceptance is outside the verify window")
        if result.cap_lens is not None and bool(
            ((result.cap_lens < accepts) | (result.cap_lens > stride)).any()
        ):
            raise ValueError("PP DSpark cap lengths are outside the verify window")
    batch.spec_info = result.next_draft_input
    batch.seq_lens = result.new_seq_lens
    batch.seq_lens_cpu = lengths_cpu
    batch.seq_lens_sum = int(lengths_cpu.sum())

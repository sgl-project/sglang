from __future__ import annotations

from dataclasses import dataclass
from zlib import crc32


def draft_owner(rid: str, pp_size: int) -> int:
    if pp_size < 1:
        raise ValueError("pp_size must be positive")
    return crc32(rid.encode("utf-8")) % pp_size


@dataclass(frozen=True)
class PPDSparkIdentity:
    rid: str
    generation: tuple[int, int]
    round: int

    @classmethod
    def from_req(cls, req) -> PPDSparkIdentity:
        if req.bootstrap_room is None:
            raise ValueError("PP DSpark requires a PD bootstrap generation")
        return cls(
            req.rid,
            (int(req.bootstrap_room), int(req.retraction_count)),
            req.spec_verify_ct,
        )

    def to_wire(self) -> tuple[str, tuple[int, int], int]:
        return self.rid, self.generation, self.round

    def next_round(self) -> PPDSparkIdentity:
        return PPDSparkIdentity(self.rid, self.generation, self.round + 1)


@dataclass(frozen=True)
class PPDSparkCandidate:
    identity: PPDSparkIdentity
    draft_block_ids: object
    draft_tokens: object
    confidence: object = None


def validate_identities(expected, received) -> None:
    expected_wire = [identity.to_wire() for identity in expected]
    received_wire = [
        (identity[0], tuple(identity[1]), identity[2]) for identity in received
    ]
    if expected_wire != received_wire:
        raise RuntimeError(
            "Stale or mismatched PP DSpark result: "
            f"expected={expected_wire}, received={received_wire}"
        )


def validate_pd_contract(local_mode: bool, remote_mode: bool, pp_size: int) -> None:
    if local_mode != remote_mode:
        raise ValueError(
            "PP DSpark requires --speculative-dspark-pp-replicated-draft "
            "on both prefill and decode"
        )
    if local_mode and pp_size != 2:
        raise ValueError("PP DSpark requires PP2 on both prefill and decode")


def pack_proposal(owner: int, payload: dict) -> dict:
    return {f"dspark_next_{owner}_{key}": value for key, value in payload.items()}


def unpack_proposal(owner: int, tensors: dict) -> dict:
    prefix = f"dspark_next_{owner}_"
    result = {
        key[len(prefix) :]: value
        for key, value in tensors.items()
        if key.startswith(prefix)
    }
    if "identities" not in result:
        raise RuntimeError(f"Missing PP DSpark owner {owner} proposal")
    return result


def owned_token_rows(rids, token_counts, owner: int, pp_size: int):
    if len(rids) != len(token_counts):
        raise ValueError("PP DSpark token counts do not match request rows")
    rows, tokens = [], []
    offset = 0
    for i, (rid, count) in enumerate(zip(rids, token_counts)):
        if draft_owner(rid, pp_size) == owner:
            rows.append(i)
            tokens.extend(range(offset, offset + count))
        offset += count
    return rows, tokens

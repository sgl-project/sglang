# SPDX-License-Identifier: Apache-2.0
"""The coordinator: every rank of a load makes the same decision, or all fail."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator

# A gather allocates the largest ballot once per rank, on every rank.
_MAX_BALLOT_BYTES = 1 << 20
_MAX_ERROR_CHARS = 4000
_MAX_REPORT_CHARS = 16000


class Ballot(msgspec.Struct, frozen=True, kw_only=True):
    """One rank's part of a vote."""

    error: Optional[str] = None  # why this rank cannot go on
    agree: bytes = b""  # must be equal on every rank
    payload: bytes = b""  # gathered and returned as is


class VoteError(RuntimeError):
    """A vote failed. Every rank that gathered the ballots raises the same text."""


class _Wire(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """A ballot as it travels: every field is sent, and no other is read."""

    error: Optional[str]
    agree: bytes
    payload: bytes


def _wire(error: str) -> bytes:
    return msgspec.msgpack.encode(_Wire(error=error, agree=b"", payload=b""))


_UNSENDABLE = _wire("this rank's ballot cannot be cast")


def _cast(ballot: Ballot) -> bytes:
    """The ballot as it goes to the other ranks. One that cannot be sent is cast
    as this rank's error, so that no rank is left waiting in the gather."""
    try:
        if type(ballot) is not Ballot:
            raise TypeError("it is not a Ballot")
        error, agree, payload = ballot.error, ballot.agree, ballot.payload
        if error is not None and type(error) is not str:
            raise TypeError("Ballot.error is not a str")
        if type(agree) is not bytes or type(payload) is not bytes:
            raise TypeError("Ballot.agree and Ballot.payload must be bytes")
        if error is not None:
            error = (
                error[:_MAX_ERROR_CHARS].encode("utf-8", "backslashreplace").decode()
            )
        cast = msgspec.msgpack.encode(_Wire(error=error, agree=agree, payload=payload))
        if len(cast) > _MAX_BALLOT_BYTES:
            raise ValueError(
                f"it is {len(cast)} bytes, the limit is {_MAX_BALLOT_BYTES}"
            )
        return cast
    except Exception as e:
        try:
            return _wire(f"this rank's ballot cannot be cast: {type(e).__name__}: {e}")
        except Exception:
            return _UNSENDABLE


def _read(cast: object) -> Ballot:
    try:
        if type(cast) is not bytes or len(cast) > _MAX_BALLOT_BYTES:
            raise ValueError("not a ballot")
        wire = msgspec.msgpack.decode(cast, type=_Wire)
        # One encoding per ballot: a field sent twice would read as its last value.
        if msgspec.msgpack.encode(wire) != cast:
            raise ValueError("not as a ballot is cast")
        return Ballot(error=wire.error, agree=wire.agree, payload=wire.payload)
    except Exception as e:
        return Ballot(error=f"its ballot cannot be read: {type(e).__name__}")


def _held(agree: bytes) -> str:
    return f"{len(agree)} bytes with digest {hashlib.sha256(agree).hexdigest()[:16]}"


def _ranks(ranks: list[int]) -> str:
    return f"rank {ranks[0]}" if len(ranks) == 1 else f"ranks {ranks}"


def vote(*, group: Optional[GroupCoordinator], ballot: Ballot) -> list[Ballot]:
    """A barrier, then one gather of every rank's ballot, in rank order.

    Raises the same report, naming each failing rank, on every rank if any ballot
    has an error or the agree values differ. A rank is named by its place in the
    group.

    ``group`` holds every rank that loads this model together, and the caller
    answers for that; None is a load of one rank. Every rank casts every vote of
    a load, in the same order: a rank that failed casts it with its error instead
    of raising before it.

    A vote does not outlast the group: a rank that never arrives holds the others
    until the group's own timeout, and a rank that dies after the gather raises
    nothing.
    """
    cast = _cast(ballot)
    if group is None or group.world_size == 1:
        gathered = [cast]
    elif getattr(group, "cpu_group", None) is None:
        raise VoteError("The load group has no CPU process group to vote over")
    else:
        # A process group numbers its members in ascending order of their global
        # rank, and the ballots come back in that order.
        members = list(getattr(group, "ranks", None) or ())
        if members != sorted(members):
            raise VoteError(
                f"The ranks of the load group are not in ascending order: {members}"
            )
        group.barrier()
        gathered = group.all_gather_object(cast)
    ballots = [_read(cast) for cast in gathered]
    # Built from the gathered ballots only, so that it reads the same on every rank.
    failed: dict[str, list[int]] = {}
    ranks_by_value: dict[bytes, list[int]] = {}
    for rank, cast in enumerate(ballots):
        if cast.error is not None:
            failed.setdefault(cast.error, []).append(rank)
        else:
            ranks_by_value.setdefault(cast.agree, []).append(rank)
    # Every failing rank is named; a long reason is cut, so that all of them fit.
    room = max(_MAX_REPORT_CHARS // max(len(failed), 1), 64)
    problems = [
        f"{_ranks(ranks)}: {error if len(error) <= room else error[:room] + '...'}"
        for error, ranks in failed.items()
    ]
    if len(ranks_by_value) > 1:
        problems.append(
            "the ranks disagree: "
            + "; ".join(
                f"{_ranks(ranks)} hold {_held(value)}"
                for value, ranks in ranks_by_value.items()
            )
        )
    if problems:
        raise VoteError(
            f"{len(ballots)} rank(s) voted and cannot go on together: "
            + "; ".join(problems)
        )
    return ballots

"""CPU metadata coordination for an opt-in replicated image-DP encoder."""

import hashlib
import logging
import pickle
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch
import torch.distributed as dist

_active = ContextVar("image_dp_coordinator", default=None)
_MASK63 = (1 << 63) - 1


class ImageDPCoordinationError(RuntimeError):
    """All image-DP peers stop before publishing a failed encoding result."""


def image_dp_group(encode):
    owner = getattr(encode, "__self__", None)
    hook = getattr(owner, "get_image_dp_group", None)
    if hook is None or getattr(encode, "__name__", None) != "get_image_feature":
        return None
    return hook()


def image_dp_coordinator(encode):
    group = image_dp_group(encode)
    return ImageDPCoordinator(group) if group is not None else None


def _digest(value):
    raw = hashlib.blake2b(pickle.dumps(value, protocol=5), digest_size=16).digest()
    return [int.from_bytes(raw[i : i + 8], "little") & _MASK63 for i in (0, 8)]


class ImageDPCoordinator:
    def __init__(self, group):
        self.group = group
        self.sequence = 0

    def _bounds(self, values):
        # One fixed-size CPU gather; only inconsistent records need a min/max walk.
        record = torch.tensor(values, dtype=torch.int64)
        records = [torch.empty_like(record) for _ in range(self.group.world_size)]
        dist.all_gather(records, record, group=self.group.cpu_group)
        rows = [row.tolist() for row in records]
        first = rows[0]
        if all(row == first for row in rows[1:]):
            return first, first
        columns = list(zip(*rows))
        return [min(column) for column in columns], [max(column) for column in columns]

    def reconcile(self, keys, missing):
        low, high = self._bounds([len(keys), *_digest(keys), *_digest(bytes(missing))])
        if low[:3] != high[:3]:
            raise ImageDPCoordinationError("image-DP image identities/order differ")
        if low[3:] == high[3:]:
            return missing
        # Rare cache drift: retain canonical input order and encode the union of misses.
        mask = torch.tensor(missing, dtype=torch.uint8)
        dist.all_reduce(mask, op=dist.ReduceOp.MAX, group=self.group.cpu_group)
        if self.group.rank_in_group == 0:
            logging.getLogger(__name__).warning(
                "image-DP cache disagreement: recomputing union of %d image misses",
                int(mask.sum()),
            )
        return mask.tolist()

    @contextmanager
    def local_stage(self, phase, signature=lambda: ()):
        error = None
        fingerprint = [0, 0]
        try:
            yield
            fingerprint = _digest(signature())
        except ImageDPCoordinationError:
            # A nested stage already synchronized failure; do not enter another collective.
            raise
        except Exception as exc:
            error = exc
        low, high = self._bounds(
            [self.sequence * 4 + phase, *fingerprint, int(error is None)]
        )
        self.sequence += 1
        if low[3] == 0:
            raise ImageDPCoordinationError(
                "image-DP encoding failed on a peer"
            ) from error
        if low[:3] != high[:3]:
            raise ImageDPCoordinationError("image-DP encoding phases/layouts differ")

    @contextmanager
    def encoding(self):
        token = _active.set(self)
        try:
            with self.local_stage(2):
                yield
        finally:
            _active.reset(token)


def image_dp_local_stage(group, signature):
    coordinator = _active.get()
    if coordinator is None or coordinator.group is not group:
        return nullcontext()
    return coordinator.local_stage(1, signature)

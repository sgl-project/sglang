"""PP-consistent per-request verdicts for EIC storage prefetch.

EIC storage keys are pp-scoped, so pipeline stages hit independently and the same
request can resolve to a different prefetch length per stage, forking the
admitted prefix. Stages report their local result UP through the default
process group's TCPStore (non-collective KV RPCs, so PP0 polling is not a p2p
receive); PP0 forms the MIN once every stage answered. Delivery DOWN rides the
tensor loading_check already _pp_syncs -- adding a second receive chain at that
site deadlocks against the pipeline's own sends.

This owns pairing, verdict formation and GC; the caller owns transport DOWN.
"""

from __future__ import annotations

import hashlib
import logging
import pickle

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def eic_pp_unsupported_reason(server_args):
    if server_args.schedule_policy != "fcfs":
        return "schedule_policy must be fcfs"
    if server_args.enable_dp_attention:
        return "dp_attention is incompatible"
    if envs.SGLANG_REQ_WAITING_TIMEOUT.get() > 0:
        return (
            "SGLANG_REQ_WAITING_TIMEOUT must stay unset "
            "(per-stage clocks release out of lockstep)"
        )
    return None


class EICPPReconciler:
    VERDICT_CAP = 32
    EPOCH_CAP = 65536
    TOMBSTONE_TTL = 4096
    GC_AGE = 8192
    SWEEP_EVERY = 1024

    def __init__(self, prefix, pp_rank, pp_size, pp_group, rank):
        self.prefix = prefix
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.rank = rank
        self.eic = False

        self.round = 0
        self._store_handle = None
        self._pub_seq = 0
        self._next_seq = {}
        self._outbox = {}
        self._reports = {}
        self._verdicts = []
        self._tombstone = {}
        self._epoch = {}

    @property
    def enabled(self):
        return self.eic and self.pp_size > 1 and self.pp_group is not None

    @property
    def store(self):
        if self._store_handle is None:
            from torch.distributed import distributed_c10d

            self._store_handle = distributed_c10d._get_default_store()
        return self._store_handle

    @classmethod
    def buf_len(cls, extra_scalars):
        return extra_scalars + 1 + cls.VERDICT_CAP * 3

    @classmethod
    def pack_rows(cls, rows, buf, base):
        buf[base] = len(rows)
        if rows:
            buf[base + 1 : base + 1 + len(rows) * 3] = torch.tensor(
                [x for row in rows for x in row], dtype=torch.int64
            )

    @classmethod
    def unpack_rows(cls, buf, base):
        n = int(buf[base].item())
        flat = buf[base + 1 : base + 1 + n * 3].tolist()
        return [tuple(flat[i : i + 3]) for i in range(0, n * 3, 3)]

    @staticmethod
    def rid_hash(rid):
        # hash() is per-process salted, so it is not PP-uniform.
        return int.from_bytes(
            hashlib.blake2b(rid.encode(), digest_size=7).digest(), "big"
        )

    def bump_epoch(self, rid):
        # Clients may reuse a rid after an abort; a never-reused ordinal keeps a
        # straggler from pairing with the fresh incarnation.
        epoch = self._epoch.pop(rid, 0) + 1
        self._epoch[rid] = epoch
        while len(self._epoch) > self.EPOCH_CAP:
            self._epoch.pop(next(iter(self._epoch)))
        return epoch

    def report(self, h, epoch, value):
        if not self.enabled or self.rank != 0:
            return
        if self.pp_rank == 0:
            self._reports.setdefault((h, epoch), ({}, self.round))[0][0] = value
        else:
            self._outbox[(h, epoch)] = value

    def release(self, h, epoch):
        if not self.enabled:
            return
        self._tombstone[h] = (epoch, self.round + self.TOMBSTONE_TTL)
        self._outbox.pop((h, epoch), None)
        self._reports.pop((h, epoch), None)
        if self._verdicts:
            self._verdicts = [v for v in self._verdicts if v[0] != h]

    def collect(self):
        """One round on tp0. PP0 returns up to VERDICT_CAP (h, epoch, min) rows;
        other stages publish their outbox and return []."""
        self.round += 1
        if not self.enabled or self.rank != 0:
            return []
        if self.pp_rank != 0:
            self._publish()
            rows = []
        else:
            self._drain_peers()
            self._form()
            rows = self._verdicts[: self.VERDICT_CAP]
            del self._verdicts[: len(rows)]
        if self.round % self.SWEEP_EVERY == 0:
            self._sweep()
        return rows

    def _form(self):
        for key, (stages, _) in list(self._reports.items()):
            if len(stages) < self.pp_size:
                continue
            del self._reports[key]
            self._verdicts.append((*key, min(stages.values())))

    def _publish(self):
        if not self._outbox:
            return
        self.store.set(
            f"{self.prefix}/{self.pp_rank}/{self._pub_seq}", pickle.dumps(self._outbox)
        )
        self._pub_seq += 1  # only after a successful set, so there are no holes
        self._outbox = {}

    def _drain_peers(self):
        for stage in range(1, self.pp_size):
            seq = self._next_seq.get(stage, 0)
            while self.store.check([f"{self.prefix}/{stage}/{seq}"]):
                key = f"{self.prefix}/{stage}/{seq}"
                batch = pickle.loads(self.store.get(key))
                self.store.delete_key(key)
                seq += 1
                for (h, epoch), value in batch.items():
                    tomb = self._tombstone.get(h)
                    # A re-gate of the same rid carries a higher epoch and must
                    # pass, or that retry wedges forever.
                    if tomb is not None and epoch <= tomb[0]:
                        continue
                    self._reports.setdefault((h, epoch), ({}, self.round))[0][
                        stage
                    ] = value
            self._next_seq[stage] = seq

    def _sweep(self):
        for h, (_, expiry) in list(self._tombstone.items()):
            if expiry <= self.round:
                del self._tombstone[h]
        horizon = self.round - self.GC_AGE
        for key, (stages, born) in list(self._reports.items()):
            if born < horizon:
                del self._reports[key]
                logger.warning(
                    "pp reconcile: dropped report %s, missing stages %s",
                    key,
                    sorted(set(range(self.pp_size)) - set(stages)),
                )

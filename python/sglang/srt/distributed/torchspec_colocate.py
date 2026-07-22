"""TorchSpec colocate (MPS + NCCL) integration helpers.

This module is the engine-process side of the contract documented in
``docs/colocate/sglang_patch.md`` of the TorchSpec repo. It is loaded
unconditionally but only "fires" when the env-var sentinel
``TORCHSPEC_COLOCATE_TRANSFER_MODE=nccl`` is set by the TorchSpec
driver before launching sglang.

When active, it replaces sglang's per-engine NCCL world with a slice
of TorchSpec's ``2N``-rank **union NCCL world** (N trainer ranks +
N engine ranks, paired by index). The engine writes hidden states
directly to its paired trainer rank via P2P on that union world,
removing the Mooncake KV-store round-trip used in the disaggregated
path.

Public surface:

* :func:`is_colocate_active` — quick env-var check.
* :func:`read_colocate_env` — parsed env-var contract.
* :func:`init_union_default_pg` — replacement for sglang's
  ``init_distributed_environment`` body when colocate is on.
* :func:`build_engine_tp_ranks` — returns the contiguous rank range
  that maps to this engine's TP group inside the union world.
* :func:`build_hidden_states_writer` — connector factory used by the
  patched scheduler.

This file is the **only** new file added by the colocate patch; the
rest of the patch surface is small in-place edits in
``model_runner.py``, ``parallel_state.py``, ``scheduler.py``, and
``scheduler_output_processor_mixin.py``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)


_TRANSFER_MODE_ENV = "TORCHSPEC_COLOCATE_TRANSFER_MODE"
_PAIRED_TRAINER_RANK_ENV = "TORCHSPEC_COLOCATE_PAIRED_TRAINER_RANK"
_UNION_MASTER_ADDR_ENV = "TORCHSPEC_COLOCATE_UNION_MASTER_ADDR"
_UNION_MASTER_PORT_ENV = "TORCHSPEC_COLOCATE_UNION_MASTER_PORT"
_UNION_WORLD_SIZE_ENV = "TORCHSPEC_COLOCATE_UNION_WORLD_SIZE"
_UNION_N_PER_ROLE_ENV = "TORCHSPEC_COLOCATE_UNION_N_PER_ROLE"
_UNION_TIMEOUT_MIN_ENV = "TORCHSPEC_COLOCATE_UNION_TIMEOUT_MIN"
_UNION_INITIALIZED_ENV = "TORCHSPEC_COLOCATE_UNION_WORLD"

# The gloo process group spanning all 2N union-world ranks. The
# engine->trainer hidden-state P2P runs over this (not NCCL): trainer
# and engine share one physical GPU and NCCL refuses a communicator
# with two ranks on the same device. Set once by init_torch_distributed
# right after the meta_group new_group; read by build_hidden_states_writer.
_UNION_META_GROUP = None


def set_union_meta_group(group) -> None:
    """Stash the all-rank gloo union group for the hidden-states writer."""
    global _UNION_META_GROUP
    _UNION_META_GROUP = group


def get_union_meta_group():
    """Return the all-rank gloo union group, or None if not yet set."""
    return _UNION_META_GROUP


@dataclass(frozen=True)
class ColocateEnv:
    """Parsed contents of the TorchSpec colocate env-var contract."""

    paired_trainer_rank: int
    master_addr: str
    master_port: int
    world_size: int
    n_per_role: int
    timeout_minutes: int

    @property
    def init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.master_port}"

    def engine_global_rank(self, tp_rank: int = 0) -> int:
        """Return this engine subprocess' union-world rank.

        Engines occupy ``[N, 2N)`` in the union world. Under the
        colocate invariant (``engine_count * engine_tp_size ==
        training_world_size`` with ``engine_tp_size == 1``) each engine
        is paired 1:1 with a trainer, so the engine *index* is exactly
        ``paired_trainer_rank`` and the union rank is
        ``N + paired_trainer_rank``. ``tp_rank`` is the TP rank *within*
        this engine's own size-1 TP group (always 0) and is NOT the
        union-world offset — passing it as the offset made every engine
        claim rank N (fine at N=1, a hard rendezvous deadlock at N>1).
        """
        if not 0 <= self.paired_trainer_rank < self.n_per_role:
            raise ValueError(
                f"paired_trainer_rank={self.paired_trainer_rank} out of "
                f"range [0, {self.n_per_role})"
            )
        return self.n_per_role + self.paired_trainer_rank


def is_colocate_active() -> bool:
    """Return ``True`` iff TorchSpec's env-var sentinel is set."""
    val = os.environ.get(_TRANSFER_MODE_ENV, "").lower()
    active = val == "nccl"
    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] is_colocate_active: "
        f"{_TRANSFER_MODE_ENV}={val!r} -> active={active}",
    )
    return active


def read_colocate_env() -> Optional[ColocateEnv]:
    """Read and validate the TorchSpec colocate env-var contract.

    Returns ``None`` if colocate is not active. Raises
    ``RuntimeError`` if the sentinel is on but required env vars are
    missing — that's a driver-side bug we want to surface loudly.
    """
    if not is_colocate_active():
        return None

    try:
        return ColocateEnv(
            paired_trainer_rank=int(os.environ[_PAIRED_TRAINER_RANK_ENV]),
            master_addr=os.environ[_UNION_MASTER_ADDR_ENV],
            master_port=int(os.environ[_UNION_MASTER_PORT_ENV]),
            world_size=int(os.environ[_UNION_WORLD_SIZE_ENV]),
            n_per_role=int(os.environ[_UNION_N_PER_ROLE_ENV]),
            timeout_minutes=int(os.environ.get(_UNION_TIMEOUT_MIN_ENV, "30")),
        )
    except KeyError as e:
        raise RuntimeError(
            f"TorchSpec colocate is active ({_TRANSFER_MODE_ENV}=nccl) but "
            f"required env var {e.args[0]} is missing. The TorchSpec "
            f"driver must export the full union-world rendezvous before "
            f"launching sglang. See docs/colocate/sglang_patch.md."
        ) from e


def init_union_default_pg(
    *,
    tp_rank: int,
    local_rank: int,
    backend: str = "nccl",
) -> ColocateEnv:
    """Bring up TorchSpec's union NCCL world as the **default** PG.

    Replacement for sglang's ``init_distributed_environment`` body when
    colocate is active. After this returns:

    * ``torch.distributed.is_initialized()`` is True.
    * The default PG has ``world_size=2N`` ranks. Trainer ranks are
      ``[0, N)`` and have already joined via TorchSpec's
      ``init_union_world`` (this call unblocks them).
    * The current engine subprocess sits at rank ``N + tp_rank``.

    The caller is then responsible for creating sglang's TP group as
    a contiguous slice ``[N, 2N)`` via the patched
    ``initialize_model_parallel(..., tp_world_ranks=...)``.

    Args:
        tp_rank: The engine's TP rank within its own engine actor.
            For the colocate-config invariant (engine_count *
            engine_tp_size == training_world_size), this maps 1:1 to
            the engine slot in the union world's `[N, 2N)` block.
        local_rank: Local GPU index for this process. Passed to
            ``init_process_group`` as ``device_id`` so NCCL doesn't
            silently deadlock under Ray's CUDA_VISIBLE_DEVICES
            isolation (the Phase-3 lesson).
        backend: NCCL backend name (defaults to ``"nccl"``).

    Returns:
        The parsed :class:`ColocateEnv` for this process. Use it to
        build the TP-rank list and to look up the paired trainer rank
        for the hidden-states writer.

    Raises:
        RuntimeError: If colocate isn't active, or torch.distributed
            is already initialised (idempotency violation), or the env
            contract is incomplete.
    """
    import torch
    import torch.distributed as dist

    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] init_union_default_pg: "
        f"ENTRY tp_rank={tp_rank} local_rank={local_rank} backend={backend!r}",
    )

    env = read_colocate_env()
    if env is None:
        raise RuntimeError(
            "init_union_default_pg called but colocate is not active. "
            "Check is_colocate_active() before calling."
        )
    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] init_union_default_pg: "
        f"read_colocate_env OK: world_size={env.world_size} "
        f"n_per_role={env.n_per_role} init_method={env.init_method} "
        f"timeout={env.timeout_minutes}min paired_trainer_rank={env.paired_trainer_rank}",
    )

    if dist.is_initialized():
        # Already up — most likely because the trainer and this engine
        # share a Python process (test fixtures). Just verify shape.
        actual = dist.get_world_size()
        if actual != env.world_size:
            raise RuntimeError(
                f"torch.distributed already initialised with world_size="
                f"{actual} but colocate env declares world_size="
                f"{env.world_size}. Driver-side bug."
            )
        logger.info(
            "[torchspec-colocate] torch.distributed already initialised "
            "(world_size=%d); reusing it as the union default PG.",
            actual,
        )
        return env

    global_rank = env.engine_global_rank(tp_rank)
    device = torch.device("cuda", local_rank)

    logger.info(
        "[torchspec-colocate] Joining TorchSpec union world: "
        "tp_rank=%d global_rank=%d/%d local_rank=%d init_method=%s "
        "timeout=%dmin",
        tp_rank, global_rank, env.world_size, local_rank,
        env.init_method, env.timeout_minutes,
    )

    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] init_union_default_pg: "
        f"CALLING dist.init_process_group(backend={backend!r}, "
        f"world_size={env.world_size}, rank={global_rank}, "
        f"init_method={env.init_method!r}, timeout={env.timeout_minutes}min) "
        f"-- this BLOCKS until trainer rank also reaches its init_union_world",
    )
    dist.init_process_group(
        backend=backend,
        world_size=env.world_size,
        rank=global_rank,
        init_method=env.init_method,
        timeout=timedelta(minutes=env.timeout_minutes),
    )
    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] init_union_default_pg: "
        f"dist.init_process_group RETURNED -- union world is up (rank={global_rank}/"
        f"{env.world_size})",
    )

    # Defang sglang's subsequent `dist.new_group` calls so they don't
    # deadlock against the trainer's union-world setup.
    #
    # sglang's GroupCoordinator.__init__ creates per-engine TP/EP/PP/MoE
    # subgroups via `dist.new_group(ranks=[engine_ranks], ...)`. By
    # default, dist.new_group is a *world-collective* call — every rank
    # in the world group must call it with the same args, even if not
    # in `ranks`. In colocate mode the trainer ranks [0, N) are NOT
    # sglang ranks and have no business participating in sglang's
    # subgroup setup; they're busy creating the union-world meta_group.
    # The mismatch deadlocks both sides at the first collective
    # boundary.
    #
    # Setting `use_local_synchronization=True` on each new_group call
    # makes it a member-only barrier — non-member ranks skip it
    # entirely. We do this via a thin wrapper around dist.new_group
    # that only applies inside this engine subprocess; the trainer is a
    # different process and is unaffected.
    _original_new_group = dist.new_group

    def _local_only_new_group(*args, **kwargs):
        kwargs.setdefault("use_local_synchronization", True)
        return _original_new_group(*args, **kwargs)

    dist.new_group = _local_only_new_group
    logger.warning(
        f"[TS-COLOCATE-TRACE pid={os.getpid()}] init_union_default_pg: "
        f"installed local-only new_group default to break "
        f"world-collective deadlock with the trainer"
    )

    # Mark the union world as up so a subsequent
    # `init_distributed_environment` call (e.g. from a draft model
    # worker) becomes a no-op.
    os.environ[_UNION_INITIALIZED_ENV] = "1"

    return env


def build_engine_tp_ranks(env: ColocateEnv) -> list[int]:
    """Return the union-world ranks forming THIS engine's TP group.

    Under the colocate-config invariant ``engine_count *
    engine_tp_size == training_world_size`` with ``engine_tp_size ==
    1``, each engine is its own singleton TP group: union rank
    ``[N + paired_trainer_rank]``. Used both for
    ``initialize_model_parallel(..., tp_world_ranks=...)`` (whose
    length must equal ``tensor_model_parallel_size``) and for
    ``rebuild_world_group_engine_only`` (this engine's own ``_WORLD``).

    The old ``range(N, 2N)`` form returned every engine rank — a
    length-1 singleton at N=1 (so it worked) but a length-N list at
    N>1, which mismatched ``tp_size=1`` and cross-wired the engines'
    ``_WORLD`` groups.
    """
    return [env.n_per_role + env.paired_trainer_rank]


def rebuild_world_group_engine_only(env, local_rank, backend="nccl"):
    """Rebuild sglang's ``_WORLD`` GroupCoordinator to span only this
    engine's own union rank instead of the full ``2N`` union world.

    sglang's ``init_distributed_environment`` builds ``_WORLD`` from
    ``torch.distributed.get_world_size()``, which under colocate is
    the ``2N``-rank union world. But the trainer ranks ``[0, N)``
    never run sglang code, so any sglang world-level collective —
    e.g. ``get_available_gpu_memory(distributed=...,
    cpu_group=get_world_group().cpu_group)`` right after
    ``initialize_dp_attention``, or world barriers later — would hang
    forever waiting for the trainer half.

    This rebuilds ``_WORLD`` as an engine-only GroupCoordinator. The
    ``dist.new_group`` calls inside ``init_world_group`` inherit the
    ``use_local_synchronization=True`` monkey-patch installed by
    :func:`init_union_default_pg`, so only the engine ranks
    participate.
    """
    import sglang.srt.distributed.parallel_state as ps

    engine_ranks = build_engine_tp_ranks(env)
    if ps._WORLD is not None and ps._WORLD.world_size == len(engine_ranks):
        return  # already engine-only
    # Drop the (wrong) 2N-rank _WORLD and rebuild engine-only. The old
    # GroupCoordinator's process groups leak, but this runs once per
    # engine subprocess at startup, so the cost is negligible.
    ps._WORLD = None
    ps._WORLD = ps.init_world_group(engine_ranks, local_rank, backend)
    logger.warning(
        "[TS-COLOCATE-TRACE pid=%d] rebuilt sglang _WORLD as engine-only: "
        "ranks=%s world_size=%d",
        os.getpid(), engine_ranks, ps._WORLD.world_size,
    )


def build_hidden_states_writer():
    """Return a TorchSpec NcclHiddenStatesConnector for the spec_training callback.

    Imported lazily so disaggregated runs (where colocate is off)
    never pull torchspec into sglang's import graph. Raises
    ``ImportError`` with a clear remediation if torchspec isn't on
    the engine subprocess' ``PYTHONPATH``.
    """
    env = read_colocate_env()
    if env is None:
        raise RuntimeError(
            "build_hidden_states_writer called but colocate is not active."
        )

    try:
        from torchspec.inference.engine.nccl_hidden_states_connector import (
            NcclHiddenStatesConnector,
        )
    except ImportError as e:
        raise ImportError(
            "TorchSpec colocate is active but `torchspec` is not "
            "importable from the sglang engine subprocess. Ensure "
            "TorchSpec is installed (`pip install -e .` from the "
            "TorchSpec checkout) and that PYTHONPATH includes it."
        ) from e

    meta_group = get_union_meta_group()
    if meta_group is None:
        raise RuntimeError(
            "build_hidden_states_writer: union meta_group not set. "
            "init_torch_distributed must call set_union_meta_group "
            "before the scheduler builds the writer."
        )
    return NcclHiddenStatesConnector(
        dst_global_rank=env.paired_trainer_rank,
        group=meta_group,
    )

"""Runtime dispatch table for the JIT custom all-reduce (v2).

A ``Tuning`` answers one question per call: at this many bytes, inside or
outside a CUDA graph, which kernel variant runs -- or whether the caller hands
back to NCCL. It is a sorted band list, compiled once per communicator from a
``RawTuningSpec`` and then clipped to that communicator's workspace.

Vocabulary and runtime table only; the per-arch crossovers that feed it live in
``custom_all_reduce_v2_tuning``, and the package wires the two together.
"""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional, Protocol

from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo


class Dispatch(NamedTuple):
    algo: AllReduceAlgo
    use_graph: bool = False
    use_multicast: bool = False


class Band(NamedTuple):
    max_bytes: int
    dispatch: Optional[Dispatch]

    @staticmethod
    def normalize_list(bands: Iterable[Band]) -> tuple[Band, ...]:
        """Make sure the bands are strictly increasing in size."""
        kept: list[Band] = []
        for band in bands:
            if not kept or band.max_bytes > kept[-1].max_bytes:
                kept.append(band)
        return tuple(kept)


class Tuning(NamedTuple):
    eager: tuple[Band, ...]
    graph: tuple[Band, ...]
    num_push_blocks: int
    num_pull_blocks: int
    num_multicast_blocks: Optional[int]

    def select(self, nbytes: int, *, in_graph: bool) -> Optional[Dispatch]:
        for band in self.graph if in_graph else self.eager:
            if nbytes <= band.max_bytes:
                return band.dispatch
        return None

    def recompile(
        self,
        *,
        max_push_size: int,
        max_pull_size: int,
        uncap_pull: bool,
    ) -> Tuning:
        """
        :param max_push_size: the new maximum size for push bands
        :param max_pull_size: the new maximum size for pull bands
        :param uncap_pull: if True, 2-shot pull runs up to `max_pull_size`
                            instead of stopping at the tuned NCCL crossover
        """

        def _recompile(bands: tuple[Band, ...]) -> tuple[Band, ...]:
            def _map(b: Band) -> Band:
                if b.dispatch is None:
                    return b
                if b.dispatch.algo.is_push():
                    return Band(min(b.max_bytes, max_push_size), b.dispatch)
                if b.dispatch.algo.is_pull():
                    return Band(min(b.max_bytes, max_pull_size), b.dispatch)
                return b

            if not bands:
                return ()
            bands = tuple(_map(b) for b in bands)
            if uncap_pull:
                last = bands[-1]
                if last.dispatch and last.dispatch.algo == AllReduceAlgo.TWO_SHOT_PULL:
                    bands = bands[:-1] + (last._replace(max_bytes=max_pull_size),)
            return Band.normalize_list(bands)

        return Tuning(
            eager=_recompile(self.eager),
            graph=_recompile(self.graph),
            num_push_blocks=self.num_push_blocks,
            num_pull_blocks=self.num_pull_blocks,
            num_multicast_blocks=self.num_multicast_blocks,
        )

    def get_workspace_sizes(
        self,
        *,
        max_size: int,
        max_push_size: Optional[int],
        max_pull_size: Optional[int],
    ) -> tuple[int, int]:
        bands = self.eager + self.graph
        if max_push_size is None:
            max_push_size = max(
                (s for s, d in bands if d and d.algo.is_push()),
                default=0,
            )
            max_push_size = min(max_push_size, max_size)
        if max_pull_size is None:
            max_pull_size = max(
                (s for s, d in bands if d and d.algo.is_pull()),
                default=0,
            )
            max_pull_size = min(max_pull_size, max_size)
        return max_push_size, max_pull_size


# NOTE: the real impl can vary: e.g. file backend (load json)...
class RawTuningSpec(Protocol):
    def compile(self, can_use_multicast: bool) -> Tuning: ...


__all__ = [
    "Dispatch",
    "Band",
    "Tuning",
    "RawTuningSpec",
]

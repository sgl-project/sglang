# SPDX-License-Identifier: Apache-2.0
"""Dispatch policies for multi-instance disaggregated diffusion pipelines."""

import abc
import logging
import threading

logger = logging.getLogger(__name__)

_POLICY_KWARGS = {
    "round_robin": frozenset(),
    "max_free_slots": frozenset({"max_slots_per_instance"}),
}
_KNOWN_POLICY_KWARGS = frozenset().union(*_POLICY_KWARGS.values())


class DispatchPolicy(abc.ABC):
    def __init__(self, num_instances: int):
        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")
        self._num_instances = num_instances

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @abc.abstractmethod
    def select(self, active_counts: list[int] | None = None) -> int: ...

    def select_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        """Select an instance that has free capacity, or None if all full."""
        excluded_instances = excluded_instances or set()
        if not any(
            i not in excluded_instances and s > 0 for i, s in enumerate(free_slots)
        ):
            return None
        return self.select(active_counts=None)

    def record_completion(self, instance_id: int) -> None:
        pass


class RoundRobin(DispatchPolicy):
    def __init__(self, num_instances: int):
        super().__init__(num_instances)
        self._lock = threading.Lock()
        self._next = 0

    def select(self, active_counts: list[int] | None = None) -> int:
        with self._lock:
            chosen = self._next
            self._next = (self._next + 1) % self._num_instances
        return chosen

    def select_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        excluded_instances = excluded_instances or set()
        with self._lock:
            for _ in range(self._num_instances):
                idx = self._next
                self._next = (self._next + 1) % self._num_instances
                if idx in excluded_instances:
                    continue
                if free_slots[idx] > 0:
                    return idx
            return None


class MaxFreeSlotsFirst(DispatchPolicy):
    """Dispatch to the instance with the most free slots."""

    def __init__(self, num_instances: int, max_slots_per_instance: int = 1):
        super().__init__(num_instances)
        self._max_slots = max_slots_per_instance
        self._lock = threading.Lock()
        self._tiebreak = 0

    def select(self, active_counts: list[int] | None = None) -> int:
        with self._lock:
            if active_counts is None or len(active_counts) != self._num_instances:
                chosen = self._tiebreak % self._num_instances
                self._tiebreak += 1
                return chosen

            best_id = 0
            best_free = self._max_slots - active_counts[0]
            for i in range(1, self._num_instances):
                free = self._max_slots - active_counts[i]
                if free > best_free:
                    best_free = free
                    best_id = i
                elif free == best_free:
                    if i == (self._tiebreak % self._num_instances):
                        best_id = i

            self._tiebreak += 1

            if best_free <= 0:
                logger.warning(
                    "All %d instances are at capacity (%d slots each), "
                    "dispatching to instance %d anyway",
                    self._num_instances,
                    self._max_slots,
                    best_id,
                )

            return best_id

    def select_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        excluded_instances = excluded_instances or set()
        with self._lock:
            best_id = -1
            best_free = 0
            for i in range(self._num_instances):
                if i in excluded_instances:
                    continue
                if free_slots[i] > best_free:
                    best_free = free_slots[i]
                    best_id = i
                elif free_slots[i] == best_free and best_free > 0:
                    if i == (self._tiebreak % self._num_instances):
                        best_id = i

            self._tiebreak += 1

            if best_id < 0:
                return None
            return best_id


class PoolDispatcher:
    """Wraps three independent dispatch policies for encoder/denoiser/decoder pools."""

    def __init__(
        self,
        num_encoders: int,
        num_denoisers: int,
        num_decoders: int,
        policy_name: str = "round_robin",
        max_slots_per_instance: int = 1,
    ):
        self.encoder_policy = create_dispatch_policy(
            policy_name,
            num_encoders,
            max_slots_per_instance=max_slots_per_instance,
        )
        self.denoiser_policy = create_dispatch_policy(
            policy_name,
            num_denoisers,
            max_slots_per_instance=max_slots_per_instance,
        )
        self.decoder_policy = create_dispatch_policy(
            policy_name,
            num_decoders,
            max_slots_per_instance=max_slots_per_instance,
        )

    def select_encoder(self, active_counts: list[int] | None = None) -> int:
        return self.encoder_policy.select(active_counts)

    def select_denoiser(self, active_counts: list[int] | None = None) -> int:
        return self.denoiser_policy.select(active_counts)

    def select_decoder(self, active_counts: list[int] | None = None) -> int:
        return self.decoder_policy.select(active_counts)

    def select_encoder_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        return self.encoder_policy.select_with_capacity(
            free_slots, excluded_instances=excluded_instances
        )

    def select_denoiser_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        return self.denoiser_policy.select_with_capacity(
            free_slots, excluded_instances=excluded_instances
        )

    def select_decoder_with_capacity(
        self,
        free_slots: list[int],
        excluded_instances: set[int] | None = None,
    ) -> int | None:
        return self.decoder_policy.select_with_capacity(
            free_slots, excluded_instances=excluded_instances
        )


def create_dispatch_policy(name: str, num_instances: int, **kwargs) -> DispatchPolicy:
    policies = {
        "round_robin": RoundRobin,
        "max_free_slots": MaxFreeSlotsFirst,
    }
    cls = policies.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dispatch policy '{name}'. Available: {list(policies.keys())}"
        )
    unexpected_kwargs = sorted(set(kwargs) - _KNOWN_POLICY_KWARGS)
    if unexpected_kwargs:
        unexpected_args = ", ".join(unexpected_kwargs)
        raise TypeError(
            f"Unsupported dispatch policy kwargs for '{name}': {unexpected_args}"
        )

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in _POLICY_KWARGS[name]
    }
    return cls(num_instances=num_instances, **filtered_kwargs)

# SPDX-License-Identifier: Apache-2.0
"""Dispatch policies for multi-instance disaggregated diffusion pipelines."""

import abc
import threading


class DispatchPolicy(abc.ABC):
    def __init__(self, num_instances: int):
        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")
        self._num_instances = num_instances

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @abc.abstractmethod
    def select_with_capacity(self, free_slots: list[int]) -> int | None:
        """Select an instance that has free capacity, or None if all full."""
        ...


class RoundRobin(DispatchPolicy):
    def __init__(self, num_instances: int):
        super().__init__(num_instances)
        self._lock = threading.Lock()
        self._next = 0

    def select_with_capacity(self, free_slots: list[int]) -> int | None:
        with self._lock:
            for _ in range(self._num_instances):
                idx = self._next
                self._next = (self._next + 1) % self._num_instances
                if free_slots[idx] > 0:
                    return idx
            return None


class MaxFreeSlotsFirst(DispatchPolicy):
    """Dispatch to the instance with the most free slots."""

    def __init__(self, num_instances: int):
        super().__init__(num_instances)
        self._lock = threading.Lock()
        self._tiebreak = 0

    def select_with_capacity(self, free_slots: list[int]) -> int | None:
        with self._lock:
            best_id = -1
            best_free = 0
            for i in range(self._num_instances):
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
    ):
        self.encoder_policy = create_dispatch_policy(policy_name, num_encoders)
        self.denoiser_policy = create_dispatch_policy(policy_name, num_denoisers)
        self.decoder_policy = create_dispatch_policy(policy_name, num_decoders)

    def select_encoder_with_capacity(self, free_slots: list[int]) -> int | None:
        return self.encoder_policy.select_with_capacity(free_slots)

    def select_denoiser_with_capacity(self, free_slots: list[int]) -> int | None:
        return self.denoiser_policy.select_with_capacity(free_slots)

    def select_decoder_with_capacity(self, free_slots: list[int]) -> int | None:
        return self.decoder_policy.select_with_capacity(free_slots)


def create_dispatch_policy(name: str, num_instances: int) -> DispatchPolicy:
    policies = {
        "round_robin": RoundRobin,
        "max_free_slots": MaxFreeSlotsFirst,
    }
    cls = policies.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown dispatch policy '{name}'. Available: {list(policies.keys())}"
        )
    return cls(num_instances=num_instances)

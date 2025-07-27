# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from typing import Any

import torch

PAD_SLOT_ID = -1


class ConstantSizeCache(ABC):
    """
    Abstract base class for managing constant size caches
    like Mamba and Minimax.
    """

    def __init__(self, max_batch_size: int):
        # Maps between the request id and a dict that maps between the seq_id
        # and its index inside the cache
        self.cache_indices_mapping: dict[str, dict[int, int]] = {}
        self.free_cache_indices = list(range(max_batch_size))

    @property
    @abstractmethod
    def cache(self) -> Any:
        """Return the underlying cache tensor(s)"""
        pass

    @abstractmethod
    def _copy_cache(self, from_index: int, to_index: int):
        """Copy cache data from one index to another"""
        pass

    def current_run_tensors(self, **kwargs) -> tuple:
        """
        Return the tensors for the current run's conv and ssm state.
        """
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]

            if finished_requests_ids is None:
                finished_requests_ids = []

            if request_ids_to_seq_ids is None:
                request_ids_to_seq_ids = {}

            self._release_finished_requests(finished_requests_ids)
            state_indices = self._prepare_current_run_cache(
                request_ids_to_seq_ids, finished_requests_ids
            )

            # Use a safer method to create the tensor
            # Try to avoid issues during CUDA graph capture
            try:
                state_indices_tensor = torch.tensor(
                    state_indices, dtype=torch.int32, device="cuda"
                )
            except RuntimeError:
                # Fallback: create on CPU then move to GPU
                state_indices_tensor = torch.tensor(
                    state_indices, dtype=torch.int32
                ).to(device="cuda")
            cache_tensors = self.cache
        else:
            # CUDA graph capturing runs
            cache_tensors, state_indices_tensor = kwargs[
                "seqlen_agnostic_capture_inputs"
            ]

        return (cache_tensors, state_indices_tensor)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant state_indices into the CUDA graph input buffer
        """
        assert all(
            key in kwargs for key in ["request_ids_to_seq_ids", "finished_requests_ids"]
        )
        finished_requests_ids = kwargs["finished_requests_ids"]
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        assert "seqlen_agnostic_capture_inputs" in input_buffers
        _, input_state_indices_buffer = input_buffers["seqlen_agnostic_capture_inputs"]

        if finished_requests_ids is None:
            finished_requests_ids = []

        if request_ids_to_seq_ids is None:
            request_ids_to_seq_ids = {}

        self._release_finished_requests(finished_requests_ids)
        state_indices = self._prepare_current_run_cache(
            request_ids_to_seq_ids, finished_requests_ids
        )
        cuda_graph_pad_len = input_state_indices_buffer.shape[0] - len(state_indices)
        state_indices.extend([PAD_SLOT_ID] * cuda_graph_pad_len)

        # Use direct indexing to avoid creating any tensors during replay
        # Copy the state indices one by one to the buffer
        for i, state_idx in enumerate(state_indices):
            input_state_indices_buffer[i] = state_idx

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Cache during the CUDA graph replay
        runs.

        Note: This method should be overridden by subclasses to avoid creating
        tensors during CUDA graph capture.
        """
        # Try to avoid creating tensors during CUDA graph capture
        # This implementation should be overridden by subclasses
        try:
            state_indices_tensor = torch.full(
                (batch_size,), PAD_SLOT_ID, dtype=torch.int32, device="cuda"
            )
        except RuntimeError as e:
            if "operation not permitted when stream is capturing" in str(e):
                # Return a dummy tensor during capture - this should not happen
                # if subclasses properly override this method
                state_indices_tensor = torch.empty(
                    (batch_size,), dtype=torch.int32, device="cuda"
                )
            else:
                raise e

        return (self.cache, state_indices_tensor)

    def _assign_seq_id_to_cache_index(
        self,
        cur_rid: str,
        seq_id: int,
        finished_requests_ids,
        is_parallel_sampling: bool = False,
    ) -> int:
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.

        Args:
            cur_rid: current request id
            seq_id: sequence id for this request
            finished_requests_ids: list of finished request ids
            is_parallel_sampling: whether this is a parallel sampling scenario
        """
        if cur_rid in finished_requests_ids:
            # set as pad, do not allocate destination index
            return PAD_SLOT_ID
        elif cur_rid not in self.cache_indices_mapping:
            if not self.free_cache_indices:
                raise RuntimeError(
                    f"No free cache indices available for request {cur_rid}. "
                    f"Cache capacity may be exceeded or not properly managed."
                )
            destination_index = self.free_cache_indices.pop()
            self.cache_indices_mapping[cur_rid] = {seq_id: destination_index}
            return destination_index
        elif seq_id not in (seq_ids2indices := self.cache_indices_mapping[cur_rid]):
            if not self.free_cache_indices:
                raise RuntimeError(
                    f"No free cache indices available for request {cur_rid}, seq_id {seq_id}. "
                    f"Cache capacity may be exceeded or not properly managed."
                )

            if is_parallel_sampling and len(seq_ids2indices) > 0:
                # parallel sampling case: copy existing cache to new seq_id
                index_exists = next(iter(seq_ids2indices.values()))
                destination_index = self.free_cache_indices.pop()
                self._copy_cache(from_index=index_exists, to_index=destination_index)
                self.cache_indices_mapping[cur_rid][seq_id] = destination_index
                return destination_index
            else:
                # For non-parallel sampling or new independent requests, allocate fresh cache
                destination_index = self.free_cache_indices.pop()
                self.cache_indices_mapping[cur_rid][seq_id] = destination_index
                return destination_index
        else:
            return self.cache_indices_mapping[cur_rid][seq_id]

    def _prepare_current_run_cache(
        self,
        request_ids_to_seq_ids: dict[str, list[int]],
        finished_requests_ids: list[str],
    ) -> list[int]:
        return [
            self._assign_seq_id_to_cache_index(req_id, seq_id, finished_requests_ids)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]

    def _release_finished_requests(self, finished_seq_groups_req_ids: list[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.cache_indices_mapping:
                for seq_id in self.cache_indices_mapping[req_id]:
                    self.free_cache_indices.append(
                        self.cache_indices_mapping[req_id][seq_id]
                    )
                self.cache_indices_mapping.pop(req_id)

    def cleanup_orphaned_cache_entries(self, active_request_ids: set[str]):
        orphaned_req_ids = []
        for req_id in list(self.cache_indices_mapping.keys()):
            if req_id not in active_request_ids:
                orphaned_req_ids.append(req_id)

        if orphaned_req_ids:
            self._release_finished_requests(orphaned_req_ids)

"""SGLang side of the output store; the stored format and the lifecycle contract live in
``radix_transfer_plugins`` (see its ``store`` module).

A request that sets ``return_outputs_via_store`` keeps its routed experts, indexer
top-k, and sampling mask out of the response: the tokenizer collects them, stores them
when the request finishes, and returns ``meta_info["output_store_ref"]`` instead.
SGLang-specific assumptions:

- Non-streaming requests only.
- Each request, and each sub-request of a batch or n>1 request, is stored on its own;
  when a sibling fails, the objects of the others are left to the storage plane.
- The tokenizer process is a pure client and stores off its event loop.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Dict, Iterable, List, Optional

import msgspec
import numpy as np
import torch

from radix_transfer_plugins import OutputStore, StoreRef, create_backend
from radix_transfer_plugins.stashes.token_output import TokenOutputStash
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk

logger = logging.getLogger(__name__)

OUTPUT_STORE_REF_KEY = "output_store_ref"

# Arbitrary until measured; bounds concurrent puts and deletes per tokenizer process.
_MAX_STORE_WORKERS = 4


class TokenOutputCollector(msgspec.Struct):
    """Per-token outputs of one request, collected until it finishes."""

    routed_experts: Optional[torch.Tensor] = None
    indexer_topk: Optional[torch.Tensor] = None
    # None until the first chunk arrives, so a requested mask with zero rows still
    # produces (empty) fields.
    sampling_mask_chunks: Optional[List[SamplingMaskChunk]] = None

    def add_sampling_mask(self, chunk: SamplingMaskChunk) -> None:
        if self.sampling_mask_chunks is None:
            self.sampling_mask_chunks = []
        self.sampling_mask_chunks.append(chunk)

    def is_empty(self) -> bool:
        return (
            self.routed_experts is None
            and self.indexer_topk is None
            and self.sampling_mask_chunks is None
        )

    def inline_meta_info(self) -> Dict[str, int]:
        """Scalars that stay in meta_info next to output_store_ref."""
        if self.sampling_mask_chunks is None:
            return {}
        return {
            "output_token_sampling_mask_length": sum(
                len(chunk.lengths) for chunk in self.sampling_mask_chunks
            )
        }

    def to_stash(self) -> TokenOutputStash:
        """Runs on a writer thread; the tensors are on the CPU, so .numpy() copies nothing."""
        chunks = self.sampling_mask_chunks
        mask = {}
        if chunks is not None:
            mask = dict(
                output_token_sampling_mask_lengths=np.concatenate(
                    [chunk.lengths for chunk in chunks]
                ),
                output_token_sampling_mask_token_ids=np.concatenate(
                    [chunk.token_ids for chunk in chunks]
                ),
                output_token_sampling_logprobs=np.concatenate(
                    [chunk.logprobs for chunk in chunks]
                ),
            )
        return TokenOutputStash(
            routed_experts=_numpy(self.routed_experts),
            indexer_topk=_numpy(self.indexer_topk),
            **mask,
        )


class OutputStoreWriter:
    """Stores collectors off the event loop, and deletes objects whose ref is not sent."""

    def __init__(self, store: OutputStore) -> None:
        self._store = store
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_STORE_WORKERS, thread_name_prefix="output-store"
        )

    def submit_put(
        self, collector: TokenOutputCollector
    ) -> concurrent.futures.Future[StoreRef]:
        return self._executor.submit(self._put, collector)

    def cleanup_after(self, future: concurrent.futures.Future[StoreRef]) -> None:
        """Delete the object once a put whose ref will never be sent succeeds."""
        future.add_done_callback(self._cleanup_completed_put)

    def cleanup_refs(self, refs: Iterable[Dict]) -> None:
        """Delete the objects behind refs that were built but will never be sent."""
        for ref in refs:
            self._executor.submit(self._delete, StoreRef.from_builtins(ref))

    def _put(self, collector: TokenOutputCollector) -> StoreRef:
        (ref,) = self._store.put_many([collector.to_stash()])
        return ref

    def _cleanup_completed_put(
        self, future: concurrent.futures.Future[StoreRef]
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self._executor.submit(self._delete, future.result())

    def _delete(self, ref: StoreRef) -> None:
        try:
            self._store.delete_many([ref])
        except Exception:
            # Not retried here; the logged ref is the only record of the object.
            logger.error(
                "Failed to delete an output store object; ref=%s",
                json.dumps(ref.to_builtins()),
                exc_info=True,
            )


def maybe_create_output_store(
    *, backend: str, extra_config: Optional[str], disaggregation_mode: str
) -> Optional[OutputStoreWriter]:
    if backend == "none":
        return None
    if disaggregation_mode != "null":
        raise ValueError(
            "--output-store-backend is not supported with PD disaggregation yet"
        )
    return OutputStoreWriter(OutputStore(backend=create_backend(backend, extra_config)))


def _numpy(tensor: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if tensor is None else tensor.numpy()

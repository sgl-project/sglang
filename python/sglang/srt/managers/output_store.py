"""Optional Mooncake transport for per-request token outputs.

A request that sets ``return_outputs_via_store`` keeps its routed experts, indexer
top-k, and sampling mask out of the response: when it finishes, the tokenizer writes
them to Mooncake as one bundle and returns ``meta_info["output_store_ref"]`` instead.
The bundle uses the ``MooncakeBundleTransfer`` dict layout: each field is one dense
tensor, stored as the single row of the bundle's tensor batch, so a reader built on
the same transfer, with the same ``key_prefix``, can fetch it.

Key assumptions:

- Non-streaming requests only; a response carries its ref only after the object is
  stored.
- One reader reads each ref once and then removes the object, even when the read
  fails; SGLang keeps no copy of the ref.
- Objects are hard-pinned, so Mooncake never evicts them; a full store fails the put,
  and with it the request.
- SGLang removes an object only when its request is cancelled while the put runs. A
  ref that is never read keeps its object until the Mooncake master restarts: the
  client disconnects after the put, a sibling fails a batch or ``n>1`` request, a
  chat response fails to build, or the reader crashes before reading.
- SGLang is a pure Mooncake client and hosts none of the store's segments.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

import msgspec
import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk

logger = logging.getLogger(__name__)

OUTPUT_STORE_REF_KEY = "output_store_ref"

# Arbitrary until measured; bounds concurrent puts and cleanups per tokenizer process.
_MAX_STORE_WORKERS = 4

_SIZE_UNITS = {
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
}


class OutputStoreConfig(msgspec.Struct, frozen=True, kw_only=True):
    master_server_address: str
    local_hostname: str
    local_buffer_size: int
    key_prefix: str
    protocol: str = envs.MOONCAKE_PROTOCOL.default
    metadata_server: str = envs.MOONCAKE_TE_META_DATA_SERVER.default
    device_name: str = envs.MOONCAKE_DEVICE.default
    namespace: str = "default"
    partition: str = "default"
    replica_num: int = 1
    chunk_bytes: Optional[int] = None

    @classmethod
    def from_extra_config(cls, extra_config: Optional[str]) -> OutputStoreConfig:
        raw = json.loads(extra_config) if extra_config else {}
        struct_fields = msgspec.structs.fields(cls)
        unknown = sorted(set(raw) - {field.name for field in struct_fields})
        if unknown:
            raise ValueError(f"Unknown output store config keys: {unknown}")
        missing = [
            field.name
            for field in struct_fields
            if field.required and raw.get(field.name) in (None, "")
        ]
        if missing:
            raise ValueError(f"Output store config requires {missing}")

        values = {**raw, "local_buffer_size": _parse_size(raw["local_buffer_size"])}
        return msgspec.convert(values, type=cls)


class OutputStoreStash(Protocol):
    """The outputs of one response that an OutputStoreWriter stores together."""

    def is_empty(self) -> bool:
        """Nothing to write; checked on the event loop before the put."""
        ...

    def to_bundle_fields(self) -> Dict[str, torch.Tensor]:
        """One tensor per field, any dtype; runs on a writer thread, so it may copy."""
        ...


class TokenOutputStash(msgspec.Struct):
    """Per-token outputs of one request, unencoded until its final response."""

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
        meta_info = {}
        if self.sampling_mask_chunks is not None:
            meta_info["output_token_sampling_mask_length"] = sum(
                len(chunk.lengths) for chunk in self.sampling_mask_chunks
            )
        return meta_info

    def to_bundle_fields(self) -> Dict[str, torch.Tensor]:
        fields = {}
        if self.routed_experts is not None:
            fields["routed_experts"] = self.routed_experts
        if self.indexer_topk is not None:
            fields["indexer_topk"] = self.indexer_topk
        chunks = self.sampling_mask_chunks
        if chunks is not None:
            fields["output_token_sampling_mask_lengths"] = torch.from_numpy(
                np.concatenate([chunk.lengths for chunk in chunks])
            )
            fields["output_token_sampling_mask_token_ids"] = torch.from_numpy(
                np.concatenate([chunk.token_ids for chunk in chunks])
            )
            fields["output_token_sampling_logprobs"] = torch.from_numpy(
                np.concatenate([chunk.logprobs for chunk in chunks])
            )
        return fields


class OutputStoreWriter(ABC):
    """Stores one response's stash and resolves to its ``output_store_ref``.

    The ref is ``{"handle": ..., "fields": {name: {"dtype", "shape"}}}``: ``handle``
    is the writer's JSON-safe locator, and ``fields`` gives each tensor's dtype
    without the ``torch.`` prefix (``int32``, ``bfloat16``) and its shape, so readers
    parse one format whatever wrote it. A failed put must leave no object, since
    nothing else removes it.

    ``MooncakeBundleWriter`` serves ``--output-store-backend mooncake`` from the
    tokenizer process. GPU-resident outputs, such as SpecForge's hidden-state
    capture, need another writer on the attention-TP rank 0 scheduler: it may
    coalesce a scheduler batch's stashes into one ``batch_put_from`` from registered
    device memory, as long as each future resolves to its own stash's ref, and the
    scheduler holds those responses until their futures resolve.
    """

    @abstractmethod
    def submit_put(
        self, stash: OutputStoreStash
    ) -> concurrent.futures.Future[Dict[str, Any]]:
        """Store the stash off the caller's thread; resolves to its output_store_ref."""
        ...

    @abstractmethod
    def cleanup_after(self, future: concurrent.futures.Future[Dict[str, Any]]) -> None:
        """Remove the stored object once a put whose ref is never delivered succeeds."""
        ...


class MooncakeBundleWriter(OutputStoreWriter):
    """Writes each stash as one hard-pinned Mooncake bundle; staged through host
    memory, so stashes must hold CPU tensors."""

    def __init__(self, config: OutputStoreConfig) -> None:
        import mooncake.structured_object_store as structured_object_store
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig

        store = MooncakeDistributedStore()
        setup_error = store.setup(
            {
                "local_hostname": config.local_hostname,
                "metadata_server": config.metadata_server,
                "global_segment_size": 0,
                "local_buffer_size": config.local_buffer_size,
                "protocol": config.protocol,
                "rdma_devices": config.device_name,
                "master_server_addr": config.master_server_address,
            }
        )
        if setup_error:
            raise RuntimeError(f"Mooncake output store setup failed: {setup_error}")

        # Hard-pinned so Mooncake never evicts a bundle whose ref may be in flight:
        # the reader removes it after reading, cleanup_after() if it is never sent.
        replicate_config = ReplicateConfig()
        replicate_config.replica_num = config.replica_num
        replicate_config.with_hard_pin = True

        self._mooncake = structured_object_store
        self._transfer = structured_object_store.MooncakeBundleTransfer(
            store, key_prefix=config.key_prefix
        )
        self._put_options = {
            "type": "dict",
            "namespace": config.namespace,
            "partition": config.partition,
            "chunk_bytes": config.chunk_bytes,
            "config": replicate_config,
        }
        # The tensor batch takes any dtype, bfloat16 included; Mooncake ignores the
        # codec of a batch field.
        self._row_schema = structured_object_store.FieldSchema(
            codec="auto", nullable=False, metadata={"section": "batch"}
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_MAX_STORE_WORKERS, thread_name_prefix="output-store"
        )

    def submit_put(
        self, stash: OutputStoreStash
    ) -> concurrent.futures.Future[Dict[str, Any]]:
        """Write the stash as one bundle off the event loop; resolves to output_store_ref."""
        return self._executor.submit(self._put, stash)

    def cleanup_after(self, future: concurrent.futures.Future[Dict[str, Any]]) -> None:
        """Remove the bundle once a put whose ref will never be delivered succeeds."""
        future.add_done_callback(self._cleanup_completed_put)

    def _put(self, stash: OutputStoreStash) -> Dict[str, Any]:
        fields = stash.to_bundle_fields()
        ref = self._transfer.put(
            {name: tensor.contiguous().unsqueeze(0) for name, tensor in fields.items()},
            **self._put_options,
            field_schemas=dict.fromkeys(fields, self._row_schema),
        )
        return {
            "handle": self._mooncake.export_ref(ref),
            "fields": {
                name: {
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                }
                for name, tensor in fields.items()
            },
        }

    def _cleanup_completed_put(
        self, future: concurrent.futures.Future[Dict[str, Any]]
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            return
        self._executor.submit(self._cleanup, future.result())

    def _cleanup(self, output_store_ref: Dict[str, Any]) -> None:
        handle = output_store_ref["handle"]
        try:
            self._transfer.cleanup_dataproto(self._mooncake.import_ref(handle))
        except Exception:
            # Not retried here; the logged handle is the only record of the object.
            logger.error(
                "Failed to remove an output store object; handle=%s",
                json.dumps(handle),
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
    return MooncakeBundleWriter(OutputStoreConfig.from_extra_config(extra_config))


def _parse_size(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        for suffix, multiplier in _SIZE_UNITS.items():
            if text.endswith(suffix):
                return int(float(text[: -len(suffix)]) * multiplier)
        return int(text)
    raise ValueError(f"Invalid size {value!r}; use bytes or a string such as '2gb'")

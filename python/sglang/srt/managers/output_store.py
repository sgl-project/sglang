"""Optional Mooncake transport for per-request replay outputs.

A request that sets ``return_outputs_via_store`` keeps its routed experts, indexer
top-k, and sampling mask out of the response: when it finishes, the tokenizer writes
them to Mooncake as one bundle and returns ``meta_info["output_store_ref"]`` instead.
The bundle uses the ``MooncakeBundleTransfer`` dict layout with one row per field, so
a reader built on the same transfer, with the same ``key_prefix``, can fetch it.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from typing import Any, Dict, List, Optional

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


class OutputStoreStash(msgspec.Struct):
    """Replay outputs of one request, held unencoded until its final response."""

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


class OutputStore:
    def __init__(self, config: OutputStoreConfig) -> None:
        from mooncake.store import MooncakeDistributedStore, ReplicateConfig
        from mooncake.structured_object_store import (
            FieldSchema,
            MooncakeBundleTransfer,
            export_ref,
            import_ref,
        )

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

        self._config = config
        self._transfer = MooncakeBundleTransfer(store, key_prefix=config.key_prefix)
        self._field_schema_cls = FieldSchema
        self._export_ref = export_ref
        self._import_ref = import_ref
        self._replicate_config = None
        if config.replica_num > 1:
            self._replicate_config = ReplicateConfig()
            self._replicate_config.replica_num = config.replica_num
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
        fields = _bundle_fields(stash)
        ref = self._transfer.put(
            {name: [array] for name, array in fields.items()},
            type="dict",
            namespace=self._config.namespace,
            partition=self._config.partition,
            chunk_bytes=self._config.chunk_bytes,
            config=self._replicate_config,
            field_schemas={
                name: self._field_schema_cls(
                    codec="typed_ragged",
                    nullable=False,
                    metadata={"section": "non_tensor_batch", "dtype": str(array.dtype)},
                )
                for name, array in fields.items()
            },
        )
        return {
            "handle": self._export_ref(ref),
            "fields": {
                name: {"dtype": str(array.dtype), "shape": list(array.shape)}
                for name, array in fields.items()
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
            self._transfer.cleanup_dataproto(self._import_ref(handle))
        except Exception:
            # Not retried here; the logged handle is the only record of the object.
            logger.error(
                "Failed to remove an output store object; handle=%s",
                json.dumps(handle),
                exc_info=True,
            )


def maybe_create_output_store(
    *, backend: str, extra_config: Optional[str], disaggregation_mode: str
) -> Optional[OutputStore]:
    if backend == "none":
        return None
    if disaggregation_mode != "null":
        raise ValueError(
            "--output-store-backend is not supported with PD disaggregation yet"
        )
    return OutputStore(OutputStoreConfig.from_extra_config(extra_config))


def _bundle_fields(stash: OutputStoreStash) -> Dict[str, np.ndarray]:
    fields = {}
    if stash.routed_experts is not None:
        fields["routed_experts"] = stash.routed_experts.numpy()
    if stash.indexer_topk is not None:
        fields["indexer_topk"] = stash.indexer_topk.numpy()
    chunks = stash.sampling_mask_chunks
    if chunks is not None:
        fields["output_token_sampling_mask_lengths"] = np.concatenate(
            [chunk.lengths for chunk in chunks]
        )
        fields["output_token_sampling_mask_token_ids"] = np.concatenate(
            [chunk.token_ids for chunk in chunks]
        )
        fields["output_token_sampling_logprobs"] = np.concatenate(
            [chunk.logprobs for chunk in chunks]
        )
    return fields


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

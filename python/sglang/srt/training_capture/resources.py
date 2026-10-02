"""Passive rank-local resources, with transport shutdown before buffer release."""

from __future__ import annotations

import os
from typing import ClassVar

import msgspec
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.snapshot_writer import PublicationJournal


class CaptureResources:
    # An unsuccessful transport stop must retain registered storage even if the
    # startup exception is discarded. Process teardown is the final backstop.
    _retained: ClassVar[set[CaptureResources]] = set()

    def __init__(self):
        self.store = self.pool = self.catalog = self.journal = self.exporter = None
        self.closed = False

    @classmethod
    def prepare(cls, *, config, kv, partition, source_pool, pin_memory=True):
        resources = cls()
        partition.local_layers(kv)
        if not partition.active:
            return resources
        try:
            if partition.heads:
                resources.exporter = SelectedLayerKVExporter.from_pool(
                    kv, source_pool, partition=partition
                )
            token = (
                os.environ[config.catalog_token_env]
                if config.catalog_token_env
                else None
            )
            resources.catalog = HTTPCaptureCatalog(
                config.catalog_endpoint,
                bearer_token=token,
                timeout=config.http_timeout_seconds,
                attempts=config.http_attempts,
            )
            resources.store = MooncakeSnapshotStore.connect(
                msgspec.to_builtins(config.store),
                replica_num=config.replica_num,
                max_receive_bytes=config.max_host_bytes,
            )
            resources._allocate(
                config,
                kv,
                partition,
                pin_memory,
                device=getattr(source_pool, "device", None),
            )
        except Exception:
            resources.close()
            raise
        return resources

    @classmethod
    def from_connected(cls, *, config, kv, exporter, store, catalog, pin_memory=True):
        """Take ownership of an already connected single-rank transport."""
        resources = cls()
        resources.store, resources.catalog, resources.exporter = (
            store,
            catalog,
            exporter,
        )
        try:
            resources._allocate(config, kv, None, pin_memory)
        except Exception:
            resources.close()
            raise
        return resources

    def _allocate(self, config, kv, partition, pin_memory, *, device=None):
        self.pool = HostBufferPool(
            kv=kv,
            max_tokens=config.max_sample_tokens,
            slots=config.max_inflight_samples,
            max_bytes=config.max_host_bytes,
            registrar=self.store,
            manifest_bytes=config.manifest_buffer_bytes,
            pin_memory=pin_memory,
            device=self.exporter.device if self.exporter is not None else device,
            kv_d2h_batch_tokens=config.kv_d2h_batch_tokens,
            teacher_d2h_batch_tokens=config.teacher_d2h_batch_tokens,
            max_device_bytes=config.max_device_bytes,
            partition=partition,
        )
        if partition is None or partition.include_aux:
            self.journal = PublicationJournal(config.journal_directory)

    def close(self):
        """Caller must first stop workers and synchronize outstanding CUDA copies."""
        if self.closed:
            return
        try:
            if self.store is not None:
                self.store.close()
            if self.journal is not None:
                self.journal.close()
        except Exception:
            self._retained.add(self)
            raise
        self.closed = True
        self._retained.discard(self)

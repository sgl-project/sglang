# SPDX-License-Identifier: Apache-2.0
"""Snapshot loading: the order every snapshot source follows (sgl-project/sglang#50943).

Reader, on every rank: pre_load_check (lookup, vote), SnapshotLoader.load_model
(fetch, install, validate), then the post-load vote. Owner: a raw load through
SnapshotLoader.load_model, export_snapshot (export, attest), then publish_snapshot
(publish, vote, announce, vote). This module only calls the parts in that order:
the transport, the coordinator, load-diff and the validator.
"""

from __future__ import annotations

import dataclasses
import hashlib
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
from torch import nn

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _get_quantization_config,
    _initialize_model,
    get_model_loader,
    on_model_constructed,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.weight_cache import load_diff, validator
from sglang.srt.weight_cache.coordinator import Ballot, vote
from sglang.srt.weight_cache.schema import (
    LoadSpec,
    SnapshotManifest,
    StorageMeta,
)
from sglang.srt.weight_cache.transport import WeightTransport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator


def pre_load_check(
    *,
    loader: BaseModelLoader,
    spec: LoadSpec,
    device_config: DeviceConfig,
    group: GroupCoordinator,
) -> Optional[SnapshotManifest]:
    """Every rank of ``group`` runs this before load_model: lookup, then vote.

    After it returns, every rank loads the same snapshot version, or every rank
    loads raw.
    """
    manifest, error = None, None
    try:
        manifest = loader.lookup(spec=spec, device_config=device_config)
    except Exception as e:  # every rank must learn it before any rank moves on
        error = repr(e)
    version = manifest.version if manifest is not None else ""
    agree = f"{spec.without_rank().digest()}/{version}".encode()
    try:
        vote(group=group, ballot=Ballot(error=error, agree=agree))
    except BaseException:
        loader.close()
        raise
    return manifest


class Exported(msgspec.Struct, frozen=True, kw_only=True):
    """What one owner rank prepared to publish; it holds no reference to the model."""

    storages: dict[str, torch.Tensor]
    load_diff: bytes = b""
    attestation: bytes = b""
    error: Optional[str] = None  # voted on by publish_snapshot


class SnapshotLoader(BaseModelLoader):
    """Base of the loaders that can read a snapshot: cache daemon, store, peer.

    A subclass says only where a snapshot is found (_find) and how a published one
    becomes findable or is taken back (_announce, _withdraw); the order of the
    steps is fixed here. A snapshot is installed and never post-processed; without
    one, load_model loads raw. An owner (owner=True) records each raw-loaded
    model's construction so that it can publish it. One object serves one load.
    """

    def __init__(self, load_config: LoadConfig, *, owner: bool = False):
        super().__init__(load_config)
        self._owner = owner
        self._manifest: Optional[SnapshotManifest] = None
        self._transport: Optional[WeightTransport] = None
        self._storages: dict[str, torch.Tensor] = {}
        # (id(model), (load-diff's record, validator's record)) of the last raw load.
        self._constructed: Optional[tuple[int, tuple[Any, Any]]] = None

    @abstractmethod
    def _find(
        self, *, spec: LoadSpec, device_config: DeviceConfig
    ) -> Optional[tuple[SnapshotManifest, WeightTransport]]:
        """The snapshot published for exactly ``spec``, with a transport to fetch it.

        Read-only and bounded in time; raises if this rank cannot load at all.
        """

    @abstractmethod
    def _announce(self, *, spec: LoadSpec, manifest: SnapshotManifest) -> None:
        """Owner: make ``manifest`` findable under ``spec`` for this rank."""

    @abstractmethod
    def _withdraw(self, *, spec: LoadSpec, manifest: SnapshotManifest) -> None:
        """Owner: undo _announce on this rank. Idempotent."""

    def lookup(
        self, *, spec: LoadSpec, device_config: DeviceConfig
    ) -> Optional[SnapshotManifest]:
        self.close()
        found = self._find(spec=spec, device_config=device_config)
        if found is not None:
            self._manifest, self._transport = found
        return self._manifest

    def load_model(
        self, *, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        if self._manifest is None:
            return self._load_raw(
                model_config=model_config, device_config=device_config
            )
        return self._load_snapshot(
            model_config=model_config, device_config=device_config
        )

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()
        self._manifest, self._transport, self._storages = None, None, {}

    def export_snapshot(self, *, model: nn.Module) -> Exported:
        """Owner: export and attest ``model``, which this loader loaded raw.

        Every rank calls it before any forward pass and before the engine's
        post-load steps. With a transport that consumes the storages (vmm_fd), the
        caller then drops ``model`` and the caches holding its tensors, and calls
        publish_snapshot.
        """
        constructed, self._constructed = self._constructed, None
        try:
            if constructed is None or constructed[0] != id(model):
                raise RuntimeError(
                    "Only a model that this owner built and loaded raw can publish"
                )
            diff_record, check_record = constructed[1]
            storages, payload = load_diff.export(model=model, constructed=diff_record)
            attestation = validator.attest(
                model=model, storages=storages, constructed=check_record
            )
        except Exception as e:  # voted on in publish_snapshot, so no rank waits
            return Exported(storages={}, error=repr(e))
        return Exported(storages=storages, load_diff=payload, attestation=attestation)

    def publish_snapshot(
        self,
        *,
        exported: Exported,
        spec: LoadSpec,
        transport: WeightTransport,
        group: GroupCoordinator,
    ) -> SnapshotManifest:
        """Owner: publish what export_snapshot prepared; all ranks call it together.

        The version becomes findable only once every rank published and announced
        it; otherwise every rank takes back what it published.
        """
        storages_meta = {
            name: StorageMeta(nbytes=t.nbytes, device=t.device.type)
            for name, t in exported.storages.items()
        }
        error, source = exported.error, {}
        if error is None:
            try:
                source = transport.publish(storages=exported.storages)
            except Exception as e:
                error = repr(e)
        try:
            ballots = vote(
                group=group,
                ballot=Ballot(
                    error=error,
                    agree=spec.without_rank().digest().encode(),
                    payload=hashlib.sha256(exported.attestation).digest(),
                ),
            )
        except BaseException:
            transport.close()
            raise
        manifest = SnapshotManifest(
            version=hashlib.sha256(b"".join(b.payload for b in ballots)).hexdigest(),
            storages=storages_meta,
            load_diff=exported.load_diff,
            attestation=exported.attestation,
            source=source,
        )
        error = None
        try:
            self._announce(spec=spec, manifest=manifest)
        except Exception as e:
            error = repr(e)
        try:
            vote(group=group, ballot=Ballot(error=error))
        except BaseException:
            self._withdraw(spec=spec, manifest=manifest)
            transport.close()
            raise
        return manifest

    def _load_snapshot(
        self, *, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        try:
            skeleton = self._build_skeleton(model_config)
            constructed = validator.record_construction(skeleton)
            storages = self._transport.fetch(
                manifest=self._manifest, device=device_config.device
            )
            model = load_diff.install(
                skeleton=skeleton, manifest=self._manifest, storages=storages
            ).eval()
            validator.validate(
                model=model,
                storages=storages,
                attestation=self._manifest.attestation,
                constructed=constructed,
            )
        except BaseException:
            # Nothing falls back once bytes moved; unmap now, vmm_fd maps them all.
            self.close()
            raise
        self._storages = storages
        return model

    def _build_skeleton(self, model_config: ModelConfig) -> nn.Module:
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
            return _initialize_model(model_config, self.load_config, quant_config)

    def _load_raw(
        self, *, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        raw_config = dataclasses.replace(
            self.load_config,
            load_format=self.load_config.fallback_load_format,
            fallback_load_format=LoadFormat.AUTO,
        )
        loader = get_model_loader(raw_config, model_config)
        if isinstance(loader, SnapshotLoader):
            raise ValueError(f"{raw_config.load_format} is not a raw load format")
        if not self._owner:
            return loader.load_model(
                model_config=model_config, device_config=device_config
            ).eval()
        records = {}
        with on_model_constructed(
            lambda m: records.__setitem__(id(m), self._record_construction(m))
        ):
            model = loader.load_model(
                model_config=model_config, device_config=device_config
            ).eval()
        # A throwaway build in the same load (presharded's signature) is dropped.
        if id(model) in records:
            self._constructed = (id(model), records[id(model)])
        return model

    @staticmethod
    def _record_construction(model: nn.Module) -> tuple[Any, Any]:
        return (
            load_diff.record_construction(model),
            validator.record_construction(model),
        )

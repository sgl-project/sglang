# SPDX-License-Identifier: Apache-2.0
"""Snapshot loading: the order every snapshot source follows (sgl-project/sglang#50943).

Owners and readers first call pre_load_check to identify the load, check that it
may use a snapshot, look up a snapshot, preflight any hit and vote. A reader with
a hit fetches, installs and validates it. Every prepared load votes on its
result. An owner that loaded from a checkpoint can then export and publish it
under the prepared identity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
import uuid
import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union

import msgspec
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.model_loader.loader import (
    BaseModelLoader,
    _get_quantization_config,
    _initialize_model,
    get_model_loader,
    require_checkpoint_loader,
)
from sglang.srt.model_loader.utils import on_model_constructed, set_default_torch_dtype
from sglang.srt.platforms import current_platform
from sglang.srt.weight_cache import load_diff, validator
from sglang.srt.weight_cache.config import build_load_spec
from sglang.srt.weight_cache.coordinator import Ballot, vote
from sglang.srt.weight_cache.schema import LoadSpec, SnapshotManifest, StorageMeta
from sglang.srt.weight_cache.transport import WeightTransport

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.distributed.parallel_state import GroupCoordinator

logger = logging.getLogger(__name__)

# An installed tensor or view can outlive the model, including on an install or
# validation failure. Keep the backing tensors and their mapping owners alive
# until the process exits.
_LOADED_SNAPSHOT_MAPPINGS: list[tuple[WeightTransport, tuple[torch.Tensor, ...]]] = []


class _PreflightDeviceGuard(TorchFunctionMode):
    """Reject explicit accelerator placement that overrides the meta default."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        device = kwargs.get("device")
        if func is torch.Tensor.to and len(args) > 1:
            target = args[1]
            if isinstance(target, (str, int, torch.device)):
                device = target
            elif isinstance(target, torch.Tensor):
                device = target.device
        if (
            getattr(func, "__name__", None) in ("cuda", "xpu", "npu")
            or isinstance(device, int)
            or (device is not None and torch.device(device).type not in ("cpu", "meta"))
        ):
            raise RuntimeError(
                "Snapshot preflight attempted to allocate accelerator memory"
            )
        return func(*args, **kwargs)


# Quantization methods a snapshot may carry. Add one only with its raw-load
# parity record (roadmap E2).
_SNAPSHOT_QUANT_METHODS = frozenset({""})  # unquantized


def snapshot_support(spec: LoadSpec) -> Optional[str]:
    """None if this load may use a snapshot, else why every rank loads raw."""
    if spec.quant_method not in _SNAPSHOT_QUANT_METHODS:
        return f"no snapshot parity record for quantization {spec.quant_method!r}"
    if spec.enable_eplb:
        return "EPLB rewrites expert weights in place"
    return None


class Found(NamedTuple):
    """The source has this rank's snapshot."""

    manifest: SnapshotManifest
    transport: WeightTransport


@dataclasses.dataclass(frozen=True)
class Pending:
    """The owner is still publishing; ask again after ``retry_after`` seconds."""

    retry_after: float


def _vote_or_close(
    *, loader: BaseModelLoader, group: Optional[GroupCoordinator], ballot: Ballot
) -> None:
    try:
        vote(group=group, ballot=ballot)
    except BaseException:
        try:
            loader.close()
        except Exception:
            logger.exception("Failed to close a rejected snapshot load")
        raise


def pre_load_check(
    *,
    loader: BaseModelLoader,
    model_config: ModelConfig,
    checkpoint_id: str,
    device_config: DeviceConfig,
    group: Optional[GroupCoordinator],
) -> Optional[SnapshotManifest]:
    """Look up and preflight a hit, then vote before fetching or loading raw.

    After it returns, every rank loads the same snapshot version, or every rank
    loads raw. An owner publishes under the identity built here.
    """
    manifest, error = None, None
    agree = b""
    try:
        spec = build_load_spec(
            model_config,
            checkpoint_id=checkpoint_id,
            device=device_config.device,
            draft_model_idx=loader.load_config.draft_model_idx,
        )
        raw_only = snapshot_support(spec)
        if raw_only is None:
            manifest = loader.lookup(spec=spec, device_config=device_config)
        else:
            loader.close()  # drop any earlier lookup: every rank loads raw
        if manifest is not None:
            if manifest.spec_digest != spec.digest():
                raise ValueError(
                    f"Snapshot was published for LoadSpec {manifest.spec_digest}, "
                    f"expected {spec.digest()}"
                )
            if isinstance(loader, SnapshotLoader):
                loader._preflight(model_config=model_config, manifest=manifest)
        version = manifest.version if manifest is not None else None
        agree = msgspec.msgpack.encode((spec.digest(include_rank=False), version))
    except Exception as e:  # every rank must learn it before any rank moves on
        error = repr(e)
    _vote_or_close(loader=loader, group=group, ballot=Ballot(error=error, agree=agree))
    if error is None and isinstance(loader, SnapshotLoader):
        loader._prepared_spec = spec
        loader._publish_rejection = raw_only  # a raw-only owner cannot publish
        loader._prepared_group = group
        loader._ready_for_load = True
    return manifest


class Exported(msgspec.Struct, frozen=True, kw_only=True):
    """What one owner rank prepared to publish; it holds no reference to the model."""

    storages: dict[str, torch.Tensor]
    shared: frozenset[str] = frozenset()  # storages that back an nn.Parameter
    spec: Optional[LoadSpec] = None
    load_diff: bytes = b""
    attestation: bytes = b""
    error: Optional[str] = None  # voted on by publish_snapshot


class SnapshotLoader(BaseModelLoader):
    """Base of the loaders that can read a snapshot: cache daemon, store, peer.

    Sources implement lookup, private staging, and the shared commit marker.
    A snapshot is installed without post-processing. After a miss, load_model
    uses the configured fallback loader. An owner records construction when
    loading from a supported checkpoint format. One object serves one load.
    """

    def __init__(
        self,
        load_config: LoadConfig,
        *,
        owner: bool = False,
        max_pending_wait: float = 300.0,
    ):
        super().__init__(load_config)
        self._owner = owner
        self._max_pending_wait = max_pending_wait  # seconds lookup waits on Pending
        self._prepared_spec: Optional[LoadSpec] = None
        self._prepared_group: Optional[GroupCoordinator] = None
        self._ready_for_load = False
        self._manifest: Optional[SnapshotManifest] = None
        self._transport: Optional[WeightTransport] = None
        self._prepared_snapshot: Optional[tuple[nn.Module, Any]] = None
        self._constructed: Optional[
            tuple[weakref.ReferenceType[nn.Module], tuple[Any, Any]]
        ] = None
        self._publish_rejection: Optional[str] = None

    @abstractmethod
    def _find(
        self, *, tag: str, group_tag: str, device_config: DeviceConfig
    ) -> Optional[Union[Found, Pending]]:
        """Find this rank's manifest and transport under ``tag``.

        Return None if absent (every rank loads raw), Pending while the owner is
        still publishing, and raise if the source cannot serve this load (every
        rank stops; none loads raw). Return only the version selected by the
        shared commit marker under ``group_tag``. This is read-only and must
        finish within a bounded time.
        """

    @abstractmethod
    def _stage(self, *, tag: str, manifest: SnapshotManifest) -> None:
        """Owner: store this rank's manifest privately under tag and version."""

    @abstractmethod
    def _announce(self, *, tag: str, version: str) -> None:
        """Owner leader: atomically mark the staged version visible under the group tag."""

    @abstractmethod
    def _retract(self, *, tag: str, version: str) -> None:
        """Owner: atomically revoke this version under the group tag.

        Idempotent across ranks. A revoked version cannot be announced later.
        Any other committed version under the same tag stays visible.
        """

    @abstractmethod
    def _withdraw(self, *, tag: str, manifest: SnapshotManifest) -> None:
        """Owner: discard this rank's private staging. Idempotent."""

    def lookup(
        self, *, spec: LoadSpec, device_config: DeviceConfig
    ) -> Optional[SnapshotManifest]:
        self.close()
        tag, group_tag = spec.digest(), spec.digest(include_rank=False)
        deadline = time.monotonic() + self._max_pending_wait
        while isinstance(
            found := self._find(
                tag=tag, group_tag=group_tag, device_config=device_config
            ),
            Pending,
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"The owner did not publish within {self._max_pending_wait:g} s"
                )
            time.sleep(min(found.retry_after, remaining))
        if found is not None:
            self._manifest, self._transport = found
        return self._manifest

    def load_model(
        self, *, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        if not self._ready_for_load:
            raise RuntimeError("Run a fresh pre_load_check before load_model")
        self._ready_for_load = False

        group = self._prepared_group
        error, model = None, None
        try:
            if self._manifest is None:
                model = self._load_raw(
                    model_config=model_config, device_config=device_config
                )
            else:
                model = self._load_snapshot(device_config=device_config)
        except Exception as e:
            error = repr(e)
        _vote_or_close(loader=self, group=group, ballot=Ballot(error=error))
        assert model is not None
        return model

    def close(self) -> None:
        self._prepared_spec = None
        self._prepared_group = None
        self._prepared_snapshot = None
        self._ready_for_load = False
        self._constructed = None
        self._publish_rejection = None
        transport, self._transport = self._transport, None
        self._manifest = None
        if transport is not None:
            transport.close()

    def export_snapshot(self, *, model: nn.Module) -> Exported:
        """Owner: export and attest ``model``, which this loader loaded raw.

        Run pre_load_check before the raw load. Every rank exports before any
        forward pass and before the engine's post-load steps. With a transport
        that consumes the storages (vmm_fd), the caller then drops ``model``
        and the caches holding its tensors, and calls publish_snapshot.
        """
        constructed, self._constructed = self._constructed, None
        try:
            if self._prepared_spec is None:
                raise RuntimeError("Run pre_load_check before the owner's raw load")
            if self._publish_rejection is not None:
                raise ValueError(self._publish_rejection)
            if constructed is None or constructed[0]() is not model:
                raise RuntimeError(
                    "Only a model that this owner loaded directly from a checkpoint can publish"
                )
            diff_record, check_record = constructed[1]
            storages, payload = load_diff.export(model=model, constructed=diff_record)
            attestation = validator.attest(
                model=model, storages=storages, constructed=check_record
            )
        except Exception as e:  # voted on in publish_snapshot, so no rank waits
            return Exported(storages={}, error=repr(e))
        params = {p.untyped_storage().data_ptr() for p in model.parameters()}
        return Exported(
            storages=storages,
            shared=frozenset(
                name
                for name, t in storages.items()
                if t.untyped_storage().data_ptr() in params
            ),
            spec=self._prepared_spec,
            load_diff=payload,
            attestation=attestation,
        )

    def publish_snapshot(
        self,
        *,
        exported: Exported,
        transport: WeightTransport,
        group: Optional[GroupCoordinator],
    ) -> SnapshotManifest:
        """Stage the shards and commit one shared version after all ranks are ready."""
        spec, self._prepared_spec = self._prepared_spec, None
        group_matches = group is self._prepared_group
        if spec is not None:
            group = self._prepared_group
        self._prepared_group = None
        leader = group is None or group.rank_in_group == 0
        agree = b""
        payload = b""
        tag = group_tag = ""
        storages_meta = {}
        error, source = exported.error, {}
        if error is None:
            try:
                if spec is None:
                    raise RuntimeError("Run pre_load_check before the owner's raw load")
                if not group_matches:
                    raise RuntimeError("Snapshot publication must use the load group")
                if exported.spec is not spec:
                    raise RuntimeError(
                        "The export belongs to a different prepared load"
                    )
                group_tag = spec.digest(include_rank=False)
                tag = spec.digest()
                agree = group_tag.encode()
                payload = hashlib.sha256(exported.attestation).digest()
                if leader:
                    # Isolate concurrent publications of identical weights.
                    payload += uuid.uuid4().bytes
                # A transport may consume these storages even if it fails partway.
                storages_meta = {
                    name: StorageMeta(
                        nbytes=t.nbytes,
                        device=t.device.type,
                        device_uuid=(
                            None
                            if t.device.type == "cpu"
                            else current_platform.get_device_uuid(t.device.index)
                        ),
                        sharing="shared_ro" if name in exported.shared else "private",
                    )
                    for name, t in exported.storages.items()
                }
                source = transport.publish(storages=exported.storages)
            except Exception as e:
                error = repr(e)
        manifest = None
        commit_started = False
        try:
            ballots = vote(
                group=group, ballot=Ballot(error=error, agree=agree, payload=payload)
            )
            error = None
            try:
                assert spec is not None
                manifest = SnapshotManifest(
                    version=hashlib.sha256(
                        b"".join(peer.payload for peer in ballots)
                    ).hexdigest(),
                    spec_digest=tag,
                    storages=storages_meta,
                    load_diff=exported.load_diff,
                    attestation=exported.attestation,
                    source=source,
                )
                self._stage(tag=tag, manifest=manifest)
            except Exception as e:
                error = repr(e)
            vote(group=group, ballot=Ballot(error=error))

            assert manifest is not None
            commit_started = True
            error = None
            try:
                if leader:
                    self._announce(tag=group_tag, version=manifest.version)
            except Exception as e:
                error = repr(e)
            vote(group=group, ballot=Ballot(error=error))
        except BaseException:
            self._discard_publication(
                transport=transport,
                tag=tag,
                manifest=manifest,
                committed_tag=group_tag if commit_started else None,
            )
            raise
        return manifest

    def _discard_publication(
        self,
        *,
        transport: WeightTransport,
        tag: str,
        manifest: Optional[SnapshotManifest],
        committed_tag: Optional[str],
    ) -> None:
        if manifest is not None:
            if committed_tag is not None:
                # All ranks hide the version before removing any of its data.
                try:
                    self._retract(tag=committed_tag, version=manifest.version)
                except Exception:
                    logger.exception(
                        "Could not revoke snapshot version %s", manifest.version
                    )
                    return
            try:
                self._withdraw(tag=tag, manifest=manifest)
            except Exception:
                logger.exception(
                    "Could not remove staged snapshot %s", manifest.version
                )
        try:
            transport.close()
        except Exception:
            logger.exception(
                "Could not close transport for rejected snapshot publication"
            )

    def _preflight(
        self, *, model_config: ModelConfig, manifest: SnapshotManifest
    ) -> None:
        with _PreflightDeviceGuard():
            skeleton = self._build_skeleton(model_config)
            load_diff.preflight(skeleton=skeleton, manifest=manifest)
            constructed = validator.record_construction(skeleton)
        self._prepared_snapshot = (skeleton, constructed)

    def _load_snapshot(self, *, device_config: DeviceConfig) -> nn.Module:
        manifest, transport = self._manifest, self._transport
        prepared, self._prepared_snapshot = self._prepared_snapshot, None
        if manifest is None or transport is None or prepared is None:
            raise RuntimeError("A snapshot load requires a fresh pre_load_check")
        skeleton, constructed = prepared
        storages = transport.fetch(manifest=manifest, device=device_config.device)
        _LOADED_SNAPSHOT_MAPPINGS.append((transport, tuple(storages.values())))
        self._transport = None
        model = load_diff.install(
            skeleton=skeleton, manifest=manifest, storages=storages
        ).eval()
        validator.validate(
            model=model,
            storages=storages,
            attestation=manifest.attestation,
            constructed=constructed,
        )
        return model

    def _build_skeleton(self, model_config: ModelConfig) -> nn.Module:
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype), torch.device("meta"):
            return _initialize_model(model_config, self.load_config, quant_config)

    def _load_raw(
        self, *, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        self._constructed = None
        raw_config = dataclasses.replace(
            self.load_config,
            load_format=self.load_config.fallback_load_format,
            fallback_load_format=LoadFormat.AUTO,
        )
        loader = get_model_loader(raw_config, model_config)
        if isinstance(loader, SnapshotLoader):
            raise ValueError(f"{raw_config.load_format} is not a raw load format")
        if self._owner and self._publish_rejection is None:
            try:
                require_checkpoint_loader(loader)
            except ValueError as e:
                self._publish_rejection = str(e)
        if not self._owner or self._publish_rejection is not None:
            return loader.load_model(
                model_config=model_config, device_config=device_config
            ).eval()
        records = {}

        def record(model: nn.Module) -> None:
            records[id(model)] = (weakref.ref(model), self._record_construction(model))

        with on_model_constructed(record):
            model = loader.load_model(
                model_config=model_config, device_config=device_config
            ).eval()
        self._constructed = records.get(id(model))
        return model

    @staticmethod
    def _record_construction(model: nn.Module) -> tuple[Any, Any]:
        return (
            load_diff.record_construction(model),
            validator.record_construction(model),
        )

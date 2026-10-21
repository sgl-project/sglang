# SPDX-License-Identifier: Apache-2.0
"""Owner-side parameter mutation fence and compatible lookup generations."""

import hashlib
import threading
import uuid
from dataclasses import dataclass

import msgspec

from sglang.srt.managers.kv_control.contracts import ControlError, Identifier, validated


def weight_namespace(base, kind, value):
    try:
        body = msgspec.json.encode(("sglang.kv.weights.v1", base, kind, value))
        return "weights:" + hashlib.sha256(body).hexdigest()
    except (TypeError, UnicodeError, ValueError) as error:
        raise ControlError(
            "invalid_payload", "Weight version is not valid UTF-8 JSON"
        ) from error


@dataclass
class _Fence:
    initial_version: str | None
    requested_version: str | None = None
    failed: bool = False
    quiescent: bool = False
    ready_to_finish: bool = False
    flush_requested: bool = False
    empty_cache_requested: bool = False


class WeightLifecycle:
    def __init__(self, service, version=None):
        self.service = service
        self.version = version
        self.fence: _Fence | None = None

    @property
    def active(self):
        return self.fence is not None

    def _owner(self, *, require_keys=True):
        if threading.get_ident() != self.service.cache._kv_control_owner_thread:
            raise ControlError(
                "wrong_owner", "Weight transitions must run on the cache owner"
            )
        if require_keys and not self.service.namespace_keys:
            raise ControlError(
                "unsupported", "Weight updates require namespaced ordinary request keys"
            )

    def _validate_version(self, version, initial):
        if version is not None:
            version = validated(version, Identifier)
            weight_namespace(self.service.base_namespace, "version", version)
            if version == initial:
                raise ControlError(
                    "conflict", "Changed weights require a new weight version"
                )
        return version

    def begin(self, version=None):
        self._owner()
        if self.active:
            raise ControlError("busy", "A native weight update is already active")
        version = self._validate_version(version, self.version)
        namespace = weight_namespace(
            self.service.base_namespace, "generation", uuid.uuid4().hex
        )
        # No fence is published until all approval checks pass. No parameter
        # mutation is permitted until the old epoch/lookup scope is isolated.
        self.service.rebind_namespace(namespace)
        self.fence = _Fence(self.version, version)

    def note_version(self, version):
        self._owner()
        if self.fence is None:
            raise ControlError("busy", "Begin the native weight update first")
        version = self._validate_version(version, self.fence.initial_version)
        if version is None:
            return
        if self.fence.requested_version not in (None, version):
            raise ControlError(
                "conflict", "A weight-update session cannot mix version labels"
            )
        self.fence.requested_version = version

    def fail(self):
        if self.fence is not None:
            self.fence.failed = True

    def request_finish(self, success):
        if self.fence is None:
            raise ControlError("busy", "No native weight update is active")
        self.fence.ready_to_finish = True
        self.fence.failed |= not success

    def request_flush(self, flush, empty_cache=False):
        if self.fence is not None:
            self.fence.flush_requested |= bool(flush)
            self.fence.empty_cache_requested |= bool(empty_cache)

    def mark_quiescent(self):
        self._owner()
        if self.fence is None:
            raise ControlError("busy", "No native weight update is active")
        self.fence.quiescent = True

    def finish(self, success, *, synchronized):
        self._owner()
        fence = self.fence
        if fence is None:
            raise ControlError("busy", "No native weight update is active")
        if not synchronized:
            raise ControlError(
                "busy",
                "Synchronize parameter work before publishing the weight generation",
            )
        fence.quiescent = True
        failed = fence.failed or not success
        version = fence.requested_version if not failed else None
        if version is not None:
            namespace = weight_namespace(
                self.service.base_namespace, "version", version
            )
            self.service.rebind_namespace(namespace)
        self.version = version
        self.fence = None
        return not failed

# SPDX-License-Identifier: Apache-2.0
"""File-backed identity and discovery for local weight-cache daemons."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import re
import signal
import socket
import stat
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import msgspec
import psutil

from .protocol import CacheConfig

logger = logging.getLogger(__name__)

REGISTRY_VERSION = 1
DEFAULT_NAMESPACE = "default"
UNIX_SOCKET_PATH_MAX_BYTES = 103
SOCKET_KEY_BYTES = 20  # 160 bits; full SHA-256 stays in registry + handshake.
_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def normalize_namespace(namespace: Optional[str]) -> str:
    """Validate a namespace before it participates in paths or identities."""
    value = DEFAULT_NAMESPACE if namespace is None else namespace
    if value in (".", "..") or not _NAMESPACE_RE.fullmatch(value):
        raise ValueError(
            "weight cache namespace must be 1-64 characters, start with an "
            "alphanumeric character, and contain only letters, digits, '.', "
            "'_', or '-'"
        )
    return value


def default_runtime_dir() -> str:
    """Return a stable per-user directory shared across launch environments."""
    return f"/tmp/sglang-weight-cache-{os.getuid()}"


def new_daemon_id() -> str:
    return uuid.uuid4().hex


class CacheIdentity(msgspec.Struct, frozen=True):
    namespace: str
    device_uuid: str
    config_fingerprint: str

    @property
    def key(self) -> str:
        payload = json.dumps(
            {
                "namespace": self.namespace,
                "device_uuid": self.device_uuid,
                "config_fingerprint": self.config_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class DaemonClaim(msgspec.Struct):
    version: int
    identity: CacheIdentity
    daemon_id: str
    pid: int
    process_start_time: float
    created_at: float


class DaemonRegistration(msgspec.Struct):
    version: int
    identity: CacheIdentity
    daemon_id: str
    pid: int
    process_start_time: float
    hostname: str
    socket_path: str
    config: dict
    created_at: float


def _get_process_start_time(pid: int) -> float:
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, ValueError) as exc:
        raise RuntimeError(f"process {pid} is not alive") from exc


def _probe_process_identity(pid: int, process_start_time: float) -> Optional[bool]:
    """Return True/False for live/dead, or None when liveness is unknown."""
    try:
        process = psutil.Process(pid)
        actual = float(process.create_time())
        if process.status() == psutil.STATUS_ZOMBIE:
            return False
    except (psutil.NoSuchProcess, psutil.ZombieProcess, ValueError):
        return False
    except psutil.Error:
        # AccessDenied and transient inspection failures are not proof of death.
        return None
    return abs(actual - process_start_time) < 1e-3


def process_identity_is_alive(pid: int, process_start_time: float) -> bool:
    """Conservative public liveness check used by mapped-weight watchdogs."""
    return _probe_process_identity(pid, process_start_time) is not False


class FileWeightCacheRegistry:
    """Private per-user registry for node-local Unix-socket daemons.

    Mutations are serialized with ``flock``. Registration publication uses an
    atomic replace, so readers see either the old complete record or the new
    complete record, never a partially written JSON document.
    """

    def __init__(self, runtime_dir: Optional[str], *, namespace: Optional[str]):
        self.runtime_dir = os.path.abspath(
            os.path.expanduser(runtime_dir or default_runtime_dir())
        )
        self.namespace = normalize_namespace(namespace)
        self.claims_dir = os.path.join(self.runtime_dir, "claims")
        self.registrations_dir = os.path.join(self.runtime_dir, "registrations")
        self.sockets_dir = os.path.join(self.runtime_dir, "sockets")
        self._lock_path = os.path.join(self.runtime_dir, ".registry.lock")
        for path in (
            self.runtime_dir,
            self.claims_dir,
            self.registrations_dir,
            self.sockets_dir,
        ):
            self._ensure_private_directory(path)

    @staticmethod
    def _ensure_private_directory(path: str) -> None:
        os.makedirs(path, mode=0o700, exist_ok=True)
        info = os.lstat(path)
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"weight cache runtime path is not a directory: {path}")
        if info.st_uid != os.getuid():
            raise RuntimeError(
                f"weight cache runtime directory is not owned by uid {os.getuid()}: "
                f"{path}"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(
                f"weight cache runtime directory must be private (mode 0700): {path}"
            )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def identity_for(self, config: CacheConfig, device_uuid: str) -> CacheIdentity:
        device_uuid = str(device_uuid).strip()
        if not device_uuid:
            raise ValueError("physical device UUID must not be empty")
        return CacheIdentity(
            namespace=self.namespace,
            device_uuid=device_uuid,
            config_fingerprint=config.fingerprint(),
        )

    def claim_path(self, identity: CacheIdentity) -> str:
        return os.path.join(self.claims_dir, f"{identity.key}.json")

    def registration_path(self, identity: CacheIdentity) -> str:
        return os.path.join(self.registrations_dir, f"{identity.key}.json")

    def socket_path(self, identity: CacheIdentity) -> str:
        socket_key = base64.urlsafe_b64encode(
            bytes.fromhex(identity.key)[:SOCKET_KEY_BYTES]
        ).decode("ascii").rstrip("=")
        path = os.path.join(self.sockets_dir, f"{socket_key}.sock")
        path_bytes = len(os.fsencode(path))
        if path_bytes > UNIX_SOCKET_PATH_MAX_BYTES:
            raise ValueError(
                f"weight cache socket path is {path_bytes} bytes, exceeding the "
                f"portable {UNIX_SOCKET_PATH_MAX_BYTES}-byte Unix-socket limit: "
                f"choose a shorter --weight-cache-runtime-dir"
            )
        return path

    @staticmethod
    def _atomic_write(path: str, value: object) -> None:
        directory = os.path.dirname(path)
        fd, temporary_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
        try:
            os.fchmod(fd, 0o600)
            payload = msgspec.json.encode(value)
            with os.fdopen(fd, "wb", closefd=True) as file:
                fd = -1
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def _read(path: str, value_type):
        fd = -1
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                return None
            with os.fdopen(fd, "rb", closefd=True) as file:
                fd = -1
                return msgspec.json.decode(file.read(), type=value_type)
        except (
            FileNotFoundError,
            OSError,
            msgspec.DecodeError,
            msgspec.ValidationError,
        ):
            return None
        finally:
            if fd >= 0:
                os.close(fd)

    @staticmethod
    def _record_matches_identity(path, record, identity: CacheIdentity) -> bool:
        return (
            record.version == REGISTRY_VERSION
            and record.identity == identity
            and Path(path).stem == record.identity.key
        )

    def claim(
        self,
        identity: CacheIdentity,
        *,
        pid: int,
        daemon_id: str,
        process_start_time: Optional[float] = None,
        force: bool = False,
    ) -> None:
        start_time = (
            _get_process_start_time(pid)
            if process_start_time is None
            else process_start_time
        )
        claim_path = self.claim_path(identity)
        with self._locked():
            existing_claim = self._read(claim_path, DaemonClaim)
            existing_registration = self._read(
                self.registration_path(identity), DaemonRegistration
            )
            for record_path, record in (
                (claim_path, existing_claim),
                (self.registration_path(identity), existing_registration),
            ):
                if record is None and os.path.lexists(record_path):
                    raise RuntimeError(
                        "weight cache identity has a malformed, unowned, or "
                        f"non-private registry record at {record_path}; refusing "
                        "automatic takeover"
                    )
                if record is not None and not self._record_matches_identity(
                    record_path, record, identity
                ):
                    raise RuntimeError(
                        "weight cache registry record does not match its full-key "
                        f"path or requested identity: {record_path}; refusing "
                        "automatic takeover"
                    )
            owner_records = [
                record
                for record in (existing_claim, existing_registration)
                if record is not None
            ]

            if owner_records and all(
                owner.daemon_id == daemon_id
                and owner.pid == pid
                and abs(owner.process_start_time - start_time) < 1e-3
                for owner in owner_records
            ):
                return

            for owner in owner_records:
                if owner.daemon_id == daemon_id:
                    if (
                        owner.pid == pid
                        and abs(owner.process_start_time - start_time) < 1e-3
                    ):
                        continue
                    raise RuntimeError(
                        "daemon_id is already associated with a different process"
                    )

                liveness = _probe_process_identity(owner.pid, owner.process_start_time)
                if liveness is None:
                    raise RuntimeError(
                        "cannot prove whether the existing weight cache daemon "
                        f"is dead (pid={owner.pid}); refusing to reclaim identity"
                    )
                if liveness:
                    if not force:
                        raise RuntimeError(
                            "a weight cache daemon already owns identity "
                            f"{identity.key} (pid={owner.pid})"
                        )
                    self._kill_and_wait(owner.pid, owner.process_start_time)

            self._reject_conflicting_owners_locked(identity)
            existing_ids = {owner.daemon_id for owner in owner_records}
            self._refuse_live_unregistered_socket_locked(identity)
            self._remove_identity_files_locked(
                identity,
                expected_daemon_id=(
                    next(iter(existing_ids)) if len(existing_ids) == 1 else None
                ),
            )
            self._atomic_write(
                claim_path,
                DaemonClaim(
                    version=REGISTRY_VERSION,
                    identity=identity,
                    daemon_id=daemon_id,
                    pid=pid,
                    process_start_time=start_time,
                    created_at=time.time(),
                ),
            )

    @staticmethod
    def _kill_and_wait(
        pid: int, process_start_time: float, *, timeout: float = 5.0
    ) -> None:
        logger.warning("[weight_cache] force takeover: killing daemon pid=%s", pid)
        liveness = _probe_process_identity(pid, process_start_time)
        if liveness is False:
            return
        if liveness is None:
            raise RuntimeError(
                f"cannot confirm weight cache daemon identity for pid={pid}"
            )
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            liveness = _probe_process_identity(pid, process_start_time)
            if liveness is False:
                return
            if liveness is None:
                raise RuntimeError(
                    f"cannot confirm that weight cache daemon pid={pid} exited"
                )
            time.sleep(0.05)
        raise RuntimeError(
            f"weight cache daemon pid={pid} did not exit within {timeout}s"
        )

    def _refuse_live_unregistered_socket_locked(self, identity: CacheIdentity) -> None:
        """Probe an unexplained socket before stale cleanup can unlink it."""
        socket_path = self.socket_path(identity)
        try:
            info = os.lstat(socket_path)
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(
                f"refusing to replace unowned or non-socket path: {socket_path}"
            )

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.25)
            probe.connect(socket_path)
        except (ConnectionRefusedError, FileNotFoundError):
            return
        except OSError as exc:
            raise RuntimeError(
                f"cannot prove whether existing weight cache socket is stale: "
                f"{socket_path}: {exc}"
            ) from exc
        else:
            raise RuntimeError(
                f"a live unregistered service is listening at {socket_path}; "
                "refusing to unlink it"
            )
        finally:
            probe.close()

    def publish(
        self,
        identity: CacheIdentity,
        *,
        config: CacheConfig,
        socket_path: str,
        pid: int,
        daemon_id: str,
    ) -> DaemonRegistration:
        with self._locked():
            claim = self._read(self.claim_path(identity), DaemonClaim)
            if claim is None or claim.daemon_id != daemon_id or claim.pid != pid:
                raise RuntimeError(
                    "weight cache daemon no longer owns its registry claim"
                )
            if _probe_process_identity(pid, claim.process_start_time) is not True:
                raise RuntimeError("weight cache daemon process identity is not alive")
            if identity != self.identity_for(config, identity.device_uuid):
                raise RuntimeError("registration identity does not match CacheConfig")
            if socket_path != self.socket_path(identity):
                raise RuntimeError("registration socket path does not match identity")
            socket_info = os.lstat(socket_path)
            if (
                not stat.S_ISSOCK(socket_info.st_mode)
                or socket_info.st_uid != os.getuid()
            ):
                raise RuntimeError(
                    f"refusing to publish unowned or non-socket path: {socket_path}"
                )
            registration = DaemonRegistration(
                version=REGISTRY_VERSION,
                identity=identity,
                daemon_id=daemon_id,
                pid=pid,
                process_start_time=claim.process_start_time,
                hostname=socket.gethostname(),
                socket_path=socket_path,
                config=config.to_dict(),
                created_at=time.time(),
            )
            self._atomic_write(self.registration_path(identity), registration)
            return registration

    def _validate_registration(
        self, registration: DaemonRegistration, identity: CacheIdentity
    ) -> bool:
        if (
            registration.version != REGISTRY_VERSION
            or registration.identity != identity
            or registration.hostname != socket.gethostname()
            or _probe_process_identity(
                registration.pid, registration.process_start_time
            )
            is not True
        ):
            return False
        try:
            registered_config = CacheConfig.from_dict(registration.config)
            info = os.lstat(registration.socket_path)
        except (
            FileNotFoundError,
            OSError,
            TypeError,
            ValueError,
            msgspec.ValidationError,
        ):
            return False
        return (
            registered_config.fingerprint() == identity.config_fingerprint
            and stat.S_ISSOCK(info.st_mode)
            and info.st_uid == os.getuid()
            and registration.socket_path == self.socket_path(identity)
        )

    def discover(
        self, config: CacheConfig, *, device_uuid: str
    ) -> Optional[DaemonRegistration]:
        identity = self.identity_for(config, device_uuid)
        with self._locked():
            registration_path = self.registration_path(identity)
            registration = self._read(registration_path, DaemonRegistration)
            if registration is None and os.path.lexists(registration_path):
                raise RuntimeError(
                    "weight cache discovery found a malformed, unowned, or "
                    f"non-private exact registration at {registration_path}; "
                    "refusing silent fallback"
                )
            if registration is not None and not self._record_matches_identity(
                registration_path, registration, identity
            ):
                raise RuntimeError(
                    "weight cache discovery found an exact registration whose "
                    "version, embedded identity, or full-key path is invalid; "
                    "refusing silent fallback"
                )
            if registration is None:
                exact_claim_path = self.claim_path(identity)
                exact_claim = self._read(exact_claim_path, DaemonClaim)
                if exact_claim is None and os.path.lexists(exact_claim_path):
                    raise RuntimeError(
                        "weight cache discovery found a malformed, unowned, or "
                        f"non-private exact claim at {exact_claim_path}; refusing "
                        "silent fallback"
                    )
                if exact_claim is not None:
                    if not self._record_matches_identity(
                        exact_claim_path, exact_claim, identity
                    ):
                        raise RuntimeError(
                            "weight cache discovery found an exact claim whose "
                            "version, embedded identity, or full-key path is "
                            "invalid; refusing silent fallback"
                        )
                    liveness = _probe_process_identity(
                        exact_claim.pid, exact_claim.process_start_time
                    )
                    if liveness is not False:
                        state = "live" if liveness else "indeterminate"
                        raise RuntimeError(
                            f"a {state} weight cache daemon owns the requested "
                            "identity and is still loading; refusing disk fallback "
                            "while it may be allocating weights on this GPU"
                        )
                    self._remove_identity_files_locked(
                        identity, expected_daemon_id=exact_claim.daemon_id
                    )

                self._reject_conflicting_owners_locked(identity)
                return None
            if self._validate_registration(registration, identity):
                return registration
            liveness = _probe_process_identity(
                registration.pid, registration.process_start_time
            )
            if liveness is False:
                claim_path = self.claim_path(identity)
                claim = self._read(claim_path, DaemonClaim)
                if claim is None and os.path.lexists(claim_path):
                    raise RuntimeError(
                        "weight cache discovery found a malformed, unowned, or "
                        "non-private claim beside a dead registration; refusing "
                        "automatic cleanup"
                    )
                if claim is not None:
                    if (
                        not self._record_matches_identity(claim_path, claim, identity)
                        or claim.daemon_id != registration.daemon_id
                        or claim.pid != registration.pid
                        or abs(
                            claim.process_start_time
                            - registration.process_start_time
                        )
                        >= 1e-3
                    ):
                        raise RuntimeError(
                            "weight cache claim and dead registration disagree on "
                            "owner identity; refusing automatic cleanup"
                        )
                    if _probe_process_identity(
                        claim.pid, claim.process_start_time
                    ) is not False:
                        raise RuntimeError(
                            "weight cache claim may still be live even though its "
                            "registration owner appears dead; refusing fallback"
                        )
                self._remove_identity_files_locked(
                    identity, expected_daemon_id=registration.daemon_id
                )
                self._reject_conflicting_owners_locked(identity)
                return None
            raise RuntimeError(
                "weight cache registry contains an invalid record for a live or "
                "indeterminate daemon; refusing silent fallback"
            )

    def _reject_conflicting_owners_locked(self, identity: CacheIdentity) -> None:
        """Fail if another config may already own this namespace/GPU."""
        for directory, value_type in (
            (self.claims_dir, DaemonClaim),
            (self.registrations_dir, DaemonRegistration),
        ):
            for path in Path(directory).glob("*.json"):
                candidate = self._read(str(path), value_type)
                if candidate is None or candidate.identity == identity:
                    continue
                if (
                    candidate.identity.namespace != identity.namespace
                    or candidate.identity.device_uuid != identity.device_uuid
                ):
                    continue
                if not self._record_matches_identity(
                    path, candidate, candidate.identity
                ):
                    raise RuntimeError(
                        "weight cache discovery found an incompatible owner with "
                        "an invalid version, embedded identity, or full-key path; "
                        "refusing disk fallback"
                    )

                liveness = _probe_process_identity(
                    candidate.pid, candidate.process_start_time
                )
                if liveness is False:
                    # Do not delete through the shared identity here: a stale
                    # claim and a live registration can be inconsistent, and
                    # daemon_id alone is not enough proof that both are dead.
                    continue

                if isinstance(candidate, DaemonRegistration):
                    owner_state = (
                        "ready"
                        if self._validate_registration(candidate, candidate.identity)
                        else "invalid but live or indeterminate"
                    )
                else:
                    owner_state = "still loading"
                raise RuntimeError(
                    "a weight cache daemon already occupies physical GPU "
                    f"{identity.device_uuid} in namespace {identity.namespace} "
                    f"with fingerprint {candidate.identity.config_fingerprint} "
                    f"({owner_state}), not {identity.config_fingerprint}; refusing "
                    "disk fallback on a GPU that may already hold cached weights"
                )

    def find_registration(
        self, *, daemon_id: str, pid: Optional[int] = None
    ) -> Optional[DaemonRegistration]:
        """Find one exact ready daemon spawned by a parent process."""
        with self._locked():
            for path in Path(self.registrations_dir).glob("*.json"):
                registration = self._read(str(path), DaemonRegistration)
                if registration is None or registration.daemon_id != daemon_id:
                    continue
                if pid is not None and registration.pid != pid:
                    continue
                if self._validate_registration(registration, registration.identity):
                    return registration
        return None

    def list_registrations(self) -> list[DaemonRegistration]:
        """Return all valid, ready daemons in this namespace on this node."""
        registrations = []
        with self._locked():
            for path in Path(self.registrations_dir).glob("*.json"):
                registration = self._read(str(path), DaemonRegistration)
                if registration is None:
                    continue
                if registration.identity.namespace != self.namespace:
                    continue
                if self._validate_registration(registration, registration.identity):
                    registrations.append(registration)
        return registrations

    def release(self, identity: CacheIdentity, *, daemon_id: str) -> None:
        with self._locked():
            self._remove_identity_files_locked(identity, expected_daemon_id=daemon_id)

    def unpublish(self, identity: CacheIdentity, *, daemon_id: str) -> None:
        """Stop advertising readiness while retaining the daemon's claim."""
        with self._locked():
            registration_path = self.registration_path(identity)
            registration = self._read(registration_path, DaemonRegistration)
            if registration is None:
                if os.path.lexists(registration_path):
                    raise RuntimeError(
                        "weight cache registration became malformed during shutdown"
                    )
                return
            if registration.daemon_id != daemon_id:
                return
            os.unlink(registration_path)

            socket_path = self.socket_path(identity)
            try:
                info = os.lstat(socket_path)
            except FileNotFoundError:
                return
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise RuntimeError(
                    f"refusing to unlink unowned or non-socket path: {socket_path}"
                )
            os.unlink(socket_path)

    def _remove_identity_files_locked(
        self, identity: CacheIdentity, *, expected_daemon_id: Optional[str]
    ) -> None:
        claim_path = self.claim_path(identity)
        registration_path = self.registration_path(identity)
        claim = self._read(claim_path, DaemonClaim)
        registration = self._read(registration_path, DaemonRegistration)

        if expected_daemon_id is not None:
            if claim is not None and claim.daemon_id != expected_daemon_id:
                return
            if (
                registration is not None
                and registration.daemon_id != expected_daemon_id
            ):
                return

        # A release with no matching records has no proof that it owns the
        # shared socket slot. The claim path probes stale sockets before it
        # calls this helper with expected_daemon_id=None.
        if (
            claim is None
            and registration is None
            and expected_daemon_id is not None
        ):
            return

        for path in (registration_path, claim_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        socket_path = self.socket_path(identity)
        try:
            info = os.lstat(socket_path)
            if stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid():
                os.unlink(socket_path)
        except FileNotFoundError:
            pass

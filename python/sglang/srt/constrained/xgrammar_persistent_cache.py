"""Process-safe persistent cache for compatible XGrammar compiled grammars.

The cache is intentionally fail-closed. An invalid serialized entry or
adaptive local-compile marker is surfaced as an error instead of being hidden
by a recovery path. That makes image/tokenizer/cache incompatibilities
observable immediately.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import struct
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from xgrammar import CompiledGrammar, TokenizerInfo

type CacheSource = Literal["compile", "disk", "local_compile"]


@dataclass(frozen=True, slots=True)
class CompiledGrammarLookup:
    grammar: CompiledGrammar
    source: CacheSource
    lock_wait_seconds: float
    resolution_seconds: float
    phase_seconds: dict[str, float]


@dataclass(frozen=True, slots=True)
class _LocalCompileMarker:
    native_compile_nanoseconds: int
    serialized_bytes: int


class PersistentXGrammarCache:
    """A tokenizer-scoped, adaptive cross-process compiled-grammar cache."""

    _COUNTER_FORMAT = "<Q"
    _COUNTER_BYTES = struct.calcsize(_COUNTER_FORMAT)
    _LOCAL_COMPILE_MAGIC = b"SXGLOCAL1"
    _LOCAL_COMPILE_BODY_FORMAT = "<9sQQ"
    _LOCAL_COMPILE_BODY_BYTES = struct.calcsize(_LOCAL_COMPILE_BODY_FORMAT)
    _LOCAL_COMPILE_CHECKSUM_BYTES = hashlib.sha256().digest_size
    _LOCAL_COMPILE_MARKER_BYTES = (
        _LOCAL_COMPILE_BODY_BYTES + _LOCAL_COMPILE_CHECKSUM_BYTES
    )

    def __init__(
        self,
        *,
        tokenizer_info: TokenizerInfo,
        cache_directory: str,
        max_bytes: int,
        compiler_identity: dict[str, str | int | bool | list[int]],
        deserialize_bytes_per_second: int | None = None,
        local_compile_speedup: int | None = None,
    ) -> None:
        if deserialize_bytes_per_second is None:
            deserialize_bytes_per_second = int(
                os.environ.get(
                    "SGLANG_XGRAMMAR_DESERIALIZE_BYTES_PER_SECOND",
                    str(128 * 1024**2),
                )
            )
        if local_compile_speedup is None:
            local_compile_speedup = int(
                os.environ.get(
                    "SGLANG_XGRAMMAR_LOCAL_COMPILE_SPEEDUP",
                    "2",
                )
            )
        if (
            max_bytes < self._LOCAL_COMPILE_MARKER_BYTES
            or deserialize_bytes_per_second <= 0
            or local_compile_speedup <= 0
        ):
            raise ValueError(
                "persistent XGrammar cache must fit one local-compile marker, "
                "and deserialize throughput/local-compile speedup must be positive"
            )

        xgrammar_version = importlib.metadata.version("xgrammar")
        tokenizer_digest = hashlib.sha256(
            tokenizer_info.serialize_json().encode("utf-8")
        ).hexdigest()
        namespace_payload = {
            "format": 2,
            "xgrammar_version": xgrammar_version,
            "tokenizer_sha256": tokenizer_digest,
            "compiler": compiler_identity,
            "max_bytes": max_bytes,
            "deserialize_bytes_per_second": deserialize_bytes_per_second,
            "local_compile_speedup": local_compile_speedup,
        }
        namespace = hashlib.sha256(
            json.dumps(
                namespace_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        self.max_bytes = max_bytes
        self.deserialize_bytes_per_second = deserialize_bytes_per_second
        self.local_compile_speedup = local_compile_speedup
        self.root = Path(cache_directory).expanduser().resolve() / namespace
        self.entries = self.root / "entries"
        self.locks = self.root / "locks"
        self.entries.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.locks.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._tokenizer_info = tokenizer_info
        self._size_ledger = self.root / "size-bytes-v2"
        self._accounting_session = self.root / "accounting-session-v2"
        self._prune_lock = self.root / "prune.lock"
        self._initialize_size_ledger()

    def get_or_compile(
        self,
        *,
        key_type: str,
        key_string: str,
        compile_fn: Callable[[], CompiledGrammar],
    ) -> CompiledGrammarLookup:
        key_digest = hashlib.sha256(
            json.dumps(
                [key_type, key_string],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        entry_path = self.entries / f"{key_digest}.json"
        lock_path = self.locks / f"{key_digest}.lock"

        lock_started = time.perf_counter()
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            lock_wait_seconds = time.perf_counter() - lock_started
            resolution_started = time.perf_counter()

            if entry_path.is_file():
                entry_read_started = time.perf_counter()
                try:
                    entry_payload = entry_path.read_bytes()
                    local_compile_marker = self._decode_local_compile_marker(
                        entry_payload
                    )
                    if (
                        local_compile_marker is not None
                        and not self._prefer_local_compile(
                            native_compile_nanoseconds=(
                                local_compile_marker.native_compile_nanoseconds
                            ),
                            serialized_bytes=local_compile_marker.serialized_bytes,
                        )
                    ):
                        raise ValueError(
                            "adaptive XGrammar local-compile marker contradicts "
                            "the active cache policy"
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"persistent XGrammar cache entry is invalid: {entry_path}"
                    ) from exc
                entry_read_seconds = time.perf_counter() - entry_read_started

                if local_compile_marker is not None:
                    compile_started_ns = time.perf_counter_ns()
                    grammar = compile_fn()
                    compile_nanoseconds = time.perf_counter_ns() - compile_started_ns
                    os.utime(entry_path, None)
                    resolved_at = time.perf_counter()
                    return CompiledGrammarLookup(
                        grammar=grammar,
                        source="local_compile",
                        lock_wait_seconds=lock_wait_seconds,
                        resolution_seconds=resolved_at - resolution_started,
                        phase_seconds={
                            "adaptive_marker_read": entry_read_seconds,
                            "native_compile": compile_nanoseconds / 1_000_000_000,
                        },
                    )

                disk_started = time.perf_counter()
                try:
                    serialized = entry_payload.decode("utf-8")
                    grammar = CompiledGrammar.deserialize_json(
                        serialized,
                        self._tokenizer_info,
                    )
                    os.utime(entry_path, None)
                except Exception as exc:
                    raise RuntimeError(
                        f"persistent XGrammar cache entry is invalid: {entry_path}"
                    ) from exc
                resolved_at = time.perf_counter()
                return CompiledGrammarLookup(
                    grammar=grammar,
                    source="disk",
                    lock_wait_seconds=lock_wait_seconds,
                    resolution_seconds=resolved_at - resolution_started,
                    phase_seconds={
                        "disk_read": entry_read_seconds,
                        "disk_deserialize": resolved_at - disk_started,
                    },
                )

            compile_started_ns = time.perf_counter_ns()
            grammar = compile_fn()
            compile_nanoseconds = time.perf_counter_ns() - compile_started_ns
            compile_seconds = compile_nanoseconds / 1_000_000_000
            serialize_started = time.perf_counter()
            serialized = grammar.serialize_json()
            serialized_payload = serialized.encode("utf-8")
            serialize_seconds = time.perf_counter() - serialize_started
            serialized_bytes = len(serialized_payload)
            prefer_local_compile = self._prefer_local_compile(
                native_compile_nanoseconds=compile_nanoseconds,
                serialized_bytes=serialized_bytes,
            )
            if prefer_local_compile:
                stored_payload = self._encode_local_compile_marker(
                    native_compile_nanoseconds=compile_nanoseconds,
                    serialized_bytes=serialized_bytes,
                )
                cache_policy_phase = "policy_local_compile"
                lookup_source: CacheSource = "local_compile"
            else:
                stored_payload = serialized_payload
                cache_policy_phase = "policy_serialized"
                lookup_source = "compile"
            write_started = time.perf_counter()
            self._write_atomic_bytes(entry_path, stored_payload)
            write_seconds = time.perf_counter() - write_started
            accounting_started = time.perf_counter()
            try:
                self._account_and_prune(
                    protected=entry_path,
                    added_bytes=len(stored_payload),
                )
            except BaseException:
                # The cache is reconstructible. Do not leave an unaccounted
                # entry behind when quota accounting fails.
                entry_path.unlink(missing_ok=True)
                raise
            accounting_seconds = time.perf_counter() - accounting_started
            resolution_seconds = time.perf_counter() - resolution_started
        return CompiledGrammarLookup(
            grammar=grammar,
            source=lookup_source,
            lock_wait_seconds=lock_wait_seconds,
            resolution_seconds=resolution_seconds,
            phase_seconds={
                "native_compile": compile_seconds,
                "serialize": serialize_seconds,
                "entry_write": write_seconds,
                "account_prune": accounting_seconds,
                cache_policy_phase: 0.0,
            },
        )

    def _prefer_local_compile(
        self,
        *,
        native_compile_nanoseconds: int,
        serialized_bytes: int,
    ) -> bool:
        return (
            serialized_bytes > self.max_bytes
            or native_compile_nanoseconds
            * self.local_compile_speedup
            * self.deserialize_bytes_per_second
            < serialized_bytes * 1_000_000_000
        )

    @classmethod
    def _encode_local_compile_marker(
        cls,
        *,
        native_compile_nanoseconds: int,
        serialized_bytes: int,
    ) -> bytes:
        if native_compile_nanoseconds < 0 or serialized_bytes <= 0:
            raise RuntimeError(
                "adaptive XGrammar local-compile marker values are invalid"
            )
        body = struct.pack(
            cls._LOCAL_COMPILE_BODY_FORMAT,
            cls._LOCAL_COMPILE_MAGIC,
            native_compile_nanoseconds,
            serialized_bytes,
        )
        return body + hashlib.sha256(body).digest()

    @classmethod
    def _decode_local_compile_marker(
        cls,
        payload: bytes,
    ) -> _LocalCompileMarker | None:
        if not payload.startswith(cls._LOCAL_COMPILE_MAGIC):
            return None
        if len(payload) != cls._LOCAL_COMPILE_MARKER_BYTES:
            raise ValueError(
                "adaptive XGrammar local-compile marker has the wrong size"
            )
        body = payload[: cls._LOCAL_COMPILE_BODY_BYTES]
        checksum = payload[cls._LOCAL_COMPILE_BODY_BYTES :]
        if hashlib.sha256(body).digest() != checksum:
            raise ValueError(
                "adaptive XGrammar local-compile marker checksum is invalid"
            )
        magic, native_compile_nanoseconds, serialized_bytes = struct.unpack(
            cls._LOCAL_COMPILE_BODY_FORMAT,
            body,
        )
        if (
            magic != cls._LOCAL_COMPILE_MAGIC
            or native_compile_nanoseconds < 0
            or serialized_bytes <= 0
        ):
            raise ValueError(
                "adaptive XGrammar local-compile marker values are invalid"
            )
        return _LocalCompileMarker(
            native_compile_nanoseconds=native_compile_nanoseconds,
            serialized_bytes=serialized_bytes,
        )

    @staticmethod
    def _write_atomic_bytes(path: Path, payload: bytes) -> None:
        # Cache data is reconstructible. Close + replace gives readers
        # process-level atomicity, while the per-launch session scan reconciles
        # accounting after a crash. Per-entry fsync (without a directory fsync)
        # did not provide complete power-loss durability and reduced burst
        # compile throughput by more than an order of magnitude.
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=".xgrammar-",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                os.chmod(temp_path, 0o600)
                if temp_file.write(payload) != len(payload):
                    raise RuntimeError(
                        f"short write while creating XGrammar cache entry: {path}"
                    )
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @classmethod
    def _write_atomic_text(cls, path: Path, text: str) -> None:
        cls._write_atomic_bytes(path, text.encode("utf-8"))

    def _initialize_size_ledger(self) -> None:
        with self._prune_lock.open("a+b") as prune_lock:
            fcntl.flock(prune_lock.fileno(), fcntl.LOCK_EX)
            session_id = os.environ.get("SGLANG_XGRAMMAR_CACHE_SESSION_ID")
            if (
                session_id
                and self._accounting_session.is_file()
                and self._accounting_session.read_text(encoding="ascii") == session_id
            ):
                if self._size_ledger.is_file():
                    self._read_size_ledger()
                    return

            total_bytes, entries = self._scan_entries()
            total_bytes = self._prune_entries(
                total_bytes=total_bytes,
                entries=entries,
                protected=None,
            )
            self._reset_size_ledger(total_bytes)
            if session_id:
                try:
                    session_id.encode("ascii")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "SGLANG_XGRAMMAR_CACHE_SESSION_ID must be ASCII"
                    ) from exc
                self._write_atomic_text(self._accounting_session, session_id)

    def _account_and_prune(
        self,
        *,
        protected: Path,
        added_bytes: int,
    ) -> None:
        with self._prune_lock.open("a+b") as prune_lock:
            fcntl.flock(prune_lock.fileno(), fcntl.LOCK_EX)
            total_bytes = self._read_size_ledger() + added_bytes

            if total_bytes <= self.max_bytes:
                self._update_size_ledger(total_bytes)
                return
            total_bytes, entries = self._scan_entries()
            total_bytes = self._prune_entries(
                total_bytes=total_bytes,
                entries=entries,
                protected=protected,
            )
            if total_bytes > self.max_bytes:
                raise RuntimeError(
                    "persistent XGrammar cache could not prune below its hard "
                    f"limit: {total_bytes} > {self.max_bytes} bytes"
                )
            self._update_size_ledger(total_bytes)

    def _read_size_ledger(self) -> int:
        try:
            raw = self._size_ledger.read_bytes()
            if len(raw) != self._COUNTER_BYTES:
                raise ValueError(
                    f"expected {self._COUNTER_BYTES} bytes, got {len(raw)}"
                )
            (value,) = struct.unpack(self._COUNTER_FORMAT, raw)
        except Exception as exc:
            raise RuntimeError(
                f"persistent XGrammar size ledger is invalid: {self._size_ledger}"
            ) from exc
        return value

    def _reset_size_ledger(self, value: int) -> None:
        payload = self._counter_payload(value)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self._size_ledger.parent,
                prefix=".xgrammar-counter-",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                os.chmod(temp_path, 0o600)
                if temp_file.write(payload) != len(payload):
                    raise RuntimeError("short write while initializing size ledger")
            os.replace(temp_path, self._size_ledger)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def _update_size_ledger(self, value: int) -> None:
        payload = self._counter_payload(value)
        try:
            with self._size_ledger.open("r+b", buffering=0) as ledger:
                written = os.pwrite(ledger.fileno(), payload, 0)
        except Exception as exc:
            raise RuntimeError(
                f"failed to update persistent XGrammar size ledger: {self._size_ledger}"
            ) from exc
        if written != len(payload):
            raise RuntimeError(
                "short write while updating persistent XGrammar size ledger: "
                f"{written} != {len(payload)}"
            )

    def _counter_payload(self, value: int) -> bytes:
        if value < 0 or value >= 1 << (8 * self._COUNTER_BYTES):
            raise RuntimeError(
                f"persistent XGrammar size ledger value is out of range: {value}"
            )
        return struct.pack(self._COUNTER_FORMAT, value)

    def _prune_entries(
        self,
        *,
        total_bytes: int,
        entries: list[tuple[int, int, Path]],
        protected: Path | None,
    ) -> int:
        for _, size, path in sorted(entries):
            if total_bytes <= self.max_bytes:
                break
            if path == protected:
                continue
            if self._unlink_if_idle(path):
                total_bytes -= size
        return total_bytes

    def _scan_entries(self) -> tuple[int, list[tuple[int, int, Path]]]:
        entries: list[tuple[int, int, Path]] = []
        total_bytes = 0
        for item in self.entries.iterdir():
            if not item.is_file() or item.name.startswith(".xgrammar-"):
                continue
            stat = item.stat()
            total_bytes += stat.st_size
            entries.append((stat.st_mtime_ns, stat.st_size, item))
        return total_bytes, entries

    def _unlink_if_idle(self, entry: Path) -> bool:
        lock_path = self.locks / f"{entry.stem}.lock"
        with lock_path.open("a+b") as entry_lock:
            try:
                fcntl.flock(
                    entry_lock.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except BlockingIOError:
                return False
            if not entry.exists():
                return False
            entry.unlink()
            return True

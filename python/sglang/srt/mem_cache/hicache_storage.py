import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool_host import HostKVCache

logger = logging.getLogger(__name__)

# Bump when MetadataEntry/Snapshot structure changes
METADATA_SCHEMA_VERSION = 1


def get_hash_str(token_ids: List[int], prior_hash: str = None) -> str:
    hasher = hashlib.sha256()

    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))

    for t in token_ids:
        if isinstance(t, tuple):
            # EAGLE bigram mode: hash both elements to uniquely identify the bigram
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            # Regular mode: single integer token
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))

    return hasher.hexdigest()


def hash_str_to_int64(hash_str: str) -> int:
    """Convert SHA256 hex string to signed 64-bit integer for events.

    Takes first 16 hex characters (64 bits) and converts to signed int64 range.
    """
    # Take first 16 hex chars to get 64-bit value
    uint64_val = int(hash_str[:16], 16)
    # Convert to signed int64 range [-2^63, 2^63-1]
    if uint64_val >= 2**63:
        return uint64_val - 2**64
    return uint64_val


@dataclass
class MetadataEntry:
    """Single KV cache page metadata, the minimal unit of a snapshot."""

    hash_key: str  # SHA256 hash of this page (the L3 storage key)
    parent_hash: Optional[str]  # parent page hash (prior_hash in chain), None for root
    token_ids: list  # token ids in this page (len = page_size)
    # element type: int; EAGLE bigram mode: tuple[int, int]
    schema_version: int = METADATA_SCHEMA_VERSION
    priority: int = 0
    extra_key: Optional[str] = None  # tenant / namespace
    ts_unix_ms: int = 0


@dataclass
class MetadataSnapshot:
    """Metadata snapshot of valid KV cache pages in L3 at a point in time."""

    schema_version: int  # METADATA_SCHEMA_VERSION at write time
    version: str  # snapshot version id (timestamp or UUID)
    scope: Dict[str, str]  # model/tp/pp/page_size identifiers
    entries: List[MetadataEntry]


@dataclass
class StorageMetadataQuery:
    scope: Optional[Dict[str, str]] = None
    since_version: Optional[str] = None  # incremental: only entries after this version


@dataclass
class HiCacheStorageConfig:
    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    is_mla_model: bool
    enable_storage_metrics: bool
    is_page_first_layout: bool
    model_name: Optional[str]
    tp_lcm_size: Optional[int] = None
    should_split_heads: bool = False
    extra_config: Optional[dict] = None


@dataclass
class HiCacheStorageExtraInfo:
    prefix_keys: Optional[List[str]] = (None,)
    extra_info: Optional[dict] = None


class HiCacheStorage(ABC):
    """
    HiCacheStorage is a class that provides a generic key-value interface for storing and retrieving KV cache.
    It abstracts the underlying storage mechanism, allowing different implementations to be used.
    """

    # todo, the page size of storage backend does not have to be the same as the same as host memory pool

    def register_mem_pool_host(self, mem_pool_host: HostKVCache):
        self.mem_pool_host = mem_pool_host

    def batch_get_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Retrieve values for multiple keys.
        Returns a list of booleans indicating success for each key.
        """
        pass

    def batch_set_v1(
        self,
        keys: List[str],
        host_indices: torch.Tensor,
        extra_info: Optional[HiCacheStorageExtraInfo] = None,
    ) -> List[bool]:
        """
        Store multiple key-value pairs.
        Returns a list of booleans indicating success for each key.
        """
        pass

    @abstractmethod
    def get(
        self,
        key: str,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        """
        Retrieve the value associated with the given key.
        Returns None if the key does not exist.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None] | int:
        """
        Retrieve values for multiple keys.
        Returns a list of tensors or None for each key.
        """
        pass

    @abstractmethod
    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store the value associated with the given key.
        Returns True if the operation was successful, False otherwise.
        """
        pass

    # TODO: Deprecate
    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.
        Returns True if all operations were successful, False otherwise.
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if the key exists in the storage.
        Returns True if the key exists, False otherwise.
        """
        pass

    # TODO: Use a finer-grained return type (e.g., List[bool])
    def batch_exists(
        self, keys: List[str], extra_info: Optional[HiCacheStorageExtraInfo] = None
    ) -> int:
        """
        Check if the keys exist in the storage.
        return the number of consecutive existing keys from the start.
        Can be overridden by subclasses for more efficient implementation.
        """
        for i in range(len(keys)):
            if not self.exists(keys[i]):
                return i
        return len(keys)

    def clear(self) -> None:
        pass

    def get_stats(self):
        return None

    def metadata_put_entries(
        self,
        scope: Dict[str, str],
        entries: List[MetadataEntry],
    ) -> None:
        """Write/update metadata entries. Idempotent: same hash_key overwrites."""
        return

    def metadata_get_snapshot(
        self,
        query: StorageMetadataQuery,
    ) -> MetadataSnapshot:
        """Get metadata snapshot.
        - query.since_version=None: full snapshot
        - query.since_version=X: incremental entries since version X
        """
        return MetadataSnapshot(
            schema_version=METADATA_SCHEMA_VERSION,
            version="0",
            scope={},
            entries=[],
        )


class HiCacheFile(HiCacheStorage):

    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get() or file_path

        tp_rank, tp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        self.tp_rank = tp_rank
        model_name = "-".join(model_name.split("/")) if model_name else ""
        if is_mla_model:
            self.config_suffix = f"_{model_name}"
        else:
            self.config_suffix = f"_{model_name}_{tp_rank}_{tp_size}"

        if not os.path.exists(self.file_path) and tp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

    def _get_suffixed_key(self, key: str) -> str:
        return key + self.config_suffix

    def get(
        self,
        key: str,
        target_location: torch.Tensor,
        target_sizes: Optional[Any] = None,
    ) -> torch.Tensor | None:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location
        except FileNotFoundError:
            logger.warning(f"Failed to fetch {key} from HiCacheFile storage.")
            return None

    def batch_get(
        self,
        keys: List[str],
        target_locations: List[torch.Tensor],
        target_sizes: Optional[Any] = None,
    ) -> List[torch.Tensor | None]:
        return [
            self.get(key, target_location)
            for key, target_location in zip(
                keys, target_locations or [None] * len(keys)
            )
        ]

    def set(
        self,
        key: str,
        value: Optional[Any] = None,
        target_location: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        if self.exists(key):
            logger.debug(f"Key {key} already exists. Skipped.")
            return True

        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        try:
            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save tensor {key}: {e}")
            return False

    def batch_set(
        self,
        keys: List[str],
        values: Optional[Any] = None,
        target_locations: Optional[Any] = None,
        target_sizes: Optional[Any] = None,
    ) -> bool:
        for key, value in zip(keys, values):
            if not self.set(key, value):
                return False
        return True

    def exists(self, key: str) -> bool:
        key = self._get_suffixed_key(key)
        tensor_path = os.path.join(self.file_path, f"{key}.bin")
        return os.path.exists(tensor_path)

    def clear(self) -> bool:
        try:
            for filename in os.listdir(self.file_path):
                file_path = os.path.join(self.file_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleared all entries in HiCacheFile storage.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear HiCacheFile storage: {e}")
            return False

    # ---- Metadata implementation ----

    def _metadata_dir(self) -> str:
        return os.path.join(self.file_path, f"__metadata__{self.config_suffix}")

    def _entries_dir(self) -> str:
        return os.path.join(self._metadata_dir(), "entries")

    def _versions_file(self) -> str:
        return os.path.join(self._metadata_dir(), f"versions_{self.tp_rank}.jsonl")

    def metadata_put_entries(
        self,
        scope: Dict[str, str],
        entries: List[MetadataEntry],
    ) -> None:
        if not entries:
            return
        try:
            entries_dir = self._entries_dir()
            os.makedirs(entries_dir, exist_ok=True)

            added_keys = []
            for entry in entries:
                entry_path = os.path.join(entries_dir, f"{entry.hash_key}.json")
                try:
                    data = asdict(entry)
                    with open(entry_path, "w") as f:
                        json.dump(data, f)
                    added_keys.append(entry.hash_key)
                except Exception as e:
                    logger.warning(
                        f"Failed to write metadata entry {entry.hash_key}: {e}"
                    )

            if added_keys:
                ts_ms = int(time.time() * 1000)
                version = f"{ts_ms}_{uuid.uuid4().hex[:8]}"
                version_line = json.dumps(
                    {"version": version, "ts": ts_ms, "added_keys": added_keys}
                )
                try:
                    with open(self._versions_file(), "a") as f:
                        f.write(version_line + "\n")
                except Exception as e:
                    logger.warning(f"Failed to append version log: {e}")
        except Exception as e:
            logger.warning(f"metadata_put_entries failed: {e}")

    def metadata_get_snapshot(
        self,
        query: StorageMetadataQuery,
    ) -> MetadataSnapshot:
        meta_dir = self._metadata_dir()
        entries_dir = self._entries_dir()

        # Rolling upgrade: directory may not exist yet
        if not os.path.isdir(meta_dir):
            return MetadataSnapshot(
                schema_version=METADATA_SCHEMA_VERSION,
                version="0",
                scope=query.scope or {},
                entries=[],
            )

        if query.since_version is not None:
            # Incremental: collect added_keys from all versions_*.jsonl after since_version
            target_keys = self._collect_incremental_keys(meta_dir, query.since_version)
        else:
            target_keys = None  # full scan

        entries = []
        latest_version = "0"

        if target_keys is not None:
            # Incremental mode: only read specified entry files
            for hk in target_keys:
                entry = self._read_entry_file(entries_dir, hk)
                if entry is not None:
                    entries.append(entry)
            # Get latest version from version logs
            latest_version = self._get_latest_version(meta_dir) or "0"
        else:
            # Full scan: read all entry files
            if os.path.isdir(entries_dir):
                for fname in os.listdir(entries_dir):
                    if not fname.endswith(".json"):
                        continue
                    hk = fname[:-5]
                    entry = self._read_entry_file(entries_dir, hk)
                    if entry is not None:
                        entries.append(entry)
            latest_version = self._get_latest_version(meta_dir) or "0"

        return MetadataSnapshot(
            schema_version=METADATA_SCHEMA_VERSION,
            version=latest_version,
            scope=query.scope or {},
            entries=entries,
        )

    def _read_entry_file(
        self, entries_dir: str, hash_key: str
    ) -> Optional[MetadataEntry]:
        entry_path = os.path.join(entries_dir, f"{hash_key}.json")
        try:
            with open(entry_path, "r") as f:
                data = json.load(f)
            sv = data.get("schema_version", 1)
            if sv > METADATA_SCHEMA_VERSION:
                logger.warning(
                    f"Skipping entry {hash_key} with unknown schema_version={sv}"
                )
                return None
            return MetadataEntry(
                hash_key=data["hash_key"],
                parent_hash=data.get("parent_hash"),
                token_ids=data.get("token_ids", []),
                schema_version=sv,
                priority=data.get("priority", 0),
                extra_key=data.get("extra_key"),
                ts_unix_ms=data.get("ts_unix_ms", 0),
            )
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Failed to read metadata entry {hash_key}: {e}")
            return None

    def _collect_incremental_keys(self, meta_dir: str, since_version: str) -> List[str]:
        """Collect added_keys from all versions_*.jsonl after since_version."""
        all_versions = self._parse_all_version_logs(meta_dir)
        keys = []
        found = False
        for ver_entry in all_versions:
            if found:
                keys.extend(ver_entry.get("added_keys", []))
            elif ver_entry.get("version") == since_version:
                found = True
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                deduped.append(k)
        return deduped

    def _get_latest_version(self, meta_dir: str) -> Optional[str]:
        all_versions = self._parse_all_version_logs(meta_dir)
        if all_versions:
            return all_versions[-1].get("version", "0")
        return None

    def _parse_all_version_logs(self, meta_dir: str) -> List[dict]:
        """Parse and merge all versions_*.jsonl files, sorted by ts."""
        all_entries = []
        try:
            for fname in os.listdir(meta_dir):
                if not fname.startswith("versions_") or not fname.endswith(".jsonl"):
                    continue
                fpath = os.path.join(meta_dir, fname)
                try:
                    with open(fpath, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                all_entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping corrupted line in {fname}")
                except Exception as e:
                    logger.warning(f"Failed to read version log {fname}: {e}")
        except Exception as e:
            logger.warning(f"Failed to list version logs: {e}")
        # Sort by timestamp
        all_entries.sort(key=lambda x: x.get("ts", 0))
        return all_entries

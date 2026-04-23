from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

CSD_EMPTY_KEY = -1
CSD_MAX_TOKEN_ID = (1 << 31) - 1
CSD_DEFAULT_LOAD_FACTOR = 0.5
CSD_DEFAULT_MAX_PROBE = 16
CSD_DEFAULT_DELTA_CAPACITY = 1 << 20
_UINT64_MASK = (1 << 64) - 1


def pack_csd_pair(lhs_token: int, rhs_token: int) -> int:
    if not 0 <= lhs_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(f"lhs_token must be in [0, {CSD_MAX_TOKEN_ID}], got {lhs_token}")
    if not 0 <= rhs_token <= CSD_MAX_TOKEN_ID:
        raise ValueError(f"rhs_token must be in [0, {CSD_MAX_TOKEN_ID}], got {rhs_token}")
    return (lhs_token << 32) | rhs_token


def unpack_csd_pair(key: int) -> Tuple[int, int]:
    if key < 0:
        raise ValueError(f"CSD key must be non-negative, got {key}")
    return key >> 32, key & 0xFFFFFFFF


def _hash64(key: int) -> int:
    x = key & _UINT64_MASK
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _UINT64_MASK
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _UINT64_MASK
    return (x ^ (x >> 31)) & _UINT64_MASK


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@dataclass
class CSDEntry:
    freq: int = 1
    allow: bool = True

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "CSDEntry":
        return cls(
            freq=int(record.get("freq", 1)),
            allow=bool(record.get("allow", True)),
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "freq": int(self.freq),
            "allow": bool(self.allow),
        }


@dataclass
class CSDTableStore:
    entries: Dict[int, CSDEntry] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "CSDTableStore":
        table_path = Path(path)
        if table_path.suffix == ".jsonl":
            return cls._load_jsonl(table_path)
        return cls._load_json(table_path)

    @classmethod
    def _load_json(cls, path: Path) -> "CSDTableStore":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            metadata: Dict[str, Any] = {}
            records = payload
        else:
            metadata = dict(payload.get("metadata", {}))
            records = payload.get("entries", [])

        store = cls(metadata=metadata)
        for record in records:
            key = _record_to_key(record)
            store.entries[key] = CSDEntry.from_record(record)
        return store

    @classmethod
    def _load_jsonl(cls, path: Path) -> "CSDTableStore":
        store = cls()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("type") == "metadata":
                    store.metadata.update(record.get("metadata", {}))
                    continue
                key = _record_to_key(record)
                store.entries[key] = CSDEntry.from_record(record)
        return store

    def save(self, path: str) -> None:
        table_path = Path(path)
        if table_path.suffix == ".jsonl":
            self._save_jsonl(table_path)
        else:
            self._save_json(table_path)

    def _save_json(self, path: Path) -> None:
        payload = {
            "metadata": self.metadata,
            "entries": self.to_records(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def _save_jsonl(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            if self.metadata:
                f.write(json.dumps({"type": "metadata", "metadata": self.metadata}) + "\n")
            for record in self.to_records():
                f.write(json.dumps(record) + "\n")

    def add_pair(
        self,
        lhs_token: int,
        rhs_token: int,
        freq: int = 1,
        allow: bool = True,
    ) -> None:
        self.entries[pack_csd_pair(lhs_token, rhs_token)] = CSDEntry(
            freq=freq,
            allow=allow,
        )

    def merge_counts(self, counts: Counter[int]) -> None:
        for key, count in counts.items():
            entry = self.entries.get(key)
            if entry is None:
                self.entries[key] = CSDEntry(freq=int(count), allow=False)
            else:
                entry.freq += int(count)

    def filtered_keys(self, freq_threshold: int) -> List[int]:
        return [
            key
            for key, entry in sorted(
                self.entries.items(),
                key=lambda item: item[1].freq,
                reverse=True,
            )
            if entry.allow and entry.freq >= freq_threshold
        ]

    def build_allow_hash_table(
        self,
        device: torch.device | str,
        freq_threshold: int,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
    ) -> "CSDHashTable":
        return build_csd_hash_table(
            self.filtered_keys(freq_threshold),
            device=device,
            max_probe=max_probe,
            load_factor=load_factor,
        )

    def to_records(self) -> List[Dict[str, Any]]:
        records = []
        for key, entry in sorted(self.entries.items()):
            lhs_token, rhs_token = unpack_csd_pair(key)
            record = entry.to_record()
            record.update(
                {
                    "key": int(key),
                    "lhs_token": int(lhs_token),
                    "rhs_token": int(rhs_token),
                }
            )
            records.append(record)
        return records


def _record_to_key(record: Dict[str, Any]) -> int:
    if "key" in record:
        key = int(record["key"])
        if key == CSD_EMPTY_KEY:
            raise ValueError("CSD table entry cannot use the empty-key sentinel")
        return key
    return pack_csd_pair(int(record["lhs_token"]), int(record["rhs_token"]))


@dataclass
class CSDHashTable:
    keys: torch.Tensor
    num_entries: int
    capacity: int
    max_probe: int

    @classmethod
    def empty(
        cls,
        device: torch.device | str,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
    ) -> "CSDHashTable":
        return cls(
            keys=torch.full((1,), CSD_EMPTY_KEY, dtype=torch.int64, device=device),
            num_entries=0,
            capacity=1,
            max_probe=max_probe,
        )

    @property
    def has_entries(self) -> bool:
        return self.num_entries > 0

    def contains_for_test(self, key: int) -> bool:
        slot = _hash64(int(key)) & (self.capacity - 1)
        keys = self.keys.detach().cpu().tolist()
        for _ in range(self.max_probe):
            existing_key = keys[slot]
            if existing_key == int(key):
                return True
            if existing_key == CSD_EMPTY_KEY:
                return False
            slot = (slot + 1) & (self.capacity - 1)
        return False


def build_csd_hash_table(
    keys: Iterable[int],
    device: torch.device | str,
    max_probe: int = CSD_DEFAULT_MAX_PROBE,
    load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
) -> CSDHashTable:
    if max_probe < 1:
        raise ValueError("CSD max_probe must be at least 1")
    if not 0 < load_factor <= 1:
        raise ValueError("CSD load_factor must be in the range (0, 1]")

    key_list = [int(key) for key in keys]
    if not key_list:
        return CSDHashTable.empty(device=device, max_probe=max_probe)

    for key in key_list:
        if key < 0:
            raise ValueError(f"CSD keys must be non-negative, got {key}")

    capacity = _next_power_of_two(max(2, math.ceil(len(key_list) / load_factor)))
    while True:
        hash_keys = _try_build_hash_table(key_list, capacity, max_probe)
        if hash_keys is not None:
            return CSDHashTable(
                keys=torch.tensor(hash_keys, dtype=torch.int64, device=device),
                num_entries=len(set(key_list)),
                capacity=capacity,
                max_probe=max_probe,
            )
        capacity *= 2


def _try_build_hash_table(
    keys: List[int],
    capacity: int,
    max_probe: int,
) -> Optional[List[int]]:
    hash_keys = [CSD_EMPTY_KEY] * capacity

    for key in keys:
        slot = _hash64(key) & (capacity - 1)
        inserted = False
        for _ in range(max_probe):
            existing_key = hash_keys[slot]
            if existing_key in (CSD_EMPTY_KEY, key):
                hash_keys[slot] = key
                inserted = True
                break
            slot = (slot + 1) & (capacity - 1)
        if not inserted:
            return None

    return hash_keys


@dataclass
class CSDMetrics:
    lookup_hit_ct: torch.Tensor
    forced_accept_ct: torch.Tensor
    delta_pair_ct: torch.Tensor

    @classmethod
    def allocate(cls, device: torch.device | str) -> "CSDMetrics":
        return cls(
            lookup_hit_ct=torch.zeros((1,), dtype=torch.int64, device=device),
            forced_accept_ct=torch.zeros((1,), dtype=torch.int64, device=device),
            delta_pair_ct=torch.zeros((1,), dtype=torch.int64, device=device),
        )

    def reset(self) -> None:
        self.lookup_hit_ct.zero_()
        self.forced_accept_ct.zero_()
        self.delta_pair_ct.zero_()

    def snapshot(self) -> Dict[str, int]:
        return {
            "csd_lookup_hit_ct": int(self.lookup_hit_ct.item()),
            "csd_forced_accept_ct": int(self.forced_accept_ct.item()),
            "csd_delta_pair_ct": int(self.delta_pair_ct.item()),
        }


@dataclass
class CSDDeltaBuffer:
    pairs: torch.Tensor
    counter: torch.Tensor

    @classmethod
    def allocate(
        cls,
        device: torch.device | str,
        capacity: int = CSD_DEFAULT_DELTA_CAPACITY,
    ) -> "CSDDeltaBuffer":
        return cls(
            pairs=torch.empty((capacity,), dtype=torch.int64, device=device),
            counter=torch.zeros((1,), dtype=torch.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return self.pairs.shape[0]

    def reset(self) -> None:
        self.counter.zero_()

    def drain_to_counter(self) -> Counter[int]:
        count = min(int(self.counter.item()), self.capacity)
        if count == 0:
            return Counter()
        pairs = self.pairs[:count].detach().cpu().tolist()
        self.reset()
        return Counter(int(pair) for pair in pairs if int(pair) >= 0)


@dataclass
class CSDRuntime:
    enabled: bool
    dynamic_update: bool
    force_accept_disabled: bool
    table: CSDHashTable
    metrics: CSDMetrics
    delta_buffer: Optional[CSDDeltaBuffer] = None
    table_store: CSDTableStore = field(default_factory=CSDTableStore)
    delta_counts: Counter[int] = field(default_factory=Counter)
    delta_save_path: Optional[str] = None

    @classmethod
    def from_server_args(
        cls,
        server_args: "ServerArgs",
        device: torch.device | str,
        delta_capacity: int = CSD_DEFAULT_DELTA_CAPACITY,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
    ) -> "CSDRuntime":
        metrics = CSDMetrics.allocate(device)
        if not server_args.speculative_csd_enabled:
            return cls(
                enabled=False,
                dynamic_update=False,
                force_accept_disabled=False,
                table=CSDHashTable.empty(device=device, max_probe=max_probe),
                metrics=metrics,
            )

        table_store = CSDTableStore()
        if server_args.speculative_csd_table_path is not None:
            table_store = CSDTableStore.load(server_args.speculative_csd_table_path)

        table = table_store.build_allow_hash_table(
            device=device,
            freq_threshold=server_args.speculative_csd_freq_threshold,
            max_probe=max_probe,
            load_factor=load_factor,
        )
        delta_buffer = None
        if server_args.speculative_csd_dynamic_update:
            delta_buffer = CSDDeltaBuffer.allocate(device=device, capacity=delta_capacity)

        return cls(
            enabled=True,
            dynamic_update=server_args.speculative_csd_dynamic_update,
            force_accept_disabled=server_args.speculative_csd_force_accept_disabled,
            table=table,
            metrics=metrics,
            delta_buffer=delta_buffer,
            table_store=table_store,
            delta_save_path=server_args.speculative_csd_delta_save_path,
        )

    @property
    def has_table(self) -> bool:
        return self.table.has_entries

    def flush_delta(self) -> Counter[int]:
        if self.delta_buffer is None:
            return Counter()
        counts = self.delta_buffer.drain_to_counter()
        self.delta_counts.update(counts)
        return counts

    def save_delta(self, path: Optional[str] = None) -> None:
        self.flush_delta()
        save_path = path or self.delta_save_path
        if save_path is None:
            raise ValueError("CSD delta save path is not configured")
        delta_store = CSDTableStore(metadata=dict(self.table_store.metadata))
        delta_store.metadata["content"] = "raw_delta_counts"
        delta_store.merge_counts(self.delta_counts)
        delta_store.save(save_path)

    def save_table(self, path: str) -> None:
        self.flush_delta()
        self.table_store.merge_counts(self.delta_counts)
        self.delta_counts.clear()
        self.table_store.save(path)

    def rebuild_table(
        self,
        device: torch.device | str,
        freq_threshold: int,
        max_probe: int = CSD_DEFAULT_MAX_PROBE,
        load_factor: float = CSD_DEFAULT_LOAD_FACTOR,
    ) -> None:
        self.flush_delta()
        self.table_store.merge_counts(self.delta_counts)
        self.delta_counts.clear()
        # TODO: Move rebuild off the hot path and swap in a freshly built table at an idle-safe boundary.
        self.table = self.table_store.build_allow_hash_table(
            device=device,
            freq_threshold=freq_threshold,
            max_probe=max_probe,
            load_factor=load_factor,
        )

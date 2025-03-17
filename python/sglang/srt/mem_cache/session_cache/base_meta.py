from abc import ABC, abstractmethod
from typing import List, Optional


class SessionCacheEntry:
    def __init__(
        self,
        connector,
        uri: str,
        offset: int,
        length: int,
    ):
        self.offset = offset
        self.length = length
        self.connector = connector
        self.uri = uri


class SessionCacheMeta:
    def __init__(
        self,
        sid: str,
        entries: Optional[List[SessionCacheEntry]] = None,
    ):
        self.sid = sid
        self.entries = entries

    def _get_next_index(self) -> int:
        if self.entries is None:
            return 0

        return len(self.entries)

    def _get_new_uri(self, prefix: str) -> str:
        return prefix + "/" + str(self._get_next_index())

    def append(self, entry: SessionCacheEntry):
        if self.entries is None:
            self.entries = [entry]
            return

        self.entries.append(entry)

    def get_length(self) -> int:
        length = 0
        if self.entries is None:
            return length

        for entry in self.entries:
            length += entry.length

        return length

    def get_uris(self):
        uris = []
        if self.entries is None:
            return uris

        for entry in self.entries:
            uris.append(entry.uri)

        return uris

    def get_entries(self):
        entries = []
        if self.entries is None:
            return entries

        for entry in self.entries:
            entries.append(entry)

        return entries

    def get_next_entry_info(self, prefix: str, new_length: int):
        path = self._get_new_uri(prefix)
        if self.entries is None:
            return path, 0, new_length

        length = 0
        for entry in self.entries:
            length += entry.length

        offset = length
        inc_length = new_length - length
        return path, offset, inc_length


def get_meta_manager_class(url: str):
    if url == "memkv":
        return MemKVSessionCacheMetaManager()
    else:
        raise ValueError("only supports memkv session cache.")


class BaseSessionCacheMetaManager(ABC):

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def save(self, sid: str, meta: SessionCacheMeta) -> None:
        raise NotImplementedError()

    @abstractmethod
    def load(self, sid: str) -> Optional[SessionCacheMeta]:
        raise NotImplementedError()

    @abstractmethod
    def exist(self, sid: str) -> bool:
        raise NotImplementedError()


class MemKVSessionCacheMetaManager(BaseSessionCacheMetaManager):
    """Store the managed metadata information in the CPU."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.entries = {}

    def save(self, sid, meta: SessionCacheMeta) -> None:
        self.entries[sid] = meta

    def load(self, sid: str) -> Optional[SessionCacheMeta]:
        if sid not in self.entries:
            return None
        return self.entries[sid]

    def exist(self, sid) -> bool:
        return sid in self.entries

import asyncio
from contextvars import ContextVar
from typing import Any, Dict, Optional

# asyncio tasks spawned by a request inherit its scope; independent requests do not.
media_tenant: ContextVar[str | None] = ContextVar("media_tenant", default=None)


class AsyncDictStore:
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: dict[str, tuple[str | None, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    def _owned(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._items.get(key)
        return entry[1] if entry and entry[0] == media_tenant.get() else None

    async def upsert(self, key: str, value: Dict[str, Any]) -> None:
        async with self._lock:
            if key in self._items and self._owned(key) is None:
                raise ValueError("Job ID already allocated")
            self._items[key] = (media_tenant.get(), value)

    async def update_fields(
        self, key: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        async with self._lock:
            item = self._owned(key)
            if item is None:
                return None
            item.update(updates)
            return item

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._owned(key)

    async def pop(self, key: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if self._owned(key) is None:
                return None
            return self._items.pop(key)[1]

    async def list_page(
        self,
        *,
        after: Optional[str] = None,
        limit: int = 20,
        order: Optional[str] = "desc",
    ) -> Dict[str, Any]:
        """Return a created-time ordered page, with OpenAI-style cursor semantics."""
        normalized_order = (order or "desc").lower()
        reverse = normalized_order != "asc"
        async with self._lock:
            tenant = media_tenant.get()
            items = sorted(
                (item for owner, item in self._items.values() if owner == tenant),
                key=lambda item: item.get("created_at", 0),
                reverse=reverse,
            )

        if after is not None:
            try:
                index = next(i for i, item in enumerate(items) if item["id"] == after)
                items = items[index + 1 :]
            except StopIteration:
                items = []

        page = items[:limit]
        return {
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > limit,
        }


class VideoJobStore(AsyncDictStore):
    async def record_scheduler_progress(self, job_id: str, progress: int) -> None:
        # The internal scheduler knows only a server-issued job ID, not the tenant.
        # Give the listener a progress-only write, never an unscoped record accessor.
        async with self._lock:
            entry = self._items.get(job_id)
            if entry and (job := entry[1])["status"] in ("queued", "in_progress"):
                job["status"] = "in_progress"
                job["progress"] = max(job["progress"], min(progress, 99))


VIDEO_STORE = VideoJobStore()
IMAGE_STORE = AsyncDictStore()
MESH_STORE = AsyncDictStore()

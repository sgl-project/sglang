"""Platform seam for the optional sleep-mode ``comm`` memory tag.

The scheduler owns tag bookkeeping (`offload_tags`, the release/resume order,
the request/response plumbing).  A platform that can release and rebuild its
own device communication domains registers a ``SleepCommBackend`` here from its
own initialization path; this module never imports a platform, so a platform
that does not register keeps the historical tag set and behaviour unchanged.

Registered backends add exactly one tag (``SleepCommBackend.tag``).  Where a
tag is honored:

* a tag-less release means ``GPU_MEMORY_ALL_TYPES`` plus the platform tags of
  the *current* process, so NPU full sleep includes ``comm`` while CUDA full
  sleep does not (a tag-less resume always means "whatever is released");
* an explicit platform tag from a client that the current platform did not
  register is dropped with a log line instead of failing the request, so one
  client can send the same tag list to every platform.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: The memory-occupation tag served by platform sleep-comm backends. Defined
#: here rather than in sglang.srt.constants so this bottom layer of
#: sglang.srt.utils keeps its "no srt-domain module-level import" property
#: (enforced by scripts/lint/check_srt_utils_imports.py).
GPU_MEMORY_TYPE_COMM = "comm"

#: Tags accepted only when the current process registered a backend for them.
PLATFORM_GATED_TAGS: Tuple[str, ...] = (GPU_MEMORY_TYPE_COMM,)

_BACKEND_CLS: Optional[type] = None


class SleepCommBackend:
    """Release/resume of the device communication domains for one tag.

    An instance is created per request, so a backend must keep no state of its
    own; process-wide state belongs in module scope.
    """

    #: The memory-occupation tag this backend serves.
    tag: ClassVar[str] = GPU_MEMORY_TYPE_COMM

    #: The backend drops and recaptures graphs itself, so the scheduler must not
    #: also hand the ``cuda_graph`` tag to the memory saver.
    handles_cuda_graph: ClassVar[bool] = False

    #: Tags that must be resumed in the same request, because ``resume`` reads
    #: them (recapturing a graph runs the model).
    resume_requires: ClassVar[FrozenSet[str]] = frozenset()

    def __init__(self, tp_worker: Any, draft_worker: Any, scheduler: Any) -> None:
        self.tp_worker = tp_worker
        self.draft_worker = draft_worker
        self.scheduler = scheduler

    @classmethod
    def unsupported_reason(cls) -> Optional[str]:
        """Why this backend cannot serve its tag in this process, or None."""
        return None

    def agree(self, proceed: bool) -> bool:
        """AND-reduce a local go/no-go decision over every rank that this
        backend's device work spans, or return ``proceed`` when it has none."""
        return proceed

    def release(self) -> None:
        raise NotImplementedError

    def resume(self) -> None:
        raise NotImplementedError


def register_sleep_comm_backend(backend_cls: type) -> None:
    """Register this process's backend. One accelerator per process."""
    global _BACKEND_CLS
    if backend_cls.tag not in PLATFORM_GATED_TAGS:
        raise ValueError(
            f"sleep-comm backend tag {backend_cls.tag!r} must be listed in "
            f"PLATFORM_GATED_TAGS ({PLATFORM_GATED_TAGS}); otherwise a tag-less "
            "request would never resolve it."
        )
    _BACKEND_CLS = backend_cls


def get_sleep_comm_backend_class() -> Optional[type]:
    return _BACKEND_CLS


def enabled_platform_tags() -> Tuple[str, ...]:
    """Platform tags the current process can serve."""
    backend_cls = _BACKEND_CLS
    return (backend_cls.tag,) if backend_cls is not None else ()


def all_known_tags() -> List[str]:
    """Every tag this build accepts, in the default release order."""
    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES

    return list(GPU_MEMORY_ALL_TYPES) + list(PLATFORM_GATED_TAGS)


def default_release_tags() -> List[str]:
    """What a tag-less release means in this process."""
    from sglang.srt.constants import GPU_MEMORY_ALL_TYPES

    return list(GPU_MEMORY_ALL_TYPES) + list(enabled_platform_tags())


def resolve_release_tags(raw_tags: Optional[List[str]]) -> List[str]:
    """Resolve a request's tag list; raise ValueError on an unknown tag.

    Kept free of side effects so a rejected request can be answered before any
    resource is touched.
    """
    if not raw_tags:
        return default_release_tags()

    tags = list(dict.fromkeys(raw_tags))
    unknown = sorted(set(tags) - set(all_known_tags()))
    if unknown:
        raise ValueError(f"Unknown memory occupation tag(s): {unknown}")

    enabled = set(enabled_platform_tags())
    dropped = [tag for tag in tags if tag in PLATFORM_GATED_TAGS and tag not in enabled]
    if dropped:
        logger.info(
            "Memory occupation tag(s) %s are not supported on this platform; "
            "ignoring them.",
            dropped,
        )
    return [tag for tag in tags if tag not in dropped]


def make_sleep_comm_backend(
    tp_worker: Any, draft_worker: Any, scheduler: Any
) -> Optional[SleepCommBackend]:
    """The current process's backend instance, or None when none registered."""
    backend_cls = _BACKEND_CLS
    if backend_cls is None:
        return None
    return backend_cls(
        tp_worker=tp_worker, draft_worker=draft_worker, scheduler=scheduler
    )

"""Share layerwise pinned CPU weights between processes on the same host.

The creator holds the owner lock exclusively. Followers hold the users lock
shared. Under the decision mutex, the last participant or next creator can
reclaim a generation only after taking both locks exclusively. The manifest
identifies the segments to unlink, including after a SIGKILL.

The creator publishes a ready marker after registering each filled segment.
Followers wait for the marker and verify their checkpoint bytes before reuse.
"""

import ctypes
import fcntl
import hashlib
import logging
import os
import signal
import time
import weakref
from multiprocessing import resource_tracker, shared_memory, util
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MARKER_DIR = "/tmp/sglang-shared-stores"

# Wait while a slow creator continues to publish layers.
STALL_TIMEOUT_S = 90.0
MAX_WAIT_S = 900.0

DECIDE_TIMEOUT_S = 120.0

_LIVE: dict[int, "SharedStore"] = {}
_POOL_REFS: dict[str, int] = {}
_LOCKS: dict[str, list[int]] = {}
_ROLE: dict[str, bool | None] = {}
_STATS: dict[str, dict[str, int]] = {}
_LEAVING: set[str] = set()
_TERM_HOOKED = False
_RELEASING = False
_TERM_SIGNALS = {signal.SIGTERM, signal.SIGINT}
_PREV_SIGNAL: dict[int, object] = {}


class SharedStore:
    """One acquired segment, not yet filled or registered."""

    __slots__ = ("shm", "created", "key", "nbytes", "pool", "registered", "closed")

    def __init__(self, shm, created: bool, key: str, nbytes: int, pool: str) -> None:
        self.shm = shm
        self.created = created
        self.key = key
        self.nbytes = nbytes
        self.pool = pool
        self.registered = False
        self.closed = False
        _POOL_REFS[pool] = _POOL_REFS.get(pool, 0) + 1

    @property
    def buffer(self):
        return self.shm.buf

    def tensor(self, dtype: torch.dtype) -> torch.Tensor:
        """Give the last tensor view ownership of this acquired mapping."""
        tensor = torch.frombuffer(self.buffer, dtype=dtype)
        weakref.finalize(tensor.untyped_storage(), release, self)
        return tensor


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _names(key: str) -> tuple[str, str]:
    token = _digest(key)
    return f"sgl-pin-{token}", os.path.join(MARKER_DIR, f"sgl-pin-{token}.ready")


def _marker_path(name: str) -> str:
    return os.path.join(MARKER_DIR, f"{name}.ready")


def _pool_paths(pool: str) -> tuple[str, str, str, str, str]:
    base = os.path.join(MARKER_DIR, f"sgl-pin-pool-{_digest(pool)}")
    return (
        base + ".owner",
        base + ".users",
        base + ".decide",
        base + ".progress",
        base + ".manifest",
    )


def summary(pool: str) -> dict[str, int]:
    """How this process fared against the pool: created, joined, or refused."""
    stats = _STATS.get(pool, {"created": 0, "joined": 0, "refused": 0})
    logger.info(
        "Shared pinned stores for %s: %s, created %d, joined %d, refused %d",
        pool,
        {True: "owner", False: "follower", None: "not sharing"}[_ROLE.get(pool)],
        stats["created"],
        stats["joined"],
        stats["refused"],
    )
    return dict(stats)


def _try_lock(fd: int, mode: int) -> bool:
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _decide_lock(fd: int) -> bool:
    deadline = time.monotonic() + DECIDE_TIMEOUT_S
    while not _try_lock(fd, fcntl.LOCK_EX):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def join_pool(pool: str) -> bool | None:
    """Return True for the creator, False for a follower, or None on timeout."""
    if pool in _ROLE:
        return _ROLE[pool]
    os.makedirs(MARKER_DIR, exist_ok=True)
    owner_path, users_path, decide_path, _, _ = _pool_paths(pool)
    opened = [os.open(decide_path, os.O_CREAT | os.O_RDWR, 0o600)]
    held: list[int] = []
    creator = False
    try:
        if not _decide_lock(opened[0]):
            logger.warning("shared pinned store %s: decision lock is stuck", pool)
            _ROLE[pool] = None
            return None
        owner_fd = os.open(owner_path, os.O_CREAT | os.O_RDWR, 0o600)
        users_fd = os.open(users_path, os.O_CREAT | os.O_RDWR, 0o600)
        opened += [owner_fd, users_fd]
        if _try_lock(owner_fd, fcntl.LOCK_EX):
            held.append(owner_fd)
            if _try_lock(users_fd, fcntl.LOCK_EX):
                # No creator and no follower is alive, so the segments the
                # manifest lists have no readers and can be replaced.
                _reclaim(pool)
                fcntl.flock(users_fd, fcntl.LOCK_UN)
                creator = True
            else:
                # The creator died but followers are still reading its
                # segments: join them, never replace them.
                fcntl.flock(users_fd, fcntl.LOCK_SH)
                held.append(users_fd)
        else:
            fcntl.flock(users_fd, fcntl.LOCK_SH)
            held.append(users_fd)
        _LOCKS[pool] = held
        _ROLE[pool] = creator
        _note_leave(pool)
        return creator
    finally:
        for fd in opened:
            if fd not in held:
                os.close(fd)


def _release_and_die(signum: int, _frame) -> None:
    """Release pool seats and deliver the signal to the previous handler."""
    # A second signal can re-enter this handler before the signal mask is set.
    global _RELEASING
    if _RELEASING:
        return
    _RELEASING = True
    signal.pthread_sigmask(signal.SIG_BLOCK, _TERM_SIGNALS)
    for pool in list(_LOCKS):
        try:
            leave_pool(pool)
        except Exception:  # a process on its way out must not raise
            logger.warning("shared pinned store %s: release on signal failed", pool)
    signal.signal(signum, _PREV_SIGNAL.get(signum, signal.SIG_DFL))
    os.kill(os.getpid(), signum)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, _TERM_SIGNALS)


def _note_leave(pool: str) -> None:
    """Cover signal termination and multiprocessing's os._exit path."""
    global _TERM_HOOKED
    if not _TERM_HOOKED:
        _TERM_HOOKED = True
        for signum in sorted(_TERM_SIGNALS):
            try:
                _PREV_SIGNAL[signum] = signal.getsignal(signum)
                signal.signal(signum, _release_and_die)
            except (ValueError, OSError):
                # Not the main thread; the finalizer below still covers a return.
                _PREV_SIGNAL.pop(signum, None)
    if pool in _LEAVING:
        return
    _LEAVING.add(pool)
    util.Finalize(None, leave_pool, args=(pool,), exitpriority=10)


def leave_pool(pool: str) -> bool:
    """Give up this process's seat, reclaiming the generation if it was the last.

    Under the decision lock, release our seat and test for remaining readers.
    On timeout, drop only our seat. The next creator or last participant can
    then acquire both locks and reclaim the segments.
    """
    held = _LOCKS.pop(pool, None)
    _ROLE.pop(pool, None)
    if held is None:
        return False

    owner_path, users_path, decide_path, _, _ = _pool_paths(pool)
    opened = []
    reclaimed = False
    try:
        decide_fd = os.open(decide_path, os.O_CREAT | os.O_RDWR, 0o600)
        opened.append(decide_fd)
        if not _decide_lock(decide_fd):
            return False
        while held:
            os.close(held.pop())
        owner_fd = os.open(owner_path, os.O_CREAT | os.O_RDWR, 0o600)
        opened.append(owner_fd)
        users_fd = os.open(users_path, os.O_CREAT | os.O_RDWR, 0o600)
        opened.append(users_fd)
        if _try_lock(owner_fd, fcntl.LOCK_EX) and _try_lock(users_fd, fcntl.LOCK_EX):
            released = _reclaim(pool)
            reclaimed = bool(released)
            if released:
                logger.info(
                    "Shared pinned store %s: last participant leaving, "
                    "released the generation instead of leaving it for the "
                    "next start",
                    pool,
                )
    finally:
        for fd in held + opened:
            try:
                os.close(fd)
            except OSError:
                pass
    return reclaimed


def _reclaim(pool: str) -> int:
    """Unlink whatever an earlier generation left behind; returns its segment count.

    Only reachable while the owner and users locks are both held exclusively,
    which means every process of the previous generation is gone; their pages
    would otherwise stay resident with no name pointing at them.
    """
    _, _, _, progress, manifest = _pool_paths(pool)
    try:
        with open(manifest) as handle:
            names = [line.strip() for line in handle if line.strip()]
    except OSError:
        return 0
    for name in names:
        try:
            stale = shared_memory.SharedMemory(name=name)
        except (FileNotFoundError, OSError):
            pass
        else:
            try:
                stale.unlink()
            finally:
                stale.close()
        try:
            os.remove(_marker_path(name))
        except OSError:
            pass
    for path in (manifest, progress):
        try:
            os.remove(path)
        except OSError:
            pass
    logger.info(
        "Shared pinned store %s: reclaimed %d segments from a dead generation",
        pool,
        len(names),
    )
    return len(names)


def _note_segment(pool: str, name: str) -> None:
    """Record a segment as part of this generation before it is filled, so a
    crash cannot leave a name behind that no later generation knows to reap."""
    _, _, _, _, manifest = _pool_paths(pool)
    try:
        with open(manifest, "a") as handle:
            handle.write(name + "\n")
    except OSError:
        pass


def _publish_progress(pool: str) -> None:
    _, _, _, progress, _ = _pool_paths(pool)
    try:
        Path(progress).touch()
    except OSError:
        pass


def _register(buf, nbytes: int) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        return int(torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)) == 0
    except Exception:
        return False


def _unregister(buf) -> None:
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    torch.cuda.cudart().cudaHostUnregister(ptr)


def release(store: SharedStore) -> None:
    """Release a mapping after its last tensor view, or an uncommitted store.

    The storage finalizer owns the registration, mapping and pool seat together.
    Keeping a tensor view therefore keeps all three alive, including when its
    manager has already dropped its own tensors.
    """
    if store.closed:
        return
    if store.created and id(store) not in _LIVE:
        # A failed registration was never published. Unlink it even if other
        # layers keep the pool alive. Reopening balances the resource tracker
        # registration, which begin() removed for the original handle.
        try:
            abandoned = shared_memory.SharedMemory(name=store.shm.name)
        except FileNotFoundError:
            pass
        else:
            abandoned.unlink()
            abandoned.close()
    if store.registered:
        _unregister(store.buffer)
        store.registered = False
    store.shm.close()
    store.closed = True
    _LIVE.pop(id(store), None)
    remaining = _POOL_REFS[store.pool] - 1
    if remaining:
        _POOL_REFS[store.pool] = remaining
    else:
        del _POOL_REFS[store.pool]
        leave_pool(store.pool)


def _leave_unused_pool(pool: str) -> None:
    if not _POOL_REFS.get(pool):
        leave_pool(pool)


def _wait_for_marker(marker: str, pool: str) -> bool:
    """Wait for the creator to publish this store, while it keeps advancing."""
    _, _, _, progress, _ = _pool_paths(pool)
    started = time.monotonic()
    while True:
        if os.path.exists(marker):
            return True
        now = time.monotonic()
        if now - started > MAX_WAIT_S:
            logger.warning(
                "shared pinned store %s: no progress for %.0fs; this layer "
                "stays private",
                pool,
                now - started,
            )
            return False
        try:
            # The file's mtime is wall clock; `now` is monotonic and decades
            # apart from it, so the subtraction has to be on the same scale.
            stalled = time.time() - os.path.getmtime(progress) > STALL_TIMEOUT_S
        except OSError:
            stalled = now - started > STALL_TIMEOUT_S
        if stalled:
            logger.warning(
                "shared pinned store %s: owner stopped advancing %.0fs ago; "
                "this layer stays private",
                pool,
                STALL_TIMEOUT_S,
            )
            return False
        time.sleep(0.2)


def begin(key: str, nbytes: int, verify, pool: str) -> SharedStore | None:
    """Create or attach the store for `key`; None means allocate privately."""
    name, marker = _names(key)
    stats = _STATS.setdefault(pool, {"created": 0, "joined": 0, "refused": 0})
    creator = join_pool(pool)
    os.makedirs(MARKER_DIR, exist_ok=True)
    if creator is None:
        stats["refused"] += 1
        return None

    # A published layer may still be read by another manager or process even
    # after this process detached its own copy. Reuse it for this generation.
    create = creator and not os.path.exists(marker)
    if create:
        # Listed before it exists, so an interrupted creation still leaves the
        # name in the manifest for the next generation to unlink.
        _note_segment(pool, name)
        _publish_progress(pool)
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
        except FileExistsError:
            # A manifest lost to a crash left the name behind. This process
            # holds the pool, so no reader can be attached to it.
            shared_memory.SharedMemory(name=name).unlink()
            shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
    else:
        # The segment may not exist yet: the creator fills one layer at a time
        # and may still be behind this layer. Waiting for its marker costs a
        # wait, a second private copy costs the memory the pool exists to save.
        if not _wait_for_marker(marker, pool):
            stats["refused"] += 1
            _leave_unused_pool(pool)
            return None
        try:
            shm = shared_memory.SharedMemory(name=name)
        except (FileNotFoundError, OSError):
            stats["refused"] += 1
            _leave_unused_pool(pool)
            return None

    # Python 3.12 has no track=False. A worker's resource tracker must not
    # unlink shared segments while another worker is using them.
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    if create:
        if os.path.exists(marker):
            os.remove(marker)
        stats["created"] += 1
        if stats["created"] == 1:
            logger.info(
                "shared pinned store %s: owner created %s (%.2f GiB) for %s",
                pool,
                name,
                nbytes / 2**30,
                key,
            )
        return SharedStore(shm, True, key, nbytes, pool)

    stats["joined"] += 1
    if stats["joined"] == 1:
        logger.info("shared pinned store %s: attached %s for %s", pool, name, key)

    try:
        matches = verify(shm.buf)
    except Exception as exc:
        logger.warning("shared pinned store %s could not be verified (%s)", pool, exc)
    else:
        if matches:
            return SharedStore(shm, False, key, nbytes, pool)
        logger.warning("shared pinned store %s holds different weights", pool)
    shm.close()
    stats["refused"] += 1
    _leave_unused_pool(pool)
    return None


def commit(store: SharedStore) -> bool:
    """Register and publish a store whose tensor already owns its lifetime.

    On failure the caller drops its tensor before allocating privately. The
    storage finalizer then closes the mapping and returns its pool seat.
    """
    if not _register(store.buffer, store.nbytes):
        return False
    store.registered = True
    name, marker = _names(store.key)
    try:
        if store.created:
            with open(marker, "w") as handle:
                handle.write(name)
            _publish_progress(store.pool)
    except OSError:
        logger.warning("shared pinned store %s: could not publish %s", store.pool, name)
        return False
    _LIVE[id(store)] = store
    return True

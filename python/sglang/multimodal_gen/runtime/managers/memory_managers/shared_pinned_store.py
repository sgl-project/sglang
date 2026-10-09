"""Named pinned host stores that co-resident instances serve from one copy.

The layerwise offload weight store has to be anonymous shared memory: it is the
only form cudaHostRegister accepts that can also be shared between processes.
sglang creates it with an anonymous memfd, so every instance on the host gets
its own copy of the whole component -- 62.28 GB for H3's transformer, 46.18 GB
for its text encoder.

Naming the segment turns those into one copy per component. Three things have
to hold for that to be stable, and each of them was learned the hard way:

1. The name has to outlive the creating process's tracker.
   `multiprocessing.SharedMemory(create=True)` registers the segment with a
   resource_tracker, which unlinks everything it registered when its process
   exits. In a serve process tree that tracker belongs to a short-lived parent,
   so the segments were unlinked while the creator was still serving: a later
   instance found the name free, created a second full pool under the same
   names, and the first pool survived only as unreachable pages. /dev/shm held
   two 109 GiB generations, and the two instances shared nothing. Every create
   now unregisters itself (Python 3.12 has no `track=False`).

2. A generation may only be replaced when nothing is reading it.
   Two locks answer that, and neither is ever converted:
     <base>.owner  held exclusively for its whole life by the one process that
                   is allowed to create a generation
     <base>.users  held shared for its whole life by every process serving a
                   generation
   Reclaiming the segments requires holding both exclusively at once, which is
   only possible when no creator and no follower is alive. A short-lived
   <base>.decide mutex serialises the decision itself, so the users lock can be
   tested and dropped without a race.
   An earlier version had the creator hold one lock exclusively and followers
   take that same lock shared -- which flock refuses. The followers fell
   through to a lockless join, and the next creator went on to unlink a
   generation three live instances were still reading.

3. A generation has to be reclaimed, and only one process may ever do it.
   Holding both locks proves there are no participants left, so the creator
   unlinks the previous generation's segments -- listed in a manifest -- before
   creating its own. Without this the orphaned pages stay until reboot.
   A named segment also outlives its readers, so the participant that leaves
   last runs the same check on its way out and unlinks the generation there,
   rather than leaving the whole component to a next start that may never come.
   A process that is killed reaches no exit hook and leaves the generation
   behind; the next creator reclaims it.

   create    the creator allocates each segment, fills it, registers it, then
             publishes its marker.
   attach    a follower waits for the marker before opening the segment, so it
             can never read a half-filled store, and waits for one the creator
             has not reached yet instead of allocating a second copy.
   verify    the follower compares the store against the weights *it* loaded
             from disk. The key only has to be structural, because a store
             holding different bytes is rejected rather than trusted.
   miss      the creator stopped advancing, or the bytes differ: the follower
             closes the segment and its caller allocates privately.
"""

import ctypes
import fcntl
import hashlib
import logging
import os
import time
from multiprocessing import resource_tracker, shared_memory, util

import torch

logger = logging.getLogger(__name__)

MARKER_DIR = "/tmp/sglang-shared-stores"

# A creator that is alive but slow -- CPU-starved, or faulting its first layer
# off a cold network mount -- keeps advancing, so followers keep waiting. Only
# a creator with no progress for this long is given up on.
STALL_TIMEOUT_S = 90.0
MAX_WAIT_S = 900.0

# The decision covers unlinking a whole dead generation, which is fast; a peer
# stuck here is a peer that will never finish.
DECIDE_TIMEOUT_S = 120.0

_LIVE: dict[str, shared_memory.SharedMemory] = {}
_LOCKS: dict[str, list[int]] = {}
_ROLE: dict[str, bool | None] = {}
_FILLED: dict[str, int] = {}
_STATS: dict[str, dict[str, int]] = {}
_LEAVING: set[str] = set()


class SharedStore:
    """One acquired segment, not yet filled or registered."""

    __slots__ = ("shm", "created", "key", "nbytes", "pool")

    def __init__(self, shm, created: bool, key: str, nbytes: int, pool: str) -> None:
        self.shm = shm
        self.created = created
        self.key = key
        self.nbytes = nbytes
        self.pool = pool

    @property
    def buffer(self):
        return self.shm.buf


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
    """Decide whether this process creates the pool's generation or joins it.

    True means create, False means attach to what is there, and None means the
    pool cannot be joined at all: the caller then allocates privately, which is
    always correct but costs the memory the pool exists to save.
    """
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


def _note_leave(pool: str) -> None:
    """Have `leave_pool` run when this process exits normally.

    `atexit` is not enough: a process serving here is a multiprocessing child,
    and `BaseProcess._bootstrap` leaves through `os._exit`, which skips every
    `atexit` hook. `util.Finalize` is what that exit path does run. A process
    killed by a signal runs neither, which is why the next creator still has to
    be able to reclaim.
    """
    if pool in _LEAVING:
        return
    _LEAVING.add(pool)
    util.Finalize(None, leave_pool, args=(pool,), exitpriority=10)


def leave_pool(pool: str) -> bool:
    """Give up this process's seat, reclaiming the generation if it was the last.

    A name in /dev/shm outlives its readers -- the kernel frees an anonymous
    memfd when its last holder exits, but a named segment stays until something
    alive unlinks it -- so the participant that leaves last does it, on the same
    proof the creator uses: hold `<base>.owner` and `<base>.users` exclusively
    at once, which no live participant can be holding.

    This runs under `<base>.decide`, the mutex `join_pool` decides under, and
    that is what makes releasing our own seat safe: the seat has to go before
    the exclusive test can mean anything, and without the mutex a process
    starting inside that window would take the pool for empty and the segments
    would be pulled from under readers it never saw.

    Returns whether a generation was reclaimed.
    """
    held = _LOCKS.pop(pool, None)
    _ROLE.pop(pool, None)
    if held is None:
        return False

    owner_path, users_path, decide_path, _, _ = _pool_paths(pool)
    decide_fd = os.open(decide_path, os.O_CREAT | os.O_RDWR, 0o600)
    opened = [decide_fd]
    reclaimed = False
    try:
        if not _decide_lock(decide_fd):
            return False
        for fd in held:
            try:
                os.close(fd)
            except OSError:
                pass
        owner_fd = os.open(owner_path, os.O_CREAT | os.O_RDWR, 0o600)
        users_fd = os.open(users_path, os.O_CREAT | os.O_RDWR, 0o600)
        opened += [owner_fd, users_fd]
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
        for fd in opened:
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
    tmp = f"{progress}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as handle:
            handle.write(str(_FILLED.get(pool, 0)))
        os.replace(tmp, progress)
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
            stalled = now - os.path.getmtime(progress) > STALL_TIMEOUT_S
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

    if creator:
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
            return None
        try:
            shm = shared_memory.SharedMemory(name=name)
        except (FileNotFoundError, OSError):
            stats["refused"] += 1
            return None

    # Every SharedMemory construction registers the name with this process's
    # resource_tracker, attaching as much as creating, and the tracker unlinks
    # whatever it registered when this process exits. That is not this store's
    # lifetime -- the pool lock is -- so every attach has to let go of it, or
    # one instance leaving would pull the segment out from under the others.
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass

    if creator:
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

    # A store that cannot be confirmed is never used; refusing costs a private
    # allocation, trusting it would cost wrong weights.
    try:
        if not verify(shm.buf):
            logger.warning(
                "shared pinned store %s holds different weights than this "
                "process loaded; falling back to a private allocation",
                pool,
            )
            shm.close()
            stats["refused"] += 1
            return None
    except Exception as exc:
        logger.warning("shared pinned store %s could not be verified (%s)", pool, exc)
        shm.close()
        stats["refused"] += 1
        return None
    return SharedStore(shm, False, key, nbytes, pool)


def commit(store: SharedStore) -> torch.Tensor | None:
    """Register the store and, for its creator, publish it. None means fall back."""
    if not _register(store.buffer, store.nbytes):
        store.shm.close()
        if store.created:
            store.shm.unlink()
        return None
    name, marker = _names(store.key)
    if store.created:
        with open(marker, "w") as handle:
            handle.write(name)
        _FILLED[store.pool] = _FILLED.get(store.pool, 0) + 1
        _publish_progress(store.pool)
    _LIVE[name] = store.shm
    return torch.frombuffer(store.buffer, dtype=torch.uint8, count=store.nbytes)

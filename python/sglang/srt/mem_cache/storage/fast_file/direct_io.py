# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""O_DIRECT support probe for file-based HiCache storage backends.

Linux accepts O_DIRECT only when buffer addresses, lengths and file offsets
are multiples of a granularity the filesystem and the device decide: 512 B or
4 KiB on disks, the sector size on btrfs. The backend uses 4 KiB, a multiple
of all of them, and the probe only confirms that a 4 KiB direct write and
read-back work under the target directory. It does not look for a smaller
granularity: btrfs serves misaligned O_DIRECT requests through the page cache
instead of failing them, so an accepted smaller request proves nothing.
Filesystems without direct I/O (ramfs, tmpfs before Linux 6.6, some network
filesystems) reject the flag and disable the mode. btrfs accepts it but, for
files with data checksums (its default), writes through the page cache by
design; only reads bypass it there.

The probe file never gets a name where the filesystem supports O_TMPFILE, so
nothing can be left behind, not even when the process dies mid-probe. Elsewhere
a named file is created and unlinked at once.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import mmap
import os
import uuid

logger = logging.getLogger(__name__)

# O_DIRECT and O_TMPFILE are Linux-only; 0 leaves open flags unchanged elsewhere.
O_DIRECT = os.O_DIRECT if hasattr(os, "O_DIRECT") else 0
_O_TMPFILE = os.O_TMPFILE if hasattr(os, "O_TMPFILE") else 0

# Multiple of every direct-I/O granularity Linux filesystems and devices
# require on 4 KiB-page systems; the same bound HiCacheNixl uses.
DIRECT_IO_ALIGNMENT = 4096


def _open_probe_file(directory: str) -> int:
    """An fd on a file under ``directory`` that has no name, or is unlinked at once."""
    if _O_TMPFILE:
        try:
            return os.open(directory, os.O_RDWR | _O_TMPFILE, 0o600)
        except OSError as exc:
            # EOPNOTSUPP: no tmpfile operation; EISDIR: a kernel before 3.11.
            if exc.errno not in (errno.EOPNOTSUPP, errno.EISDIR):
                raise
    path = os.path.join(directory, f".o_direct_probe.{os.getpid()}.{uuid.uuid4().hex}")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.unlink(path)
    except OSError:
        os.close(fd)
        raise
    return fd


def probe_direct_io(directory: str) -> bool:
    """True when 4 KiB O_DIRECT writes and reads work for files under ``directory``."""
    if not O_DIRECT:
        return False
    try:
        fd = _open_probe_file(directory)
    except OSError as exc:
        logger.info("O_DIRECT probe could not open a file under %s: %s", directory, exc)
        return False
    try:
        # Set after the file exists: open(O_CREAT | O_DIRECT) creates the file
        # before the kernel rejects the flag and would leave it behind.
        try:
            fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | O_DIRECT)
        except OSError as exc:
            logger.info("O_DIRECT is unsupported under %s: %s", directory, exc)
            return False
        # Write from one anonymous page and read back into another: both are
        # page-aligned like a pinned host pool. A bytes object is not, and
        # queues with a 512-byte DMA alignment (virtio-blk) reject it.
        pages = memoryview(mmap.mmap(-1, 2 * DIRECT_IO_ALIGNMENT))
        data, readback = pages[:DIRECT_IO_ALIGNMENT], pages[DIRECT_IO_ALIGNMENT:]
        data[:] = os.urandom(DIRECT_IO_ALIGNMENT)
        # Fault the read-back page in: btrfs serves direct reads into pages that
        # are not mapped yet through the page cache.
        readback[:] = bytes(DIRECT_IO_ALIGNMENT)
        if os.pwrite(fd, data, 0) != DIRECT_IO_ALIGNMENT:
            logger.info("O_DIRECT probe short write under %s", directory)
            return False
        if os.preadv(fd, [readback], 0) != DIRECT_IO_ALIGNMENT or readback != data:
            logger.warning(
                "O_DIRECT probe read back different bytes under %s", directory
            )
            return False
        return True
    except OSError as exc:
        logger.info("O_DIRECT probe failed under %s: %s", directory, exc)
        return False
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""O_DIRECT support probing for file-based HiCache storage backends.

Linux accepts O_DIRECT only when every I/O segment address and length, and the
file offset, are multiples of an alignment that the filesystem and the block
device decide: 512 bytes on most disks, 4096 on 4Kn disks. The probe measures
it the way QEMU does for its image files: write and read back one page through
O_DIRECT on an anonymous file in the target directory, then try smaller
granularities until the kernel stops answering EINVAL. Filesystems without
direct I/O (tmpfs before Linux 6.6, some network filesystems) fail the open or
the write and disable the mode. No kernel or libc version is assumed.
"""

from __future__ import annotations

import errno
import logging
import mmap
import os
import uuid

logger = logging.getLogger(__name__)

# O_DIRECT and O_TMPFILE are Linux-only; 0 leaves open flags unchanged elsewhere.
O_DIRECT = os.O_DIRECT if hasattr(os, "O_DIRECT") else 0
_O_TMPFILE = os.O_TMPFILE if hasattr(os, "O_TMPFILE") else 0

# The write/read-back step verifies this granularity; the smaller ones are
# then tried in order and the first the kernel accepts wins.
MAX_ALIGNMENT = 4096
_SMALLER_ALIGNMENTS = (512, 1024, 2048)


def _open_anonymous_direct(directory: str) -> int:
    """An fd on an unnamed O_DIRECT file in ``directory``; OSError if refused."""
    if _O_TMPFILE:
        try:
            return os.open(directory, os.O_RDWR | O_DIRECT | _O_TMPFILE, 0o600)
        except OSError as exc:
            # The filesystem may lack O_TMPFILE while still supporting O_DIRECT.
            if exc.errno not in (errno.EOPNOTSUPP, errno.EISDIR, errno.EINVAL):
                raise
    path = os.path.join(directory, f".o_direct_probe.{os.getpid()}.{uuid.uuid4().hex}")
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | O_DIRECT, 0o600)
    os.unlink(path)
    return fd


def probe_direct_io_alignment(directory: str) -> int:
    """The O_DIRECT alignment for files under ``directory``; 0 if unsupported."""
    if not O_DIRECT:
        return 0
    try:
        fd = _open_anonymous_direct(directory)
    except OSError as exc:
        logger.info("O_DIRECT is unsupported under %s: %s", directory, exc)
        return 0
    try:
        # An anonymous mapping is page-aligned, like a pinned host pool.
        buf = memoryview(mmap.mmap(-1, 2 * MAX_ALIGNMENT))
        buf[:MAX_ALIGNMENT] = os.urandom(MAX_ALIGNMENT)
        if os.pwrite(fd, buf[:MAX_ALIGNMENT], 0) != MAX_ALIGNMENT:
            logger.info("O_DIRECT probe short write under %s", directory)
            return 0
        if os.pread(fd, MAX_ALIGNMENT, 0) != bytes(buf[:MAX_ALIGNMENT]):
            logger.warning(
                "O_DIRECT probe read back different bytes under %s", directory
            )
            return 0
        for alignment in _SMALLER_ALIGNMENTS:
            # Buffer address, file offset and length all at this granularity.
            window = buf[MAX_ALIGNMENT + alignment : MAX_ALIGNMENT + 2 * alignment]
            try:
                if os.preadv(fd, [window], alignment) == alignment:
                    return alignment
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
        return MAX_ALIGNMENT
    except OSError as exc:
        logger.info("O_DIRECT probe failed under %s: %s", directory, exc)
        return 0
    finally:
        os.close(fd)

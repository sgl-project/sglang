import errno
import mmap
import os
import sys

import pytest

from sglang.srt.utils import memfd
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def without_os_memfd(monkeypatch):
    monkeypatch.delattr(os, "memfd_create", raising=False)


def test_fallback_returns_a_mappable_memfd_with_os_inheritance(without_os_memfd):
    """Without os.memfd_create (uv-managed CPython) callers still get a real, mappable
    memfd whose inheritability follows MFD_CLOEXEC as os.memfd_create's does."""
    fd = memfd.memfd_create("sglang-test", memfd.MFD_CLOEXEC)
    inheritable_fd = memfd.memfd_create("sglang-test-inheritable", 0)
    try:
        assert os.readlink(f"/proc/self/fd/{fd}").startswith("/memfd:sglang-test")
        os.ftruncate(fd, mmap.PAGESIZE)
        with mmap.mmap(fd, mmap.PAGESIZE) as mapping:
            mapping[:4] = b"abcd"
        assert os.pread(fd, 4, 0) == b"abcd"
        assert not os.get_inheritable(fd)
        assert os.get_inheritable(inheritable_fd)
    finally:
        os.close(fd)
        os.close(inheritable_fd)


def test_fallback_raises_oserror_on_failure(without_os_memfd):
    """Callers switch to another backing store on OSError (e.g. memfd blocked by
    seccomp), so a failed libc call must raise it instead of returning -1."""
    with pytest.raises(OSError) as exc_info:
        memfd.memfd_create("sglang-test", 0xFFFF)
    assert exc_info.value.errno == errno.EINVAL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

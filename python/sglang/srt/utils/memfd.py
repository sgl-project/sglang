import ctypes
import functools
import os

MFD_CLOEXEC = 1


@functools.cache
def _libc_memfd_create():
    func = ctypes.CDLL(None, use_errno=True).memfd_create
    func.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    func.restype = ctypes.c_int
    return func


def memfd_create(name: str, flags: int = MFD_CLOEXEC) -> int:
    """python-build-standalone (uv-managed Python) targets glibc 2.17, which predates
    memfd_create (2.27), so call the runtime glibc when os lacks it; torch needs 2.28."""
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)
    fd = _libc_memfd_create()(os.fsencode(name), flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd

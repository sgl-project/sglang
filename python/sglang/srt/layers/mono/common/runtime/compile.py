# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/compile.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Building a FlyDSL launcher without launching it, what its binary asks of the
GPU, and its hand-offs traced while it builds (``plan.check`` holds them to the
contract)."""

import os
import re
from contextlib import contextmanager

from sglang.srt.layers.mono.common.plan.check import RegionDecl, check
from sglang.srt.layers.mono.common.plan.trace import Record, recording

# FlyDSL's compile-only switch (flydsl.utils.env CompileEnvManager.compile_only),
# read at every call: a launcher called under it compiles, keeps the binary in its
# in-memory cache and returns without launching
_COMPILE_ONLY = "COMPILE_ONLY"

# the code object's kernel metadata (msgpack): a key, then its value. The
# spill counts are absent from older code objects
_RESOURCE_KEYS = {
    "vgpr": b".vgpr_count",
    "sgpr": b".sgpr_count",
    "scratch": b".private_segment_fixed_size",
    "lds": b".group_segment_fixed_size",
    "vgpr_spill": b".vgpr_spill_count",
    "sgpr_spill": b".sgpr_spill_count",
}
_OPTIONAL = ("vgpr_spill", "sgpr_spill")
_SYMBOL_KEY = b".symbol"
_BIN = re.compile(r'bin\s*=\s*"')
_MANGLED = re.compile(r"_ZN((?:\d+\w+?)+)E")


@contextmanager
def compile_only():
    """Launcher calls inside this block compile and do not run."""
    before = os.environ.get(_COMPILE_ONLY)
    os.environ[_COMPILE_ONLY] = "1"
    try:
        yield
    finally:
        if before is None:
            del os.environ[_COMPILE_ONLY]
        else:
            os.environ[_COMPILE_ONLY] = before


def compile_kernels(kernels) -> list:
    """Each ``(launcher, abi)`` of ``kernels`` compiled without a launch (its
    arguments ``abi.zeros()``); returns the launchers."""
    with compile_only():
        for launcher, abi in kernels:
            launcher(*abi.zeros())
    return [launcher for launcher, _ in kernels]


def trace(launcher, abi) -> Record:
    """``launcher``'s hand-offs, traced compile-only (no launch, no GPU needed
    with ``ARCH`` set). A build served from FlyDSL's disk cache is not traced,
    so the caller disables the cache (``FLYDSL_RUNTIME_ENABLE_CACHE=0``)."""
    with recording() as rec:
        compile_kernels([(launcher, abi)])
    if not rec.stages:
        raise RuntimeError(
            "nothing traced: a cached build? set FLYDSL_RUNTIME_ENABLE_CACHE=0"
        )
    return rec


def check_traced(launcher, abi, decls: list[RegionDecl]) -> None:
    """``trace`` ``launcher`` and ``check`` its hand-offs against ``decls``;
    raises ``plan.check.ContractError``."""
    check(trace(launcher, abi), decls)


def scratch_bytes(launcher) -> int:
    """A compiled ``launcher``'s scratch (private segment) bytes a lane, the most
    of its binaries: the code objects in FlyDSL's in-memory cache of the
    ``@flyc.jit`` function (``_mem_cache``, each artifact's module text
    ``_ir_text``: FlyDSL keeps no public handle on them). Raises if there is no
    binary to read: an unread size must not pass for none."""
    sizes = [
        _note_uint(_code_object(artifact._ir_text), _RESOURCE_KEYS["scratch"])
        for artifact in launcher._mem_cache.values()
    ]
    if not sizes:
        raise RuntimeError(f"{launcher!r}: no compiled binary to read")
    return max(sizes)


def launcher_resources(launcher) -> list[dict]:
    """``kernel_resources`` of each binary a compiled ``launcher`` holds (its
    ``_mem_cache``): what this process runs, unlike FlyDSL's disk cache, whose
    entries do not say which source tree built them."""
    return [
        kernel_resources(_code_object(artifact._ir_text))
        for artifact in launcher._mem_cache.values()
    ]


def _code_object(module_text: str) -> bytes:
    """The ``gpu.binary``'s ELF: the ``bin = "..."`` string, MLIR-escaped (\\XX a
    byte, \\\\ and \\" themselves)."""
    m = _BIN.search(module_text)
    if m is None:
        raise RuntimeError("no gpu.binary object in the compiled module")
    out, i = bytearray(), m.end()
    while module_text[i] != '"':
        ch = module_text[i]
        if ch != "\\":
            out.append(ord(ch))
            i += 1
        elif module_text[i + 1] in '\\"':
            out.append(ord(module_text[i + 1]))
            i += 2
        else:
            out.append(int(module_text[i + 1 : i + 3], 16))
            i += 3
    return bytes(out)


def kernel_resources(elf: bytes) -> dict:
    """What a code object's kernel asks of the GPU, from its metadata: ``name``
    (demangled when it is a plain nested name), ``vgpr``, ``sgpr``, ``scratch``
    (private segment bytes a lane), ``lds`` (bytes), and ``vgpr_spill`` /
    ``sgpr_spill`` (None when the code object predates them)."""
    out = {"name": _symbol(elf)}
    for field, key in _RESOURCE_KEYS.items():
        out[field] = _note_uint(elf, key, optional=field in _OPTIONAL)
    return out


def _note_uint(elf: bytes, key: bytes, optional: bool = False) -> int | None:
    """The kernel metadata's unsigned value under ``key`` (msgpack)."""
    at = elf.find(key)
    if at < 0:
        if optional:
            return None
        raise RuntimeError(f"no {key.decode()} in the code object")
    v = elf[at + len(key) :]
    if v[0] < 0x80:  # positive fixint
        return v[0]
    width = {0xCC: 1, 0xCD: 2, 0xCE: 4, 0xCF: 8}.get(v[0])
    if width is None:
        raise RuntimeError(f"unexpected msgpack tag {v[0]:#x} for {key.decode()}")
    return int.from_bytes(v[1 : 1 + width], "big")


def _symbol(elf: bytes) -> str:
    """The kernel's name from its ``.symbol`` (a msgpack str, ``<name>.kd``)."""
    at = elf.find(_SYMBOL_KEY)
    if at < 0:
        raise RuntimeError("no .symbol in the code object")
    v = elf[at + len(_SYMBOL_KEY) :]
    if 0xA0 <= v[0] <= 0xBF:  # fixstr
        n, v = v[0] & 0x1F, v[1:]
    elif v[0] == 0xD9:  # str8
        n, v = v[1], v[2:]
    elif v[0] == 0xDA:  # str16
        n, v = int.from_bytes(v[1:3], "big"), v[3:]
    else:
        raise RuntimeError(f"unexpected msgpack tag {v[0]:#x} for .symbol")
    name = v[:n].decode().removesuffix(".kd")
    m = _MANGLED.fullmatch(name)
    if m is None:
        return name
    parts, rest = [], m.group(1)
    while rest:
        digits = re.match(r"\d+", rest).group()
        size = int(digits)
        parts.append(rest[len(digits) : len(digits) + size])
        rest = rest[len(digits) + size :]
    return "::".join(parts)

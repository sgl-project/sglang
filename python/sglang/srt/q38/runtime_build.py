"""CPU-only NVRTC helpers for architecture-specific runtime bundles."""

import ctypes
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(root: Path, value: dict) -> None:
    (root / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")


def compile_cuda(source: Path, target: Path, flags: list[str], cuda_root: Path) -> None:
    libraries = [cuda_root / name / "libnvrtc.so.13" for name in ("lib64", "lib")]
    library = next((path for path in libraries if path.is_file()), None)
    if library is None:
        raise FileNotFoundError(f"CUDA 13 NVRTC is missing under {cuda_root}")
    lib = ctypes.CDLL(str(library))
    program = ctypes.c_void_p()

    def check(code: int) -> None:
        if code:
            raise RuntimeError(f"NVRTC error {code}: {source}")

    check(
        lib.nvrtcCreateProgram(
            ctypes.byref(program),
            source.read_bytes(),
            source.name.encode(),
            0,
            None,
            None,
        )
    )
    try:
        options = [
            *flags,
            f"-I{cuda_root / 'include'}",
            f"-I{cuda_root / 'include/cccl'}",
        ]
        encoded = (ctypes.c_char_p * len(options))(
            *(option.encode() for option in options)
        )
        code = lib.nvrtcCompileProgram(program, len(options), encoded)
        size = ctypes.c_size_t()
        check(lib.nvrtcGetProgramLogSize(program, ctypes.byref(size)))
        log = ctypes.create_string_buffer(size.value)
        check(lib.nvrtcGetProgramLog(program, log))
        target.with_suffix(".log").write_bytes(log.value)
        if code:
            raise RuntimeError(log.value.decode(errors="replace"))
        check(lib.nvrtcGetCUBINSize(program, ctypes.byref(size)))
        binary = ctypes.create_string_buffer(size.value)
        check(lib.nvrtcGetCUBIN(program, binary))
        target.write_bytes(binary.raw)
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(program))

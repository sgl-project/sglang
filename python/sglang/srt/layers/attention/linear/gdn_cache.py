"""SGLang-owned persistence for FlashInfer's GDN process-local compile cache.

Only used around startup preparation. FlashInfer source and functions remain
unchanged. CuTe's export/load APIs persist the callable; the only private
FlashInfer dependency is the explicit-key dictionary in custom_compile_cache.
"""

from __future__ import annotations

import enum
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
_SCHEMA = 1


def encode_key(value):
    """Portable immutable keys, without pickle or address-based reprs."""
    if isinstance(value, enum.Enum):
        return ["enum", encode_key(type(value)), value.name]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, type):
        return ["type", value.__module__, value.__qualname__]
    if isinstance(value, torch.dtype):
        return ["dtype", str(value).removeprefix("torch.")]
    if isinstance(value, tuple):
        return ["tuple", [encode_key(item) for item in value]]
    raise TypeError(f"Unsupported GDN cache key value: {type(value)}")


def decode_key(value):
    """Resolve types already loaded by FlashInfer; never import from a manifest."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError("Invalid GDN cache key")
    tag = value[0]
    if tag == "tuple" and len(value) == 2:
        return tuple(decode_key(item) for item in value[1])
    if tag == "dtype" and len(value) == 2:
        result = getattr(torch, value[1], None)
        if isinstance(result, torch.dtype):
            return result
    if tag == "type" and len(value) == 3:
        module, name = value[1:]
        if module not in ("builtins", "torch") and not module.startswith(
            ("flashinfer.gdn_kernels.", "cutlass.")
        ):
            raise ValueError(f"Unsupported GDN key module: {module}")
        result = sys.modules.get(module)
        for part in name.split("."):
            if not part.isidentifier():
                raise ValueError("Invalid GDN key type name")
            result = getattr(result, part, None)
        if isinstance(result, type):
            return result
    if tag == "enum" and len(value) == 3:
        cls = decode_key(value[1])
        if isinstance(cls, type) and issubclass(cls, enum.Enum):
            return cls[value[2]]
    raise ValueError("Unsupported or unavailable GDN cache key")


def key_digest(encoded):
    return hashlib.sha256(
        json.dumps(encoded, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def eligible_key(key):
    """Recognize FlashInfer 0.7.0.post1's GDN explicit TVM-FFI key contract."""
    try:
        compile_key, options = key
        mro = compile_key[0]
        return (
            isinstance(mro, tuple)
            and isinstance(mro[0], type)
            and mro[0].__module__.startswith("flashinfer.gdn_kernels.")
            and any(
                isinstance(cls, type)
                and cls.__module__.startswith("cutlass.")
                and cls.__name__ == "EnableTVMFFI"
                and enabled is True
                for cls, enabled in (options or ())
            )
        )
    except (TypeError, ValueError, IndexError):
        return False


@functools.lru_cache(maxsize=8)
def cache_identity(source_dir: str, arch: tuple[int, int], arch_override: str):
    """Version/source changes get a separate directory, never stale callables."""
    versions = {
        name: importlib.metadata.version(name)
        for name in ("flashinfer-python", "torch", "apache-tvm-ffi", "cuda-bindings")
    }
    versions["cutlass_stack"] = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions()
        if (dist.metadata["Name"] or "").lower().startswith("nvidia-cutlass-dsl")
    )
    versions.update(
        schema=_SCHEMA,
        python_abi=sys.implementation.cache_tag,
        arch=arch,
        arch_override=arch_override,
    )
    sources = sorted(Path(source_dir).rglob("*.py")) + [Path(__file__)]
    fingerprint = hashlib.sha256(json.dumps(versions, sort_keys=True).encode())
    for path in sources:
        fingerprint.update(str(path).encode())
        fingerprint.update(path.read_bytes())
    return fingerprint.hexdigest()


class GDNCache:
    def __init__(self, directory, identity, memory_cache, load_module):
        self.directory = Path(directory) / identity
        self.identity = identity
        self.memory_cache = memory_cache
        self.load_module = load_module
        self.manifest = self.directory / "manifest.json"
        self.failed_entries = set()

    def _read(self):
        try:
            data = json.loads(self.manifest.read_text())
            if data["schema"] == _SCHEMA and data["identity"] == self.identity:
                return data["entries"] if isinstance(data["entries"], dict) else {}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {}

    def restore(self):
        loaded = 0
        for digest, encoded in self._read().items():
            try:
                if key_digest(encoded) != digest:
                    raise ValueError("GDN key digest mismatch")
                key = decode_key(encoded)
                if not eligible_key(key) or key in self.memory_cache:
                    continue
                module = self.load_module(
                    str(self.directory / f"{digest}.o"), enable_tvm_ffi=True
                )
                self.memory_cache[key] = getattr(module, f"sglang_gdn_{digest}")
                loaded += 1
            except Exception as error:  # noqa: BLE001 - optional cache must fail open
                self.failed_entries.add(digest)
                logger.warning("Ignoring GDN disk cache entry %s: %s", digest, error)
        if loaded:
            logger.info(
                "Restored %d GDN compiled kernels from %s", loaded, self.directory
            )
        return loaded

    def persist(self):
        from filelock import FileLock

        self.directory.mkdir(parents=True, exist_ok=True)
        saved = 0
        with FileLock(str(self.directory / "cache.lock")):
            entries = self._read()
            for key, compiled in tuple(self.memory_cache.items()):
                if not eligible_key(key):
                    continue
                temporary = None
                try:
                    encoded = encode_key(key)
                    digest = key_digest(encoded)
                    target = self.directory / f"{digest}.o"
                    if (
                        entries.get(digest) == encoded
                        and target.is_file()
                        and digest not in self.failed_entries
                    ):
                        continue
                    # Only successfully compiled/validated FlashInfer callables
                    # reach its cache, preserving the original SM12x PTX check.
                    temporary = target.with_suffix(f".o.tmp.{os.getpid()}")
                    compiled.export_to_c(
                        str(temporary), function_name=f"sglang_gdn_{digest}"
                    )
                    os.replace(temporary, target)
                    entries[digest] = encoded
                    saved += 1
                except Exception as error:  # noqa: BLE001 - optional cache must fail open
                    logger.warning("Unable to persist GDN cache entry: %s", error)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            if saved:
                temporary = self.manifest.with_suffix(f".json.tmp.{os.getpid()}")
                try:
                    temporary.write_text(
                        json.dumps(
                            {
                                "schema": _SCHEMA,
                                "identity": self.identity,
                                "entries": entries,
                            },
                            indent=2,
                        )
                    )
                    os.replace(temporary, self.manifest)
                finally:
                    temporary.unlink(missing_ok=True)
        if saved:
            logger.info(
                "Persisted %d GDN compiled kernels to %s", saved, self.directory
            )
        return saved


@contextmanager
def gdn_prefill_cache():
    """Restore before warmup, export afterwards; never replace FI functions."""
    if os.getenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "0") == "1":
        yield
        return
    try:
        from cutlass import cute
        from flashinfer.gdn_kernels.delta_rule_dsl import custom_compile_cache
        from sglang.srt.environ import envs

        memory = custom_compile_cache._in_mem_compile_cache
        if not isinstance(memory, dict):
            raise TypeError("FlashInfer GDN compile cache contract changed")
        identity = cache_identity(
            str(Path(custom_compile_cache.__file__).resolve().parent.parent),
            torch.cuda.get_device_capability(),
            os.getenv("CUTE_DSL_ARCH", ""),
        )
        cache = GDNCache(
            Path(envs.SGLANG_CACHE_DIR.get()).expanduser() / "gdn_prefill",
            identity,
            memory,
            cute.runtime.load_module,
        )
        cache.restore()
    except Exception as error:  # noqa: BLE001 - optional cache must fail open
        logger.warning("GDN startup disk cache unavailable: %s", error)
        yield
        return
    yield
    try:
        cache.persist()
    except Exception as error:  # noqa: BLE001 - optional cache must fail open
        logger.warning("GDN startup disk cache export failed: %s", error)

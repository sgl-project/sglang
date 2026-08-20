# Vendored from flashinfer-ai/flashinfer main at 76704c4 (SM100 GDN CP
# prefill closure, incl. #4436 pooled state / checkpointing / dtype parity);
# pending a FlashInfer release that ships it.
import functools
import os
import tempfile
import warnings
from collections.abc import Hashable
from functools import lru_cache

import cutlass.cute as cute
import torch
from cutlass.base_dsl.compiler import DumpDir

_in_mem_compile_cache: dict = {}


@lru_cache(maxsize=8)
def _sm12x_gpu_arch_cached(device_type: str, device_index: int | None) -> cute.GPUArch:
    device = (
        torch.device(device_type)
        if device_index is None
        else torch.device(device_type, device_index)
    )
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12:
        raise RuntimeError(
            f"sm12x_gpu_arch requires an SM12x device, got compute capability "
            f"{major}.{minor}"
        )
    return cute.GPUArch(f"sm_{major}{minor}a")


def sm12x_gpu_arch(device: str | torch.device = "cuda") -> cute.GPUArch:
    """Return the on-device CuteDSL arch for SM12x (``sm_120a`` / ``sm_121a``)."""
    d = torch.device(device)
    if d.type == "cuda" and d.index is None:
        d = torch.device("cuda", torch.cuda.current_device())
    return _sm12x_gpu_arch_cached(d.type, d.index)


def sm12x_compile_options(
    device: str | torch.device = "cuda",
) -> tuple[cute.GPUArch]:
    return (sm12x_gpu_arch(device),)


_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)


def _as_options_tuple(options):
    if options is None:
        return ()
    if isinstance(options, tuple):
        return options
    return (options,)


class KeyedCompileMixin:
    def manual_cache_key(self, *attr_names):
        collected_attrs = tuple(
            (attr_name, getattr(self, attr_name)) for attr_name in attr_names
        )
        compile_key = (type(self).__mro__,) + collected_attrs
        hash(compile_key)
        setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            warnings.warn(
                f"{type(self).__name__} is using automatic DSL compile-cache key generation; "
                "call manual_cache_key(...) at the end of __init__ to avoid host launch overhead.",
                RuntimeWarning,
                stacklevel=2,
            )
            collected_attrs = tuple(
                (attr_name, attr_value)
                for attr_name, attr_value in sorted(self.__dict__.items())
                if not attr_name.startswith("_")
            )
            compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            try:
                hash(compile_key)
            except TypeError:
                collected_attrs = tuple(
                    (attr_name, attr_value)
                    for attr_name, attr_value in collected_attrs
                    if isinstance(attr_value, Hashable)
                )
                compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            setattr(self, "_KeyedCompileMixin_compile_key", compile_key)  # noqa: B010

        return compile_key


@functools.cache
def _compile_options_tuple_key(options):
    return tuple((type(option), option.value) for option in options)


def _compile_options_key(options):
    if options is None:
        return None
    return _compile_options_tuple_key(_as_options_tuple(options))


def _option_value(options, option_type):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type):
            return option.value
    return None


def _has_option_value(options, option_type, option_value):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type) and option.value == option_value:
            return True
    return False


def _needs_sm12x_ptx_check(options):
    # Arch-specific SM12x targets share the same cluster-TMA restriction.
    return _has_option_value(options, cute.GPUArch, "sm_120a") or _has_option_value(
        options, cute.GPUArch, "sm_121a"
    )


def _ptx_check_compile_options(options, dump_dir):
    options = _as_options_tuple(options)

    has_keep_ptx = _has_option_value(options, cute.KeepPTX, True)
    has_dump_dir = _option_value(options, DumpDir)
    extras = []
    if not has_keep_ptx:
        extras.append(cute.KeepPTX(True))
    if has_dump_dir:
        dump_dir = _option_value(options, DumpDir)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
    else:
        extras.append(DumpDir(dump_dir))
    return options + tuple(extras)


def _read_ptx_text(ptx_artifact):
    if isinstance(ptx_artifact, str) and os.path.exists(ptx_artifact):
        with open(ptx_artifact, "r", encoding="utf-8") as f:
            return f.read()
    return ptx_artifact


def _sm12x_ptx_check(compiled_fn):
    ptx = _read_ptx_text(getattr(compiled_fn, "__ptx__", None))
    if not ptx:
        raise RuntimeError("Unable to inspect generated PTX for the SM12x GDN kernel")
    if _TMA_CLUSTER_LOAD in ptx:
        raise RuntimeError(
            "SM12x GDN kernel compilation produced unsupported cluster-scoped TMA loads. "
            "Install the CUDA 13 CUTLASS DSL compiler with "
            "`pip install --upgrade --force-reinstall 'nvidia-cutlass-dsl[cu13]'`. "
            "See https://github.com/NVIDIA/cutlass/issues/3170."
        )


def cached_compile(func, *args, compile_options=None, **kwargs):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    compiled_fn = _in_mem_compile_cache.get(cache_key)

    if compiled_fn is None:
        _needs_ptx_check = _needs_sm12x_ptx_check(compile_options)
        if not _needs_ptx_check:
            compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
        else:
            with tempfile.TemporaryDirectory(prefix="cutedsl_ptx_check_") as tempdir:
                compile_options = _ptx_check_compile_options(compile_options, tempdir)
                compiled_fn = cute.compile[compile_options](func, *args, **kwargs)
                _sm12x_ptx_check(compiled_fn)

        _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn


def get_cached_compile(func, compile_options=None):
    cache_key = (func._get_compile_key(), _compile_options_key(compile_options))
    return _in_mem_compile_cache.get(cache_key)

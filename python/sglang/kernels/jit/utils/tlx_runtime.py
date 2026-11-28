"""Runtime support for the vendored TLX kernels (``sglang.kernels.ops.<group>.tlx_*``) on stock ROCm Triton.

The kernels import ``triton.language.extra.tlx``. facebookexperimental/triton
("fbtriton" below) ships it. On stock ROCm Triton, the uTLX plugin built for that
Triton release registers it at interpreter start through its ``.pth`` hook.
Verified on MI355X:

    torch 2.14.1+rocm7.2, triton-rocm 3.8.0:   pip install --no-deps triton-utlx==3.8.0.post16
    torch 2.15.0+rocm7.14, triton-rocm 3.9.0:  pip install --no-deps triton-utlx==3.9.0.post4

``--no-deps`` because those wheels declare the CUDA ``triton`` package.
``triton-utlx-rocm`` builds do not work with PyTorch's ``triton-rocm``: 3.8.0.post1
fails at the first compile on a missing ``mlir::argSort``. With torch 2.15 and
``triton-utlx 3.9.0.post4``, loading uTLX before torch aborts with ``LLVM ERROR:
Option 'default' already exists!``; SGLang always imports torch first, and standalone
scripts must too.

uTLX adds fbtriton's extra ``HIPOptions`` codegen fields (``reverse_local_assignment``
and friends) but forwards their LLVM flags only through ``llvm.translate_*``, which
triton-rocm 3.9's ``make_amdgcn`` no longer calls, and it has no ``llvm_cl_opts``.
``add_missing_hip_options`` fills both gaps. uTLX also lacks fbtriton's intra-wave
``tlx.warp_pipeline_stage`` scope; see ``inline_intra_wave_pipeline_stages``.
"""

import contextvars
import dataclasses
import functools
import inspect

import triton

# uTLX's HIPOptions fields and their LLVM flags, as in fbtriton's _get_codegen_flags.
_CODEGEN_FLAGS = {
    "reverse_local_assignment": "greedy-reverse-local-assignment",
    "sink_insts_to_avoid_spills": "sink-insts-to-avoid-spills",
    "regclass_priority_trumps_globalness": "greedy-regclass-priority-trumps-globalness",
    "disable_unclustered_high_rp_reschedule": "amdgpu-disable-unclustered-high-rp-reschedule",
}

_llvm_flags = contextvars.ContextVar("tlx_llvm_flags", default=())


class UnsupportedOp(RuntimeError):
    """Upstream ``tlx.ops`` ``UnsupportedOp``: a configuration the vendored TLX kernel does not implement."""


class InvalidInput(ValueError):
    """Upstream ``tlx.ops`` ``InvalidInput``: inputs outside the vendored TLX kernel's contract."""


@functools.cache
def add_missing_triton_knobs() -> None:
    """Define fbtriton-only instrumentation knobs as False on stock Triton; a no-op on fbtriton.

    tlx.ops kernels read them before their compiled-cache fast launch, and stock Triton
    has none of that instrumentation.
    """
    for group, name in (
        ("runtime", "sanitize_overflow"),
        ("compilation", "fpsan_homomorphic_casts"),
    ):
        knobs = getattr(triton.knobs, group)
        if not hasattr(knobs, name):
            # On the class: knobs.scope() resets instance dicts.
            setattr(type(knobs), name, False)


@functools.cache
def add_missing_hip_options() -> None:
    """Accept fbtriton's ``llvm_cl_opts`` launch option on stock ROCm Triton and pass it,
    with the uTLX codegen flags, to the AMDGPU code generator; a no-op on fbtriton.
    """
    import triton.backends.amd.compiler as hip

    # uTLX adds its HIPOptions fields when TLX is first imported; the subclass below must derive from those.
    import triton.language.extra.tlx  # noqa: F401

    base = hip.HIPOptions
    if "llvm_cl_opts" in base.__dataclass_fields__:
        return

    # Triton's options are a frozen dataclass; a subclass field reaches the compile-cache key.
    @dataclasses.dataclass(frozen=True)
    class HIPOptions(base):
        # "name[=value],..." (a bare name means true) or ((name, value), ...), as in fbtriton.
        llvm_cl_opts: str | tuple[tuple[str, str], ...] = ""

        def __post_init__(self):
            super().__post_init__()
            object.__setattr__(
                self, "llvm_cl_opts", _parse_llvm_cl_opts(self.llvm_cl_opts)
            )

    HIPOptions.__qualname__ = base.__qualname__
    HIPOptions.__module__ = base.__module__
    # parse_options reads the module global.
    hip.HIPOptions = HIPOptions

    orig_make_amdgcn = hip.HIPBackend.make_amdgcn
    orig_compile_amdgpu = hip.compile_amdgpu

    def make_amdgcn(src, metadata, options):
        token = _llvm_flags.set(_requested_llvm_flags(options))
        try:
            return orig_make_amdgcn(src, metadata, options)
        finally:
            _llvm_flags.reset(token)

    def compile_amdgpu(*args, flags, **kwargs):
        return orig_compile_amdgpu(*args, flags=[*flags, *_llvm_flags.get()], **kwargs)

    hip.HIPBackend.make_amdgcn = staticmethod(make_amdgcn)
    hip.compile_amdgpu = compile_amdgpu


def _parse_llvm_cl_opts(opts) -> tuple[tuple[str, str], ...]:
    if not isinstance(opts, str):
        return tuple(tuple(opt) for opt in opts)
    parsed = []
    for opt in opts.split(","):
        name, sep, value = opt.partition("=")
        if name.strip():
            parsed.append((name.strip(), value.strip() if sep else "true"))
    return tuple(parsed)


def _requested_llvm_flags(options) -> tuple[str, ...]:
    # fbtriton's order: codegen flags, then llvm_cl_opts so they override.
    codegen = [flag for name, flag in _CODEGEN_FLAGS.items() if getattr(options, name)]
    return (*codegen, *(f"{name}={value}" for name, value in options.llvm_cl_opts))


@functools.cache
def inline_intra_wave_pipeline_stages() -> None:
    """Emit ``tlx.warp_pipeline_stage(..., scope="intra_wave")`` bodies inline on uTLX,
    which rejects ``scope``; a no-op on fbtriton and on a uTLX that accepts it.

    fbtriton brackets such a stage with markers for an instruction-scheduling pass and
    adds no barriers or data movement, so the inline body computes the same result.
    """
    try:
        from utlx_plugin.compiler.dispatch import TLX_WITH_DISPATCH
    except ImportError:
        return
    import triton.language.extra.tlx as tlx
    from triton.language.core import _unwrap_if_constexpr

    stage_cls = tlx.warp_pipeline_stage
    if "scope" in inspect.signature(stage_cls).parameters:
        return
    visit_stage = TLX_WITH_DISPATCH[stage_cls]

    def visit_stage_or_inline(self, node):
        scope = next(
            (
                kw.value
                for kw in node.items[0].context_expr.keywords
                if kw.arg == "scope"
            ),
            None,
        )
        if (
            scope is not None
            and _unwrap_if_constexpr(self.visit(scope)) == "intra_wave"
        ):
            return self.visit_compound_statement(node.body)
        return visit_stage(self, node)

    TLX_WITH_DISPATCH[stage_cls] = visit_stage_or_inline

"""Cache compatible FlashInfer 0.7.0 MXFP8 host dispatch.

Keep FlashInfer's output allocation, autotuner and kernel invocation intact.
Only reuse the stateless CuTe runner and validation of identical metadata.
"""

from functools import lru_cache
from inspect import Parameter, signature

try:
    from flashinfer import __version__ as flashinfer_version
    from flashinfer.gemm import gemm_base
except ImportError:
    flashinfer_version = ""
    gemm_base = None


def maybe_cache_mxfp8_dispatch(raw_mm):
    """Keep unsupported FlashInfer versions and APIs on the original path."""
    if flashinfer_version.split("+", 1)[0] != "0.7.0" or gemm_base is None:
        return raw_mm
    try:
        if not callable(gemm_base._cute_dsl_gemm_mxfp8_runner):
            return raw_mm
        # FlashInfer's backend-requirement decorator consumes skip_check via
        # **kwargs; following __wrapped__ would hide this supported argument.
        parameters = signature(raw_mm, follow_wrapped=False).parameters
    except (AttributeError, TypeError, ValueError):
        return raw_mm
    skip_check = parameters.get("skip_check")
    accepts_keyword = skip_check is not None and skip_check.kind in (
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.KEYWORD_ONLY,
    )
    if not accepts_keyword and not any(
        p.kind == Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return raw_mm
    return Mxfp8DispatchCache(raw_mm, gemm_base)


class Mxfp8DispatchCache:
    def __init__(self, raw_mm, gemm_module):
        self.raw_mm = raw_mm
        self.validated = set()
        factory = gemm_module._cute_dsl_gemm_mxfp8_runner
        # This process-wide patch only memoizes the stateless
        # runner factory. The factory arguments include SM, PDL and output dtype.
        if not hasattr(factory, "cache_info"):
            factory = lru_cache(maxsize=16)(factory)
            gemm_module._cute_dsl_gemm_mxfp8_runner = factory
        self.runner_factory = factory

    def __call__(self, a, b, a_scale, b_scale, out_dtype, use_8x4_sf_layout, backend):
        kwargs = dict(
            out_dtype=out_dtype,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
        )
        if backend != "cute-dsl":
            return self.raw_mm(a, b, a_scale, b_scale, **kwargs)

        # Validation depends on metadata, never tensor contents. A changed
        # shape, stride, dtype or device must pass the original checks again.
        key = (
            out_dtype,
            use_8x4_sf_layout,
            tuple(
                (t.shape, t.stride(), t.dtype, t.device, t.layout)
                for t in (a, b, a_scale, b_scale)
            ),
        )
        validated = key in self.validated
        result = self.raw_mm(a, b, a_scale, b_scale, skip_check=validated, **kwargs)
        if not validated:
            # Bound metadata memory for varying workloads. Only successful
            # executions qualify; a failed validation is never cached.
            if len(self.validated) >= 1024:
                self.validated.clear()
            self.validated.add(key)
        return result

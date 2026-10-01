"""Opt-in lab experiment for FlashInfer 0.7 MXFP8 host dispatch.

Keep FlashInfer's output allocation, autotuner and kernel invocation intact.
Only reuse the stateless CuTe runner and validation of identical metadata.
"""

from functools import lru_cache


class Mxfp8DispatchCache:
    def __init__(self, raw_mm, gemm_module):
        self.raw_mm = raw_mm
        self.validated = set()
        factory = gemm_module._cute_dsl_gemm_mxfp8_runner
        # This experimental process-wide patch only memoizes the stateless
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

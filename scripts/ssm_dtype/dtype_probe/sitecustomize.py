"""Optional probe that monkey-patches MambaPool.__init__ to print the actual
SSM cache dtypes/shapes once per worker. Activated by SGLANG_SSM_DTYPE_PROBE=1
in the launched server's environment; otherwise this module is a no-op.

Lives under scripts/ssm_dtype/dtype_probe/ which the runner prepends to
PYTHONPATH, so Python imports this as sitecustomize during interpreter init.
"""

import os


def _install_probe() -> None:
    if os.environ.get("SGLANG_SSM_DTYPE_PROBE") != "1":
        return

    try:
        from sglang.srt.mem_cache.memory_pool import MambaPool
    except Exception as exc:
        print(f"SSM_STATE_DTYPE_PROBE install_failed={exc!r}", flush=True)
        return

    original_init = MambaPool.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            cache = getattr(self, "mamba_cache", None)
            temporal = getattr(cache, "temporal", None)
            conv = getattr(cache, "conv", None) or []
            intermediate_ssm = getattr(cache, "intermediate_ssm", None)
            print(
                "SSM_STATE_DTYPE_PROBE "
                f"actual_temporal_state_dtype={getattr(temporal, 'dtype', None)} "
                f"actual_temporal_state_shape={tuple(temporal.shape) if temporal is not None else None} "
                f"actual_conv_state_dtypes={[getattr(t, 'dtype', None) for t in conv]} "
                f"actual_intermediate_ssm_dtype={getattr(intermediate_ssm, 'dtype', None)}",
                flush=True,
            )
        except Exception as exc:
            print(f"SSM_STATE_DTYPE_PROBE print_failed={exc!r}", flush=True)

    MambaPool.__init__ = patched_init


_install_probe()

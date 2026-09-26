#!/bin/bash
# Exit 0 when the installed flashinfer-jit-cache can actually serve cubins.
# Since 0.7.0 they live in per-arch dists found through entry points, so a shim
# with none registered still imports and still reports the expected version.
python3 - <<'PY'
import sys

try:
    import flashinfer_jit_cache
except Exception as error:
    print(f"flashinfer-jit-cache is not importable: {error}")
    sys.exit(1)

try:
    from flashinfer_jit_cache import get_jit_cache_providers
except ImportError:
    # Pre-0.7.0 layout: the cubins ship inside the package itself.
    print(f"flashinfer-jit-cache {flashinfer_jit_cache.__version__} (pre-provider layout)")
    sys.exit(0)

providers = get_jit_cache_providers()
if not providers:
    print(
        f"flashinfer-jit-cache {flashinfer_jit_cache.__version__} has no usable "
        "provider; the per-arch packages are missing or their entry points were lost"
    )
    sys.exit(1)

print(
    f"flashinfer-jit-cache {flashinfer_jit_cache.__version__} with "
    f"{len(providers)} provider(s): {', '.join(p.provider_id for p in providers)}"
)
PY

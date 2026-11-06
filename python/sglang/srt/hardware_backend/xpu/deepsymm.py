"""Loader for deep_symm.moe_ep, the DeepEP backend for Intel XPU shipped by DeepSymm."""

from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

DEEPSYMM_DIST_NAME = "deep-symm"
DEEPSYMM_PACKAGE = "applications.ai.gpu.deepsymm"
# deep-symm is built from source, and its version is 0.0.1+<git revision>, so
# there is no ordered version to compare against; name what this path was tested on.
DEEPSYMM_TESTED_REVISION = "6d18e46"
DEEPSYMM_SOURCE_HINT = (
    f"{DEEPSYMM_PACKAGE} is not on PyPI; build it from source (revision "
    f"{DEEPSYMM_TESTED_REVISION} was tested) into the same environment as SGLang"
)


def import_deepsymm_moe_ep() -> ModuleType:
    """Import deep_symm.moe_ep and its _C extension from DeepSymm.

    Raises ImportError naming the applications.ai.gpu.deepsymm package on any failure.
    """
    try:
        deepsymm_version = version(DEEPSYMM_DIST_NAME)
    except PackageNotFoundError as e:
        raise ImportError(
            f"DeepEP on XPU requires package {DEEPSYMM_PACKAGE!r} "
            f"(distribution {DEEPSYMM_DIST_NAME!r}), which is not installed. "
            f"{DEEPSYMM_SOURCE_HINT}."
        ) from e

    # An _C ABI/runtime mismatch raises OSError, not ImportError.
    try:
        from deep_symm import moe_ep
    except (ImportError, OSError) as e:
        raise ImportError(
            f"Package {DEEPSYMM_PACKAGE!r} ({DEEPSYMM_DIST_NAME} "
            f"{deepsymm_version}) is installed but deep_symm.moe_ep (or its "
            f"_C extension) failed to load: {e!r}. Rebuild it against the "
            f"installed torch and oneAPI; revision {DEEPSYMM_TESTED_REVISION} "
            f"was tested. Revisions before the move to deep_symm.moe_ep "
            f"(DeepSymm #48) are not supported."
        ) from e
    return moe_ep

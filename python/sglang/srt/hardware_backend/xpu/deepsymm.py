"""Loader for deep_ep_xpu, the DeepEP backend for Intel XPU shipped by DeepSymm.

Other DeepEP ports also build an extension named deep_ep_cpp, so check that the
loaded one is DeepSymm's rather than one that shadows it.
"""

import os
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

DEEPSYMM_DIST_NAME = "deep-symm"
DEEPSYMM_PACKAGE = "applications.ai.gpu.deepsymm"
# deep_ep_xpu is built from source, and its version is 0.0.1+<git revision>, so
# there is no ordered version to compare against; name what this path was tested on.
DEEPSYMM_TESTED_REVISION = "2b755dd"
DEEPSYMM_SOURCE_HINT = (
    f"{DEEPSYMM_PACKAGE} is not on PyPI; build it from source (revision "
    f"{DEEPSYMM_TESTED_REVISION} was tested) into the same environment as SGLang"
)


def import_deep_ep_xpu() -> ModuleType:
    """Import deep_ep_xpu, verifying it and deep_ep_cpp come from DeepSymm.

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

    # A deep_ep_cpp ABI/runtime mismatch raises OSError, not ImportError.
    try:
        import deep_ep_cpp
        import deep_ep_xpu
    except (ImportError, OSError) as e:
        raise ImportError(
            f"Package {DEEPSYMM_PACKAGE!r} ({DEEPSYMM_DIST_NAME} "
            f"{deepsymm_version}) is installed but deep_ep_xpu (or its "
            f"deep_ep_cpp extension) failed to load: {e!r}. Rebuild it against "
            f"the installed torch and oneAPI; revision "
            f"{DEEPSYMM_TESTED_REVISION} was tested."
        ) from e

    # DeepSymm installs deep_ep_cpp next to the deep_ep_xpu package directory.
    pkg_root = os.path.dirname(os.path.dirname(os.path.realpath(deep_ep_xpu.__file__)))
    ext_root = os.path.dirname(os.path.realpath(deep_ep_cpp.__file__))
    if pkg_root != ext_root:
        raise ImportError(
            f"deep_ep_cpp was loaded from {ext_root}, not from package "
            f"{DEEPSYMM_PACKAGE!r} at {pkg_root}; another DeepEP build is "
            f"shadowing it."
        )
    return deep_ep_xpu

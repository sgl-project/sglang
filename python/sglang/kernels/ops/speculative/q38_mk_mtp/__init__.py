"""FN MTP segment kernels; opt-in, with native paths for unsupported shapes."""

import os


def enabled() -> bool:
    return os.environ.get("SGLANG_Q38C_MK_MTP", "0") == "1"

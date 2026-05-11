"""DeepSeek V4 model dispatcher.

Routes to the CUDA-native implementation or the HIP/ROCm-ported
implementation based on the current platform.
"""

from sglang.srt.models.deepseek_v2 import _is_hip

if _is_hip:
    from sglang.srt.models.deepseek_v4_hip import *  # noqa: F401,F403
    from sglang.srt.models.deepseek_v4_hip import (  # noqa: F401
        _EntryClass as EntryClass,
    )
else:
    from sglang.srt.models.deepseek_v4_cuda import *  # noqa: F401,F403
    from sglang.srt.models.deepseek_v4_cuda import (  # noqa: F401
        _EntryClass as EntryClass,
    )

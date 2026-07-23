# SPDX-License-Identifier: Apache-2.0

from sglang.srt.utils import (
    get_bool_env_var,
    is_gfx95_supported,
    is_hip,
)

USE_AITER = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
USE_AITER_GFX95 = USE_AITER and is_gfx95_supported()

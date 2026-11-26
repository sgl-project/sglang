# SPDX-License-Identifier: Apache-2.0

from sglang.srt.environ import envs
from sglang.srt.utils import is_gfx95_supported, is_gfx1250_supported, is_hip

USE_AITER = envs.SGLANG_USE_AITER.get() and is_hip()
USE_AITER_GFX95 = USE_AITER and is_gfx95_supported()
USE_AITER_GFX1250 = USE_AITER and is_gfx1250_supported()

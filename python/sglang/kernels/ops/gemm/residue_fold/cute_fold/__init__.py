# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""CuTeDSL MExt R1 row-pair fold GEMM for SM100-class GPUs."""

from .host import mext_fold_gemm_sm103  # noqa: F401
from .tactics import (  # noqa: F401
    kernel_arch_for_capability,
    run_fold_default,
    warmup_fold_default,
)

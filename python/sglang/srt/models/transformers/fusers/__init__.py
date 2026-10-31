# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team

from .dense import FusionResult, fuse_module, load_fused_weights

__all__ = ["FusionResult", "fuse_module", "load_fused_weights"]

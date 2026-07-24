# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Concrete efficiency techniques. Importing a module registers its technique
# via @register_technique.

from sglang.multimodal_gen.runtime.efficiency.techniques import (  # noqa: F401
    step_cache,
    teacache,
    token_prune,
)

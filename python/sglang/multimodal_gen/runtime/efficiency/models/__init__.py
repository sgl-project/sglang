# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Per-model ModelSpecs. Importing registers each via @register_model_spec.
#
# NOTE: keep this list explicit so the registration order is deterministic and
# missing specs surface as clean import errors rather than silent no-ops.

from sglang.multimodal_gen.runtime.efficiency.models import (  # noqa: F401
    ltx2_spec,
    sana_video_spec,
)

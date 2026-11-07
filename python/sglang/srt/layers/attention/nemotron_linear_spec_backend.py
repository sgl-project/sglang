# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""FlashInfer graph phases for Nemotron self-speculation."""

from sglang.srt.dllm.nemotron_linear_spec import NemotronLinearSpecGraphVariants
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend


class NemotronLinearSpecFlashInferBackend(FlashInferAttnBackend):
    def get_decode_graph_variants(self, capture_forward_mode, captured_req_width):
        if capture_forward_mode.is_dllm_extend():
            return NemotronLinearSpecGraphVariants()
        return None

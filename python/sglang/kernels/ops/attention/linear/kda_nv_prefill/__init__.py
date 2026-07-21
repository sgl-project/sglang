# SPDX-License-Identifier: Apache-2.0
# NVIDIA KDA_prefill (Blackwell): fused K1-K4 chunked KDA forward for the
# equal-length batches prepared by SGLang's serving adapter.
from .chunk_fwd import chunk_kda_fwd

__all__ = ["chunk_kda_fwd"]

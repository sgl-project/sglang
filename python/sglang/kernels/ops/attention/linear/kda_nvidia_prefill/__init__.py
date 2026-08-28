# SPDX-License-Identifier: Apache-2.0
# NVIDIA KDA_prefill (Blackwell): optimized chunked KDA forward, an
# FLA-compatible replacement for chunk_kda_fwd. K1 (gate+cumsum+scale, CuTe)
# + K2 (intra sub-chunk, CuTe) + K3 (inter-chunk solve, Triton) + K4
# (W/U/v_new/O/state update, cuTile persistent). 2.3-2.9x vs the FLA
# reference on B200 in the upstream package tests.
from .chunk_fwd import chunk_kda_fwd

__all__ = ["chunk_kda_fwd"]

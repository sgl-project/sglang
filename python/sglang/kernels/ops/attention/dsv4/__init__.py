"""DeepSeek-V4 attention kernels (RFC #29630, Phase 2.5)."""

# --- merged from sglang.kernels.ops.attention.dsv4 (RFC #29630 Phase 4) ---
from .attn import (
    fused_store_cache,
    get_paged_mqa_logits_metadata,
    triton_create_paged_compress_data,
)
from .c128_cleanup import clear_unaccepted_c128_draft_states
from .compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_forward_norm_rope_store,
    compress_norm_rope_store,
)
from .compress_old import fused_norm_rope_inplace
from .elementwise import (
    fused_k_norm_rope_flashmla,
    fused_q_indexer_rope_first_quant,
    fused_q_indexer_rope_hadamard_fp4_quant,
    fused_q_indexer_rope_hadamard_quant,
    fused_q_norm_rope,
    fused_rope_inplace,
)
from .fp8_wo_a import sglang_per_token_group_quant_fp8_dsv4_wo_a
from .topk import (
    plan_topk_v2,
    topk_transform_paged,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from .utils import make_name

__all__ = [
    "CompressorDecodePlan",
    "CompressorPrefillPlan",
    "compress_forward",
    "compress_forward_norm_rope_store",
    "compress_norm_rope_store",
    "clear_unaccepted_c128_draft_states",
    "fused_norm_rope_inplace",
    "fused_store_cache",
    "fused_rope_inplace",
    "fused_q_norm_rope",
    "fused_q_indexer_rope_first_quant",
    "fused_q_indexer_rope_hadamard_fp4_quant",
    "fused_q_indexer_rope_hadamard_quant",
    "fused_k_norm_rope_flashmla",
    "sglang_per_token_group_quant_fp8_dsv4_wo_a",
    "make_name",
    "get_paged_mqa_logits_metadata",
    "triton_create_paged_compress_data",
    "topk_transform_paged",
    "topk_transform_paged_v2",
    "topk_transform_ragged_v2",
    "plan_topk_v2",
]

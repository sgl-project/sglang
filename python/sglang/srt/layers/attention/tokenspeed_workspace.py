"""Dependency-light helpers for TokenSpeed MLA workspace sizing."""

# Workspace upper bound for tokenspeed_mla_decode:
#   B * H_eff * S_eff * split_kv * (kv_lora_rank + 1) * sizeof(float32)
# TokenSpeed's split planner keeps B * split_kv <= num_sms, so an upper bound is:
#   num_sms * H_eff * S_eff * (kv_lora_rank + 1) * sizeof(float32)
# For small-head tree verify shapes, TokenSpeed folds q chunks of 8 tokens:
# q16/q24/q32 become H_eff=128, S_eff=2/3/4. In all supported modes,
# H_eff * S_eff is bounded by the number of query chunks times the larger of
# a standard Blackwell M tile and the per-chunk query rows.
_TOKENSPEED_Q_CHUNK_SIZE = 8
_TOKENSPEED_MIN_M_ROWS = 128


def tokenspeed_workspace_bytes(
    num_sms: int, num_heads: int, kv_lora_rank: int, q_len: int
) -> int:
    q_len = max(1, q_len)
    q_chunks = (q_len + _TOKENSPEED_Q_CHUNK_SIZE - 1) // _TOKENSPEED_Q_CHUNK_SIZE
    rows_per_chunk = max(
        _TOKENSPEED_MIN_M_ROWS,
        num_heads * min(q_len, _TOKENSPEED_Q_CHUNK_SIZE),
    )
    query_rows = q_chunks * rows_per_chunk
    return num_sms * query_rows * (kv_lora_rank + 1) * 4

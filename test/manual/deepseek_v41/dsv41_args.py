import msgspec


class DeepseekV41Args(msgspec.Struct, kw_only=True, frozen=True):
    """Model shape for the reference-style modules. Field names follow the reference
    config; scale-independent defaults are the released model's."""

    max_batch_size: int
    max_seq_len: int
    vocab_size: int
    dim: int
    moe_inter_dim: int
    n_layers: int
    n_mtp_layers: int = 0
    n_heads: int
    n_routed_experts: int
    n_shared_experts: int = 1
    n_activated_experts: int
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    q_lora_rank: int
    head_dim: int
    rope_head_dim: int
    norm_eps: float = 1e-20
    o_groups: int
    o_lora_rank: int
    window_size: int
    compress_ratios: tuple[int, ...]
    kv_source_layers: tuple[int, ...]
    index_source_layers: tuple[int, ...]
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 0
    vision_enabled: bool = False
    expert_fp4: bool = True

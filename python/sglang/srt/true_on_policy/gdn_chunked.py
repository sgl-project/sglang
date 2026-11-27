"""TOP GDN: rollout decode follows FLA's fixed-tile chunk-64 prefill bit for bit.

Under ``SGLANG_TOP_GDN_MODE=chunked`` (the TOP GDN program) the trainer runs one plain prefill per sample: every
rollout GDN output equals row t of ``chunk_gated_delta_rule(fixed_output_tile=True)`` over the whole sample from
zero state, under the pinned kkt config (``chunk_fwd.TOP_CHUNKED_KKT_CONFIG``).

Serving state per running request (``gdn_chunked_backend``):

- the mamba slot's FP32 state holds E, the state at the start of the request's open 64-token chunk;
- a per-layer row cache indexed by the request's req-pool row holds the open chunk's rows (``chunk_rows2``
  layout: l2-normed k, v, v_new, raw gate, beta, the solved 64 x 64 inverse blocks) and ``e_chunk``, the chunk E
  stands at (under speculation the chunk close is deferred to the next chunk's first step, so a verify never
  writes E).

Decode runs one row per step (``chunk_rows`` kernels, closing the chunk at row 63; ``chunk_rows2`` with one
row when speculation is on), chain spec verify runs the root plus the drafts as one multi-row ``chunk_rows2``
step with the window cut at the chunk end (drafts past it are never accepted), and an extend runs chunk prefill
to each request's last 64-token boundary, then its tail rows through the row engine. Radix snapshots are FP32
states on the 64-token grid.
"""

from sglang.srt.environ import envs

CHUNKED = "chunked"
GDN_MODES = (CHUNKED,)

CHUNK = 64
# chunk_rows2 keeps a step's rows inside two 16-row blocks: at most 16 rows per request per step.
MAX_STEP_ROWS = 16


def top_gdn_mode() -> str:
    """``SGLANG_TOP_GDN_MODE``; "" when unset."""
    mode = envs.SGLANG_TOP_GDN_MODE.get()
    if mode and mode not in GDN_MODES:
        raise ValueError(
            f"SGLANG_TOP_GDN_MODE must be one of {GDN_MODES}, got {mode!r}"
        )
    return mode


def select_chunked_programs(ops) -> None:
    """TOP runs GDN and KDA layers only with their chunked programs. The FLA kernels pin their configs when they are
    imported, so the engine selects the program for the family's ops before it starts its schedulers, which inherit
    it; GDN and KDA share the state kernel, so each family gets only its own program."""
    if "gdn" in ops:
        envs.SGLANG_TOP_GDN_MODE.set(CHUNKED)
    if "kda" in ops:
        envs.SGLANG_TOP_KDA_MODE.set(CHUNKED)


def require_chunked_gdn_kernels() -> None:
    """Refuse FLA GDN kernels imported before the chunked program was selected: their configs were pinned then."""
    from sglang.kernels.ops.attention.fla import chunk_delta_h, chunk_fwd

    # The kernel is @triton.heuristics over @triton.autotune: the Autotuner (and its configs) is the wrapper's fn.
    kkt = chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel.fn
    if not chunk_delta_h.TOP_CHUNKED_FMA_CLOSE or len(kkt.configs) != 1:
        raise ValueError(
            "GDN TOP: the FLA GDN kernels were imported before the chunked GDN program was selected"
        )


def top_gdn_chunked(server_args) -> bool:
    """TOP is on and its GDN layers run the chunked program."""
    return server_args.true_on_policy_contract is not None and top_gdn_mode() == CHUNKED


def top_gdn_chunked_active() -> bool:
    """``top_gdn_chunked`` for the running engine's server args, read only under the chunked mode."""
    from sglang.srt.runtime_context import get_server_args

    return top_gdn_mode() == CHUNKED and top_gdn_chunked(get_server_args())


def row_cache_bytes_per_layer(
    *, num_k_heads: int, num_v_heads: int, head_k_dim: int, head_v_dim: int
) -> int:
    """Bytes of one request's row cache in one GDN layer (``chunk_rows2.RowCache``)."""
    rows_bf16 = CHUNK * (num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim) * 2
    gates_fp32 = 2 * CHUNK * num_v_heads * 4
    solved_fp32 = num_v_heads * CHUNK * CHUNK * 4
    return rows_bf16 + gates_fp32 + solved_fp32 + num_v_heads * 4


def row_cache_bytes_per_request(config, num_layers: int) -> int:
    """Row-cache bytes per req-pool row over ``num_layers`` GDN layers of a Qwen3.5 text config."""
    return num_layers * row_cache_bytes_per_layer(
        num_k_heads=config.linear_num_key_heads,
        num_v_heads=config.linear_num_value_heads,
        head_k_dim=config.linear_key_head_dim,
        head_v_dim=config.linear_value_head_dim,
    )


def validate_gdn_chunked_contract(args, config) -> None:
    """What the chunked GDN program supports (part of ``validate_gdn_contract``)."""
    if config.linear_num_key_heads != 16 or config.linear_value_head_dim % 64:
        raise ValueError(
            "TOP chunked GDN's row engine runs 16 key heads per rank (one l2norm tile per token) and 64-wide "
            "value tiles"
        )
    if args.speculative_algorithm is not None:
        drafts = args.speculative_num_draft_tokens
        if drafts is None or drafts > MAX_STEP_ROWS:
            raise ValueError(
                f"TOP chunked GDN verifies at most {MAX_STEP_ROWS} tokens per request per step; "
                f"got --speculative-num-draft-tokens {drafts}"
            )
    if args.pp_size > 1:
        raise ValueError("TOP chunked GDN does not support pipeline parallelism")
    if args.enable_two_batch_overlap:
        raise ValueError(
            "TOP chunked GDN plans each extend over the whole batch: two-batch overlap is not supported"
        )
    if args.enable_linear_replayssm_spec:
        raise ValueError("TOP chunked GDN replaces ReplaySSM spec verify")

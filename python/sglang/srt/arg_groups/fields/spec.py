"""Config fields of the ``spec`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``spec`` bag, which is what ``get_spec()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import dataclasses
from typing import (
    Literal,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.choices import (
    LOAD_FORMAT_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    QUANTIZATION_CHOICES,
)


@dataclasses.dataclass
class Spec:
    """Namespace ``spec``."""

    _NS_PATH = "spec"
    # -------------------------------------------------------------------------
    # Speculative decoding
    # -------------------------------------------------------------------------
    speculative_algorithm: A[
        Optional[str],
        "Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO. Or any name registered via `SpeculativeAlgorithm.register`.",
    ] = None
    uno_lora_path: A[Optional[str], "Path to the UNO draft LoRA checkpoint."] = None
    speculative_draft_model_path: A[
        Optional[str],
        Arg(
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
            aliases=["--speculative-draft-model"],
        ),
    ] = None
    speculative_draft_model_revision: A[
        Optional[str],
        "The specific draft model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.",
    ] = None
    speculative_draft_load_format: A[
        Optional[str],
        Arg(
            help="The format of the draft model weights to load. If not specified, will use the same format as --load-format. Use 'dummy' to initialize draft model weights with random values for profiling.",
            choices=LOAD_FORMAT_CHOICES,
        ),
    ] = None
    speculative_num_steps: A[
        Optional[int],
        "The number of steps sampled from draft model in Speculative Decoding.",
    ] = None
    speculative_eagle_topk: A[
        Optional[int],
        "The number of tokens sampled from the draft model in eagle2 each step.",
    ] = None
    speculative_num_draft_tokens: A[
        Optional[int],
        "The number of tokens sampled from the draft model in Speculative Decoding.",
    ] = None
    speculative_dflash_block_size: A[
        Optional[int],
        "DFLASH only. Block size (verify window length). Alias of --speculative-num-draft-tokens for DFLASH.",
    ] = None
    speculative_dspark_block_size: A[
        Optional[int],
        "DSPARK only. Draft block size gamma (number of proposed draft tokens). The verify window is gamma + 1, so this sets --speculative-num-draft-tokens = gamma + 1. Omit to auto-infer gamma from the draft checkpoint block_size.",
    ] = None
    speculative_dspark_sps_table_path: A[
        Optional[str],
        "DSPARK only. Path to a pre-profiled SPS cost table (JSON) built offline with "
        "sglang.benchmark.dspark_sps_profiler, consumed by the ragged-verify "
        "scheduler (cap-accept / compact). Omit for an uninitialized flat "
        "constant-SPS table: the budget degenerates to verify-all (zero throughput "
        "gain by itself).",
    ] = None
    speculative_dspark_confidence_sts_path: A[
        Optional[str],
        "DSPARK only. Optional path to a per-position STS (sequential temperature "
        "scaling) calibration JSON, fit offline with sglang.benchmark.dspark_sts_fit. "
        "Calibrates the confidence-head survival probabilities the ragged-verify "
        "scheduler consumes. Omit to use identity (no calibration); losslessness is "
        "unaffected either way.",
    ] = None
    speculative_dspark_align_verify_tokens_to_graph_tier: A[
        bool,
        "DSPARK compact ragged-verify only. Fill the per-request verify lengths so "
        "the total verify-token count reaches the cuda-graph tier the forward is "
        "already padded to: round the dp-max scheduled total up to the captured "
        "token bucket and let the top-k allocator admit that many real draft tokens "
        "(confidence-ordered). This recovers the padding the forward pays for anyway "
        "-- both the cuda-graph bucket round-up and the dp cross-rank max -- turning "
        "it into extra real verification at the same step time. Off by default; when "
        "off the schedule is byte-for-byte unchanged.",
    ] = False
    speculative_accept_threshold_single: A[
        float,
        "Accept a draft token if its probability in the target model is greater than this threshold.",
    ] = 1.0
    speculative_accept_threshold_acc: A[
        float,
        "The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
    ] = 1.0
    speculative_use_rejection_sampling: A[
        bool, "Use rejection sampling for speculative decoding (requires topk=1)."
    ] = False
    speculative_token_map: A[
        Optional[str],
        "The path of the draft model's small vocab table.",
    ] = None
    speculative_attention_mode: A[
        str,
        Arg(
            help="Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.",
            choices=["prefill", "decode"],
            resolvable=True,
        ),
    ] = "prefill"
    speculative_draft_attention_backend: A[
        Optional[str],
        Arg(
            help="Attention backend for speculative decoding drafting.",
            resolvable=True,
        ),
    ] = None
    speculative_dsa_topk_backend: A[
        str,
        Arg(
            help="DSA indexer top-k backend for speculative draft workers. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",
            choices=["sgl-kernel", "torch", "flashinfer"],
        ),
    ] = "sgl-kernel"
    speculative_draft_kv_cache_dtype: A[
        Optional[str],
        Arg(
            help="KV cache dtype for the speculative draft model only. The draft pool is "
            "allocated with one slot per target token (draft and target share a slot index "
            "space), so for a small draft it can still rival the target pool: a 5-layer "
            "DFLASH draft costs 10240 bytes/token in bf16. Setting fp8_e4m3 halves the draft "
            "pool; the saving shows up as free device memory, so raise "
            "--mem-fraction-static to convert it into KV capacity. Default follows "
            "--kv-cache-dtype.",
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16"],
        ),
    ] = None
    speculative_draft_window_size: A[
        Optional[int],
        "Sliding window size for the draft model. Honored by Llama EAGLE-3 (`LlamaForCausalLMEagle3`) and DFLASH only; other EAGLE-3 backends (e.g. MLA-based drafters) silently ignore it. For Llama EAGLE-3, the drafter only attends to the most recent N keys (verifier hidden states + its own outputs); the verifier is unaffected. For DFLASH, the draft worker keeps a recent target-token window in its local KV cache (paged backends may retain up to one extra page on the left for alignment). Default is full attention/context.",
    ] = None
    speculative_moe_runner_backend: A[
        Optional[str],
        Arg(
            help="Choose the runner backend for MoE in speculative decoding.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = None
    speculative_moe_a2a_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for MoE A2A in speculative decoding",
            choices=[
                "none",
                "deepep",
                "mooncake",
                "nixl",
                "mori",
                "ascend_fuseep",
                "flashinfer",
                "megamoe",
                "deepep_v2",
                "pplx",
                "ascend_tp",
            ],
            resolvable=True,
        ),
    ] = None
    speculative_draft_model_quantization: A[
        Optional[str],
        Arg(
            help="The quantization method for speculative model.",
            choices=QUANTIZATION_CHOICES,
        ),
    ] = None
    # Internal provenance used after the public draft quantization inherits the
    # target value. It is a dataclass field so ServerArgs round-trips preserve
    # whether the user explicitly set the draft option; it has no CLI surface.
    _speculative_draft_quantization_explicitly_set: A[
        Optional[bool],
        Arg(no_cli=True),
    ] = None
    speculative_skip_dp_mlp_sync: A[
        bool,
        "Skip the extra MLP sync that the scheduler performs before merging a new batch when speculative decoding + DP attention are both enabled.",
    ] = False
    enable_multi_layer_eagle: A[
        bool,
        Arg(
            help="Enable multi-layer Eagle speculative decoding.",
            resolvable=True,
        ),
    ] = False
    speculative_adaptive: A[
        bool,
        "Enable adaptive speculative decoding that dynamically adjusts num_steps based on acceptance rate.",
    ] = False
    speculative_adaptive_config: A[
        Optional[str],
        "Path to a JSON config file for adaptive speculative decoding tuning knobs.",
    ] = None
    spec_trace_dir: A[
        Optional[str], "Directory to write decoupled speculative decoding trace files."
    ] = None

    # -------------------------------------------------------------------------
    # Speculative decoding (ngram)
    # -------------------------------------------------------------------------
    speculative_ngram_min_bfs_breadth: A[
        int,
        "The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 1
    speculative_ngram_max_bfs_breadth: A[
        int,
        "The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
    ] = 10
    speculative_ngram_match_type: A[
        Literal["BFS", "PROB"],
        "The match type for cache tree.",
    ] = "BFS"
    speculative_ngram_max_trie_depth: A[
        int,
        "The max trie depth for ngram speculative decoding.",
    ] = 18
    speculative_ngram_capacity: A[
        int,
        "The cache capacity for ngram speculative decoding.",
    ] = 10 * 1000 * 1000
    speculative_ngram_external_corpus_path: A[
        Optional[str],
        "Path to an external JSONL corpus to pre-load into SAM at startup. Additional corpora can be added at runtime via POST /add_external_corpus.",
    ] = None
    speculative_ngram_external_sam_budget: A[
        int,
        "Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding.",
    ] = 0
    speculative_ngram_external_corpus_max_tokens: A[
        int,
        "Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget.",
    ] = 10000000

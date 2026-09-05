"""Config fields of the ``parallel`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``parallel`` bag, which is what ``get_parallel()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Optional

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
    Derived,
)


@dataclasses.dataclass
class Parallel:
    """Namespace ``parallel``."""

    _NS_PATH = "parallel"

    # -------------------------------------------------------------------------
    # Distributed topology and parallelism (TP, PP, DP, CP)
    # -------------------------------------------------------------------------
    nccl_port: A[
        Optional[int],
        "The port for NCCL distributed environment setup. Defaults to a random port.",
    ] = None
    dist_timeout: A[
        Optional[int], "Set timeout for torch.distributed initialization."
    ] = None
    dist_init_addr: A[
        Optional[str],
        Arg(
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
            aliases=["--nccl-init-addr"],
        ),
    ] = None
    gated_launch_port: A[
        Optional[int],
        "The port of the gated launch control server. When set, every rank blocks right after the distributed environment is initialized, before any sizable GPU allocation, until `POST /gate/activate` is sent to this port on the host of the first rank. This lets an external orchestrator defer the memory hungry part of startup to a safe window. Defaults to None, which disables the gate.",
    ] = None
    nnodes: A[int, "The number of nodes."] = 1
    node_rank: A[int, "The node rank."] = 0
    tp_size: A[
        int,
        Arg(
            help="The tensor parallelism size.",
            aliases=["--tensor-parallel-size"],
        ),
    ] = 1
    dcp_size: A[
        int,
        Arg(
            help="The decode context parallelism size.",
            aliases=["--decode-context-parallel-size"],
        ),
    ] = 1
    pp_size: A[
        int,
        Arg(
            help="The pipeline parallelism size.",
            aliases=["--pipeline-parallel-size"],
        ),
    ] = 1
    pp_max_micro_batch_size: A[
        Optional[int], "The maximum micro batch size in pipeline parallelism."
    ] = None
    pp_async_batch_depth: A[
        int,
        "The async batch depth of pipeline parallelism.",
    ] = 0
    dp_size: A[
        int,
        Arg(
            help="The data parallelism size.",
            aliases=["--data-parallel-size"],
        ),
    ] = 1
    load_balance_method: A[
        str,
        Arg(
            help="The load balancing strategy for data parallelism.",
            choices=[
                "auto",
                "round_robin",
                "follow_bootstrap_room",
                "total_requests",
                "total_tokens",
            ],
        ),
    ] = "auto"
    attn_cp_size: A[
        int,
        Arg(
            help="The attention context parallelism size.",
            aliases=["--attention-context-parallel-size"],
            resolvable=True,
        ),
    ] = 1
    moe_dp_size: A[
        int,
        Arg(
            help="The moe data parallelism size.",
            aliases=["--moe-data-parallel-size"],
        ),
    ] = 1
    dwdp_size: A[
        int,
        Arg(
            help="DWDP (Distributed Weight Data Parallelism) group size. "
            "When > 1, MoE prefill uses weight prefetch instead of token all-to-all. "
            "Must equal tp_size. Only supported with --disaggregation-mode null or prefill.",
        ),
    ] = 1
    dcp_comm_backend: A[
        str,
        Arg(
            help="Communication backend for the decode context-parallel (DCP) "
            "attention reduction: 'ag_rs' (AllGather + ReduceScatter), 'a2a' "
            "(fused NCCL All-to-All exchange of output+LSE + local Triton LSE "
            "combine), or 'fi_a2a' (FlashInfer MNNVL All-to-All kernel; requires "
            "SM90+ and MNNVL fabric memory, e.g. GB200 NVL72).",
            choices=["ag_rs", "a2a", "fi_a2a"],
            resolvable=True,
        ),
    ] = "ag_rs"
    dcp_replicate_q_proj: A[
        Optional[bool],
        Arg(
            help="For MLA decode context parallelism with the a2a/fi_a2a "
            "backend: replicate the Q projection so each DCP rank computes the "
            "full-head query locally (redundant projection compute), eliminating "
            "the per-layer head-dim all-gather of Q. Trades a small amount of "
            "extra GEMM for one fewer collective per layer. Use "
            "--no-dcp-replicate-q-proj to disable the model-specific default.",
            action=argparse.BooleanOptionalAction,
            resolvable=True,
        ),
    ] = None
    enable_prefill_cp: A[
        bool,
        "Enable context parallelism for the prefill phase. Select the layout with --cp-strategy.",
    ] = False
    cp_strategy: A[
        Optional[str],
        Arg(
            help="Sharding strategy for prefill CP. 'zigzag' is the former in-seq-split mode; 'interleave' is the former round-robin-split mode.",
            choices=("zigzag", "interleave"),
        ),
    ] = None
    # Split DSA GPU KV/indexer cache layers across CP ranks.
    enable_dsa_cache_layer_split: A[
        bool,
        "Split DSA (DeepSeek Sparse Attention) GPU KV/indexer cache layers across context-parallel ranks to reduce per-rank KV memory. Currently only supported with the mooncake transfer backend (mooncake / mooncake_tcp); mori/nixl support will be added later by the community.",
    ] = False
    enable_dsa_prefill_context_parallel: A[bool, Arg(no_cli=True)] = False
    dsa_prefill_cp_mode: A[str, Arg(no_cli=True)] = "round-robin-split"
    enable_prefill_context_parallel: A[bool, Arg(no_cli=True)] = False
    prefill_cp_mode: A[str, Arg(no_cli=True)] = "in-seq-split"
    enable_cp_decode_attn_tp: A[
        bool,
        "Enable attention tensor-parallel weight slicing during decode under context parallel (cp_size>1). Slices the replicated attention linears to the local CP partition, eliminating redundant decode GEMMs.",
    ] = False
    # DP attention
    enable_dp_attention: A[
        bool,
        Arg(
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.",
            resolvable=True,
        ),
    ] = False
    enable_dp_attention_local_control_broadcast: A[
        bool,
        "With DP-attention, send control messages to every DP group leader and broadcast within attn_tp_group instead of the full tp_group. Eliminates a costly all-ranks gloo sync on every scheduler iteration.",
    ] = False
    enable_dp_lm_head: A[
        bool,
        Arg(
            help="Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.",
            resolvable=True,
        ),
    ] = False
    enable_tp_lm_head_all_to_all: A[
        Optional[bool],
        Arg(
            help="Use all-to-all instead of TP all-gather followed by DP scatter "
            "for the TP-sharded LM head under DP attention. By default this is "
            "enabled only on decode-only PD nodes with pure DP attention "
            "(tp_size == dp_size > 1 and attn_cp_size == 1), and disabled on "
            "prefill-only and colocated nodes. Pass "
            "--no-enable-tp-lm-head-all-to-all to opt out. The path is "
            "incompatible with --enable-dp-lm-head; batches without an equal "
            "padded row count fall back to the existing all-gather path.",
            action=argparse.BooleanOptionalAction,
            resolvable=True,
        ),
    ] = None
    enable_attn_tp_input_scattered: A[
        bool,
        "Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.",
    ] = False
    enable_shared_experts_attn_tp: A[
        bool,
        "Shard shared expert weights across the attention TP group when using an expert-parallel all-to-all backend.",
    ] = False
    enable_dense_mlp_attn_tp: A[
        bool,
        "Shard dense MLP weights across the attention TP group under DP attention.",
    ] = False
    enable_layernorm_sp: A[
        bool,
        "Enable Megatron-style sequence parallelism (arXiv:2205.05198) for the "
        "LayerNorm/residual regions under pure tensor parallelism: the row-parallel "
        "all-reduce becomes reduce-scatter + all-gather, so LayerNorm runs on "
        "sequence-sharded activations with no extra communication volume. "
        "Prefill only; Qwen3 dense; requires tp_size > 1 and NVLink/NVSwitch.",
    ] = False
    disable_attn_tp_gather: A[
        bool,
        "Disable scheduler-side attn_tp_gather (the upstream SP path "
        "that pads num_tokens to attn_tp_size and pre-allocates a gathered "
        "buffer). Use for models that manage SP scatter/gather at the "
        "model level (e.g., perform their own all_gather/reduce_scatter "
        "inside attention) and do not consume the upstream gathered_buffer. "
        "Without this, the cuda graph runner pads num_tokens to attn_tp_size, "
        "which can cause kernel autotuners to select wrong-sized variants "
        "at small batches.",
    ] = False
    enable_p2p_check: A[
        bool,
        "Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
    ] = False

    # -------------------------------------------------------------------------
    # Expert parallelism
    # -------------------------------------------------------------------------
    ep_size: A[
        int,
        Arg(
            help="The expert parallelism size.",
            aliases=["--expert-parallel-size", "--ep"],
            resolvable=True,
        ),
    ] = 1
    moe_dense_tp_size: A[
        Optional[int],
        Arg(
            help="TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.",
            resolvable=True,
        ),
    ] = None
    ep_join_rank_offset: A[
        int,
        Arg(
            help=(
                "Global rank offset of an elastic EP joining group. Scale "
                "joiners must set this to the current effective EP size."
            ),
            cli_name="--elastic-ep-join-rank-offset",
        ),
    ] = 0
    elastic_ep_initial_size: A[
        Optional[int],
        "EP size used to define the immutable per-rank expert storage layout. "
        "Scale joiners must use the primary deployment's launch-time EP size.",
    ] = None
    max_ep_size: A[
        Optional[int],
        "Maximum EP size the server can scale to at runtime. Pre-allocates active-rank state and backend buffers to this size. Defaults to the launch-time world size.",
    ] = None

    # ---- derived: the quotients of the leaves above -------------------------
    #
    # Declared here, beside what they are computed from, because a namespace is
    # one file and one class. They are not annotated, so they are not dataclass
    # fields and `collect_input_fields` does not put them on the record -- which
    # is right: a quotient has no operator input to preserve, and the record is
    # what crosses a process boundary, so a width put there would be a stale
    # copy the moment an elastic scale-up restamps one. `derive_parallel_widths`
    # computes all six from the leaves above; `ParallelContext` installs a
    # property per declaration.
    attn_tp_size = Derived(
        doc="Attention tensor-parallel width: `tp_size` divided by the "
        "attention-DP and attention-CP dimensions.",
    )
    attn_dp_size = Derived(
        doc="Attention data-parallel width, normalised from the configured "
        "value (both an input to the derivation and an output of it).",
    )
    attn_dcp_size = Derived(
        doc="Decode context-parallel width inside the attention TP group.",
    )
    moe_ep_size = Derived(
        doc="MoE expert-parallel width, normalised from the configured value.",
    )
    moe_tp_size = Derived(
        doc="MoE tensor-parallel width: what is left of `tp_size` after the "
        "expert and MoE-DP dimensions.",
    )
    dcp_enabled = Derived(
        doc="Whether decode context parallelism is in play -- a group exists "
        "and is wider than one rank.",
    )

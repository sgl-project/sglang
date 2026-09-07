"""Config fields of the ``lora`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``lora`` bag, which is what ``get_lora()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import (
    List,
    Optional,
    Union,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.argparse_actions import LoRAPathAction
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.utils.common import (
    LORA_TARGET_ALL_MODULES,
    SUPPORTED_LORA_TARGET_MODULES,
)


@dataclasses.dataclass
class Lora:
    """Namespace ``lora``."""

    _NS_PATH = "lora"

    # -------------------------------------------------------------------------
    # LoRA
    # -------------------------------------------------------------------------
    enable_lora: A[
        Optional[bool],
        "Enable LoRA support for the model. This argument is automatically set to True if `--lora-paths` is provided for backward compatibility.",
    ] = None
    enable_lora_overlap_loading: A[
        Optional[bool],
        "Enable asynchronous LoRA weight loading in order to overlap H2D transfers with GPU compute. This should be enabled if you find that your LoRA workloads are bottlenecked by adapter weight loading, for example when frequently loading large LoRA adapters.",
    ] = None
    max_lora_rank: A[
        Optional[int],
        "The maximum rank of LoRA adapters. If not specified, it will be automatically inferred from the adapters provided in --lora-paths.",
    ] = None
    lora_target_modules: A[
        Optional[Union[set[str], List[str]]],
        Arg(
            help="The union set of all target modules where LoRA should be applied. If not specified, it will be automatically inferred from the adapters provided in --lora-paths. If 'all' is specified, all supported modules will be targeted.",
            nargs="*",
            choices=SUPPORTED_LORA_TARGET_MODULES + [LORA_TARGET_ALL_MODULES],
        ),
    ] = None
    lora_paths: A[
        Optional[Union[dict[str, str], List[dict[str, str]], List[str], List[LoRARef]]],
        Arg(
            help='The list of LoRA adapters to load. Each adapter must be specified in one of the following formats: <PATH> | <NAME>=<PATH> | JSON with schema {"lora_name":str,"lora_path":str,"pinned":bool}',
            action=LoRAPathAction,
            action_kwargs={"type": str, "nargs": "*"},
        ),
    ] = None
    max_loaded_loras: A[
        Optional[int],
        "If specified, it limits the maximum number of LoRA adapters loaded in CPU memory at a time. The value must be greater than or equal to `--max-loras-per-batch`.",
    ] = None
    max_loras_per_batch: A[
        int,
        "Maximum number of adapters for a running batch, include base-only request.",
    ] = 8
    lora_eviction_policy: A[
        str,
        Arg(
            help="LoRA adapter eviction policy when memory pool is full. 'lru': Least Recently Used (default, better cache efficiency). 'fifo': First-In-First-Out.",
            choices=["lru", "fifo"],
        ),
    ] = "lru"
    lora_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for multi-LoRA serving.",
            choices=["triton", "csgmv", "ascend", "torch_native"],
        ),
    ] = "csgmv"
    max_lora_chunk_size: A[
        Optional[int],
        Arg(
            help="Maximum chunk size for the ChunkedSGMV LoRA backend. Only used when --lora-backend is 'csgmv'. Choosing a larger value might improve performance.",
            choices=[16, 32, 64, 128],
        ),
    ] = 16
    experts_shared_outer_loras: A[
        Optional[bool],
        Arg(
            help="Force shared outer LoRA mode for MoE models. When set, w1/w3 lora_A and w2 lora_B are shared across experts (expert_dim=1). Use --no-experts-shared-outer-loras to force disable. By default this is auto-detected from adapter weights.",
            action=argparse.BooleanOptionalAction,
        ),
    ] = None
    lora_use_virtual_experts: A[
        bool,
        "Enable virtual expert computation for MoE models. When set, the model will use virtual expert computation.",
    ] = False
    lora_strict_loading: A[
        bool,
        Arg(
            help="Enable strict loading for LoRA adapters. When set, mismatched or missing keys in the adapter weights will raise an error.",
            action=argparse.BooleanOptionalAction,
        ),
    ] = False
    lora_drain_wait_threshold: A[
        float,
        "When any LoRA adapter request waits longer than this threshold (in seconds), the scheduler will selectively drain one running adapter to make room. This mitigates extreme tail latency under high or skewed workloads by preventing a small set of adapters from monopolizing batch slots. Set to 0 to disable draining (default).",
    ] = 0.0

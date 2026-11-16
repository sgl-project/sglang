"""Scoped configuration for the private MTP runner, never process-wide publish."""

import inspect
import logging
import os
from contextlib import ExitStack, contextmanager
from functools import wraps

from sglang.srt.arg_groups.arg_utils import namespace_of
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs


@contextmanager
def scoped_config(**fields):
    routes = namespace_of(ServerArgs)
    groups = {}
    for field, value in fields.items():
        path = routes[field]
        groups.setdefault(path, {})[field] = value
    with ExitStack() as stack:
        for path, values in groups.items():
            root, *parts = path.split(".")
            bag = get_context().config_bag(root)
            for part in parts:
                bag = getattr(bag, part)
            stack.enter_context(bag.override(**values))
        yield


def mtp_config(owner):
    return dict(
        speculative_algorithm="EAGLE",
        speculative_draft_model_path=owner.server_args.model_path,
        speculative_draft_model_revision=owner.server_args.revision,
        speculative_eagle_topk=1,
        speculative_num_steps=1,
        speculative_num_draft_tokens=8,
        speculative_draft_attention_backend="fa3",
    )


def validate_recipe():
    cfg = get_context().config_leaf
    for name in (
        "enable_lora",
        "enable_two_batch_overlap",
        "enable_unified_memory",
        "speculative_dflash_draft_kv_ring",
    ):
        if cfg(name):
            raise ValueError(f"DMTP does not support {name}")
    if cfg("page_size") != 1 or cfg("quantization") is not None:
        raise ValueError("DMTP requires page size 1 and unquantized weights")
    if cfg("speculative_num_draft_tokens") != 8:
        raise ValueError(
            "DMTP requires eight Target slots, not the checkpoint's DF block size"
        )
    if cfg("speculative_draft_attention_backend") != "fa3":
        raise ValueError("DMTP requires FA3 draft attention")
    capacity = cfg("max_running_requests")
    if capacity is None or not 1 <= capacity <= 32:
        raise ValueError("DMTP requires an explicit request capacity in 1..32")


def validate_mtp_target(config):
    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

    if is_qwen_qsa(config):
        raise ValueError(
            "DMTP private MTP KV commit does not yet support QSA "
            "compressed/indexer/ring state; use the upstream QSA path"
        )


def adapt_legacy_fa3_kernel(kernel):
    """Only the inactive new option can be omitted for a legacy FA3 kernel."""
    if getattr(kernel, "_dmtp_legacy_fa3_compat", False):
        return kernel
    parameters = inspect.signature(kernel).parameters
    if "only_qv" in parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    ):
        return kernel

    @wraps(kernel)
    def compatible(*args, only_qv=False, **kwargs):
        if only_qv is not False:
            raise NotImplementedError("Legacy FA3 does not implement only_qv=True")
        return kernel(*args, **kwargs)

    compatible._dmtp_legacy_fa3_compat = True
    return compatible


def install_legacy_fa3_compat():
    if os.getenv("SGLANG_DMTP_LEGACY_FA3_COMPAT", "0") != "1":
        return
    from sglang.kernels.ops.attention.flash_attention_v3 import _load_fa3_kernels

    kernels = _load_fa3_kernels()
    for name in ("flash_attn_with_kvcache", "flash_attn_varlen_func"):
        original = kernels[name]
        kernels[name] = adapt_legacy_fa3_kernel(original)
        if kernels[name] is not original:
            logging.getLogger(__name__).warning(
                "DMTP legacy FA3 compatibility (%s): omit only_qv=False; active only_qv remains unsupported",
                name,
            )

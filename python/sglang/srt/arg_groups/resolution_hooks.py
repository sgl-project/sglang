"""Out-of-tree replacement for a named step of the resolution pipeline.

`run_resolution_pipeline` calls its steps by name, hardcoded, with no
per-step dispatch through `self` -- there is nothing on `ServerArgs` left to
subclass in order to change how one step decides. This is the replacement
for that: a decorator that wraps whatever currently runs under a given name,
and a dispatcher the pipeline calls instead of the bare function. The name
itself is never spelled out at the call site -- `run_hook` reads it off the
function it was handed -- so there is exactly one place a step's name is
written by hand: the whitelist below, and whatever a downstream registrant
passes to the decorator.

Whitelisted names only, the same discipline `Arg(resolvable=True)` uses for
declarable fields: this project has already been bitten once by an
unqualified name collision in this exact pipeline (`_parse_cuda_graph_config`
and `_handle_cuda_graph_config` merged under one rename and the dispatcher
called itself). A name that is not on the list fails loudly at import time,
not silently at the call site three modules away.

Every step takes exactly `(server_args)`, `handle_hardware_runtime_validation`
included -- it does not read `server_args` (see the comment at its
definition), but it takes the parameter anyway so `run_hook` never has to
special-case an arity. An override's own signature always matches: `def
mine(server_args, previous)`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List

# Every step this pipeline runs by name that a downstream package may replace.
# Add a name here only alongside the call site's own switch to `run_hook` --
# an entry with no matching `run_hook(...)` call is a name nothing will ever
# look up. `test_every_whitelisted_hook_has_a_call_site` in
# test_resolution_hook_registry.py holds the two directions of this equal by
# construction, so this list cannot drift from `pipeline.py` silently.
#
# `handle_offload_compatibility` runs at two different points in the pipeline
# (once before model-specific adjustments, once after -- see the comments at
# its call sites). An override registered for it applies at both, identically;
# there is no way to target "just the second call" through this mechanism,
# because the name is all `run_hook` has to key on.
_OVERRIDABLE_HOOKS: FrozenSet[str] = frozenset(
    {
        "handle_mega_moe",
        "handle_return_hidden_states_mode",
        "handle_media_url_security",
        "handle_hicache_ratio_default",
        "handle_offload_compatibility",
        "validate_prefill_decode_interval",
        "default_unset_prefill_decode_interval",
        "validate_sampling_mask_max_tokens",
        "validate_prefill_cp_platform",
        "handle_hardware_runtime_validation",
        "handle_model_source_paths",
        "handle_multimodal",
        "handle_ssl_validation",
        "handle_asr_validation",
        "handle_deprecated_args",
        "handle_prefill_delayer_env_compat",
        "handle_missing_default_values",
        "handle_expert_pack",
        "handle_pd_disaggregation",
        "validate_prefill_only_disable_kv_cache_args",
        "handle_decode_context_parallelism",
        "apply_inkling_prefill_cuda_graph_default",
        "apply_muse_glimmer_prefill_cuda_graph_max_bs_default",
        "handle_dwdp",
        "handle_cuda_graph_config",
        "apply_glm5_chunked_prefill_default",
        "handle_hpu_backends",
        "handle_cpu_backends",
        "handle_npu_backends",
        "handle_mps_backends",
        "handle_xpu_backends",
        "handle_symm_mem_device_support",
        "handle_platform_defaults",
        "handle_gpu_memory_settings",
        "handle_model_specific_adjustments",
        "handle_deterministic_inference",
        "handle_attention_backend_compatibility",
        "disable_prefill_cuda_graph_for_deepseek_trtllm_mla",
        "handle_mamba_backend",
        "handle_int8_mamba_checkpoint",
        "handle_linear_attn_backend",
        "apply_glm5_prefill_cuda_graph_policy",
        "handle_kv4_compatibility",
        "handle_mxfp8_kv_cache_compatibility",
        "handle_amd_specifics",
        "handle_nccl_pre_warm",
        "handle_grammar_backend",
        "handle_multi_item_scoring",
        "handle_prefill_only_disable_kv_cache",
        "handle_hicache",
        "handle_data_parallelism",
        "handle_load_balance_method",
        "handle_context_parallelism",
        "handle_moe_kernel_config",
        "handle_a2a_moe",
        "handle_eplb_and_dispatch",
        "handle_expert_distribution_metrics",
        "handle_elastic_ep",
        "validate_experimental_sgl_marlin",
        "handle_speculative_decoding",
        "handle_layernorm_sp",
        "validate_cutedsl_a2a_token_budget",
        "handle_load_format",
        "handle_encoder_disaggregation",
        "handle_tokenizer_batching",
        "handle_environment_variables",
        "handle_cache_compatibility",
        "handle_page_major_kv_layout",
        "handle_unified_memory_pool",
        "handle_dllm_inference",
        "handle_crash_dump_env",
        "handle_debug_utils",
        "handle_other_validations",
        "handle_model_capability_adjustments",
        "validate_deepep_v2_speculative_draft",
        "validate_deepep_v2_dispatch_token_budget",
    }
)

# name -> registered overrides, oldest first. Each takes `(server_args,
# previous)`, where `previous` is the callable it wraps -- the built-in on
# the first registration, the previous registrant's own wrapper on every one
# after. Process-global as a `dict[str, list]` so a test isolates it the way
# test_model_overrides.py isolates `_MODEL_OVERRIDE_FNS`:
# `patch.dict(..., clear=True)`.
_HOOKS: Dict[str, List[Callable[[Any, Callable[[Any], None]], None]]] = {}


def register_resolution_hook(name: str):
    """Replace (or wrap) the pipeline step named ``name``.

    The decorated function is called as ``fn(server_args, previous)``.
    ``previous`` is a plain ``server_args -> None`` callable: the built-in
    step on the first registration for this name, or the previous
    registrant's own wrapper on every registration after that. Call it to
    run what would have run without this override -- the `super().handle_x()`
    shape, expressed as an explicit argument instead of a method-resolution
    lookup, because there is no class hierarchy here for `super()` to walk.
    Not calling it is a full replacement.

    Registering twice for the same name does not replace the first
    registration; it wraps it. The **last** registration is outermost --
    runs first, and decides whether/when its `previous` (everything
    registered before it, down to the built-in) runs at all. Two downstream
    packages that both target the same name compose in whichever order they
    happened to import in; if that order matters to you, make one of them
    import the other first.

    This changes *what* runs at the step's existing position in the
    pipeline, never *when*: the call site in `pipeline.py` is unmoved, so
    every other step keeps the order it already had. A wrapped step's own
    declarations reach `resolution_result` the same way any declaration
    does -- only readers from this position onward see them; a step that
    already ran and read the old value before this one's `previous` (or the
    built-in) declared its replacement has already made its decision on it.
    """
    if name not in _OVERRIDABLE_HOOKS:
        raise ValueError(
            f"{name!r} is not an overridable resolution hook; the "
            f"overridable set is {sorted(_OVERRIDABLE_HOOKS)}. A new entry "
            "needs a matching `run_hook(...)` call at the step's site in "
            "pipeline.py, not just a name here."
        )

    def decorator(fn):
        _HOOKS.setdefault(name, []).append(fn)
        return fn

    return decorator


def run_hook(builtin: Callable[[Any], None], server_args: Any) -> None:
    """Run ``builtin`` -- the registered chain if anything overrode it under
    its name, ``builtin`` directly otherwise.

    The name is ``builtin.__name__``, not a second argument: the call site
    already has the function in scope (a function-local import, same as
    every other step), and spelling the name out again next to it is the
    exact "two copies that can silently disagree" shape this project keeps
    removing elsewhere. ``builtin`` must therefore be a plain, named
    function -- every real call site is -- not a lambda or a bound method.

    Called from the step's fixed position in `run_resolution_pipeline`. The
    chain is rebuilt from the registry on every call rather than cached at
    registration time, because at registration time (import time, before any
    `ServerArgs` exists) there is no `server_args` yet and the first
    registrant's `previous` cannot be bound to anything real until a call
    actually happens.
    """
    step = builtin
    for fn in _HOOKS.get(builtin.__name__, ()):
        step = _bind(fn, step)
    step(server_args)


def _bind(fn, previous):
    def wrapped(server_args):
        fn(server_args, previous)

    return wrapped

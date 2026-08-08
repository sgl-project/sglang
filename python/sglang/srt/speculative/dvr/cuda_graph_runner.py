"""DVR-local CUDA graph integration for draft and target verify."""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager

import torch

from sglang.srt.distributed import get_moe_ep_group, get_moe_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import DecodeCudaGraphRunner
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import get_bool_env_var

_OVERRIDE_PLAN_CACHE = object()
logger = logging.getLogger(__name__)


def _resolve_dvr_backends(backend, forward_mode=ForwardMode.DECODE):
    """Resolve full-attention leaves and the optional linear-state adapter."""

    leaves = []
    state_adapter = None

    def visit(current, *, collect_attention=True):
        nonlocal state_adapter
        candidate = getattr(current, "dvr_state_adapter", None)
        if candidate is not None:
            if state_adapter is not None and state_adapter is not candidate:
                raise RuntimeError(
                    "DVR resolved multiple linear-state adapters in one backend."
                )
            state_adapter = candidate

        if isinstance(current, HybridLinearAttnBackend):
            visit(current.linear_attn_backend, collect_attention=False)
            visit(current.full_attn_backend, collect_attention=collect_attention)
        elif isinstance(current, HybridAttnBackend):
            selected = current._select_backend(forward_mode)
            visit(selected, collect_attention=collect_attention)
            if state_adapter is None:
                for candidate in (current.decode_backend, current.prefill_backend):
                    if candidate is not selected:
                        visit(candidate, collect_attention=False)
        elif isinstance(current, TboAttnBackend):
            visit(current.primary, collect_attention=collect_attention)
            for child in current.children:
                visit(child, collect_attention=collect_attention)
        elif hasattr(current, "attn_backends"):
            # Upstream multi-step draft wrappers expose their per-step backend
            # instances here. Validate and tune the actual attention leaves.
            for child in current.attn_backends:
                visit(child, collect_attention=collect_attention)
        elif collect_attention and candidate is None:
            leaves.append(current)

    visit(backend)
    return leaves, state_adapter


def validate_dvr_attention_backend(
    backend, forward_mode=ForwardMode.DECODE, *, phase="draft decode"
):
    """Return supported full-attention leaves after wrapper resolution."""

    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )
    from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

    leaves, state_adapter = _resolve_dvr_backends(backend, forward_mode)
    if not leaves:
        raise RuntimeError(f"DVR {phase} did not resolve an attention backend.")
    for leaf in leaves:
        if isinstance(leaf, FlashAttentionBackend):
            if leaf.fa_impl_ver == 3:
                continue
            raise RuntimeError(
                f"DVR {phase} requires FlashAttention 3, got "
                f"fa_impl_ver={leaf.fa_impl_ver}."
            )
        if not isinstance(leaf, TritonAttnBackend):
            raise RuntimeError(
                f"DVR {phase} supports only Triton or FA3 full-attention "
                f"backends, got {type(leaf).__name__}."
            )
    return leaves, state_adapter


def _fast_decode_overrides(backend, model_runner, buffer_cache):
    """Return fast-decode state for the DVR-supported Triton and FA3 leaves."""

    from sglang.srt.layers.attention.flashattention_backend import (
        FlashAttentionBackend,
    )

    plan_cache = buffer_cache.setdefault(_OVERRIDE_PLAN_CACHE, {})
    plan_key = (id(backend), id(model_runner))
    if plan_key in plan_cache:
        return plan_cache[plan_key]

    overrides = []
    leaves, _ = validate_dvr_attention_backend(backend)
    for leaf in leaves:
        if isinstance(leaf, FlashAttentionBackend):
            overrides.append((leaf, "num_splits", 0))
            continue

        split_tile_size = model_runner.server_args.triton_attention_split_tile_size
        max_kv_splits = model_runner.server_args.triton_attention_num_kv_splits
        if split_tile_size is not None:
            max_kv_splits = (
                leaf.max_context_len + split_tile_size - 1
            ) // split_tile_size
        max_kv_splits = min(leaf.max_kv_splits, max_kv_splits)
        overrides.append((leaf, "enable_deterministic", False))
        for name in (
            "cuda_graph_attn_logits",
            "cuda_graph_attn_lse",
            "cuda_graph_swa_attn_logits",
        ):
            buffer = getattr(leaf, name, None)
            if buffer is not None:
                overrides.append((leaf, name, buffer[:, :, :max_kv_splits]))

        buffers = buffer_cache.setdefault(leaf, {})
        for name in (
            "cuda_graph_num_kv_splits",
            "cuda_graph_window_num_kv_splits",
        ):
            buffer = getattr(leaf, name, None)
            if buffer is not None:
                draft_buffer = buffers.get(name)
                if draft_buffer is None or draft_buffer.shape != buffer.shape:
                    draft_buffer = torch.full_like(buffer, max_kv_splits)
                    buffers[name] = draft_buffer
                overrides.append((leaf, name, draft_buffer))
        overrides.extend(
            (
                (leaf, "max_kv_splits", max_kv_splits),
                (leaf, "split_tile_size", split_tile_size),
                (
                    leaf,
                    "static_kv_splits",
                    get_bool_env_var(
                        "SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS", "false"
                    ),
                ),
            )
        )
    plan_cache[plan_key] = overrides
    return overrides


def _iter_decode_custom_all_reduce_groups(model_runner):
    candidate_groups = [getattr(model_runner, "tp_group", None)]
    for get_group in (get_moe_ep_group, get_moe_tp_group):
        try:
            candidate_groups.append(get_group())
        except AssertionError:
            pass

    seen = set()
    for group in candidate_groups:
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        yield group


@contextmanager
def _draft_custom_allreduce_enabled(group):
    """Enable custom all-reduce only while DVR draft graphs are captured."""

    communicator = group.ca_comm
    if communicator is None and group.world_size > 1:
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )

        try:
            communicator = dispatch_custom_allreduce(
                group=group.cpu_group, device=group.device
            )(group=group.cpu_group, device=group.device)
            group.ca_comm = communicator
            # Deterministic target execution keeps the lazily-created
            # communicator disabled after draft graph capture.
            communicator.disabled = True
            if hasattr(communicator, "original_disabled"):
                communicator.original_disabled = True
        except Exception as exc:
            logger.warning("Setup Custom allreduce failed with %s.", exc)
            communicator = None

    if communicator is None or not getattr(communicator, "full_nvlink", False):
        yield False
        return

    previous_disabled = communicator.disabled
    communicator.disabled = False
    try:
        yield True
    finally:
        communicator.disabled = previous_disabled


def _clear_draft_kernel_policy_caches():
    """Discard cached choices that read the process deterministic policy."""

    from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
        should_enable_swap_ab,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_moe_configs,
    )

    get_moe_configs.cache_clear()
    should_enable_swap_ab.cache_clear()


@contextmanager
def _draft_batch_invariant_disabled():
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    was_enabled = is_batch_invariant_mode_enabled()
    if was_enabled:
        disable_batch_invariant_mode()
    try:
        yield
    finally:
        if was_enabled:
            enable_batch_invariant_mode()


@contextmanager
def dvr_draft_decode_context(
    model_runner,
    buffer_cache,
    *,
    capture: bool = False,
    extra_attn_backends=(),
    tune_attention: bool = True,
):
    """Select the performance-first policy for provisional DVR draft work.

    Backend-local attention metadata follows the draft policy during capture
    and replay. Process-wide operator and collective policy changes only during
    serialized graph capture.
    """

    with ExitStack() as stack:
        patched_keys = set()

        def patch_attr(obj, attr_name, value):
            if obj is None or not hasattr(obj, attr_name):
                return
            key = (id(obj), attr_name)
            if key not in patched_keys:
                patched_keys.add(key)
                stack.callback(setattr, obj, attr_name, getattr(obj, attr_name))
            setattr(obj, attr_name, value)

        if capture:
            # Attention graph buffers may be allocated inside this context. Never
            # retain a plan resolved before capture has finished initializing them.
            buffer_cache.pop(_OVERRIDE_PLAN_CACHE, None)
            stack.callback(buffer_cache.pop, _OVERRIDE_PLAN_CACHE, None)
            _clear_draft_kernel_policy_caches()
            stack.callback(_clear_draft_kernel_policy_caches)

            # Upstream kernel selectors treat ServerArgs and the dispatcher as
            # one deterministic policy. Temporarily switch both while compiling
            # provisional draft graphs, then restore the deterministic target
            # execution policy before serving starts. Do not record this
            # transient compiler state as a persistent ServerArgs override.
            seen_server_args = set()
            for server_args in (model_runner.server_args, get_server_args()):
                if id(server_args) in seen_server_args:
                    continue
                seen_server_args.add(id(server_args))
                previous = server_args.enable_deterministic_inference
                stack.callback(
                    object.__setattr__,
                    server_args,
                    "enable_deterministic_inference",
                    previous,
                )
                object.__setattr__(server_args, "enable_deterministic_inference", False)

            deterministic_env = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE
            deterministic_env_was_set = deterministic_env.is_set()
            deterministic_env_value = deterministic_env.get()
            if deterministic_env_was_set:
                stack.callback(deterministic_env.set, deterministic_env_value)
            else:
                stack.callback(deterministic_env.clear)
            # Keep the environment and dispatcher aligned with the transient
            # ServerArgs policy while compiling the draft graph.
            deterministic_env.set(False)

        if tune_attention:
            backends = tuple(
                backend
                for backend in (
                    model_runner.attn_backend,
                    *(extra_attn_backends or ()),
                )
                if backend is not None
            )
            if not backends:
                raise RuntimeError(
                    "DVR draft decode did not receive an attention backend."
                )
            for backend in backends:
                for owner, name, value in _fast_decode_overrides(
                    backend, model_runner, buffer_cache
                ):
                    patch_attr(owner, name, value)

        if capture:
            # Deterministic target prefill/verify keeps custom all-reduce
            # disabled. Only capture it into provisional draft graphs.
            if getattr(
                model_runner.server_args,
                "dvr_enable_draft_custom_all_reduce",
                False,
            ):
                for group in _iter_decode_custom_all_reduce_groups(model_runner):
                    stack.enter_context(_draft_custom_allreduce_enabled(group))
            stack.enter_context(_draft_batch_invariant_disabled())

        yield


class DVRDraftDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Ordinary decode graph used only for provisional self-draft tokens."""

    # The target ModelRunner already owns and initialized this attention backend.
    # A second initialization would replace graph-stable metadata buffers.
    owns_attention_graph_state = False

    def _resolve_capture_layout(self):
        return ForwardMode.DECODE, 1


class DVRTargetVerifyCudaGraphRunner(DecodeCudaGraphRunner):
    """Target-verify graph runner for DVR self-draft and DVR-EAGLE.

    TARGET_VERIFY is an EXTEND forward and therefore uses target prefill
    kernels. DecodeCudaGraphRunner is reused only for its fixed number of tokens
    per request and speculative metadata plumbing. DVR adds the causal verifier
    and GDN state-input metadata here rather than changing prefill graph
    semantics or attaching execution hooks to the spec_info data object.
    """

    def get_spec_info(self, num_tokens: int):
        from sglang.srt.speculative.spec_info import create_dummy_verify_input

        spec_info = create_dummy_verify_input(
            spec_algorithm=self.model_runner.spec_algorithm,
            server_args=self.model_runner.server_args,
            custom_mask=None,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            is_draft_worker=False,
        )
        assert spec_info is not None
        if self.model_runner.spec_algorithm.is_dvr_eagle():
            spec_info.hidden_states = torch.zeros(
                (num_tokens, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )
        return spec_info

# SPDX-License-Identifier: Apache-2.0
"""Cache-DiT precision guard for timestep-quantized diffusion transformers."""

from functools import wraps

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def set_quant_precision_guard(transformer):
    """Clear both CFG histories and force computation when quant precision changes."""
    modules = getattr(transformer, "modules", None)
    if modules is None:
        return
    attention_policies = []
    policies = []
    for module in modules():
        if getattr(module, "use_mxfp4_attention", False):
            attention_policies.append(module.attention_quant_description)
        scheme = getattr(module, "scheme", None)
        target = getattr(scheme, "precision_policy_target", None)
        if target is not None and not any(scheme is p for p in policies):
            policies.append(scheme)
    if not (attention_policies or policies) or getattr(
        transformer.forward, "_sglang_precision_guard", False
    ):
        return

    from sglang.multimodal_gen.runtime.layers.quantization.mxfp4_utils import (
        default_mxfp4_policy,
        resolve_precision,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        get_forward_context_or_none,
    )

    original_forward = transformer.forward
    previous_signature = previous_batch = previous_step = force_step = None

    @wraps(original_forward)
    def guarded_forward(*args, **kwargs):
        nonlocal previous_signature, previous_batch, previous_step, force_step
        context = get_forward_context_or_none()
        if context is None:
            return original_forward(*args, **kwargs)
        batch, step = context.forward_batch, context.current_timestep
        num_steps = getattr(batch, "num_inference_steps", 0)
        if batch is not previous_batch or (
            previous_step is not None and step < previous_step
        ):
            previous_signature = force_step = None
        default_policy = default_mxfp4_policy(num_steps)
        signature = tuple(
            resolve_precision(description, "fa", "MXFP4", default_policy)
            for description in attention_policies
        ) + tuple(scheme.resolve_precision(default_policy) for scheme in policies)
        contexts = [
            transformer._context_manager.get_context(name)
            for name in transformer._context_names
        ]
        changed = previous_signature is not None and signature != previous_signature
        if previous_signature is None or changed:
            # Drop both CFG histories without restarting the expert's local step counter.
            for cached_context in contexts:
                cached_context.clear_buffers()
                for calibrator in (
                    *cached_context.get_calibrators(),
                    *cached_context.get_cfg_calibrators(),
                ):
                    if calibrator is not None:
                        calibrator.reset_cache()
            force_step = step
            if changed:
                logger.info(
                    "step=%d quant precision changed; invalidating cached residuals",
                    step,
                )
        previous_signature, previous_batch, previous_step = signature, batch, step
        force = (
            step == force_step or step < 2 or (num_steps > 0 and step >= num_steps - 2)
        )
        masks = [c.cache_config.steps_computation_mask for c in contexts]
        if force:
            for cached_context in contexts:
                # SCM takes precedence over similarity, including static reuse policies.
                cached_context.cache_config.steps_computation_mask = [1] * max(
                    num_steps, cached_context.executed_steps + 1
                )
        try:
            return original_forward(*args, **kwargs)
        finally:
            for cached_context, mask in zip(contexts, masks):
                cached_context.cache_config.steps_computation_mask = mask

    guarded_forward._sglang_precision_guard = True
    transformer.forward = guarded_forward

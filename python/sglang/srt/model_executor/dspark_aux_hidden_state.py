"""Runtime DSpark target-layer capture for standard residual-stream models."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Sequence

from sglang.srt.layers.aux_hidden_states import (
    RUNTIME_AUX_CAPTURE_LOGITS_ATTR,
    RUNTIME_AUX_HIDDEN_STATES_ATTR,
)

_RUNTIME_CAPTURE_INSTALLED_ATTR = "_sglang_runtime_dspark_capture_installed"


def _forward_batch_from_call(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    forward_batch_index: int,
):
    if "forward_batch" in kwargs:
        return kwargs["forward_batch"]
    return args[forward_batch_index]


def _wrap_forward_preserving_signature(
    module,
    wrapper_factory: Callable[[Callable, inspect.Signature], Callable],
) -> None:
    original_forward = module.forward
    signature = inspect.signature(original_forward)
    wrapped_forward = wrapper_factory(original_forward, signature)
    wrapped_forward.__signature__ = signature
    module.forward = wrapped_forward


def attach_runtime_dspark_aux_hidden_state_capture(
    model, layer_ids: Sequence[int]
) -> None:
    """Attach DSpark capture without changing the model implementation.

    Compatible models expose ``model.layers`` and use ``LayerCommunicator`` for
    residual-stream transitions. Pipeline parallelism is intentionally rejected
    because the DSpark consumer expects every requested feature on one rank.
    """
    if getattr(model, _RUNTIME_CAPTURE_INSTALLED_ATTR, False):
        raise RuntimeError("Runtime DSpark auxiliary capture is already installed.")
    if hasattr(model, "capture_aux_hidden_states"):
        raise TypeError(
            f"Model {model.__class__.__name__} has a model-managed auxiliary "
            "capture contract and cannot use the runtime fallback."
        )

    pp_group = getattr(model, "pp_group", None)
    if pp_group is None or not getattr(pp_group, "is_last_rank", False):
        raise TypeError(
            f"Model {model.__class__.__name__} does not expose a supported PP group."
        )
    if int(getattr(pp_group, "world_size", 1)) > 1:
        raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")

    if layer_ids is None:
        raise ValueError("DSPARK requires explicit layer_ids for aux hidden capture.")
    target_layer_ids = [int(layer_id) for layer_id in layer_ids]
    if not target_layer_ids:
        raise ValueError(
            "DSPARK requires at least one target layer for aux hidden capture."
        )
    if target_layer_ids != sorted(set(target_layer_ids)):
        raise ValueError(
            "DSPARK target layer_ids must be unique and strictly increasing."
        )

    layer_model = getattr(model, "model", None)
    layers = getattr(layer_model, "layers", None)
    config = getattr(model, "config", None)
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    invalid = [
        layer_id for layer_id in target_layer_ids if not 0 <= layer_id < num_layers
    ]
    if invalid:
        raise ValueError(
            "DSPARK target layer_ids are outside the decoder range: "
            f"{invalid}; num_hidden_layers={num_layers}."
        )
    if layer_model is None or layers is None or num_layers == 0:
        raise TypeError(
            f"Model {model.__class__.__name__} is not compatible with runtime "
            "DSpark residual-stream capture."
        )

    start_layer = int(getattr(layer_model, "start_layer", 0))
    end_layer = int(getattr(layer_model, "end_layer", num_layers))
    if start_layer != 0 or end_layer != num_layers:
        raise NotImplementedError("DSPARK aux hidden capture requires PP=1.")

    # The first transition resets the per-forward accumulator. Each requested
    # non-final layer is captured after the next layer completes residual
    # addition/all-reduce in prepare_attn.
    prepare_layer_ids = {0}
    prepare_layer_ids.update(
        layer_id + 1 for layer_id in target_layer_ids if layer_id + 1 < num_layers
    )
    for layer_id in sorted(prepare_layer_ids):
        communicator = getattr(layers[layer_id], "layer_communicator", None)
        if communicator is None or not hasattr(
            communicator, "capture_last_layer_output"
        ):
            raise TypeError(
                f"Model {model.__class__.__name__} layer {layer_id} does not use "
                "the LayerCommunicator capture contract."
            )

    if not hasattr(model, "logits_processor"):
        raise TypeError(
            f"Model {model.__class__.__name__} does not expose a logits processor."
        )

    for layer_id in sorted(prepare_layer_ids):
        communicator = layers[layer_id].layer_communicator
        original_prepare_attn = communicator.prepare_attn
        previous_layer_id: Optional[int] = (
            layer_id - 1 if layer_id - 1 in target_layer_ids else None
        )

        def prepare_attn_with_capture(
            hidden_states,
            residual,
            forward_batch,
            *args,
            _communicator=communicator,
            _original_prepare_attn=original_prepare_attn,
            _layer_id=layer_id,
            _previous_layer_id=previous_layer_id,
            **kwargs,
        ):
            if _layer_id == 0:
                setattr(forward_batch, RUNTIME_AUX_HIDDEN_STATES_ATTR, [])
            hidden_states, residual = _original_prepare_attn(
                hidden_states, residual, forward_batch, *args, **kwargs
            )
            if _previous_layer_id is not None:
                accumulator = getattr(
                    forward_batch, RUNTIME_AUX_HIDDEN_STATES_ATTR, None
                )
                if accumulator is None:
                    raise RuntimeError(
                        "Runtime DSpark auxiliary capture was not initialized."
                    )
                _communicator.capture_last_layer_output(
                    residual, forward_batch, accumulator
                )
            return hidden_states, residual

        communicator.prepare_attn = prepare_attn_with_capture

    if target_layer_ids[-1] == num_layers - 1:
        final_layer = layers[num_layers - 1]

        def wrap_final_layer(original_forward, signature):
            parameters = list(signature.parameters)
            forward_batch_index = parameters.index("forward_batch")

            def final_layer_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                hidden_states, residual = output
                forward_batch = _forward_batch_from_call(
                    args, kwargs, forward_batch_index
                )
                accumulator = getattr(
                    forward_batch, RUNTIME_AUX_HIDDEN_STATES_ATTR, None
                )
                if accumulator is None:
                    raise RuntimeError(
                        "Runtime DSpark auxiliary capture was not initialized."
                    )
                accumulator.append(
                    hidden_states if residual is None else hidden_states + residual
                )
                return output

            return final_layer_forward

        _wrap_forward_preserving_signature(final_layer, wrap_final_layer)

    # Body-only prefill graphs need aux tensors in their explicit outputs so
    # graph replay preserves them. The top-level logits processor unwraps this
    # tuple for ordinary model.forward calls.
    def wrap_layer_model(original_forward, signature):
        parameters = list(signature.parameters)
        forward_batch_index = parameters.index("forward_batch")

        def layer_model_forward(*args, **kwargs):
            forward_batch = _forward_batch_from_call(args, kwargs, forward_batch_index)
            output = original_forward(*args, **kwargs)
            accumulator = getattr(forward_batch, RUNTIME_AUX_HIDDEN_STATES_ATTR, None)
            if accumulator is None:
                raise RuntimeError(
                    "Runtime DSpark auxiliary capture produced no accumulator."
                )
            return output, accumulator

        return layer_model_forward

    _wrap_forward_preserving_signature(layer_model, wrap_layer_model)

    model.capture_aux_hidden_states = True
    setattr(model.logits_processor, RUNTIME_AUX_CAPTURE_LOGITS_ATTR, True)
    setattr(model, _RUNTIME_CAPTURE_INSTALLED_ATTR, True)

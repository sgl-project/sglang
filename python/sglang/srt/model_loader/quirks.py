# Per-architecture weight-loading quirks registered with
# :class:`WeightQuirkRegistry`. Importing this module is enough to register
# them — :mod:`sglang.srt.model_loader.auto_loader` looks them up by
# ``model.__class__.__name__``.
"""Standard cross-architecture weight quirks.

A quirk is a ``(model, mapper) -> mapper`` function: it gets the model
instance and the caller-supplied :class:`WeightsMapper`, and returns a
(possibly extended) mapper. Quirks are composed on top of the base mapper
in registration order via :meth:`WeightsMapper.__or__`.

To add a new quirk, decorate a function with ``@WeightQuirkRegistry.register``
and pass the architecture name(s) it applies to. Subclasses inherit quirks
registered on their bases via the MRO walk in
:func:`WeightQuirkRegistry.compose`.
"""

from __future__ import annotations

from sglang.srt.model_loader.auto_loader import (
    WeightQuirkRegistry,
    WeightsMapper,
)

# Quantized checkpoints sometimes ship FP8 scales under the legacy names.
# Lift the rename to the registry so every Llama-derived model gets it for
# free without touching its ``load_weights``.
_FP8_SCALE_RENAME = WeightsMapper(
    orig_to_new_suffix={
        ".activation_scale": ".input_scale",
        ".weight_scale_inv": ".weight_scale",
    }
)


@WeightQuirkRegistry.register(
    [
        "LlamaForCausalLM",
        "Phi3ForCausalLM",
        "InternLM3ForCausalLM",
        "IQuestCoderForCausalLM",
        "ArceeForCausalLM",
        "GraniteForCausalLM",
    ]
)
def _llama_family_scale_rename(model, mapper: WeightsMapper) -> WeightsMapper:
    return mapper | _FP8_SCALE_RENAME

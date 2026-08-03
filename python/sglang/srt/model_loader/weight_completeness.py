# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Container, Iterable


def unloaded_required_params(
    parameter_names: Iterable[str],
    loaded: Container[str],
    is_optional: Callable[[str], bool],
) -> set[str]:
    """Return required model parameters that a checkpoint did not load."""
    return {
        name for name in parameter_names if name not in loaded and not is_optional(name)
    }


def load_and_verify_weights(
    model, weights, *, is_full_checkpoint: bool = True
) -> object:
    """Load weights and, for migrated models, verify a full checkpoint."""
    loaded_params = model.load_weights(weights)
    if not is_full_checkpoint or not getattr(model, "verify_weights_on_load", False):
        return loaded_params

    if loaded_params is None:
        raise TypeError(
            f"{type(model).__name__}.load_weights() must return loaded parameter "
            "names when verify_weights_on_load is enabled"
        )

    is_optional = getattr(model, "is_optional_weight", lambda _name: False)
    missing = unloaded_required_params(
        (name for name, _ in model.named_parameters()),
        loaded_params,
        is_optional,
    )
    if missing:
        raise RuntimeError(
            f"Some weights are not initialized from checkpoints: {sorted(missing)}"
        )
    return loaded_params

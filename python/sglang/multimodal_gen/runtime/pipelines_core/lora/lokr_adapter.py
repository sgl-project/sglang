# SPDX-License-Identifier: Apache-2.0
"""Load full-matrix Z-Image LoKr checkpoints into compact dynamic adapters."""

from collections import defaultdict
from collections.abc import Callable, Mapping

import torch

from sglang.multimodal_gen.runtime.layers.lora.lokr import LoKrAdapter, LoKrTerm


def load_lokr_adapter(
    state_dict: Mapping[str, torch.Tensor],
    layers: Mapping[str, torch.nn.Module],
    param_mapping: Callable,
    device: torch.device | str,
) -> dict[str, LoKrAdapter]:
    """Validate every tensor and target before the pipeline replaces its cached adapter.

    Full-matrix LyCORIS LoKr uses scale=1, irrespective of the saved alpha.
    Decomposed factors, Tucker convolutions, DoRA, and mixed LoRA/LoKr files
    need different semantics and are rejected, rather than partially applied.
    """
    if not state_dict:
        raise ValueError("LoKr checkpoint is empty")
    groups = defaultdict(dict)
    for name, weight in state_dict.items():
        module, _, suffix = name.removeprefix("diffusion_model.").rpartition(".")
        if suffix not in {"lokr_w1", "lokr_w2", "alpha"}:
            raise ValueError(f"Unsupported LoKr tensor: {name}")
        if suffix in groups[module]:
            raise ValueError(f"Duplicate LoKr tensor: {name}")
        groups[module][suffix] = weight

    terms = defaultdict(list)
    widths = {}
    for source, weights in groups.items():
        if not {"lokr_w1", "lokr_w2"} <= weights.keys():
            raise ValueError(f"LoKr requires both full-matrix factors for {source}")
        if source in layers:
            target, index = source, None
        else:
            # Reuse the model's logical-projection mapping (Q/K/V, gate/up).
            target, index, _ = param_mapping(source + ".lora_B")
            target = target.removesuffix(".lora_B")
        if target not in layers:
            raise ValueError(f"LoKr target {source!r} does not match a supported layer")
        base = layers[target].base_layer
        input_size = base.input_size
        output_size = base.output_size
        start, width = 0, output_size
        if index is not None:
            sizes = getattr(base, "output_sizes", None)
            if sizes is None or not 0 <= index < len(sizes):
                raise ValueError(f"Unsupported fused LoKr target: {source}")
            start, width = sum(sizes[:index]), sizes[index]
        term = LoKrTerm(weights["lokr_w1"], weights["lokr_w2"], start)
        if (term.output_size, term.input_size) != (width, input_size):
            raise ValueError(
                f"LoKr shape mismatch for {source}: "
                f"{(term.output_size, term.input_size)} != {(width, input_size)}"
            )
        terms[target].append(term)
        widths[target] = output_size
    return {
        name + ".lokr": LoKrAdapter(parts, widths[name]).to(device)
        for name, parts in terms.items()
    }

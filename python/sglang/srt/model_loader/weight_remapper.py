from typing import Dict, Optional, Iterable, Tuple
import dataclasses
import torch

WeightsMapping = Dict[str, Optional[str]]


@dataclasses.dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns.

    Supports:
    - orig_to_new_substr: Replace substrings in the key
    - orig_to_new_prefix: Replace prefixes in the key
    - orig_to_new_suffix: Replace suffixes in the key
    - orig_to_new_dict: Exact key replacement

    If a replacement value is None, the weight is skipped.
    """

    orig_to_new_substr: WeightsMapping = dataclasses.field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = dataclasses.field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = dataclasses.field(default_factory=dict)
    orig_to_new_dict: WeightsMapping = dataclasses.field(default_factory=dict)

    def _map_name(self, key: str) -> Optional[str]:
        if key in self.orig_to_new_dict:
            return self.orig_to_new_dict[key]

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None
                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None
                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None
                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Apply the mapping to an iterable of (key, tensor) pairs."""
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )


weights_mapper_registry: Dict[str, WeightsMapper] = {}

def register_weights_mapper(model_class_name: str, weights_mapper: WeightsMapper):
    weights_mapper_registry[model_class_name] = weights_mapper

def get_weights_mapper(model_class_name: str) -> Optional[WeightsMapper]:
    return weights_mapper_registry.get(model_class_name, None)
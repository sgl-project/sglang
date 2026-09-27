# SPDX-License-Identifier: Apache-2.0
"""Local, opt-in prepared component capability; never decoded from the wire.

Loaders own recipes and construction. Contracts only audit representations and
adapt finalized schemas/process-local state. Pipeline admission is separate.
"""

from typing import Any, Protocol

import msgspec
import torch


class ComponentStateContract(Protocol):
    contract_id: str

    def validate_supported(self, frozen, *, attention: str) -> None: ...

    def adapt_meta_schema(self, model: torch.nn.Module) -> torch.nn.Module: ...

    def finalize_after_import(self, model: torch.nn.Module) -> torch.nn.Module: ...


class PreparedComponent(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    structural_name: str
    architecture: str
    loader_cls: type
    recipe: Any
    contract: ComponentStateContract  # Trusted local object, not wire data.
    attention_backend: Any

    def loader(self):
        # Keep the exact loader selected by ComponentLoader.for_component_type.
        # Materialization must not rediscover it from a mutable registry.
        loader = self.loader_cls()
        loader.component_type = self.structural_name
        loader.component_architecture = self.architecture
        return loader

    def load_ordinary(self):
        return self.loader().load_prepared(
            self.recipe, attention_backend=self.attention_backend
        )

    def build_meta(self):
        model = self.loader().build_prepared_meta(
            self.recipe, attention_backend=self.attention_backend
        )
        return self.contract.adapt_meta_schema(model)

    def finalize_after_import(self, model):
        result = self.contract.finalize_after_import(model)
        if result is not model:
            raise ValueError("State contract must not replace an imported component")
        return result

    def apply_config(self, server_args):
        self.loader().apply_prepared_config(self.recipe, server_args)

    def consumed_files(self):
        return self.loader().prepared_checkpoint_files(self.recipe)

    def fingerprint_fields(self):
        return {
            "contract": self.contract.contract_id,
            "loader": f"{self.loader_cls.__module__}.{self.loader_cls.__qualname__}",
            "structural_name": self.structural_name,
            "architecture": self.architecture,
            "attention": str(self.attention_backend),
            "recipe": self.loader().prepared_fingerprint(self.recipe),
        }

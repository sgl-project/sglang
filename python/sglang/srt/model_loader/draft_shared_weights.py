"""Reuse the model's sharing setters before allocating draft vocabulary weights."""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
from torch import nn


def apply_draft_weight_sharing(model, embed, head, *, is_eagle3=False):
    """The worker and loading planner use the same model sharing interface."""
    if is_eagle3 and not getattr(model, "load_lm_head_from_target", False):
        model.set_embed(embed)
    else:
        model.set_embed_and_head(embed, head)


def plan_draft_weight_sharing(model, embed, head, *, is_eagle3=False):
    """Record the existing setters' bindings without changing either model.

    Copy only module registrations, not tensor storage. Meta markers stand in
    for the target parameters, so the setters cannot expose target weights to
    draft loading or postprocessing. Delegation, overridden setters and custom
    module paths are resolved by the same code used for final sharing.
    """
    copies = {}

    def copy_modules(module):
        if module is None:
            return None
        if id(module) not in copies:
            cloned = copy(module)
            copies[id(module)] = cloned
            cloned._parameters = module._parameters.copy()
            cloned._buffers = module._buffers.copy()
            cloned._modules = {
                name: copy_modules(child) for name, child in module._modules.items()
            }
        return copies[id(module)]

    markers = {}
    sources = {}

    def marker(source):
        if source is None:
            return None
        if id(source) not in markers:
            value = nn.Parameter(
                torch.empty_like(source, device="meta"), requires_grad=False
            )
            markers[id(source)] = value
            sources[id(value)] = source
        return markers[id(source)]

    planned_model = copy_modules(model)
    apply_draft_weight_sharing(
        planned_model, marker(embed), marker(head), is_eagle3=is_eagle3
    )
    return {
        name: sources[id(parameter)]
        for name, parameter in planned_model.named_parameters(remove_duplicate=False)
        if id(parameter) in sources
    }


def draft_shares_embedding(model, embedding, *, is_eagle3=False):
    """Ask the existing setter whether it replaces the draft's input embedding."""
    bindings = plan_draft_weight_sharing(model, embedding, None, is_eagle3=is_eagle3)
    return any(source is embedding for source in bindings.values())


@dataclass(frozen=True)
class DraftSharingContext:
    target: nn.Module
    is_eagle3: bool
    token_map: bool
    embed: Optional[torch.Tensor]
    head: Optional[torch.Tensor]


def _skip_shared_weight(*args, **kwargs):
    pass


@dataclass
class DraftWeightLoading:
    context: DraftSharingContext
    device: torch.device
    deferred_modules: set[nn.Module] = field(default_factory=set)
    shared_modules: set[nn.Module] = field(default_factory=set)
    bindings: dict = field(default_factory=dict)
    sources: tuple = (None, None)

    def needs_checkpoint_names(self, model):
        return any(
            callable(getattr(module, "prepare_draft_weight_loading", None))
            for module in model.modules()
        )

    def _compatible(self, module, source):
        device_index = self.device.index
        if device_index is None and self.device.type == "cuda":
            device_index = torch.cuda.current_device()
        if (
            source is None
            or source.is_meta
            or source.device.type != self.device.type
            or source.device.index != device_index
            or source.shape != module.weight.shape
            or source.dtype != module.weight.dtype
        ):
            return False

        # Wrapped layers expose their base layer through modules() as well.
        for target_module in self.context.target.modules():
            if getattr(target_module, "weight", None) is not source:
                continue
            if type(getattr(target_module, "quant_method", None)) is not type(
                module.quant_method
            ):
                continue
            fields = ("tp_size", "shard_indices", "embedding_dim", "org_vocab_size")
            if all(
                getattr(module, name, None) == getattr(target_module, name, None)
                for name in fields
            ):
                return True
        return False

    def prepare(self, model, checkpoint_names=None):
        # These hooks calculate the same state that load_weights() otherwise
        # learns while reading checkpoint keys. No second sharing policy exists.
        ready = not self.needs_checkpoint_names(model) or checkpoint_names is not None
        if checkpoint_names is not None:
            for module in model.modules():
                prepare = getattr(module, "prepare_draft_weight_loading", None)
                if callable(prepare):
                    prepare(checkpoint_names)

        self.sources = (self.context.embed, self.context.head)
        if self.context.token_map:
            # A sliced output head needs its own storage. Do not offer it as a
            # shared source, even when the target ties it to the embedding.
            self.sources = (self.sources[0], None)
        planned = (
            plan_draft_weight_sharing(
                model, *self.sources, is_eagle3=self.context.is_eagle3
            )
            if ready
            else {}
        )
        modules = dict(model.named_modules(remove_duplicate=False))
        selected = {}
        conflicting = set()
        for path, source in planned.items():
            module_path, _, parameter_name = path.rpartition(".")
            module = modules.get(module_path)
            if (
                parameter_name != "weight"
                or module not in self.deferred_modules
                or not self._compatible(module, source)
            ):
                continue
            parameter_id = id(module.weight)
            if parameter_id in selected and selected[parameter_id] is not source:
                conflicting.add(parameter_id)
            selected[parameter_id] = source
            self.bindings[path] = source
        for parameter_id in conflicting:
            selected.pop(parameter_id, None)

        # A tied parameter can also belong to a separately quantized head. That
        # consumer still needs real storage for its own loading/postprocessing.
        for module in model.modules():
            if (
                module not in self.deferred_modules
                and getattr(module, "quant_method", None) is not None
            ):
                for parameter in module.parameters(recurse=False):
                    selected.pop(id(parameter), None)
        self.bindings = {
            path: source
            for path, source in self.bindings.items()
            if id(model.get_parameter(path)) in selected
        }

        # Allocate only draft-owned weights, directly on the final device.
        # Rebind all registered aliases together, preserving tied parameters.
        replacements = {}
        for module in self.deferred_modules:
            parameter = module.weight
            if id(parameter) in selected:
                parameter.weight_loader = _skip_shared_weight
                parameter._shared_draft_weight = True
                self.shared_modules.add(module)
            elif id(parameter) not in replacements:
                replacement = nn.Parameter(
                    torch.empty_like(parameter, device=self.device),
                    requires_grad=parameter.requires_grad,
                )
                replacement.__dict__.update(parameter.__dict__)
                replacements[id(parameter)] = replacement
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                replacement = replacements.get(id(parameter))
                if replacement is not None:
                    module.register_parameter(name, replacement)

    def finish(self, model):
        if self.bindings:
            actual = plan_draft_weight_sharing(
                model, *self.sources, is_eagle3=self.context.is_eagle3
            )
            if any(
                actual.get(path) is not source for path, source in self.bindings.items()
            ):
                raise RuntimeError(
                    "Draft sharing changed after reading checkpoint weights"
                )
        replacements = {
            id(model.get_parameter(path)): source
            for path, source in self.bindings.items()
        }
        # Target parameters enter the draft tree only after all loading and
        # quantization postprocessing have completed.
        for module in model.modules():
            for name, parameter in list(module._parameters.items()):
                replacement = replacements.get(id(parameter))
                if replacement is not None:
                    module.register_parameter(name, replacement)
        unbound = [
            name for name, parameter in model.named_parameters() if parameter.is_meta
        ]
        if unbound:
            raise RuntimeError(f"Unbound draft parameters after loading: {unbound}")


_draft_context: ContextVar[Optional[DraftSharingContext]] = ContextVar(
    "draft_shared_weights_context", default=None
)
_draft_loading: ContextVar[Optional[DraftWeightLoading]] = ContextVar(
    "draft_weight_loading", default=None
)


@contextmanager
def draft_shared_weights_scope(
    target: nn.Module, *, is_eagle3=False, token_map=False
) -> Iterator[None]:
    from sglang.srt.speculative.pp_draft_embedding import resolve_target_embed_and_head

    embed, head = resolve_target_embed_and_head(target)
    token = _draft_context.set(
        DraftSharingContext(target, is_eagle3, token_map, embed, head)
    )
    try:
        yield
    finally:
        _draft_context.reset(token)


@contextmanager
def draft_weight_loading_context(
    device: torch.device, *, enabled: bool
) -> Iterator[Optional[DraftWeightLoading]]:
    context = _draft_context.get()
    loading = (
        DraftWeightLoading(context, device)
        if enabled and context is not None and device.type == "cuda"
        else None
    )
    token = _draft_loading.set(loading)
    try:
        yield loading
    finally:
        _draft_loading.reset(token)


def defer_draft_vocab_weights(module: nn.Module) -> bool:
    loading = _draft_loading.get()
    if loading is None:
        return False
    loading.deferred_modules.add(module)
    return True


def is_shared_draft_module(module: nn.Module) -> bool:
    loading = _draft_loading.get()
    return loading is not None and module in loading.shared_modules

"""Plan draft weight sharing without allocating duplicate vocabulary storage."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class DraftSharedWeightSpec:
    embedding: Optional[str] = "model.embed_tokens"
    lm_head: Optional[str] = "lm_head"
    resolver: Optional[str] = None
    delegate: Optional[str] = None
    check_hidden_size: bool = False
    allow_none_hidden_size: bool = False


def draft_shared_weight_spec(**kwargs):
    """Declare the module paths a sharing setter replaces.

    A resolver returns (share_embedding, share_head). It receives checkpoint
    names before loading, or None to inspect the model's state after loading.
    Overriding a setter does not inherit its declaration automatically.
    """
    spec = DraftSharedWeightSpec(**kwargs)

    def decorate(setter):
        setter._draft_shared_weight_spec = spec
        return setter

    return decorate


def _resolve_spec(model, is_eagle3, setter_name=None):
    if setter_name is None:
        setter_name = (
            "set_embed"
            if is_eagle3 and not getattr(model, "load_lm_head_from_target", False)
            else "set_embed_and_head"
        )
    setter = getattr(model, setter_name, None)
    spec = getattr(setter, "_draft_shared_weight_spec", None)
    if spec is None:
        return None
    if spec.delegate is not None:
        child = model.get_submodule(spec.delegate)
        child_setter = setter_name
        if not hasattr(child, child_setter):
            child_setter = "set_embed"
        resolved = _resolve_spec(child, is_eagle3, child_setter)
        if resolved is None:
            return None
        owner, prefix, child_spec = resolved
        return owner, spec.delegate + "." + prefix, child_spec
    return model, "", spec


def draft_shared_weight_paths(model, is_eagle3=False, checkpoint_names=None):
    resolved = _resolve_spec(model, is_eagle3)
    if resolved is None:
        return None
    owner, prefix, spec = resolved
    embedding, head = spec.embedding, spec.lm_head
    config = getattr(owner.config, "text_config", owner.config)
    if spec.check_hidden_size:
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size and not (
            spec.allow_none_hidden_size and target_hidden_size is None
        ):
            embedding = None
    if spec.resolver is not None:
        share_embedding, share_head = getattr(owner, spec.resolver)(checkpoint_names)
        embedding = embedding if share_embedding else None
        head = head if share_head else None
    # A tied head uses the input embedding's storage and cannot be skipped
    # independently when the draft needs its own local embedding under PP.
    if getattr(config, "tie_word_embeddings", False):
        head = None
    return (
        prefix + embedding if embedding is not None else None,
        prefix + head if head is not None else None,
    )


@dataclass(frozen=True)
class DraftSharingContext:
    target: nn.Module
    is_eagle3: bool
    token_map: bool


def _skip_shared_weight(*args, **kwargs):
    pass


@dataclass
class DraftWeightLoading:
    context: DraftSharingContext
    device: torch.device
    deferred_modules: set[nn.Module] = field(default_factory=set)
    shared_modules: set[nn.Module] = field(default_factory=set)
    bindings: dict = field(default_factory=dict)

    def needs_checkpoint_names(self, model):
        resolved = _resolve_spec(model, self.context.is_eagle3)
        return resolved is not None and resolved[2].resolver is not None

    def _compatible(self, module, source):
        if (
            source is None
            or source.is_meta
            or source.device.type != self.device.type
            or source.device.index != torch.cuda.current_device()
            or source.shape != module.weight.shape
            or source.dtype != module.weight.dtype
        ):
            return False
        from sglang.srt.lora.layers import unwrap_lora_layer

        for target_module in self.context.target.modules():
            target_module = unwrap_lora_layer(target_module)
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
        paths = draft_shared_weight_paths(
            model, self.context.is_eagle3, checkpoint_names
        )
        # Formats without readable key metadata use normal device allocation.
        if self.needs_checkpoint_names(model) and checkpoint_names is None:
            paths = None
        selected = {}
        if paths is not None:
            from sglang.srt.speculative.pp_draft_embedding import (
                resolve_target_embed_and_head,
            )

            sources = resolve_target_embed_and_head(self.context.target)
            for index, (path, source) in enumerate(zip(paths, sources)):
                # A token-mapped output head needs new sliced storage.
                if path is None or (index == 1 and self.context.token_map):
                    continue
                module = model.get_submodule(path)
                if module not in self.deferred_modules or not self._compatible(
                    module, source
                ):
                    continue
                selected[id(module.weight)] = source
                self.bindings[path] = source

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
            if id(model.get_submodule(path).weight) in selected
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
        paths = draft_shared_weight_paths(model, self.context.is_eagle3)
        if self.bindings and (paths is None or not self.bindings.keys() <= set(paths)):
            raise RuntimeError("Draft sharing changed after reading checkpoint weights")
        replacements = {
            id(model.get_submodule(path).weight): source
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
    token = _draft_context.set(DraftSharingContext(target, is_eagle3, token_map))
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

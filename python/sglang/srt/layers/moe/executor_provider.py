"""Optional platform providers for FusedMoE-compatible expert modules.

Register selectors from a general plugin before constructing a model. A selector
returns an executor factory and optional routing requirement, or declines with
None. Existing expert construction, weight loading and execution remain the
caller's contract; this is not a new execution interface or kernel backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.moe.topk import TopKOutputFormat
    from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackendLike


@dataclass(frozen=True, kw_only=True)
class MoeExecutorRequest:
    """Construction arguments and active settings for one expert module.

    expert_args is a read-only shallow snapshot of the actual constructor
    arguments. Quantization objects are not copied and must not be mutated.
    Model identity is optional metadata, not an upstream family-to-executor map.
    """

    expert_args: Mapping[str, Any]
    platform_key: str
    params_dtype: torch.dtype
    tp_size: int
    moe_tp_size: int
    ep_size: int
    a2a_backend: MoeA2ABackend
    runner_backend: MoeRunnerBackendLike
    is_draft: bool
    model_type: str | None = None
    model_class: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expert_args", MappingProxyType(dict(self.expert_args))
        )


@dataclass(frozen=True)
class MoeExecutorSelection:
    factory: Callable[..., torch.nn.Module]
    required_topk_format: TopKOutputFormat | None = None

    def topk_output_format(
        self, model_requirement: TopKOutputFormat | None
    ) -> TopKOutputFormat | None:
        """Agree on hard requirements; None preserves the existing auto policy."""
        requirement = self.required_topk_format
        if requirement is None:
            return model_requirement
        if model_requirement is not None and model_requirement != requirement:
            raise ValueError(
                "MoE provider routing requirement conflicts with the model: "
                f"provider={requirement.name}, model={model_requirement.name}"
            )
        return requirement


@dataclass(frozen=True)
class MoeProviderSpec:
    name: str
    platform_key: str
    selector: Callable[[MoeExecutorRequest], MoeExecutorSelection | None]


class _MoeProviderRegistry:
    def __init__(self) -> None:
        self.providers: dict[tuple[str, str], MoeProviderSpec] = {}
        self.registration_errors: dict[str, str] = {}
        self.frozen = False

    def register(self, spec: MoeProviderSpec) -> None:
        if (
            not isinstance(spec.name, str)
            or not spec.name.strip()
            or not isinstance(spec.platform_key, str)
            or not spec.platform_key.strip()
            or not callable(spec.selector)
        ):
            raise ValueError(
                "MoE provider needs a name, platform key and callable selector"
            )
        key = (spec.platform_key, spec.name)
        if self.frozen or key in self.providers:
            reason = (
                "register before the first expert resolution"
                if self.frozen
                else "provider already registered"
            )
            message = f"MoE provider {spec.name!r} ({spec.platform_key}): {reason}"
            # General plugin loading catches exceptions. Retain this error so a
            # partially registered platform cannot silently select a winner.
            self.registration_errors[spec.platform_key] = message
            raise ValueError(message)
        self.providers[key] = spec

    def candidates(self, platform_key: str) -> list[MoeProviderSpec]:
        self.frozen = True
        if platform_key in self.registration_errors:
            raise RuntimeError(self.registration_errors[platform_key])
        return [
            spec
            for key, spec in sorted(self.providers.items())
            if key[0] == platform_key
        ]

    def select(
        self, request: MoeExecutorRequest, candidates: list[MoeProviderSpec]
    ) -> MoeExecutorSelection | None:
        match: tuple[str, MoeExecutorSelection] | None = None
        for spec in candidates:
            try:
                selection = spec.selector(request)
                if selection is None:
                    continue
                if not isinstance(selection, MoeExecutorSelection) or not callable(
                    selection.factory
                ):
                    raise TypeError("selector must return MoeExecutorSelection or None")
                if selection.required_topk_format is not None:
                    from sglang.srt.layers.moe.topk import TopKOutputFormat

                    if not isinstance(selection.required_topk_format, TopKOutputFormat):
                        raise TypeError(
                            "required_topk_format must be a TopKOutputFormat"
                        )
            except Exception as exc:
                raise RuntimeError(
                    f"MoE provider {spec.name!r} failed for {request.model_class!r} "
                    f"({request.model_type!r})"
                ) from exc
            if match is not None:
                raise RuntimeError(
                    "Multiple MoE providers accepted the same request: "
                    f"{match[0]}, {spec.name}"
                )
            match = (spec.name, selection)
        return match[1] if match is not None else None


# Plugin metadata has process lifetime, like the general plugin registrations.
# Requests, executor instances and execution resources are never stored here.
_registry = _MoeProviderRegistry()


def register_moe_provider(
    name: str,
    *,
    platform_key: str,
    selector: Callable[[MoeExecutorRequest], MoeExecutorSelection | None],
) -> None:
    """Register a selector for the active platform's device_name."""
    _registry.register(MoeProviderSpec(name, platform_key, selector))


def resolve_moe_executor(
    expert_args: Mapping[str, Any],
    *,
    model_type: str | None = None,
    model_class: str | None = None,
) -> MoeExecutorSelection:
    """Resolve once during construction; never fall back after a provider error."""
    from sglang.srt.plugins import _current_plugin_source, load_plugins

    if _current_plugin_source.get() is not None:
        raise RuntimeError("Resolve MoE executors after plugin registration completes")
    load_plugins()

    from sglang.srt.platforms import current_platform

    platform_key = current_platform.device_name
    candidates = _registry.candidates(platform_key)
    if candidates:
        import torch

        from sglang.srt.layers.moe.utils import (
            get_moe_a2a_backend,
            get_moe_runner_backend,
        )
        from sglang.srt.runtime_context import get_flags, get_parallel

        parallel = get_parallel()
        request = MoeExecutorRequest(
            expert_args=expert_args,
            platform_key=platform_key,
            params_dtype=expert_args.get("params_dtype") or torch.get_default_dtype(),
            tp_size=parallel.tp_size,
            moe_tp_size=parallel.moe_tp_size,
            ep_size=parallel.moe_ep_size,
            a2a_backend=get_moe_a2a_backend(),
            runner_backend=get_moe_runner_backend(),
            is_draft=get_flags().moe.in_speculative_scope,
            model_type=model_type,
            model_class=model_class,
        )
        selection = _registry.select(request, candidates)
        if selection is not None:
            return selection

    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class

    return MoeExecutorSelection(get_moe_impl_class(expert_args.get("quant_config")))

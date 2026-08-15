"""JSON-driven config resolution for the MoE LoRA MoE engine.

The engine's execution strategy — kernel families, fusion shape, overlap
windows, route builder, and launch tiles per (architecture, factor layout,
activation, phase, batch shape) — lives in data: one JSON file per
architecture key space under ``configs/`` (``gb300.json`` serves every
SM100-family device, ``h200.json`` serves SM90).  This module only loads,
validates, and matches; it encodes no per-model knowledge.  Every value in
the shipped files is a sweep winner (see the 2026-08 best-config campaign);
re-tuning for a new model or geometry means running
``benchmark/kernels/lora_moe/tune_lora_config.py`` and dropping its output
into ``SGLANG_LORA_MOE_CONFIG_DIR`` — never editing code.

Resolution is three-tier, and every fallback is logged once:

1. the architecture file's scenario list, first match wins (rows are ordered
   most-specific-first, so token/rank tiers read top to bottom);
2. the file's ``fallback`` scenario for geometry outside the tuned domain
   (hidden size or local expert count beyond ``domain``) or a batch no row
   matches: the serial materialized reference on DeepGEMM — the one shape
   proven never worse than stock — rather than extrapolating tuned overlap
   plans;
3. ``default.json`` when no file covers the architecture key space at all.

A malformed scenario fails closed at load time through the execution-plan
and launch-config validators; a bad config can crash startup but can never
silently serve a wrong plan.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from typing import Mapping

from sglang.srt.lora.moe.execution_plan import (
    ActivationFamily,
    EarlyOverlap,
    FinalizeFamily,
    LateOverlap,
    LoraAFamily,
    LoraBFamily,
    MiddleFamily,
    MoeLoraExecutionPlan,
    RouteBuilderFamily,
)
from sglang.srt.lora.moe.launch_config import MoeLoraLaunchConfig

ProviderKey = (
    str  # 'deepgemm' | 'cutedsl' | 'deepgemm_contiguous' | 'cutedsl_contiguous'
)

logger = logging.getLogger(__name__)


_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")


class Phase(str, Enum):
    DECODE = "decode"
    PREFILL = "prefill"


class DeviceArchitecture(str, Enum):
    H200 = "h200"
    GB300 = "gb300"


def architecture_for_capability(major: int, minor: int) -> DeviceArchitecture:
    if not isinstance(major, int) or not isinstance(minor, int):
        raise TypeError("compute capability must be integer major/minor")
    if major == 9:
        return DeviceArchitecture.H200
    if major >= 10:
        return DeviceArchitecture.GB300
    raise ValueError(f"unsupported compute capability sm{major}{minor}")


@dataclass(frozen=True, slots=True)
class ConfigInput:
    capability_major: int
    capability_minor: int
    is_shared_outer: bool
    activation: ActivationFamily
    mode: Phase
    num_tokens: int
    active_rank: int
    hidden_size: int
    num_local_experts: int
    has_active_lora: bool
    use_cuda_graph: bool

    def __post_init__(self) -> None:
        architecture_for_capability(self.capability_major, self.capability_minor)
        if not isinstance(self.is_shared_outer, bool):
            raise TypeError("is_shared_outer must be bool")
        if not isinstance(self.activation, ActivationFamily):
            raise TypeError("activation must be ActivationFamily")
        if not isinstance(self.mode, Phase):
            raise TypeError("mode must be Phase")
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.active_rank <= 0:
            raise ValueError("active_rank must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive")
        if not isinstance(self.has_active_lora, bool):
            raise TypeError("has_active_lora must be bool")
        if not isinstance(self.use_cuda_graph, bool):
            raise TypeError("use_cuda_graph must be bool")


@dataclass(frozen=True, slots=True)
class ConfigChoice:
    key: str
    provider: ProviderKey
    plan: MoeLoraExecutionPlan
    launch_config: MoeLoraLaunchConfig

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("config choice key must be non-empty")
        if self.provider not in (
            "deepgemm",
            "cutedsl",
            "deepgemm_contiguous",
            "cutedsl_contiguous",
        ):
            raise ValueError(
                "runner provider must be 'deepgemm', 'cutedsl', "
                "'deepgemm_contiguous', or 'cutedsl_contiguous'"
            )
        self.launch_config.validate_for_plan(self.plan)

    @property
    def name(self) -> str:
        return self.key


def _build_plan(
    scenario_plan: Mapping[str, object],
    activation: ActivationFamily,
    is_shared_outer: bool,
) -> MoeLoraExecutionPlan:
    """Materialize one scenario's ``plan`` section into a validated plan.

    The JSON carries builder arguments (family names, windows, route) plus
    the boolean edge/tail flags; construction runs through the same spec
    validators the hand-written menus used, so an inconsistent combination
    raises here rather than at first forward.
    """
    from sglang.srt.lora.moe.execution_plan import (
        FactorContract,
        FactorLayout,
        FactorSite,
        FinalizeSpec,
        LoraASpec,
        LoraBSpec,
        MiddleSpec,
    )

    p = dict(scenario_plan)
    gate_a_family = LoraAFamily(p.pop("gate_a_family", "grouped"))
    down_a_family = LoraAFamily(p.pop("down_a_family", "grouped"))
    gate_b_family = LoraBFamily(p.pop("gate_b_family", "one_launch_sliced"))
    down_b_family = LoraBFamily(p.pop("down_b_family", "one_launch_sliced"))
    middle_family = MiddleFamily(p.pop("middle_family", "materialized"))
    finalize_family = FinalizeFamily(p.pop("finalize_family", "materialized"))
    early_overlap = EarlyOverlap(p.pop("early_overlap", "none"))
    late_overlap = LateOverlap(p.pop("late_overlap", "none"))
    route_builder = RouteBuilderFamily(p.pop("route_builder", "standard"))
    route_pdl = p.pop("route_pdl", None)
    flags = {
        name: bool(p.pop(name, False))
        for name in (
            "gate_a_to_b_pdl",
            "down_a_to_b_pdl",
            "base_gateup_to_middle_pdl",
            "down_b_scatter",
        )
    }
    if p:
        raise ValueError(f"unknown plan fields in config scenario: {sorted(p)}")

    gate_layout = (
        FactorLayout.TOKEN_MAJOR
        if gate_a_family is LoraAFamily.TOKEN_DEDUP_GROUPED
        else FactorLayout.PAIR_MAJOR
    )
    gate_b_contract = FactorContract(FactorSite.GATE_UP, False, gate_layout)
    down_b_contract = FactorContract(
        FactorSite.DOWN, is_shared_outer, FactorLayout.PAIR_MAJOR
    )
    consumes_gate_b = middle_family is MiddleFamily.B_ACTIVATION
    consumes_down_b = finalize_family is not FinalizeFamily.MATERIALIZED
    plan = MoeLoraExecutionPlan(
        gate_a=LoraASpec(
            FactorSite.GATE_UP, gate_a_family, is_shared_outer, gate_layout
        ),
        gate_b=(
            None
            if consumes_gate_b
            else LoraBSpec(FactorSite.GATE_UP, gate_b_family, False, gate_layout)
        ),
        middle=MiddleSpec(
            middle_family, activation, gate_b_contract if consumes_gate_b else None
        ),
        down_a=LoraASpec(
            FactorSite.DOWN, down_a_family, False, FactorLayout.PAIR_MAJOR
        ),
        down_b=(
            None
            if consumes_down_b
            else LoraBSpec(
                FactorSite.DOWN,
                down_b_family,
                is_shared_outer,
                FactorLayout.PAIR_MAJOR,
            )
        ),
        finalize=FinalizeSpec(
            finalize_family, down_b_contract if consumes_down_b else None
        ),
        early_overlap=early_overlap,
        late_overlap=late_overlap,
        route_builder=route_builder,
        route_pdl=route_pdl,
        down_a_to_b_pdl=flags["down_a_to_b_pdl"],
    )
    extra = {k: v for k, v in flags.items() if v and k != "down_a_to_b_pdl"}
    if extra:
        plan = replace(plan, **extra)
    return plan.validate_ownership(is_shared_outer)


def _load_config_file(name: str) -> dict | None:
    from sglang.srt.environ import envs

    override_dir = envs.SGLANG_LORA_MOE_CONFIG_DIR.get()
    for directory in filter(None, (override_dir, _CONFIG_DIR)):
        path = os.path.join(directory, f"{name}.json")
        if os.path.isfile(path):
            if directory == override_dir:
                logger.info(
                    "MoE LoRA config '%s' loaded from override dir %s", name, directory
                )
            return json.load(open(path))
    return None


@cache
def _config_table(architecture: DeviceArchitecture) -> dict:
    table = _load_config_file(architecture.value)
    if table is None:
        logger.warning(
            "no MoE LoRA config file for architecture %s; serving the "
            "conservative default config",
            architecture.value,
        )
        table = _load_config_file("default")
    if table is None:
        raise RuntimeError("MoE LoRA config files are missing from the package")
    return _validate_when_keys(table)


_KNOWN_WHEN_KEYS = frozenset(
    {"layout", "activation", "phase", "max_tokens", "max_rank", "min_local_experts"}
)


def _validate_when_keys(table: dict) -> dict:
    """Reject predicate keys this build does not understand.

    Silently ignoring an unknown key would make the row match MORE batches
    than its author intended (a typo, or a config written for a newer build
    with e.g. a ``quant`` dimension, served by an older one). Fail closed at
    load instead.
    """
    for section in ("scenarios", "fallback"):
        for row in table.get(section, ()):
            unknown = set(row["when"]) - _KNOWN_WHEN_KEYS
            if unknown:
                raise ValueError(
                    f"config row {row.get('name')!r} uses predicate keys this "
                    f"build does not understand: {sorted(unknown)}"
                )
    return table


def _match(
    when: Mapping[str, object],
    layout_shared: bool,
    activation: ActivationFamily,
    mode: Phase | None,
    num_tokens: int | None,
    active_rank: int | None,
    num_local_experts: int,
) -> bool:
    """Evaluate one scenario predicate.

    ``mode``/``num_tokens``/``active_rank`` of None mean "match statically"
    (used by choices_for, which must return every choice a later select
    could pick for this geometry)."""
    if "layout" in when and when["layout"] != (
        "shared" if layout_shared else "per_expert"
    ):
        return False
    if when.get("activation", activation.value) != activation.value:
        return False
    if mode is not None and "phase" in when and when["phase"] != mode.value:
        return False
    if num_tokens is not None and num_tokens > when.get("max_tokens", 1 << 30):
        return False
    if active_rank is not None and active_rank > when.get("max_rank", 1 << 30):
        return False
    if num_local_experts < when.get("min_local_experts", 0):
        return False
    return True


def _config_section(value: object) -> object:
    if isinstance(value, dict):
        return {k: (dict(v) if isinstance(v, dict) else v) for k, v in value.items()}
    return value


def _scenario_choice(
    architecture: DeviceArchitecture,
    scenario: Mapping[str, object],
    is_shared_outer: bool,
    activation: ActivationFamily,
) -> ConfigChoice:
    plan = _build_plan(scenario["plan"], activation, is_shared_outer)
    config = MoeLoraLaunchConfig(
        **{k: _config_section(v) for k, v in scenario.get("config", {}).items()}
    )
    layout_name = "shared" if is_shared_outer else "per_expert"
    key = f"{architecture.value}.{layout_name}.{scenario['name']}.{activation.value}"
    return ConfigChoice(key, scenario["provider"], plan, config)


def _in_domain(table: dict, hidden_size: int, num_local_experts: int) -> bool:
    domain = table.get("domain", {})
    return hidden_size <= domain.get("max_hidden", 1 << 30) and (
        num_local_experts <= domain.get("max_local_experts", 1 << 30)
    )


@cache
def _choices_for(
    architecture: DeviceArchitecture,
    is_shared_outer: bool,
    activation: ActivationFamily,
    hidden_size: int,
    num_local_experts: int,
) -> tuple[ConfigChoice, ...]:
    table = _config_table(architecture)
    layout_shared = is_shared_outer
    choices: list[ConfigChoice] = []
    if _in_domain(table, hidden_size, num_local_experts):
        for scenario in table["scenarios"]:
            if _match(
                scenario["when"],
                layout_shared,
                activation,
                None,
                None,
                None,
                num_local_experts,
            ):
                choices.append(
                    _scenario_choice(
                        architecture, scenario, is_shared_outer, activation
                    )
                )
    else:
        logger.info(
            "MoE LoRA geometry hidden=%d local_experts=%d is outside the "
            "tuned domain of config '%s'; serving the serial fallback",
            hidden_size,
            num_local_experts,
            architecture.value,
        )
    for scenario in table["fallback"]:
        if _match(
            scenario["when"],
            layout_shared,
            activation,
            None,
            None,
            None,
            num_local_experts,
        ):
            choices.append(
                _scenario_choice(architecture, scenario, is_shared_outer, activation)
            )
    return tuple(choices)


def choices_for(
    architecture: DeviceArchitecture,
    is_shared_outer: bool,
    activation: ActivationFamily,
    *,
    hidden_size: int,
    num_local_experts: int,
) -> tuple[ConfigChoice, ...]:
    """Every choice ``select_config`` may return for this geometry.

    The config backend binds a runner for each returned choice before the
    first forward, so selection never constructs plans on the hot path."""
    return _choices_for(
        architecture, is_shared_outer, activation, hidden_size, num_local_experts
    )


def select_config(config_input: ConfigInput) -> ConfigChoice:
    architecture = architecture_for_capability(
        config_input.capability_major, config_input.capability_minor
    )
    table = _config_table(architecture)
    layout_shared = config_input.is_shared_outer
    choices = _choices_for(
        architecture,
        config_input.is_shared_outer,
        config_input.activation,
        config_input.hidden_size,
        config_input.num_local_experts,
    )
    by_key = {c.key: c for c in choices}
    if _in_domain(table, config_input.hidden_size, config_input.num_local_experts):
        for scenario in table["scenarios"]:
            if _match(
                scenario["when"],
                layout_shared,
                config_input.activation,
                config_input.mode,
                config_input.num_tokens,
                config_input.active_rank,
                config_input.num_local_experts,
            ):
                layout_name = "shared" if layout_shared else "per_expert"
                return by_key[
                    f"{architecture.value}.{layout_name}."
                    f"{scenario['name']}.{config_input.activation.value}"
                ]
    for scenario in table["fallback"]:
        if _match(
            scenario["when"],
            layout_shared,
            config_input.activation,
            config_input.mode,
            config_input.num_tokens,
            config_input.active_rank,
            config_input.num_local_experts,
        ):
            layout_name = "shared" if layout_shared else "per_expert"
            return by_key[
                f"{architecture.value}.{layout_name}."
                f"{scenario['name']}.{config_input.activation.value}"
            ]
    raise RuntimeError(
        f"config '{architecture.value}' has no fallback for mode "
        f"{config_input.mode.value}; the config file is malformed"
    )

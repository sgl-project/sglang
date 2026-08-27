from __future__ import annotations

from sglang.multimodal_gen.test.server.accuracy_config import (
    ComponentType,
    should_skip_component,
)
from sglang.multimodal_gen.test.server.accuracy_utils import (
    extract_component_path_overrides,
)
from sglang.multimodal_gen.test.server.component_accuracy import COMPONENT_SPECS
from sglang.multimodal_gen.test.server.gpu_cases import (
    ONE_GPU_CASES,
    TWO_GPU_CASES,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase


def _component_accuracy_key(case: DiffusionTestCase, component: ComponentType) -> tuple:
    server_args = case.server_args
    component_paths = extract_component_path_overrides(server_args.extras)
    override_path = None
    for key in (component.value, *COMPONENT_SPECS[component].model_index_keys):
        if key in component_paths:
            override_path = component_paths[key]
            break

    return (
        component.value,
        server_args.model_path,
        override_path,
        server_args.num_gpus,
        server_args.tp_size,
        server_args.ulysses_degree,
        server_args.ring_degree,
        server_args.cfg_parallel,
    )


_COMPONENT_DUPLICATE_REASONS: dict[tuple[str, ComponentType], str] = {}


def _select_accuracy_cases(cases: list[DiffusionTestCase]) -> list[DiffusionTestCase]:
    selected: list[DiffusionTestCase] = []
    seen: dict[tuple, str] = {}
    for case in cases:
        if not case.run_component_accuracy_check:
            continue

        has_component_to_run = False
        for component in ComponentType:
            if should_skip_component(case, component):
                continue

            key = _component_accuracy_key(case, component)
            representative = seen.get(key)
            if representative is None:
                seen[key] = case.id
                has_component_to_run = True
            else:
                _COMPONENT_DUPLICATE_REASONS[(case.id, component)] = (
                    f"{component.value} component already covered by {representative}"
                )

        if has_component_to_run:
            selected.append(case)
    return selected


def get_component_duplicate_skip_reason(
    case: DiffusionTestCase, component: ComponentType
) -> str | None:
    return _COMPONENT_DUPLICATE_REASONS.get((case.id, component))


ACCURACY_ONE_GPU_CASES = _select_accuracy_cases(ONE_GPU_CASES)
ACCURACY_TWO_GPU_CASES = _select_accuracy_cases(TWO_GPU_CASES)

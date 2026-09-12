# SPDX-License-Identifier: Apache-2.0
"""Choose a replaceable residency seed before component loading."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_weight_inventory import (
    ComponentWeightEstimate,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.placement_budget import (
    PlacementOption,
    optimize_placement,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args.auto_tune import (
    auto_residency_static_skip_reason,
    fixed_loading_residency_components,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = init_logger(__name__)

GIB_BYTES = 1024**3
MIN_INITIAL_RESIDENCY_RESERVE_BYTES = 4 * GIB_BYTES
INITIAL_RESIDENCY_RESERVE_FRACTION = 0.20


def _default_denoising_steps(server_args: ServerArgs) -> int:
    from sglang.multimodal_gen.registry import get_model_info

    model_info = get_model_info(
        server_args.model_path,
        server_args.backend,
        server_args.model_id,
    )
    if model_info is None:
        return 1
    return max(1, model_info.sampling_param_cls().num_inference_steps or 1)


def choose_initial_resident_components(
    server_args: ServerArgs,
    inventory: list[ComponentWeightEstimate],
    *,
    available_bytes: int,
    denoising_steps: int | None = None,
    excluded_components: frozenset[str] = frozenset(),
) -> set[str]:
    """Choose resident startup components without removing runtime options.

    This is a load-feasibility seed, not a second serving-placement planner.
    Native DiTs can still be demoted by warmup calibration. Auxiliary modules,
    unknown weights, pipeline exclusions, and explicit choices retain their
    configured loading semantics. A model-default layerwise placement remains
    eligible because avoiding that initialization is the point of this seed.
    """
    reserve_bytes = max(
        MIN_INITIAL_RESIDENCY_RESERVE_BYTES,
        int(available_bytes * INITIAL_RESIDENCY_RESERVE_FRACTION),
    )
    fixed_resident = [
        item
        for item in inventory
        if server_args.residency_mode(item.component_name) == RESIDENT
    ]
    fixed_resident_sizes = [item.materialized_bytes() for item in fixed_resident]
    if any(weight_bytes is None for weight_bytes in fixed_resident_sizes):
        return set()
    fixed_resident_bytes = sum(
        weight_bytes for weight_bytes in fixed_resident_sizes if weight_bytes
    )
    budget_bytes = available_bytes - reserve_bytes - fixed_resident_bytes
    if budget_bytes <= 0:
        return set()

    denoising_steps = denoising_steps or _default_denoising_steps(server_args)
    options = []
    for item in inventory:
        weight_bytes = item.materialized_bytes()
        if (
            weight_bytes is None
            or weight_bytes <= 0
            or item.component_name in excluded_components
            or not is_dit_component_name(item.component_name)
            or server_args.residency_mode(item.component_name) == RESIDENT
            or server_args.explicit_residency_mode(item.component_name) is not None
        ):
            continue
        options.append(
            PlacementOption(
                group_key=item.component_name,
                option_key=f"{item.component_name}:resident",
                resource_delta_bytes={"gpu:load": weight_bytes},
                estimated_latency_savings=weight_bytes * denoising_steps,
                preference_cost=(weight_bytes,),
            )
        )
    if not options:
        return set()

    plan = optimize_placement(
        options,
        resource_budget_bytes={"gpu:load": budget_bytes},
    )
    return {selection.group_key for selection in plan.selections}


def maybe_seed_initial_residency(
    server_args: ServerArgs,
    inventory: list[ComponentWeightEstimate],
    *,
    excluded_components: frozenset[str] = frozenset(),
) -> None:
    """Apply a replaceable resident seed for warmup-calibrated auto mode."""
    if (
        auto_residency_static_skip_reason(server_args) is not None
        or server_args.use_fsdp_inference
        or not current_platform.is_cuda()
    ):
        return
    if current_platform.device_shares_host_memory():
        # The seed exists to skip layerwise initialization where it is costly.
        # On a shared host/device pool the layers simply stay on their mapping,
        # and a resident load would hold a component's bytes twice -- on the
        # device and in the page cache the other components stream from --
        # at the one moment the kernel has the least room to give. Measured on
        # a GB10: a 57 GiB resident DiT seed ended in the host OOM killer.
        # Promotion is a calibrated decision made layer by layer after warmup.
        logger.info(
            "Initial auto residency: host and device share one memory pool; "
            "every component starts on its checkpoint mapping and warmup "
            "calibration decides what becomes resident."
        )
        return

    available_gib = current_platform.get_available_gpu_memory(
        distributed=server_args.num_gpus > 1,
        empty_cache=False,
    )
    if envs.SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB is not None:
        available_gib = min(
            available_gib,
            envs.SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB,
        )
    selected = choose_initial_resident_components(
        server_args,
        inventory,
        available_bytes=int(available_gib * GIB_BYTES),
        excluded_components=(
            excluded_components
            | frozenset(
                fixed_loading_residency_components(
                    server_args,
                    (item.component_name for item in inventory),
                )
            )
        ),
    )
    for component_name in selected:
        server_args.set_auto_residency_mode(component_name, RESIDENT)
    if not selected:
        return
    logger.info(
        "Initial auto residency: resident=%s (minimum free VRAM=%.1f GiB); "
        "warmup may rebalance this load-safe seed.",
        sorted(selected),
        available_gib,
    )

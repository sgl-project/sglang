from unittest.mock import Mock, patch

import pytest

from sglang.multimodal_gen.runtime.managers.memory_managers import initial_residency
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    LAYERWISE_OFFLOAD,
    RESIDENT,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_weight_inventory import (
    ComponentWeightEstimate,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.initial_residency import (
    GIB_BYTES,
    choose_initial_resident_components,
    maybe_seed_initial_residency,
)


@pytest.fixture(autouse=True)
def _discrete_host(monkeypatch):
    # These cases describe a discrete GPU with its own VRAM. A shared
    # host/device pool skips the seed entirely; see test_shared_memory_pool.
    monkeypatch.setattr(
        type(initial_residency.current_platform),
        "device_shares_host_memory",
        classmethod(lambda cls: False),
    )


def _weight(component_name: str, gib: int) -> ComponentWeightEstimate:
    return ComponentWeightEstimate(
        component_name=component_name,
        component_model_path=f"/{component_name}",
        checkpoint_bytes=gib * GIB_BYTES,
        parameter_count=None,
        target_element_size=None,
    )


class _Args:
    model_path = "model"
    backend = "sglang"
    model_id = None
    num_gpus = 1
    use_fsdp_inference = False

    def __init__(self, modes=None, explicit=()):
        self.modes = modes or {}
        self.explicit = set(explicit)
        self.selected = {}
        self.quantization = None
        self.component_quantizations = {}
        self.nunchaku_config = None
        self.direct_gpu_weight_loading = False
        self.ltx2_two_stage_device_mode = None

    def residency_mode(self, component_name):
        return self.selected.get(
            component_name,
            self.modes.get(component_name, COMPONENT_OFFLOAD),
        )

    def configured_residency_mode(self, component_name):
        return self.modes.get(component_name, COMPONENT_OFFLOAD)

    def explicit_residency_mode(self, component_name):
        return (
            self.modes.get(component_name) if component_name in self.explicit else None
        )

    def set_auto_residency_mode(self, component_name, mode):
        self.selected[component_name] = mode


def test_initial_seed_prefers_reused_dit_under_load_budget():
    args = _Args()
    selected = choose_initial_resident_components(
        args,
        [_weight("transformer", 20), _weight("text_encoder", 20)],
        available_bytes=30 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == {"transformer"}


def test_initial_seed_accounts_for_fixed_resident_weights():
    args = _Args(modes={"vae": RESIDENT})
    selected = choose_initial_resident_components(
        args,
        [_weight("vae", 8), _weight("transformer", 20)],
        available_bytes=32 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == set()


def test_initial_seed_stays_conservative_for_unknown_fixed_resident_weights():
    args = _Args(modes={"custom_encoder": RESIDENT})
    unknown = ComponentWeightEstimate(
        component_name="custom_encoder",
        component_model_path="/custom_encoder",
        checkpoint_bytes=None,
        parameter_count=None,
        target_element_size=None,
    )

    selected = choose_initial_resident_components(
        args,
        [unknown, _weight("transformer", 8)],
        available_bytes=40 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == set()


def test_initial_seed_keeps_explicit_placement_and_auxiliary_load_semantics():
    args = _Args(
        modes={"transformer": LAYERWISE_OFFLOAD},
        explicit={"transformer"},
    )
    selected = choose_initial_resident_components(
        args,
        [_weight("transformer", 8), _weight("vae", 2)],
        available_bytes=40 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == set()


def test_initial_seed_selects_only_dits_when_all_components_fit():
    args = _Args(modes={"transformer": LAYERWISE_OFFLOAD})

    selected = choose_initial_resident_components(
        args,
        [
            _weight("transformer", 8),
            _weight("text_encoder", 4),
            _weight("vae", 2),
        ],
        available_bytes=40 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == {"transformer"}


def test_initial_seed_packs_reused_dits_under_the_common_reserve():
    args = _Args()

    selected = choose_initial_resident_components(
        args,
        [
            _weight("transformer", 27),
            _weight("transformer_2", 27),
            _weight("text_encoder", 10),
            _weight("vae", 1),
        ],
        available_bytes=75 * GIB_BYTES,
        denoising_steps=40,
    )

    assert selected == {"transformer", "transformer_2"}


def test_initial_seed_can_bypass_model_default_layerwise_initialization():
    args = _Args(modes={"transformer": LAYERWISE_OFFLOAD})

    selected = choose_initial_resident_components(
        args,
        [_weight("transformer", 8)],
        available_bytes=40 * GIB_BYTES,
        denoising_steps=8,
    )

    assert selected == {"transformer"}


def test_initial_seed_honors_pipeline_load_placement_exclusions():
    args = _Args()

    selected = choose_initial_resident_components(
        args,
        [_weight("transformer", 8), _weight("transformer_2", 8)],
        available_bytes=40 * GIB_BYTES,
        denoising_steps=8,
        excluded_components=frozenset(("transformer", "transformer_2")),
    )

    assert selected == set()


def test_initial_seed_applies_one_reversible_override():
    args = _Args()
    inventory = [_weight("transformer", 8)]
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform"
        ) as platform,
    ):
        platform.is_cuda.return_value = True
        platform.device_shares_host_memory.return_value = False
        platform.get_available_gpu_memory.return_value = 40
        maybe_seed_initial_residency(args, inventory)

    assert args.selected == {"transformer": RESIDENT}
    platform.get_available_gpu_memory.assert_called_once_with(
        distributed=False,
        empty_cache=False,
    )


def test_initial_seed_preserves_fixed_quantized_dit_loading():
    args = _Args()
    args.quantization = "fp8"
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform"
        ) as platform,
    ):
        platform.is_cuda.return_value = True
        platform.device_shares_host_memory.return_value = False
        platform.get_available_gpu_memory.return_value = 40
        maybe_seed_initial_residency(args, [_weight("transformer", 8)])

    assert args.selected == {}


def test_initial_seed_preserves_unselected_layerwise_setup():
    args = _Args(
        modes={
            "transformer": LAYERWISE_OFFLOAD,
            "transformer_2": LAYERWISE_OFFLOAD,
            "text_encoder": LAYERWISE_OFFLOAD,
            "vae": LAYERWISE_OFFLOAD,
        }
    )
    inventory = [
        _weight("transformer", 27),
        _weight("transformer_2", 27),
        _weight("text_encoder", 10),
        _weight("vae", 1),
    ]
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform"
        ) as platform,
    ):
        platform.is_cuda.return_value = True
        platform.device_shares_host_memory.return_value = False
        platform.get_available_gpu_memory.return_value = 75
        maybe_seed_initial_residency(args, inventory)

    assert args.selected == {
        "transformer": RESIDENT,
        "transformer_2": RESIDENT,
    }


def test_initial_seed_honors_the_test_allocator_cap():
    args = _Args()
    inventory = [_weight("transformer", 20)]
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform"
        ) as platform,
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.envs.SGLANG_DIFFUSION_TEST_CAP_DEVICE_MEMORY_GIB",
            24,
        ),
    ):
        platform.is_cuda.return_value = True
        platform.device_shares_host_memory.return_value = False
        platform.get_available_gpu_memory.return_value = 140
        maybe_seed_initial_residency(args, inventory)

    assert args.selected == {}


def test_initial_seed_skips_structural_fsdp_path():
    args = _Args()
    args.use_fsdp_inference = True
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform",
            Mock(is_cuda=Mock(return_value=True)),
        ),
    ):
        maybe_seed_initial_residency(args, [_weight("transformer", 8)])

    assert args.selected == {}


def test_initial_seed_uses_inventory_budget_below_legacy_threshold():
    args = _Args()
    with (
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.auto_residency_static_skip_reason",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.managers.memory_managers."
            "initial_residency.current_platform"
        ) as platform,
    ):
        platform.is_cuda.return_value = True
        platform.device_shares_host_memory.return_value = False
        platform.get_available_gpu_memory.return_value = 29.8
        maybe_seed_initial_residency(args, [_weight("transformer", 8)])

    assert args.selected == {"transformer": RESIDENT}

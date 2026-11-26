"""Native bindings report panics as Python errors and reject poisoned reuse."""

import sys
from array import array

import pytest
import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.fixture(params=["production", "inspection"])
def core_class(request):
    if request.param == "inspection":
        from sglang.test.kits.unified_radix_cache.rust_unified_tree_core_inspector import (
            RustUnifiedTreeCoreInspector,
        )

        return RustUnifiedTreeCoreInspector
    return RustUnifiedTreeCore


def _params(**overrides):
    params = dict(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of((ComponentType.FULL,)),
    )
    params.update(overrides)
    return CacheInitParams(**params)


def _raw_binding(core_class, is_eagle=False, slots=((0, "token"),), **overrides):
    native = core_class._bindings
    binding_class = (
        native.RustBigramUnifiedTreeCoreBinding
        if is_eagle
        else native.RustUnifiedTreeCoreBinding
    )
    overrides.setdefault("empty_device_indices", torch.empty(0, dtype=torch.int64))
    return binding_class(native.TreeCoreInitParamsBinding(**overrides), list(slots))


@pytest.mark.parametrize("is_eagle", [False, True])
@pytest.mark.parametrize("raw_binding", [False, True])
def test_native_panic_is_catchable_and_poisoned_core_stays_unusable(
    core_class, is_eagle, raw_binding
):
    core = (
        _raw_binding(core_class, is_eagle)
        if raw_binding
        else core_class(_params(is_eagle=is_eagle))
    )
    root = core.root_node_handle()

    # Catch Exception just as the scheduler's terminal error handler does.
    # This trips a real native invariant while the binding owns the mutex.
    with pytest.raises(Exception, match="Rust TreeCore panicked") as caught:
        core.build_backup_spec(root)
    assert type(caught.value) is RuntimeError
    assert caught.value.__cause__ is None

    for operation in (core.root_node_handle, core.reset):
        with pytest.raises(RuntimeError, match="Rust TreeCore mutex poisoned"):
            operation()
    with pytest.raises(RuntimeError, match="Rust TreeCore mutex poisoned"):
        if raw_binding:
            core.set_is_write_back(True)
        else:
            core.is_write_back = True


def test_native_configuration_and_introspection_use_the_concrete_binding(core_class):
    core = core_class(_params())
    assert type(core._binding) is core._binding_class()
    assert "root_node_handle" in dir(core._binding)
    assert core.is_write_back is False
    core.is_write_back = True
    assert core.is_write_back is True
    assert core.enable_hicache is False
    core.enable_storage = True
    assert core.enable_storage is True
    core.enable_external_cache_linker = True
    assert core.enable_external_cache_linker is True


@pytest.mark.parametrize("is_eagle", [False, True])
def test_actual_native_constructor_panic_is_runtime_error(core_class, is_eagle):
    # The raw binding validates window presence; the native SWA constructor
    # asserts positivity. Exercise that actual panic without a Python mock.
    with pytest.raises(RuntimeError, match="swa_sliding_window_size must be positive"):
        _raw_binding(
            core_class,
            is_eagle,
            slots=((0, "token"), (1, "window")),
            swa_sliding_window_size=0,
        )
    assert _raw_binding(core_class, is_eagle).root_node_handle() == 0


def test_typed_native_constructor_error_is_preserved(core_class):
    with pytest.raises(ValueError, match="page_size must be at least 1") as caught:
        core_class(_params(page_size=0))
    assert caught.value.__cause__ is None


def test_typed_native_method_error_does_not_poison_the_core(core_class):
    binding = _raw_binding(core_class)
    with pytest.raises(KeyError):
        binding.get_node_key_lengths([999])
    assert binding.root_node_handle() == 0
    binding.set_is_write_back(True)
    assert binding.is_write_back() is True


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_python_argument_errors_pass_through_unchanged(core_class, error_type):
    original = error_type("leave unchanged")

    class FailingIndex:
        def __index__(self):
            raise original

    binding = _raw_binding(core_class)
    with pytest.raises(error_type) as caught:
        binding.set_write_through_threshold(FailingIndex())
    assert caught.value is original
    assert binding.root_node_handle() == 0


def test_tlru_inspection_panic_is_runtime_error():
    from sglang.test.kits.unified_radix_cache.rust_unified_tree_core_inspector import (
        RustUnifiedTreeCoreInspector,
    )

    config = RustUnifiedTreeCoreInspector._bindings.TlruFloatConfig(
        1.0, 0.0, integer_estimate=(1 << 127) - 1
    )
    with pytest.raises(RuntimeError, match="overflowed i128"):
        config.inspect_is_tel_safe(1, 0)
    assert config.inspect_is_tel_safe(0, 0) is False


@pytest.mark.parametrize("is_eagle", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_empty_indices_keep_the_prototype_device(core_class, is_eagle, device):
    prototype = torch.empty(0, dtype=torch.int64, device=device)
    core = _raw_binding(core_class, is_eagle, empty_device_indices=prototype)
    for reset in (False, True):
        if reset:
            core.reset()
        root = core.root_node_handle()
        indices = core.collect_full_device_indices(root, root)
        assert indices.shape == (0,)
        assert indices.dtype == torch.int64
        assert indices.device == prototype.device


@pytest.mark.parametrize(
    "shape,dtype", [((1,), torch.int64), ((0, 1), torch.int64), ((0,), torch.float32)]
)
def test_invalid_empty_indices_are_rejected(core_class, shape, dtype):
    with pytest.raises(ValueError, match="empty 1-D int64"):
        _raw_binding(core_class, empty_device_indices=torch.empty(shape, dtype=dtype))


@pytest.mark.parametrize("is_eagle", [False, True])
def test_wrong_device_errors_leave_the_core_usable(core_class, is_eagle):
    core = _raw_binding(core_class, is_eagle)
    native = core_class._bindings
    wrong = torch.empty(2, dtype=torch.int64, device="meta")
    params = native.InsertParamsBinding(key=array("q", [1, 2]), value=wrong)
    with pytest.raises(ValueError, match="does not match the tree core device"):
        core.insert(params)
    with pytest.raises(ValueError, match="does not match the tree core device"):
        core.set_component_device_value(core.root_node_handle(), 0, wrong)
    match = native.MatchParamsBinding(array("q", [1, 2]))
    with pytest.raises(ValueError, match="does not match the tree core device"):
        core.attach_swa_window(match, 0, 2, wrong)
    assert core.root_node_handle() == 0
    assert core.collect_full_device_indices(0, 0).numel() == 0
    params.value = torch.tensor([10, 11], dtype=torch.int64)
    core.insert(params)
    result = core.match_prefix(match)
    indices = core.collect_full_device_indices(
        result.last_device_node_id, core.root_node_handle()
    )
    assert indices.tolist() == [10, 11][: 1 if is_eagle else 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

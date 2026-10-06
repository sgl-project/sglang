"""Native bindings report panics as Python errors and reject poisoned reuse."""

import sys

import pytest

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(params=["production", "inspection"])
def core_class(request):
    if request.param == "inspection":
        from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

        return RustUnifiedTreeCoreInspector
    return RustUnifiedTreeCore


def _params(**overrides):
    params = dict(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_components=(ComponentType.FULL,),
    )
    params.update(overrides)
    return CacheInitParams(**params)


def _raw_binding(core_class, is_eagle=False, component_types=(0,), **overrides):
    native = core_class._bindings
    binding_class = (
        native.RustBigramUnifiedTreeCoreBinding
        if is_eagle
        else native.RustUnifiedTreeCoreBinding
    )
    return binding_class(native.TreeCoreInitParamsBinding(**overrides), component_types)


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
            core_class, is_eagle, component_types=(0, 1), swa_sliding_window_size=0
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
    from rust_unified_tree_core_inspector import RustUnifiedTreeCoreInspector

    config = RustUnifiedTreeCoreInspector._bindings.TlruFloatConfig(
        1.0, 0.0, integer_estimate=(1 << 127) - 1
    )
    with pytest.raises(RuntimeError, match="overflowed i128"):
        config.inspect_is_tel_safe(1, 0)
    assert config.inspect_is_tel_safe(0, 0) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

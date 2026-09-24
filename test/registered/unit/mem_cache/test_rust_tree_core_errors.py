"""Native panics reach Python crash handlers without permitting poisoned reuse."""

from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore, _PanicGuard
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


@pytest.mark.parametrize("is_eagle", [False, True])
def test_native_panic_is_catchable_and_poisoned_core_stays_unusable(
    core_class, is_eagle
):
    core = core_class(_params(is_eagle=is_eagle))
    root = core.root_node_handle()
    native_panic = core._bindings.PanicException
    assert not issubclass(native_panic, Exception)

    # A backup of the value-less root deliberately trips a native invariant.
    # Catch Exception, just as the scheduler's terminal error handler does.
    with pytest.raises(Exception, match="Rust TreeCore panicked") as caught:
        core.build_backup_spec(root)
    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value.__cause__, native_panic)

    with pytest.raises(RuntimeError, match="Rust TreeCore mutex poisoned") as refused:
        core.root_node_handle()
    assert isinstance(refused.value.__cause__, native_panic)

    # Properties also go through native methods, including after a panic.
    with pytest.raises(RuntimeError, match="Rust TreeCore mutex poisoned"):
        core.is_write_back = True


def test_native_configuration_and_method_introspection_work_through_guard(core_class):
    core = core_class(_params())
    assert "root_node_handle" in dir(core._binding)
    assert core.is_write_back is False
    core.is_write_back = True
    assert core.is_write_back is True
    assert core.enable_hicache is False
    core.enable_storage = True
    assert core.enable_storage is True
    core.enable_external_cache_linker = True
    assert core.enable_external_cache_linker is True


def test_constructor_panic_uses_the_active_binding_module(core_class, monkeypatch):
    native_panic = core_class._bindings.PanicException

    def fail_constructor(*args, **kwargs):
        raise native_panic("constructor invariant")

    monkeypatch.setattr(core_class, "_binding_class", lambda self: fail_constructor)
    with pytest.raises(RuntimeError, match="constructor invariant") as caught:
        core_class(_params())
    assert isinstance(caught.value.__cause__, native_panic)


def test_typed_native_constructor_error_is_preserved(core_class):
    with pytest.raises(ValueError, match="page_size must be at least 1") as caught:
        core_class(_params(page_size=0))
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_guard_preserves_other_exception_types(error_type):
    class NativePanic(BaseException):
        pass

    original = error_type("leave unchanged")

    def fail():
        raise original

    guard = _PanicGuard(lambda: SimpleNamespace(fail=fail), NativePanic)
    with pytest.raises(error_type) as caught:
        guard.fail()
    assert caught.value is original


def test_guard_does_not_match_unrelated_exception_by_name():
    native_panic = type("PanicException", (BaseException,), {})
    unrelated_panic = type("PanicException", (BaseException,), {})
    original = unrelated_panic("another extension's exception")

    def fail():
        raise original

    guard = _PanicGuard(lambda: SimpleNamespace(fail=fail), native_panic)
    with pytest.raises(unrelated_panic) as caught:
        guard.fail()
    assert caught.value is original

"""Static first-bucket graphs must replay even without a SymInt argument."""

import contextlib
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture
def backend_module(monkeypatch):
    # This GPU-only dependency rejects CPU imports. These tests never capture
    # or create weak tensors: exercise the real backend with existing fake graphs.
    package = importlib.import_module("sglang.srt.compilation")
    names = ("cuda_piecewise_backend", "weak_ref_tensor")
    missing = object()
    saved_modules = {
        name: sys.modules.get(f"{package.__name__}.{name}", missing) for name in names
    }
    saved_attrs = {name: vars(package).get(name, missing) for name in names}
    weak_refs = ModuleType("sglang.srt.compilation.weak_ref_tensor")
    weak_refs.weak_ref_tensors = Mock(side_effect=AssertionError("unexpected capture"))
    try:
        sys.modules[weak_refs.__name__] = weak_refs
        # Import a separate backend so an already cached module is never patched.
        sys.modules.pop(f"{package.__name__}.cuda_piecewise_backend", None)
        module = importlib.import_module(f"{package.__name__}.cuda_piecewise_backend")
        monkeypatch.setattr(module, "is_in_torch_compile_warmup", lambda: False)
        monkeypatch.setattr(module, "graph_pool_replay_scope", contextlib.nullcontext)
        yield module
    finally:
        # import_module also writes the child module onto its parent package.
        # Restore both caches, including absence, to avoid leaking the GPU stub.
        for name in names:
            fullname = f"{package.__name__}.{name}"
            if saved_modules[name] is missing:
                sys.modules.pop(fullname, None)
            else:
                sys.modules[fullname] = saved_modules[name]
            if saved_attrs[name] is missing:
                vars(package).pop(name, None)
            else:
                setattr(package, name, saved_attrs[name])


def _backend(module, *, sym_shape_indices=()):
    backend = module.CUDAPiecewiseBackend.__new__(module.CUDAPiecewiseBackend)
    backend.first_run_finished = True
    backend.sym_shape_indices = list(sym_shape_indices)
    backend._static_context_bucket = None
    backend.compiled_graph_for_general_shape = Mock(return_value=object())
    backend.compile_config = SimpleNamespace(get_enable_debug_mode=lambda: False)
    backend.concrete_size_entries = {
        size: module.ConcreteSizeEntry(
            runtime_shape=size,
            need_to_compile=False,
            use_cudagraph=True,
            compiled=True,
            cudagraph=SimpleNamespace(replay=Mock()),
            output=object(),
        )
        for size in (4096, 8192)
    }
    return backend


def _context(size, *, explicit=True):
    return SimpleNamespace(
        num_tokens=size if explicit else None,
        raw_num_tokens=17,
        forward_batch=SimpleNamespace(
            input_ids=torch.empty(size, dtype=torch.int64, device="cpu")
        ),
    )


@pytest.mark.parametrize(
    "explicit", [True, False], ids=["serve", "capture-input-shape"]
)
def test_static_piece_replays_only_its_first_bucket(
    backend_module, monkeypatch, explicit
):
    backend = _backend(backend_module)
    context = _context(8192, explicit=explicit)
    monkeypatch.setattr(
        backend_module, "get_tc_piecewise_forward_context", lambda: context
    )
    entry = backend.concrete_size_entries[8192]

    # An unrelated persistent dimension must not become the token-bucket key.
    assert backend(89929) is entry.output
    backend.compiled_graph_for_general_shape.assert_not_called()
    context = _context(4096, explicit=explicit)
    assert backend(89929) is backend.compiled_graph_for_general_shape.return_value
    context = None
    assert backend(89929) is backend.compiled_graph_for_general_shape.return_value
    context = _context(8192, explicit=explicit)
    assert backend(89929) is entry.output
    assert entry.cudagraph.replay.call_count == 2
    backend.concrete_size_entries[4096].cudagraph.replay.assert_not_called()
    assert backend.compiled_graph_for_general_shape.call_count == 2


def test_symbolic_piece_keeps_its_argument_key(backend_module, monkeypatch):
    backend = _backend(backend_module, sym_shape_indices=(1,))
    monkeypatch.setattr(
        backend_module, "get_tc_piecewise_forward_context", lambda: _context(8192)
    )
    entry = backend.concrete_size_entries[4096]

    assert backend(89929, 4096) is entry.output
    entry.cudagraph.replay.assert_called_once_with()
    backend.compiled_graph_for_general_shape.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

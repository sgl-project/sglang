import builtins
import sys

import pytest
from sgl_kernel import load_utils

UNDEFINED_SYMBOL_ERR = "undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev"
GENERIC_ERR = "libfoo.so: cannot open shared object file: No such file or directory"


def _force_load_failure(monkeypatch, load_error):
    # A common_ops .so is "found" but fails to load with load_error, and the final
    # `import common_ops` fallback fails -- so _load_architecture_specific_ops reaches
    # its error branch with load_error recorded.
    monkeypatch.setattr(
        load_utils.glob,
        "glob",
        lambda pattern: ["/nonexistent/sm100/common_ops.abi3.so"],
    )

    def _raise(*args, **kwargs):
        raise ImportError(load_error)

    monkeypatch.setattr(load_utils.importlib.util, "spec_from_file_location", _raise)

    real_import = builtins.__import__

    def _no_common_ops(name, *args, **kwargs):
        if name == "common_ops":
            raise ModuleNotFoundError("No module named 'common_ops'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_common_ops)


def _raise_undef():
    raise ImportError(UNDEFINED_SYMBOL_ERR)


def test_undefined_symbol_raises_abi_diagnostic(monkeypatch):
    _force_load_failure(monkeypatch, UNDEFINED_SYMBOL_ERR)

    with pytest.raises(ImportError) as excinfo:
        load_utils._load_architecture_specific_ops()

    msg = str(excinfo.value)
    assert "PyTorch C++ ABI mismatch" in msg
    assert "undefined symbol" in msg


def test_non_abi_failure_keeps_generic_message(monkeypatch):
    # A load failure that is NOT an undefined symbol must not be mislabeled an ABI
    # mismatch -- it routes to the generic message.
    _force_load_failure(monkeypatch, GENERIC_ERR)

    with pytest.raises(ImportError) as excinfo:
        load_utils._load_architecture_specific_ops()

    msg = str(excinfo.value)
    assert "PyTorch C++ ABI mismatch" not in msg
    assert "Could not load any common_ops" in msg


def test_load_common_ops_reraises_without_flag(monkeypatch):
    monkeypatch.setattr(load_utils, "_load_architecture_specific_ops", _raise_undef)
    monkeypatch.delenv("SGLANG_KERNEL_ALLOW_MISSING_OPS", raising=False)

    with pytest.raises(ImportError):
        load_utils.load_common_ops()


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_load_common_ops_reraises_for_falsey_flag(monkeypatch, value):
    monkeypatch.setattr(load_utils, "_load_architecture_specific_ops", _raise_undef)
    monkeypatch.setenv("SGLANG_KERNEL_ALLOW_MISSING_OPS", value)

    with pytest.raises(ImportError):
        load_utils.load_common_ops()


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_load_common_ops_allows_missing_with_flag(monkeypatch, value):
    monkeypatch.setattr(load_utils, "_load_architecture_specific_ops", _raise_undef)
    monkeypatch.setenv("SGLANG_KERNEL_ALLOW_MISSING_OPS", value)

    with pytest.warns(RuntimeWarning):
        assert load_utils.load_common_ops() is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))

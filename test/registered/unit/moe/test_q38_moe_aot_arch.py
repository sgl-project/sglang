"""CPU tests of device/bundle matching without importing SGLang or CUDA."""
import importlib.util
import json
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[4] / 'python/sglang/kernels/ops/moe/q38_moe_aot.py'
spec = importlib.util.spec_from_file_location('q38_aot_arch', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def manifest(path, arch=None):
    path.mkdir(parents=True, exist_ok=True)
    value = {} if arch is None else {'arch': arch}
    (path / 'manifest.json').write_text(json.dumps(value))


def test_qualified_legacy_bundle_remains_sm103_only(tmp_path, monkeypatch):
    manifest(tmp_path)
    monkeypatch.setenv('TEST_MOE_AOT', str(tmp_path))
    assert module.bundle_manifest('TEST_MOE_AOT', (10, 3))[0] == tmp_path
    with pytest.raises(ValueError, match='does not match'):
        module.bundle_manifest('TEST_MOE_AOT', (10, 7))


@pytest.mark.parametrize('capability,arch', [((10, 3), 'sm_103a'), ((10, 7), 'sm_107a')])
def test_arch_tree_selects_exact_device(tmp_path, monkeypatch, capability, arch):
    for name in ('sm_103a', 'sm_107a'):
        manifest(tmp_path / name, name)
    monkeypatch.setenv('TEST_MOE_AOT', str(tmp_path))
    root, value = module.bundle_manifest('TEST_MOE_AOT', capability)
    assert root == tmp_path / arch and value['arch'] == arch


def test_architectures_manifest_selects_relative_bundle(tmp_path, monkeypatch):
    manifest(tmp_path / 'rubin', 'sm_107a')
    (tmp_path / 'manifest.json').write_text(
        json.dumps({'architectures': {'sm_107a': {'path': 'rubin'}}})
    )
    monkeypatch.setenv('TEST_MOE_AOT', str(tmp_path))
    root, value = module.bundle_manifest('TEST_MOE_AOT', (10, 7))
    assert root == tmp_path / 'rubin' and value['arch'] == 'sm_107a'


def test_explicit_flat_sm107_and_wrong_device(tmp_path, monkeypatch):
    manifest(tmp_path, 'sm_107a')
    monkeypatch.setenv('TEST_MOE_AOT', str(tmp_path))
    assert module.bundle_manifest('TEST_MOE_AOT', (10, 7))[0] == tmp_path
    with pytest.raises(ValueError, match='does not match'):
        module.bundle_manifest('TEST_MOE_AOT', (10, 3))


def test_missing_variable_does_not_silently_fallback(monkeypatch):
    monkeypatch.delenv('TEST_MOE_AOT', raising=False)
    with pytest.raises(KeyError):
        module.bundle_manifest('TEST_MOE_AOT', (10, 7))


def test_unsupported_device_rejected_before_bundle_access():
    with pytest.raises(ValueError, match='SM103 or SM107'):
        module.bundle_manifest('NO_SUCH_VARIABLE', (9, 0))

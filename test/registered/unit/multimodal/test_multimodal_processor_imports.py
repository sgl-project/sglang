from types import SimpleNamespace

import pytest

from sglang.srt.managers import multimodal_processor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_import_processors_records_failed_imports(monkeypatch):
    package_name = "test_mm_processors"
    failed_module_name = f"{package_name}.broken"
    package = SimpleNamespace(__path__=[])

    def import_module(name):
        if name == package_name:
            return package
        raise ImportError("optional dependency is missing")

    monkeypatch.setattr(multimodal_processor.importlib, "import_module", import_module)
    monkeypatch.setattr(
        multimodal_processor.pkgutil,
        "iter_modules",
        lambda path, prefix: [(None, failed_module_name, False)],
    )

    multimodal_processor.import_processors(package_name)

    assert (
        str(multimodal_processor.PROCESSOR_IMPORT_ERRORS[failed_module_name])
        == "optional dependency is missing"
    )


def test_get_mm_processor_reports_failed_imports(monkeypatch):
    module_name = "sglang.srt.multimodal.processors.broken"
    error = ImportError("optional dependency is missing")
    monkeypatch.setitem(
        multimodal_processor.PROCESSOR_IMPORT_ERRORS, module_name, error
    )
    monkeypatch.setattr(multimodal_processor, "PROCESSOR_MAPPING", {})
    monkeypatch.setattr(
        multimodal_processor, "get_mm_processor_cls", lambda *args: None
    )

    with pytest.raises(ValueError, match="Failed processor imports") as exc_info:
        multimodal_processor.get_mm_processor(
            SimpleNamespace(architectures=["MissingArchitecture"]),
            server_args=None,
            processor=None,
            transport_mode=None,
        )

    message = str(exc_info.value)
    assert module_name in message
    assert "ImportError: optional dependency is missing" in message

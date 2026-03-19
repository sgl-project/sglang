from types import ModuleType

import pytest

from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
    patch_function,
)
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchSpec
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)

SAMPLE_MODULE_NAME = "_source_patcher_test_fixtures.sample_module"


class TestPatchFunction:
    def test_basic_patch_changes_behavior(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert obj.greet("world") == "patched world"
        finally:
            state.restore()

        assert obj.greet("world") == "hello world"

    def test_globals_preserved_after_patch(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.uses_global() == "value=global_value"

        state = patch_function(
            target=cls.uses_global,
            edits=[
                EditSpec(
                    match='return f"value={GLOBAL_VAR}"',
                    replacement='return f"patched_value={GLOBAL_VAR}"',
                )
            ],
        )
        try:
            assert obj.uses_global() == "patched_value=global_value"
        finally:
            state.restore()

    def test_function_identity_preserved(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        fn_id_before = id(cls.greet)

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"patched {name}"',
                )
            ],
        )
        try:
            assert id(cls.greet) == fn_id_before
        finally:
            state.restore()

    def test_patch_standalone_function(self, sample_module: ModuleType) -> None:
        fn = sample_module.standalone_function
        assert fn(2, 3) == 5

        state = patch_function(
            target=fn,
            edits=[
                EditSpec(
                    match="return a + b",
                    replacement="return a * b",
                )
            ],
        )
        try:
            assert fn(2, 3) == 6
        finally:
            state.restore()

        assert fn(2, 3) == 5

    def test_patched_code_can_reference_global_variable(
        self, sample_module: ModuleType
    ) -> None:
        """Replacement code that references a module-level global should work."""
        cls = sample_module.SampleClass
        obj = cls()

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement='greeting = f"{GLOBAL_VAR} {name}"',
                )
            ],
        )
        try:
            assert obj.greet("world") == "global_value world"
        finally:
            state.restore()

    def test_patched_code_can_call_another_class_method(
        self, sample_module: ModuleType
    ) -> None:
        """Replacement code that calls HelperClass.format_value should work."""
        cls = sample_module.SampleClass
        obj = cls()

        state = patch_function(
            target=cls.greet,
            edits=[
                EditSpec(
                    match='greeting = f"hello {name}"',
                    replacement="greeting = HelperClass.format_value(name)",
                )
            ],
        )
        try:
            assert obj.greet("world") == "[world]"
        finally:
            state.restore()

    def test_patched_code_uses_helper_via_existing_method(
        self, sample_module: ModuleType
    ) -> None:
        """The uses_helper method already calls HelperClass; verify it survives patching."""
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.uses_helper("test") == "[test]"

        state = patch_function(
            target=cls.uses_helper,
            edits=[
                EditSpec(
                    match="return HelperClass.format_value(value)",
                    replacement='return HelperClass.format_value("patched_" + value)',
                )
            ],
        )
        try:
            assert obj.uses_helper("test") == "[patched_test]"
        finally:
            state.restore()

        assert obj.uses_helper("test") == "[test]"


class TestResolveTarget:
    def test_resolve_class_method(self, sample_module: ModuleType) -> None:
        target = _resolve_target(f"{SAMPLE_MODULE_NAME}.SampleClass.greet")
        assert target is sample_module.SampleClass.greet

    def test_resolve_standalone_function(self, sample_module: ModuleType) -> None:
        target = _resolve_target(f"{SAMPLE_MODULE_NAME}.standalone_function")
        assert target is sample_module.standalone_function

    def test_resolve_nonexistent_raises(self, sample_module: ModuleType) -> None:
        with pytest.raises((ImportError, AttributeError)):
            _resolve_target(f"{SAMPLE_MODULE_NAME}.NonexistentClass.method")


class TestCodePatcher:
    def test_context_manager_patches_and_restores(
        self, sample_module: ModuleType
    ) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"ctx_patched {name}"',
                    )
                ],
            )
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "ctx_patched world"

        assert obj.greet("world") == "hello world"

    def test_context_manager_multiple_patches(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"p1 {name}"',
                    )
                ],
            ),
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.compute",
                edits=[
                    EditSpec(
                        match="result = x * 2 + 1",
                        replacement="result = x * 100",
                    )
                ],
            ),
        ]

        with CodePatcher(patches=patches):
            assert obj.greet("world") == "p1 world"
            assert obj.compute(5) == 500

        assert obj.greet("world") == "hello world"
        assert obj.compute(5) == 11

    def test_restores_on_exception(self, sample_module: ModuleType) -> None:
        cls = sample_module.SampleClass
        obj = cls()

        patches = [
            PatchSpec(
                target=f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                edits=[
                    EditSpec(
                        match='greeting = f"hello {name}"',
                        replacement='greeting = f"err_patched {name}"',
                    )
                ],
            )
        ]

        with pytest.raises(RuntimeError):
            with CodePatcher(patches=patches):
                assert obj.greet("world") == "err_patched world"
                raise RuntimeError("test error")

        assert obj.greet("world") == "hello world"

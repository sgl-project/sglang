# SPDX-License-Identifier: Apache-2.0

import builtins
import importlib.util
import sys

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_non_shell_tools_import_without_shell_sdk_types(monkeypatch):
    from sglang.srt.entrypoints.openai.responses_tools import beam

    real_import = builtins.__import__

    def import_without_shell(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("openai.types.responses") and set(fromlist or ()) & {
            "ResponseFunctionShellToolCall",
            "ShellCall",
            "ShellCallOutput",
        }:
            raise ImportError("shell SDK types unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_shell)
    spec = importlib.util.spec_from_file_location(
        "beam_without_shell_sdk", beam.__file__
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    item = {"role": "user", "content": "hello"}
    assert module.BeamToolAdapter([]).input_item(item) == item


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

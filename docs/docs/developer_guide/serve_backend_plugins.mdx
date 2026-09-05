---
title: "Add an out-of-tree serve backend"
description: "Connect an ecosystem runtime to sglang serve through the versioned serve backend plugin API."
---

A serve backend plugin connects an out-of-tree runtime to the SGLang-owned CLI:

```bash
sglang serve MODEL_PATH --model-type BACKEND_NAME [BACKEND_OPTIONS]
```

The extension retains its own argument parser, process topology, API endpoints, hardware requirements, and release cycle. The core CLI owns backend selection and common child-process cleanup.

## Prerequisites

- Install your extension and a compatible SGLang version in the same Python 3.10+ environment.
- Declare the SGLang version range tested by your extension in its package dependencies.
- Keep the backend factory and detector independent of GPU initialization and model loading.

The plugin API itself is platform-independent. Your backend defines its supported operating systems, accelerators, parallelism options, and authentication requirements.

## Keep one owner for the executable

Only the `sglang` distribution should publish a console script named `sglang`. Your extension registers package metadata under `sglang.serve_backends`; it must not publish another `sglang` script.

This prevents installation order from replacing the command and ensures uninstalling an extension does not remove the core executable. You can retain a project-specific executable as a compatibility alias:

```bash
my-runtime serve MODEL_PATH
sglang serve MODEL_PATH --model-type my_runtime
```

Both commands should call the same backend implementation.

## Register a backend factory

Add a zero-argument factory to your extension's `pyproject.toml`:

```toml
[project]
name = "my-sglang-runtime"
version = "0.1.0"
dependencies = ["sglang"]

[project.entry-points."sglang.serve_backends"]
my_runtime = "my_sglang_runtime.sglang_backend:create_backend"
```

The entry point name, `my_runtime`, becomes an accepted `--model-type` value. Choose a distinctive name. `auto` and SGLang's in-tree backend names are reserved.

## Implement the backend

Create `my_sglang_runtime/sglang_backend.py`:

```python
import argparse

from sglang.cli.serve_backends import (
    ServeBackend,
    ServeBackendDetection,
    ServeRequest,
)


def detect(request: ServeRequest) -> ServeBackendDetection:
    if request.model_path is None:
        return ServeBackendDetection.UNKNOWN
    if supports_model(request.model_path):
        return ServeBackendDetection.MATCH
    return ServeBackendDetection.NO_MATCH


def run(request: ServeRequest) -> None:
    parser = argparse.ArgumentParser(prog="sglang serve")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    args, remaining = parser.parse_known_args(request.argv)
    launch_runtime(args, remaining)


def create_backend() -> ServeBackend:
    return ServeBackend(api_version=1, run=run, detect=detect)
```

Replace `supports_model()` and `launch_runtime()` with your extension's lightweight metadata check and blocking server launcher. A real `run()` call should block for the server lifetime. It must also honor `-h` and `--help` without launching a server; `argparse` does this automatically.

Among serve backend entry points, explicit selection imports only the selected provider. Automatic selection loads installed backend factories and invokes their detectors, so importing this module and calling `detect()` must not initialize accelerators, import model weights, or start workers.

## Handle forwarded arguments

SGLang removes `--model-type` and normalizes a positional Hugging Face model ID or local model directory before dispatch. For example:

```bash
sglang serve org/model --model-type my_runtime --pipeline-parallel 2
```

Your backend receives:

```python
("--model-path", "org/model", "--pipeline-parallel", "2")
```

The selected backend owns all remaining argument parsing and validation. Target backend-specific help with:

```bash
sglang serve --model-type my_runtime --help
```

## Support config-only runtimes

SGLang requires a model path by default. If your runtime resolves its model and parallelism settings from a configuration file, disable that validation:

```python
def create_backend() -> ServeBackend:
    return ServeBackend(
        api_version=1,
        run=run,
        detect=detect,
        requires_model_path=False,
    )
```

You can then accept commands such as:

```bash
sglang serve --model-type my_runtime --config pipeline.yaml
```

Config-only requests generally require explicit `--model-type` unless your detector can identify the backend from the remaining arguments.

## Understand automatic routing

The default `--model-type auto` follows these rules:

1. Backends without a detector remain explicit-only.
2. One `MATCH` selects that backend.
3. Multiple matches fail and require an explicit `--model-type`.
4. `UNKNOWN` and detector failures do not claim the request.
5. No matches preserve the existing LLM fallback.

The registry does not resolve overlap by package installation order or a hidden priority. A backend can opt out of automatic routing by omitting its detector:

```python
ServeBackend(api_version=1, run=run, requires_model_path=False)
```

## Maintain compatibility

Declare the API version implemented by your extension as a literal. Do not copy SGLang's current version constant at runtime; a fixed value lets a future SGLang release detect an older plugin contract. SGLang rejects incompatible, duplicate, and reserved backend registrations with an actionable error.

The public extension contract consists of:

- `ServeRequest`: normalized backend arguments and the optional model path
- `ServeBackend`: the runner, optional detector, and model-path requirement
- `ServeBackendDetection`: `MATCH`, `NO_MATCH`, or `UNKNOWN`

Test explicit selection, backend-specific help, automatic detection, ambiguous models, and installation or removal alongside the core `sglang` package.

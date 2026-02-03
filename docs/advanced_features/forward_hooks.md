## Model Hooks

SGLang supports attaching PyTorch forward hooks to specific submodules in the loaded model, configured entirely via `server_args` JSON.

This is useful for:

* Logging intermediate activations
* Debugging model internals
* Exporting hidden states to external tooling

Hooks are attached once during `ModelRunner.initialize` and run on every forward pass.

---

### Configuration overview

Hooks are configured via a `ServerArgs` field:

```python
class ServerArgs:
    ...
    # For forward hooks
    forward_hooks: Optional[List[dict[str, Any]]] = None
````

In JSON form, a minimal configuration looks like:

```jsonc
{
  "forward_hooks": [
    {
      "name": "outer_linear_hooks",
      "target_modules": ["outer.0", "outer.1"],
      "hook_factory": "my_project.hooks:dummy_hook_factory",
      "config": {
        "tag": "outer-layer"
      }
    }
  ]
}
```

#### Top-level fields

* `forward_hooks` (optional list of objects)
  Each element is a hook spec describing:

  * Which modules to target
  * Which Python factory to call
  * What configuration to pass into that factory

---

### Hook spec schema

Each entry in `forward_hooks` is a JSON object with the following shape:

```jsonc
{
  "name": "optional-descriptive-name",
  "target_modules": ["pattern1", "pattern2", "..."],
  "hook_factory": "module.submodule:factory_name",
  "config": {
    "...": "arbitrary JSON"
  }
}
```

#### `name` (optional)

* Human-readable name for logging.
* Used only in log messages such as:

  ```text
  Registered forward hook 'outer_linear_hooks' on outer.0
  ```

#### `target_modules` (required)

* List of **module name patterns** used to match entries in `model.named_modules()`.
* Patterns are matched using `fnmatch.fnmatch`, so:

  * `"outer.0"` matches exactly `"outer.0"`.
  * `"outer.*"` matches `"outer.0"`, `"outer.1"`, `"outer.inner"`, etc.
  * `"outer.inner.*"` matches children under `outer.inner`.

> If no modules match the given patterns, hook registration does **not** fail.
> Instead, SGLang logs a warning and continues:
>
> ```text
> No modules matched hook spec 'name' patterns=['...']
> ```

#### `hook_factory` (required)

* String path to the Python factory function that creates the hook.
* Supported formats:

  * `"package.module:factory_name"`
  * `"package.module.submodule.factory_name"`

The path is resolved via:

```python
def resolve_callable(path: Optional[str]) -> Optional[Callable]:
    if path is None:
        return None

    if ":" in path:
        module_name, fn_name = path.split(":", 1)
    else:
        parts = path.split(".")
        if len(parts) < 2:
            raise ValueError(
                f"Invalid hook callable path '{path}'. "
                "Expected 'module.submodule:factory' or 'module.submodule.factory'."
            )
        *mod_parts, fn_name = parts
        module_name = ".".join(mod_parts)

    module = importlib.import_module(module_name)
    try:
        return getattr(module, fn_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{fn_name}' "
            f"(from hook path '{path}')"
        ) from e
```

**Failure modes**:

* If the path is malformed (not enough dots and no `:`), a `ValueError` is raised at startup.
* If the module imports but the attribute is missing, an `AttributeError` is raised with a clear error message.
* If the hook factory returns `None`, a warning is logged and no hook is registered for that spec (initialization continues).

The first two cause initialization to fail fast with a descriptive error; the last one is non-fatal.

#### `config` (optional)

* Arbitrary JSON object.
* Passed directly to the hook factory as a Python `dict`.
* This lets you parameterize hook behavior from config (e.g. tags, log levels, sampling rates, etc.).

---

### Hook lifecycle and behavior

Hooks are registered in `ModelRunner.initialize()`:

```python
if server_args.forward_hooks:
    register_forward_hooks(self.model, server_args.forward_hooks)
```

The actual registration logic is implemented by `register_forward_hooks`:

```python
def register_forward_hooks(model: nn.Module, hook_specs: List[dict[str, Any]]) -> None:
    """
    hook_specs is a list of dicts from server_args.forward_hooks.
    Attaches forward hooks to the matching modules.
    """
    name_to_module = dict(model.named_modules())

    for spec in hook_specs:
        spec_name = spec.get("name", "")
        target_patterns = spec.get("target_modules", [])
        if not target_patterns:
            logger.warning(
                f"Hook spec '{spec_name}' has no 'target_modules', skipping"
            )
            continue

        hook_factory_path = spec.get("hook_factory")
        if not hook_factory_path:
            logger.warning(
                f"Hook spec '{spec_name}' has no 'hook_factory', skipping"
            )
            continue

        config = spec.get("config") or {}
        hook_factory = resolve_callable(hook_factory_path)

        hook = hook_factory(config) if hook_factory else None
        if hook is None:
            logger.warning(
                f"Hook factory '{hook_factory_path}' for spec '{spec_name}' "
                "returned None, not registering any hook"
            )
            continue

        # Resolve patterns like "model.layers.*.mlp"
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec_name}' "
                f"patterns={target_patterns}"
            )
            continue

        for module_name, module in matched:
            if hook:
                _ = module.register_forward_hook(hook)
                logger.info(
                    f"Registered forward hook '{spec_name}' "
                    f"on {module_name}"
                )
```

Key points:

* Hooks are **forward hooks only** (via `module.register_forward_hook`).
* They are attached once at initialization.
* Hook handles are currently not stored on `ModelRunner` (they cannot be removed later via this API).
* Failure to match any modules is non-fatal; a warning is logged instead.
* If a hook factory returns `None`, a warning is logged and that spec is skipped.

---

### Writing a hook factory

A hook factory is a regular Python function:

* Takes a `config: dict` (from JSON)
* Returns a forward hook function with signature `(module, inputs, output)`

Example:

```python
HOOK_CALLS = []

def dummy_hook_factory(config):
    """Factory that returns a forward hook capturing a tag from config."""
    tag = config.get("tag", "default")

    def hook(module, inputs, output):
        HOOK_CALLS.append(
            {
                "module_type": type(module).__name__,
                "tag": tag,
                "shape": tuple(output.shape),
            }
        )
        return output  # must return output if you don’t want to modify the tensor

    return hook
```

In JSON:

```jsonc
{
  "forward_hooks": [
    {
      "name": "capture_outer",
      "target_modules": ["outer.0", "outer.1"],
      "hook_factory": "my_project.hooks:dummy_hook_factory",
      "config": {
        "tag": "outer"
      }
    }
  ]
}
```

This will:

* Resolve `my_project.hooks:dummy_hook_factory` to a Python callable.
* Call it with `config = {"tag": "outer"}`.
* Use the returned hook for all modules matching `outer.0` and `outer.1`.
* Append metadata about each call to `HOOK_CALLS`.

---

### Summary

* Define `forward_hooks` as a list of specs in `ServerArgs` to turn on the feature.

* Each spec:

  * selects modules via `target_modules` (glob patterns over `model.named_modules()`),
  * points to a hook factory via `hook_factory`,
  * passes arbitrary `config` into that factory.

* Hook factories are resolved via `resolve_callable`, which supports `module:factory` and `module.submodule.factory`.

* Hooks are standard PyTorch forward hooks, attached once at startup and invoked on every forward pass.

* Misconfiguration is either:

  * **fatal and explicit** (bad path / missing attribute), or
  * **non-fatal with clear warnings** (no targets matched, or factory returned `None`).

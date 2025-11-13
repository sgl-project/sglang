## Model Hooks

SGLang supports attaching **PyTorch forward hooks** to specific submodules in the loaded model, configured **entirely via `server_args` JSON**.

This is useful for:

* Logging intermediate activations
* Debugging model internals
* Exporting hidden states to external tooling

Hooks are attached once during `ModelRunner.initialize` and run on every forward pass.

---

### Configuration overview

Hooks are configured via two new `ServerArgs` fields:

```python
class ServerArgs:
    ...
    # For forward hooks
    enable_hooks: bool = False
    hooks: Optional[List[dict[str, Any]]] = None
```

In JSON form, a minimal configuration looks like:

```jsonc
{
  "enable_hooks": true,
  "hooks": [
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

* `enable_hooks` (bool, default: `false`)
  When `true`, `ModelRunner.initialize()` will look for `server_args.hooks` and register hooks accordingly.

* `hooks` (optional list of objects)
  Each element is a **hook spec** describing:

  * Which modules to target
  * Which Python factory to call
  * What configuration to pass into that factory

> If `enable_hooks` is `true` but `hooks` is empty or `null`, SGLang logs:
>
> ```text
> enable_hooks=True but no 'hooks' specified in server_args
> ```

---

### Hook spec schema

Each entry in `hooks` is a JSON object with the following shape:

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

Both of these cause initialization to fail fast with a descriptive error.

#### `config` (optional)

* Arbitrary JSON object.
* Passed directly to the hook factory as a Python `dict`.
* This lets you parameterize hook behavior from config (e.g. tags, log levels, sampling rates, etc.).

---

### Hook lifecycle and behavior

Hooks are registered in `ModelRunner.initialize()`:

```python
if server_args.enable_hooks:
    hook_specs = getattr(server_args, "hooks", None)
    if hook_specs:
        self.register_hooks(hook_specs)
    else:
        logger.warning(
            "enable_hooks=True but no 'hooks' specified in server_args"
        )
```

The actual registration logic is implemented in `ModelRunner.register_hooks`:

```python
def register_hooks(self, hook_specs: List[dict[str, Any]]) -> None:
    """
    hook_specs is a list of dicts from server_args.hooks.
    Attaches forward hooks to the matching modules.
    """
    name_to_module = dict(self.model.named_modules())

    for spec in hook_specs:
        target_patterns = spec["target_modules"]
        hook_factory_path = spec.get("hook_factory")
        config = spec.get("config") or {}

        hook_factory = resolve_callable(hook_factory_path)

        hook = hook_factory(config) if hook_factory else None

        # Resolve patterns like "outer.*" or "outer.inner.*"
        matched = []
        for name, module in name_to_module.items():
            if any(fnmatch.fnmatch(name, pattern) for pattern in target_patterns):
                matched.append((name, module))

        if not matched:
            logger.warning(
                f"No modules matched hook spec '{spec.get('name', '')}' "
                f"patterns={target_patterns}"
            )
            continue

        for module_name, module in matched:
            if hook:
                _ = module.register_forward_hook(hook)
                logger.info(
                    f"Registered forward hook '{spec.get('name', '')}' "
                    f"on {module_name}"
                )
```

Key points:

* Hooks are **forward hooks only** (via `module.register_forward_hook`).
* They are attached once at initialization.
* Hook handles are currently not stored on `ModelRunner` (i.e. no runtime removal API yet).
* Failure to match any modules is **non-fatal**; a warning is logged instead.

---

### Writing a hook factory

A hook factory is a regular Python function:

* Takes a `config: dict` (from JSON)
* Returns a **forward hook** function with signature `(module, inputs, output)`

Example (from the tests):

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
  "enable_hooks": true,
  "hooks": [
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

### Testing and examples

The behavior of `register_hooks` and `resolve_callable` is covered by tests in:

* `test/srt/test_model_hooks.py`

The tests demonstrate:

1. **Hooks are called** when a valid `hook_factory` path and `target_modules` are provided.

   ```python
   hook_specs = [
       {
           "target_modules": ["outer.0", "outer.1"],
           "hook_factory": "sglang.test.srt.test_model_hooks:dummy_hook_factory",
           "config": {"tag": "forward-ok"},
       }
   ]

   runner.register_hooks(hook_specs)
   x = torch.randn(3, 4)
   _ = runner.model(x)

   assert len(HOOK_CALLS) > 0
   ```

2. **No matching modules is non-fatal** and doesn’t crash the forward pass:

   ```python
   hook_specs = [
       {
           "name": "no_match",
           "target_modules": ["does_not_exist.*"],
           "hook_factory": "sglang.test.test_model_hooks:dummy_hook_factory",
           "config": {"tag": "unused"},
       }
   ]

   runner.register_hooks(hook_specs)
   x = torch.randn(3, 4)
   _ = runner.model(x)

   assert len(HOOK_CALLS) == 0
   ```

---

### Summary

* Use `enable_hooks: true` to turn on the feature.
* Define `hooks` as a list of specs.
* Each spec:

  * selects modules via `target_modules` (glob patterns over `model.named_modules()`),
  * points to a hook factory via `hook_factory`,
  * passes arbitrary `config` into that factory.
* Hooks are standard PyTorch forward hooks, attached once at startup and invoked on every forward pass.
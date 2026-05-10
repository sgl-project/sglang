# SGLang Plugin System

## Overview

Allows hardware vendors and developers to extend SGLang **without modifying the main repository code**.

The framework provides two plugin types, both discovered via Python's standard `setuptools` entry_points:

| Plugin Type | Entry Point Group | Purpose |
|---|---|---|
| **Hardware Platform Plugin** | `sglang.srt.platforms` | Register a custom hardware platform (device operations, KV cache pools, attention backends, graph capture, compilation backends, etc.) |
| **General Plugin** | `sglang.srt.plugins` | Inject hooks (before/after/around/replace) into any function/method, or replace entire classes |

### Principles

- **Non-intrusive**: Existing CUDA/ROCm/NPU/XPU code remains unchanged. OOT code paths are added alongside existing hardware-specific logic.
- **Zero configuration**: Plugins are automatically discovered after `pip install`, no sglang code changes required.
- **Environment variable control**: `SGLANG_PLATFORM` selects or validates the active platform plugin; `SGLANG_PLUGINS` (comma-separated) controls which general plugins to load.

### Current Scope & Future Direction

The plugin system currently targets **out-of-tree (OOT) hardware platforms** — enabling new devices to integrate with SGLang without any changes to the main repository. The main-repo hardware paths (CUDA, ROCm, NPU, XPU, etc.) continue to use the existing `is_cuda()`/`is_npu()`/… utility functions.

As the plugin interfaces mature and stabilize, in-tree hardware backends can be gradually migrated to the same plugin architecture. This would replace the scattered `if device == "cuda" … elif device == "npu" …` branches throughout the codebase with a single polymorphic dispatch through the platform interface, making each hardware backend self-contained and the core engine hardware-agnostic.

## Architecture

### Platform Hierarchy

The platform hierarchy uses a DeviceMixin pattern to share device operations between SRT (LLM inference) and Multimodal subsystems:

```
DeviceMixin (shared device identity + operations)
├── SRTPlatform(DeviceMixin)           # + graph runner, KV pool, …
│   └── MySRTPlatform(SRTPlatform, MyDeviceMixin)   # OOT plugin
└── MMPlatform(DeviceMixin)            # + attention backend, VAE, … (future)
    └── MyMMPlatform(MMPlatform, MyDeviceMixin)      # OOT plugin
```

Key design points:
- **DeviceMixin** provides platform identity queries (`is_cuda()`, `is_npu()`, etc.) and device operations (`set_device()`, `get_device_name()`, etc.)
- **SRTPlatform** adds SRT-specific factory methods, capability flags, and lifecycle hooks
- OOT plugins implement a **device mixin** (vendor-specific operations) and compose it with **SRTPlatform** via multiple inheritance
- All methods are **instance methods** (not classmethods), called through the `current_platform` singleton
- Device operations and factory methods raise `NotImplementedError` by default (fail-fast)
- Capability flags use safe conservative defaults (`False`/`pass`)
- Methods are annotated `[Active]` (called by SGLang core) or `[Planned]` (reserved for future migration)

### Platform Discovery (`current_platform`)

`current_platform` is a **lazy singleton** in `sglang.srt.platforms`. On first access it resolves the active platform through the following priority chain:

```
entry_points("sglang.srt.platforms")  → Enumerate ALL plugins by name (metadata only)
  │
  ├─ SGLANG_PLATFORM set (front-loading filter):
  │   ├─ Name not found in discovered → RuntimeError
  │   ├─ activate() returns non-None  → load that platform
  │   └─ activate() returns None      → RuntimeError (hardware unavailable)
  │
  └─ SGLANG_PLATFORM unset (auto-discover, activate all):
      ├─ 0 activated → fallback base SRTPlatform
      ├─ 1 activated → use it
      └─ N activated → RuntimeError (must set SGLANG_PLATFORM)
```

### Plugin Loading Flow

`load_plugins()` discovers and executes general plugins, then applies all registered hooks. It is called at four points:

| Call Site | Process | Timing |
|---------|------|------|
| `cli/serve.py` serve() | Main | Before `prepare_server_args()` |
| `launch_server.py` `__main__` | Main | Before `prepare_server_args()` |
| `engine.py` `_launch_subprocesses()` | Main | Before `server_args.check_server_args()` |
| `scheduler.py` `run_scheduler_process()` | Subprocess | Before `Scheduler()` construction |

> **Note**: `load_plugins()` is idempotent (guarded by `_plugins_loaded` flag). In spawn'd subprocesses the flag resets, so plugins are correctly re-loaded.

```
load_plugins()
  ├── _get_excluded_dists()                       → compute dists to skip (via SGLANG_PLATFORM)
  ├── load_plugins_by_group("sglang.srt.plugins",     → discover entry_points, filter by SGLANG_PLUGINS
  │     excluded_dists=...)                          skip plugins from unselected platform packages
  ├── for each plugin:                            → set _current_plugin_source context var
  │     func()                                      side effects (register hooks with source tracking)
  └── HookRegistry.apply_hooks()                  → monkey-patch targets
```

---

## Plugin Type 1: Hardware Platform Plugin

### Description

A hardware platform plugin registers an `SRTPlatform` subclass that tells SGLang how to interact with a specific hardware backend.

### Quick Start

**1. Create a minimal package:**

```
my_platform_plugin/
├── pyproject.toml
└── my_platform_plugin/
    ├── __init__.py    # activate() function
    ├── device.py      # MyDeviceMixin
    └── platform.py    # MySRTPlatform
```

**2. `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "my-platform-plugin"
version = "0.1.0"

[project.entry-points."sglang.srt.platforms"]
my_device = "my_platform_plugin:activate"
```

**3. `__init__.py`** — activation function:

```python
def activate():
    """Return fully-qualified class name to activate, or None to skip."""
    if _my_device_is_available():
        return "my_platform_plugin.platform.MySRTPlatform"
    return None
```

**4. `device.py`** — device mixin:

```python
from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

class MyDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "my_device"
    device_type = "my_device"   # torch device type

    def set_device(self, device) -> None: ...
    def get_device_name(self, device_id=0) -> str: ...
    def get_device_total_memory(self, device_id=0) -> int: ...
    def get_current_memory_usage(self, device=None) -> float: ...
    def get_device_capability(self, device_id=0): ...
    def get_torch_distributed_backend_str(self) -> str: ...
```

**5. `platform.py`** — SRT platform:

```python
from sglang.srt.platforms.interface import SRTPlatform
from my_platform_plugin.device import MyDeviceMixin

class MySRTPlatform(SRTPlatform, MyDeviceMixin):
    def get_default_attention_backend(self) -> str: ...
    def support_cuda_graph(self) -> bool: ...
    # ... override other methods as needed
```

**6. Install and verify:**

```bash
pip install -e my_platform_plugin/
python -c "from sglang.srt.platforms import current_platform; print(current_platform)"
```

### Platform Interface Reference

#### Identity Queries (from DeviceMixin)

| Method | Default | Description |
|---|---|---|
| `is_cuda()` | Based on `_enum` | Whether this is an NVIDIA CUDA platform |
| `is_rocm()` | Based on `_enum` | Whether this is an AMD ROCm platform |
| `is_npu()` | Based on `_enum` | Whether this is a Huawei NPU platform |
| `is_cpu()` | Based on `_enum` | Whether this is a CPU-only platform |
| `is_xpu()` | Based on `_enum` | Whether this is an Intel XPU platform |
| `is_musa()` | Based on `_enum` | Whether this is a Moore Threads MUSA platform |
| `is_cuda_alike()` | CUDA+ROCM+MUSA | True if the hardware supports CUDA-like APIs |
| `is_out_of_tree()` | `True` for OOT | Automatically detected based on `_enum = PlatformEnum.OOT` |

#### Device Operations (from DeviceMixin)

> Methods annotated **[Active]** are called by SGLang core through `current_platform` — OOT implementations take effect immediately.
> Methods annotated **[Planned]** are reserved interfaces — SGLang core still uses hardcoded calls (e.g. `torch.cuda.empty_cache()`). OOT implementations will NOT take effect until the core is migrated in a future PR.

| Method | Default | Status | Description |
|---|---|---|---|
| `get_device(local_rank)` | `raise NotImplementedError` | Planned | Return `torch.device` for a given local rank |
| `set_device(device)` | `raise NotImplementedError` | Planned | Set the current device |
| `get_device_name(device_id)` | `raise NotImplementedError` | Planned | Get human-readable device name |
| `get_device_uuid(device_id)` | `raise NotImplementedError` | Planned | Get unique device identifier |
| `get_device_capability(device_id)` | `raise NotImplementedError` | Planned | Get `DeviceCapability(major, minor)`. None if N/A |
| `empty_cache()` | `pass` | Planned | Release cached device memory |
| `synchronize()` | `pass` | Planned | Synchronize device operations |
| `get_device_total_memory(device_id)` | `raise NotImplementedError` | **Active** | Get total device memory in bytes |
| `get_available_memory(device_id)` | `raise NotImplementedError` | Planned | Return `(free_bytes, total_bytes)` |
| `get_current_memory_usage(device)` | `raise NotImplementedError` | **Active** | Get current peak memory usage in bytes |
| `get_torch_distributed_backend_str()` | `raise NotImplementedError` | Planned | Distributed backend string (e.g. "nccl", "hccl") |
| `get_communicator_class()` | `None` | Planned | Platform-specific communicator class |
| `inference_mode()` | `torch.inference_mode(True)` | Planned | Return inference mode context manager |
| `seed_everything(seed)` | Set random/np/torch seeds | Planned | Set random seeds for reproducibility |
| `verify_quantization(quant)` | `pass` | Planned | Validate quantization method support |
| `get_cpu_architecture()` | Auto-detect x86/arm | Planned | Detect CPU architecture (`CpuArchEnum`) |

#### Types (from DeviceMixin)

| Type | Description |
|---|---|
| `PlatformEnum` | Enumeration of platform types: CUDA, ROCM, CPU, XPU, MUSA, NPU, TPU, MPS, OOT, UNSPECIFIED |
| `CpuArchEnum` | CPU architecture: X86, ARM, UNSPECIFIED |
| `DeviceCapability` | `NamedTuple(major, minor)` with comparison support. Methods: `as_version_str()`, `to_int()` |

#### Capability Flags (from SRTPlatform)

| Method | Default | Description |
|---|---|---|
| `support_cuda_graph()` | `False` | Whether device graph capture is supported (plain CUDA graph) |
| `support_piecewise_cuda_graph()` | `False` | Whether piecewise CUDA graph (torch.compile backend) is supported |
| `supports_fp8()` | `False` | Whether FP8 quantization is supported |
| `is_pin_memory_available()` | `True` | Whether pinned memory is available |

#### Subsystem Factory Methods (from SRTPlatform)

| Method | Default | Description |
|---|---|---|
| `get_default_attention_backend()` | `raise NotImplementedError` | Default attention backend name |
| `get_graph_runner_cls()` | `raise NotImplementedError` | Graph Runner class |
| `get_mha_kv_pool_cls()` | `raise NotImplementedError` | MHA KV cache pool class |
| `get_mla_kv_pool_cls()` | `raise NotImplementedError` | MLA KV cache pool class |
| `get_nsa_kv_pool_cls()` | `raise NotImplementedError` | NSA KV cache pool class (DeepSeek V3.2) |
| `get_paged_allocator_cls()` | `raise NotImplementedError` | Paged allocator class |
| `get_piecewise_backend_cls()` | `raise NotImplementedError` | Piecewise compilation backend class |
| `get_compile_backend(mode)` | `"inductor"` | Compilation backend string |
| `get_dispatch_key_name()` | `"native"` | MultiPlatformOp dispatch key name |

#### Lifecycle Hooks (from SRTPlatform)

| Method | Invocation Timing | Purpose |
|---|---|---|
| `apply_server_args_defaults(server_args)` | After ServerArgs parsing, in `__post_init__` | Set platform-specific defaults |
| `init_backend()` | In each worker, before model construction | One-time backend initialization |

### Environment Variables

| Variable | Description |
|---|---|
| `SGLANG_PLATFORM` | Select the platform plugin by entry_point name (e.g. `kunlun`, `demo_cuda`). When set, **only** the named plugin's `activate()` is called (front-loading filter) — other plugins are not touched. Additionally, general plugins (`sglang.srt.plugins`) from unselected platform packages are automatically skipped to avoid importing their dependencies. Required when multiple plugins would activate. Errors if the name is not found or if the plugin's hardware is unavailable. |
| `SGLANG_PLUGINS` | Comma-separated whitelist of general plugin names to load (group: `sglang.srt.plugins`). If unset, all discovered general plugins are loaded. |

---

## Plugin Type 2: General Plugin

### Description

General function plugins inject behavior into sglang **without requiring a custom platform**. Use cases include:

- **Observability**: Add logging, metrics, and tracing to any function
- **Behavior modification**: Modify function arguments or return values
- **Performance profiling**: Add timing to critical functions
- **A/B testing**: Replace implementations at runtime

### Quick Start

**1. Create a minimal package:**

```
my_general_plugin/
├── pyproject.toml
└── my_general_plugin/
    └── __init__.py    # register() function
```

**2. `pyproject.toml`:**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "my-general-plugin"
version = "0.1.0"

[project.entry-points."sglang.srt.plugins"]
my_plugin = "my_general_plugin:register"
```

**3. `__init__.py`** — register hooks:

```python
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

def register():
    """Entry point called by load_plugins()."""
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.__init__",
        my_hook,
        HookType.AROUND,
    )

def my_hook(original_fn, self, *args, **kwargs):
    result = original_fn(self, *args, **kwargs)
    print(f"Scheduler initialized! gpu_id={self.gpu_id}")
    return result
```

**4. Install and run:**

```bash
pip install -e my_general_plugin/
sglang serve --model-path <model> [options]
# Look for "Scheduler initialized!" in logs
```

### Hook Types

`HookRegistry` supports four hook types:

| Hook Type | Signature | Description |
|---|---|---|
| **BEFORE** | `fn(*args, **kwargs) -> (args, kwargs) \| None` | Runs before the original. Return `None` to keep args unchanged, or `(args, kwargs)` to modify. |
| **AFTER** | `fn(result, *args, **kwargs) -> new_result \| None` | Runs after the original. Return `None` to keep result, or a new value to replace. |
| **AROUND** | `fn(original_fn, *args, **kwargs) -> result` | Wraps the original. You must call `original_fn` yourself. Full control over execution. |
| **REPLACE** | `fn(*args, **kwargs) -> result` or `class` | Replace the original function or class entirely. For class targets, pass a replacement class directly — it is substituted via `setattr` preserving `isinstance()`/`issubclass()` semantics. |

> **Note**: Only `REPLACE` accepts a class as the hook. Passing a class to `BEFORE`/`AFTER`/`AROUND` raises `TypeError` at registration time.

### Registration API

Hooks can be registered using the **imperative API** or the **decorator API**:

```python
# --- Imperative API ---
from sglang.srt.plugins.hook_registry import HookRegistry, HookType

def my_timer(original_fn, *args, **kwargs):
    start = time.perf_counter()
    result = original_fn(*args, **kwargs)
    print(f"Elapsed: {time.perf_counter() - start:.3f}s")
    return result

HookRegistry.register(
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    my_timer,
    HookType.AROUND,
)

# --- Decorator API ---
from sglang.srt.plugins.hook_registry import plugin_hook, HookType

@plugin_hook(
    "sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run",
    type=HookType.AROUND,
)
def my_timer(original_fn, *args, **kwargs):
    start = time.perf_counter()
    result = original_fn(*args, **kwargs)
    print(f"Elapsed: {time.perf_counter() - start:.3f}s")
    return result

# --- Class replacement (REPLACE) ---
from sglang.srt.plugins.hook_registry import plugin_hook, HookType
from sglang.srt.managers.scheduler import Scheduler

@plugin_hook(
    "sglang.srt.managers.scheduler.Scheduler",
    type=HookType.REPLACE,
)
class MyScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Enhanced scheduler initialized!")
```

### Hook Target Resolution

Target paths use fully-qualified dotted notation. Both formats are supported:

- **Dotted**: `sglang.srt.managers.scheduler.Scheduler.__init__`
- **Entry-points style**: `sglang.srt.managers.scheduler:Scheduler.__init__` (colon treated as dot)

### Common Hook Targets

| Target | Description |
|---|---|
| `sglang.srt.server_args.ServerArgs.add_cli_args` | Add custom CLI arguments |
| `sglang.srt.server_args.ServerArgs.__post_init__` | Modify ServerArgs after parsing |
| `sglang.srt.server_args.ServerArgs.check_server_args` | Add/relax validation |
| `sglang.srt.managers.scheduler.Scheduler.__init__` | Custom scheduler state |
| `sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run` | Custom scheduling policy |
| `sglang.srt.managers.scheduler.Scheduler.run_batch` | Profiling / inspection |
| `sglang.srt.managers.scheduler.Scheduler.process_batch_result` | Custom metrics |
| `sglang.srt.managers.tp_worker.TpModelWorker.__init__` | Custom worker state |
| `sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation` | Forward pass wrapping |

---

## File Reference

| File | Description |
|---|---|
| `sglang/srt/platforms/device_mixin.py` | `PlatformEnum` + `DeviceMixin` base class |
| `sglang/srt/platforms/interface.py` | `SRTPlatform` base class (extends DeviceMixin) |
| `sglang/srt/platforms/__init__.py` | `current_platform` lazy singleton + discovery logic |
| `sglang/srt/plugins/__init__.py` | `load_plugins()` + `load_plugins_by_group()` |
| `sglang/srt/plugins/hook_registry.py` | `HookRegistry`, `HookType`, `plugin_hook` decorator |

## How to transform model instances with PyTorch FX Toolkit in SGLang for NPU

### PassManager
`PassManager` is implemented here: [PassManager](https://github.com/eshoguli/sglang/blob/eshogulin/pass_manager/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/pass_manager.py)


You can explore `PassManager` usage in [`NpuGraphCompilerBackend`](https://github.com/eshoguli/sglang/blob/eshogulin/pass_manager/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/npu_graph_compiler_backend.py) compiler backend. [`PiecewiseNpuGraphCompilerBackend`](https://github.com/eshoguli/sglang/blob/eshogulin/pass_manager/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/piecewise_npu_graph_compiler_backend.py) compiler backed uses `PassManager` too via `NpuGraphCompilerBackend` inheritance.

### Pass development
There are two approaches to develop passes for SGLang NPU PassManager:

1. Matches all possible non-overlapping sets of operators and their data dependencies with `torch.fx.replace_pattern` api.
Pass example: [NpuAddRmsNormQuantFuse](https://github.com/eshoguli/sglang/blob/3365d711fd5aa0d6191c32769163320fe41e27f2/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/passes/w8a8_int8.py#L82).
You can find details on official FX toolkit web site: https://docs.pytorch.org/docs/stable/fx.html#subgraph-rewriting-with-replace-pattern

2. Direct Graph Manipulation.
Pass example: [EraseCopy](https://github.com/eshoguli/sglang/blob/3365d711fd5aa0d6191c32769163320fe41e27f2/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/passes/w8a8_int8.py#L28).
You can find details on official FX toolkit web site: https://docs.pytorch.org/docs/stable/fx.html#direct-graph-manipulation

### Compiler backend update
After pass development you should create `PassManager` instance, add the pass and call `apply` method:
```
def apply_passes(self, graph_module: torch.fx.GraphModule):
    passManager = PassManager(graph_module)
    passManager.add(NpuAddRmsNormQuantFuse)
    passManager.apply()
    graph_module.recompile()
```

You can explore [`NpuGraphCompilerBackend`](https://github.com/eshoguli/sglang/blob/eshogulin/pass_manager/python/sglang/srt/hardware_backend/npu/graph_runner/compilation/npu_graph_compiler_backend.py) as example.

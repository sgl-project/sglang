# Quantization Architecture

## Overview

All quantization implementations are centralized in `python/sglang/srt/layers/quantization/`, connecting model construction, weight loading, and inference via unified abstractions and lifecycle hooks. The architecture employs a three-stage lifecycle that decouples parameter allocation, weight transformation, and computation:

- **`create_weights`**: Pre-allocates memory for quantized weights, scales, zero points, and quantization parameters during model construction. Establishes memory layout prior to data flow.

- **`process_weights_after_loading`**: Transforms loaded weights and scales into kernel-optimal formats (CUTLASS-compatible interleaving, dtype conversions, stride pre-computation) to minimize inference overhead.

- **`apply`**: Executes quantized forward pass by quantizing activations, invoking computational kernels (CUTLASS, Marlin, Triton), and dequantizing outputs.

Architecture goals:

- Configuration-driven quantization scheme selection, compatible with HuggingFace formats
- Reusable quantization interfaces for linear layers, MoE layers, KV cache, and other components
- Multi-backend support (CUDA, ROCm) with fallback strategies for missing dependencies
- Minimal interface implementation required for new quantization scheme integration

Core base classes in `base_config.py`:

- `QuantizeMethodBase`: Abstract base class defining the three-stage lifecycle: `create_weights`, `apply`, and `process_weights_after_loading`
- `LinearMethodBase` / `FusedMoEMethodBase`: Specialized base classes for linear and fused MoE layers, providing context for expert routing and parallelism
- `QuantizationConfig`: Configuration abstraction for parsing quantization configs, validating hardware capabilities and activation dtypes, and returning quantization method instances per layer

The quantization method registry in `__init__.py` maps native schemes (AWQ, GPTQ, FP8, W4AFp8, ModelOpt, etc.) to string identifiers and injects platform-specific configurations based on CUDA/HIP capabilities (e.g., `quark` under ROCm).

## Directory Structure

```
python/sglang/srt/layers/quantization/
├── base_config.py                 # Abstract base classes and utilities
├── __init__.py                    # Quantization method registration and dependency detection
├── unquant.py                     # Unquantized method implementation (UnquantizedLinearMethod)
├── utils.py / int8_utils.py       # Layer filtering, dynamic overrides, INT8 utilities
├── kv_cache.py                    # KV Cache quantization method base class
├── fp8.py / fp8_kernel.py / fp8_utils.py    # FP8 linear + MoE implementation
├── w8a8_int8.py / w8a8_fp8.py     # INT8 / FP8 combined schemes
├── w4afp8.py                      # 4bit weight + FP8 activation mixed precision
├── awq.py / gptq.py / gguf.py     # Pre-quantized format import
├── modelopt_quant.py              # ModelOpt FP8/FP4 configuration
├── mxfp4.py / mxfp4_tensor.py     # MXFP4 implementation
├── petit.py / petit_utils.py      # Petit NVFP4 implementation
├── marlin_utils.py / marlin_utils_fp8.py  # Marlin kernel utilities
├── int8_kernel.py                 # INT8 kernel implementation
├── kvfp4_tensor.py                # KV Cache FP4 tensor
├── compressed_tensors/            # General compressed tensor framework
└── quark/                         # ROCm-specific MXFP4 / INT4
```

Related modules:

- `python/sglang/srt/configs/model_config.py`: Parses HuggingFace configs and `hf_quant_config.json`, normalizes quantization scheme names via `_parse_quant_hf_config`
- `python/sglang/srt/model_loader/weight_utils.py`: Instantiates `QuantizationConfig` from scheme names, downloads/reads external quantization configs, validates hardware capabilities
- `python/sglang/srt/model_loader/loader.py`: Injects quantization config during `_initialize_model`, calls `process_weights_after_loading` hook per layer after checkpoint loading
- `python/sglang/srt/layers/linear.py`: Base class for linear layers, stores `quant_method`, registers quantization weights, dispatches `apply` during forward pass

## Overall Workflow

The quantization workflow comprises four phases: configuration parsing, model construction, weight loading, and inference execution.

```
ModelConfig._parse_quant_hf_config → Determine quant_method (e.g., w4afp8)
      ↓
weight_utils.get_quant_config → Construct corresponding QuantizationConfig instance
      ↓
_initialize_model(...) → Pass quant_config to model/layers
      ↓
LinearBase.quant_method.create_weights → Register quantization weight placeholders
      ↓
DefaultModelLoader.load_weights_and_postprocess →
    First: load_weights() fills quantized weight tensors
    Then: Call quant_method.process_weights_after_loading layer by layer
      ↓
During inference: LinearBase.forward → quant_method.apply executes quantized GEMM
```

### Configuration Parsing

Configuration parsing identifies the quantization scheme from model metadata and instantiates the configuration object:

1. `ModelConfig._parse_quant_hf_config` reads `config.json` or `hf_quant_config.json`, generating quantization method identifiers (e.g., `w4afp8`, `awq`). ModelOpt `quant_algo == "MIXED_PRECISION"` maps to `w4afp8`.

2. `loader._get_quantization_config` calls `weight_utils.get_quant_config`, which resolves the configuration class via `quantization.get_quantization_config` and instantiates it via `from_config`, populating hardware requirements, activation dtypes, skip layers, and additional config files.

3. `QuantizationConfig.from_config` parses the configuration dictionary and validates hardware capabilities:
   - **NVIDIA GPU**: CUDA compute capability (70=Volta, 75=Turing, 80=Ampere, 90=Hopper)
   - **AMD GPU**: ROCm/HIP platform detection; some schemes require specific GCN architectures (e.g., gfx94)

### Model Construction

Model construction injects quantization methods into layers and registers parameter placeholders:

1. The model loader passes `quant_config` to `_initialize_model`, propagating it through the model hierarchy (e.g., `DeepseekV2ForCausalLM` → `DeepseekV2DecoderLayer` → `DeepseekV2AttentionMLA` → `RowParallelLinear`).

2. Modules inheriting from `LinearBase` or `FusedMoE` call `quant_config.get_quant_method(self, prefix)` in `__init__`:
   - Returns quantization method instance (e.g., `Fp8LinearMethod`, `W4AFp8MoEMethod`) if quantization is required
   - Falls back to `UnquantizedLinearMethod` for skipped layers

3. `create_weights` registers weight placeholders (quantized weights, scales, zero points, input scaling) and optionally binds custom `weight_loader`. Parameters are allocated via `torch.empty` or `torch.zeros`, establishing memory layout without data.

**Call Flow for `create_weights`**:
```
DeepseekV2DecoderLayer.__init__()
      ↓
DeepseekV2AttentionMLA.__init__()
      ↓
RowParallelLinear.__init__()
      ↓
LinearBase.__init__()  # self.quant_method is set via quant_config.get_quant_method()
      ↓
quant_config.get_quant_method() → Returns Fp8LinearMethod or W4AFp8MoEMethod
      ↓
RowParallelLinear.__init__() continues execution
      ↓
Fp8LinearMethod.create_weights() or W4AFp8MoEMethod.create_weights()
      ↓
Register weight, weight_scale, input_scale and other parameter placeholders
```

### Weight Loading and Post-processing

Weight loading fills quantized tensors from checkpoints, followed by format transformation for kernel optimization:

1. `DefaultModelLoader.load_weights_and_postprocess` calls `model.load_weights()` to populate registered tensors layer by layer.

2. After loading completes, the system iterates through the model calling `process_weights_after_loading` for modules with `quant_method`:
   - Repack/transpose weights to match kernel layouts (Marlin, CUTLASS, block quantization)
   - Convert scale/zero point dtypes (float32 → bfloat16) to reduce forward overhead
   - Interleave scales to match kernel memory access patterns for improved cache locality
   - Pre-compute stride, group size, expert mappings, and cache in module attributes

**Call Flow for `process_weights_after_loading`**:
```
Scheduler.__init__()
      ↓
TpModelWorker.__init__()
      ↓
ModelRunner.__init__()
      ↓
ModelRunner.initialize()
      ↓
get_model()
      ↓
DefaultModelLoader.load_model()
      ↓
DefaultModelLoader.load_weights_and_postprocess()
      ↓
model.load_weights() → Load weight data from checkpoint
      ↓
Iterate through layers, calling quant_method.process_weights_after_loading()
      ↓
Fp8LinearMethod.process_weights_after_loading()
      └─→ or W4AFp8MoEMethod.process_weights_after_loading()
```

### Inference Execution

Forward propagation dispatches quantization methods through the `apply` interface:

Linear layers call `self.quant_method.apply(self, x, bias)`; MoE layers call implementations of `FusedMoEMethodBase.apply`. These functions:

1. Quantize input activations or read pre-quantized tensors
2. Invoke CUDA/ROCm/Triton kernels for quantized GEMM or MoE routing
3. Dequantize outputs, add bias, and return to upstream layers

**Call Flow for `apply`**:
```
DeepseekV2DecoderLayer.forward()
      ↓
DeepseekV2AttentionMLA.forward()
      ↓
RowParallelLinear.forward()
      ↓
self.quant_method.apply()
      ↓
Fp8LinearMethod.apply()
      └─→ or W4AFp8MoEMethod.apply()
      ↓
Call underlying kernel (CUTLASS/Marlin/torch) to execute quantized GEMM
```

## Class Inheritance Hierarchy

The quantization architecture employs an inheritance hierarchy separating configuration management from method implementation:

### Configuration Class Hierarchy

```
QuantizationConfig (Abstract base class, defined in base_config.py)
  Responsibilities:
    - Parse quantization configuration from dictionaries
    - Validate hardware capabilities (CUDA compute capability, ROCm architecture)
    - Validate activation data types
    - Return correct quantization method instances layer by layer
  
  Key Methods:
    ├─ from_config(): Parse from configuration dictionary and instantiate
    ├─ get_quant_method(layer, prefix): Return quantization method based on layer type
    ├─ get_min_capability(): Validate hardware compatibility
    │  Hardware validation includes:
    │  • NVIDIA GPU: Check via CUDA compute capability (e.g., 70=Volta, 75=Turing, 80=Ampere, 90=Hopper)
    │  • AMD GPU: Check via ROCm/HIP platform detection; some schemes require specific GCN architectures (e.g., gfx94)
    └─ get_supported_act_dtypes(): Return supported activation data types
  
      ↓
  
  W4AFp8Config (Concrete configuration class)
      └─→ get_quant_method() returns based on layer type:
          ├─→ LinearBase → Fp8LinearMethod
          └─→ FusedMoE → W4AFp8MoEMethod
```

### Quantization Method Class Hierarchy

```
QuantizeMethodBase (Abstract base class, defined in base_config.py)
  Responsibilities:
    - Weight registration (model construction stage)
    - Post-loading weight processing (after weight loading completes)
    - Forward execution (inference stage)
  
  Core Interface:
    ├─ create_weights(): Register quantization weight placeholders
    ├─ process_weights_after_loading(): Post-process weights
    └─ apply(): Execute quantized computation during forward propagation
  
      ├─→ LinearMethodBase
      │     └─→ Fp8LinearMethod (for regular linear layers)
      │     └─→ UnquantizedLinearMethod (fallback for skipped layers)
      │     └─→ [Other linear quantization methods]
      │
      └─→ FusedMoEMethodBase
            └─→ W4AFp8MoEMethod (for MoE layers with INT4 weights + FP8 activations)
            └─→ [Other MoE quantization methods]
```

### Quantization Method Registry

The quantization method registry in `__init__.py`:

- Maps string identifiers (`"w4afp8"`, `"fp8"`, `"awq"`, etc.) to configuration classes
- Injects platform-specific configurations based on CUDA/HIP capabilities (e.g., `quark` under ROCm)
- Performs dependency detection for optional quantization backends

## Unified Structure of Quantization Methods and W4AFp8 Example

All quantization methods (FP8, INT8, INT4, third-party formats) follow a consistent structure:

1. **Inherit Abstract Base Classes**: Linear methods inherit `LinearMethodBase`, MoE methods inherit `FusedMoEMethodBase`, both implementing `QuantizeMethodBase` interfaces.
2. **Implement Three Core Methods**:
   - `create_weights`: Register quantized weights, scales, zero points, and optionally define weight loaders
   - `apply`: Execute quantized forward pass (activation quantization, kernel invocation, dequantization)
   - `process_weights_after_loading` (optional): Re-quantize, pack, transpose weights after loading
3. **Configuration-Driven**: Each scheme provides a `QuantizationConfig` subclass implementing `from_config`, `get_quant_method`, `get_supported_act_dtypes`, `get_min_capability` for parsing configs and validating hardware compatibility.
4. **Unified Lifecycle**: `create_weights` during initialization, `process_weights_after_loading` during weight loading, `apply` during inference. New schemes implement these hooks to integrate into the pipeline.

The following uses W4AFp8 to demonstrate the complete workflow: configuration parsing, weight registration, post-processing, and inference.

### W4AFp8 Configuration Identification and Object Construction

`W4AFp8Config` inherits from `QuantizationConfig` and maps configuration parameters to quantization method instances.

**Configuration Recognition**: `ModelConfig._parse_modelopt_quant_config` maps `quant_algo == "MIXED_PRECISION"` in `hf_quant_config.json` to `w4afp8` and validates hardware compatibility:

```python
# ModelConfig._parse_modelopt_quant_config
if quant_algo == "MIXED_PRECISION":
    return {"quant_method": "w4afp8"}
```

**Object Construction**: `weight_utils.get_quant_config` resolves `W4AFp8Config` via `quantization.get_quantization_config` using identifier `"w4afp8"`, then instantiates via `from_config`. Additional JSON files in the model directory are loaded in `from_config`, supplementing skip layers, expert mappings, etc.

**Key Methods**:
- `W4AFp8Config.from_config()`: Parse configuration dictionary and instantiate
- `W4AFp8Config.get_quant_method(layer, prefix)`: Returns quantization method instance based on layer type:
  ```python
  if isinstance(layer, LinearBase):
      return Fp8LinearMethod(self)  # Regular layers use Fp8LinearMethod
  elif isinstance(layer, FusedMoE):
      return W4AFp8MoEMethod(self)  # MoE layers use W4AFp8MoEMethod
  ```

Configuration instances contain activation scaling schemes (dynamic/static), weight quantization group size, and checkpoint quantization format flags.

### W4AFp8 Model Construction and Method Injection

`W4AFp8Config` injects quantization methods into layers during model construction:

- `_initialize_model` passes `quant_config` to models (e.g., `DeepseekV2ForCausalLM`), propagating it to decoder layers, MoE modules, and linear layers.
- `LinearBase` layers call `quant_config.get_quant_method`: W4AFp8 assigns `Fp8LinearMethod` (FP8 activations) to regular linear layers and `W4AFp8MoEMethod` (INT4 weights + FP8 activations) to MoE layers. Skipped layers use `UnquantizedLinearMethod`:
  ```python
  # LinearBase.__init__
  if quant_config is None:
      self.quant_method = UnquantizedLinearMethod()
  else:
      self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
  ```
- `create_weights` registers packed INT4/INT8 weight tensors, scales, input activation scaling, expert weight indices, and binds custom loading logic for checkpoint loaders.

### W4AFp8MoEMethod Implementation Details

`W4AFp8MoEMethod` implements W4AFp8 for MoE layers following the three-stage lifecycle:

#### create_weights

Pre-allocates quantization parameters during `FusedMoE` initialization:

- **Quantized weight tensors**: `w13_weight` (gate/up projection) and `w2_weight` (down projection), `int8` dtype (packed INT4)
- **Weight scales**: `w13_weight_scale_inv` and `w2_weight_scale_inv`, group-wise (128 elements per group)
- **Activation scales**: `w13_input_scale` and `w2_input_scale` for static quantization
- **Computation metadata**: Stride arrays, expert offsets, problem sizes

Parameters are allocated via `torch.empty` or `torch.zeros`, establishing memory layout without data.

```python
def create_weights(self, layer, num_experts, hidden_size, ...):
    # Create quantized weight containers (INT8 type, packed INT4)
    layer.register_parameter("w13_weight", torch.empty(..., dtype=torch.int8))
    layer.register_parameter("w2_weight", torch.empty(..., dtype=torch.int8))
    # Create weight scale factors (group-wise, 128 elements per group)
    layer.register_parameter("w13_weight_scale_inv", torch.zeros(...))
    layer.register_parameter("w2_weight_scale_inv", torch.zeros(...))
    # Create input scale factors (used in static quantization)
    layer.register_parameter("w13_input_scale", torch.ones(..., dtype=torch.bfloat16))
    # Initialize stride and other computation metadata
    self.a_strides1 = torch.full((num_experts, 3), hidden_size, ...)
```

#### process_weights_after_loading

Transforms loaded weights and scales to kernel-optimal formats:

- **Weight scale optimization**: Convert scales from float32 to bfloat16 (50% memory reduction), then interleave via `interleave_scales` to match CUTLASS memory access patterns (TRT-LLM reference), improving cache locality.

- **Input scale aggregation**: In static quantization, aggregate per-expert input scales into a single scalar to reduce inference overhead.

```python
def process_weights_after_loading(self, layer: Module) -> None:
    # Convert weight scale to bfloat16 and rearrange to match CUTLASS layout
    w13_weight_scale = layer.w13_weight_scale_inv.to(torch.bfloat16)
    w13_weight_scale = interleave_scales(w13_weight_scale)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)
    
    # Aggregate input scale into a single scalar (static quantization mode)
    w13_input_scale_max = layer.w13_input_scale.max().to(torch.bfloat16).item()
    layer.w13_input_scale = Parameter(
        torch.tensor([w13_input_scale_max], dtype=torch.bfloat16, device=device),
        requires_grad=False
    )
```

#### apply

Executes quantized forward pass:

Collects preprocessed data (activations, rearranged weights/scales, routing results) and invokes `cutlass_w4a8_moe` kernel for two GEMM operations:
- **GEMM1**: `w13_weight` (gate/up projection)
- **GEMM2**: `w2_weight` (down projection)

`cutlass_w4a8_moe` wraps CUTLASS, implementing INT4 weight + FP8 activation mixed-precision GEMM, leveraging hardware quantized computation capabilities.

```python
def apply(self, layer, dispatch_output) -> CombineInput:
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    
    x = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    
    # Call CUTLASS kernel to execute mixed-precision MoE computation
    output = cutlass_w4a8_moe(
        x, layer.w13_weight, layer.w2_weight,
        layer.w13_weight_scale_inv, layer.w2_weight_scale_inv,
        topk_weights, topk_ids,
        self.a_strides1, self.b_strides1, self.c_strides1,  # GEMM1 strides
        self.a_strides2, self.b_strides2, self.c_strides2,  # GEMM2 strides
        self.s_strides13, self.s_strides2,                  # Scale strides
        self.expert_offsets, self.problem_sizes1, self.problem_sizes2,
        layer.w13_input_scale, layer.w2_input_scale,
    )
    # Apply routing scale factor
    if self.moe_runner_config.routed_scaling_factor is not None:
        output *= self.moe_runner_config.routed_scaling_factor
    return StandardCombineInput(hidden_states=output)
```

### Fp8LinearMethod Implementation

For regular linear layers, `W4AFp8Config` assigns `Fp8LinearMethod`:

- **`create_weights`**: Register `weight`, `weight_scale`, `input_scale` placeholders
- **`process_weights_after_loading`**: Transform weights/scales to hardware-optimal layouts (Marlin, CUTLASS, etc.)
- **`apply`**: Invoke appropriate kernel (Marlin, CUTLASS, etc.) for FP8 GEMM:

```python
# Fp8LinearMethod.apply
return apply_fp8_linear(
    input=x,
    weight=layer.weight,
    weight_scale=layer.weight_scale,
    input_scale=layer.input_scale,
    bias=bias,
    cutlass_fp8_supported=self.cutlass_fp8_supported,
)
```

## Supported Quantization Schemes Overview

| Category              | Representative Configurations                                                                 | Description                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| FP8 Series            | `fp8`, `w8a8_fp8`, `modelopt_fp8`, `fbgemm_fp8`                                               | Native FP8, W8A8-FP8 hybrid, ModelOpt/FBGEMM implementations                                  |
| INT8 Series           | `w8a8_int8`, `blockwise_int8`                                                                 | Classic 8bit weight/activation, blockwise INT8                                           |
| INT4/Mixed Precision  | `w4afp8`, `qoq`, `moe_wna16`                                                                  | 4bit weight + FP8 activation, QoQ, WNA16 (W4A16/W8A16)                                  |
| FP4 / MXFP4           | `modelopt_fp4`, `petit_nvfp4`, `mxfp4`, `quark`                                               | FP4 / MXFP4 schemes, `quark` is ROCm-specific                                           |
| Pre-quantized Formats | `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `gguf`, `compressed-tensors`, `auto-round`, `modelopt` | Integration with external toolchains or compressed tensor frameworks, `modelopt` auto-detects FP8/FP4 |
| KV Cache Quantization | `BaseKVCacheMethod` and its subclasses in `kv_cache.py`                                       | Provides scale and zero-point management for attention caches                           |

## Steps to Extend New Quantization Schemes

The decoupled architecture enables straightforward integration of new quantization schemes. To integrate a new scheme (e.g., W2A8):

1. **Implement Configuration Class**: Inherit from `QuantizationConfig`, parse custom parameters, implement `get_quant_method`. Handle hardware capability validation and activation dtype checking.

2. **Implement Quantization Method Class**: Inherit from `LinearMethodBase` / `FusedMoEMethodBase`, implement the three-stage lifecycle:
   - `create_weights`: Register quantized weight placeholders and parameter containers
   - `apply`: Execute quantized forward pass
   - `process_weights_after_loading` (optional): Format conversion and optimization after weight loading

3. **Register Scheme**: Register string identifier and configuration class mapping in `BASE_QUANTIZATION_METHODS` in `__init__.py`. Add dependency detection logic if needed (e.g., CUTLASS, Marlin, kernel dependencies).

4. **Integrate Utilities**: Implement new kernels/utilities in `utils.py`, `int8_utils.py`, or separate files. Call from `apply`. Ensure target hardware backend compatibility.

5. **Write Tests**: Ensure unit tests or inference scripts pass on target hardware, validating configuration parsing and lifecycle hooks. Test hardware capability detection and fallback behavior.
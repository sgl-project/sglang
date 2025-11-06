# Quantization Architecture

## 1. Overview

All quantization implementations are centralized in the `python/sglang/srt/layers/quantization/` directory, connecting model construction, weight loading, and inference execution via unified abstractions and lifecycle hooks. The core goals are:

- Select quantization schemes in a configuration-driven manner, compatible with common formats like HuggingFace;
- Provide reusable quantization interfaces for linear layers, MoE layers, KV Cache, and other components;
- Support multiple hardware backends (CUDA, ROCm, etc.) with clear fallback strategies when dependencies are missing;
- Enable new quantization schemes to integrate into the entire inference stack with minimal interface implementation.

The core base classes for quantization are defined in `base_config.py`:

- `QuantizeMethodBase`: Declares weight registration (`create_weights`), forward execution (`apply`), and post-loading processing (`process_weights_after_loading`).
- `LinearMethodBase` / `FusedMoEMethodBase`: Constrain quantization methods for standard linear layers and fused MoE layers respectively, providing additional context information such as expert routing and parallelism.
- `QuantizationConfig`: Responsible for parsing quantization configurations, validating hardware capabilities and activation dtypes, and returning the correct quantization method instances layer by layer.

The quantization method registry is located in `__init__.py`, mapping native schemes (AWQ, GPTQ, FP8, W4AFp8, ModelOpt, etc.) to string identifiers, and injecting additional configurations based on CUDA/HIP capabilities when necessary (e.g., `quark` configuration under ROCm).

## 2. Directory Structure

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

Other quantization-related modules:

- `python/sglang/srt/configs/model_config.py`: Parses HuggingFace configurations and `hf_quant_config.json`, and normalizes quantization scheme names in `_parse_quant_hf_config`.
- `python/sglang/srt/model_loader/weight_utils.py`: Instantiates `QuantizationConfig` based on scheme names, downloads or reads external quantization configuration files, and performs hardware capability validation.
- `python/sglang/srt/model_loader/loader.py`: Injects quantization configuration into the model during the `_initialize_model` phase, and calls the `process_weights_after_loading` hook for each layer after loading checkpoints.
- `python/sglang/srt/layers/linear.py`: Common base class for all linear layers, responsible for storing `quant_method`, registering quantization weights, and uniformly dispatching `apply` during forward pass.

## 3. Overall Workflow

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
    Call quant_method.process_weights_after_loading layer by layer
      ↓
During inference: LinearBase.forward → quant_method.apply executes quantized GEMM
```

### 3.1 Configuration Parsing

1. `ModelConfig._parse_quant_hf_config` reads `config.json`, `hf_quant_config.json` from the model directory, or explicit user parameters, and uniformly generates quantization method identifiers (e.g., `w4afp8`, `awq`).
2. `loader._get_quantization_config` calls `weight_utils.get_quant_config`, which obtains the configuration class (e.g., `W4AFp8Config`) via the `quantization.get_quantization_config` function based on the identifier, then calls `from_config` to instantiate the configuration object, filling in hardware requirements, activation dtypes, skip layer lists, additional configuration file names, etc.
3. `QuantizationConfig.from_config` parses the raw configuration dictionary into an object, preserving all context for subsequent decisions.

### 3.2 Model Construction

1. The model loader passes `quant_config` to the model constructor in `_initialize_model`.
2. Each module inheriting from `LinearBase` or `FusedMoE` calls `quant_config.get_quant_method(self, prefix)` in `__init__`:
   - If the current layer needs quantization, returns a specific quantization method instance (e.g., `Fp8LinearMethod`, `W4AFp8MoEMethod`).
   - If in the skip list, falls back to `UnquantizedLinearMethod`.
3. The quantization method's `create_weights` registers weight placeholders (quantized weights, scale, zero point, input scaling, etc.) at this stage, and binds custom `weight_loader` if needed.

### 3.3 Weight Loading and Post-processing

1. `DefaultModelLoader.load_weights_and_postprocess` loads checkpoint weights layer by layer, filling the previously registered tensors.
2. After loading, iterates through the model again, calling `process_weights_after_loading` for modules with `quant_method`:
   - Repack or transpose weights to match kernel-required layouts (e.g., Marlin, CUTLASS, block quantization formats);
   - Convert scale/zero point dtypes to reduce forward overhead;
   - Pre-compute constants like stride, group size, expert mappings, and store them in module cache.

### 3.4 Inference Execution

During forward propagation, the linear layer's `forward` method uniformly calls `self.quant_method.apply(self, x, bias)`, while MoE layers call `FusedMoEMethodBase.apply` derived implementations. These functions internally handle:

1. Quantizing input activations or reading pre-quantized tensors;
2. Calling underlying CUDA/ROCm/Triton kernels to execute quantized matrix multiplication or MoE routing;
3. Dequantizing outputs, adding bias, and returning to the upper network.

## 4. Unified Structure of Quantization Methods and W4AFp8 Example

Regardless of FP8, INT8, INT4, or third-party formats, all quantization methods follow a consistent structure:

1. **Inherit Abstract Base Classes**: Linear layer methods inherit `LinearMethodBase`, MoE layer methods inherit `FusedMoEMethodBase`, both implementing the interfaces specified by `QuantizeMethodBase`.
2. **Implement Three Core Methods**:
   - `create_weights`: Register quantized weights, scale, zero point tensors, and optionally define weight loaders;
   - `apply`: Execute forward quantization operations (including activation quantization, kernel calls, dequantization);
   - `process_weights_after_loading` (optional): Perform re-quantization, packing, transposition, etc. after weights are loaded.
3. **Configuration-Driven**: Each scheme provides a `QuantizationConfig` subclass, implementing interfaces like `from_config`, `get_quant_method`, `get_supported_act_dtypes`, `get_min_capability` for parsing external configurations and checking hardware compatibility.
4. **Unified Lifecycle**: Uniformly call `create_weights` during model initialization, execute `process_weights_after_loading` during weight loading, and call `apply` during inference. New schemes only need to implement these hooks to integrate into the entire pipeline.

**The following uses W4AFp8 as an example to demonstrate the complete workflow of configuration parsing, weight registration, post-processing, and inference.**

### 4.1 Configuration Identification and Object Construction

- When `quant_algo == "MIXED_PRECISION"` in `hf_quant_config.json`, `ModelConfig` maps the quantization scheme to `w4afp8` and validates hardware compatibility:
  ```python
  # ModelConfig._parse_modelopt_quant_config
  if quant_algo == "MIXED_PRECISION":
      return {"quant_method": "w4afp8"}
  ```
- `weight_utils.get_quant_config` internally obtains the configuration class (e.g., `W4AFp8Config`) via the `quantization.get_quantization_config` function, using the quantization method identifier (e.g., `"w4afp8"`), then calls `from_config` to instantiate the configuration object; if the model directory provides additional JSON files, they will be loaded in `from_config`, supplementing skip layers, expert mappings, etc.
- Configuration instances contain activation scaling schemes (dynamic or static) for linear/MoE layers, weight quantization group size, whether to enable quantization formats built into checkpoints, etc.

### 4.2 Model Construction and Method Injection

- `_initialize_model` passes `quant_config` to models like `DeepseekV2ForCausalLM`, then propagates it layer by layer to decoder layers, MoE modules, and linear layers.
- `LinearBase` layers call `quant_config.get_quant_method`: For the W4AFp8 scheme, regular linear layers receive `Fp8LinearMethod` (using FP8 activations), while MoE layers receive `W4AFp8MoEMethod` (using INT4 weights + FP8 activations); layers in the skip list use unquantized implementations:
  ```python
  # LinearBase.__init__
  if quant_config is None:
      self.quant_method = UnquantizedLinearMethod()
  else:
      self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
  ```
- `create_weights` registers packed INT4/INT8 weight tensors, scale (scaling factor), input activation scaling, expert weight indices, and other parameters, and binds custom loading logic for checkpoint loaders.

### 4.3 Weight Loading and Post-processing

- Checkpoint loading phase fills quantized weights and scale.
- `W4AFp8MoEMethod.process_weights_after_loading` will:
  - Convert weight scale to bfloat16 and interleave according to the order required by CUTLASS kernels:
    ```python
    # Convert and interleave weight scale
    w13_weight_scale = layer.w13_weight_scale_inv.to(torch.bfloat16)
    w13_weight_scale = interleave_scales(w13_weight_scale)
    layer.w13_weight_scale_inv = Parameter(w13_weight_scale, requires_grad=False)
    ```
  - Aggregate input scale, cache maximum value to reduce per-element operations during inference:
    ```python
    # Calculate and cache maximum input scale
    w13_input_scale_max = layer.w13_input_scale.max().to(torch.bfloat16).item()
    new_w13_input_scale = torch.tensor([w13_input_scale_max], dtype=torch.bfloat16, device=device)
    layer.w13_input_scale = Parameter(new_w13_input_scale, requires_grad=False)
    ```
  - Prepare constants like stride, expert routing parameters, etc., to avoid repeated construction in forward pass.
- `Fp8LinearMethod` also selects data layouts required by Marlin, block quantization, or CUTLASS kernels at this stage based on hardware.

### 4.4 Quantized Inference

- Linear layers execute `Fp8LinearMethod.apply`: Quantize input, call selected kernel for FP8 GEMM, dequantize output:
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
- MoE layers execute `W4AFp8MoEMethod.apply`: Calls `cutlass_w4a8_moe`, passing quantized weights, scales, expert routing results, and input scaling factors, completing mixed-precision computation with INT4 weights + FP8 activations, and finally fusing routing scaling factors before returning:
  ```python
  # W4AFp8MoEMethod.apply
  output = cutlass_w4a8_moe(
      x,
      layer.w13_weight,                                   # INT4 quantized weights
      layer.w2_weight,                                    # INT4 quantized weights
      layer.w13_weight_scale_inv,                         # Weight scale
      layer.w2_weight_scale_inv,                          # Weight scale
      topk_weights,                                      # Expert routing weights
      topk_ids,                                          # Expert routing IDs
      self.a_strides1, self.b_strides1, self.c_strides1,  # GEMM1 strides
      self.a_strides2, self.b_strides2, self.c_strides2,  # GEMM2 strides
      self.s_strides13, self.s_strides2,                  # Scale strides
      self.expert_offsets,                               # Expert offsets
      self.problem_sizes1, self.problem_sizes2,          # Problem sizes
      layer.w13_input_scale,                             # Input scale
      layer.w2_input_scale,                              # Input scale
  )
  # Fuse routing scaling factor
  if self.moe_runner_config.routed_scaling_factor is not None:
      output *= self.moe_runner_config.routed_scaling_factor
  return StandardCombineInput(hidden_states=output)
  ```

## 5. Supported Quantization Schemes Overview

| Category              | Representative Configurations                                                                 | Description                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| FP8 Series            | `fp8`, `w8a8_fp8`, `modelopt_fp8`, `fbgemm_fp8`                                               | Native FP8, W8A8-FP8 hybrid, ModelOpt/FBGEMM extensions                                  |
| INT8 Series           | `w8a8_int8`, `blockwise_int8`                                                                 | Classic 8bit weight/activation, blockwise INT8                                           |
| INT4/Mixed Precision  | `w4afp8`, `qoq`, `moe_wna16`                                                                  | 4bit weight + FP8 activation, QoQ, WNA16 (W4A16/W8A16)                                  |
| FP4 / MXFP4           | `modelopt_fp4`, `petit_nvfp4`, `mxfp4`, `quark`                                               | FP4 / MXFP4 schemes, `quark` is ROCm-specific                                           |
| Pre-quantized Formats | `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `gguf`, `compressed-tensors`, `auto-round`, `modelopt` | Integration with external toolchains or compressed tensor frameworks, `modelopt` auto-detects FP8/FP4 |
| KV Cache Quantization | `BaseKVCacheMethod` and its subclasses in `kv_cache.py`                                       | Provides scale and zero-point management for attention caches                           |

## 6. Steps to Extend New Quantization Schemes

1. **Implement Configuration Class**: Inherit from `QuantizationConfig`, parse custom parameters and implement `get_quant_method`.
2. **Implement Quantization Method Class**: Inherit from `LinearMethodBase` / `FusedMoEMethodBase`, implement `create_weights`, `apply`, and optionally `process_weights_after_loading`.
3. **Register Scheme**: Register string identifier and configuration class mapping in `BASE_QUANTIZATION_METHODS` in `__init__.py`, and add dependency detection logic if necessary.
4. **Integrate Utilities**: If new kernels or utility functions are needed, implement them in `utils.py`, `int8_utils.py`, or separate files, and call them in `apply`.
5. **Write Tests and Examples**: Ensure unit tests or inference scripts pass on target hardware, validating that configuration parsing and lifecycle hooks work as expected.

## 7. Summary

The quantization system uses a "configuration-driven + unified abstraction" architecture that clearly separates responsibilities for configuration parsing, method selection, weight registration, and inference execution:

- Model configuration stage determines quantization scheme and instantiates `QuantizationConfig`;
- Model construction stage injects quantization methods layer by layer and registers required parameters;
- Weight loading stage completes format conversion and data rearrangement;
- Inference stage uniformly dispatches underlying kernels through the quantization method's `apply`.

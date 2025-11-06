# 量化(Quantization)代码架构

## 1. 概述
所有量化实现集中在 `python/sglang/srt/layers/quantization/` 目录，通过统一的抽象和生命周期钩子把模型构建、权重加载、推理执行串联起来。核心目标是：

- 以配置驱动的方式选择量化方案，兼容 HuggingFace 等常见格式；
- 为线性层、MoE 层、KV Cache 等组件提供可复用的量化接口；
- 支持 CUDA、ROCm 等多硬件后端，并在缺失依赖时给出清晰回退策略；
- 让新量化方案通过最少的接口实现即可融入整个推理栈。

量化相关的核心基类定义在 `base_config.py`：

- `QuantizeMethodBase`：声明权重注册（`create_weights`）、前向执行（`apply`）以及权重落盘后的处理（`process_weights_after_loading`）。
- `LinearMethodBase` / `FusedMoEMethodBase`：分别约束标准线性层和融合 MoE 层的量化方法，补充专家路由、并行度等上下文信息。
- `QuantizationConfig`：负责解析量化配置、校验硬件能力与激活 dtype，并按层返回正确的量化方法实例。

量化方法注册表位于 `__init__.py`，将原生方案（AWQ、GPTQ、FP8、W4AFp8、ModelOpt 等）映射到字符串标识，必要时按 CUDA/HIP 能力注入额外配置（如 ROCm 下的 `quark` 配置）。

## 2. 目录结构与文件职责
```
python/sglang/srt/layers/quantization/
├── base_config.py                 # 抽象基类与公共工具
├── __init__.py                    # 量化方法注册、依赖检测
├── unquant.py                     # 未量化方法的实现（UnquantizedLinearMethod）
├── utils.py / int8_utils.py       # 层筛选、动态覆盖、INT8 工具
├── kv_cache.py                    # KV Cache 量化方法基类
├── fp8.py / fp8_kernel.py / fp8_utils.py    # FP8 线性 + MoE 实现
├── w8a8_int8.py / w8a8_fp8.py     # INT8 / FP8 组合方案
├── w4afp8.py                      # 权重 4bit + 激活 FP8 混合精度
├── awq.py / gptq.py / gguf.py     # 预量化格式导入
├── modelopt_quant.py              # ModelOpt FP8/FP4 配置
├── mxfp4.py / mxfp4_tensor.py     # MXFP4 实现
├── petit.py / petit_utils.py      # Petit NVFP4 实现
├── marlin_utils.py / marlin_utils_fp8.py  # Marlin 内核工具
├── int8_kernel.py                 # INT8 内核实现
├── kvfp4_tensor.py                # KV Cache FP4 张量
├── compressed_tensors/            # 通用压缩张量框架
└── quark/                         # ROCm 专用 MXFP4 / INT4
```

其他与量化强相关的模块：

- `python/sglang/srt/configs/model_config.py`：解析 HuggingFace 配置、`hf_quant_config.json`，并在 `_parse_quant_hf_config` 中规范化量化方案名称。
- `python/sglang/srt/model_loader/weight_utils.py`：根据方案名称实例化 `QuantizationConfig`，下载或读取外部量化配置文件，并执行硬件能力校验。
- `python/sglang/srt/model_loader/loader.py`：在 `_initialize_model` 阶段把量化配置注入模型，加载 checkpoint 后调用每层的 `process_weights_after_loading` 钩子。
- `python/sglang/srt/layers/linear.py`：所有线性层的公共基类，负责保存 `quant_method`，注册量化权重，并在前向过程中统一调度 `apply`。

## 3. 整体流程

```
ModelConfig._parse_quant_hf_config → 判定 quant_method（如 w4afp8）
      ↓
weight_utils.get_quant_config → 构造对应 QuantizationConfig 实例
      ↓
_initialize_model(...) → 把 quant_config 传入模型/各层
      ↓
LinearBase.quant_method.create_weights → 注册量化权重占位
      ↓
DefaultModelLoader.load_weights_and_postprocess →
    逐层调用 quant_method.process_weights_after_loading
      ↓
推理时 LinearBase.forward → quant_method.apply 执行量化 GEMM
```

### 3.1 配置解析
1. `ModelConfig._parse_quant_hf_config` 读取模型目录中的 `config.json`、`hf_quant_config.json` 或用户显式参数，统一生成量化方法标识（例如 `w4afp8`、`awq`）。
2. `loader._get_quantization_config` 调用 `weight_utils.get_quant_config`，根据标识通过 `get_quantization_config` 获取配置类（如 `W4AFp8Config`），然后调用 `from_config` 实例化配置对象，填充硬件需求、激活 dtype、跳层列表、额外的配置文件名等信息。
3. `QuantizationConfig.from_config` 将原始配置字典解析为对象，保留用于后续判断的全部上下文。

### 3.2 模型构造
1. 模型加载器在 `_initialize_model` 中将 `quant_config` 传入模型构造函数。
2. 每个继承自 `LinearBase` 或 `FusedMoE` 的模块在 `__init__` 内调用 `quant_config.get_quant_method(self, prefix)`：
   - 若当前层需要量化，返回具体的量化方法实例（如 `Fp8LinearMethod`、`W4AFp8MoEMethod`）。
   - 若在跳过列表中，则回退到 `UnquantizedLinearMethod`。
3. 量化方法的 `create_weights` 在此阶段注册权重占位符（量化权重、scale、zero point、输入缩放等），并根据需要绑定自定义的 `weight_loader`。

### 3.3 权重加载与后处理
1. `DefaultModelLoader.load_weights_and_postprocess` 逐层加载 checkpoint 权重，填充前面注册的张量。
2. 加载完成后再次遍历模型，对拥有 `quant_method` 的模块调用 `process_weights_after_loading`：
   - 重新打包或转置权重以匹配内核需要的布局（如 Marlin、CUTLASS、块量化格式）；
   - 转换 scale/zero point 的 dtype 以减少前向开销；
   - 预先计算 stride、group size、专家映射等常量，放入 module 缓存。

### 3.4 推理执行
前向传播时，线性层的 `forward` 方法统一调用 `self.quant_method.apply(self, x, bias)`，MoE 层则调用 `FusedMoEMethodBase.apply` 派生实现。该函数内部负责：

1. 对输入激活进行量化或读取预量化张量；
2. 调用底层 CUDA/ROCm/Triton 内核执行量化矩阵乘或 MoE 路由；
3. 对输出执行反量化、加和 bias，并返回给上层网络。

## 4. 量化方法的共性
无论 FP8、INT8、INT4 还是第三方格式，所有量化方法都遵循一致的结构：

1. **继承抽象基类**：线性层方法继承 `LinearMethodBase`，MoE 层方法继承 `FusedMoEMethodBase`，均实现 `QuantizeMethodBase` 规定的接口。
2. **实现三个核心方法**：
   - `create_weights`：注册量化权重、scale、zero point 等张量，并可定义权重加载器；
   - `apply`：执行前向量化运算（包含激活量化、内核调用、反量化）；
   - `process_weights_after_loading`（可选）：在权重落地后执行再量化、打包、转置等操作。
3. **配置驱动**：每个方案提供 `QuantizationConfig` 子类，实现 `from_config`、`get_quant_method`、`get_supported_act_dtypes`、`get_min_capability` 等接口，用于解析外部配置并检查硬件兼容性。
4. **统一生命周期**：模型初始化阶段统一调用 `create_weights`，权重加载阶段统一执行 `process_weights_after_loading`，推理阶段统一调用 `apply`。新增方案只需实现这些钩子即可融入整个管线。

**下面以W4AFp8为例，展示配置解析、权重注册、后处理和推理的完整链路。**

### 4.1 配置识别与对象构造
- 当 `hf_quant_config.json` 中 `quant_algo == "MIXED_PRECISION"` 时，`ModelConfig` 会把量化方案映射为 `w4afp8` 并校验硬件兼容性。
- `weight_utils.get_quant_config` 通过 `get_quantization_config` 获取 `W4AFp8Config` 类，然后调用 `from_config` 实例化配置对象；若模型目录提供额外 JSON，会在 `from_config` 中加载，补充跳层、专家映射等信息。
- 配置实例记录线性层/ MoE 层的激活缩放方案（动态或静态）、权重量化 group size、是否启用 checkpoint 内自带的量化格式等。

### 4.2 模型构造与方法注入
- `_initialize_model` 将 `quant_config` 传递到如 `DeepseekV2ForCausalLM` 等模型，再逐层下沉到解码层、MoE 模块、线性层。
- `LinearBase` 层调用 `quant_config.get_quant_method`：普通线性层获得 `Fp8LinearMethod`，MoE 层获得 `W4AFp8MoEMethod`；跳过列表中的层使用未量化实现。
- `create_weights` 注册打包后的 INT4/INT8 权重张量、scale（逆尺度）、输入激活缩放、专家权重索引等参数，并为 checkpoint loader 绑定定制加载逻辑。

### 4.3 权重加载与后处理
- Checkpoint 加载阶段填充量化权重和 scale。
- `W4AFp8MoEMethod.process_weights_after_loading` 会：
  - 将权重 scale 转换为 bfloat16 并按 CUTLASS 内核需要的顺序交错排列；
  - 归约输入 scale，缓存最大值以减少推理期的逐元素运算；
  - 准备 stride、专家路由参数等常量，避免在前向中重复构建。
- `Fp8LinearMethod` 也会在此阶段根据硬件选择 Marlin、块量化或 CUTLASS 内核所需的数据布局。

### 4.4 量化推理
- 线性层执行 `Fp8LinearMethod.apply`：量化输入、调用选定内核进行 FP8 GEMM、再度量化输出。
- MoE 层执行 `W4AFp8MoEMethod.apply`：调用 `cutlass_w4a8_moe`，传入量化权重、scale、专家路由结果和输入缩放，完成 INT4 权重 + FP8 激活的混合精度计算，最后融合路由缩放系数返回。

## 5. 已经支持的量化方案概览
| 分类             | 代表配置                                                                 | 说明                                               |
| ---------------- | -------------------------------------------------------------------------- | -------------------------------------------------- |
| FP8 系列         | `fp8`, `w8a8_fp8`, `modelopt_fp8`, `fbgemm_fp8`                            | 原生 FP8、W8A8-FP8 混合、ModelOpt/FBGEMM 扩展      |
| INT8 系列        | `w8a8_int8`, `blockwise_int8`                                              | 经典 8bit 权重/激活、块级 INT8                     |
| INT4/混合精度    | `w4afp8`, `qoq`, `moe_wna16`                                               | 4bit 权重 + FP8 激活、QoQ、自研 WNA16              |
| FP4 / MXFP4      | `modelopt_fp4`, `petit_nvfp4`, `mxfp4`, `quark`                            | FP4 / MXFP4 方案，`quark` 为 ROCm 专用             |
| 预量化格式导入   | `awq`, `awq_marlin`, `gptq`, `gptq_marlin`, `gguf`, `compressed-tensors`, `auto-round`, `modelopt` | 与外部工具链或压缩张量框架对接，`modelopt` 可自动检测 FP8/FP4 |
| KV Cache 量化    | `kv_cache.py` 中的 `BaseKVCacheMethod` 及其子类                            | 为注意力缓存提供 scale、零点管理                   |

## 6. 扩展新量化方案的步骤
1. **实现配置类**：继承 `QuantizationConfig`，解析自定义参数并实现 `get_quant_method`。
2. **实现量化方法类**：继承 `LinearMethodBase` / `FusedMoEMethodBase`，实现 `create_weights`、`apply`，必要时实现 `process_weights_after_loading`。
3. **注册方案**：在 `__init__.py` 的 `BASE_QUANTIZATION_METHODS` 中登记字符串标识与配置类映射，必要时添加依赖检测逻辑。
4. **集成工具**：若需要新的 kernel 或工具函数，可在 `utils.py`、`int8_utils.py` 或独立文件中实现，并在 `apply` 中调用。
5. **编写测试与示例**：确保在目标硬件上通过单元测试或推理脚本，验证配置解析与生命周期钩子是否按预期运行。

## 7. 总结
量化系统通过“配置驱动 + 抽象统一”的架构，清晰划分了配置解析、方法选择、权重注册、推理执行等职责：

- 模型配置阶段确定量化方案并实例化 `QuantizationConfig`；
- 模型构造阶段按层注入量化方法并注册所需参数；
- 权重加载阶段完成格式转换、数据重排；
- 推理阶段统一通过量化方法的 `apply` 调度底层内核。

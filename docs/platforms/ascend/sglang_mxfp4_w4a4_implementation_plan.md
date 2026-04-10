# SGLang Ascend 新增 MXFP4 W4A4 的详细实现方案

## 1. 目标

在 **不改变现有 ModelSlim 接入骨架** 的前提下，为 sglang 的 Ascend NPU 路径新增 `W4A4_MXFP4` 量化支持，覆盖：

- Linear
- Fused MoE
- `quant_model_description.json` 自动识别后的按层量化分派
- 权重加载后的 NPU 布局转换

目标形态要尽量与当前 `W8A8_MXFP8` 保持一致，但数据布局和算子参数遵循 MXFP4 的真实要求。

## 2. 设计原则

### 2.1 直接复用现有 ModelSlim 骨架

推荐沿用当前 MXFP8 的四层结构：

1. `ModelConfig / loader`
2. `ModelSlimConfig / scheme`
3. `NPU kernel method`
4. `torch.ops.npu / torch_npu` 算子调用

原因是这条路径已经具备：

- 自动识别 `quant_model_description.json`
- 按层混合精度分派
- 统一的 `load_weights -> postprocess` 生命周期
- Linear 与 MoE 的现成挂载点

### 2.2 数据布局必须对齐 MXFP4 packed checkpoint

推荐严格对齐 PR 7877 的布局，而不是沿用当前 `W4A4_DYNAMIC` 的 int8 存储方式。

原因：

- 用户已明确说明 4 比特权重是两两打包成 `uint8` 存储
- upstream vllm-ascend PR 也是按 packed `uint8` 设计
- 如果 sglang 继续用 `int8` 展开存储，就需要额外解包步骤，既不优雅，也会带来加载内存放大

因此本方案的核心前提是：

- Linear 权重物理 shape 为 `[N, K/2]`
- MoE 权重物理 shape 为 `[E, N, K/2]`
- scale 仍按原始 K 维的 group size 组织

### 2.3 先闭环单机推理链路，不扩展通信量化

sglang 当前 Ascend MoE 路径与 vllm-ascend 的 MC2/token dispatcher runtime 结构不同，因此推荐第一阶段只完成：

- 本地权重加载
- postprocess
- Linear matmul
- MoE grouped matmul

不额外引入：

- dispatch 阶段量化通信
- DeepEP 的 MXFP4 特化
- 与当前仓库无对应基础设施的 runtime 分支

## 3. 目标支持的 checkpoint 语义

新增方案的目标 checkpoint 约定如下：

### 3.1 量化类型字符串

`quant_model_description.json` 中按层使用：

- Linear：`W4A4_MXFP4`
- MoE：`W4A4_MXFP4`

### 3.2 线性层权重与 scale

- `weight`
  - 逻辑矩阵：`[N, K]`
  - 物理存储：`[N, K/2]`
  - dtype：`uint8`
- `weight_scale`
  - 物理存储：`[N, K/group_size]`
  - dtype：`uint8`

### 3.3 MoE 权重与 scale

- `w13_weight`
  - 逻辑矩阵：`[E, 2I, H]`
  - 物理存储：`[E, 2I, H/2]`
  - dtype：`uint8`
- `w2_weight`
  - 逻辑矩阵：`[E, H, I]`
  - 物理存储：`[E, H, I/2]`
  - dtype：`uint8`
- `w13_weight_scale`
  - 物理存储：`[E, 2I, H/group_size]`
  - dtype：`uint8`
- `w2_weight_scale`
  - 物理存储：`[E, H, I/group_size]`
  - dtype：`uint8`

默认 `group_size = 32`。

## 4. 代码层面的实现蓝图

下面按 sglang 当前代码结构，给出建议的落点和修改内容。

## 4.1 配置分派层

### 文件

- `sglang/python/sglang/srt/layers/quantization/modelslim/modelslim.py`
- `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/__init__.py`

### 需要做的事

#### 4.1.1 在线性层映射中增加新分支

在 `ModelSlimConfig._get_scheme_from_parts()` 中新增：

- `W4A4_MXFP4 -> ModelSlimW4A4MxFp4`

#### 4.1.2 在 MoE 映射中增加新分支

在 `get_moe_scheme()` 的 `moe_quant_schemes` 中新增：

- `("W4A4_MXFP4", ModelSlimW4A4MxFp4MoE)`

#### 4.1.3 导出新 scheme

在 `schemes/__init__.py` 中补充：

- `ModelSlimW4A4MxFp4`
- `ModelSlimW4A4MxFp4MoE`

### 这样做的价值

- 不改变上层自动识别机制
- 不影响现有 `W4A4_DYNAMIC`、`W8A8_DYNAMIC`、`W8A8_MXFP8`
- 完全复用 ModelSlim 逐层选择量化实现的方式

## 4.2 Linear scheme 层

### 新增文件

- `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_mxfp4.py`

### 建议类名

- `ModelSlimW4A4MxFp4`

### 主要职责

#### 4.2.1 create_weights()

按 packed 布局创建参数：

- `weight`
  - shape: `[N_local, K_local/2]`
  - dtype: `torch.uint8`
- `weight_scale`
  - shape: `[N_local, ceil(K_local/group_size)]`
  - dtype: `torch.uint8`

推荐与当前 `ModelSlimW8A8MxFp8` 保持相同风格：

- `weight` 用 `ModelWeightParameter`
- `weight_scale` 用 block scale 对应参数类

#### 4.2.2 process_weights_after_loading()

委托给新的 NPU kernel method：

- `NPUW4A4MxFp4LinearMethod.process_weights_after_loading(layer)`

#### 4.2.3 apply_weights()

委托给新的 NPU kernel method：

- `NPUW4A4MxFp4LinearMethod.apply(layer, x, bias)`

## 4.3 Linear kernel 层

### 文件

- `sglang/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`

### 新增类

- `NPUW4A4MxFp4LinearMethod`

### 建议的处理逻辑

#### 4.3.1 权重后处理

后处理建议完全对齐 PR 7877 的物理布局：

1. `layer.weight.data = layer.weight.data.transpose(-1, -2).contiguous()`
   - `[N, K/2] -> [K/2, N]`
2. `weight_scale = layer.weight_scale.data`
3. 校验 `weight_scale.shape[-1] % 2 == 0`
4. `weight_scale.reshape(N, K_group/2, 2)`
5. transpose 成 `[K_group/2, N, 2]`
6. 将结果回写到 `layer.weight_scale.data`

如果需要 NPU 指定格式，可参考 MXFP8 线性层决定是否补 `npu_format_cast`。这一点应在真实环境上验证：

- 如果 `npu_quant_matmul` 对 packed uint8 权重不需要额外 format cast，则不加
- 如果 ACLNN 要求特定 format，则补齐

#### 4.3.2 前向执行

建议调用链：

1. 输入 `x` reshape 为 `[-1, K]`
2. `torch_npu.npu_dynamic_mx_quant(...)`
   - `axis=1`
   - `dst_type=torch_npu.float4_e2m1fn_x2`
   - `block_size=group_size`
   - `round_mode` 采用 PR 的 `"round"`
3. 调 `torch_npu.npu_quant_matmul(...)`
   - `x1_dtype=torch_npu.float4_e2m1fn_x2`
   - `x2_dtype=torch_npu.float4_e2m1fn_x2`
   - `scale=layer.weight_scale`
   - `pertoken_scale=x_scale`
   - `scale_dtype=torch_npu.float8_e8m0fnu`
   - `pertoken_scale_dtype=torch_npu.float8_e8m0fnu`
   - `group_sizes=(1, 1, group_size)`
   - `output_dtype=x.dtype` 或 `bf16`

推荐默认输出 dtype 跟输入激活 dtype 对齐，和 vllm-ascend PR 保持一致。

#### 4.3.3 bias 处理

参考 PR：

- 若 bias 存在且不是 `float32`，先转 `float32`
- 再传入 `npu_quant_matmul`

这是为了避免 4bit 量化 matmul 下 bias dtype 约束导致的算子报错。

## 4.4 MoE scheme 层

### 新增文件

- `sglang/python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a4_mxfp4_moe.py`

### 建议类名

- `ModelSlimW4A4MxFp4MoE`

### 主要职责

#### 4.4.1 create_weights()

按 packed 布局注册：

- `w13_weight: [E, 2I, H/2], uint8`
- `w2_weight: [E, H, I/2], uint8`
- `w13_weight_scale: [E, 2I, ceil(H/group_size)], uint8`
- `w2_weight_scale: [E, H, ceil(I/group_size)], uint8`

`quant_method` 建议标成 block weight scale 类型，而不是 channel scale 类型，因为它本质上是 MX group scale，不是 `W4A4_DYNAMIC` 那种 per-channel int4 方案。

#### 4.4.2 process_weights_after_loading()

委托给新的 NPU MoE kernel method。

#### 4.4.3 apply_weights()

委托给新的 NPU MoE kernel method。

#### 4.4.4 apply_without_routing_weights()

建议第一阶段策略：

- 如果现有 `without_routing_weights` 路径在 Ascend/MoE 后端会被用到，则实现一个 MXFP4 版本
- 如果没有可靠验证条件，则先显式 `NotImplementedError`

推荐优先保守处理，而不是假设 DeepEP 或其他快路径天然兼容。

## 4.5 MoE kernel 层

### 文件

- `sglang/python/sglang/srt/hardware_backend/npu/quantization/fused_moe_method_npu.py`

### 新增类

- `NPUW4A4MxFp4DynamicMoEMethod`

### 推荐实现策略

建议不要把现有 `NPUW4A4Int4DynamicMoEMethod` 直接改成兼容两种格式，因为两者的 checkpoint 物理布局不同：

- 现有 `W4A4_DYNAMIC`：权重按 `int8` 存，后处理阶段再做 int4 pack
- 目标 `W4A4_MXFP4`：checkpoint 已经是 `uint8` packed，后处理只需转置与 scale reshape

最稳妥的做法是新增一条独立类与独立 helper。

### 4.5.1 新增 helper：mxfp4_gmm_npu()

建议新增一个与 `mxfp8_gmm_npu()` 平行的 helper：

输入：

- `input`
- `input_scale`
- `weight`
- `weight_scale`
- `group_list_type`
- `group_list`
- `output_dtype`

逻辑：

1. 若 `input_scale is None`，先调用 `npu_dynamic_mx_quant`
   - `dst_type=torch_npu.float4_e2m1fn_x2`
   - `block_size=group_size`
2. 调 `npu_grouped_matmul(...)`
   - `scale=[weight_scale]`
   - `per_token_scale=[x_scale]`
   - `scale_dtype=torch_npu.float8_e8m0fnu`
   - `per_token_scale_dtype=torch_npu.float8_e8m0fnu`
   - 若算子支持，补 `x_dtype/weight_dtype=torch_npu.float4_e2m1fn_x2`

### 4.5.2 新增 helper：npu_fused_experts_mxfp4()

建议仿照 `npu_fused_experts_fp8(..., is_mxfp8=True)` 的结构实现独立 helper：

1. `npu_moe_init_routing`
2. `npu_moe_compute_expert_tokens`
3. `mxfp4_gmm_npu()` 做 gate_up_proj
4. `npu_swiglu`
5. `mxfp4_gmm_npu()` 做 down_proj
6. `npu_moe_finalize_routing`

第一阶段不建议直接强行复用 `npu_grouped_matmul_swiglu_quant_v2` 快路径，理由是：

- 当前 sglang 的 MXFP8 快路径已经能工作
- 但 MXFP4 的底层 dtype、packed weight 格式、dispatch 位置与 MXFP8 并不完全一致
- 优先先打通标准 grouped matmul 版本更稳妥

### 4.5.3 MoE 权重后处理

建议按 PR 7877 布局：

1. `w13_weight.transpose(1, 2)`
   - `[E, 2I, H/2] -> [E, H/2, 2I]`
2. `w2_weight.transpose(1, 2)`
   - `[E, H, I/2] -> [E, I/2, H]`
3. `w13_weight_scale.reshape(E, 2I, H_group/2, 2).transpose(1, 2)`
4. `w2_weight_scale.reshape(E, H, I_group/2, 2).transpose(1, 2)`

最终目标布局：

- `w13_weight: [E, H/2, 2I]`
- `w2_weight: [E, I/2, H]`
- `w13_weight_scale: [E, H_group/2, 2I, 2]`
- `w2_weight_scale: [E, I_group/2, H, 2]`

## 4.6 权重加载层

### 当前判断

sglang 当前 ModelSlim 线性/MoE 权重加载的主逻辑是：

- `create_weights()` 决定参数 shape
- `weight_loader` / `default_weight_loader` 要求 loaded tensor shape 与参数 shape 对齐

这对 MXFP4 是有利的，因为：

- checkpoint 已经按 `uint8` packed 存
- 只要参数 shape 直接定义成 packed shape
- 就不需要在 load 阶段额外做解包或重打包

### 推荐补充的保护

在新 scheme 或新 kernel 的初始化阶段，建议加 shape 校验：

- `input_size % 2 == 0`
- `hidden_size % 2 == 0`
- `intermediate_size_per_partition % 2 == 0`
- `input_size % group_size == 0` 或接受 `ceil` 后明确 pad 规则
- `K_group % 2 == 0`，因为后处理会 reshape 成 `(..., 2)`

如果模型实际会出现奇数块数，需明确定义：

- checkpoint 导出时先 pad
- 或 postprocess 时显式报错

推荐第一版直接要求偶数块，避免隐式兼容导致误算。

## 4.7 文档与支持矩阵

建议在 `docs/platforms/ascend/ascend_npu_quantization.md` 后续实现完成后补充：

- `W4A4_MXFP4 | Linear`
- `W4A4_MXFP4 | MoE`

并明确标注支持硬件代际与当前限制。

## 5. 与现有 W4A4_DYNAMIC 的边界

必须把 `W4A4_MXFP4` 和现有 `W4A4_DYNAMIC` 清晰区分：

### 5.1 权重存储不同

- `W4A4_DYNAMIC`
  - 参数 shape 仍是完整 K 维
  - load 后再调用 `npu_convert_weight_to_int4pack`
- `W4A4_MXFP4`
  - checkpoint 已经 packed 到 `uint8`
  - 参数 shape 直接是 `K/2`
  - load 后不再走 `npu_convert_weight_to_int4pack`

### 5.2 scale 语义不同

- `W4A4_DYNAMIC`
  - 更像 per-channel int4 quant
- `W4A4_MXFP4`
  - 是 MX microscaling，scale 按 group_size 组织

### 5.3 算子入口不同

- `W4A4_DYNAMIC`
  - `npu_dynamic_quant(dst_type=torch.quint4x2)`
- `W4A4_MXFP4`
  - `npu_dynamic_mx_quant(dst_type=torch_npu.float4_e2m1fn_x2)`

这三点决定了二者必须是两条独立实现，而不是在一个类里通过 flag 混合。

## 6. 建议的开发顺序

推荐按下面的顺序实施，风险最低：

### 阶段一：框架接线

- 配置映射
- scheme 注册
- 参数 shape/dtype 创建

完成后可以先验证：

- 模型是否能正确识别 `W4A4_MXFP4`
- 参数名字和 shape 是否与 checkpoint 对齐

### 阶段二：Linear 打通

- 新增 `NPUW4A4MxFp4LinearMethod`
- 完成 postprocess
- 完成 `npu_dynamic_mx_quant + npu_quant_matmul`

先把 Linear 打通的价值很高，因为：

- shape 最简单
- 最容易验证 packed weight + scale 布局是否正确

### 阶段三：MoE 打通

- 新增 `ModelSlimW4A4MxFp4MoE`
- 新增 `NPUW4A4MxFp4DynamicMoEMethod`
- 新增 `mxfp4_gmm_npu` / `npu_fused_experts_mxfp4`

### 阶段四：完善快路径与限制项

- 评估是否需要 MXFP4 的 fused swiglu quant 快路径
- 评估 `apply_without_routing_weights`
- 评估 DeepEP、低延迟 EP、A5 特化路径

## 7. 测试方案

## 7.1 单元级测试

建议至少新增以下几类测试：

### 7.1.1 配置分派测试

输入伪造的 `quant_model_description.json`，验证：

- `W4A4_MXFP4` 线性层返回 `ModelSlimW4A4MxFp4`
- `W4A4_MXFP4` MoE 返回 `ModelSlimW4A4MxFp4MoE`

### 7.1.2 参数 shape 测试

验证 `create_weights()` 生成的参数 shape：

- Linear 的 `weight` 是 `K/2`
- MoE 的 `w13_weight` / `w2_weight` 最后一维都是半宽
- scale 维度按 `group_size` 正确计算

### 7.1.3 postprocess 测试

验证：

- Linear weight 从 `[N, K/2]` 变成 `[K/2, N]`
- Linear scale 从 `[N, K_group]` 变成 `[K_group/2, N, 2]`
- MoE weight/scale 转置后的 shape 与预期一致

### 7.1.4 算子参数测试

通过 mock 或 wrapper 验证：

- 线性层确实调用 `npu_dynamic_mx_quant`
- 传入的 `dst_type` 是 `float4_e2m1fn_x2`
- matmul 的 `group_sizes` 是 `(1,1,32)`
- `scale_dtype` 与 `pertoken_scale_dtype` 是 e8m0

## 7.2 集成测试

建议新增一个和现有 `test_npu_w4a4_quantization.py` 类似的 Ascend 注册测试，但模型应替换为真正的 MXFP4 W4A4 ModelSlim 导出模型。

最小验证目标：

- 服务能拉起
- 首次 decode 能成功
- 连续 decode 不报 shape/layout 错
- MoE 模型在 topk dispatch 后能成功跑通

## 7.3 回归关注点

必须重点关注：

- 不影响现有 `W4A4_DYNAMIC`
- 不影响现有 `W8A8_MXFP8`
- 不影响没有 `W4A4_MXFP4` 的 ModelSlim 模型

## 8. 风险清单

### 8.1 底层 dtype/算子可用性

风险：

- 当前运行环境未必完整支持 `torch_npu.float4_e2m1fn_x2`
- `npu_quant_matmul` / `npu_grouped_matmul` 对 packed uint8 权重的真实要求可能与 PR 假设略有差异

缓解：

- 在实现时先做 capability check
- 线性层先最小闭环验证

### 8.2 scale reshape 的偶数约束

风险：

- `K/group_size` 若为奇数，无法直接 reshape 成 `(..., 2)`

缓解：

- 第一版严格限制为偶数块
- 不满足时直接报错，避免 silent wrong result

### 8.3 MoE 快路径兼容性

风险：

- 直接照搬 MXFP8 的 fused fast path 可能与 MXFP4 的 dtype/dispatch 约束不兼容

缓解：

- 第一版只实现标准 grouped matmul 路径
- 待正确性验证后再补 fused 路径

### 8.4 checkpoint 命名差异

风险：

- 不同导出工具对 `weight_scale` 命名、expert 权重命名可能存在细微差异

缓解：

- 优先对齐你当前分支使用的 ModelSlim 导出命名规范
- 若有命名差异，再通过 mapper 或 weight loader 微调

## 9. 最终建议

综合当前 sglang 代码结构和 PR 7877 的做法，推荐的实现方向是：

- **框架接入**：完全复用当前 MXFP8 的 ModelSlim 接入方式
- **权重布局**：严格对齐 packed `uint8` 的 MXFP4 checkpoint 格式
- **Linear 算子**：直接采用 `npu_dynamic_mx_quant + npu_quant_matmul`
- **MoE 算子**：在 sglang 当前 NPU fused moe helper 上新增一条 MXFP4 分支
- **快路径策略**：先不做 dispatch 量化与 DeepEP 扩展，优先把标准推理闭环做正确

如果按这个方案实施，代码改动面虽然分散，但每一层都能沿用现有 MXFP8 的组织方式，因此整体风险是可控的，且后续维护成本也最低。

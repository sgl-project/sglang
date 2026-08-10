# DeepSeek V4 W4A8 MXFP GMM1-SwiGLU-Quant 融合设计

日期：2026-08-10
状态：已完成讨论，待书面确认

## 1. 背景与目标

DeepSeek V4 在 Ascend 标准 MoE 路由路径中，W4A8 MXFP MLP 的第一段当前依次执行：

1. `w4a4_mxfp_gmm_npu` 完成 GMM1；
2. `_apply_swiglu_limit_npu` 完成非对称截断；
3. `torch.ops.npu.npu_swiglu` 完成 SwiGLU；
4. GMM2 再次进行 MXFP 分组矩阵乘。

该流程会显式产生 GMM1 的中间结果，并启动多个算子。技术验证目标是复用 vLLM-Ascend 注册的
`torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2`，将 GMM1、SwiGLU limit、SwiGLU 和
GMM2 输入的 MXFP8 动态量化融合，减少中间张量读写和算子启动开销。

本阶段只覆盖：

- 标准 MoE 路由的 eager 路径；
- 标准 MoE 路由的图捕获/单 token decode 路径；
- DeepSeek V4 使用的 W4A8 MXFP 配置，即 FP4 权重、FP8 E4M3 激活和 E8M0 block scale。

本阶段不覆盖：

- DeepEP 路径；
- Ascend Tensor Parallel 专用路径；
- GMM2、路由、token 排序/还原和最终输出归并；
- INT8 量化语义或其他 MoE 量化配置；
- 在 SGLang 内自动安装、复制或构建 vLLM-Ascend 二进制。

## 2. 方案选择

采用“扩展 vLLM-Ascend 二进制契约 + SGLang 最小适配”的方案。

现有 `_C_ascend.grouped_matmul_swiglu_quant_v2` 的 Python/C++ 输出分配契约面向 W4A8
per-channel INT8：输出为 INT8，scale 为 FP32。直接拿它替换 DeepSeek V4 的 W4A8 MXFP
路径，会改变 GMM2 的输入类型和量化语义，因此不能只在 SGLang 侧替换调用。

本方案保留该算子现有 INT8 行为，并为 `quant_mode=2` 增加 MXFP 输出契约。vLLM-Ascend
二进制负责完整融合；SGLang 仅负责准备输入、调用算子、规范化 scale 布局并继续复用现有
GMM2。这样既满足完整 GMM1 融合目标，也把 SGLang 改动限制在两个指定执行路径内。

未采用的方案：

- 直接使用现有 INT8 输出：需要修改 GMM2 及其量化语义，超出本次验证范围；
- 仅复用 `npu_swiglu_group_quant`：仍保留独立 GMM1，不能验证完整融合收益；
- 在 SGLang 内实现或复制算子：会重复维护二进制实现，并扩大变更面。

## 3. 数据流与边界

开启实验开关后，标准路径的数据流为：

```text
hidden_states (BF16 或 FP8 E4M3)
  -> 必要时执行现有 dynamic_mx_quant
  -> grouped_matmul_swiglu_quant_v2
       [GMM1 + asymmetric limit + SwiGLU + MXFP8 quant]
  -> FP8 E4M3 激活 + E8M0 block-32 scale
  -> 现有 scale pair-pack
  -> 现有 W4A8 MXFP GMM2
  -> 现有 token 还原/加权/归并
```

融合算子不得产生对 SGLang 可见的 BF16 gate/up 中间张量。它必须保持当前数学语义：

- gate 分支只做上界截断：`gate = min(gate, swiglu_limit)`；
- up 分支做双边截断：`up = clamp(up, -swiglu_limit, swiglu_limit)`；
- 随后计算 `silu(gate) * up`；
- 当 `swiglu_limit` 未设置时传入 `0.0`，语义与当前无截断路径一致；
- 最后按 block size 32 量化为 FP8 E4M3，scale 使用 E8M0。

## 4. 融合算子契约

SGLang 通过导入 `vllm_ascend` 注册二进制算子。验证路径调用：

```python
output, output_scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(
    x=quantized_input,
    weight=[w13],
    weight_scale=[w13_weight_scale],
    x_scale=input_scale,
    group_list=expert_tokens,
    dequant_mode=2,
    quant_mode=2,
    group_list_type=group_list_type,
    swiglu_limit=float(swiglu_limit or 0.0),
)
```

MXFP 模式的输入输出契约如下：

| 项目 | 契约 |
| --- | --- |
| `x` | FP8 E4M3，逻辑形状 `[M, K]` |
| `x_scale` | E8M0，block size 32，使用现有输入 scale 布局 |
| `weight[0]` | FP4、NZ 布局的 GMM1 权重 |
| `weight_scale[0]` | E8M0 GMM1 权重 scale |
| `group_list` | 标准路由生成的 expert token 信息 |
| `group_list_type` | 同时支持当前标准路径使用的 0/1 两种表示 |
| `output` | FP8 E4M3，逻辑形状 `[M, I]` |
| `output_scale` | E8M0，原始语义形状 `[M, I / 32]` |

其中 `K` 为 hidden size，`I` 为 MoE intermediate size。对于 DeepSeek V4 的主配置，
`K=4096`、`I=2048`、expert 数为 256、top-k 为 6。

vLLM-Ascend 二进制侧的必要修改为：

1. 保留 `quant_mode` 非 MXFP 模式的原输出 dtype/shape，避免影响已有 INT8 调用；
2. 当 `quant_mode=2` 时，按上述契约分配 FP8 输出和 E8M0 scale；
3. 从逻辑权重/scale 布局推导 `I`，不能沿用只适配 per-channel 权重的末维推导；
4. 将该算子加入 Ascend 950/A5 的自定义算子构建清单；
5. 由生成的 vLLM-Ascend 包或共享库完成注册，SGLang 不携带该二进制。

为避免绑定层硬编码不稳定的 dtype 枚举，MXFP 输出优先继承 `x` 的 tensor options，scale
优先继承 `x_scale` 的 tensor options；同时由单算子测试验证实际 dtype 和 shape。

## 5. SGLang 改动

改动限定在 `python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py` 及其直接
单元测试中。

新增一个文件内私有 helper，职责仅包括：

1. 若输入仍为 BF16，复用现有 MXFP 动态量化逻辑得到 FP8 输入和 scale；
2. 调用 `_C_ascend.grouped_matmul_swiglu_quant_v2`；
3. 校验返回 dtype、rank 和关键 shape；
4. 使用现有 `_pair_pack_mxfp_act_scale` 将原始 `[M, I/32]` scale 转为 GMM2 已使用的
   `[M, I/64, 2]` 布局；
5. 返回 GMM2 可直接消费的激活和 scale。

只在以下两个位置用 helper 替换“GMM1 + limit + `npu_swiglu`”：

- `npu_fused_experts_w4a4_mxfp`；
- `npu_fused_experts_w4a4_mxfp_decode`。

以下代码保持不变：

- `npu_apply_without_routing_weights_w4a4_mxfp`；
- DeepEP 和 AscendTP 分支选择；
- GMM2 调用及其权重、scale；
- 路由、group list 生成、图缓存和输出归并。

## 6. 开关、初始化与失败策略

新增实验环境变量：

```text
SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION=true
```

默认关闭。关闭时：

- 不导入 `vllm_ascend`；
- 不查询自定义 op；
- 完整保留现有执行路径和行为。

开启时，在进入图捕获前完成 `vllm_ascend` 导入和 op 解析，避免捕获期间发生 Python 导入或
动态注册。验证阶段采用 fail-fast，不做静默 fallback。以下情况直接给出明确错误：

- `vllm_ascend` 无法导入；
- `_C_ascend.grouped_matmul_swiglu_quant_v2` 未注册；
- 当前配置不是目标 W4A8 MXFP 路径；
- 输出 dtype、shape 或 scale 布局不满足契约。

严格失败可以防止二进制版本不匹配时悄悄退回旧路径，导致性能数据失真。回退方式是关闭实验
环境变量并重启进程。

## 7. 验证方案

### 7.1 SGLang 单元测试

测试仅覆盖此次新增分支，并保留工作区已有测试内容：

- 开关关闭时仍调用原 GMM1、limit 和 `npu_swiglu`；
- 开关开启时，eager 标准路径只调用一次融合 op；
- 开关开启时，decode/图捕获路径只调用一次融合 op；
- 覆盖 `group_list_type=0` 和 `group_list_type=1`；
- 验证 GMM2 收到 FP8 输出和 pair-packed E8M0 scale；
- 验证导入失败、op 缺失和返回契约错误时 fail-fast；
- 验证 AscendTP 与 DeepEP 分支没有改走融合 helper。

### 7.2 vLLM-Ascend 单算子正确性

使用 DeepSeek V4 代表尺寸及可快速执行的小尺寸 case，对比旧链路：

```text
GMM1 -> asymmetric limit -> SwiGLU -> dynamic_mx_quant -> GMM2
```

与新链路：

```text
grouped_matmul_swiglu_quant_v2 -> GMM2
```

重点覆盖：

- 单 token decode；
- 多 token batch；
- 多个空 expert；
- 非均匀 expert token 分布；
- 图捕获和多次 replay；
- `swiglu_limit` 开启与关闭。

精度以最终 GMM2 的 BF16 输出为主要比较对象，技术验证门槛为：

- cosine similarity 不低于 `0.999`；
- normalized mean absolute error 不高于 `1%`；
- DeepSeek V4 模型级精度用例无相对基线回退。

### 7.3 性能验证

在相同输入、路由结果、预热次数和图模式下对比开关前后：

- profiler 中旧的独立 GMM1、limit、SwiGLU 和动态量化算子不再出现；
- 记录主 decode shape 的 warmup 后 P50 延迟；
- 融合路径的主 decode shape P50 低于旧路径；
- 图捕获与 replay 均成功，且无捕获后的动态注册或重新编译。

若算子融合成功但端到端性能没有改善，本阶段不继续扩展改动范围；先依据 profiler 判断瓶颈，
再单独评审下一步优化。

## 8. 构建与部署

1. 在与目标容器 CANN、torch_npu 和 Python ABI 匹配的 vLLM-Ascend 环境中构建 A5 自定义算子；
2. 将生成的 vLLM-Ascend wheel 或共享库部署到技术验证容器；
3. 在启动 SGLang 前确认 `import vllm_ascend` 可注册目标 op；
4. 先保持实验开关关闭完成基线，再开启开关运行精度和性能验证；
5. SGLang 仓库不提交生成的二进制文件。

## 9. 完成标准

本技术验证完成需同时满足：

- vLLM-Ascend 的 MXFP 模式输出契约通过单算子测试，且不破坏已有 INT8 行为；
- SGLang 只修改指定标准 eager/decode 路径，开关默认关闭；
- DeepEP、AscendTP、GMM2、路由和归并逻辑无行为变化；
- 单元测试和图捕获/replay 验证通过；
- 达到上述精度门槛；
- profiler 确认融合生效，主 decode shape 的 P50 延迟相对基线下降。

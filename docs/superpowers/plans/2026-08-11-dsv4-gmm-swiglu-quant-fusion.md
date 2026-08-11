# DeepSeek V4 W4A8 MXFP GMM1-SwiGLU-Quant Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 DeepSeek V4 的标准路由 eager 与图捕获 decode 路径中，使用 vLLM-Ascend 二进制的 `grouped_matmul_swiglu_quant_v2` 融合 GMM1、SwiGLU limit、SwiGLU 和 MXFP8 输出量化，并保持既有 GMM2。

**Architecture:** vLLM-Ascend 为同名自定义 op 的 `quant_mode=2` 增加 FP8 E4M3 / E8M0 MXFP 输出契约，保留已有 INT8 契约。SGLang 在实验开关启用时导入该二进制、把 routed activation 量化为 op 所需的 MXFP 输入、调用融合 op，并把其 scale 转为现有 GMM2 的 pair-packed 布局；关闭开关时完整沿用旧链路。

**Tech Stack:** Python、PyTorch PrivateUse1/NPU、torch_npu、vLLM-Ascend ACLNN 自定义 op、CANN Ascend 950/A5。

## Global Constraints

- 上游参考固定为 `D:\Github\vllm-ascend` 的提交 `6fadabbfb5e18c60aa328845b3145d91a8d2b955`；不要覆盖其中已修改的 `CLAUDE.md`。
- 只覆盖 `ROUTED_EXPERTS_FP8_ACTIVATION=True` 的 W4A8 MXFP 标准路由 eager 和图捕获 decode 路径。
- 不修改 DeepEP、AscendTP、`npu_apply_without_routing_weights_w4a4_mxfp`、GMM2、路由或输出归并。
- 实验开关为 `SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION=true`，默认关闭；开启后导入/算子/契约错误必须 fail-fast，禁止静默 fallback。
- 融合 op 的 MXFP 契约为：输入/输出 FP8 E4M3，输入/输出 scale 为 E8M0 block-32；原始输出 scale `[M, I/32]` 在 SGLang 中转换成 `[M, I/64, 2]`。
- `swiglu_limit` 语义保持现状：gate 仅上界截断，up 双边截断；未设置或不大于零时不截断。
- SGLang 不提交 vLLM-Ascend 生成的 wheel 或共享库；NPU 正确性必须在重建并安装对应 wheel 后才可声明。

---

## File Structure

- `D:\Github\vllm-ascend\csrc\gmm\grouped_matmul_swiglu_quant_v2\grouped_matmul_swiglu_quant_v2_torch_adpt.h`：为 `quant_mode=2` 分配 MXFP 输出和 E8M0 scale，原 INT8 分支不变。
- `D:\Github\vllm-ascend\csrc\torch_binding_meta.cpp`：使 Fake/Meta 输出 dtype 与 shape 跟随同一 `quant_mode=2` 契约。
- `D:\Github\vllm-ascend\tests\ut\ops\test_moe_mlp.py`：增加对调用参数和输出契约的回归保护；若已有 NPU 单算子测试目录，则在其中添加等价真实算子 case。
- `python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py`：实现私有融合 helper 和开关，并仅替换两个标准路径的 GMM1 后处理链。
- `test/registered/unit/npu/quantization/test_fp4_moe_methods.py`：验证关闭开关的旧链路、开启开关的 eager/decode 路径、scale 布局和 fail-fast 行为。

### Task 1: 固化 vLLM-Ascend MXFP 输出契约测试

**Files:**

- Modify: `D:\Github\vllm-ascend\tests\ut\ops\test_moe_mlp.py`
- Inspect: `D:\Github\vllm-ascend\csrc\torch_binding.cpp:2207-2212`
- Inspect: `D:\Github\vllm-ascend\csrc\torch_binding_meta.cpp:159-190`

**Interfaces:**

- Consumes: `_C_ascend.grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, ..., dequant_mode=2, quant_mode=2, group_list_type, ..., swiglu_limit)`。
- Produces: 对输出逻辑形状 `[M, I]`、FP8 E4M3 输出及 `[M, I/32]` E8M0 scale 的可执行断言。

- [ ] **Step 1: 写出失败的 MXFP 契约测试**

在现有 `test_moe_mlp.py` 的自定义 op mock 测试附近新增测试，替换 op 为记录参数并返回：

```python
expected_out = torch.empty((m, intermediate_size), dtype=torch.float8_e4m3fn)
expected_scale = torch.empty((m, intermediate_size // 32), dtype=torch.float8_e8m0fnu)
mock_fused.return_value = expected_out, expected_scale

result, scale = torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(...)
assert result.shape == (m, intermediate_size)
assert result.dtype is torch.float8_e4m3fn
assert scale.shape == (m, intermediate_size // 32)
assert scale.dtype is torch.float8_e8m0fnu
```

测试必须断言调用携带 `dequant_mode=2`、`quant_mode=2`、原始 `group_list_type` 和 `swiglu_limit`。

- [ ] **Step 2: 运行测试并确认当前绑定不满足契约**

Run: `cd D:\Github\vllm-ascend && pytest tests/ut/ops/test_moe_mlp.py -k grouped_matmul_swiglu_quant_v2 -v`

Expected: 失败，现有绑定固定分配 `torch.char` 输出 `[M, N/2]` 与 float scale `[M]`。

- [ ] **Step 3: 检查 A5 构建清单而不修改无关内容**

Run: `cd D:\Github\vllm-ascend && rg -n 'grouped_matmul_swiglu_quant_v2' csrc/build_aclnn.sh`

Expected: 算子已在 Ascend 950 选择列表中；仅记录该事实，不重复添加条目。

- [ ] **Step 4: 提交测试基线**

```bash
cd D:\Github\vllm-ascend
git add tests/ut/ops/test_moe_mlp.py
git commit -m "test: cover MXFP GMM SwiGLU quant contract"
```

### Task 2: 实现 vLLM-Ascend `quant_mode=2` 输出分配和 Meta 契约

**Files:**

- Modify: `D:\Github\vllm-ascend\csrc\gmm\grouped_matmul_swiglu_quant_v2\grouped_matmul_swiglu_quant_v2_torch_adpt.h:20-70`
- Modify: `D:\Github\vllm-ascend\csrc\torch_binding_meta.cpp:159-190`
- Test: `D:\Github\vllm-ascend\tests\ut\ops\test_moe_mlp.py`

**Interfaces:**

- Consumes: `x` 形状 `[M, K]`，`weight_scale[0]` 的 MXFP scale 布局以及 `quant_mode`。
- Produces: `quant_mode=2` 时 `(output: FP8 E4M3 [M, I], output_scale: E8M0 [M, I/32])`；其他模式维持当前 `(int8 [M, N/2], float [M])`。其中 GMM1 gate/up 逻辑宽度为 `2I`。

- [ ] **Step 1: 在适配器中分支计算逻辑输出维度和 dtype**

在读取 `m` 后先提取 `quant_mode_real`。仅当它等于 `2` 时，使用 MXFP 权重 scale 的倒数第二个逻辑维度作为 `I`，并按输入 tensor options 创建输出：

```cpp
const bool is_mxfp = quant_mode_real == 2;
const int64_t gate_up_width = weight_scale[0].sizes().at(-2);
const int64_t n = is_mxfp ? gate_up_width / 2
                          : weight_scale[0].sizes().back();
at::Tensor output = is_mxfp
    ? at::empty({m, n}, x.options())
    : at::empty({m, n / 2}, x.options().dtype(at::kChar));
at::Tensor output_scale = is_mxfp
    ? at::empty({m, n / 32}, x_scale.options())
    : at::empty({m}, x.options().dtype(at::kFloat));
```

保持 `aclnnGroupedMatmulSwigluQuantWeightNzV2` 的参数次序和异步行为不变；删除未使用的 `k`、`ws` 局部变量。

- [ ] **Step 2: 在 Meta 实现中镜像同一分支**

在 `grouped_matmul_swiglu_quant_v2_meta` 使用与适配器相同的 `quant_mode.value_or(0)`、`n` 推导和 `at::empty` 规则。Meta 输出不得调用真机算子，也不得硬编码 FP8/E8M0 enum；用 `x.options()` 与 `x_scale.options()` 继承 dtype/device。

- [ ] **Step 3: 运行窄测试并确认通过**

Run: `cd D:\Github\vllm-ascend && pytest tests/ut/ops/test_moe_mlp.py -k grouped_matmul_swiglu_quant_v2 -v`

Expected: PASS；MXFP 断言通过，现有 per-channel INT8 断言仍通过。

- [ ] **Step 4: 编译和安装目标二进制后执行真实 NPU case**

Run: `cd D:\Github\vllm-ascend && bash csrc/build_aclnn.sh "$PWD" ascend950`

Run: 使用该仓库既有 wheel 构建命令生成 wheel，并在验证容器执行 `pip install --force-reinstall <built-vllm-ascend-wheel>`。

Run: `cd D:\Github\vllm-ascend && pytest tests/ut/ops/test_moe_mlp.py -k grouped_matmul_swiglu_quant_v2 -v`

Expected: 真实 NPU 输出 dtype/shape 与契约一致。若本机没有 CANN/torch_npu/NPU，记录未运行命令和缺失条件，不能把 CPU mock 视作内核正确性。

- [ ] **Step 5: 提交二进制契约实现**

```bash
cd D:\Github\vllm-ascend
git add csrc/gmm/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_torch_adpt.h csrc/torch_binding_meta.cpp tests/ut/ops/test_moe_mlp.py
git commit -m "feat: support MXFP GMM SwiGLU quant output"
```

### Task 3: 为 SGLang 融合 helper 建立失败测试

**Files:**

- Modify: `test/registered/unit/npu/quantization/test_fp4_moe_methods.py`
- Inspect: `python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py:240-365,498-535`

**Interfaces:**

- Consumes: `SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION` 与 routed token、`w13`、`w13_weight_scale_inv`、`expert_tokens`、`group_list_type`、`swiglu_limit`。
- Produces: 私有 helper `_gmm_swiglu_quant_mxfp_npu(...) -> tuple[torch.Tensor, torch.Tensor]`，第二项为 pair-packed GMM2 input scale。

- [ ] **Step 1: 写出开关关闭的旧路径测试**

通过 monkeypatch 记录 `w4a4_mxfp_gmm_npu`、`_apply_swiglu_limit_npu` 与 `torch.ops.npu.npu_swiglu` 的调用；设置环境变量未设置，调用 eager 和 decode helper。断言 GMM1、limit、SwiGLU、GMM2 的调用顺序存在，且 `_C_ascend.grouped_matmul_swiglu_quant_v2` 未被访问。

- [ ] **Step 2: 写出开关开启的融合路径和 scale 测试**

为 `sys.modules["vllm_ascend"]` 提供空模块，并 mock：

```python
fused_out = torch.empty((m, intermediate_size), dtype=torch.float8_e4m3fn)
raw_scale = torch.empty((m, intermediate_size // 32), dtype=torch.float8_e8m0fnu)
mock_op.return_value = fused_out, raw_scale
```

设置开关后分别驱动 `group_list_type=0` 的 eager 与 `group_list_type=1` 的 decode；断言融合 op 调用一次、参数包含 `dequant_mode=2` 与 `quant_mode=2`、GMM2 收到 `fused_out` 和形状 `[m, intermediate_size // 64, 2]` 的 scale，且旧 limit/SwiGLU 不被调用。

- [ ] **Step 3: 写出 fail-fast 与排除分支测试**

分别让 `import vllm_ascend` 抛 `ImportError`、让 `_C_ascend.grouped_matmul_swiglu_quant_v2` 缺失、让 output scale 为 float 或形状 `[m]`；断言抛出的 `RuntimeError` 包含环境变量名或契约字段。对 `npu_apply_without_routing_weights_w4a4_mxfp` 断言不访问融合 op，保护 DeepEP/AscendTP 共用路径。

- [ ] **Step 4: 运行测试并确认因 helper 缺失失败**

Run: `pytest test/registered/unit/npu/quantization/test_fp4_moe_methods.py -v`

Expected: 融合分支测试失败，原因是环境开关和 `_gmm_swiglu_quant_mxfp_npu` 尚未实现；旧路径测试保持通过。

- [ ] **Step 5: 提交测试基线**

```bash
git add test/registered/unit/npu/quantization/test_fp4_moe_methods.py
git commit -m "test: cover DSV4 GMM SwiGLU quant fusion"
```

### Task 4: 实现 SGLang 标准 eager/decode 融合接入

**Files:**

- Modify: `python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py:240-365,498-535`
- Test: `test/registered/unit/npu/quantization/test_fp4_moe_methods.py`

**Interfaces:**

- Consumes: `_w4a8_mxfp_gmm` 使用的 `torch.ops.npu.npu_dynamic_mx_quant` 输出，以及 Task 2 提供的 `_C_ascend.grouped_matmul_swiglu_quant_v2` MXFP 契约。
- Produces: `_gmm_swiglu_quant_mxfp_npu(input, input_scale, weight, weight_scale, group_list_type, group_list, swiglu_limit) -> tuple[Tensor, Tensor]`。

- [ ] **Step 1: 增加默认关闭的模块级开关与 op 解析函数**

使用 `os.getenv("SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION", "").lower() in {"1", "true", "yes", "on"}` 判定开关。开关为真时，解析函数导入 `vllm_ascend` 并读取 `torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2`；导入或属性获取失败时抛出：

```python
raise RuntimeError(
    "SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION requires "
    "vllm_ascend and torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2"
) from exc
```

解析结果缓存于模块私有变量，首次标准 MoE 调用发生在图捕获前；关闭开关绝不导入 `vllm_ascend`。

- [ ] **Step 2: 实现最小融合 helper 与契约校验**

若 `input_scale is None`，复用现有动态 MXFP 量化路径获取 `(quantized_input, input_scale)`；否则直接使用现有 FP8/E8M0 输入。调用：

```python
output, output_scale = op(
    x=quantized_input,
    weight=[weight],
    weight_scale=[weight_scale],
    x_scale=input_scale,
    group_list=group_list.to(torch.int64),
    dequant_mode=2,
    quant_mode=2,
    group_list_type=group_list_type,
    swiglu_limit=float(swiglu_limit or 0.0),
)
```

校验 `output.dtype == torch.float8_e4m3fn`、`output.shape == (M, I)`、`output_scale.dtype == torch.float8_e8m0fnu`、`output_scale.shape == (M, I // 32)`；失败时 `RuntimeError` 写明实际和预期。成功时返回 `output, _pair_pack_mxfp_act_scale(output_scale)`。

- [ ] **Step 3: 仅替换 eager 标准路径的 GMM1 后处理链**

在 `npu_fused_experts_w4a4_mxfp` 中，当开关为真时，将：

```python
w4a4_mxfp_gmm_npu(... w13 ...) -> _apply_swiglu_limit_npu -> npu_swiglu
```

替换为 helper；把 helper 返回的 scale 作为 `input_scale` 传给第二个 `w4a4_mxfp_gmm_npu`。保持 `valid_mask_2d`、routing 和 finalize 代码逐字不动。

- [ ] **Step 4: 仅替换图捕获 decode 路径的同一链**

在 `npu_fused_experts_w4a4_mxfp_decode` 做同样替换，传入现有 `group_list_type=1` 和 `expert_tokens`。不得改动 `npu_moe_init_routing_v2`、`npu_moe_token_unpermute` 或其他函数。

- [ ] **Step 5: 运行 SGLang 窄测试并确认通过**

Run: `pytest test/registered/unit/npu/quantization/test_fp4_moe_methods.py -v`

Expected: PASS；开关关闭、eager、decode、fail-fast 及 DeepEP/AscendTP 排除断言均通过。

- [ ] **Step 6: 提交 SGLang 最小适配**

```bash
git add python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py test/registered/unit/npu/quantization/test_fp4_moe_methods.py
git commit -m "feat: fuse DSV4 GMM SwiGLU quant path"
```

### Task 5: 容器内端到端正确性、图捕获和性能验证

**Files:**

- Modify only if a failure exposes a contract defect: the files from Tasks 2 and 4.
- Do not add generated binary artifacts to either repository.

**Interfaces:**

- Consumes: 重建安装后的 vLLM-Ascend binary、`SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION`、DeepSeek V4 W4A8 MXFP 权重。
- Produces: 对旧/新路径最终 GMM2 BF16 输出、图 replay 和主 decode P50 的验证记录。

- [ ] **Step 1: 在同一容器建立关闭开关的基线**

启动 SGLang 前取消设置 `SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION`。以单 token decode、多 token batch、空 expert 和非均匀 expert token case 运行现有 DeepSeek V4 单算子对比脚本，保存最终 GMM2 BF16 输出与 warmup 后主 decode P50。

- [ ] **Step 2: 验证二进制注册与开启开关的路径选择**

Run: `python -c "import vllm_ascend, torch; print(torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2)"`

Expected: 打印可调用 op。随后设置 `SGLANG_NPU_USE_GMM_SWIGLU_QUANT_FUSION=true` 并重启 SGLang；不允许在已捕获图后再变更环境变量或导入二进制。

- [ ] **Step 3: 比较最终输出与图 replay**

对每个 case 比较基线与融合后的最终 GMM2 BF16 输出：

```python
cosine = torch.nn.functional.cosine_similarity(
    baseline.float().flatten(), fused.float().flatten(), dim=0
)
normalized_mae = (baseline.float() - fused.float()).abs().mean() / baseline.float().abs().mean().clamp_min(1e-6)
assert cosine >= 0.999
assert normalized_mae <= 0.01
```

连续执行至少两次同一 graph replay；两次均完成且输出满足相同门槛。

- [ ] **Step 4: 采集 profiler 证据和 P50**

在同样输入、路由结果、预热次数和图模式下采集旧/新 profiler。确认融合路径没有独立的 GMM1、limit、`npu_swiglu` 或 GMM1 后动态 MXFP quant；记录主 decode shape 的 P50，要求融合 P50 小于基线 P50。

- [ ] **Step 5: 运行模型级精度回归并记录未执行项**

运行项目已有 DeepSeek V4 accuracy case，确认相对基线无回退。若本地没有匹配的 CANN/torch_npu/NPU，明确记录 Task 2 Step 4 和本任务的未执行命令及原因，不将 mock 测试结果表述为 NPU 正确性或性能结论。

- [ ] **Step 6: 进行最终 diff 审查并提交修复（如有）**

Run: `git diff --check` 和 `git diff -- python/sglang/srt/hardware_backend/npu/quantization/fp4_moe_methods.py test/registered/unit/npu/quantization/test_fp4_moe_methods.py`

Expected: 无空白错误；修改仅限本计划列出的标准路径、测试和必要的 vLLM-Ascend 契约文件。仅在本任务确有修复时再提交对应文件，提交信息使用 `fix: validate DSV4 fused MXFP path`。

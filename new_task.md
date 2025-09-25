# SGLang MoE「可配置激活」实现手册（**纯文字，无代码**）

> 目标：在 **不新增文件**、**注册和获取合一**、**少 if-else** 的前提下，为 SGLang 的 MoE 路径（Triton / Cutlass，含 bf16、fp8、fp4）提供**可从 `config.json` 灵活定义激活与其参数**（如 Swish β、SwiGLU α、clamp limit）的机制，并**保持对现有官方 `config.json` 的完全兼容**。
> 备注：FlashInfer 融合 MoE 可在第二阶段对齐，本阶段先打通 Triton/Cutlass。

---

## 0. 一句话总览

* **做一件事**：把“激活类型 + 参数”（如 `hidden_act="swiglu"`, `alpha=1.702`, `limit=7`）从 **`config.json` → 模型类 → MoeRunnerConfig → MoE 内核** 持续透传，在 MoE 的激活步骤**通过一个统一入口函数**完成“选择 & 调用”，普通 `silu/gelu` 继续走原有高性能 fused op，其他激活（`swish`/`swiglu`/`reglu`/`geglu`…）走**单一的 Triton 通用核**。
* **不新增文件**：仅在**现有文件**内补充极少量逻辑与轻量数据结构。
* **注册/获取合一**：提供**一个入口函数**，既可“注册映射”，又可“按名取配置”；内部用**字典映射**避免冗长 if-else。
* **兼容默认**：若 `config.json` 未给某参数（如 `swiglu` 的 α），使用**模型约定的默认**（例如 `gpt-oss` 用 1.702；`swiglu_limit` 已存在则沿用 7）。

---

## 1. 改哪些文件（**新增文件：无**）

> 下列为“**最小侵入**”的修改清单与职责。路径以常见的 SGLang 目录结构描述（你已有 `sglang-main`，按对应位置查找）。

1. **`python/sglang/srt/layers/activation.py`**
   **职责**：提供**唯一入口函数**（注册+获取合一）与一个**极轻的数据载体**（概念上类似 ActivationSpec），作为**Triton/Cutlass 的共同门面**。
   **你要补充的内容（纯文字描述）**：

   * 维护一个\*\*（名称 → 构造器）映射表\*\*（如 `"silu"`, `"gelu"`, `"swish"`, `"swiglu"`, `"geglu"`, `"reglu"` 等）。
   * 入口函数职责：

     * 当传入“构造器”时，把该名称注册进映射表（**注册**）。
     * 当仅传字符串名称时，根据名称 + 关键字参数（alpha、limit、up\_shift 等）**返回一个 ActivationSpec**（**获取**）。
   * **ActivationSpec（数据载体）**：仅需描述

     * `mode`（如 `SILU_GLU` / `GELU_GLU` / `SWISH_GLU` / `RELU_GLU`），
     * `alpha`（或 β）、`limit`、`up_shift`（如 `swiglu` 的 `(up+1)` 形态）、
     * 以及是否可走**fast-path**（`silu/gelu` 可直达现有 fused op）。
   * 暴露一个**统一的“应用函数”**（概念：`apply_glu_activation(x, out, spec)`），内部策略：

     * 若 `spec` 指向 `silu`/`gelu` 且参数是默认值 → **走已有 sgl-kernel fused op**；
     * 否则 → **走 Triton 通用核**（见第 2 点）。

2. **`python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`**（或同目录中现有 Triton MoE 内核文件）
   **职责**：实现一个**通用、可参数化的 GLU 激活 Triton 内核**，并在 MoE 的激活阶段接入“统一入口”。
   **你要补充的内容（纯文字描述）**：

   * **一个通用 Triton 激活内核**，输入：

     * 上游 GEMM1 的输出 `X`（按惯例含 gate/up 两路），
     * 输出 `out`，
     * 编译期常量 `mode`（等效 switch/case）、
     * 运行时/编译期常量组合的参数：`alpha`、`limit`、`up_shift`。
   * **内核逻辑**（编译期分支，等效“switch/case 且每个 case 自带 break”）：

     * `SILU_GLU`：`gate * sigmoid(alpha * gate)` 与 `(up + up_shift)` 相乘；
     * `SWISH_GLU`：与上同，但 `alpha` 语义为 β；
     * `GELU_GLU`：GELU 近似再乘 `up`；
     * `RELU_GLU`：`relu(gate) * up`；
     * 若 `limit` 存在，对 `gate/up` 的中间值进行 clamp（对齐 `gpt-oss` 的 `swiglu_limit` 语义）。
   * \*\*在 `fused_experts_impl`（或等价位置）\*\*的激活阶段：

     * 用第 1 步的入口函数由 `hidden_act` 名称**拿到 ActivationSpec**；
     * 调用**统一应用函数**，从而：

       * `silu/gelu` → 仍走原 fused op（性能不退化）；
       * 其他激活 → 走**该通用 Triton 内核**。

3. **`python/sglang/srt/layers/moe/cutlass_moe.py`**（以及 `cutlass_w4a8_moe.py`、`fp4/fp8` 相关 Cutlass MoE 文件）
   **职责**：让 Cutlass 路径在激活环节也**统一走“门面函数”**（从而获得灵活激活），但**尽量复用原有 fused op**。
   **你要补充的内容（纯文字描述）**：

   * 把原来直接调用 `silu_and_mul` / `gelu_and_mul` 的地方，统一改成调用**第 1 步的“应用函数”**：

     * 如果是 `silu/gelu` 且默认参数 → **继续走现有 fused op**；
     * 否则 → \*\*（复用 Triton 通用核的路径）\*\*完成激活，再进入量化/GEMM2。

4. **`python/sglang/srt/models/gpt_oss.py`**（以及其他使用 MoE 的模型类文件）
   **职责**：把 `config.json` 中与激活相关的信息**透传**到 MoE Runner：**不强制新增配置项**、**保留默认值**。
   **你要补充的内容（纯文字描述）**：

   * 从 `config` 读取：

     * `hidden_act`（如 `"silu"`, `"swiglu"` 等）；
     * `hidden_act_alpha`（若无 → 对 `gpt-oss` 默认 **1.702**）；
     * `hidden_act_limit` 或 **已有的** `swiglu_limit`（`gpt-oss` 已定义为 **7**）。
   * 创建或填充 `MoeRunnerConfig` 时，**把上面三个量**原样透传：

     * `activation = hidden_act`
     * `gemm1_alpha = hidden_act_alpha（或默认 1.702）`
     * `gemm1_clamp_limit = hidden_act_limit（无则用 swiglu_limit，仍无则 None）`
   * **不修改官方 `config.json`**：若开发者后来在 config 里加上 `hidden_act_alpha/limit`，就自动覆盖默认。

> **新增文件**：**无**。
> 以上改动均发生在现有文件中，且每处改动仅增加**极少量文字/枚举/映射**与**调用路径**，不创建新模块。

---

## 2. 激活语义与默认值（约定清单）

> 所有默认值仅在 `config.json` 未提供时生效，优先使用**已有字段**（如 `swiglu_limit`）。

* `silu`：标准 SiLU-GLU（若实现形态为 GLU）。`alpha` 默认 = 1。
* `swish`：Swish-GLU，`alpha` 视为 β（`x * sigmoid(βx)`），默认 = 1。
* `swiglu`（**gpt-oss 变体**）：`alpha` 默认 = **1.702**；`limit` = `swiglu_limit`（**7**）；`up_shift` = 1（对应 `(up + 1)`）。
* `gelu`/`geglu`：GELU-GLU（近似实现），无额外参数。
* `relu`/`reglu`：ReLU-GLU，无额外参数。

> 说明：不同模型对 `swiglu` 的具体定义可能有“是否 `up+1`”等差异。为兼容主流实现，**把 `up_shift` 作为 ActivationSpec 的一个可选参数**（默认 0；`gpt-oss` 设为 1）。

---

## 3. 数据流与调用顺序（从 `config.json` 到 Kernel）

```
config.json
   └─ ModelConfig（各模型类，如 gpt_oss）
       └─ MoeRunnerConfig(activation, gemm1_alpha, gemm1_clamp_limit, ...)
           └─ MoE Runner（Triton/Cutlass 通用）
               ├─ GEMM1（W1/W3）
               ├─ 激活（统一入口）
               │    ├─ fast-path：silu/gelu → 现有 fused op
               │    └─ 其他：通用 Triton 内核（带 alpha/limit/up_shift）
               ├─ 量化（fp8/fp4 等）
               └─ GEMM2（W2）→ 汇聚/路由
```

---

## 4. 如何做到“注册 + 获取合一 & 少 if-else”

* **一个入口函数**负责两件事：

  1. 传入“名称 + 构造器” → **注册**到一个本地映射表（名称→构造器）；
  2. 仅传入“名称 + 参数字典” → **获取**一个 ActivationSpec（构造器处理默认值与参数装配）。
* **少 if-else 的关键**：

  * 不写 `if act == "silu"` / `elif act == "gelu"` … 的链式分支；
  * 而是 `映射表[name] → 构造器 → ActivationSpec`；
  * Triton 内核**用编译期常量 `mode`** 做静态分支（等效 switch/case 且天然“每个 case 都 break”）。

---

## 5. 与现有内核的性能兼容

* **保留 fast-path**：

  * `silu/gelu`（参数为默认值）时，仍直接调用**现有 sgl-kernel fused 激活**（不回退性能）。
* **统一 fallback**：

  * `swish/swiglu/reglu/geglu/...` 与“参数非常规”的 `silu/gelu` → 统一由**通用 Triton 内核**计算。
* **Cutlass 路径**：

  * 同样先尝试 fast-path（`silu/gelu`），不行再走通用核。
* **量化顺序不变**：

  * 仍是“激活 → 量化 → GEMM2”。

---

## 6. 与 `gpt-oss` 的兼容要点

* **拉取更新**：`gpt_oss` 的 MoE 引入了一个使用 Triton `_kernels` 的 `swiglu` 变体。
* **你要做的**：

  * 在模型类中**仍然只读取官方给的字段**（`hidden_act`, `swiglu_limit`）；
  * 对 `alpha` 用 `getattr` 默认 **1.702**；
  * **不要强迫**在 `config.json` 新增 `swiglu_alpha` 一类的键；
  * 这样能保证**不改官方配置就能跑通**，同时允许开发者在自定义模型里**覆盖默认**（例如定义 `hidden_act_alpha`）。

---

## 7. 编译与测试建议（**仍然不写代码，仅流程**）

1. **编译 sgl-kernel**（与你当前仓库方式一致）。
2. 本地快速验证：

   * 把 `num_hidden_layers=1`；
   * 启动参数用 `--load-format dummy`（不加载权重）；
   * 选择 MoE 模型（如 `gpt-oss`），`hidden_act="swiglu"`，不改配置文件也应启动成功。
3. 测试集：

   * 运行现有 MoE 测试（Cutlass、FP4、FP8、W4A8 等），应全部通过（新增逻辑默认不影响 `silu/gelu` 路径）。
   * 自测：给一个最小 dummy 输入，对比

     * fast-path（`silu/gelu`） vs 旧实现数值一致，
     * `swiglu(swiglu_limit=7, alpha=1.702)` 与 PyTorch 逐元素基线一致。
4. 性能回归：

   * 重点观测 `silu/gelu` 场景（应不退化）；
   * 其他激活初期可接受略低但正确（后续再考虑更深度 fused）。

---

## 8. 常见坑位与规避

* **不同仓的 `swiglu` 变体**：确认是否有 `up+1`，本方案用 `up_shift` 参数统一表示（默认 0，`gpt-oss` 用 1）。
* **limit 的落点**：明确 clamp 应用在 gate/up 的哪个阶段（对齐 `gpt-oss` 的实现语义）。
* **精度/类型**：激活在 bf16/fp16 完成，量化再做；保持与现有数据流一致。
* **开关回滚**：若某模型不稳定，暂时把 `hidden_act` 置回 `silu/gelu` 以快速回避。
* **“switch/case + break”问题**：在 Triton/Python 中用**编译期常量分支**即可等效，避免运行期贯穿。

---

## 9. 交付验收清单（对着打勾）

* [ ] **不新增文件**，仅在上述 4 处现有文件内少量补充。
* [ ] **注册/获取合一的入口函数**存在且被 Triton/Cutlass 路径共同调用。
* [ ] **silu/gelu** 默认走**原 fused op**（无性能回退）。
* [ ] **swish/swiglu/reglu/geglu** 可通过 **通用 Triton 内核**正确执行。
* [ ] `gpt-oss` 在**不改官方 `config.json`** 情况下运行成功：`alpha` 默认为 1.702，`limit` 用 `swiglu_limit=7`。
* [ ] 现有 MoE 测试全部通过；新增最小功能性检查通过。

---

## 10.（可选）未来阶段：FlashInfer 融合 MoE

* 第一阶段：仍在激活后，按现有流程进入 FlashInfer 的量化/GEMM；
* 第二阶段：若 FlashInfer 开放“可配置激活”接口，则把 `mode/alpha/limit/up_shift` 作为枚举与常量传下去，做真正的**融合激活**，避免中间写回。

---

### 附：名词与参数对齐速查

| 名称     | 语义（GLU 形态）                                      | 关键参数                       | 默认值（无配置时）                                              |
| ------ | ----------------------------------------------- | -------------------------- | ------------------------------------------------------ |
| silu   | `gate * sigmoid(gate)`                          | `alpha`（可视作 1）             | `alpha = 1`                                            |
| swish  | `gate * sigmoid(β * gate)`                      | `alpha`（即 β）               | `alpha = 1`                                            |
| swiglu | `SwiGLU(gate; alpha) * (up + up_shift)` + clamp | `alpha`、`limit`、`up_shift` | `alpha = 1.702`; `limit=7`; `up_shift=1`（以 gpt-oss 为例） |
| gelu   | `gelu(gate) * up`                               | —                          | —                                                      |
| geglu  | `gelu(gate) * up`                               | —                          | —                                                      |
| relu   | `relu(gate) * up`                               | —                          | —                                                      |
| reglu  | `relu(gate) * up`                               | —                          | —                                                      |

> `limit` 取自 `hidden_act_limit` 或已有 `swiglu_limit`（优先使用已有字段以兼容官方配置）。

---

如需，我可以基于你本地 `sglang-main` 的具体目录结构，逐个文件**列出建议插入的位置、函数名与段落标题**（仍然不写代码），方便你对着仓库修改。

## A01_B16: Cache-DiT Diffusion 深度排雷与 4090 压力测试总结

### 1. 测试目标与策略

- **目标**: 在本地 RTX 4090 上，系统性验证 SGLang Diffusion 模块（特别是 Cache-DiT 路径）的可用性与稳定性，找出 dev 镜像在真实环境中的工程化缺陷。
- **策略**: 采用分阶段、由浅入深的“排雷式”测试，从依赖补完 → 源码修复 → 硬件极限挑战，一步步逼近问题源头。

---

### 2. 分阶段测试历程

#### 阶段一：依赖补完（Dependency Patch）

- **现象**:
  - 运行 Wan 2.1 等 Diffusion 模型时：
    - 首先因为缺少 `accelerate` 无法使用 `device_map=cuda` 进行设备映射。
    - 随后因缺少 `ftfy`，导致文本清洗（text normalization）相关函数崩溃。
- **行动**:
  - 在容器内部手动补全缺失依赖：
    - `pip install accelerate`
    - `pip install ftfy`
- **结论**:
  - `lmsysorg/sglang:dev` 镜像在 Diffusion 模块维度仍然存在**依赖声明不完整**的问题。
  - 与 A01_B13 中的结论一致：镜像在 LLM 侧较完整，在 Diffusion 侧仍是“半成品”。

---

#### 阶段二：源码修复（Source-Level Fix）

- **现象**:
  - 切换到 SGLang 原生后端（绕开部分兼容层）后，触发 `ImportError`/`AttributeError`：
    - 提示在 `fsdp_load.py` 里找不到 `set_default_dtype`，并建议是否指 `set_default_torch_dtype`。
- **分析**:
  - 这是典型的**重构后忘记更新调用点**：
    - 函数定义从 `set_default_dtype` 重命名为 `set_default_torch_dtype`。
    - 某些下游模块（如 `comfyui_qwen_image_pipeline.py`）仍然引用旧名字。
- **行动**:
  - 通过 `sed` 等方式，将旧引用批量替换为 `set_default_torch_dtype`，使模块恢复可加载。
- **结论**:
  - 代码层面存在“坏味道”（code smell）：
    - 重构未配套全局重命名与测试。
    - 多模态/ComfyUI 相关路径明显缺乏 CI 覆盖。

---

#### 阶段三：硬件极限与 OOM 挑战（4090 + WSL2）

- **测试环境关键约束**:
  - WSL2 默认内存上限约 **17.8 GB**。
  - GPU: RTX 4090（显存 24 GB）。

- **现象 1：系统级 OOM (`Exit Code -9`)**
  - 加载 Wan 2.1 1.3B 等大模型时：
    - T5 风格的文本编码器体积巨大。
    - 触发 WSL2 级别的 OOM，被宿主系统直接 `SIGKILL` → `Exit Code -9`。

- **现象 2：VRAM 占满与 PCIe 交换**
  - 即便抬高 WSL2 内存上限，当模型勉强加载成功后：
    - 显存几乎被吃满（> 23 GB）。
    - 触发 GPU ↔ CPU 之间的 PCIe 内存交换。
    - 单步推理耗时飙升至 **1500 秒级别**。
    - 调度器因长时间无响应触发超时，任务失败。

- **结论**:
  - 对于 DiT / T5 编码器这一类**极重的多模态骨干网络**，SGLang 在：
    - 内存预测、
    - Offload 策略、
    - 超时控制与反馈
    三个维度都**缺乏友好的保护措施与自动降级逻辑**。

---

### 3. 不同模型的对比测试结果

| 模型名称        | 架构类型 | CPU 内存 / GPU 显存压力          | 运行表现          | 结论与症结                        |
|-----------------|----------|----------------------------------|-------------------|-----------------------------------|
| Wan 2.1 1.3B    | DiT + T5 | RAM 爆表，触发 WSL2 OOM (-9)    | 直接被系统杀死    | 编码器体积远超 WSL2 默认内存      |
| Z-Image-Turbo   | DiT      | VRAM 占满（接近 24GB）          | 极慢，最终超时    | 触发 PCIe 交换，调度器等待超时    |
| SD-Turbo        | UNet     | 显存占用可控，CPU/RAM 压力较小  | 流畅完成          | 经典 SD UNet 结构，对资源友好     |

- **关键观察**:
  - **DiT / T5 型** Diffusion：对内存/显存极其敏感，在默认 WSL2 环境下“几乎注定失败”。
  - **UNet 型** Diffusion（如 SD-Turbo）：在同一环境下可以**顺利跑通**，说明：
    - 驱动、CUDA、SGLang 基础推理链路是通的。
    - 真正的问题出在**模型结构 + 内存管理策略**的组合上。

---

### 4. 已锁定的几个问题点（面向 PR / Issue 的输入）

结合 A01_B12 / B13 / B15 以及本轮 Cache-DiT 测试，可以认为你已经帮项目方“锁定”了以下待修复点：

1. **镜像缺陷（Dockerfile / 依赖声明层）**
   - dev 镜像在构建时**漏掉了 Diffusion 模块必备依赖**：
     - `accelerate`（设备映射 / low_cpu_mem_usage）
     - `ftfy`（文本清洗 / 正规化）
   - 建议：
     - 在 Dockerfile 或 `pyproject.toml` 中补齐依赖；
     - 或提供带 `diffusion` extras 的专用镜像标签（如 `sglang:diffusion`）。

2. **代码坏味道（Refactor Hygiene）**
   - 典型案例：`set_default_dtype` → `set_default_torch_dtype` 重命名后：
     - `comfyui_qwen_image_pipeline.py` 等路径未更新引用。
   - 反映问题：
     - 多模态/ComfyUI 相关代码缺乏测试与 CI 保护；
     - 重构流程缺少“全局引用检查 + 全量测试”的步骤。

3. **调度策略冲突与报错友好性**
   - Cache-DiT 加速器与 `layerwise-offload` 之间存在**隐性互斥**：
     - 某些组合会在运行期触发极端慢推理或资源争用。
   - 当前报错信息对这种策略级冲突提示不够：
     - 用户难以从日志中直接看出“是策略冲突，而非单纯 OOM 或驱动问题”。

4. **环境准入门槛 / 自动降级缺失**
   - 对于 DiT / T5 参数量级的 Diffusion 模型：
     - SGLang 缺乏对“低内存环境”（例如 16–24GB RAM + 单卡 4090）的预检测和自动降级机制。
   - 建议：
     - 在模型加载前做一次**粗粒度内存评估**并给出友好提示；
     - 在极端内存吃紧时自动关闭某些昂贵特性（如 Cache-DiT、部分缓存）或建议使用 UNet 型轻量模型。

---

### 5. 建议的后续行动路径

结合你目前“打算提交 PR 修复兼容性问题”的计划，可以按优先级这样推进：

1. **P0：镜像 / 依赖层修复**
   - 在 Dockerfile 中补齐：
     - `pip install accelerate ftfy`
   - 或在 `pyproject.toml` 的 `dependencies` / extras 中声明：
     - `"accelerate"`, `"ftfy"`.
   - 这是最容易落地、收益最大的改动，也最适合单独提一个小 PR。

2. **P1：源码命名 Bug 修复**
   - 修正 `set_default_dtype` 引用为 `set_default_torch_dtype`；
   - 或在 `utils.py` 中加一个向后兼容别名：
     - `set_default_dtype = set_default_torch_dtype`.

3. **P2：调度策略与 OOM 体验优化**
   - 在 Cache-DiT + layerwise-offload 组合上增加显式检查与清晰错误信息；
   - 对 DiT/T5 大模型增加内存前置检查，避免用户在 1500 秒之后才看到超时。

4. **Issue / PR 维度的输出**
   - 依赖缺失与函数命名问题：可以对应两个独立的 Issue + 小 PR；
   - Cache-DiT + 大模型 + 4090/WSL2 的极限案例：可以作为一篇“环境限制与建议配置”的补充文档，帮助官方完善文档与默认参数。

---

### 6. 总结：这轮“Cache-DiT 深度排雷”的价值

- 你已经从**依赖层 → 源码层 → 调度层 → 硬件边界**，完整走了一遍 SGLang Diffusion 模块在本地 4090 环境下的“死亡路径”。
- 这些结论不仅能支撑你后续的 PR 设计，也为官方在：
  - 镜像构建、
  - 依赖管理、
  - 多模态代码质量、
  - 大模型运行体验
  四个维度提供了**非常具体可操作的改进点**。

这篇文档可作为 **Issue #17671 后续深度调查的“第二阶段总结”**，也可以在未来 PR/Issue 中直接引用其中的结论与表格。


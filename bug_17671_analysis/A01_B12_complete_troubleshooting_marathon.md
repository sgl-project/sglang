# A01_B12: 完整的排障马拉松记录 - RTX 4090 Diffusion 测试

## 概述

这是一次非常有代表性的排障过程：从试图让一台 RTX 4090 画图开始，最后以它流畅地讲了一个程序员的冷笑话结束。完整记录了在 Windows + Docker + SGLang 环境下测试 Diffusion 模型的全部过程。

**测试日期**: 2025年1月
**测试环境**: Windows + Docker + WSL2 + RTX 4090 + CUDA 12.9
**测试目标**: 验证 SGLang 对 Diffusion 模型的支持（Issue #17671）

---

## 1. 测试历程总结

### 阶段一：Diffusion 初试

**操作**:
- 尝试运行 `segmind/small-sd` 模型
- 通过 Docker 挂载 D 盘缓存目录

**结果**: ❌ **失败**

**关键问题**:
- 镜像极其庞大（10GB+）
- 下载和解压耗时极长
- 模型加载过程异常缓慢

**命令示例**:
```bash
docker run --gpus all \
  -v D:\docker_data\huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:dev \
  sglang generate --model-path segmind/small-sd \
  --prompt "test" --save-output
```

---

### 阶段二：文件系统博弈

**操作**:
- 试图通过挂载本地目录来持久化模型数据
- 期望避免重复下载

**结果**: ❌ **严重失败**

**关键问题**:
- 触发了 SGLang 内部 `maybe_download_model` 的死循环
- 即使下载了 5.7GB，系统仍判定文件不完整
- 文件校验机制在 Windows + Docker 挂载环境下失效

**错误表现**:
```
ValueError: Downloaded model at /root/.cache/huggingface/... is still incomplete 
after forced re-download. The model repository may be missing required components 
(model_index.json, transformer/, or vae/).
```

**根本原因**:
- SGLang 的 diffusers 后端在校验文件时，对 Linux 路径结构有深度依赖
- Windows 磁盘挂载到 Docker 时，文件系统的延迟或锁定会导致校验函数返回 False
- 触发无休止的 `force_download` 循环

---

### 阶段三：Windows 锁定

**操作**:
- 尝试手动清理报错的本地模型目录
- 希望重新开始下载

**结果**: 😫 **折磨**

**关键问题**:
- Windows/WSL2 锁定了 Docker 挂载的文件夹
- 导致无法删除，产生"删不掉的垃圾"
- 需要重启 Docker 或 WSL2 才能释放锁定

**解决方案**:
```powershell
# 需要重启 Docker Desktop 或 WSL2
wsl --shutdown
# 或者重启 Docker Desktop
```

---

### 阶段四：LLM 核心测试

**操作**:
- 放弃 Diffusion，改用 `Qwen2.5-1.5B` 进行零挂载、纯容器运行
- 测试 SGLang 的 LLM 功能

**结果**: ✅ **100% 成功**

**关键发现**:
- 模型秒开
- API 调用成功
- RTX 4090 的超高推理速度得到验证
- 推理几乎是瞬时完成

**命令示例**:
```bash
# 启动服务器（无挂载）
docker run --gpus all -p 30000:30000 \
  lmsysorg/sglang:dev \
  python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct

# API 调用测试
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "讲一个程序员的冷笑话"}]
  }'
```

**结果**: 4090 流畅地讲了一个程序员的冷笑话 ✅

---

## 2. 核心结论与发现

### A. SGLang 是一个"严重偏科"的学霸

#### LLM 模块 (极强) ⭐⭐⭐⭐⭐

**优势**:
- `launch_server` 逻辑非常成熟
- 对于 Qwen、Llama 等模型的支持是工业级的
- API 兼容性好（OpenAI 兼容 API）
- 性能优异（4090 上几乎是瞬时完成）
- 文档完善，使用简单

**结论**: SGLang 的 LLM 模块已经达到生产级别

#### Diffusion 模块 (实验性/脆弱) ⭐⭐

**问题**:
- 扩散模型加载器逻辑非常"轴"
- 有一套极其死板的文件校验机制
- 在 Windows + Docker 的挂载环境下极易崩溃
- 容错率几乎为零
- 对文件系统要求过于苛刻

**结论**: SGLang 的 Diffusion 模块目前还处于实验阶段，工程化程度不够

---

### B. Windows/Docker 路径是"万恶之源"

**核心问题**:
1. **文件系统差异**:
   - SGLang 的 diffusers 后端在校验文件时，对 Linux 路径结构有深度依赖
   - Windows 路径挂载到 Docker 时，文件系统语义不同

2. **文件锁定问题**:
   - Windows/WSL2 会锁定 Docker 挂载的文件夹
   - 导致无法删除，产生"删不掉的垃圾"

3. **校验机制失效**:
   - 文件系统的延迟或锁定会导致校验函数返回 False
   - 触发无休止的 `force_download` 循环

**根本原因**:
- SGLang 的 `maybe_download_model` 函数在 Windows + Docker 挂载环境下无法正确识别已下载的文件
- 文件校验逻辑假设了纯 Linux 文件系统环境

---

### C. 4090 的性能没问题，是"翻译官"坏了

**硬件验证**:
- ✅ RTX 4090 显卡性能正常
- ✅ CUDA 12.9 环境配置正确
- ✅ 只要避开 SGLang 那个还没写好的 Diffusion 校验逻辑，4090 的表现非常惊人
- ✅ LLM 推理几乎是瞬时完成

**结论**: 问题不在硬件和环境，而在 SGLang 的 Diffusion 模块实现

---

## 3. 说明了什么问题？

### 复现工作的难点

**为什么难跑通**:
- SGLang 的 dev 镜像在**多模态/图像生成**这一块的工程化程度还不够
- 它对环境（尤其是文件系统）的要求过于苛刻
- Windows + Docker 挂载的组合是"死亡组合"

### 避坑指南

**在 SGLang 里跑 Diffusion，目前绝不要使用 Windows 本地挂载**

**推荐方案**:
1. ✅ **让它在容器内的虚拟磁盘（内存中）完成所有操作**
   ```bash
   # 不挂载任何目录，让模型下载到容器内
   docker run --gpus all --shm-size 32g \
     lmsysorg/sglang:dev \
     sglang generate --model-path <model> --prompt "test"
   ```

2. ✅ **在纯 Linux 环境下运行**
   - 使用 Linux 服务器或 WSL2 的 Linux 文件系统
   - 避免 Windows 路径挂载

3. ⚠️ **如果必须挂载，使用 Linux 路径**
   ```bash
   # 在 WSL2 中，使用 Linux 路径
   docker run --gpus all \
     -v /mnt/wsl/docker-desktop-data/... \
     lmsysorg/sglang:dev ...
   ```

### 工具选型建议

**如果你复现的工作重点是**:

- ✅ **速度优化**: SGLang 是对的
  - LLM 推理速度极快
  - 4090 性能得到充分发挥

- ⚠️ **稳定产出**: 目前 SGLang 的这个模块还没达到 ComfyUI 或 WebUI 那样的稳定性
  - Diffusion 模块还在实验阶段
  - 建议使用更成熟的工具（ComfyUI、Stable Diffusion WebUI）

---

## 4. 下一步建议

### 选项 1: 查找官方 Bug 修复记录

**目标**: 查找 SGLang 官方仓库里，针对 Diffusion 模块最新的 Bug 修复记录（Hotfixes）

**可能的问题**:
- `maybe_download_model` 的死循环问题
- Windows + Docker 挂载环境下的文件校验问题
- 文件锁定问题

**查找方向**:
- GitHub Issues 和 PRs
- 最近的 commits 记录
- 官方文档中的已知问题

### 选项 2: 纯净测试（推荐）

**使用官方文档里明确标注为"Stable"的模型**

**推荐模型**:
- `runwayml/stable-diffusion-v1-5` (官方测试过的)
- `stabilityai/stable-diffusion-xl-base-1.0` (SDXL)
- `stabilityai/sdxl-turbo` (SDXL Turbo)

**测试方法**:
```bash
# 完全不挂载目录，让模型下载到容器内
docker run --gpus all --shm-size 32g \
  --rm -it \
  lmsysorg/sglang:dev \
  sglang generate \
  --model-path runwayml/stable-diffusion-v1-5 \
  --backend diffusers \
  --prompt "A logo With Bold Large text: SGL Diffusion" \
  --save-output
```

**预期结果**:
- 如果成功：说明问题确实在 Windows 挂载
- 如果失败：说明问题更深层，需要进一步调查

---

## 5. 技术细节记录

### 问题代码位置

**可能的问题函数**:
- `sglang.multimodal_gen.runtime.entrypoints.diffusion_generator.DiffGenerator`
- `maybe_download_model` 函数
- 文件校验逻辑

### 错误模式

1. **死循环模式**:
   ```
   下载 → 校验失败 → force_download → 下载 → 校验失败 → ...
   ```

2. **文件锁定模式**:
   ```
   Windows 锁定 → 无法删除 → 无法重新下载 → 卡死
   ```

3. **路径问题模式**:
   ```
   Windows 路径 → Docker 挂载 → Linux 校验 → 路径不匹配 → 失败
   ```

---

## 6. 经验总结

### 成功经验

1. ✅ **LLM 模块测试成功**：证明了 SGLang 的核心功能是可靠的
2. ✅ **硬件验证成功**：4090 性能得到验证
3. ✅ **API 调用成功**：OpenAI 兼容 API 工作正常

### 失败教训

1. ❌ **不要使用 Windows 挂载**：在 Diffusion 模块中这是"死亡组合"
2. ❌ **不要过度依赖文件持久化**：在实验阶段，容器内运行更可靠
3. ❌ **不要忽略环境差异**：Windows + Docker 的组合需要特别小心

### 最佳实践

1. ✅ **先测试 LLM**：验证环境配置是否正确
2. ✅ **使用官方测试过的模型**：避免模型本身的问题
3. ✅ **避免 Windows 挂载**：在 Diffusion 模块中尤其重要
4. ✅ **记录详细日志**：便于问题定位和复现

---

## 7. 相关文档链接

- [A01_B01_original_issue.md](./A01_B01_original_issue.md) - 原始问题
- [A01_B08_reproduction_results.md](./A01_B08_reproduction_results.md) - 复现测试结果
- [A01_B09_issue_reply_to_kevin.md](./A01_B09_issue_reply_to_kevin.md) - 给 Kevin 的回复
- [A01_B10_disk_space_troubleshooting.md](./A01_B10_disk_space_troubleshooting.md) - 磁盘空间问题排查
- [A01_B11_diffusion_parameter_issue.md](./A01_B11_diffusion_parameter_issue.md) - Parameter 参数问题

---

## 8. 待办事项

- [ ] 查找 SGLang 官方仓库中 Diffusion 模块的最新 Bug 修复记录
- [ ] 使用官方标注为"Stable"的模型进行纯净测试
- [ ] 如果问题持续，考虑向官方提交详细的 Bug 报告
- [ ] 记录成功的配置方案，供后续参考

---

**最后更新**: 2025年1月
**测试人员**: [待填写]
**测试环境**: Windows + Docker + WSL2 + RTX 4090 + CUDA 12.9

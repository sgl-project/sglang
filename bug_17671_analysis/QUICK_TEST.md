# 快速验证命令（按顺序执行）

## 1. 检查镜像版本信息

```powershell
docker image inspect lmsysorg/sglang:dev --format "{{.Id}} {{.Created}}"
```

**目的**: 确认测试的是哪个版本的镜像

---

## 2. 明确验证 diffusers 是否存在

```powershell
docker run --rm -it --gpus all lmsysorg/sglang:dev python -c "import diffusers; print('diffusers ok', diffusers.__version__)"
```

**预期结果**:
- **如果输出**: `diffusers ok x.y.z` → 镜像已包含 diffusers（可能已修复）
- **如果报错**: `ModuleNotFoundError: No module named 'diffusers'` → 镜像缺少 diffusers（复现成功）

---

## 3. 如果 diffusers 存在，测试 sglang generate（可选）

```powershell
docker run --rm -it --gpus all -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format --backend diffusers --prompt "test" --save-output
```

**目的**: 如果 diffusers 存在，测试 sglang generate 是否能正常工作

---

## 结果解读

### 情况1: diffusers 存在
- **说明**: 镜像可能已经修复，包含了 diffusers
- **下一步**: 测试 sglang generate 是否能正常工作
- **Issue状态**: 可能需要更新文档说明版本，或关闭issue

### 情况2: diffusers 不存在
- **说明**: 成功复现了问题
- **下一步**: 执行 Step 3（验证安装后恢复）
- **Issue状态**: 确认是bug，可以提交修复PR

# 磁盘空间问题排查和解决方案

## 问题描述

测试 `runwayml/stable-diffusion-v1-5` 时遇到磁盘空间不足错误：
```
RuntimeError: Data processing error: CAS service error : IO Error: No space left on device (os error 28)
```

错误信息显示：
- 需要空间：3438.17 MB (~3.4 GB)
- 可用空间：1337.31 MB (~1.3 GB)

## 当前状态检查

### 1. Docker 系统磁盘使用情况

```powershell
docker system df
```

**结果**：
- Images: 167.9GB (可回收 31.19GB)
- Containers: 25.23GB (可回收 25.23GB，99%可回收)

### 2. 宿主机磁盘空间

- **C 盘**：已用 916GB，可用 1082GB ✅
- **D 盘**：已用 2884GB，可用 1116GB ✅
- **HuggingFace 缓存**：54.04 GB（位于 `C:\Users\nickc\.cache\huggingface`）

### 3. Docker 容器内部空间

```powershell
docker run --rm lmsysorg/sglang:dev df -h /
```

**结果**：
- 总空间：1007G
- 已用：158G
- 可用：798G ✅

## 问题分析

**关键发现**：
1. ✅ 宿主机有足够的磁盘空间（C 盘 1TB+ 可用）
2. ✅ Docker 容器内部有足够的空间（798G 可用）
3. ❌ 但下载时仍然报错"磁盘空间不足"

**可能的原因**：
1. **Docker Desktop 磁盘配额限制**：Docker Desktop 可能设置了磁盘使用上限
2. **挂载目录的临时文件**：下载过程中的临时文件可能占用了空间
3. **Docker 镜像层占用**：多个镜像层可能占用了大量空间
4. **HuggingFace 缓存目录碎片**：54GB 的缓存可能包含很多小文件，导致碎片化

## 解决方案

### 方案 1：清理 Docker 未使用的资源（推荐）

```powershell
# 查看可以清理的空间
docker system df

# 清理所有未使用的资源（包括停止的容器、未使用的镜像、网络、构建缓存）
docker system prune -a --volumes

# 或者只清理未使用的镜像和容器（更安全）
docker system prune -a
```

**注意**：`docker system prune -a` 会删除所有未使用的镜像和容器，请确保没有重要数据。

### 方案 2：清理 HuggingFace 缓存

```powershell
# 查看缓存大小
$hfCache = "$env:USERPROFILE\.cache\huggingface"
$size = (Get-ChildItem -Path $hfCache -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Current size: $([math]::Round($size, 2)) GB"

# 手动清理不需要的模型（谨慎操作）
# 进入缓存目录，删除不需要的模型文件夹
```

### 方案 3：检查 Docker Desktop 磁盘配额

1. 打开 **Docker Desktop**
2. 进入 **Settings** → **Resources** → **Advanced**
3. 检查 **Disk image size** 设置
4. 如果设置了上限，可以：
   - 增加磁盘配额
   - 或者清理未使用的资源释放空间

### 方案 4：使用更小的模型进行测试

由于 `runwayml/stable-diffusion-v1-5` 需要下载约 3.4GB 的文件，可以改用更小的模型：

```powershell
# 使用 segmind/tiny-sd（约 500MB）
docker run --rm -it --gpus all -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path segmind/tiny-sd --prompt "test" --save-output
```

### 方案 5：更改挂载目录到 D 盘（如果 C 盘空间紧张）

```powershell
# 创建 D 盘的 HuggingFace 缓存目录
New-Item -ItemType Directory -Force -Path "D:\docker_data\huggingface"

# 使用 D 盘目录挂载
docker run --rm -it --gpus all -v D:\docker_data\huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path segmind/tiny-sd --prompt "test" --save-output
```

## 推荐操作步骤

1. **首先清理 Docker 未使用的资源**：
   ```powershell
   docker system prune -a
   ```

2. **检查 Docker Desktop 磁盘配额**：
   - 打开 Docker Desktop → Settings → Resources → Advanced
   - 确保 Disk image size 足够大（建议至少 100GB）

3. **如果仍然失败，使用更小的模型测试**：
   ```powershell
   docker run --rm -it --gpus all -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface lmsysorg/sglang:dev sglang generate --model-path segmind/tiny-sd --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
   ```

## 验证

清理后，再次检查 Docker 系统空间：

```powershell
docker system df
```

应该看到可回收空间明显减少。

---

**最后更新**: 2026年1月27日

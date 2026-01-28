# A01_B01: Issue #17671 原始内容

## 相关文档
- [A01: Diffusion模型启动问题详解](./A01_diffusion_launch_issue.md) - 了解整体问题（待创建）

---

## Issue 链接
https://github.com/sgl-project/sglang/issues/17671

## 问题标题
[Bug] Can't launch diffusion models by following the official doc

## 提交者
@kevin85421 (Collaborator)

## 创建时间
opened 3 days ago · edited by kevin85421

## 问题描述

### Bug描述
按照官方文档启动docker容器时，无法启动diffusion模型。

### 复现步骤
```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```

### 环境信息
- Docker镜像: `lmsysorg/sglang:dev`
- 模型: `black-forest-labs/FLUX.1-dev`

---

## 讨论和回复

### @Kangyan-Zhou 的回复（2天前）
**问题**: What's the last commit in the dev docker file? I couldn't reproduce it on my end (although I hit a different issue that needs to manually install diffusion in pyproject). The error looks like some transient network issue with hugginface.

**关键信息**:
- 遇到了不同的问题：需要手动在pyproject中安装diffusion
- 可能是HuggingFace的网络问题

### @kevin85421 的澄清（6小时前）
**核心问题**: 
> "although I hit a different issue that needs to manually install diffusion in pyproject"
> 
> Sorry for the misleading, the issue I want to express is that **SGLang images didn't have SGLang diffusion**.

**问题本质**:
- **SGLang Docker镜像没有包含SGLang diffusion功能**
- 需要手动安装diffusion到pyproject

---

## 问题总结

### 核心问题
**SGLang Docker镜像（`lmsysorg/sglang:dev`）缺少diffusion功能支持**

### 问题表现
1. 按照官方文档启动diffusion模型失败
2. Docker镜像中没有包含SGLang diffusion
3. 需要手动安装diffusion依赖

### 影响范围
- 使用Docker部署diffusion模型的用户
- 按照官方文档操作的用户
- 使用`lmsysorg/sglang:dev`镜像的用户

---

## 相关链接
- [Issue #17671](https://github.com/sgl-project/sglang/issues/17671)
- [SGLang官方文档](https://docs.sglang.ai/)

---

**最后更新**: 2025年1月

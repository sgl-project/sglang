# SGLang Serve 命令差异说明

## 问题

为什么这个命令能直接运行：
```bash
sudo sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --backend sglang
```

而我们的命令需要额外参数：
```bash
sglang serve \
    --model-path zai-org/GLM-Image \
    --backend sglang \
    --port 30000 \
    --trust-remote-code
```

## 主要区别

### 1. **模型类型差异**（最关键）

#### Wan2.1-T2V-1.3B-Diffusers
- **模型格式**: 标准的 Diffusers 格式模型
- **自定义代码**: ❌ **不需要** `--trust-remote-code`
- **原因**: 这是一个标准的 Diffusers 模型，使用 HuggingFace 的标准架构，不需要执行自定义的 Python 代码

#### GLM-Image
- **模型格式**: 包含自定义代码的模型
- **自定义代码**: ✅ **必须**使用 `--trust-remote-code`
- **原因**: GLM-Image 使用了自定义的模型架构代码（在 `modeling_*.py` 等文件中），HuggingFace 默认不允许执行远程代码，必须显式授权

### 2. **端口参数**

#### 他们的命令（Wan2.1-T2V）
```bash
# 没有指定 --port
sudo sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --backend sglang
```
- **默认端口**: SGLang 默认使用端口 **30000**
- **不需要显式指定**: 如果不指定 `--port`，会自动使用默认端口

#### 我们的命令（GLM-Image）
```bash
# 显式指定了 --port 30000
sglang serve --model-path zai-org/GLM-Image --backend sglang --port 30000 --trust-remote-code
```
- **显式指定端口**: 虽然默认也是 30000，但我们显式指定了
- **原因**: 
  - 确保端口一致性（便于脚本管理）
  - 如果将来需要更改端口，更容易修改
  - 明确性更好（代码可读性）

### 3. **sudo 的使用**

#### 他们的命令
```bash
sudo sglang serve ...
```
- **使用 sudo**: 以 root 权限运行
- **可能的原因**:
  - 访问某些系统资源
  - 绑定特权端口（但 30000 不是特权端口）
  - 环境配置问题（可能不需要）

#### 我们的命令
```bash
# 不使用 sudo
sglang serve ...
```
- **不使用 sudo**: 以普通用户权限运行
- **推荐做法**: 除非必要，否则不应该使用 sudo 运行服务

### 4. **环境激活**

#### 他们的命令
- 可能直接在系统环境中运行（如果 sglang 是全局安装的）
- 或者已经在激活的虚拟环境中

#### 我们的命令
```bash
# 在脚本中显式激活环境
cd /data/users/yandache/workspaces/sglang
source env_sglang/bin/activate
```
- **显式激活虚拟环境**: 确保使用正确的 Python 环境和依赖
- **更可靠**: 不依赖系统环境配置

## 总结对比表

| 特性 | Wan2.1-T2V 命令 | GLM-Image 命令 | 说明 |
|:-----|:----------------|:---------------|:-----|
| **模型** | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | `zai-org/GLM-Image` | 不同的模型 |
| **--trust-remote-code** | ❌ 不需要 | ✅ **必须** | GLM-Image 有自定义代码 |
| **--port** | ❌ 未指定（使用默认 30000） | ✅ 显式指定 30000 | 我们的更明确 |
| **sudo** | ✅ 使用 | ❌ 不使用 | 我们的更安全 |
| **环境激活** | 可能隐式 | ✅ 显式激活 | 我们的更可靠 |

## 为什么 GLM-Image 需要 --trust-remote-code？

### HuggingFace 的安全机制

HuggingFace 默认**不允许执行远程仓库中的自定义代码**，这是为了防止恶意代码执行。

### GLM-Image 的特殊性

GLM-Image 模型仓库包含自定义的模型架构代码，例如：
- `modeling_glm_image.py` - 自定义的模型类
- `configuration_glm_image.py` - 自定义的配置类
- 其他自定义的 Python 模块

这些代码需要被加载和执行才能正确加载模型。

### 如何判断模型是否需要 --trust-remote-code？

1. **查看模型仓库**: 如果仓库中有 `modeling_*.py`、`configuration_*.py` 等自定义代码文件
2. **查看模型卡片**: 模型说明中通常会提到需要 `trust_remote_code=True`
3. **尝试加载**: 如果不加 `--trust-remote-code` 会报错，提示需要这个参数

## 我们的脚本设计更优的原因

### 1. **显式参数**
- 明确指定所有必要参数
- 代码可读性更好
- 便于维护和调试

### 2. **环境管理**
- 显式激活虚拟环境
- 确保依赖版本一致
- 避免环境冲突

### 3. **安全性**
- 不使用 sudo（除非必要）
- 遵循最小权限原则

### 4. **可配置性**
- 通过脚本参数控制后端（sglang/diffusers）
- 便于切换和测试

## 如果我们要运行 Wan2.1-T2V

如果我们要用同样的方式运行 Wan2.1-T2V，命令应该是：

```bash
# 不需要 --trust-remote-code
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --backend sglang \
    --port 30000
```

或者更简单（使用默认端口）：
```bash
sglang serve \
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --backend sglang
```

## 结论

**两个命令都能运行，但原因不同**：

1. **Wan2.1-T2V**: 标准 Diffusers 模型，不需要 `--trust-remote-code`
2. **GLM-Image**: 包含自定义代码，**必须**使用 `--trust-remote-code`

**我们的脚本设计更规范**：
- 显式参数更清晰
- 环境管理更可靠
- 安全性更好（不使用 sudo）

**建议**: 保持我们当前的脚本设计，这是更好的实践。

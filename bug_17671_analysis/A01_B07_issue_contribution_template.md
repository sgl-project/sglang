# A01_B07: Issue贡献模板

**参考文档**: 
- [A01_B06_reproduction_steps.md](./A01_B06_reproduction_steps.md) ⭐ **Bug复现步骤**

---

## 1. 在Issue #17671中回复的模板

### 1.1 基本回复模板

```markdown
## 复现确认

我在4090上成功复现了这个问题。

### Step 1: 验证 diffusers 模块缺失

```bash
$ docker run --rm -it lmsysorg/sglang:dev python -c "import diffusers"
```

**输出**:
```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'diffusers'
```

✅ **确认**: 镜像确实缺少 `diffusers` 模块

---

### Step 2: 镜像信息

```bash
$ docker image inspect lmsysorg/sglang:dev | head -n 40
```

**输出**:
```
[
    {
        "Id": "sha256:...",
        "RepoTags": [
            "lmsysorg/sglang:dev"
        ],
        "Created": "2025-01-XX...",
        ...
    }
]
```

---

### Step 3: 最小可复现示例（使用 tiny 模型）

使用 tiny 模型验证，不需要下载大模型：

```bash
$ docker run --gpus all --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:dev \
  sglang generate \
    --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format \
    --prompt "test" \
    --save-output \
    --backend diffusers
```

**输出**: 同样的 `ModuleNotFoundError: No module named 'diffusers'`

---

### Step 4: 安装后恢复确认

在容器内安装 `sglang[diffusion]` 后：

```bash
# 进入容器
$ docker run --gpus all --rm -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:dev bash

# 在容器内
$ uv pip install 'sglang[diffusion]' --prerelease=allow
$ python -c "import diffusers; print('✓ diffusers installed')"
✓ diffusers installed

$ sglang generate \
    --model-path hf-internal-testing/tiny-stable-diffusion-pipe-variants-right-format \
    --prompt "test" \
    --save-output \
    --backend diffusers
# 成功运行
```

✅ **结论**: 
- 安装 `sglang[diffusion]` 后可以成功运行
- 问题确实是镜像缺少 diffusion 依赖
- 这是文档/镜像不一致问题

---

## 建议的修复方案

1. **文档修复**（推荐）: 在Docker使用文档中明确说明需要安装 `sglang[diffusion]`
2. **Docker镜像修复**: 提供包含diffusion的专用镜像tag，或添加build arg支持

如果需要，我可以提交PR修复文档。
```

---

## 2. 收集的信息清单

### 2.1 必须收集的信息

- [ ] Step 1 的完整错误信息（截图或文本）
- [ ] 镜像信息（`docker image inspect` 的输出）
- [ ] 安装后是否恢复（是/否）

### 2.2 可选收集的信息

- [ ] Step 3 的错误信息（如果执行了）
- [ ] 安装过程的输出（如果执行了Step 4）
- [ ] 运行成功的输出（如果执行了Step 4）

---

## 3. 截图建议

### 3.1 关键截图

1. **Step 1 的错误信息**:
   - 显示完整的 `ModuleNotFoundError`
   - 包含命令和输出

2. **镜像信息**:
   - 显示镜像的Created时间
   - 显示镜像的标签信息

3. **安装后恢复**（如果执行了）:
   - 显示成功导入 diffusers
   - 显示成功运行模型

---

## 4. 贡献价值

### 4.1 对维护者的帮助

1. **快速确认问题**: 
   - 提供明确的复现步骤
   - 确认问题本质

2. **最小可复现示例**:
   - 使用 tiny 模型，不需要大模型
   - 节省时间和资源

3. **解决方案验证**:
   - 确认安装后可以恢复
   - 帮助决定修复方向

### 4.2 对社区的帮助

1. **其他用户可以参考**:
   - 提供workaround
   - 提供复现步骤

2. **加速问题解决**:
   - 明确问题本质
   - 提供修复方向

---

**参考文档**:
- [A01_B06_reproduction_steps.md](./A01_B06_reproduction_steps.md) - Bug复现步骤
- [A01_B05_PR_draft.md](./A01_B05_PR_draft.md) - PR修改草案

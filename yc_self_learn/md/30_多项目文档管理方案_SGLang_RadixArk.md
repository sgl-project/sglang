# 多项目文档管理方案 - SGLang & RadixArk

## 📋 问题

**现状**：
- 已有大量 SGLang 相关文档（30+ 个文件）
- 想 pull RadixArk 的代码进行研究
- 希望文档能在两个项目之间通用

**挑战**：
- 两个项目共享核心技术（RadixAttention、调度器、KV Cache 等）
- 但实现细节可能不同
- 如何让文档既通用又准确？

---

## 🎯 解决方案

### 方案 1：项目标记 + 通用文档（推荐）

**核心思路**：
- 在文档中标记适用的项目（SGLang / RadixArk / 通用）
- 创建通用文档目录和项目特定文档目录
- 使用文档索引/映射

**目录结构**：
```
yc_self_learn/
├── md/
│   ├── common/                    # 通用文档（两个项目都适用）
│   │   ├── RadixAttention详解.md
│   │   ├── KV_Cache详解.md
│   │   ├── 调度器原理.md
│   │   └── ...
│   ├── sglang/                    # SGLang 特定文档
│   │   ├── SGLang_调度器实现详解.md
│   │   ├── SGLang_API调用流程.md
│   │   └── ...
│   ├── radixark/                  # RadixArk 特定文档
│   │   ├── RadixArk_架构设计.md
│   │   ├── RadixArk_训练管线.md
│   │   └── ...
│   └── projects/                  # 项目对比和关系文档
│       ├── 29_SGLang_Contribution_现状分析.md
│       ├── SGLang_vs_RadixArk_对比.md
│       └── ...
```

**文档标记格式**：
```markdown
# 文档标题

**适用项目**: [SGLang] [RadixArk] [通用]
**最后更新**: 2025-01-XX

## 内容...

<!-- SGLang 特定内容 -->
**SGLang 实现**：
- 代码位置：`python/sglang/srt/...`
- 实现细节：...

<!-- RadixArk 特定内容 -->
**RadixArk 实现**：
- 代码位置：`radixark/...`
- 实现细节：...

<!-- 通用原理 -->
**核心原理**：
- 这是两个项目都使用的技术
- ...
```

---

### 方案 2：使用 Git Submodule（适合代码仓库）

**⚠️ 澄清**：这不是关于"环境"的，而是关于**代码仓库管理**的。

**核心思路**：
- 将 RadixArk 的代码仓库作为 submodule 添加到当前 SGLang 仓库中
- 这样可以在一个仓库中同时访问两个项目的代码
- 文档放在主仓库（SGLang），代码通过 submodule 引用

**什么是 Git Submodule？**
- Git Submodule 允许在一个 Git 仓库中**引用另一个 Git 仓库**
- 不是复制代码，而是**链接**到另一个仓库
- 可以独立更新 submodule 的代码，不影响主仓库

**操作步骤**：
```bash
# 1. 添加 RadixArk 作为 submodule
# 这样 RadixArk 的代码会出现在当前仓库的 radixark/ 目录下
git submodule add <radixark_repo_url> radixark

# 2. 更新 submodule（第一次需要）
git submodule update --init --recursive

# 3. 拉取 RadixArk 最新代码
cd radixark
git pull origin main
cd ..

# 4. 更新主仓库中的 submodule 引用
cd ..
git add radixark
git commit -m "Update RadixArk submodule"
```

**目录结构**：
```
yc_research/sglang/                # 当前 SGLang 仓库（主仓库）
├── yc_self_learn/
│   └── md/                        # 文档在主仓库
│       ├── common/                # 通用文档
│       ├── sglang/                # SGLang 特定文档
│       └── radixark/              # RadixArk 特定文档
├── python/                        # SGLang 代码
├── sgl-kernel/                    # SGLang 代码
└── radixark/                      # RadixArk submodule（链接到 RadixArk 仓库）
    ├── src/                       # RadixArk 代码
    └── ...
```

**实际效果**：
- ✅ 可以在一个仓库中同时看到 SGLang 和 RadixArk 的代码
- ✅ 文档统一管理在主仓库
- ✅ 可以对比两个项目的代码实现
- ✅ RadixArk 代码更新时，只需要 `cd radixark && git pull`

**优点**：
- 代码和文档分离（文档在主仓库，代码通过 submodule）
- 可以独立更新 RadixArk 代码（不影响主仓库）
- 文档统一管理（都在主仓库）
- 可以轻松对比两个项目的代码

**缺点**：
- 需要管理 submodule（需要学习 Git Submodule 的使用）
- 如果 RadixArk 不是公开仓库，可能无法添加
- 其他人 clone 你的仓库时，需要额外步骤初始化 submodule

**适用场景**：
- ✅ 你想在一个仓库中同时研究两个项目的代码
- ✅ 你想对比 SGLang 和 RadixArk 的实现差异
- ✅ 你想统一管理文档，但代码分开管理

**不适用场景**：
- ❌ 如果你只是想要"共享文档"，方案 1 更简单
- ❌ 如果你想要"共享开发环境"（Python 环境、依赖等），这不是 submodule 的作用

---

### 方案 3：符号链接（适合本地开发）

**核心思路**：
- 创建通用文档目录
- 使用符号链接在两个项目之间共享

**操作步骤**（Windows）：
```powershell
# 1. 创建通用文档目录
New-Item -ItemType Directory -Path "I:\yc_research\common_docs\md"

# 2. 移动通用文档到共享目录
Move-Item "I:\yc_research\sglang\yc_self_learn\md\06_RadixAttention详解.md" "I:\yc_research\common_docs\md\"

# 3. 创建符号链接
New-Item -ItemType SymbolicLink -Path "I:\yc_research\sglang\yc_self_learn\md\06_RadixAttention详解.md" -Target "I:\yc_research\common_docs\md\06_RadixAttention详解.md"

# 4. 在 RadixArk 项目中创建相同的符号链接
New-Item -ItemType SymbolicLink -Path "I:\yc_research\radixark\docs\06_RadixAttention详解.md" -Target "I:\yc_research\common_docs\md\06_RadixAttention详解.md"
```

**目录结构**：
```
I:\yc_research\
├── common_docs\                   # 共享文档目录
│   └── md\
│       ├── 06_RadixAttention详解.md
│       ├── 25_KV_Cache详解.md
│       └── ...
├── sglang\
│   └── yc_self_learn\
│       └── md\
│           ├── 06_RadixAttention详解.md  # 符号链接
│           └── ...（SGLang 特定文档）
└── radixark\
    └── docs\
        ├── 06_RadixAttention详解.md      # 符号链接
        └── ...（RadixArk 特定文档）
```

**优点**：
- 文档真正共享，修改一处，两处都更新
- 不需要复制文件

**缺点**：
- 符号链接在不同系统上可能有问题
- Git 需要特殊配置才能正确处理符号链接

---

### 方案 4：文档索引 + 条件标记（最灵活）

**核心思路**：
- 保持现有文档结构
- 在文档开头添加项目标记
- 创建文档索引，按项目分类

**文档模板**：
```markdown
# 文档标题

<!-- 项目标记 -->
**适用项目**: 
- ✅ SGLang
- ✅ RadixArk
- ✅ 通用原理

**项目差异**:
- SGLang: 代码位置 `python/sglang/srt/...`
- RadixArk: 代码位置 `radixark/...`（待补充）

---

## 核心内容（通用）

这部分内容适用于两个项目...

<!-- SGLang 特定 -->
## SGLang 实现细节

...

<!-- RadixArk 特定 -->
## RadixArk 实现细节

（待补充）
```

**创建文档索引**：
```markdown
# 文档索引 - SGLang & RadixArk

## 通用文档（两个项目都适用）

1. [RadixAttention 详解](./06_RadixAttention详解.md) - ✅ SGLang ✅ RadixArk
2. [KV Cache 详解](./25_KV_Cache详解与Decode带宽瓶颈.md) - ✅ SGLang ✅ RadixArk
3. [调度器原理](./12_调度器waiting_queue与get_new_batch详解.md) - ✅ SGLang ✅ RadixArk

## SGLang 特定文档

1. [SGLang API 调用流程](./10_API调用完整流程详解.md) - ✅ SGLang
2. [SGLang 调度器实现](./12_调度器waiting_queue与get_new_batch详解.md) - ✅ SGLang

## RadixArk 特定文档

1. [RadixArk 架构设计](./radixark/RadixArk_架构设计.md) - ✅ RadixArk（待创建）
2. [RadixArk 训练管线](./radixark/RadixArk_训练管线.md) - ✅ RadixArk（待创建）

## 项目对比文档

1. [SGLang vs RadixArk 对比](./29_SGLang_Contribution_现状分析.md)
```

---

## 🎯 推荐方案：方案 1 + 方案 4 结合

### 实施步骤

**步骤 1：创建目录结构**
```bash
cd yc_self_learn/md
mkdir -p common sglang radixark projects
```

**步骤 2：分类现有文档**

**通用文档**（移动到 `common/`）：
- `06_RadixAttention详解.md` - RadixAttention 是核心技术，两个项目都用
- `25_KV_Cache详解与Decode带宽瓶颈.md` - KV Cache 原理通用
- `03_TTFT_为什么重要.md` - 概念通用
- `24_为什么攒Batch会让TTFT变大_排队等待详解.md` - 原理通用

**SGLang 特定文档**（移动到 `sglang/`）：
- `10_API调用完整流程详解.md` - SGLang API
- `12_调度器waiting_queue与get_new_batch详解.md` - SGLang 调度器实现
- `22_SGLang完整请求流程详解_纠正版.md` - SGLang 特定流程
- `04_Docker_SGLang_本地开发环境设置.md` - SGLang 环境

**项目对比文档**（移动到 `projects/`）：
- `29_SGLang_Contribution_现状分析.md`
- 新建：`SGLang_vs_RadixArk_对比.md`

**步骤 3：更新文档，添加项目标记**

在每个文档开头添加：
```markdown
**适用项目**: 
- ✅ SGLang
- ✅ RadixArk（如果适用）
- ✅ 通用原理

**项目差异**:
- SGLang: [具体实现细节]
- RadixArk: [待补充或具体实现细节]
```

**步骤 4：创建文档索引**

创建 `00_文档索引_SGLang_RadixArk.md`：
```markdown
# 文档索引 - SGLang & RadixArk

## 📚 快速导航

### 通用文档（两个项目都适用）
- [RadixAttention 详解](./common/06_RadixAttention详解.md)
- [KV Cache 详解](./common/25_KV_Cache详解与Decode带宽瓶颈.md)
- [调度器原理](./common/12_调度器原理.md)

### SGLang 特定文档
- [SGLang API 调用流程](./sglang/10_API调用完整流程详解.md)
- [SGLang 调度器实现](./sglang/12_调度器waiting_queue与get_new_batch详解.md)

### RadixArk 特定文档
- [RadixArk 架构设计](./radixark/RadixArk_架构设计.md)（待创建）
- [RadixArk 训练管线](./radixark/RadixArk_训练管线.md)（待创建）

### 项目对比文档
- [SGLang vs RadixArk 对比](./projects/29_SGLang_Contribution_现状分析.md)
```

---

## 🔧 实际操作建议

### 如果 RadixArk 是公开仓库

**方案 A：Git Submodule**
```bash
# 1. 在当前仓库添加 RadixArk submodule
git submodule add <radixark_github_url> radixark

# 2. 文档放在主仓库，通过路径引用
# 文档中引用代码：`radixark/src/...`
```

### 如果 RadixArk 是私有仓库或本地项目

**方案 B：文档标记 + 索引**
```bash
# 1. 保持现有文档结构
# 2. 在文档中添加项目标记
# 3. 创建文档索引
# 4. 根据项目标记筛选文档
```

### 如果两个项目在同一台机器上

**方案 C：符号链接（Windows）**
```powershell
# 创建共享文档目录
New-Item -ItemType Directory -Path "I:\yc_research\common_docs"

# 移动通用文档
Move-Item "I:\yc_research\sglang\yc_self_learn\md\06_RadixAttention详解.md" "I:\yc_research\common_docs\"

# 创建符号链接
New-Item -ItemType SymbolicLink `
  -Path "I:\yc_research\sglang\yc_self_learn\md\06_RadixAttention详解.md" `
  -Target "I:\yc_research\common_docs\06_RadixAttention详解.md"
```

---

## 📝 文档模板

### 通用文档模板

```markdown
# [文档标题]

**适用项目**: 
- ✅ SGLang
- ✅ RadixArk
- ✅ 通用原理

**最后更新**: 2025-01-XX

---

## 核心原理（通用）

这部分内容适用于两个项目...

---

## SGLang 实现

**代码位置**: `python/sglang/srt/...`

**实现细节**:
- ...

---

## RadixArk 实现

**代码位置**: `radixark/...`（待补充）

**实现细节**:
- ...

---

## 项目差异

| 维度 | SGLang | RadixArk |
|------|--------|----------|
| 实现方式 | ... | ... |
| 性能特点 | ... | ... |
```

---

## 🎯 具体实施计划

### 阶段 1：准备（1-2 小时）

1. **创建目录结构**
   ```bash
   cd yc_self_learn/md
   mkdir -p common sglang radixark projects
   ```

2. **分析现有文档**
   - 列出所有文档
   - 判断哪些是通用的，哪些是 SGLang 特定的

3. **创建文档索引**
   - 新建 `00_文档索引_SGLang_RadixArk.md`
   - 列出所有文档及其适用项目

### 阶段 2：分类和移动（2-3 小时）

1. **移动通用文档到 `common/`**
2. **移动 SGLang 特定文档到 `sglang/`**
3. **移动项目对比文档到 `projects/`**

### 阶段 3：更新文档（持续）

1. **在文档开头添加项目标记**
2. **补充 RadixArk 相关信息**（当 pull 到代码后）
3. **更新文档索引**

### 阶段 4：Pull RadixArk 代码

1. **找到 RadixArk 仓库地址**
2. **Clone 或添加 submodule**
3. **分析代码结构**
4. **补充 RadixArk 特定文档**

---

## 💡 最佳实践

1. **保持文档同步**：
   - 通用原理更新时，两个项目都受益
   - 项目特定内容分开维护

2. **使用清晰的标记**：
   - ✅ 表示适用
   - ❌ 表示不适用
   - ⚠️ 表示部分适用或待确认

3. **定期更新索引**：
   - 新增文档时更新索引
   - 文档分类变化时更新索引

4. **版本控制**：
   - 文档变更记录在 git 中
   - 重要更新添加日期标记

---

## 📌 下一步行动

1. **立即行动**：
   - [ ] 创建目录结构（common, sglang, radixark, projects）
   - [ ] 创建文档索引文件
   - [ ] 分析现有文档，分类

2. **找到 RadixArk 仓库**：
   - [ ] 搜索 RadixArk GitHub 仓库
   - [ ] 确认是否开源
   - [ ] 决定使用 submodule 还是独立 clone

3. **Pull RadixArk 代码**：
   - [ ] Clone 或添加 submodule
   - [ ] 分析代码结构
   - [ ] 对比与 SGLang 的差异

4. **补充文档**：
   - [ ] 在通用文档中补充 RadixArk 信息
   - [ ] 创建 RadixArk 特定文档
   - [ ] 更新文档索引

---

**记住**：文档是活的，需要持续更新。先建立结构，然后逐步完善。
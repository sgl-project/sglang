# Z-Image-Turbo 文档归档索引

> 本目录包含 Claude Code 在 Z-Image-Turbo 性能分析过程中生成的参考文档。
> 已通过 `.gitignore` 排除，**不会上传到 GitHub**。

---

## 目录结构

```
zimage_256_256/
├── analysis_report/
│   └── ANALYSIS_REPORT.md          ← 【主报告，已 git track，可上传】
├── deep_gemm/
│   ├── introduce.md                ← DeepGemm 技术介绍（手写/已 track）
│   └── report.md                   ← FP8 量化逻辑调研（手写/已 track）
├── review/
│   └── pre_transpose.md            ← Code Review 记录（手写/已 track）
│
├── docs/                           ← 【以下全部 gitignore，仅本地参考】
│   ├── fp8_dispatch_analysis/      ← FP8 Python dispatch 开销分析
│   ├── ffn_codebase_search/        ← FFN 代码路径搜索记录
│   ├── nsys_profiling_guide/       ← nsys profiling 工具使用指南
│   ├── torch_profiler_guide/       ← torch.profiler 集成指南
│   └── cudaprofiler_integration/   ← CudaProfilerApi 实现分析
│
└── README_DOCS.md                  ← 本文件
```

---

## 分类说明

### 可上传到 GitHub 的文档

| 文件 | 用途 | 说明 |
|------|------|------|
| `analysis_report/ANALYSIS_REPORT.md` | **主性能分析报告** | 包含 Section 1-10 的完整分析，含 nsys kernel 对比、GEMM 加速比、优化路线图 |
| `deep_gemm/introduce.md` | DeepGemm 技术介绍 | 原创调研文档 |
| `deep_gemm/report.md` | FP8 量化逻辑调研 | 原创调研文档 |
| `review/pre_transpose.md` | Code Review 记录 | Pre-transpose weight scale 方案评审 |

### 仅本地参考的文档（gitignore）

#### 1. `docs/fp8_dispatch_analysis/` — FP8 Python Dispatch 开销分析

**用途**：分析 FP8 DeepGemm 每次 GEMM 调用的 CPU 侧开销（tensor 分配、JIT check、quant launch），解释 256×256 FP8 E2E 负收益的根因。

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `README_FP8_ANALYSIS.md` | 215 | 入口索引 |
| `FP8_DOCS_INDEX.md` | 217 | 文档导航 |
| `FP8_DISPATCH_ANALYSIS.md` | 654 | **核心**：BF16 vs FP8 dispatch 路径对比 |
| `FP8_GEMM_FORWARD_PATH.md` | 547 | FP8 GEMM forward 完整代码路径追踪 |
| `FP8_CODE_SNIPPETS.md` | 445 | 关键函数的源码摘录 |
| `FP8_QUICK_REFERENCE.md` | 193 | 快速查阅 |
| `FP8_ANALYSIS_SUMMARY.md` | 163 | 结论摘要 |
| `VISUAL_COMPARISON.md` | 373 | BF16 vs FP8 dispatch 流程图 |
| `FP8_VISUAL_SUMMARY.txt` | 289 | ASCII 架构图 |

**何时查阅**：优化 FP8 dispatch overhead 时（CUDA Graph、fused quant+GEMM 等）。

#### 2. `docs/ffn_codebase_search/` — FFN 代码路径搜索记录

**用途**：搜索 sglang 中 FFN（FeedForward）层的实现位置，确认 `"feed_forward" not in key` bug。

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `README_FFN_SEARCH.md` | 409 | 入口索引 |
| `FFN_CODE_REFERENCE.md` | 512 | FFN 代码路径完整参考 |
| `FFN_SEARCH_INDEX.md` | 308 | 搜索结果索引 |
| `FFN_SEARCH_SUMMARY.md` | 331 | 搜索结论摘要 |

**何时查阅**：修改 DiT FFN 层或排查 FP8 量化覆盖范围时。

#### 3. `docs/nsys_profiling_guide/` — nsys 工具使用指南

**用途**：NVIDIA Nsight Systems 在 sglang 中的使用方法、分析技巧、脚本位置。

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `NSYS_PROFILING_COMPREHENSIVE_GUIDE.md` | 809 | **核心**：nsys 完整使用手册 |
| `NSYS_TOOLS_INDEX.md` | 261 | 工具快速索引 |
| `PROFILING_ANALYSIS_SUMMARY.md` | 640 | 256×256 profiling 分析总结 |
| `PROFILING_EXPLORATION_INDEX.md` | 379 | 代码探索记录 |
| `ANALYSIS_FILES.txt` | 208 | 分析文件清单 |

**何时查阅**：做 nsys profiling、分析 kernel trace、对比不同配置时。

#### 4. `docs/torch_profiler_guide/` — torch.profiler 集成指南

**用途**：sglang 中 torch.profiler 的集成方式、使用命令、trace 分析方法。

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `SGLANG_PROFILER_COMPLETE_GUIDE.md` | 708 | **核心**：完整技术参考 |
| `PROFILER_QUICK_START.md` | 293 | 快速上手 cheat sheet |
| `PROFILER_IMPLEMENTATION_GUIDE.md` | 563 | 5 种实现模式 |
| `README_PROFILER_DOCS.md` | 296 | 入口索引 |

**何时查阅**：用 `--profile` 抓 CPU+GPU trace、分析 launch gap 时。

#### 5. `docs/cudaprofiler_integration/` — CudaProfilerApi 实现分析

**用途**：分析 scheduler.py 中 `cudaProfilerStart/Stop` 的实现，验证 warmup 排除逻辑。

| 文件 | 行数 | 内容 |
|------|:---:|------|
| `CUDAPROFILER_INTEGRATION_ANALYSIS.md` | 526 | **核心**：完整技术分析 |
| `README_CUDAPROFILER.md` | 281 | 入口索引 |
| `CUDAPROFILER_QUICK_REFERENCE.txt` | 201 | ASCII 快速参考 |

**何时查阅**：修改 nsys profiling 的 capture range、验证 warmup 排除时。

---

## 统计

| 类别 | 文件数 | 总行数 | 是否上传 |
|------|:---:|:---:|:---:|
| 主报告 (`analysis_report/`) | 1 | 867 | **是** |
| 手写调研 (`deep_gemm/`, `review/`) | 3 | 822 | **是** |
| FP8 dispatch 分析 | 9 | 3,296 | 否 |
| FFN 搜索记录 | 4 | 1,560 | 否 |
| nsys profiling 指南 | 5 | 2,297 | 否 |
| torch.profiler 指南 | 4 | 1,860 | 否 |
| CudaProfilerApi 分析 | 3 | 1,008 | 否 |
| **合计** | **29** | **11,710** | — |

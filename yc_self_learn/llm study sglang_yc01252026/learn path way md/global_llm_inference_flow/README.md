# LLM 推理全局架构（从 GPU 到客户使用）

本文件夹存放「从 GPU 到最终用户」的 LLM 推理全局设计文档，用于建立端到端图景。

## 文档列表

| 文件 | 说明 |
|------|------|
| [00_LLM推理全局架构_从GPU到客户使用.md](./00_LLM推理全局架构_从GPU到客户使用.md) | 主文档：五层架构、各层职责、与 SGLang 对应关系、自测清单 |
| [01_硬件层_GPU与Kernel详解_行业明星与最佳实践.md](./01_硬件层_GPU与Kernel详解_行业明星与最佳实践.md) | 第⑤层展开：Kernel 细节、最佳公司与案例、行业 super star（Tri Dao） |

## 建议阅读顺序

1. 先读 **00_LLM推理全局架构_从GPU到客户使用.md** 建立整体图景。
2. 再结合 `../casestudy/` 与 `../sglang_day01/` 等具体模块深入。

## 相关路径

- 系统级项目计划：`../sglang study plan/02_整体框架学习路径_系统级项目计划.md`
- SGLang Case Study：`../casestudy/`

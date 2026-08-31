<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="logo" width="400" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/sglang)](https://pypi.org/project/sglang)
![PyPI - Downloads](https://static.pepy.tech/badge/sglang?period=month)
[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/sgl-project/sglang/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang)](https://github.com/sgl-project/sglang/issues)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://www.sglang.io/"><b>🌐 官方网站</b></a> |
<a href="https://lmsys.org/blog/"><b>技术博客</b></a> |
<a href="https://docs.sglang.io/"><b>官方文档</b></a> |
<a href="https://roadmap.sglang.io/"><b>研发路线图</b></a> |
<a href="https://slack.sglang.io/"><b>加入 Slack</b></a> |
<a href="https://meet.sglang.io/"><b>每周开发者周会</b></a> |
<a href="https://github.com/sgl-project/sgl-learning-materials?tab=readme-ov-file#slides"><b>演讲幻灯片</b></a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 最新动态 (News)
- [2026/07] 🔥 SGLang 与 Miles 率先实现对 Kimi K3 的首发即时支持（Day-0）([博客](https://lmsys.org/blog/2026-07-27-kimi-k3-day0-support/))。
- [2026/07] RadixArk 与 Google 合作，将 SGLang 的全部全功能特性完整带入 Google TPU 生态 ([博客](https://lmsys.org/blog/2026-07-30-sglang-google-tpu/))。
- [2026/07] 使用 SGLang 支撑 GLM5.2 NVFP4 智能体高负载：两周内达成 500 TPS 吞吐 ([博客](https://lmsys.org/blog/2026-07-13-glm52-optimization/))。
- [2026/06] 🔥 下一代投机解码（Speculative Decoding）：DFlash 与 Spec V2 架构 ([博客](https://lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/))。
- [2026/06] SGLang 首发即时支持最新开源大模型（[Nemotron 3 Ultra](https://lmsys.org/blog/2026-06-04-nvidia-run-nemotron-3-ultra/)、[Nemotron 3 Super](https://lmsys.org/blog/2026-03-11-run-nvidia-nemotron-3-super/)、[Higgs Audio v3 TTS 语音大模型](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/))。
- [2026/04] 🔥 首发即时支持 DeepSeek-V4：从极速推理到基于 SGLang 和 Miles 的强化学习验证 ([博客](https://lmsys.org/blog/2026-04-25-deepseek-v4/))。
- [2026/02] 🔥 在 NVIDIA GB300 NVL72 机架上解锁 25 倍推理性能飞跃 ([博客](https://lmsys.org/blog/2026-02-20-gb300-inferencex/))。
- [2026/01] SGLang Diffusion 加速视频与图像生成扩散模型推理 ([博客](https://lmsys.org/blog/2026-01-16-sglang-diffusion/))。

<details>
<summary>展开查看更多历史动态</summary>

- [2025/12] SGLang 首发即时支持多款前沿模型（[MiMo-V2-Flash](https://lmsys.org/blog/2025-12-16-mimo-v2-flash/)、[Nemotron 3 Nano](https://lmsys.org/blog/2025-12-15-run-nvidia-nemotron-3-nano/)、[Mistral Large 3](https://github.com/sgl-project/sglang/pull/14213)、[LLaDA 2.0 扩散大语言模型](https://lmsys.org/blog/2025-12-19-diffusion-llm/)、[MiniMax M2](https://lmsys.org/blog/2025-11-04-miminmax-m2/))。
- [2025/11] SGLang Diffusion 深度优化多模态视频/图像生成 ([博客](https://lmsys.org/blog/2025-11-07-sglang-diffusion/))。
- [2025/10] SGLang 通过 SGLang-Jax 后端在 Google TPU 上实现原生运行 ([博客](https://lmsys.org/blog/2025-10-29-sglang-jax/))。
- [2025/10] PyTorch Conference 2025 SGLang 主题演讲 ([幻灯片](https://github.com/sgl-project/sgl-learning-materials/blob/main/slides/sglang_pytorch_2025.pdf))。
- [2025/09] 在 GB200 NVL72 上部署 DeepSeek（解耦 PD 与超大规模专家并行 EP）：Prefill 提升 3.8 倍，Decode 吞吐提升 4.8 倍 ([博客](https://lmsys.org/blog/2025-09-25-gb200-part-2/))。
- [2025/09] SGLang 首发支持具备 Sparse Attention 的 DeepSeek-V3.2 ([博客](https://lmsys.org/blog/2025-09-29-deepseek-V32/)).
- [2025/08] SGLang 首发即时支持 OpenAI gpt-oss 模型 ([指南](https://github.com/sgl-project/sglang/issues/8833))。
- [2025/06] SGLang 荣获 a16z 第三期开源 AI 资助项目 ([a16z 博客](https://a16z.com/advancing-open-source-ai-through-benchmarks-and-bold-experimentation/))。
- [2025/05] 在 96 块 H100 GPU 集群上通过 PD 分离与大规模 EP 部署 DeepSeek ([博客](https://lmsys.org/blog/2025-05-05-large-scale-ep/))。
- [2025/03] SGLang 正式加入 PyTorch 生态系统 ([PyTorch 博客](https://pytorch.org/blog/sglang-joins-pytorch/))。
- [2025/01] 在 NVIDIA 和 AMD GPU 上为 DeepSeek V3/R1 提供首发专属优化支持 ([指南](https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3))。
- [2024/12] v0.4 发布：零开销 Batch 调度器、感知缓存的负载均衡器、超快结构化输出 ([博客](https://lmsys.org/blog/2024-12-04-sglang-v0-4/))。
- [2024/09] v0.3 发布：DeepSeek MLA 提速 7 倍、torch.compile 提速 1.5 倍 ([博客](https://lmsys.org/blog/2024-09-04-sglang-v0-3/))。
- [2024/01] 提出 **RadixAttention** 前缀缓存机制，推理速度提升高达 **5 倍** ([博客](https://lmsys.org/blog/2024-01-17-sglang/))。

</details>

## 关于 SGLang

SGLang 是一个专为大语言模型（LLM）与多模态大模型打造的高性能推理服务框架。
旨在从单张消费级 GPU 到超大规模分布式集群中，均能提供超低延迟与极高吞吐量的推理服务。核心特性包括：

- **极速运行时 (Fast Runtime)**：基于 **RadixAttention** 实现高效 KV Cache 前缀缓存；内置零开销 CPU 调度器；支持 Prefill 与 Decode 解耦（PD Disaggregation）、投机解码（Speculative Decoding）、连续批处理（Continuous Batching）、Paged Attention、张量/流水线/专家/数据并行（TP / PP / EP / DP）、结构化 JSON 输出、分块 Prefill（Chunked Prefill）、多种量化方案（FP4 / FP8 / INT4 / AWQ / GPTQ）以及 Multi-LoRA 并发批处理。
- **广泛的模型支持**：全面支持主流大语言模型（Llama、Qwen、DeepSeek、Kimi、GLM、GPT、Gemma、Mistral 等）、Embedding 向量模型（e5-mistral、gte、mcdse）、Reward 奖励模型（Skywork）以及 Diffusion 扩散模型（WAN、Qwen-Image），具备极佳的扩展性。完全兼容 Hugging Face 生态与 OpenAI API 标准。
- **丰富的硬件生态支持**：支持 NVIDIA GPU（GB200 / B300 / H100 / A100 / RTX 5090 / Spark）、AMD GPU（MI355 / MI300）、Intel Xeon CPU、Google TPU、华为昇腾（Ascend NPU）等各类芯片平台。
- **繁荣活跃的开源社区**：由开源组织 LMSYS 孵化维护，在全球工业界得到广泛落地与部署，驱动全球超过 **400,000 张 GPU** 稳定运行。
- **后训练与强化学习主力基础设施**：SGLang 是众多前沿基座模型训练的基石 Rollout 后端，已被知名后训练/强化学习框架广泛采用，如 [**AReaL**](https://github.com/inclusionAI/AReaL)、[**Miles**](https://github.com/radixark/miles)、[**slime**](https://github.com/THUDM/slime)、[**Tunix**](https://github.com/google/tunix)、[**verl**](https://github.com/volcengine/verl) 等。

## 快速入门指南

- [安装 SGLang](https://docs.sglang.io/get_started/install.html)
- [快速上手 (Quick Start)](https://docs.sglang.io/basic_usage/send_request.html)
- [后端推理教程 (OpenAI API 兼容)](https://docs.sglang.io/basic_usage/openai_api_completions.html)
- [前端编程语言教程 (Frontend Tutorial)](https://docs.sglang.io/references/frontend/frontend_tutorial.html)
- [开发者贡献指南](https://docs.sglang.io/developer_guide/contribution_guide.html)

## 基准测试与性能报告

详细评测请查阅版本发布技术博客：[v0.2 评测](https://lmsys.org/blog/2024-07-25-sglang-llama3/)、[v0.3 评测](https://lmsys.org/blog/2024-09-04-sglang-v0-3/)、[v0.4 评测](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)、[大规模专家并行 (EP)](https://lmsys.org/blog/2025-05-05-large-scale-ep/)、[GB200 机架级并行](https://lmsys.org/blog/2025-09-25-gb200-part-2/)、[GB300 超长上下文优化](https://lmsys.org/blog/2026-02-19-gb300-longctx/)。

## 业界落地与赞助支持

SGLang 已在生产环境中大规模部署，每天支撑数万亿 Token 的稳定生成。深受全球顶级科技企业与顶尖学术机构的信赖与采用，包括 xAI、NVIDIA、AMD、Intel、LinkedIn、Cursor、Oracle Cloud、Google Cloud、Microsoft Azure、AWS、Modal、MIT、斯坦福大学、加利福尼亚大学伯克利分校、清华大学、百度、蚂蚁集团、阿里巴巴、腾讯等。
作为开源 LLM 推理引擎的事实标准，SGLang 在全球超过 40 万张 GPU 上持续运行。
SGLang 目前托管于非营利开源组织 [LMSYS](https://lmsys.org/about/)。

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="logo" width="800" margin="10px"></img>

## 联系我们

企业客户如需大规模部署 SGLang、获取技术咨询、探讨商业赞助或商务合作机会，请邮件联系：[sglang@lmsys.org](mailto:sglang@lmsys.org)。

长期活跃的 SGLang 核心贡献者可申请 AI 编程助手赞助（如 Cursor、Claude Code 或 OpenAI Codex）。欢迎将您最重要的 Commit 或 Pull Request 发送至 [sglang@lmsys.org](mailto:sglang@lmsys.org)。

## 致谢 (Acknowledgment)

SGLang 的设计汲取了以下优秀开源项目的灵感并复用了部分代码：[Guidance](https://github.com/guidance-ai/guidance)、[vLLM](https://github.com/vllm-project/vllm)、[LightLLM](https://github.com/ModelTC/lightllm)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)、[Outlines](https://github.com/outlines-dev/outlines) 与 [LMQL](https://github.com/eth-sri/lmql)。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年8月31日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！

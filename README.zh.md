# SGLang: 面向大语言模型与多模态模型的高性能推理引擎

<p align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png" alt="SGLang" width="400">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/sglang/"><img src="https://img.shields.io/pypi/v/sglang?style=flat&amp;label=PyPI&amp;labelColor=555555&amp;color=orange" alt="PyPI version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green?style=flat&amp;labelColor=555555" alt="License: Apache 2.0"></a>
  <a href="https://pypistats.org/packages/sglang"><img src="https://img.shields.io/pypi/dm/sglang?style=flat&amp;label=Downloads&amp;labelColor=555555&amp;color=blue" alt="PyPI downloads per month"></a>
</p>

<p align="center">
  <a href="https://docs.sglang.io/">文档</a> |
  <a href="https://cookbook.sglang.io/">Cookbook</a> |
  <a href="https://www.sglang.io/">官方网站</a> |
  <a href="https://lmsys.org/blog/">博客</a> |
  <a href="https://slack.sglang.io/">Slack 社区</a>
</p>

SGLang 是一个面向大语言模型（LLM）、视觉-语言模型（VLM）与扩散模型（Diffusion）的开源高性能推理框架，专为 Agent 智能体工作负载、强化学习（RL）Rollout 生成和大规模线上服务进行了深度优化。[SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion) 是其内置的图像与视频生成引擎，已直接包含在本代码仓库与 `sglang` Python 软件包中。

👋 请参考下方指引快速上手，或参加 [SGLang 活动日历](https://www.sglang.io/events) 与社区交流（涵盖 Meetup、开发者例会、研讨会及 Office Hours）。

## 快速上手 (Get Started)

拉取包含 SGLang 及其全部依赖项的官方 Docker 镜像：

```bash
docker pull lmsysorg/sglang:latest
```

或者，在已激活的 Python 虚拟环境中使用 uv 安装 SGLang：

```bash
uv pip install --prerelease=allow sglang
```

接下来，启动您的模型：

- [快速入门 (Quickstart)](https://docs.sglang.io/docs/get-started/quickstart)：运行您的首个模型并发送推理请求。
- [Cookbook 指南](https://cookbook.sglang.io/)：根据您的模型与硬件配置获取开箱即用的启动命令。

## 硬件平台支持 (Supported Hardware)

SGLang 支持广泛的 GPU、TPU、NPU、CPU 以及 Apple Silicon 平台。

| 平台 | 代表性硬件 |
| --- | --- |
| [NVIDIA](https://docs.sglang.io/docs/hardware-platforms/nvidia-gpus) | A100；H100/H200/H800/H20；B200/B300/GB200/GB300；精选 RTX 30/40/50 系列、RTX 6000 Ada / PRO 6000；DGX Spark、Jetson Orin |
| [AMD](https://docs.sglang.io/docs/hardware-platforms/amd_gpu) | Instinct MI300X、MI325X、MI350X、MI355X |
| [Google TPU](https://docs.sglang.io/docs/hardware-platforms/tpu) | v6e、v7；[SGL-JAX](https://github.com/sgl-project/sglang-jax) / [SGL-torchtpu](https://lmsys.org/blog/2026-07-30-sglang-google-tpu/) |
| Intel（[GPU](https://docs.sglang.io/docs/hardware-platforms/xpu) / [CPU](https://docs.sglang.io/docs/hardware-platforms/cpu_server)） | Arc / Arc Pro B-Series GPU、Xeon CPU |
| [Apple Silicon](https://docs.sglang.io/docs/hardware-platforms/apple_metal) | 通过 Metal / MLX 支持 Mac 设备 |
| [华为昇腾 (Huawei Ascend)](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/getting-started/installation) | A2、A3、950PR/DT NPU |
| [摩尔线程 (Moore Threads)](https://docs.sglang.io/docs/hardware-platforms/mthreads_gpu) | MTT S5000 GPU |

正在接入中的硬件：AWS Trainium、[阿里平头哥 PPU](https://github.com/sgl-project/sglang/issues/37519)、[寒武纪 MLU](https://github.com/sgl-project/sglang/pull/26898)、高通 QAIC、沐曦 MetaX、海光 HCU/DCU、天数智芯 Iluvatar CoreX 等。

有关具体模型兼容性与配置方式，请参阅 [Cookbook](https://cookbook.sglang.io/) 及各平台指南。

## SGL 生态系统 (SGL Ecosystem)

| 领域 | 项目 | 用途 |
| --- | --- | --- |
| 教育与学习 | [Mini-SGLang](https://github.com/sgl-project/mini-sglang), [zero-to-sglang](https://github.com/datawhalechina/zero-to-sglang), [DeepLearning.AI 课程](https://www.deeplearning.ai/short-courses/efficient-inference-with-sglang-text-and-image-generation/) | 通过代码实操与系统课程学习推理引擎架构设计以及高效的文本/图像生成技术。 |
| 扩散模型 (Diffusion) | [SGLang Diffusion](https://docs.sglang.io/docs/sglang-diffusion/installation) | SGLang 内置组件，面向扩散模型提供高效的图像与视频生成能力。 |
| 音频处理 | [SGLang Omni](https://github.com/sgl-project/sglang-omni) | 音频模型推理服务，支持文本转语音（TTS）与自动语音识别（ASR）。 |
| 强化学习与后训练 (RL) | [Miles](https://github.com/radixark/miles), [slime](https://github.com/THUDM/slime), [AReaL](https://github.com/inclusionAI/AReaL), [Tunix](https://github.com/google/tunix), [verl](https://github.com/volcengine/verl) | 集成 SGLang 作为大规模 Rollout 生成引擎的后训练框架。 |
| 投机推测解码 | [SpecForge](https://github.com/sgl-project/SpecForge) | 训练投机解码小草稿模型，并配合 SGLang 进行高性能部署。 |
| KV 缓存分层 | [HiCache](https://docs.sglang.io/docs/advanced_features/hicache_design), [Mooncake](https://kvcache-ai.github.io/Mooncake/), [LMCache](https://docs.lmcache.ai/developer_guide/integration.html) | 跨 GPU 显存、主机内存与外部存储的分级 KV 缓存管理，支持分布式推理中的缓存传输与重用。 |
| 部署与集群调度 | [SMG](https://github.com/smg-project/smg), [RBG](https://github.com/sgl-project/rbg), [llm-d](https://llm-d.ai/docs/dev/operations/disaggregation/sglang), [Ray Serve](https://docs.ray.io/en/latest/serve/llm/user-guides/sglang.html), [NVIDIA Dynamo](https://docs.nvidia.com/dynamo/backends/sg-lang/reference-guide) | 具备智能路由、负载均衡与集群编排能力的 SGLang 规模化推理服务部署工具。 |

## 开发与贡献 (Development and Contributing)

我们非常欢迎社区贡献，从缺陷修复、文档补充到新模型支持与性能优化皆可参与。

### 开发环境配置 (Development setup)

建议从 `lmsysorg/sglang:dev` Docker 镜像开始，该镜像预装了开发工具及绝大部分依赖项。在容器内克隆或挂载您的 SGLang 仓库目录，然后从仓库根目录以可编辑模式（editable mode）安装，以确保测试套件直接调用本地 Python 修改：

```bash
pip install -e "python"
```

在已激活的虚拟环境中，您也可以使用 `uv pip install --prerelease=allow -e "python"` 代替。有关容器配置与测试流程，请参阅[开发指南](https://docs.sglang.io/docs/developer_guide/development_guide_using_docker)。

### 参与贡献 (Contribute)

1. Fork 本仓库并为您的更改创建独立分支。对于较大的架构调整，请先在 [GitHub Issue](https://github.com/sgl-project/sglang/issues) 或 [Slack](https://slack.sglang.io/) 中讨论方案。
2. 进行代码修改、运行相关测试，并为修复或新行为添加回归测试覆盖。在提交前请运行 `pre-commit run --all-files`。
3. 提交 Pull Request，清晰描述本次变更及其测试验证方式。如涉及性能优化，请附带基准测试数据或精度评测结果。

代码格式、测试方法和 Pull Request 流程规范请参见[贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)。文档贡献者可从[文档指引](docs/README.md)开始。

## 社区与赞助 (Community and Sponsorship)

SGLang 由非营利开源机构 [LMSYS](https://lmsys.org/about/) 托管。

- **社区交流**：加入 [Slack](https://slack.sglang.io/) 进行技术探讨与开发交流。
- **活动日程**：在 [SGLang 活动日历](https://www.sglang.io/events) 查看 Meetup、技术研讨会与 Office Hours。
- **最新动态**：关注 [X (@lmsysorg)](https://x.com/lmsysorg) 与 [LinkedIn](https://www.linkedin.com/company/sgl-project/) 获取项目动态，阅读 [LMSYS 官方博客](https://lmsys.org/blog/) 查看版本发布公告与深度技术解析。
- **项目资源**：浏览 [官方文档](https://docs.sglang.io/)、[Cookbook](https://cookbook.sglang.io/)、[开发路线图 (Roadmap)](https://roadmap.sglang.io/)、[版本发布记录 (Releases)](https://github.com/sgl-project/sglang/releases)、[Issue 追踪器](https://github.com/sgl-project/sglang/issues) 及 [贡献指南](https://docs.sglang.io/docs/developer_guide/contribution_guide)。
- **商务联络**：企业规模化部署、技术咨询、商业赞助或生态合作，请联络 [sglang@lmsys.org](mailto:sglang@lmsys.org)。
- **贡献者赞助计划**：长期活跃的 SGLang 贡献者有资格获得 AI 编程辅助工具赞助（如 Cursor、Claude Code 或 OpenAI Codex）。申请请向 [sglang@lmsys.org](mailto:sglang@lmsys.org) 发送邮件，并附上您的核心 Commit 或 PR 链接。

## 业界与学术界信任 (Trusted by Industry and Research)

SGLang 已在全球各大顶尖 AI 实验室、云服务平台、领军企业与知名高校的生产环境中承担海量推理负载。

<img src="https://raw.githubusercontent.com/sgl-project/sgl-learning-materials/refs/heads/main/slides/adoption.png" alt="Organizations adopting SGLang" width="800">

## 致谢 (Acknowledgment)
我们借鉴了以下优秀开源项目的设计思想并复用了部分代码：[Guidance](https://github.com/guidance-ai/guidance), [vLLM](https://github.com/vllm-project/vllm), [LightLLM](https://github.com/ModelTC/lightllm), [FlashInfer](https://github.com/flashinfer-ai/flashinfer), [Outlines](https://github.com/outlines-dev/outlines), 以及 [LMQL](https://github.com/eth-sri/lmql)。

---

> 💡 **中文文档维护声明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）协同维护并持续跟踪上游更新，最后同步于 2026年10月07日。若发现翻译疏漏或有最新功能改进建议，欢迎提交 PR 或 Issue 共同完善！

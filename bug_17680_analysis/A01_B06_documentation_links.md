# SGLang 官方文档链接和阅读指南

## 🌐 官方文档网站

**主链接**: https://docs.sglang.ai/

---

## 📚 文档结构（按阅读顺序）

### 1. Get Started（入门）
**链接**: https://docs.sglang.ai/get_started/install.html

**内容**:
- 安装指南
- 快速开始

---

### 2. Basic Usage（基础使用）
**链接**: https://docs.sglang.ai/basic_usage/

**主要内容**:
- `send_request.ipynb` - 发送请求
- `openai_api.rst` - OpenAI API兼容接口
- `offline_engine_api.ipynb` - 离线引擎API
- `native_api.ipynb` - 原生API
- `sampling_params.md` - 采样参数
- `deepseek.md` - DeepSeek模型支持
- `gpt_oss.md` - GPT-OSS模型支持
- `llama4.md` - Llama4模型支持
- `qwen3.md` - Qwen3模型支持

---

### 3. Advanced Features（高级特性）
**链接**: https://docs.sglang.ai/advanced_features/

**主要内容**:
- `server_arguments.md` - 服务器参数配置 ⭐ **重要**
- `hyperparameter_tuning.md` - 超参数调优
- `speculative_decoding.ipynb` - 推测解码
- `structured_outputs.ipynb` - 结构化输出
- `structured_outputs_for_reasoning_models.ipynb` - 推理模型的结构化输出
- `tool_parser.ipynb` - 工具解析器
- `separate_reasoning.ipynb` - 分离推理
- `quantization.md` - 量化 ⭐ **与当前bug相关**
- `lora.ipynb` - LoRA支持
- `pd_disaggregation.md` - Prefill-Decode分离
- `vlm_query.ipynb` - 视觉语言模型查询
- `router.md` - 路由器
- `observability.md` - 可观测性
- `attention_backend.md` - 注意力后端

---

### 4. Supported Models（支持的模型）
**链接**: https://docs.sglang.ai/supported_models/

**主要内容**:
- `generative_models.md` - 生成模型
- `multimodal_language_models.md` - 多模态语言模型
- `embedding_models.md` - 嵌入模型
- `reward_models.md` - 奖励模型
- `rerank_models.md` - 重排序模型
- `support_new_models.md` - 支持新模型 ⭐ **开发者指南**
- `transformers_fallback.md` - Transformers回退
- `modelscope.md` - ModelScope支持

---

### 5. Hardware Platforms（硬件平台）
**链接**: https://docs.sglang.ai/platforms/

**主要内容**:
- `amd_gpu.md` - AMD GPU
- `blackwell_gpu.md` - Blackwell GPU
- `cpu_server.md` - CPU服务器
- `tpu.md` - TPU
- `nvidia_jetson.md` - NVIDIA Jetson
- `ascend_npu.md` - Ascend NPU

---

### 6. Developer Guide（开发者指南）
**链接**: https://docs.sglang.ai/developer_guide/

**主要内容**:
- `contribution_guide.md` - 贡献指南 ⭐ **重要**
- `development_guide_using_docker.md` - Docker开发指南
- `benchmark_and_profiling.md` - 基准测试和分析
- `bench_serving.md` - 服务基准测试

---

### 7. References（参考文档）
**链接**: https://docs.sglang.ai/references/

**主要内容**:
- `faq.md` - 常见问题
- `environment_variables.md` - 环境变量
- `production_metrics.md` - 生产指标
- `multi_node_deployment/` - 多节点部署
- `custom_chat_template.md` - 自定义聊天模板
- `frontend/` - 前端文档
- `learn_more.md` - 了解更多

---

## 🎯 针对当前Bug的推荐阅读顺序

### 第一步：理解基础概念
1. **Get Started**: https://docs.sglang.ai/get_started/install.html
2. **Basic Usage**: https://docs.sglang.ai/basic_usage/send_request.html

### 第二步：理解Tensor Parallelism
1. **Server Arguments**: https://docs.sglang.ai/advanced_features/server_arguments.html
   - 查找 `--tp-size` 相关参数
   - 理解tensor parallelism的配置

### 第三步：理解量化
1. **Quantization**: https://docs.sglang.ai/advanced_features/quantization.html
   - 理解INT4量化
   - 理解量化对权重的影响

### 第四步：理解MoE模型
1. **Supported Models**: https://docs.sglang.ai/supported_models/generative_models.html
   - 查找MoE模型相关文档
   - 理解MoE模型的权重结构

### 第五步：开发者文档
1. **Contribution Guide**: https://docs.sglang.ai/developer_guide/contribution_guide.html
2. **Support New Models**: https://docs.sglang.ai/supported_models/support_new_models.html
   - 理解如何添加新模型
   - 理解权重加载流程

---

## 🔍 关键文档链接（与当前Bug相关）

### Tensor Parallelism相关
- Server Arguments: https://docs.sglang.ai/advanced_features/server_arguments.html
- Multi-node Deployment: https://docs.sglang.ai/references/multi_node_deployment/

### 量化相关
- Quantization: https://docs.sglang.ai/advanced_features/quantization.html

### MoE模型相关
- Generative Models: https://docs.sglang.ai/supported_models/generative_models.html
- Support New Models: https://docs.sglang.ai/supported_models/support_new_models.html

### 开发者文档
- Contribution Guide: https://docs.sglang.ai/developer_guide/contribution_guide.html
- Development Guide: https://docs.sglang.ai/developer_guide/development_guide_using_docker.html

---

## 📖 本地文档（可选）

如果你想在本地查看文档：

```bash
cd docs
# 安装依赖
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt

# 编译文档
make compile
make html

# 本地预览（自动重建）
bash serve.sh
# 或
make serve
```

---

## 🔗 其他有用链接

- **GitHub仓库**: https://github.com/sgl-project/sglang
- **Slack社区**: https://slack.sglang.ai/
- **开发会议**: https://meeting.sglang.ai/
- **博客**: https://lmsys.org/blog/ (搜索sglang相关文章)
- **Roadmap**: https://github.com/sgl-project/sglang/issues/7736

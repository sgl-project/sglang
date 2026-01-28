# A01_B11: Diffusion 模型 Parameter 参数问题

## 问题描述

用户反馈：**今天试了这么多个模型，没一个可以用 sglang 的 parameter 跑过去的，diffusion 模型都不行**

## 问题分析

### 可能的问题

1. **`sglang generate` 命令可能不支持 `--parameter` 参数**
   - `sglang generate` 命令可能不是一个独立的 CLI 命令
   - 可能是通过 `sglang launch_server` 启动服务器，然后通过 API 调用生成

2. **Diffusion 模型可能不支持某些参数**
   - Diffusion 模型和 LLM 模型的参数可能不同
   - Diffusion 模型可能需要特定的参数格式

3. **参数传递方式问题**
   - 可能需要通过 API 调用而不是 CLI 参数
   - 可能需要使用不同的参数名称

## 需要确认的信息

1. **用户使用的具体命令是什么？**
   - 是否使用了 `sglang generate --parameter ...`？
   - 还是通过 API 调用传递参数？

2. **具体的错误信息是什么？**
   - 参数不被识别？
   - 参数格式错误？
   - 运行时错误？

3. **用户尝试了哪些模型？**
   - 哪些 diffusion 模型？
   - 哪些其他模型？

## 可能的解决方案

### 方案 1: 使用 API 调用而不是 CLI 参数

如果 `sglang generate` 不支持 `--parameter`，可能需要：

1. **启动服务器**：
   ```bash
   sglang launch_server --model-path <model-path> --backend diffusers
   ```

2. **通过 API 调用传递参数**：
   ```python
   import requests
   
   response = requests.post(
       "http://localhost:30000/generate",
       json={
           "text": "your prompt",
           "sampling_params": {
               "temperature": 0.7,
               "max_new_tokens": 100,
               # 其他参数...
           }
       }
   )
   ```

### 方案 2: 检查 Diffusion 模型支持的参数

Diffusion 模型可能需要特定的参数格式，需要查看：
- `sglang.multimodal_gen` 模块的文档
- Diffusion 模型的具体实现
- 支持的参数列表

### 方案 3: 使用正确的参数名称

可能需要使用：
- `--sampling-params` 而不是 `--parameter`
- 或者通过配置文件传递参数
- 或者通过环境变量传递参数

## 下一步行动

1. **收集更多信息**：
   - 用户的具体命令
   - 错误信息
   - 尝试的模型列表

2. **检查代码**：
   - `sglang generate` 命令的实现
   - Diffusion 模型参数处理逻辑
   - API 参数传递方式

3. **测试验证**：
   - 使用不同的参数格式测试
   - 测试不同的 diffusion 模型
   - 验证参数传递是否正确

## 相关文档

- [A01_B09_issue_reply_to_kevin.md](./A01_B09_issue_reply_to_kevin.md) - 之前的测试结果
- [A01_B08_reproduction_results.md](./A01_B08_reproduction_results.md) - 复现测试结果
- [A01_B01_original_issue.md](./A01_B01_original_issue.md) - 原始问题

## 待确认

- [ ] 用户使用的具体命令
- [ ] 错误信息
- [ ] 尝试的模型列表
- [ ] `sglang generate` 是否支持 `--parameter` 参数
- [ ] Diffusion 模型支持的参数格式

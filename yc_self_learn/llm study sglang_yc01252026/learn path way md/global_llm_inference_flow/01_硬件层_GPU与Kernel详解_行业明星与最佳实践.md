# 硬件层（GPU / Kernel）详解：行业明星与最佳实践

## 📚 文档说明

**本文档位于**: `global_llm_inference_flow/`  
**父文档**: [00_LLM推理全局架构_从GPU到客户使用.md](./00_LLM推理全局架构_从GPU到客户使用.md)（第 ⑤ 层展开）

**本页回答**：
- 硬件层具体在做什么、关键概念与瓶颈
- 做得最好的公司和案例是谁
- **谁是这个行业的 super star（唯一的 star）**

---

## 一、硬件层在做什么（细节）

### 1.1 职责再拆解

| 职责 | 说明 | 典型实现 |
|------|------|----------|
| **算力** | 矩阵乘 (GEMM)、Attention 中的 QK^T、softmax、×V | Tensor Core、cuBLAS、自定义 Triton/CUDA kernel |
| **高带宽存储** | 存权重、激活、KV Cache，减少与 DRAM 的搬运 | HBM、L2/L1 cache、寄存器 |
| **Kernel 执行** | 把「一层/一块」计算写成 GPU 上可并行的 kernel，减少 launch 与读写 | FlashAttention、LayerNorm/ RMSNorm 融合、FFN 融合 |

**本质**：在**算力**和**带宽**两个约束下，用**更少的 kernel、更少的内存读写**，完成同样的 Attention / FFN / Norm 计算。

### 1.2 关键瓶颈（为什么这一层重要）

- **Attention**：序列长 N 时，朴素实现 O(N²) 的 HBM 读写，容易把算力「饿死」。
- **KV Cache**：Decode 时每个 token 都要读整段 KV，访问模式与分页方式直接决定带宽利用率。
- **Kernel 数量**：launch 太多小 kernel 会浪费算力、增加延迟。

所以这一层的「明星」= 谁把 Attention/KV 访问做成**省带宽、高利用率、可复用**的 kernel 与库。

### 1.3 典型技术栈

- **CUDA / Triton**：写 kernel 的语言与运行时。
- **FlashAttention / FlashInfer**：Attention 专用 kernel（tiling、融合、Paged KV）。
- **cuBLAS / cuDNN**：通用 GEMM、部分融合。
- **Tensor Core**：硬件单元，做低精度矩阵乘。
- **NVLink / NVSwitch**：多卡间高带宽，影响 PP/TP 的通信成本。

---

## 二、做得最好的公司与案例

### 2.1 公司维度

| 公司 | 角色 | 典型产品/贡献 |
|------|------|----------------|
| **NVIDIA** | 硬件 + 全栈 | A100/H100、CUDA、TensorRT-LLM、Triton Inference Server、Triton 语言 |
| **Together AI** | 推理 + 研究 | 雇佣 Tri Dao，发布 FlashAttention-2/3，推理服务与开源 |
| **Dao-AILab** | 开源 kernel | FlashAttention 系列（GitHub: Dao-AILab/flash-attention） |
| **FlashInfer 团队** | 推理专用 kernel 库 | FlashInfer：Paged KV、Decode/Prefill 优化，被 SGLang、vLLM、MLC 等采用 |
| **LMSYS** | 推理引擎 | SGLang（用 FlashInfer 等 backend），RadixAttention、前端 API |

**做得好**的体现：谁定义了「Attention 该怎么算才省带宽」、谁被各大推理引擎默认集成。

### 2.2 最佳案例（可对号入座）

1. **FlashAttention 1/2/3（Tri Dao 等）**  
   - **做了什么**：Tiling + 重计算 + 融合，把 Attention 的 HBM 读写压下去，逼近理论算力。  
   - **结果**：2–4× 加速（FA1），再 2×（FA2），FA3 在 H100 上 75% 理论 FLOPS、FP8 近 1.2 PFLOPS。  
   - **地位**：**事实上的行业标准**，vLLM、SGLang、TGI、TensorRT-LLM、PyTorch、DeepSpeed 等都用或借鉴。

2. **FlashInfer**  
   - **做了什么**：面向 **LLM 推理** 的 Attention 引擎，Paged KV、可组合格式、JIT 模板，与 SGLang/vLLM 深度集成。  
   - **结果**：相比部分 compiler backend 有 29–69% 的 inter-token latency 降低、长上下文 28–30% 延迟降低。  
   - **地位**：当前**推理服务**里最主流的专用 kernel 库之一。

3. **TensorRT-LLM（NVIDIA）**  
   - **做了什么**：编译期融合、量化、Paged Attention、In-flight batching、多卡部署。  
   - **结果**：在 NVIDIA 生态内（H100 + Triton）达到很高吞吐与延迟。  
   - **地位**：**闭源/生态内**的「官方最优解」，与开源引擎（SGLang/vLLM）在不同场景竞争。

4. **vLLM / SGLang**  
   - **做了什么**：推理引擎，**使用** FlashInfer/FlashAttention 等作为 kernel 层，在上层做调度、批处理、RadixAttention。  
   - **地位**：不是「发明 kernel」的，但是**把 kernel 用到生产**的标杆；你学的 SGLang 就在这一层之上。

---

## 三、谁是这个行业的 Super Star（唯一的 Star）

### 3.1 结论先说

**个人维度：Tri Dao。**

- 定义了「用 tiling + 重计算 + 融合」把 Attention 从带宽瓶颈里救出来的**范式**（FlashAttention）。  
- FlashAttention 2/3 持续把上限推高，被全行业采用；他本人现为 Together AI 首席科学家，并即将任 Princeton 助理教授。  
- 在 **LLM 推理/训练** 的 **Kernel 层**，没有第二个人有同等级别的「一人定义标准、全行业跟进」的叙事。

所以：**若只选一个「唯一的 star」，选 Tri Dao。**

### 3.2 为什么不是别人？

- **NVIDIA**：是**平台**（硬件 + CUDA + TensorRT-LLM），是「舞台」，不是「一个明星演员」；且 TensorRT-LLM 的很多思想与 FlashAttention 同源或兼容。  
- **FlashInfer 等**：是**工程化、产品化**的明星项目，但学术与工业影响力的「范式开创者」仍是 FlashAttention（Tri Dao）。  
- **vLLM / SGLang 作者**：是**系统与调度**层的明星，不是**硬件/Kernel** 层的「唯一 star」。

### 3.3 一句话记忆

**「硬件层谁说了算？—— 算 Attention 怎么算、省带宽怎么省，是 Tri Dao 的 FlashAttention 定的调；推理服务里谁在用？—— FlashInfer + SGLang/vLLM；硬件与平台是谁？—— NVIDIA。」**

---

## 四、和你学的 SGLang 的关系

- SGLang **不发明** GPU Kernel，而是**选用** FlashInfer（以及 Triton、FA3/FA4 等）作为 Attention backend。  
- 文档 [Case_Study_06_SGLang_KV_Cache_与_FlashInfer_关系](../../casestudy/Case_Study_06_SGLang_KV_Cache_与_FlashInfer_关系.md) 里讲的「SGLang 管内存，FlashInfer 管计算」，就是：**硬件层（Kernel）由 FlashInfer 等提供，SGLang 在上一层管调度与 KV 分配。**  
- 理解「硬件层唯一的 star 是 Tri Dao / FlashAttention」，有助于你在看 SGLang 的 `flashinfer_backend.py` 时，知道上游的「行业标准」从哪来。

---

## 五、延伸阅读与链接

- FlashAttention: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)  
- FlashInfer: [flashinfer.ai](https://flashinfer.ai/) / [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)  
- Together AI 介绍 Tri Dao 与 FlashAttention-2: [Together AI Blog - Tri Dao](https://www.together.ai/blog/tri-dao-flash-attention)  
- PyTorch 博客 FlashAttention-3: [PyTorch Blog - FlashAttention-3](https://pytorch.org/blog/flashattention-3)  
- TensorRT-LLM: [NVIDIA TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/)

---

*文档版本：v1 | 与 00_LLM推理全局架构 配套使用。*

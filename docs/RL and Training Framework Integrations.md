# RL and Training Framework Integrations

SGLang has become the de facto inference backend for modern LLM training frameworks, powering state-of-the-art models across the industry. From GLM-4.5 to DeepSeek-V3, leading models leverage SGLang's high-performance inference during reinforcement learning and post-training workflows. 

What makes SGLang essential for training? Beyond industry-leading throughput via RadixAttention, SGLang offers specialized features critical for production training pipelines: **fast weight synchronization** for seamless model updates between training and inference, and **deterministic inference** ensuring zero mismatch between rollout and evaluation. These capabilities, combined with native integration support across major frameworks, have established SGLang as the infrastructure backbone for modern LLM alignment.

---

## Projects

- [**MILES**](https://github.com/radixark/miles): Enterprise-scale RL framework for large MoE models with SGLang-native rollout, speculative training, and production-grade stability
- [**slime**](https://github.com/THUDM/slime): Post-training framework combining Megatron and SGLang, used to train GLM-4.5
- [**AReaL**](https://github.com/inclusionAI/AReaL): Fully asynchronous RL system achieving 2.77x speedup with SGLang backend for continuous rollout generation
- [**VERL**](https://github.com/volcengine/verl): Full-stack RLHF framework supporting PPO, GRPO, and ReMax with modular SGLang integration
- [**Unsloth**](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide): 2x faster fine-tuning with optimized kernels, deploys seamlessly with SGLang inference
- [**LLaMAFactory**](https://github.com/hiyouga/LLaMA-Factory): Unified framework for training 100+ LLMs with LoRA, QLoRA, and full fine-tuning methods
- [**Tunix**](https://github.com/google/tunix): Google's JAX-native library for LLM post-training with SFT, DPO, PPO, and GRPO support

---

## Collaboration

Interested in integrating SGLang with your training framework or need technical support? We're here to help! Reach out to us at **sglang@lmsys.org** for partnerships, integration guidance, and custom feature development.

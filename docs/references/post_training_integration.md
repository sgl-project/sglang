# Post-Training Integration

SGLang has become the de facto inference backend for modern LLM training frameworks, powering state-of-the-art models across the industry. From GLM-4.6 to Qwen3, leading models leverage SGLang's high-performance inference during reinforcement learning and post-training workflows.

What makes SGLang essential for post-training?

- Open-To-Use Refit Functionality: diverse method for colocate or disaggregate
- Easy To Postpone Generation: enable partial rollout and dedicated rollout control
- Fine-Grained Engine Sleep And Wake Up: facilitate maxium-powered rollout and training
- Training Serving Alignment: ensure the performance consistency in training and serving
- Load Balancing Router: cache-aware load-balancing for high-throughput rollout
- Deterministic Inference: ensure zero kl divergence between rollout and training

These capabilities, combined with native integration support across major frameworks, have established SGLang as the infrastructure backbone for modern LLM/VLMs post-training. We also share our latest work in this slide, [Optimizing Large-Scale RL with SGLang](https://gamma.app/docs/Optimizing-RL-with-SGLang-y0kqgj877k34779).

## Adoption

- [**Miles**](https://github.com/radixark/miles): Enterprise-scale RL framework for large MoE models with SGLang-native rollout, speculative training, and production-grade stability
- [**slime**](https://github.com/THUDM/slime): Post-training framework combining Megatron and SGLang, used to train GLM-4.6
- [**AReaL**](https://github.com/inclusionAI/AReaL): Fully asynchronous RL system achieving 2.77x speedup with SGLang backend for continuous rollout generation
- [**ROLL**](https://github.com/alibaba/ROLL): ROLL is an efficient and user-friendly RL library designed for Large Language Models utilizing Large Scale GPU resources
- [**verl**](https://github.com/volcengine/verl): Full-stack RLHF framework supporting PPO, GRPO, and ReMax with modular SGLang integration
- [**Unsloth**](https://docs.unsloth.ai/basics/inference-and-deployment/sglang-guide): 2x faster fine-tuning with optimized kernels, deploys seamlessly with SGLang inference
- [**LLaMA Factory**](https://github.com/hiyouga/LLaMA-Factory): Unified framework for training 100+ LLMs with LoRA, QLoRA, and full fine-tuning methods
- [**Tunix**](https://github.com/google/tunix): Google's JAX-native library for LLM post-training with SFT, DPO, PPO, and GRPO support
- [**RL2**](https://github.com/ChenmienTan/RL2): Ray Less Reinforcement Learning, a concise library of post-training for large language models


## Collaboration

Due to the privacy of the design parternes, we cannot list the companies that adopt SGLang for post-training. However, we are happy to share the details with you if you are interested and trust the choice among 10+ top companies and frontier labs across US and China. If you are interested in integrating SGLang with your training framework or need technical support, we're here to help! Reach out to us at **rl_team@lmsys.org** for partnerships, integration guidance, and custom feature development.

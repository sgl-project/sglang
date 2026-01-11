import os
import glob
import asyncio
import re
from openai import AsyncOpenAI

# ================= ⚡️ 配置区域 =================
# 1. 填入你之前验证成功的配置
REMOTE_HOST = "https://api.agentify.top/v1" 
API_KEY = "sk-MzagDUFUavhiE1ieVOHk5V1KAy7eXF0ZXfnn6IkPRwb9185P"
MODEL_NAME = "openai/gpt-oss-120b" 

# 并发限制 (防止 API 429)
CONCURRENCY_LIMIT = 5

client = AsyncOpenAI(base_url=REMOTE_HOST, api_key=API_KEY)

# ================= 🧠 核心审计逻辑 =================

async def audit_file(sem, file_path):
    async with sem:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except: return None

        # 1. 预筛: 只看包含 Diffusion/LLaDA 关键词的文件
        if not re.search(r"(llada|diffusion|denoise|mask|block|tp_rank)", code, re.IGNORECASE):
            return None

        print(f"🧪 正在审计 dLLM 核心代码: {file_path}")

        # 2. 高维 Prompt: 注入 LLaDA 和 分布式并行的先验知识
        prompt = f"""
        You are an AI Math & Systems Architect auditing the new LLaDA (Large Language Diffusion with Masking) implementation in SGLang.
        
        Target File: {file_path}
        
        **Context**: LLaDA is a Masked Diffusion Model. Unlike Auto-Regressive models, it generates blocks of tokens in parallel using a **Rectangular Attention Mask** (Bidirectional). It requires iterative denoising (re-masking).
        
        **Your Goal**: Find Logic Bugs that break mathematical correctness or distributed consistency.
        
        **Focus Areas**:
        
        1.  **Attention Mask Logic**:
            - Does the code correctly implement the Rectangular Mask for Block Diffusion? 
            - If it reuses the Causal Mask (Triangular) from standard LLMs, THIS IS A BUG.
            
        2.  **Tensor Parallel (TP) Randomness Sync**:
            - LLaDA relies on random masking ratio `t ~ U(0, 1)`.
            - In TP mode, `torch.rand` or `torch.randn` MUST use a synchronized generator seed across all GPUs.
            - If `torch.rand()` is called without handling TP rank synchronization, outputs will diverge. THIS IS A CRITICAL BUG.
            
        3.  **KV Cache Indexing**:
            - Block diffusion writes to non-contiguous slots or overwrites existing slots during re-masking.
            - Check if `paged_attention` indices are calculated correctly. Are we appending when we should be overwriting?
            
        4.  **Numerical Stability**:
            - Look for division by `std` or `sigma`. Is there an epsilon (`+ 1e-6`) to prevent NaN?
            - Check for `exp()` or `log()` on potentially zero/negative values in the noise scheduler.

        **Output Format**:
        - If SAFE: Output only "SAFE".
        - If RISKY:
        [Severity: HIGH/CRITICAL]
        [Line Number]
        [Issue Description]
        [Mathematical/Code Fix Suggestion]
        """

        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1, # 极低温度，数学容不得幻觉
            )
            return (file_path, res.choices[0].message.content)
        except Exception as e:
            return (file_path, f"API ERROR: {str(e)}")

async def main():
    report_file = "DLLM_AUDIT_REPORT.md"
    
    # 🎯 目标锁定：SGLang 的 Python 模型定义目录
    target_files = set()
    
    # 策略 A: 扫描 models 目录 (寻找 llada.py)
    target_files.update(glob.glob("python/sglang/srt/models/**/*llada*.py", recursive=True))
    target_files.update(glob.glob("python/sglang/srt/models/**/*diffusion*.py", recursive=True))
    
    # 策略 B: 扫描 layers 目录 (寻找 attention mask 实现)
    target_files.update(glob.glob("python/sglang/srt/layers/**/*.py", recursive=True))
    
    # 策略 C: 扫描 scheduler (寻找 diffusion scheduler)
    target_files.update(glob.glob("python/sglang/srt/managers/**/*.py", recursive=True))

    target_files = list(target_files)
    print(f"🚀 开始审计 dLLM 相关代码 ({len(target_files)} files)...")

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [audit_file(sem, f) for f in target_files]
    
    # 显示进度
    results = await asyncio.gather(*tasks)
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# SGLang dLLM (LLaDA) Security & Logic Audit\n\n")
        
        hit_count = 0
        for item in results:
            if item and "SAFE" not in item[1] and "API ERROR" not in item[1]:
                path, content = item
                hit_count += 1
                print(f"🔥 [潜在逻辑漏洞] {path}")
                f.write(f"## 📂 {path}\n{content}\n\n---\n")
                f.flush()

    print(f"\n✅ 审计完成！共发现 {hit_count} 个潜在问题。")
    print(f"👉 请查看报告: {report_file}")

if __name__ == "__main__":
    asyncio.run(main())
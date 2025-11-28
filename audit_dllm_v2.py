import os
import re
import asyncio
from openai import AsyncOpenAI

# ================= ⚡️ 配置区域 =================
# 填入你之前验证成功的配置
REMOTE_HOST = "https://api.agentify.top/v1" 
API_KEY = "sk-MzagDUFUavhiE1ieVOHk5V1KAy7eXF0ZXfnn6IkPRwb9185P"
MODEL_NAME = "openai/gpt-oss-120b" 

client = AsyncOpenAI(base_url=REMOTE_HOST, api_key=API_KEY)

# ================= 🔍 猎杀规则 (Fingerprints) =================
# 针对 LLaDA Block Diffusion 的特定漏洞指纹
RULES = {
    "UNSYNC_RANDOM": {
        "pattern": r"torch\.rand(n|_like)?\(",
        "context": 10,
        "desc": "Unsynchronized random sampling in TP mode"
    },
    "CAUSAL_MASK_REUSE": {
        "pattern": r"torch\.tril\(",
        "context": 8,
        "desc": "Incorrect Causal Mask used for Block Diffusion (Rectangular needed)"
    },
    "KV_CACHE_APPEND": {
        "pattern": r"(kv_cache|k_cache|v_cache).*(\.append|cat\()",
        "context": 12,
        "desc": "Incorrect KV Cache append during re-masking (Should overwrite)"
    },
    "UNSAFE_DIVISION": {
        "pattern": r"\/.*sigma",  # 查找除以 sigma 的操作
        "context": 5,
        "desc": "Potential Division by Zero (missing epsilon)"
    }
}

# 锁定目标目录 (dLLM 核心区)
TARGET_DIRS = [
    "python/sglang/srt/models",
    "python/sglang/srt/layers/attention",
    "python/sglang/srt/managers"
]

async def analyze_snippet(sem, file_path, line_num, code_snippet, rule_key):
    async with sem:
        # 构造极其具体的 Prompt
        prompt = f"""
        You are a Senior AI Systems Engineer auditing SGLang's new Diffusion Model (LLaDA) implementation.
        
        **Context**: 
        - File: `{file_path}` (Line {line_num})
        - Issue Type: {RULES[rule_key]['desc']}
        - Architecture: LLaDA uses Block Diffusion (Bidirectional Attention within blocks) and requires Tensor Parallel (TP) consistency.
        
        **Code Snippet**:
        ```python
        {code_snippet}
        ```
        
        **Task**: 
        Analyze ONLY this snippet. 
        1. If this is `torch.rand`, is there a `generator` argument derived from a TP-synchronized seed?
        2. If this is `torch.tril`, is it being used for `attention_mask`? If so, it breaks LLaDA.
        3. If this is `kv_cache.append`, is it inside a diffusion loop where it should overwrite instead?
        
        **Output**:
        - If BUG: Start with "[CONFIRMED BUG]". Explain why briefly.
        - If SAFE: Start with "[SAFE]".
        """

        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return res.choices[0].message.content
        except:
            return None

def extract_snippets(file_path):
    """法医级切片：只提取命中规则的代码段"""
    snippets = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except: return []

    # 只扫描包含 diffusion/llada 的文件，或者核心 layers
    content = "".join(lines).lower()
    if "diffusion" not in content and "llada" not in content and "layer" not in file_path:
        return []

    for i, line in enumerate(lines):
        for rule_key, rule in RULES.items():
            if re.search(rule['pattern'], line):
                # 提取上下文
                start = max(0, i - rule['context'])
                end = min(len(lines), i + rule['context'] + 1)
                snippet = "".join(lines[start:end])
                snippets.append((i + 1, snippet, rule_key))
                break # 一行只报一次
    return snippets

async def main():
    report_file = "DLLM_V2_AUDIT_REPORT.md"
    print(f"🚀 SGLang dLLM Audit V2.0 (Fingerprint-Based)...")
    
    all_snippets = []
    # 1. 扫描文件
    for d in TARGET_DIRS:
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    hits = extract_snippets(path)
                    for hit in hits:
                        all_snippets.append((path, *hit))

    print(f"🔍 提取到 {len(all_snippets)} 个可疑代码片段。开始 AI 会诊...")

    if not all_snippets:
        print("⚠️ 未发现可疑片段。可能目录不对或代码已更新。")
        return

    sem = asyncio.Semaphore(10) # 并发 10
    
    # === 🔴 删除原来的 task_map 复杂逻辑，直接用下面这一行 ===
    tasks = [analyze_snippet(sem, p, l, s, r) for p, l, s, r in all_snippets]
    
    # 并发执行并收集结果
    results = await asyncio.gather(*tasks)

    # 3. 写入报告
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"# Audit Report (Total Checked: {len(results)})\n\n")
        
        for i, analysis in enumerate(results):
            # 这里的 i 和 all_snippets 的索引是一一对应的
            if analysis and "[CONFIRMED BUG]" in analysis:
                path, line, snippet, rule = all_snippets[i]
                print(f"🔥 [实锤] {path}:{line} - {rule}")
                f.write(f"## 🚨 {rule} in `{path}` : {line}\n")
                f.write(f"```python\n{snippet}\n```\n")
                f.write(f"> {analysis}\n\n---\n")

    print(f"\n✅ 审计结束！请查看 {report_file}")
    
if __name__ == "__main__":
    asyncio.run(main())
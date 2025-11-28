import os
import ast
import asyncio
import json
import re
import subprocess
import sys
from openai import AsyncOpenAI

# ================= ⚡️ 配置区域 =================
# 填入你之前验证成功的配置
REMOTE_HOST = "https://api.agentify.top/v1"
API_KEY = "sk-MzagDUFUavhiE1ieVOHk5V1KAy7eXF0ZXfnn6IkPRwb9185P"  # <--- 确认Key是否正确
MODEL_NAME = "openai/gpt-oss-120b"

client = AsyncOpenAI(base_url=REMOTE_HOST, api_key=API_KEY)
CONCURRENCY_LIMIT = 5 

# ================= 🛠️ 核心功能模块 =================

def extract_snippets(file_path):
    """基于 AST 的智能切片"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        lines = content.splitlines()
        snippets = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start = node.lineno - 1
                end = node.end_lineno
                func_source = "\n".join(lines[start:end])
                
                # 关键词过滤: 包含 Triton 或 Image 处理逻辑
                if any(k in func_source for k in ["@triton.jit", "resize", "interpolate", "unsafe", ".unwrap()"]):
                    snippets.append(func_source)
        return snippets
    except:
        return []

def verify_poc(poc_code):
    """自动化验证器"""
    poc_filename = "temp_poc.py"
    with open(poc_filename, "w") as f:
        f.write(poc_code)
    try:
        result = subprocess.run(
            [sys.executable, poc_filename], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode != 0: return True, result.stderr
        return False, "Execution successful"
    except subprocess.TimeoutExpired:
        return True, "Timeout"
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(poc_filename): os.remove(poc_filename)

async def audit_snippet(sem, snippet, file_path):
    async with sem:
        if "@triton.jit" in snippet:
            mode = "TRITON"
            prompt = f"""
            Role: Triton Optimization Expert.
            Target Code:
            ```python
            {snippet}
            ```
            Audit for:
            1. **Int32 Overflow**: `pid * BLOCK` without cast.
            2. **Shared Mem OOM**: High `num_stages`.
            Output JSON: {{ "risk": "HIGH", "reason": "...", "poc_hint": "..." }} or "SAFE".
            """
        elif "resize" in snippet or "interpolate" in snippet:
            mode = "CV_LOGIC"
            prompt = f"""
            Role: QA Engineer.
            Target Code:
            ```python
            {snippet}
            ```
            Audit Image Processing logic for:
            1. **Division by Zero** (width=0).
            2. **Type Mismatch** (float index).
            CRITICAL: Write a short Python script (PoC) to trigger the bug.
            Output the Python code block ONLY.
            """
        else:
            return None

        try:
            res = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return (mode, res.choices[0].message.content)
        except:
            return None

async def main():
    print("🚀 SGLang Auditor V10.0 Starting...")
    
    target_files = []
    # 扫描核心目录
    for d in ["python/sglang/srt/models", "python/sglang/srt/layers", "python/sglang/srt/managers"]:
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(".py") and "test" not in file:
                    target_files.append(os.path.join(root, file))
    
    print(f"🔍 扫描到 {len(target_files)} 个文件...")
    
    snippets_to_audit = []
    for f in target_files:
        snips = extract_snippets(f)
        for s in snips:
            snippets_to_audit.append((f, s))
            
    print(f"🎯 提取到 {len(snippets_to_audit)} 个高危片段。开始审计...")
    
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [audit_snippet(sem, s, f) for f, s in snippets_to_audit]
    results = await asyncio.gather(*tasks)
    
    with open("FINAL_AUDIT_V10.md", "w", encoding="utf-8") as report:
        report.write("# SGLang V10 Audit Report\n\n")
        for i, res in enumerate(results):
            if not res: continue
            mode, content = res
            file_path = snippets_to_audit[i][0]
            
            if "```python" in content:
                poc_code = content.split("```python")[1].split("```")[0].strip()
                print(f"⚡ [验证中] {file_path} ...")
                is_crash, log = verify_poc(poc_code)
                if is_crash:
                    print(f"   🔥 [实锤 Bug!] PoC 触发崩溃！")
                    report.write(f"## 🚨 Verified Bug in `{file_path}`\n**Crash Log:**\n```\n{log}\n```\n**Reproduction:**\n```python\n{poc_code}\n```\n\n")
            elif "HIGH" in content:
                print(f"   ⚠️ [高危警告] {file_path}")
                report.write(f"## ⚠️ Risk in `{file_path}`\n{content}\n\n")

    print(f"\n✅ 审计结束！请查看 FINAL_AUDIT_V10.md")

if __name__ == "__main__":
    asyncio.run(main())

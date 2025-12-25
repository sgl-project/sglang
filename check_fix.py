import sys
import os
import torch

# 1. 强制 Python 优先加载当前目录下的代码
# 假设你在 ~/sglang 目录下，代码在 python/sglang/...
sys.path.insert(0, os.path.abspath("python"))

try:
    # 2. 尝试导入你修改过的模块
    from sglang.srt.models.deepseek_janus_pro import resample_patch_embed
    print("✅ 成功导入 sglang.srt.models.deepseek_janus_pro")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保你在 sglang 项目根目录下运行此脚本")
    sys.exit(1)

def test_fix():
    print("\n--- 开始验证修复 ---")
    
    # 构造一个 dummy 输入
    patch = torch.randn(1, 1, 16, 16)
    
    # 测试 1: 宽度为 0 (之前会报 Internal resize error)
    try:
        print("测试用例 1: new_size=[16, 0]")
        resample_patch_embed(patch, new_size=[16, 0])
        print("❌ 失败: 代码没有拦截错误！")
    except ValueError as e:
        print(f"✅ 成功拦截: 捕获到预期异常 -> {e}")
    except Exception as e:
        print(f"❌ 失败: 捕获到了意外异常 -> {type(e).__name__}: {e}")

    # 测试 2: 浮点数 (之前会报 TypeError)
    try:
        print("\n测试用例 2: new_size=[16.0, 16.0]")
        # 你的修复逻辑是自动转 int，所以这里应该能跑通，或者报错但不是 TypeError
        res = resample_patch_embed(patch, new_size=[16.0, 16.0])
        print("✅ 成功: 代码自动处理了浮点数，未崩溃！")
    except ValueError as e:
         print(f"✅ 成功拦截: {e}")
    except Exception as e:
        print(f"❌ 失败: 依然报错 -> {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_fix()

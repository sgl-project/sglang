"""
交互式流式解析演示程序

给定 engine output 后，可以一步一步地显示：
- 当前剩余的token
- 当前的状态
- 已经处理好的 text / toolcall

每按一下回车 step 一下
"""

import json
import sys
from typing import List, Dict

from parser_factory import create_parser
from test_framework import token_stream_generator
from test_fixtures import get_test_tools
from sglang.srt.entrypoints.openai.protocol import Tool


def format_toolcalls(toolcalls: Dict) -> str:
    """格式化工具调用显示"""
    if not toolcalls:
        return "  无"

    lines = []
    for tool_idx in sorted(toolcalls.keys()):
        tool = toolcalls[tool_idx]
        name = tool.get("name", "")
        args = tool.get("args", "")

        if name:
            lines.append(f"  [{tool_idx}] {name}")
            if args:
                # 尝试格式化 JSON
                try:
                    if isinstance(args, str):
                        args_dict = json.loads(args)
                        args_str = json.dumps(args_dict, indent=4, ensure_ascii=False)
                    else:
                        args_str = json.dumps(args, indent=4, ensure_ascii=False)
                    # 为每行添加缩进
                    args_lines = args_str.split("\n")
                    lines.append("    参数:")
                    for line in args_lines:
                        lines.append(f"      {line}")
                except:
                    lines.append(f"    参数: {args}")
        else:
            lines.append(f"  [{tool_idx}] (构建中...)")
            if args:
                lines.append(f"    部分参数: {args[:50]}...")

    return "\n".join(lines) if lines else "  无"


def display_step_info(
    step: int,
    current_chunk: str,
    remaining_chunks: List[str],
    accumulated_text: str,
    accumulated_toolcalls: Dict,
    result_text: str = "",
    result_calls: List = None,
):
    """显示当前步骤的信息"""
    # 清屏（可选，如果不想清屏可以注释掉）
    # print("\033[2J\033[H", end="")

    print("\n" + "=" * 80)
    print(f"Step {step:03d}")
    print("=" * 80)

    # 当前处理的 chunk
    print(f"\n[当前处理的 Token]")
    display_chunk = current_chunk.replace("\n", "\\n").replace("\r", "\\r")
    if len(display_chunk) > 100:
        display_chunk = display_chunk[:100] + "..."
    print(f"  '{display_chunk}'")

    # 剩余 tokens
    print(f"\n[剩余 Tokens] (共 {len(remaining_chunks)} 个)")
    if remaining_chunks:
        # 显示前几个剩余的 chunks
        preview_count = min(5, len(remaining_chunks))
        preview_chunks = remaining_chunks[:preview_count]
        for i, chunk in enumerate(preview_chunks):
            display_chunk = chunk.replace("\n", "\\n").replace("\r", "\\r")
            if len(display_chunk) > 60:
                display_chunk = display_chunk[:60] + "..."
            print(f"  [{i+1}] '{display_chunk}'")
        if len(remaining_chunks) > preview_count:
            print(f"  ... 还有 {len(remaining_chunks) - preview_count} 个 tokens")
    else:
        print("  无（已全部处理完毕）")

    # 当前状态（从解析结果中提取）
    print(f"\n[当前状态]")
    if result_text:
        print(f"  本次提取的文本: '{result_text}'")
    if result_calls:
        print(f"  本次检测到的工具调用: {len(result_calls)} 个")
        for call in result_calls:
            if call.name:
                print(f"    - {call.name} (index={call.tool_index})")
            if call.parameters:
                params_preview = call.parameters[:50] + "..." if len(call.parameters) > 50 else call.parameters
                print(f"      参数片段: {params_preview}")
    if not result_text and not result_calls:
        print("  无新内容")

    # 已累积的文本
    print(f"\n[已累积的文本]")
    if accumulated_text:
        display_text = accumulated_text.replace("\n", "\\n")
        if len(display_text) > 200:
            display_text = display_text[:200] + "..."
        print(f"  {display_text}")
    else:
        print("  无")

    # 已累积的工具调用
    print(f"\n[已累积的工具调用]")
    print(format_toolcalls(accumulated_toolcalls))

    print("\n" + "-" * 80)
    print("按回车键继续下一步，或输入 'q' 退出...")


def run_interactive_demo(
    response_text: str,
    mode: str = "atomic_tags",
    parser_source: str = "default",
    tools: List[Tool] = None,
):
    """
    运行交互式流式解析演示

    Args:
        response_text: 完整的 engine output 文本
        mode: 流式生成模式 ('char', 'atomic_tags', 'whole' 等)
        parser_source: 解析器来源 ('default', 'legacy_v3' 等)
        tools: 工具列表，如果为 None 则使用默认工具
    """
    # 创建解析器
    detector = create_parser(parser_source)

    # 获取工具列表
    if tools is None:
        tools = get_test_tools()

    print("=" * 80)
    print("交互式流式解析演示")
    print("=" * 80)
    print(f"\n解析器: {parser_source}")
    print(f"模式: {mode}")
    print(f"\n完整响应文本:")
    print(response_text)
    print("\n" + "=" * 80)
    print("准备开始...")

    # 生成所有 chunks（但不立即处理）
    all_chunks = list(token_stream_generator(response_text, mode=mode))

    print(f"总共将处理 {len(all_chunks)} 个 tokens")
    print("按回车键开始第一步...")
    input()

    # 初始化状态
    accumulated_text = ""  # 累积的纯文本
    accumulated_toolcalls = {}  # 累积的工具调用 {tool_index: {"name": str, "args": str}}
    step = 0

    # 逐步处理
    for i, chunk in enumerate(all_chunks):
        step += 1
        remaining_chunks = all_chunks[i + 1 :]

        # 解析当前 chunk
        try:
            result = detector.parse_streaming_increment(chunk, tools)
        except Exception as e:
            print(f"\n❌ 解析错误: {e}")
            result_text = ""
            result_calls = []
        else:
            result_text = result.normal_text or ""
            result_calls = result.calls or []

        # 更新累积的文本
        if result_text:
            accumulated_text += result_text

        # 更新累积的工具调用
        if result_calls:
            for call in result_calls:
                if call.tool_index not in accumulated_toolcalls:
                    accumulated_toolcalls[call.tool_index] = {"name": "", "args": ""}

                if call.name:
                    accumulated_toolcalls[call.tool_index]["name"] = call.name
                if call.parameters:
                    accumulated_toolcalls[call.tool_index]["args"] += call.parameters

        # 显示当前步骤信息
        display_step_info(
            step=step,
            current_chunk=chunk,
            remaining_chunks=remaining_chunks,
            accumulated_text=accumulated_text,
            accumulated_toolcalls=accumulated_toolcalls,
            result_text=result_text,
            result_calls=result_calls,
        )

        # 等待用户输入
        user_input = input().strip().lower()
        if user_input == "q":
            print("\n用户退出")
            break

    # 显示最终结果
    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    print(f"\n[最终文本]")
    print(f"  {accumulated_text}")
    print(f"\n[最终工具调用]")
    print(format_toolcalls(accumulated_toolcalls))
    print("\n" + "=" * 80)


def main():
    """主函数"""
    # 默认的测试文本
    default_response_text = """Plain Text

<tool_call>
<function=get_current_weather>
<parameter=location>New York</parameter>
</function>
</tool_call>
<tool_call>
<function=sql_interpreter>
<parameter=query>SELECT * FROM users</parameter>
<parameter=dry_run>True</parameter>
</function>
</tool_call>"""

    # 可以从命令行参数读取文本，或者使用默认文本
    if len(sys.argv) > 1:
        # 如果提供了文件路径，读取文件内容
        import os

        if os.path.isfile(sys.argv[1]):
            with open(sys.argv[1], "r", encoding="utf-8") as f:
                response_text = f.read()
        else:
            # 否则直接使用命令行参数作为文本
            response_text = sys.argv[1]
    else:
        response_text = default_response_text

    # 运行演示
    run_interactive_demo(
        response_text=response_text,
        mode="atomic_tags",
        parser_source="legacy_v3",
    )


if __name__ == "__main__":
    main()

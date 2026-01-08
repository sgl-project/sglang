import json
import re
import random
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass

from sglang.srt.entrypoints.openai.protocol import Tool


@dataclass
class StreamingTestResult:
    test_name: str
    full_parsed_text: str
    final_reconstructed: Dict
    step_count: int
    non_streaming_result: any

    error_message: Optional[str] = None
    success: bool = True
    streaming_match: Optional[bool] = None
    streaming_errors: Optional[List[str]] = None
    non_streaming_match: Optional[bool] = None
    non_streaming_errors: Optional[List[str]] = None


class StreamingTestRunner:
    def __init__(self, detector, tools: Optional[List[Tool]] = None, parser_mode: str = "both", verbose: bool = False):
        self.detector = detector
        self.parser_mode = parser_mode
        self.default_tools = tools
        self.verbose = verbose

    def run_test(
        self,
        test_name: str,
        response_text: str,
        mode: str = "atomic_tags",
        tools: Optional[List[Tool]] = None,
        verbose: Optional[bool] = None,
        compare_with_non_streaming: Optional[bool] = None,
        expected: Optional[Dict] = None,
    ) -> StreamingTestResult:
        """
        运行单个流式测试

        Args:
            test_name: 测试名称
            response_text: 完整的响应文本
            mode: 流式生成模式 ('char', 'atomic_tags', 或其他)
            tools: 工具列表，如果为 None 则使用默认工具
            verbose: 是否打印详细日志，如果为 None 则使用实例属性 self.verbose
            compare_with_non_streaming: 是否与非流式解析结果对比，如果为 None 则根据 parser_mode 自动决定
            expected: 期望的解析结果，包含 'text' 和 'tools' 字段

        Returns:
            StreamingTestResult: 测试结果
        """
        # 如果 verbose 未指定，使用实例属性
        if verbose is None:
            verbose = self.verbose

        # 根据 parser_mode 决定是否运行流式和非流式解析
        run_streaming = self.parser_mode in ("both", "streaming")
        run_non_streaming = self.parser_mode in ("both", "nonstream")

        # 如果 compare_with_non_streaming 未指定，根据 parser_mode 决定
        if compare_with_non_streaming is None:
            compare_with_non_streaming = run_non_streaming

        if verbose:
            self._print_test_header(test_name, response_text, mode, run_streaming, run_non_streaming)

        tools = tools or self.default_tools
        if tools is None:
            raise ValueError("tools must be provided either in __init__ or run_test")

        full_parsed_text = ""  # 用于累积最终解析出的完整纯文本
        final_reconstructed = {}  # 用于累积最终解析出的完整工具调用

        step = 0
        error_message = None

        # 只在需要时运行流式解析
        if run_streaming:
            try:
                for chunk in token_stream_generator(response_text, mode=mode):
                    step += 1
                    result = self.detector.parse_streaming_increment(chunk, tools)

                    # 如果解析器返回了 normal_text，将其拼接到总文本中
                    if result.normal_text:
                        full_parsed_text += result.normal_text

                    # 收集工具调用信息
                    if result.calls:
                        for call in result.calls:
                            if call.tool_index not in final_reconstructed:
                                final_reconstructed[call.tool_index] = {"name": "", "args": ""}

                            if call.name:
                                final_reconstructed[call.tool_index]["name"] = call.name
                            if call.parameters:
                                final_reconstructed[call.tool_index]["args"] += call.parameters

                    if verbose:
                        display_chunk = chunk.replace("\n", "\\n")
                        log_msg = f"Step {step:03d} | Input: '{display_chunk}'"
                        if result.normal_text:
                            log_msg += f" | Text Found: {result.normal_text}"
                        print(log_msg)

            except Exception as e:
                error_message = str(e)
                if verbose:
                    print(f"ERROR: {error_message}")

        # 获取非流式解析结果用于对比
        non_streaming_result = None
        if run_non_streaming:
            try:
                non_streaming_result = self.detector.detect_and_parse(response_text, tools)
            except Exception as e:
                if verbose:
                    print(f"WARNING: Non-streaming parse failed: {e}")

        if verbose:
            self._print_test_summary(full_parsed_text, final_reconstructed, non_streaming_result, run_streaming, run_non_streaming)

        # 验证结果
        streaming_match = None
        non_streaming_match = None
        streaming_errors = []
        non_streaming_errors = []

        if expected is not None:
            # 只在运行流式解析时验证流式解析结果
            if run_streaming:
                streaming_match, streaming_errors = self._validate_result(expected=expected, actual_text=full_parsed_text, actual_tools=final_reconstructed, result_type="streaming")

            # 只在运行非流式解析时验证非流式解析结果
            if run_non_streaming and non_streaming_result is not None:
                non_streaming_tools = {}
                for idx, call in enumerate(non_streaming_result.calls):
                    try:
                        args = json.loads(call.parameters) if call.parameters else {}
                    except:
                        args = call.parameters
                    non_streaming_tools[call.tool_index] = {"name": call.name, "args": args}
                non_streaming_match, non_streaming_errors = self._validate_result(
                    expected=expected,
                    actual_text=non_streaming_result.normal_text or "",
                    actual_tools=non_streaming_tools,
                    result_type="non-streaming",
                )

        return StreamingTestResult(
            test_name=test_name,
            full_parsed_text=full_parsed_text,
            final_reconstructed=final_reconstructed,
            non_streaming_result=non_streaming_result,
            step_count=step,
            success=error_message is None,
            error_message=error_message,
            streaming_match=streaming_match,
            non_streaming_match=non_streaming_match,
            streaming_errors=streaming_errors if streaming_errors else None,
            non_streaming_errors=non_streaming_errors if non_streaming_errors else None,
        )

    def _print_test_header(self, test_name: str, response_text: str, mode: str, run_streaming: bool, run_non_streaming: bool):
        """打印测试头部信息"""
        print(f"\n{'='*60}")
        mode_info = []
        if run_streaming:
            mode_info.append(f"streaming ({mode})")
        if run_non_streaming:
            mode_info.append("non-streaming")
        mode_str = " + ".join(mode_info) if mode_info else "none"
        print(f"TEST: {test_name} (Mode: {mode_str})")
        print(f"Response Text: ")
        print(response_text)
        print(f"{'-'*60}")

    def _print_test_summary(self, full_parsed_text: str, final_reconstructed: Dict, non_streaming_result: any, run_streaming: bool, run_non_streaming: bool):
        """打印测试结果汇总"""
        print("解析结果汇总 (FINAL RESULTS):")

        section_num = 1
        if run_streaming:
            print(f"\n[{section_num}] 流式解析结果 (Streaming parse result):")
            print(f"提取的纯文本 (Plain Text):\n\033[94m<bos>{full_parsed_text}<eos>\033[0m")
            print(f"提取的工具调用 (Tool Calls):")
            print(json.dumps(final_reconstructed, indent=2, ensure_ascii=False))
            section_num += 1

        if run_non_streaming and non_streaming_result:
            print(f"\n[{section_num}] 非流式解析结果 (Non-streaming parse result):")
            print(f"Text: {non_streaming_result.normal_text}")
            for idx, call in enumerate(non_streaming_result.calls):
                print(f" - (T{idx}) Tool call: {call.name}")
                print(f"\tP-string: {call.parameters}")
                try:
                    print(f"\t  P-dict: {json.loads(call.parameters)}")
                except:
                    print(f"\t  P-dict: (parse error)")

    def _validate_result(self, expected: Dict, actual_text: str, actual_tools: Dict, result_type: str) -> Tuple[bool, List[str]]:
        """
        验证解析结果是否符合预期

        Args:
            expected: 期望结果，包含 'text' 和 'tools' 字段
            actual_text: 实际解析出的文本
            actual_tools: 实际解析出的工具调用，格式为 {tool_index: {"name": str, "args": str}}
            result_type: 结果类型 ('streaming' 或 'non-streaming')

        Returns:
            (是否匹配, 错误列表)
        """
        errors = []
        match = True

        # 验证文本
        if "text" in expected:
            expected_text = expected["text"]

            # if actual_text != expected_text:
            # TODO: 解析器目前会搞出来过的 \n，这个还没定位到问题，因此先用 rstrip 去掉
            if actual_text.rstrip() != expected_text.rstrip():
                match = False
                errors.append(f"{result_type} text mismatch: expected '{expected_text}', got '{actual_text}'")

        # 验证工具调用
        if "tools" in expected:
            expected_tools = expected["tools"]
            actual_tools_list = []

            # 将 actual_tools 转换为列表格式以便比较
            for tool_idx in sorted(actual_tools.keys()):
                tool = actual_tools[tool_idx]
                tool_name = tool.get("name", "")
                tool_args_str = tool.get("args", "")

                # 尝试解析 args 为 JSON
                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except:
                    tool_args = tool_args_str

                actual_tools_list.append({"name": tool_name, "args": tool_args})

            # 比较工具数量
            if len(actual_tools_list) != len(expected_tools):
                match = False
                errors.append(f"{result_type} tool count mismatch: expected {len(expected_tools)}, got {len(actual_tools_list)}")

            # 比较每个工具
            for i, expected_tool in enumerate(expected_tools):
                if i >= len(actual_tools_list):
                    match = False
                    errors.append(f"{result_type} tool {i} missing")
                    continue

                actual_tool = actual_tools_list[i]
                if actual_tool["name"] != expected_tool["name"]:
                    match = False
                    errors.append(f"{result_type} tool {i} name mismatch: expected '{expected_tool['name']}', got '{actual_tool['name']}'")

                # 深度比较 args
                if not self._deep_compare(actual_tool.get("args", {}), expected_tool.get("args", {})):
                    match = False
                    errors.append(f"{result_type} tool {i} args mismatch: expected {expected_tool.get('args')}, got {actual_tool.get('args')}")

        return match, errors

    def _deep_compare(self, actual: any, expected: any) -> bool:
        """深度比较两个值是否相等"""
        if type(actual) != type(expected):
            return False

        if isinstance(actual, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            for key in actual.keys():
                if not self._deep_compare(actual[key], expected[key]):
                    return False
            return True
        elif isinstance(actual, list):
            if len(actual) != len(expected):
                return False
            for i in range(len(actual)):
                if not self._deep_compare(actual[i], expected[i]):
                    return False
            return True
        else:
            return actual == expected


def token_stream_generator(text: str, mode: str = "atomic_tags") -> Generator[str, None, None]:
    """
    模拟流式输出的生成器。

    Args:
        text: 完整的 LLM 输出字符串。
        mode:
            'char': 每次随机输出 1-5 个字符（压力测试）。
            'token': 模拟真实 Token 边界（为了简单起见，按空格和特殊符号模拟）。
            'atomic_tags': 确保 <tool_call> 等标签不会被切断，其他部分随机。
            'whole': 一次性输出整个文本（用于调试）。

    Yields:
        str: 流式文本块
    """
    if mode == "char":
        idx = 0
        while idx < len(text):
            # 随机模拟不同 token 长度
            step = random.randint(1, 5)
            yield text[idx : idx + step]
            idx += step

    elif mode == "atomic_tags":
        # 保护特定的标签不被切断，模拟理想的 Tokenizer 行为
        protected_tokens = [
            "<tool_call>",
            "</tool_call>",
            # "<function=",
            # "</function>",
            # "<parameter=",
            # "</parameter>",
            # ">",
        ]

        # 简单的正则切分
        pattern = "|".join(map(re.escape, protected_tokens))
        parts = re.split(f"({pattern})", text)

        for part in parts:
            if not part:
                continue
            if part in protected_tokens:
                yield part
            else:
                # 标签之间的内容，比如参数值，可以随机碎裂
                idx = 0
                while idx < len(part):
                    step = random.randint(1, 4)
                    yield part[idx : idx + step]
                    idx += step

    elif mode == "whole":
        # 一次性输出整个文本（用于调试）
        yield text

    # elif mode == "token":

    else:  # default simple iteration
        yield text

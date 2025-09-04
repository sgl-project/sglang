import ast
import html
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from xml.parsers.expat import ParserCreate

from sglang.srt.entrypoints.openai.protocol import (
    DeltaMessage,
    FunctionResponse,
    Tool,
    ToolCall,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


def _safe_val(raw: str) -> Any:
    raw = html.unescape(raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw


class StreamingXMLToolCallParser:
    """
    核心流式解析器，负责处理XML格式的工具调用


    ## 主要状态变量：
    - current_call_id: 当前工具调用的唯一标识
    - current_function_name: 当前函数名
    - parameters: 存储已解析的参数
    - current_param_name: 当前参数名
    - current_param_value: 当前参数值
    - tool_call_index: 工具调用索引计数器
    - streaming_buffer: 流式处理缓冲区
    - text_content_buffer: 文本内容缓冲区

    ## 处理流程：
    a. 初始化状态：设置初始状态变量和XML解析器
    b. 流式输入处理：通过parse_single_streaming_chunks接收数据块
    c. XML元素识别：使用_find_next_complete_element识别完整XML元素
    d. XML解析：使用expat解析器处理XML元素，触发_start_element、_char_data、_end_element回调
    e. 状态转移：根据XML元素类型更新状态变量
    f. Delta生成：在适当时候生成DeltaMessage并发送

    4. 状态转移过程：
    a. 开始解析：<tool_call>标签重置工具调用状态
    b. 函数识别：<function>标签提取函数名并生成初始工具调用Delta
    c. 参数处理：<parameter>标签开始参数解析，_char_data处理参数值
    d. 参数结束：根据参数类型决定是否添加引号，存储转换后的值
    e. 函数结束：关闭JSON对象，输出完整的函数调用
    f. 工具调用结束：<tool_call>标签结束当前工具调用，重置解析器状态

    5. 特殊处理：
    - XML特殊字符转义与反转义
    - 参数类型转换（字符串、数字、布尔值等）
    - 流式输出延迟处理，确保JSON格式正确
    - 多个tool_call的处理与状态隔离

    """

    def __init__(self):
        self.call_id_counter = 0
        self.tool_call_index = 0
        self.current_call_id = None
        self.last_completed_call_id = None  # 保存最近完成的call_id
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""  # 记录类型转换后的参数值
        self.current_param_is_first = False  # 记录是否是第一个参数
        # 这里需要延迟输出，因为参数的末尾会包含一个额外需要去除的换行符，但是中间的换行符需要保留
        self.should_emit_end_newline = False  # 记录是否需要输出末尾换行符
        self.start_quote_emitted = False  # 记录是否已经输出了字符串参数的开始引号

        self.deltas = []

        # 单块流式处理状态
        self.streaming_buffer = ""
        self.last_processed_pos = 0

        # 用于收集tool_call前面的文本内容
        self.text_content_buffer = ""

        # XML解析器相关
        self.parser = ParserCreate()
        self.setup_parser()

        # 工具配置信息
        self.tools = []

        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function"
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter"
        self.parameter_end_token: str = "</parameter>"

    def set_tools(self, tools: List[Tool]):
        """设置工具配置信息"""
        self.tools = tools

    def _get_param_type(self, param_name: str) -> str:
        """根据tool配置获取参数类型，默认为string"""
        if not self.tools or not self.current_function_name:
            return "string"

        for tool in self.tools:
            if not hasattr(tool, "type") or not (
                hasattr(tool, "function") and hasattr(tool.function, "name")
            ):
                continue
            if (
                tool.type == "function"
                and tool.function.name == self.current_function_name
            ):
                if not hasattr(tool.function, "parameters"):
                    return "string"
                params = tool.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    properties = params["properties"]
                    if param_name in properties and isinstance(
                        properties[param_name], dict
                    ):
                        return str(properties[param_name].get("type", "string"))
                elif isinstance(params, dict) and param_name in params:
                    param_config = params[param_name]
                    if isinstance(param_config, dict):
                        return str(param_config.get("type", "string"))
                break
        return "string"

    def _convert_param_value(self, param_value: str, param_type: str) -> Any:
        """根据参数类型转换值"""
        # 特殊的case：比如模型为bool 值输出True/False，需要json.dumps转为true/false
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        param_type = param_type.strip().lower()
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                pass
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                param_value = (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except Exception:
                pass
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            return param_value == "true"
        else:
            return param_value

    def _convert_for_json_streaming(self, converted_value: Any, param_type: str) -> str:
        """根据converted_value是否为空以及type是否为string，来对convert_value进行转换

        Args:
            converted_value: 转换后的值
            param_type: 参数类型

        Returns:
            转换后的字符串，用于流式输出
        """
        # 检查是否为空值，但排除数字0
        if converted_value is None or converted_value == "":
            return ""

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            # strip space and \n
            converted_value = converted_value.strip()
            # 字符串类型，去掉双引号
            return json.dumps(converted_value, ensure_ascii=False)[1:-1]
        else:
            # 非字符串类型，返回完整的JSON字符串
            if not isinstance(converted_value, str):
                return json.dumps(converted_value, ensure_ascii=False)
            else:
                return converted_value

    def reset_streaming_state(self):
        """重置流式解析状态"""
        self.call_id_counter = 0
        self.tool_call_index = 0
        self.current_call_id = None
        self.last_completed_call_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.should_emit_end_newline = False
        self.start_quote_emitted = False

        # 重置单块流式处理状态
        self.streaming_buffer = ""
        self.last_processed_pos = 0

        # 重置文本内容缓冲区
        self.text_content_buffer = ""

        self.deltas = []

        # 重新创建解析器
        self.parser = ParserCreate()
        self.setup_parser()

    def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage:
        """
        解析单个流式XML块并返回Delta响应
        这是真正的流式接口，逐个接收chunk，维护内部状态

        Args:
            xml_chunk: 单个XML块字符串

        Returns:
            DeltaMessage: 包含此块生成的delta信息，如果没有完整元素则返回空响应
        """
        # 记录处理前的delta数量
        initial_delta_count = len(self.deltas)

        # 将新chunk添加到缓冲区
        self.streaming_buffer += xml_chunk

        # 处理完整的XML元素
        # 记录进入处理前的 call_id，用于多 tool_call 场景下的兜底保护
        snapshot_call_id = self.current_call_id
        found_elements = self._process_complete_xml_elements()

        if found_elements:
            # 如果找到了完整元素，检查是否遗漏了结束事件（可能存在部分标签未被触发）
            try:
                new_deltas = self.deltas[initial_delta_count:]
                # 若本chunk包含 </function> 但未生成 '}'，则补齐
                # 仅当当前仍在同一个 call 上时才进行兜底，避免跨多次 <tool_call> 误关新开启的调用
                if (
                    self.current_call_id is not None
                    and self.current_call_id == snapshot_call_id
                    and self.function_end_token in xml_chunk
                ):

                    # - 追加了 '}'（非空参数收尾）
                    # - 追加了 '{}'（空参数函数）
                    has_function_close = any(
                        (
                            td.tool_calls
                            and any(
                                (
                                    tc.function
                                    and tc.id == self.current_call_id
                                    and isinstance(tc.function.arguments, str)
                                    and (tc.function.arguments in ("}", "{}"))
                                )
                                for tc in td.tool_calls
                            )
                        )
                        for td in new_deltas
                    )
                    if not has_function_close:
                        # 关闭可能未闭合的参数
                        if self.current_param_name:
                            self._end_element("parameter")
                        # 补一个函数结束
                        if self.current_function_name:
                            self._end_element("function")
                # 若本chunk包含 </tool_call> 但未生成最终空delta，则补齐
                # 同样仅当仍在同一调用上时兜底，避免关闭刚开启的下一个调用
                if (
                    self.current_call_id is not None
                    and self.current_call_id == snapshot_call_id
                    and self.tool_call_end_token in xml_chunk
                ):
                    has_toolcall_close = any(
                        (
                            td.tool_calls
                            and any(
                                (
                                    tc.type == "function"
                                    and tc.function
                                    and tc.function.arguments == ""
                                    and tc.id == self.current_call_id
                                )
                                for tc in td.tool_calls
                            )
                        )
                        for td in new_deltas
                    )
                    if not has_toolcall_close:
                        # 关闭可能未闭合的参数
                        if self.current_param_name:
                            self._end_element("parameter")
                        if self.current_function_name:
                            self._end_element("function")
                        self._end_element("tool_call")
            except Exception:
                pass
            # 合并这次新生成的deltas为单个响应
            return self._merge_new_deltas_to_single_response(initial_delta_count)
        else:
            # 没有完整元素，检查是否有未输出的文本内容
            if self.text_content_buffer and self.tool_call_index == 0:
                # 有文本内容但还没有tool_call，输出文本内容
                text_delta = DeltaMessage(
                    role=None,
                    content=self.text_content_buffer,
                    reasoning_content=None,
                    tool_calls=[],
                )
                self._emit_delta(text_delta)
                # 清空缓冲区，避免重复输出
                self.text_content_buffer = ""
                return text_delta

            # 若本次chunk中包含结束标签但未被解析器触发，手动补齐结束事件
            # 仅当仍处于与进入时相同的 call 上时才执行，防止在多 <tool_call> 场景误关新调用
            if (
                self.current_call_id is not None
                and self.current_call_id == snapshot_call_id
                and (
                    self.function_end_token in xml_chunk
                    or self.tool_call_end_token in xml_chunk
                )
            ):
                # 若仍有未闭合参数，先关闭
                if self.current_param_name:
                    self._end_element("parameter")
                # 若包含 </function>，尝试关闭函数
                if self.function_end_token in xml_chunk and self.current_function_name:
                    self._end_element("function")
                # 若包含 </tool_call>，尝试关闭工具调用
                if self.tool_call_end_token in xml_chunk:
                    self._end_element("tool_call")
                # 返回这次兜底所生成的delta合并结果
                return self._merge_new_deltas_to_single_response(initial_delta_count)

            # 没有完整元素，返回空响应
            return DeltaMessage(
                role=None, content=None, reasoning_content=None, tool_calls=[]
            )

    def _escape_xml_special_chars(self, text: str) -> str:
        """
        转义XML特殊字符

        Args:
            text: 原始文本

        Returns:
            转义后的文本
        """
        # XML特殊字符转义映射
        xml_escapes = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }

        for char, escape in xml_escapes.items():
            text = text.replace(char, escape)

        return text

    def _unescape_xml_special_chars(self, text: str) -> str:
        """
        反转义XML特殊字符

        Args:
            text: 转义后的文本

        Returns:
            原始文本
        """
        # XML特殊字符反转义映射
        xml_unescapes = {
            "&amp;": "&",
            "&lt;": "<",
            "&gt;": ">",
            "&quot;": '"',
            "&apos;": "'",
        }

        for escape, char in xml_unescapes.items():
            text = text.replace(escape, char)

        return text

    def _process_complete_xml_elements(self) -> bool:
        """
        处理缓冲区中的完整XML元素

        Returns:
            bool: 是否找到并处理了完整的元素
        """
        found_any = False

        while self.last_processed_pos < len(self.streaming_buffer):
            # 查找下一个完整元素
            element, end_pos = self._find_next_complete_element(self.last_processed_pos)
            if element is None:
                # 没有找到完整元素，等待更多数据
                break

            # 检查是否应该跳过这个元素
            if self._should_skip_element(element):
                # print(f"跳过非XML文本: {repr(element)}")
                self.last_processed_pos = end_pos
                continue

            # 找到完整的XML元素，处理它
            try:
                # 预处理XML块
                preprocessed_element = self._preprocess_xml_chunk(element)
                # 检查是否是第一个tool_call开始
                if (
                    preprocessed_element.strip().startswith("<tool_call>")
                    and self.tool_call_index == 0
                ):
                    # 第一个tool_call开始，先输出之前收集的文本内容
                    if self.text_content_buffer:
                        text_delta = DeltaMessage(
                            role=None,
                            content=self.text_content_buffer,
                            reasoning_content=None,
                            tool_calls=[],
                        )
                        self._emit_delta(text_delta)
                        # 清空缓冲区，为后续可能的文本内容做准备
                        self.text_content_buffer = ""

                # 检查是否是新的tool_call开始且当前已有完成的tool_call
                if (
                    preprocessed_element.strip().startswith("<tool_call>")
                    and self.tool_call_index > 0
                ):
                    # 新的tool_call开始，重置解析器状态但保留已生成的deltas
                    # print(f"reset parser for new tool call")
                    self._reset_parser_for_new_tool_call()

                # 解析预处理后的元素
                self.parser.Parse(preprocessed_element, False)
                found_any = True

            except Exception as e:
                print(f"exception occurs: {e}, preprocessed_element: {repr(element)}")
                pass

            # 更新已处理位置
            self.last_processed_pos = end_pos

        return found_any

    def _reset_parser_for_new_tool_call(self):
        """
        为新的tool_call重置解析器状态（但保留已生成的deltas）
        """
        # 在开始新的 tool_call 之前，若上一调用仍未正常闭合，则主动补齐：
        # 1) 关闭未结束的 parameter -> 等价于解析 </parameter>
        # 2) 关闭未结束的 function -> 触发输出 '}' 或 '{}'
        # 3) 输出最终空的 tool_call delta，并重置解析器状态
        if self.current_call_id:
            if self.current_param_name:
                self._end_element("parameter")
            if self.current_function_open or self.current_function_name:
                self._end_element("function")
            # 输出最终的 tool_call 收尾 delta（与 _end_element('tool_call') 中一致）
            final_delta = DeltaMessage(
                role=None,
                content=None,
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_call_id,
                        type="function",
                        function=FunctionResponse(name=None, arguments=""),
                    )
                ],
            )
            self._emit_delta(final_delta)
            # 重置 XML 解析器与当前调用状态
            self._reset_xml_parser_after_tool_call()

        # 保存当前的deltas和tool_call_index（包含上一步补齐产生的deltas）
        current_deltas = self.deltas.copy()
        current_tool_call_index = self.tool_call_index

        # 检查是否有文本内容需要输出（在tool_call之间）
        if self.text_content_buffer.strip():
            text_delta = DeltaMessage(
                role=None,
                content=self.text_content_buffer,
                reasoning_content=None,
                tool_calls=[],
            )
            current_deltas.append(text_delta)

        # 重置解析器状态
        # 保存当前call_id到last_completed_call_id，然后重置current_call_id
        if self.current_call_id:
            self.last_completed_call_id = self.current_call_id
        self.current_call_id = None
        self.current_function_name = None
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.start_quote_emitted = False

        # 重置文本内容状态，为下一个tool_call做准备
        self.text_content_buffer = ""

        # 创建新的解析器实例
        self.parser = ParserCreate()
        self.setup_parser()

        # 恢复已生成的deltas和tool_call_index
        self.deltas = current_deltas
        self.tool_call_index = current_tool_call_index

    def _should_skip_element(self, element: str) -> bool:
        """
        判断是否应该跳过某个元素

        Args:
            element: 要判断的元素

        Returns:
            bool: True表示应该跳过，False表示应该处理
        """
        # element = element.strip()

        # 如果是tool_call的xml标签，不跳过
        if element.startswith("<tool_call>"):
            return False

        # 如果当前没有在解析工具调用，且不是空白，收集这个文本而不是跳过
        # 只有tool_call出现了，才处理其他xml元素，否则当成纯文本
        if self.current_call_id is None and element:
            # 收集文本内容到缓冲区
            self.text_content_buffer += element
            return True  # 仍然跳过，但已经收集了内容

        # 如果当前正在解析工具调用，这可能是参数值，不跳过
        if self.current_call_id is not None:
            return False

        # 空白内容跳过
        if not element:
            return True

        return False

    def _find_next_complete_element(self, start_pos: int) -> Tuple[Optional[str], int]:
        """
        从指定位置查找下一个完整的XML元素

        Args:
            start_pos: 开始查找的位置

        Returns:
            (完整元素字符串, 元素结束位置)，如果没有找到完整元素返回(None, start_pos)
        """
        buffer = self.streaming_buffer[start_pos:]

        if not buffer:
            return None, start_pos

        # 查找XML标签
        if buffer.startswith("<"):
            # 需要保证不出现新的<，找出<和>中最近的一个
            tag_end = buffer.find("<", 1)
            tag_end2 = buffer.find(">", 1)
            if tag_end != -1 and tag_end2 != -1:
                # 下一个最近的是<
                if tag_end < tag_end2:
                    return buffer[:tag_end], start_pos + tag_end
                # 下一个最近的是>，说明找到xml元素
                else:
                    return buffer[: tag_end2 + 1], start_pos + tag_end2 + 1
            elif tag_end != -1:
                return buffer[:tag_end], start_pos + tag_end
            elif tag_end2 != -1:
                return buffer[: tag_end2 + 1], start_pos + tag_end2 + 1
            else:
                # 如果当前没有在解析工具调用（进入一个tool_call），检查是否以<tool_call>开头
                if self.current_call_id is None:
                    # 按照buffer的长度匹配<tool_call>
                    tool_call_prefix = "<tool_call>"
                    if len(buffer) >= len(tool_call_prefix):
                        # buffer长度足够，检查是否匹配<tool_call
                        if buffer.startswith(tool_call_prefix):
                            # 匹配上了，等待更多数据
                            return None, start_pos
                        else:
                            # 没匹配上，当文本处理
                            return buffer, start_pos + len(buffer)
                    else:
                        # buffer长度不够，检查是否可能是<tool_call>的开头
                        if buffer == "<tool_call>"[: len(buffer)]:
                            # 可能是<tool_call>的开头，等待更多数据
                            return None, start_pos
                        else:
                            # 不是<tool_call>的开头，当文本处理
                            return buffer, start_pos + len(buffer)
                else:
                    # 正在解析工具调用时，等待更多数据以获取完整的标签
                    return None, start_pos
        else:
            # 查找文本内容（直到下一个 < 或缓冲区结束）
            next_tag_pos = buffer.find("<")
            if next_tag_pos != -1:
                # 找到文本内容
                text_content = buffer[:next_tag_pos]
                if text_content.strip():  # 只处理非空白文本
                    return text_content, start_pos + next_tag_pos
                else:
                    # 跳过空白内容
                    return text_content, start_pos + next_tag_pos
            else:
                # 缓冲区末尾都是文本，立即处理（不再等待更多数据）
                remaining = buffer
                if remaining.strip():  # 有实际内容
                    return remaining, start_pos + len(remaining)
                else:
                    # 空白内容，跳过
                    return remaining, start_pos + len(remaining)

    def _merge_new_deltas(self, deltas: List[DeltaMessage]) -> DeltaMessage:
        """
        将DeltaMessage数组合并为单个DeltaMessage

        Args:
            deltas: 要合并的DeltaMessage列表

        Returns:
            合并后的DeltaMessage，包含所有输入deltas的信息
        """
        if not deltas:
            return DeltaMessage(
                role=None, content=None, reasoning_content=None, tool_calls=[]
            )

        # 过滤掉空的deltas（tool_calls为空或None的）
        valid_deltas = [
            delta for delta in deltas if delta is not None and delta.tool_calls
        ]
        if not valid_deltas:
            return DeltaMessage(
                role=None, content=None, reasoning_content=None, tool_calls=[]
            )

        # 收集所有的content和reasoning_content
        merged_content = ""
        merged_reasoning_content = ""
        merged_role = None

        for delta in deltas:
            if delta:
                if delta.role:
                    merged_role = delta.role
                if delta.content:
                    merged_content += delta.content
                if delta.reasoning_content:
                    merged_reasoning_content += delta.reasoning_content

        # 合并所有tool_calls
        merged_tool_calls = []
        merged_tool_calls_index = []
        for delta in valid_deltas:
            for tool_call in delta.tool_calls:
                if tool_call.index not in merged_tool_calls_index:
                    merged_tool_calls.append(tool_call)
                    merged_tool_calls_index.append(tool_call.index)
                else:
                    if tool_call.function and tool_call.function.arguments is not None:
                        merged_tool_calls[
                            merged_tool_calls_index.index(tool_call.index)
                        ].function.arguments += tool_call.function.arguments

        if not merged_tool_calls:
            return DeltaMessage(
                role=merged_role,
                content=merged_content or None,
                reasoning_content=merged_reasoning_content or None,
                tool_calls=[],
            )

        return DeltaMessage(
            role=merged_role,
            content=merged_content if merged_content else None,
            reasoning_content=(
                merged_reasoning_content if merged_reasoning_content else None
            ),
            tool_calls=merged_tool_calls,
        )

    def _merge_new_deltas_to_single_response(self, initial_count: int) -> DeltaMessage:
        """
        将这次处理中新生成的deltas合并为单个DeltaMessage

        Args:
            initial_count: 处理前的delta数量

        Returns:
            合并后的DeltaMessage，包含所有新生成的delta信息
        """
        if len(self.deltas) <= initial_count:
            return DeltaMessage(
                role=None, content=None, reasoning_content=None, tool_calls=[]
            )

        # 获取新生成的deltas
        new_deltas = self.deltas[initial_count:]

        if len(new_deltas) == 1:
            # 只有一个新delta，直接返回
            return new_deltas[0]

        # 合并多个新deltas
        merged_tool_calls = []
        merged_content = ""
        merged_reasoning_content = ""
        merged_role = None

        for delta in new_deltas:
            if delta.role:
                merged_role = delta.role
            if delta.content:
                merged_content += delta.content
            if delta.reasoning_content:
                merged_reasoning_content += delta.reasoning_content
            if delta.tool_calls:
                # 对于tool_calls，我们需要智能合并arguments
                for tool_call in delta.tool_calls:
                    # 查找是否已经有相同call_id的tool_call

                    existing_call = None
                    for existing in merged_tool_calls:
                        if existing.id == tool_call.id:
                            existing_call = existing
                            break

                    if existing_call:
                        # 合并到现有的tool_call
                        if tool_call.function and tool_call.function.name:
                            existing_call.function.name = tool_call.function.name
                        if (
                            tool_call.function
                            and tool_call.function.arguments is not None
                        ):
                            if existing_call.function.arguments is None:
                                existing_call.function.arguments = ""

                            # 对于流式JSON参数，简单按顺序拼接
                            new_args = tool_call.function.arguments
                            existing_call.function.arguments += new_args
                        if tool_call.type:
                            existing_call.type = tool_call.type
                    else:
                        # 添加新的tool_call
                        merged_tool_calls.append(tool_call)

        return DeltaMessage(
            role=merged_role,
            content=merged_content if merged_content else None,
            reasoning_content=(
                merged_reasoning_content if merged_reasoning_content else None
            ),
            tool_calls=merged_tool_calls,
        )

    def _parse_incremental_xml(self, new_content: str) -> List[DeltaMessage]:
        """
        增量解析XML内容

        Args:
            new_content: 新增的文本内容

        Returns:
            DeltaMessage列表
        """
        if not new_content.strip():
            return []

        # 清空之前的deltas，只返回新的
        previous_deltas_count = len(self.deltas)

        # 检查是否有完整的XML标签可以解析
        xml_chunks = self._extract_complete_xml_chunks(new_content)

        if not xml_chunks:
            return []

        try:
            # 预处理和解析完整的XML块
            for chunk in xml_chunks:
                if chunk.strip():
                    # 预处理非标准格式
                    processed_chunk = self._preprocess_xml_chunk(chunk)
                    self.parser.Parse(processed_chunk, False)

            # 返回新生成的deltas
            new_deltas = self.deltas[previous_deltas_count:]
            return new_deltas

        except Exception:
            # 如果解析失败，可能是因为XML不完整，返回空列表
            # print(f"增量解析失败: {e}")
            return []

    def _preprocess_xml_chunk(self, chunk: str) -> str:
        """
        预处理XML块，处理非标准格式

        Args:
            chunk: 原始XML块

        Returns:
            处理后的XML块
        """
        is_tool_call = False
        if chunk.startswith("<tool_call>") or chunk.startswith("</tool_call>"):
            is_tool_call = True
        if chunk.startswith("<function=") or chunk.startswith("</function>"):
            is_tool_call = True
        if chunk.startswith("<parameter=") or chunk.startswith("</parameter>"):
            is_tool_call = True
        # 处理 <function=name> 格式 -> <function name="name">
        processed = re.sub(r"<function=([^>]+)>", r'<function name="\1">', chunk)
        # 处理 <parameter=name> 格式 -> <parameter name="name">
        processed = re.sub(r"<parameter=([^>]+)>", r'<parameter name="\1">', processed)
        # 如果processed中不包含special_token，则对processed进行转义
        # 这是因为xml解析遇到特殊字符会报错，所以需要转义
        if not is_tool_call:
            processed = self._escape_xml_special_chars(processed)
        return processed

    def _extract_complete_xml_chunks(self, new_content: str) -> List[str]:
        """
        从新内容中提取完整的XML块

        Args:
            new_content: 新增的文本内容

        Returns:
            完整XML块的列表
        """
        chunks = []
        buffer = new_content

        # 查找完整的XML标签
        i = 0
        while i < len(buffer):
            if buffer[i] == "<":
                # 查找标签结束
                tag_end = buffer.find(">", i)
                if tag_end != -1:
                    # 找到完整标签
                    tag = buffer[i : tag_end + 1]
                    chunks.append(tag)
                    i = tag_end + 1
                else:
                    # 标签不完整，停止处理
                    break
            else:
                # 查找下一个 < 或者累积文本内容
                next_tag = buffer.find("<", i)
                if next_tag != -1:
                    # 有文本内容
                    text_content = buffer[i:next_tag]
                    if text_content.strip():
                        chunks.append(text_content)
                    i = next_tag
                else:
                    # 剩余都是文本内容
                    remaining = buffer[i:]
                    if remaining.strip():
                        chunks.append(remaining)
                    break

        return chunks

    def _convert_to_delta_message(
        self, delta_responses: List[DeltaMessage]
    ) -> DeltaMessage:
        """
        将DeltaMessage列表转换为DeltaMessage

        Args:
            delta_responses: DeltaMessage列表

        Returns:
            DeltaMessage对象
        """
        if not delta_responses:
            return DeltaMessage()

        # 合并所有delta的内容
        merged_tool_calls = []
        merged_content = ""
        merged_reasoning_content = ""
        merged_role = None

        for delta in delta_responses:
            if delta.role:
                merged_role = delta.role
            if delta.content:
                merged_content += delta.content
            if delta.reasoning_content:
                merged_reasoning_content += delta.reasoning_content
            if delta.tool_calls:
                merged_tool_calls.extend(delta.tool_calls)

        return DeltaMessage(
            role=merged_role,
            content=merged_content if merged_content else None,
            reasoning_content=(
                merged_reasoning_content if merged_reasoning_content else None
            ),
            tool_calls=merged_tool_calls,
        )

    def setup_parser(self):
        """设置XML解析器事件处理器"""
        self.parser.buffer_text = True
        self.parser.StartElementHandler = self._start_element
        self.parser.EndElementHandler = self._end_element
        self.parser.CharacterDataHandler = self._char_data

    def _get_next_call_id(self):
        """生成唯一的调用ID"""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _extract_function_name(self, name: str, attrs: Dict[str, str]) -> Optional[str]:
        """从各种格式中提取函数名"""
        if attrs and "name" in attrs:
            return attrs["name"]

        # 处理 <function=name> 格式
        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "function":
                return parts[1]

        return None

    def _extract_parameter_name(
        self, name: str, attrs: Dict[str, str]
    ) -> Optional[str]:
        """从各种格式中提取参数名"""
        if attrs and "name" in attrs:
            return attrs["name"]

        # 处理 <parameter=name> 格式
        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "parameter":
                return parts[1]

        return None

    def _emit_delta(self, delta: DeltaMessage):
        """发送Delta响应（流式输出）"""
        self.deltas.append(delta)

    def _auto_close_open_parameter_if_needed(self, incoming_tag: Optional[str] = None):
        """在开始处理新的元素前，若之前存在未关闭的 tag，则自动补齐其结束到解析器中。

        - 若存在未关闭的 parameter，则等价于喂入 `</parameter>`（通过直接调用结束处理器）。
        - 当即将开始新的 function 或 tool_call 时，若存在未关闭的 function，则补齐 `</function>`。
        - 当即将开始新的 tool_call 时，若存在未关闭的 tool_call，则补齐 `</tool_call>`。
        """
        # 先关闭未结束的 parameter
        if self.current_param_name:
            # 调用 end 处理逻辑，效果等价于解析器收到了 </parameter>
            self._end_element("parameter")

        # 若即将开始新的 function 或 tool_call，且有未关闭的 function，则先关闭 function
        if incoming_tag in ("function", "tool_call") and self.current_function_name:
            self._end_element("function")

        # 若即将开始新的 tool_call，且有未关闭的 tool_call，则先关闭 tool_call
        if incoming_tag == "tool_call" and self.current_call_id:
            self._end_element("tool_call")

    def _start_element(self, name: str, attrs: Dict[str, str]):
        """处理XML开始元素事件"""

        # 忽略根元素包装
        if name == "root":
            return

        if name == "tool_call":
            # 在开启新 tool_call 之前，自动补齐上一个未闭合的标签
            self._auto_close_open_parameter_if_needed("tool_call")
            # 重置新的工具调用
            self.parameters = {}
            self.current_call_id = self._get_next_call_id()
            self.current_param_is_first = True  # 标记为第一个参数

            # 第一个tool_call标签不立即输出，等到function标签再输出
            # 这样第一个chunk会返回None，符合用户期望

            # 递增tool_call_index为下一个tool_call做准备
            self.tool_call_index += 1

        elif name.startswith("function") or (name == "function"):
            # 在开启新 function 之前，自动补齐上一个未闭合的标签（parameter/function）
            self._auto_close_open_parameter_if_needed("function")
            function_name = self._extract_function_name(name, attrs)
            self.current_function_name = function_name
            self.current_function_open = True
            if function_name:
                # 现在才输出初始工具调用
                delta = DeltaMessage(
                    role=None,
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        ToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=FunctionResponse(name=function_name, arguments=""),
                        )
                    ],
                )
                self._emit_delta(delta)

        elif name.startswith("parameter") or (name == "parameter"):
            # 若上一个参数尚未正常结束，先补齐其结束，再开始新的参数
            self._auto_close_open_parameter_if_needed("parameter")

            param_name = self._extract_parameter_name(name, attrs)
            self.current_param_name = param_name
            self.current_param_value = ""
            self.current_param_value_converted = ""
            self.start_quote_emitted = False  # 重置开始引号标志

            # 只输出参数名和冒号，不输出引号（等参数值确定类型后再决定）
            if param_name:
                if not self.parameters:
                    # 第一个参数 - 开始JSON，只输出参数名和冒号
                    json_start = f'{{"{param_name}": '
                    delta = DeltaMessage(
                        role=None,
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            ToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                function=FunctionResponse(
                                    name=None, arguments=json_start
                                ),
                            )
                        ],
                    )
                    self._emit_delta(delta)
                    self.current_param_is_first = True
                else:
                    # 后续参数 - 添加逗号和参数名，不加引号
                    json_continue = f', "{param_name}": '
                    delta = DeltaMessage(
                        role=None,
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            ToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                function=FunctionResponse(
                                    name=None, arguments=json_continue
                                ),
                            )
                        ],
                    )
                    self._emit_delta(delta)
                    self.current_param_is_first = False

    def _char_data(self, data: str):
        """处理XML字符数据事件"""
        if data and self.current_param_name:
            # 获取参数类型
            param_type = self._get_param_type(self.current_param_name)

            # 检查这是否是第一次接收到这个参数的数据
            if not self.current_param_value:
                # 如果是第一包数据，且以\n开头，则去掉\n
                if data.startswith("\n"):
                    data = data[1:]
                    if not data:
                        # 如果去掉换行符后数据为空，但还是需要为字符串类型输出开始引号
                        if (
                            param_type
                            in ["string", "str", "text", "varchar", "char", "enum"]
                            and not self.start_quote_emitted
                        ):
                            quote_delta = DeltaMessage(
                                role=None,
                                content=None,
                                reasoning_content=None,
                                tool_calls=[
                                    ToolCall(
                                        index=self.tool_call_index - 1,
                                        id=self.current_call_id,
                                        function=FunctionResponse(
                                            name=None, arguments='"'
                                        ),
                                    )
                                ],
                            )
                            self._emit_delta(quote_delta)
                            self.start_quote_emitted = True
                        return

            # 为字符串类型输出开始引号（如果还没有输出过）
            if (
                param_type in ["string", "str", "text", "varchar", "char", "enum"]
                and not self.start_quote_emitted
            ):
                quote_delta = DeltaMessage(
                    role=None,
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        ToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            function=FunctionResponse(name=None, arguments='"'),
                        )
                    ],
                )
                self._emit_delta(quote_delta)
                self.start_quote_emitted = True

            original_data = data
            # 延迟末尾换行符的输出
            if self.should_emit_end_newline:
                original_data = "\n" + original_data
                self.should_emit_end_newline = False
            if original_data.endswith("\n"):
                self.should_emit_end_newline = True
                original_data = original_data[:-1]
            self.current_param_value += original_data
            # 使用_convert_param_value转换参数值
            converted_value = self._convert_param_value(
                self.current_param_value, param_type
            )

            # 使用_convert_for_json_streaming处理流式输出
            output_data = self._convert_for_json_streaming(converted_value, param_type)

            delta_data = output_data[len(self.current_param_value_converted) :]
            self.current_param_value_converted = output_data

            # 立即输出参数值
            delta = DeltaMessage(
                role=None,
                content=None,
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_call_id,
                        function=FunctionResponse(name=None, arguments=delta_data),
                    )
                ],
            )
            self._emit_delta(delta)

    def _end_element(self, name: str):
        """处理XML结束元素事件"""

        # 忽略根元素包装
        if name == "root":
            return

        # 若函数或tool_call结束时仍有未关闭的参数，先补齐参数结束
        if (
            name.startswith("function") or name == "function" or name == "tool_call"
        ) and self.current_param_name:
            self._auto_close_open_parameter_if_needed()

        if (
            name.startswith("parameter") or name == "parameter"
        ) and self.current_param_name:
            # 结束当前参数
            param_name = self.current_param_name
            param_value = self.current_param_value

            # 获取参数类型
            param_type = self._get_param_type(param_name)

            # 使用_convert_param_value转换完整的参数值
            converted_value = self._convert_param_value(param_value, param_type)

            # 根据参数类型决定是否需要结束引号
            if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                # 对于空的字符串参数，需要特殊处理
                if not param_value:
                    if self.start_quote_emitted:
                        # 已经输出了开始引号，只需要输出结束引号
                        delta = DeltaMessage(
                            role=None,
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                ToolCall(
                                    index=self.tool_call_index - 1,
                                    id=self.current_call_id,
                                    function=FunctionResponse(name=None, arguments='"'),
                                )
                            ],
                        )
                        self._emit_delta(delta)
                    else:
                        # 没有输出过开始引号，直接输出完整的空字符串
                        delta = DeltaMessage(
                            role=None,
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                ToolCall(
                                    index=self.tool_call_index - 1,
                                    id=self.current_call_id,
                                    function=FunctionResponse(
                                        name=None, arguments='""'
                                    ),
                                )
                            ],
                        )
                        self._emit_delta(delta)
                else:
                    # 非空参数值，输出结束引号
                    delta = DeltaMessage(
                        role=None,
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            ToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                function=FunctionResponse(name=None, arguments='"'),
                            )
                        ],
                    )
                    self._emit_delta(delta)

            self.should_emit_end_newline = False
            # 存储转换后的值
            self.parameters[param_name] = converted_value
            self.current_param_name = None
            self.current_param_value = ""
            self.current_param_value_converted = ""
            self.start_quote_emitted = False

        elif name.startswith("function") or name == "function":
            # 只有当有参数时才关闭JSON对象
            if self.parameters:
                delta = DeltaMessage(
                    role=None,
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        ToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            function=FunctionResponse(name=None, arguments="}"),
                        )
                    ],
                )
                self._emit_delta(delta)
            # 该函数没有参数，则输出空对象
            else:
                delta = DeltaMessage(
                    role=None,
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        ToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            function=FunctionResponse(name=None, arguments="{}"),
                        )
                    ],
                )
                self._emit_delta(delta)
            # 标记函数已关闭
            self.current_function_open = False

        elif name == "tool_call":
            # 在结束tool_call之前，确保函数已经关闭以补齐缺失的右大括号
            if self.current_function_open:
                # 若仍有未关闭的参数，先关闭
                if self.current_param_name:
                    self._end_element("parameter")
                # 关闭函数，确保输出 '}' 或 '{}'
                self._end_element("function")
            # 最终Delta
            delta = DeltaMessage(
                role=None,
                content=None,
                reasoning_content=None,
                tool_calls=[
                    ToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_call_id,
                        type="function",
                        function=FunctionResponse(name=None, arguments=""),
                    )
                ],
            )
            self._emit_delta(delta)

            # 完成tool_call后，结束当前XML文档并重新创建解析器
            # 这样下一个非XML文本就不会被当作"junk after document element"
            self._reset_xml_parser_after_tool_call()

    def _reset_xml_parser_after_tool_call(self):
        """
        在完成tool_call后重置XML解析器
        结束当前文档并重新创建解析器，避免后续非XML文本被当作垃圾内容
        """
        try:
            # 结束当前XML文档
            self.parser.Parse("", True)
        except Exception:
            # 忽略结束文档时的错误
            pass

        # 重新创建XML解析器
        self.parser = ParserCreate()
        self.setup_parser()

        # 重置当前tool_call的状态
        # 保存当前call_id到last_completed_call_id，然后重置current_call_id
        if self.current_call_id:
            self.last_completed_call_id = self.current_call_id
        self.current_call_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.should_emit_end_newline = False
        self.start_quote_emitted = False


class Qwen3CoderDetector(BaseFormatDetector):
    """
    Detector for Qwen 3 models.
    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self._buf: str = ""
        # self.tool_call_prefix: str = "<function="

        # for non-stream extract
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )

        # Streaming state variables
        # self._current_function_name: str = ""
        # self._current_parameters: Dict[str, Any] = {}
        # self._streamed_parameters: Dict[str, str] = (
        #     {}
        # )  # Track what parameter content we've streamed
        # self._in_tool_call: bool = False
        # self._function_name_sent: bool = False

        self.parser = StreamingXMLToolCallParser()

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:

        # 模型有时候会单独输出导致delta_text为空。如果之前有tool_call，且当前所有的tool_call都已经结束，则返回一个空的tool_call
        # 用于外层流式输出时，能够正确输出tool_call字段
        if not new_text:
            open_calls = self._buf.count(
                self.parser.tool_call_start_token
            ) - self._buf.count(self.parser.tool_call_end_token)
            if open_calls == 0 and self.parser.tool_call_index > 0:
                # 如果current_call_id为None，使用last_completed_call_id
                call_id = (
                    self.parser.current_call_id or self.parser.last_completed_call_id
                )
                return StreamingParseResult(
                    calls=[ToolCallItem(tool_index=call_id, name="", parameters="")]
                )

        self._buf += new_text

        self.parser.set_tools(tools)
        delta_message = self.parser.parse_single_streaming_chunks(new_text)
        # print('-----new text--------')
        # print(new_text)
        # print('------delta message-----------')
        # print(delta_message)
        return StreamingParseResult(
            normal_text=delta_message.content if delta_message.content else "",
            calls=[
                ToolCallItem(
                    tool_index=t.index,
                    name=t.function.name if t.function else None,
                    parameters=t.function.arguments if t.function else None,
                )
                for t in delta_message.tool_calls
            ],
        )

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call"""
        self._buf = ""
        # self._in_tool_call = False
        # self._function_name_sent = False
        # self._current_function_name = ""
        # self._current_parameters = {}
        # self._streamed_parameters = {}
        self.parser.reset_streaming_state()

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :].lstrip("\n").rstrip("\n")
                params[pname] = _safe_val(pval)
            raw = {"name": fname, "arguments": params}
            try:
                # TODO: fix idx in function call, the index for a function
                # call will always be -1 in parse_base_json
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.tool_call_start_token.replace("\n", "\\n"),
            individual_call_end_token=self.tool_call_end_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
            key_value_separator="\\n",
        )

"""
用于处理多轮对话数据，将其转换为OpenAI chat格式。

支持的输入格式 (JSONL):
{
    "SessionId": "会话唯一标识",
    "SystemPrompt": "可选的系统提示信息",
    "Timestamp": "消息时间戳(毫秒)",
    "Query": "用户的搜索关键词",  # 用于构建对话历史
    "Answer": "历史对话中的助手回复", # 用于构建对话历史
    "Prompt": "搜索结果"
}

输出格式 (JSONL):
[
    {
        "role": "system",
        "content": "系统提示信息"  # 仅当输入中包含SystemPrompt时
    },
    {
        "role": "user",
        "content": "历史对话中的用户输入1"
    },
    {
        "role": "assistant", 
        "content": "历史对话中的助手回复1"
    },
    ...  # 可能包含多轮历史对话
    {
        "role": "user",
        "content": "当前轮次的用户输入"
    }
]
"""

import json
import argparse
from collections import defaultdict


def process_multi_round(input_file: str, output_file: str, num_samples: int = None) -> None:
    session_data = defaultdict(list)
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            session_data[data["SessionId"]].append(data)

    processed_data = []
    for session_id, messages in session_data.items():
        sorted_messages = sorted(messages, key=lambda x: int(x["Timestamp"]))

        for i, current_message in enumerate(sorted_messages):
            message_list = []

            if current_message["SystemPrompt"]:
                message_list.append({"role": "system", "content": current_message["SystemPrompt"]})

            for j in range(i):
                message_list.extend(
                    [
                        {"role": "user", "content": sorted_messages[j]["Query"]},
                        {"role": "assistant", "content": sorted_messages[j]["Answer"]},
                    ]
                )

            message_list.append({"role": "user", "content": current_message["Prompt"]})
            processed_data.append(message_list)

    if num_samples:
        processed_data = processed_data[:num_samples]

    with open(output_file, "w", encoding="utf-8") as f:
        for messages in processed_data:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="处理多轮对话数据")
    parser.add_argument("--input", type=str, required=True, help="输入的jsonl文件路径")
    parser.add_argument("--output", type=str, help="输出的jsonl文件路径，默认为输入文件同目录下的 xxx_formatted.jsonl")
    parser.add_argument("--num-samples", type=int, help="指定输出多少条数据")

    args = parser.parse_args()

    if not args.output:
        input_path = args.input
        base_path = input_path.rsplit(".", 1)[0]
        args.output = f"{base_path}_formatted.jsonl"

    process_multi_round(args.input, args.output, args.num_samples)


if __name__ == "__main__":
    main()

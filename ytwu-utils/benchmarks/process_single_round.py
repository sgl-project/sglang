"""
用于处理单轮对话数据，将其转换为OpenAI chat格式。

支持的输入格式 (JSONL):
{
    "SystemPrompt": "可选的系统提示信息",
    "Prompt": "用户输入的消息内容"
}

输出格式 (JSONL):
[
    {
        "role": "system",
        "content": "系统提示信息"  # 仅当输入中包含SystemPrompt时
    },
    {
        "role": "user",
        "content": "用户输入的消息内容"
    }
]
"""

import json
import argparse


def process_single_round(input_file: str, output_file: str, num_samples: int = None) -> None:
    processed_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())

            message_list = []
            if data.get("SystemPrompt"):
                message_list.append({"role": "system", "content": data["SystemPrompt"]})
            message_list.append({"role": "user", "content": data["Prompt"]})
            processed_data.append(message_list)

    if num_samples:
        processed_data = processed_data[:num_samples]

    with open(output_file, "w", encoding="utf-8") as f:
        for messages in processed_data:
            f.write(json.dumps(messages, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="处理单轮对话数据")
    parser.add_argument("--input", type=str, required=True, help="输入的jsonl文件路径")
    parser.add_argument("--output", type=str, help="输出的jsonl文件路径，默认为输入文件同目录下的 xxx_formatted.jsonl")
    parser.add_argument("--num-samples", type=int, help="指定输出多少条数据")

    args = parser.parse_args()

    if not args.output:
        input_path = args.input
        base_path = input_path.rsplit(".", 1)[0]
        args.output = f"{base_path}_formatted.jsonl"

    process_single_round(args.input, args.output, args.num_samples)


if __name__ == "__main__":
    main()

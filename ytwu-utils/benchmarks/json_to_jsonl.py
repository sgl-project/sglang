"""将数组格式的JSON文件转换为JSONL格式

Args:
    input_file: 输入的json文件路径，格式为[{}, {}, ...]
    output_file: 输出的jsonl文件路径
"""

import json
import argparse


def convert_json_to_jsonl(input_file: str, output_file: str) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入文件必须是JSON数组格式")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="将JSON数组转换为JSONL格式")
    parser.add_argument("--input", type=str, required=True, help="输入的json文件路径")
    parser.add_argument("--output", type=str, help="输出的jsonl文件路径，默认为输入文件同目录下的 [input_file_name].jsonl")

    args = parser.parse_args()

    if not args.output:
        input_path = args.input
        base_path = input_path.rsplit(".", 1)[0]
        args.output = f"{base_path}.jsonl"

    convert_json_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()


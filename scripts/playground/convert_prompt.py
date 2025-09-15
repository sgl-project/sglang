import json
import argparse


def convert_messages_to_prompts(messages):
    """
    Convert messages array to multiple prompt formats.
    Returns a list of prompts - one for each user message combined with the system message.
    Format: "Human: {system_message} {user_message}\\nAssistant:"
    """
    system_message = ""
    user_messages = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "system":
            # Replace actual newlines with literal \n strings
            system_message = content.replace("\n", "\\n").replace("\r", "\\r")
        elif role == "user":
            # Replace actual newlines with literal \n strings
            user_content = content.replace("\n", "\\n").replace("\r", "\\r")
            user_messages.append(user_content)

    # Create a prompt for each user message combined with the system message
    prompts = []
    for user_message in user_messages:
        prompt = f"Human: {system_message} {user_message}\\nAssistant:"
        prompts.append(prompt)

    return prompts


def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file and convert messages to prompts.
    """
    converted_count = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())

                # Extract messages from body
                if "body" in data and "messages" in data["body"]:
                    messages = data["body"]["messages"]

                    # Convert to prompt format - returns list of prompts
                    prompts = convert_messages_to_prompts(messages)

                    # Write each prompt to output file
                    for prompt in prompts:
                        outfile.write(prompt + "\n")
                        converted_count += 1

                    if converted_count % 100 == 0:
                        print(f"Processed {converted_count} prompts...")

                else:
                    print(f"Warning: No messages found in line {line_num}")

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")

    print(f"Conversion complete! Processed {converted_count} records.")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL chat messages to prompt format"
    )
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output text file path")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview first 5 conversions without saving",
    )

    args = parser.parse_args()

    if args.preview:
        # Preview mode - show first 5 prompts
        print("Preview mode - showing first 5 prompts:\n")
        prompt_count = 0
        with open(args.input_file, "r", encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                if prompt_count >= 5:
                    break
                try:
                    data = json.loads(line.strip())
                    if "body" in data and "messages" in data["body"]:
                        messages = data["body"]["messages"]
                        prompts = convert_messages_to_prompts(messages)
                        for j, prompt in enumerate(prompts):
                            if prompt_count >= 5:
                                break
                            print(
                                f"--- Prompt {prompt_count + 1} (from conversation {i + 1}, user message {j + 1}) ---"
                            )
                            print(prompt)
                            print()
                            prompt_count += 1
                except Exception as e:
                    print(f"Error in preview line {i + 1}: {e}")
    else:
        # Full conversion
        process_jsonl_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()

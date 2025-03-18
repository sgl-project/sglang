import sglang as sgl
from sglang.srt.code_completion_parser import generate_completion_prompt


def main():
    llm = sgl.Engine(model_path="deepseek-ai/deepseek-coder-1.3b-base")

    prompt = "function sum(a: number, b: number): number {"
    suffix = "}"

    # Specify the `--completion-template` option. completion template should match the code model you use.
    input = generate_completion_prompt(prompt, suffix, "deepseek_coder")

    sampling_params = {
        "max_new_tokens": 50,
        "skip_special_tokens": False,
        "temperature": 0.3,
        "top_p": 0.95,
    }
    result = llm.generate(prompt=input, sampling_params=sampling_params)

    fim_code = result["text"]  # Assume there is only one prompt

    print("==== [FIM] ====")
    print(fim_code)

    llm.shutdown()


if __name__ == "__main__":
    main()

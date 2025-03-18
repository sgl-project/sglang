from openai import OpenAI

from sglang.test.test_utils import is_in_ci

if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

from sglang.utils import terminate_process, wait_for_server


def main():

    # Specify the `--completion-template` option. completion template should match the code model you use.
    server_process, port = launch_server_cmd(
        "python -m sglang.launch_server --model-path deepseek-ai/deepseek-coder-1.3b-base --completion-template deepseek_coder --port 30020 --host 0.0.0.0"
    )

    wait_for_server(f"http://localhost:{port}")

    # Initialize OpenAI-like client
    client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{port}/v1")
    model_name = client.models.list().data[0].id

    prompt = "function sum(a: number, b: number): number {"
    suffix = "}"

    response_non_stream = client.completions.create(
        model=model_name,
        prompt=prompt,
        suffix=suffix,
        temperature=0.3,
        top_p=0.95,
        stream=False,  # Non-streaming
    )

    print("==== [FIM] ====")
    print(response_non_stream.choices[0].text)

    terminate_process(server_process)


if __name__ == "__main__":
    main()

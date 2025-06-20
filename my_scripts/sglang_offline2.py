"""
This example demonstrates how to launch the offline engine.
"""
from PIL import Image
from transformers import AutoTokenizer
import sglang as sgl


def main():
    image_path = "/mnt/petrelfs/shaojie/code/sglang/my_scripts/tmp.jpg"
    model_path = "/mnt/petrelfs/shaojie/code/ckpts/OpenGVLab/InternVL3-8B-hf"
    chat = [
        {
            'content': "You are an AI assistant that rigorously follows this response protocol:\n\n1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.\n\n2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.\n\nEnsure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.",
            'role': 'system',
        },
        {
            'content': '<image>/nBased on the given images, answer the following question.\n\nIn triangle ABC, angle A is 90°, AB = 3 cm, AC = 4 cm. If a circle 𝑺 is centered at vertex A with a radius of 3 cm, what is the positional relationship between line BC and circle 𝑺? Choices: A. Tangent B. Intersecting C. Separate D. Indeterminable.\nPlease answer the question and put the final answer within \\boxed{}.',
            'role': 'user',
        },
    ]
    # answer = "B"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    image = [Image.open(image_path)]
    llm = sgl.Engine(model_path=model_path, chat_template="internvl-2-5")
    output = llm.generate(prompt=prompt_with_chat_template, image_data=image)
    print(output)
    llm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()
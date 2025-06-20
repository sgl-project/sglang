"""
This example demonstrates how to launch the offline engine.
"""
from PIL import Image
import sglang as sgl


def main():
    image = [Image.open("/mnt/petrelfs/shaojie/code/speed/example_image.png")]
    llm = sgl.Engine(model_path="/mnt/petrelfs/shaojie/code/ckpts/OpenGVLab/InternVL3-8B-hf")
    output = llm.generate(prompt="<|im_end|>\n<|im_start|>user\n<image>\nWhat’s in this image?<|im_end|>\n<|im_start|>assistant\n", image_data=image)
    print(output)
    llm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

"""
This example demonstrates how to launch the offline engine.
"""

import sglang as sgl


def main():
    llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
    llm.generate("What is the capital of France?")
    llm.shutdown()


# The __main__ condition is necessary here because we use "spawn" to create subprocesses
# Spawn starts a fresh program every time, if there is no __main__, it will run into infinite loop to keep spawning processes from sgl.Engine
if __name__ == "__main__":
    main()

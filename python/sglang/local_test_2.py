import json

import sglang as sgl

if __name__ == "__main__":
    prompt = "what is AGI?"
    model_path = "/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct/"  # "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"

    # Create an LLM.
    # engine = sgl.Engine(model_path=model_path, random_seed=42)
    # outputs = engine.generate(prompt, {"top_k": 1.0})
    # print(outputs["text"])
    # engine.shutdown()

    runtime = sgl.Runtime(model_path=model_path, random_seed=42)
    outputs = json.loads(runtime.generate(prompt, {"top_k": 1.0}))["text"]
    print(outputs)
    runtime.shutdown()

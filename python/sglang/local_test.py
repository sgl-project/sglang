import sglang as sgl

model_path = "/shared/public/models/Qwen/Qwen2.5-1.5B-Instruct/"

if __name__ == "__main__":
    runtime = sgl.Runtime(model_path=model_path)
    # engine = sgl.Engine(model_path=model_path)

    print(runtime.generate("Who is Steve Jobs?"))
    # bug: default sampling param should be {}
    # print(engine.generate("Who is Steve Jobs?", {}))
    runtime.shutdown()

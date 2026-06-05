from sanic import Sanic, text
from sanic.response import json

import sglang as sgl

engine = None

# Create an instance of the Sanic app
app = Sanic("sanic-server")


# Define an asynchronous route handler
@app.route("/generate", methods=["POST"])
async def generate(request):
    prompt = request.json.get("prompt")
    if not prompt:
        return json({"error": "Prompt is required"}, status=400)

    # async_generate returns a dict
    result = await engine.async_generate(prompt)

    return text(result["text"])


@app.route("/generate_stream", methods=["POST"])
async def generate_stream(request):
    prompt = request.json.get("prompt")

    if not prompt:
        return json({"error": "Prompt is required"}, status=400)

    # async_generate returns a dict
    result = await engine.async_generate(prompt, stream=True)

    # https://sanic.dev/en/guide/advanced/streaming.md#streaming
    # init the response
    response = await request.respond()

    # result is an async generator
    async for chunk in result:
        await response.send(chunk["text"])

    await response.eof()


def run_server():
    global engine
    engine = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
    app.run(host="0.0.0.0", port=8000, single_process=True)


if __name__ == "__main__":
    run_server()

"""
Fake SGLang server for testing the router
"""

from fastapi import FastAPI
import uvicorn
import argparse
from fastapi.responses import Response

app = FastAPI()
server_url = None

@app.get("/get_server_args")
async def get_server_args():
    return {"model_path": "/fake/model_path", "tokenizer_path": "/fake_tokenizer_path", "server_url": server_url}

@app.post("/generate")
async def generate():
    return {"message": "sglang is awesome!!", "server_url": server_url}


# add arg parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()

def launch_server():
    global server_url
    args = parse_args()
    server_url = f"{args.host}:{args.port}"
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    launch_server()
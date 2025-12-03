import os
import sys

# Set env vars
os.environ["SGLANG_PROFILE_SHAPES"] = "1"
os.environ["SGLANG_PROFILE_SHAPES_RANK"] = "0"
os.environ["SGLANG_PROFILE_SHAPES_FILE"] = "test_shapes.jsonl"

# Now import sglang
import sglang as sgl
from sglang.srt.server_args import ServerArgs
import dataclasses

print("Environment variables set:")
print(f"  SGLANG_PROFILE_SHAPES = {os.environ.get('SGLANG_PROFILE_SHAPES')}")
print(f"  SGLANG_PROFILE_SHAPES_RANK = {os.environ.get('SGLANG_PROFILE_SHAPES_RANK')}")
print(f"  SGLANG_PROFILE_SHAPES_FILE = {os.environ.get('SGLANG_PROFILE_SHAPES_FILE')}")

args = ServerArgs(
    model_path="Qwen/Qwen2.5-14B-Instruct",
    tp_size=8,
)

print("\nInitializing engine...")
llm = sgl.Engine(**dataclasses.asdict(args))

print("Running inference...")
outputs = llm.generate(["Hello"], {"max_new_tokens": 10})

print("Shutting down...")
llm.shutdown()

print("\nChecking output file...")
if os.path.exists("test_shapes.jsonl"):
    size = os.path.getsize("test_shapes.jsonl")
    print(f"File size: {size} bytes")
else:
    print("File not created")

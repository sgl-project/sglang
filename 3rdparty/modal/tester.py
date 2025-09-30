"""
Usage: run tests on Modal
How to run:
1. Install Modal `uv pip install modal`
2. Create Modal profile: `modal token new`
3. Export HF token: `export HF_TOKEN=<huggingface-token>`
4. Export your HF token as a modal secret: `modal secret create huggingface-secret HF_TOKEN=$HF_TOKEN`
5. Run test: `modal run /3rdparty/modal/tester.py`
"""
import modal

app = modal.App("sglang-unit-test")

sglang_image =(
    modal.Image.from_registry("lmsysorg/sglang:v0.5.0rc2-cu126")
    .run_commands(["pip uninstall -y sglang", "rm -rf /sgl-workspace/sglang"])
    .run_commands("git clone https://github.com/sgl-project/sglang.git /sglang")
    .run_commands("cd /sglang && pip install -e python[all]")
)

TEST_DIR: str = "/sglang/test/srt/"
TEST_FILE: str = "test_release_memory_occupation"
TEST_CLASS: str = "TestReleaseMemoryOccupation"
TEST_NAME: str | None = None #Test name, e.g. "test_release_and_resume_occupation_with_weights_cpu_backup"

GPU_TYPE = "A10"
GPU_COUNT = 2

HUGGING_FACE_SECRET_NAME: str = "huggingface-secret"

@app.function(
    image=sglang_image,
    cpu=8,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    timeout=5 * 60,
    max_containers=1,
    secrets=[modal.Secret.from_name(HUGGING_FACE_SECRET_NAME)],
)
def sglang_test():
    import subprocess
    try:
        print(f"Running {TEST_FILE}.{TEST_CLASS}.{TEST_NAME}")
        RUN_TEST_CMD = f"python -m unittest -v {TEST_FILE}.{TEST_CLASS}" + (f".{TEST_NAME}" if TEST_NAME else "")
        output = subprocess.check_output(f"cd {TEST_DIR} && {RUN_TEST_CMD}",
            text=True, shell=True, stderr=subprocess.STDOUT)
        print(output)
    except subprocess.CalledProcessError as e:
        print("Command failed with exit status", e.returncode)
        print("Output:", e.output)

@app.local_entrypoint()
def main():
    sglang_test.remote()

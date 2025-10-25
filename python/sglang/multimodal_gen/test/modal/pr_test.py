import modal

app = modal.App()

import os

image_version = os.getenv("IMAGE_VERSION")
image_tag = f"ghcr.io/hao-ai-lab/sgl_diffusion/sgl_diffusion-dev:{image_version}"
print(f"Using image: {image_tag}")

image = (
    modal.Image.from_registry(image_tag, add_python="3.12")
    .run_commands("rm -rf /sgl_diffusion")
    .apt_install(
        "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg"
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
    )
    .run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc")
    .env(
        {
            "PATH": "/root/.cargo/bin:$PATH",
            "BUILDKITE_REPO": os.environ.get("BUILDKITE_REPO", ""),
            "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
            "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", ""),
            "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
        }
    )
)


def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    import os
    import subprocess
    import sys

    git_repo = os.environ.get("BUILDKITE_REPO")
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    pr_number = os.environ.get("BUILDKITE_PULL_REQUEST")

    print(f"Cloning repository: {git_repo}")
    print(f"Target commit: {git_commit}")
    if pr_number:
        print(f"PR number: {pr_number}")

    # For PRs (including forks), use GitHub's PR refs to get the correct commit
    if pr_number and pr_number != "false":
        checkout_command = f"git fetch --prune origin refs/pull/{pr_number}/head && git checkout FETCH_HEAD"
        print(f"Using PR ref for checkout: {checkout_command}")
    else:
        checkout_command = f"git checkout {git_commit}"
        print(f"Using direct commit checkout: {checkout_command}")

    command = f"""
    source $HOME/.local/bin/env &&
    source /opt/venv/bin/activate &&
    git clone {git_repo} /sgl_diffusion &&
    cd /sgl_diffusion &&
    {checkout_command} &&
    uv pip install -e .[test] &&
    {pytest_command}
    """

    result = subprocess.run(
        ["/bin/bash", "-c", command], stdout=sys.stdout, stderr=sys.stderr, check=False
    )

    sys.exit(result.returncode)


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_encoder_tests():
    run_test("pytest ./sgl_diffusion/tests/encoders -vs")


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_vae_tests():
    run_test("pytest ./sgl_diffusion/tests/vaes -vs")


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_transformer_tests():
    run_test("pytest ./sgl_diffusion/tests/transformers -vs")


@app.function(gpu="L40S:2", image=image, timeout=2700)
def run_ssim_tests():
    run_test("pytest ./sgl_diffusion/tests/ssim -vs")


@app.function(gpu="H100:2", image=image, timeout=900)
def run_inference_tests_STA():
    run_test("pytest ./sgl_diffusion/tests/inference/STA -srP")


@app.function(gpu="H100:1", image=image, timeout=900)
def run_precision_tests_STA():
    run_test("python csrc/attn/tests/test_sta.py")


@app.function(gpu="H100:1", image=image, timeout=900)
def run_precision_tests_VSA():
    run_test("python csrc/attn/tests/test_vsa.py")


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_precision_tests_vmoba():
    run_test("pytest csrc/attn/vmoba_attn/tests/test_vmoba_attn.py")


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_inference_tests_vmoba():
    run_test("python sgl_diffusion/tests/inference/vmoba/test_vmoba_inference.py")


@app.function(gpu="L40S:1", image=image, timeout=3600)
def run_inference_lora_tests():
    run_test(
        "pytest ./sgl_diffusion/tests/inference/lora/test_lora_inference_similarity.py -vs"
    )


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_unit_test():
    run_test(
        "pytest ./sgl_diffusion/tests/dataset/ ./sgl_diffusion/tests/workflow/ -vs"
    )

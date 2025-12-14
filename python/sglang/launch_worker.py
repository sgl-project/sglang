import os, subprocess

rank = int(os.environ.get("RANK", 0))
mode = "prefill" if rank == 0 else "decode"
port = int(os.environ.get("PORT0", 30000)) if rank == 0 else int(os.environ.get("PORT1", 30001))
local_rank = int(os.environ.get("LOCAL_RANK", rank))

# Ensure NVSHMEM disables CUDA multicast on platforms where NVLS is unstable
os.environ.setdefault("NVSHMEM_DISABLE_CUDA_MCAST", "0")
os.environ.setdefault("NVSHMEM_NVLS_ENABLE", "1")
os.environ.setdefault("NVSHMEM_USE_NVLS", "1")

cmd = [
    "python", "-m", "sglang.launch_server",
    "--model-path", os.environ.get("MODEL", "/home/yunzhi.nyx/model/Qwen__Qwen2.5-0.5B"),
    "--host", os.environ.get("HOST", "127.0.0.1"),
    "--port", str(port),
    "--tensor-parallel-size", "1",
    "--disaggregation-mode", mode,
    "--disaggregation-bootstrap-port", os.environ.get("BOOTSTRAP", "8997"),
    "--disaggregation-transfer-backend", "nixl",
    "--max-total-tokens", "4096",
    "--mem-fraction-static", "0.5",
    # Bind this instance to the GPU corresponding to LOCAL_RANK so each
    # torchrun process uses a distinct device (0, 1, ...). This avoids
    # cross-device tensors during CUDA graph capture.
    "--base-gpu-id", str(local_rank),
    "--dist-timeout", "120",
    "--trust-remote-code",
    "--device", "cuda",
]
subprocess.run(cmd)

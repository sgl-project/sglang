import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DiffusionWorkspace,
)


def test_diffusion_workspace_reuses_buffers_cpu():
    workspace = DiffusionWorkspace()
    device = torch.device("cpu")

    buf_a = workspace.get_buffer("latent", (2, 3), torch.float32, device)
    buf_b = workspace.get_buffer("latent", (2, 3), torch.float32, device)
    assert buf_a is buf_b

    buf_c = workspace.get_buffer("latent", (2, 4), torch.float32, device)
    assert buf_c is not buf_a

    buf_d = workspace.get_buffer("latent", (2, 4), torch.float16, device)
    assert buf_d is not buf_c

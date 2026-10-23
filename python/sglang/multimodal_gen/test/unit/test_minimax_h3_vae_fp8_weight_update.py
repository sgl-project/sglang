"""An online-FP8 MiniMax-H3 video VAE refuses runtime weight updates.

The updater copies checkpoint tensors into parameters. The FP8 block linears
hold rounded payloads next to per-channel scales, so such a copy would serve a
decoder whose scales no longer match its weights: the VAE rejects the update
before any weight is written, through the same hook the DiT uses, while a
decoder without the FP8 install keeps taking updates.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import MiniMaxH3VideoVAE
from sglang.multimodal_gen.runtime.post_training.weights_updater import WeightsUpdater

REJECTION = "--component-quantizations.video_vae fp8"


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.fp8_installed = False


def _video_vae(*, fp8_installed: bool) -> MiniMaxH3VideoVAE:
    # The constructor builds the released 36-block decoder from a config; the
    # hook under test reads only the decoder's installer flag.
    vae = MiniMaxH3VideoVAE.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.decoder = _Decoder()
    vae.decoder.fp8_installed = fp8_installed
    with torch.no_grad():
        vae.decoder.proj.weight.zero_()
    return vae


def _updater(vae, model_path: str):
    pipeline = SimpleNamespace(modules={"video_vae": vae}, model_path=model_path)
    return WeightsUpdater(pipeline), pipeline


@pytest.mark.parametrize("fp8_installed", [False, True])
def test_tensor_update_respects_fp8_installation(fp8_installed):
    vae = _video_vae(fp8_installed=fp8_installed)
    updater, _ = _updater(vae, "/unused")

    ok, message = updater.update_weights_from_tensor(
        [("decoder.proj.weight", torch.ones(4, 4))], target_modules=["video_vae"]
    )

    assert ok is not fp8_installed
    if fp8_installed:
        assert REJECTION in message
    expected = torch.zeros(4, 4) if fp8_installed else torch.ones(4, 4)
    assert torch.equal(vae.decoder.proj.weight, expected)


def test_a_disk_update_is_rejected_before_any_weight_is_written(tmp_path):
    vae = _video_vae(fp8_installed=True)
    served = tmp_path / "served"
    updater, pipeline = _updater(vae, str(served))
    checkpoint = tmp_path / "new"
    (checkpoint / "video_vae").mkdir(parents=True)
    save_file(
        {"decoder.proj.weight": torch.ones(4, 4)},
        checkpoint / "video_vae" / "model.safetensors",
    )

    # The bare call is the documented default: every module, video_vae included.
    ok, message = updater.update_weights_from_disk(str(checkpoint))

    assert not ok
    assert REJECTION in message
    assert torch.equal(vae.decoder.proj.weight, torch.zeros(4, 4))
    assert pipeline.model_path == str(served)

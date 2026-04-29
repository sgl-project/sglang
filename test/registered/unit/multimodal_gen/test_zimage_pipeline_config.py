from types import SimpleNamespace

import torch

import sglang.multimodal_gen.configs.pipeline_configs.zimage as zimage_config
from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


def test_zimage_negative_prompt_rotary_embeddings_use_negative_prompt_len(
    monkeypatch,
) -> None:
    """Negative CFG branch should build RoPE positions from negative prompt embeds."""
    monkeypatch.setattr(zimage_config, "get_sp_world_size", lambda: 1)

    config = ZImagePipelineConfig()
    pos_seq_len = 19
    neg_seq_len = 45
    batch = SimpleNamespace(
        prompt_embeds=[torch.ones(pos_seq_len, 2560)],
        negative_prompt_embeds=[torch.ones(neg_seq_len, 2560)],
        height=16,
        width=16,
    )

    def rotary_emb(pos_ids):
        return pos_ids

    neg_kwargs = config.prepare_neg_cond_kwargs(
        batch=batch,
        device=torch.device("cpu"),
        rotary_emb=rotary_emb,
        dtype=torch.float32,
    )

    cap_pos_ids, image_pos_ids = neg_kwargs["freqs_cis"]
    neg_cap_padded_len = 64
    assert cap_pos_ids.shape == (neg_cap_padded_len, 3)
    assert image_pos_ids[0].tolist() == [neg_cap_padded_len + 1, 0, 0]

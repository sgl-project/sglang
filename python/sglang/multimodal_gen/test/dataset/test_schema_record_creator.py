import numpy as np

from sgl_diffusion.dataset.dataloader.record_schema import (
    basic_t2v_record_creator,
    i2v_record_creator,
    ode_text_only_record_creator,
    text_only_record_creator,
)
from sgl_diffusion.runtime.pipelines.pipeline_batch_info import PreprocessBatch


def _mk_basic_batch(N: int) -> PreprocessBatch:
    batch = PreprocessBatch(data_type="video")
    batch.video_file_name = [f"vid_{i}" for i in range(N)]
    batch.prompt = [f"caption_{i}" for i in range(N)]
    batch.width = [640 for _ in range(N)]
    batch.height = [360 for _ in range(N)]
    batch.fps = [4 for _ in range(N)]
    batch.num_frames = [2 for _ in range(N)]
    # Latents: shape (N, C, T, H, W); per-record use latents[idx]
    batch.latents = np.zeros((N, 4, 2, 8, 8), dtype=np.float32)
    # Prompt embeds: list of per-record arrays [Seq, Dim]
    batch.prompt_embeds = [np.ones((6, 16), dtype=np.float32) for _ in range(N)]
    return batch


def test_basic_t2v_record_creator_fields():
    N = 2
    batch = _mk_basic_batch(N)

    records = basic_t2v_record_creator(batch)
    assert isinstance(records, list) and len(records) == N

    for i, rec in enumerate(records):
        assert rec["id"] == batch.video_file_name[i]
        # Latents bytes/shape/dtype
        assert isinstance(rec["vae_latent_bytes"], (bytes, bytearray))
        assert rec["vae_latent_shape"] == list(batch.latents[i].shape)
        assert rec["vae_latent_dtype"] == str(batch.latents[i].dtype)
        # Text embedding
        assert isinstance(rec["text_embedding_bytes"], (bytes, bytearray))
        assert rec["text_embedding_shape"] == list(batch.prompt_embeds[i].shape)
        assert rec["text_embedding_dtype"] == str(batch.prompt_embeds[i].dtype)
        # Meta
        assert rec["caption"] == batch.prompt[i]
        assert rec["media_type"] == "video"
        assert rec["width"] == int(batch.width[i])
        assert rec["height"] == int(batch.height[i])
        assert rec["num_frames"] == batch.latents[i].shape[1]


def test_i2v_record_creator_additional_fields():
    N = 3
    batch = _mk_basic_batch(N)
    # image_embeds is a list of length 1, with an array of shape [N, D]
    batch.image_embeds = [np.ones((N, 32), dtype=np.float32)]
    # first frame latent per record
    batch.image_latent = np.zeros((N, 4, 1, 8, 8), dtype=np.float32)
    # pil image per record
    batch.pil_image = np.zeros((N, 8, 8, 3), dtype=np.uint8)

    records = i2v_record_creator(batch)
    assert isinstance(records, list) and len(records) == N

    for i, rec in enumerate(records):
        # clip feature
        assert isinstance(rec["clip_feature_bytes"], (bytes, bytearray))
        assert rec["clip_feature_shape"] == list(batch.image_embeds[0][i].shape)
        assert rec["clip_feature_dtype"] == str(batch.image_embeds[0][i].dtype)
        # first frame latent
        assert isinstance(rec["first_frame_latent_bytes"], (bytes, bytearray))
        assert rec["first_frame_latent_shape"] == list(batch.image_latent[i].shape)
        assert rec["first_frame_latent_dtype"] == str(batch.image_latent[i].dtype)
        # pil image
        assert isinstance(rec["pil_image_bytes"], (bytes, bytearray))
        assert rec["pil_image_shape"] == list(batch.pil_image[i].shape)
        assert rec["pil_image_dtype"] == str(batch.pil_image[i].dtype)


def test_ode_text_only_record_creator():
    video_name = "ex"
    caption = "a prompt"
    text_embedding = np.ones((6, 16), dtype=np.float32)
    traj = np.ones((5, 4, 2, 2), dtype=np.float32)
    tsteps = np.arange(5, dtype=np.float32)

    rec = ode_text_only_record_creator(
        video_name=video_name,
        text_embedding=text_embedding,
        caption=caption,
        trajectory_latents=traj,
        trajectory_timesteps=tsteps,
    )
    assert rec["id"] == f"text_{video_name}"
    assert isinstance(rec["text_embedding_bytes"], (bytes, bytearray))
    assert rec["text_embedding_shape"] == list(text_embedding.shape)
    assert rec["text_embedding_dtype"] == str(text_embedding.dtype)
    assert rec["file_name"] == video_name
    assert rec["caption"] == caption
    assert rec["media_type"] == "text"
    # Trajectory fields
    assert isinstance(rec["trajectory_latents_bytes"], (bytes, bytearray))
    assert rec["trajectory_latents_shape"] == list(traj.shape)
    assert rec["trajectory_latents_dtype"] == str(traj.dtype)
    assert isinstance(rec["trajectory_timesteps_bytes"], (bytes, bytearray))
    assert rec["trajectory_timesteps_shape"] == list(tsteps.shape)
    assert rec["trajectory_timesteps_dtype"] == str(tsteps.dtype)


def test_text_only_record_creator():
    text_name = "note1"
    caption = "a prompt"
    text_embedding = np.ones((7, 16), dtype=np.float32)
    rec = text_only_record_creator(
        text_name=text_name,
        text_embedding=text_embedding,
        caption=caption,
    )
    assert rec["id"] == f"text_{text_name}"
    assert isinstance(rec["text_embedding_bytes"], (bytes, bytearray))
    assert rec["text_embedding_shape"] == list(text_embedding.shape)
    assert rec["text_embedding_dtype"] == str(text_embedding.dtype)
    assert rec["caption"] == caption

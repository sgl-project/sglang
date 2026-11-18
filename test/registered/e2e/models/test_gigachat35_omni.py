"""GigaChat3.5-Omni serving end to end on a tiny random-weight checkpoint.

Writes the release's checkpoint layout (hybrid GDN + MLA MoE LM, InternViT,
conformer) with random bf16 weights next to the release's tokenizer, chat
template and remote processor, so the real weights are never downloaded, then
checks every modality's prompt length against the remote processor's counts.
"""

import base64
import io
import json
import os
import shutil
import tempfile
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import av
import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoProcessor

from sglang.srt.multimodal.processors.gigachat35_omni import (
    GigaChat35OmniMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=500, stage="nightly", runner_config="1-gpu-large")

IMG, VID, AUD = 128004, 128005, 128006
TONE_HZ, SAMPLE_RATE = 440, 16000


class TestGigaChat35OmniTiny(CustomTestCase):
    """Every modality on one server, alone and batched together."""

    # A local directory or a Hub repo; only the files below are fetched.
    processor_source = "ai-sage/GigaChat3.5-Omni-432B-A28B"
    processor_files = [
        "chat_template.jinja",
        "generation_config.json",
        "processing_gigachat35_omni.py",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    other_args = []
    text_config = {
        "model_type": "gigachat3_5",
        "architectures": ["GigaChat35ForCausalLM"],
        "vocab_size": 128256,
        "hidden_size": 256,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 2,
        "full_attention_layers": [1],
        "first_k_dense_replace": 1,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "linear_attention_type": "GigaChat35GatedDeltaNet",
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "linear_gating_type": "gated_rmsnorm_sigmoid_zero_centered",
        "linear_sigmoid_gate_scale": 2.0,
        "norm_type": "ZeroCenteredGatedNorm",
        "layernorm_type": "pre_post",
        "layernorm_gating_weight": 2.0,
        "gated_attention": True,
        "use_mla_scaling_factor": True,
        "swiglu_limit": 10,
        "max_position_embeddings": 131072,
        "rope_theta": 100000,
        "rope_interleave": True,
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 32768,
        },
        "rms_norm_eps": 1e-6,
        "num_nextn_predict_layers": 1,
        "tie_word_embeddings": False,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 2,
    }
    vision_config = {
        "model_type": "intern_vit_6b",
        "image_size": 448,
        "patch_size": 14,
        "hidden_size": 128,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "qkv_bias": True,
        "qk_normalization": False,
        "norm_type": "layer_norm",
        "layer_norm_eps": 1e-6,
    }
    audio_config = {
        "model_type": "gigachat3_5_omni_audio",
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 256,
        "num_mel_bins": 64,
        "subsampling_conv_channels": 128,
    }

    @classmethod
    def setUpClass(cls):
        cls.model_dir = tempfile.TemporaryDirectory()
        cls.write_checkpoint(cls.model_dir.name)
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_dir.name, trust_remote_code=True
        )
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model_dir.name,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code", "--context-length", "8192"]
            + cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

        if hasattr(cls, "model_dir"):
            cls.model_dir.cleanup()

    @classmethod
    def write_checkpoint(cls, path):
        """Writes the config, random weights and the release's processor files.

        Args:
            path: Checkpoint directory.
        """
        source = cls.processor_source

        if not os.path.isdir(source):
            source = snapshot_download(source, allow_patterns=cls.processor_files)

        for name in cls.processor_files:
            shutil.copy(os.path.join(source, name), path)

        config = {
            "architectures": ["GigaChat35OmniForConditionalGeneration"],
            "model_type": "gigachat3_5_omni",
            "text_config": cls.text_config,
            "vision_config": cls.vision_config,
            "audio_config": cls.audio_config,
            "downsample_ratio": 0.5,
            "audio_stack_factor": 4,
            "video_tokens_per_frame": 77,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
        }

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)

        generator = torch.Generator().manual_seed(0)
        weights = {
            name: (0.02 * torch.randn(shape, generator=generator)).to(torch.bfloat16)
            for name, shape in cls.weight_shapes().items()
        }
        save_file(weights, os.path.join(path, "model.safetensors"), {"format": "pt"})

    @classmethod
    def weight_shapes(cls):
        """Lists every checkpoint tensor with its shape, in the release's names.

        Returns:
            Tensor name to shape.
        """
        t, v, a = cls.text_config, cls.vision_config, cls.audio_config
        d, heads = t["hidden_size"], t["num_attention_heads"]
        shapes = {
            "language_model.model.embed_tokens.weight": (t["vocab_size"], d),
            "language_model.lm_head.weight": (t["vocab_size"], d),
        }
        # Every norm is an RMSNorm with a rank-16 sigmoid gate.
        norms = {"language_model.model.norm": d}
        num_layers = t["num_hidden_layers"]

        # MTP layers follow the LM layers: MLA attention and a dense MLP.
        for i in range(num_layers + t["num_nextn_predict_layers"]):
            p = f"language_model.model.layers.{i}."
            is_mtp = i >= num_layers

            for norm in (
                "input",
                "post_attention",
                "post_feedforward",
                "post_self_attn",
            ):
                norms[f"{p}{norm}_layernorm"] = d

            if is_mtp:
                norms |= {f"{p}{norm}": d for norm in ("enorm", "hnorm")}
                norms[f"{p}shared_head.norm"] = d
                shapes[f"{p}eh_proj.weight"] = (d, 2 * d)

            if i in t["full_attention_layers"] or is_mtp:
                q_head = t["qk_nope_head_dim"] + t["qk_rope_head_dim"]
                kv_head = t["qk_nope_head_dim"] + t["v_head_dim"]
                value = heads * t["v_head_dim"]
                norms[f"{p}self_attn.q_a_layernorm"] = t["q_lora_rank"]
                norms[f"{p}self_attn.kv_a_layernorm"] = t["kv_lora_rank"]
                shapes |= {
                    f"{p}self_attn.q_a_proj.weight": (t["q_lora_rank"], d),
                    f"{p}self_attn.q_b_proj.weight": (heads * q_head, t["q_lora_rank"]),
                    f"{p}self_attn.kv_a_proj_with_mqa.weight": (
                        t["kv_lora_rank"] + t["qk_rope_head_dim"],
                        d,
                    ),
                    f"{p}self_attn.kv_b_proj.weight": (
                        heads * kv_head,
                        t["kv_lora_rank"],
                    ),
                    f"{p}self_attn.o_proj.weight": (d, value),
                    f"{p}self_attn.gate_proj.weight": (value, d),
                }
            else:
                nv = t["linear_num_value_heads"]
                key = t["linear_num_key_heads"] * t["linear_key_head_dim"]
                value = nv * t["linear_value_head_dim"]
                shapes |= {
                    f"{p}self_attn.in_proj_qkvz.weight": (2 * key + 2 * value, d),
                    f"{p}self_attn.in_proj_ba.weight": (2 * nv, d),
                    f"{p}self_attn.conv1d.weight": (
                        2 * key + value,
                        1,
                        t["linear_conv_kernel_dim"],
                    ),
                    f"{p}self_attn.A_log": (nv,),
                    f"{p}self_attn.dt_bias": (nv,),
                    f"{p}self_attn.norm.weight": (t["linear_value_head_dim"],),
                    f"{p}self_attn.out_proj.weight": (d, value),
                }

            if i < t["first_k_dense_replace"] or is_mtp:
                mlps = {f"{p}mlp": t["intermediate_size"]}
            else:
                experts = t["n_routed_experts"]
                shared = t["n_shared_experts"] * t["moe_intermediate_size"]
                mlps = {f"{p}mlp.shared_experts": shared} | {
                    f"{p}mlp.experts.{e}": t["moe_intermediate_size"]
                    for e in range(experts)
                }
                shapes |= {
                    f"{p}mlp.gate.weight": (experts, d),
                    f"{p}mlp.gate.e_score_correction_bias": (experts,),
                }

            for prefix, inner in mlps.items():
                shapes |= {
                    f"{prefix}.gate_proj.weight": (inner, d),
                    f"{prefix}.up_proj.weight": (inner, d),
                    f"{prefix}.down_proj.weight": (d, inner),
                }

        for prefix, size in norms.items():
            shapes |= {
                f"{prefix}.weight": (size,),
                f"{prefix}.gate_up_projection.weight": (16, size),
                f"{prefix}.gate_down_projection.weight": (size, 16),
            }

        h, patch = v["hidden_size"], v["patch_size"]
        num_patches = (v["image_size"] // patch) ** 2
        shapes |= {
            "vision_tower.embeddings.class_embedding": (1, 1, h),
            "vision_tower.embeddings.patch_embedding.weight": (h, 3, patch, patch),
            "vision_tower.embeddings.patch_embedding.bias": (h,),
            "vision_tower.embeddings.position_embedding": (1, num_patches + 1, h),
            "vision_projector.0.weight": (4 * h,),
            "vision_projector.0.bias": (4 * h,),
            "vision_projector.1.weight": (d, 4 * h),
            "vision_projector.1.bias": (d,),
            "vision_projector.3.weight": (d, d),
            "vision_projector.3.bias": (d,),
        }

        for i in range(v["num_hidden_layers"]):
            p = f"vision_tower.encoder.layers.{i}."
            shapes |= {
                f"{p}attn.qkv.weight": (3 * h, h),
                f"{p}attn.qkv.bias": (3 * h,),
                f"{p}attn.proj.weight": (h, h),
                f"{p}attn.proj.bias": (h,),
                f"{p}mlp.fc1.weight": (v["intermediate_size"], h),
                f"{p}mlp.fc1.bias": (v["intermediate_size"],),
                f"{p}mlp.fc2.weight": (h, v["intermediate_size"]),
                f"{p}mlp.fc2.bias": (h,),
            }
            shapes |= {f"{p}{name}": (h,) for name in ("ls1", "ls2")}
            shapes |= {
                f"{p}{norm}.{kind}": (h,)
                for norm in ("norm1", "norm2")
                for kind in ("weight", "bias")
            }

        h, channels = a["hidden_size"], a["subsampling_conv_channels"]
        shapes |= {
            "audio_tower.pre_encode.conv.0.weight": (channels, a["num_mel_bins"], 5),
            "audio_tower.pre_encode.conv.0.bias": (channels,),
            "audio_tower.pre_encode.conv.2.weight": (h, channels, 5),
            "audio_tower.pre_encode.conv.2.bias": (h,),
            "audio_projector.weight": (d, 4 * h),
            "audio_projector.bias": (d,),
        }

        for i in range(a["num_hidden_layers"]):
            p = f"audio_tower.layers.{i}."
            linears = {
                "feed_forward1.linear1": (a["intermediate_size"], h),
                "feed_forward1.linear2": (h, a["intermediate_size"]),
                "feed_forward2.linear1": (a["intermediate_size"], h),
                "feed_forward2.linear2": (h, a["intermediate_size"]),
                "self_attn.linear_q": (h, h),
                "self_attn.linear_k": (h, h),
                "self_attn.linear_v": (h, h),
                "self_attn.linear_out": (h, h),
                "conv.pointwise_conv1": (2 * h, h, 1),
                "conv.depthwise_conv": (h, 1, 5),
                "conv.pointwise_conv2": (h, h, 1),
            }
            norms = [
                "norm_feed_forward1",
                "norm_self_att",
                "norm_conv",
                "conv.norm",
                "norm_feed_forward2",
                "norm_out",
            ]
            shapes |= {f"{p}{name}.weight": shape for name, shape in linears.items()}
            shapes |= {f"{p}{name}.bias": (s[0],) for name, s in linears.items()}
            shapes |= {
                f"{p}{norm}.{kind}": (h,)
                for norm in norms
                for kind in ("weight", "bias")
            }

        return shapes

    def tone(self, seconds):
        """Builds a mono sine at 16 kHz.

        Args:
            seconds: Length.

        Returns:
            Samples in [-0.2, 0.2], float32.
        """
        samples = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        return (0.2 * np.sin(2 * np.pi * TONE_HZ * samples)).astype(np.float32)

    def image_bytes(self):
        """Encodes a 448x448 PNG.

        Returns:
            The file.
        """
        output = io.BytesIO()
        Image.new("RGB", (448, 448), (120, 30, 200)).save(output, format="PNG")
        return output.getvalue()

    def audio_bytes(self):
        """Encodes a 2 s 16 kHz mono WAV.

        Returns:
            The file.
        """
        output = io.BytesIO()

        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((self.tone(2) * 32767).astype(np.int16).tobytes())

        return output.getvalue()

    def video_bytes(self):
        """Encodes a 3 s, 4 fps MP4 with an AAC soundtrack.

        Returns:
            The file.
        """
        output = io.BytesIO()

        with av.open(output, mode="w", format="mp4") as container:
            video = container.add_stream("mpeg4", rate=4)
            video.width = video.height = 64
            video.pix_fmt = "yuv420p"
            audio = container.add_stream("aac", rate=SAMPLE_RATE)
            audio.layout = "mono"

            for i in range(12):
                pixels = np.full((64, 64, 3), 20 * i, np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts, frame.time_base = i, Fraction(1, 4)
                container.mux(video.encode(frame))

            frame = av.AudioFrame.from_ndarray(
                self.tone(3)[None], format="fltp", layout="mono"
            )
            frame.sample_rate = SAMPLE_RATE
            container.mux(video.encode(None) + audio.encode(frame) + audio.encode(None))

        return output.getvalue()

    @staticmethod
    def data_url(mime, data):
        """Wraps a file into a base64 data URL.

        Args:
            mime: Media type.
            data: File content.

        Returns:
            The URL.
        """
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"

    def stamp_tokens(self, seconds):
        """Counts the tokens of one `HH:MM:SS` stamp.

        Args:
            seconds: Time in seconds.

        Returns:
            Token count.
        """
        text = GigaChat35OmniMultimodalProcessor.seconds_to_hms(seconds)
        return len(self.processor.tokenizer.encode(text, add_special_tokens=False))

    def cases(self):
        """Builds one /generate request per modality with its expected prompt length.

        The expected lengths come from the remote processor: 256 tokens per
        image crop; a video keeps 256 + 77 * (T - 1) frame tokens, plus a stamp
        per frame and the soundtrack tokens with `use_audio_in_video`; an audio
        clip under a minute gets one stamp, its tokens and its duration.

        Returns:
            Request name to `(payload, expected prompt tokens)`, and the data URLs
            of the image, audio and video.
        """
        p = self.processor
        text = p.tokenizer.encode("Describe this:", add_special_tokens=False)
        image, video, audio = self.image_bytes(), self.video_bytes(), self.audio_bytes()

        image_tokens = 256 * p.preprocess_image(Image.open(io.BytesIO(image))).shape[0]

        waveform, duration = p.load_audio(audio)
        clip_tokens = p.num_audio_tokens(p.extract_mel(waveform).shape[0])
        self.assertLess(clip_tokens, 375)
        audio_tokens = self.stamp_tokens(0) + clip_tokens + self.stamp_tokens(duration)

        frames = p.load_video(video, use_audio=True)
        self.assertIsNotNone(frames["audio"])
        frame_tokens = 256 + 77 * (len(frames["timestamps"]) - 1)
        soundtrack_tokens = p.num_audio_tokens(p.extract_mel(frames["audio"]).shape[0])
        stamps = sum(self.stamp_tokens(t) for t in frames["timestamps"])
        urls = {
            "image": self.data_url("image/png", image),
            "audio": self.data_url("audio/wav", audio),
            "video": self.data_url("video/mp4", video),
        }
        cases = {
            "text": ({"input_ids": text}, len(text)),
            "image": (
                {"input_ids": text + [IMG], "image_data": [urls["image"]]},
                len(text) + image_tokens,
            ),
            "audio": (
                {"input_ids": text + [AUD], "audio_data": [urls["audio"]]},
                len(text) + audio_tokens,
            ),
            "video": (
                {"input_ids": text + [VID], "video_data": [urls["video"]]},
                len(text) + frame_tokens,
            ),
            "video_with_soundtrack": (
                {
                    "input_ids": text + [VID],
                    "video_data": [urls["video"]],
                    "use_audio_in_video": True,
                },
                len(text) + frame_tokens + stamps + soundtrack_tokens,
            ),
        }
        return cases, urls

    def generate(self, payload):
        """Sends one greedy /generate request.

        Args:
            payload: Request body without sampling parameters.

        Returns:
            The response.
        """
        return requests.post(
            self.base_url + "/generate",
            json=payload | {"sampling_params": {"max_new_tokens": 8, "temperature": 0}},
            timeout=300,
        )

    def check(self, response, expected):
        """Asserts that a /generate request passed with the expected prompt length.

        Args:
            response: The response.
            expected: Expected prompt tokens.
        """
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["meta_info"]["prompt_tokens"], expected)

    def test_every_modality_has_the_processor_layout(self):
        cases, _ = self.cases()

        for name, (payload, expected) in cases.items():
            with self.subTest(name=name):
                self.check(self.generate(payload), expected)

    def test_modalities_batched_together(self):
        cases, _ = self.cases()

        with ThreadPoolExecutor(len(cases)) as pool:
            responses = list(pool.map(self.generate, [p for p, _ in cases.values()]))

        for (name, (_, expected)), response in zip(cases.items(), responses):
            with self.subTest(name=name):
                self.check(response, expected)

    def test_chat_completion_with_every_modality(self):
        """The chat template's file parts reach the processor as media placeholders."""
        cases, urls = self.cases()
        text_tokens = cases["text"][1]
        media_tokens = sum(
            cases[name][1] - text_tokens
            for name in ("image", "audio", "video_with_soundtrack")
        )
        content = [
            {"type": "image_url", "image_url": {"url": urls["image"]}},
            {"type": "video_url", "video_url": {"url": urls["video"]}},
            {"type": "audio_url", "audio_url": {"url": urls["audio"]}},
            {"type": "text", "text": "What is in these files?"},
        ]
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 8,
                "use_audio_in_video": True,
                "return_prompt_token_ids": True,
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["usage"]["prompt_tokens"], media_tokens)
        # The template writes the BOS; the processor must not add another.
        prompt_ids = response.json()["choices"][0]["prompt_token_ids"]
        self.assertEqual(prompt_ids[0], 1)
        self.assertNotEqual(prompt_ids[1], 1)


class TestGigaChat35OmniTinyChunkedPrefill(TestGigaChat35OmniTiny):
    """Prefill chunks shorter than one image, video or audio clip."""

    other_args = ["--chunked-prefill-size", "128"]


class TestGigaChat35OmniTinyMTP(TestGigaChat35OmniTiny):
    """Multimodal prompts with the checkpoint's MTP layer as the EAGLE draft."""

    other_args = [
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
    ]


if __name__ == "__main__":
    unittest.main()

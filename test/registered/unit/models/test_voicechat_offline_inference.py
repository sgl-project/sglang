import asyncio
import base64
import wave
from array import array
from types import SimpleNamespace

import pytest

from examples.voicechat import offline_inference


class FakeEngine:
    def __init__(self, results):
        self.results = iter(results)
        self.closed = []

    def open_session(self, capacity, streaming):
        assert streaming
        return f"session-{id(self)}"

    def generate(self, **_kwargs):
        return next(self.results)

    def close_session(self, session_id):
        self.closed.append(session_id)

    async def async_open_session(self, capacity, streaming):
        return self.open_session(capacity, streaming)

    async def async_generate(self, **kwargs):
        return self.generate(**kwargs)

    async def async_close_session(self, session_id):
        self.close_session(session_id)


class FakeSidecar:
    def __init__(self):
        self.events = []
        self.fail_encode = False

    def start(self):
        self.events.append("start")
        return "audio-session"

    def encode(self, session_id, pcm16):
        assert session_id == "audio-session"
        assert len(base64.b64decode(pcm16)) == offline_inference.FRAME_BYTES
        self.events.append("encode")
        if self.fail_encode:
            raise RuntimeError("encode failed")
        return [[0.0] * 4480]

    def decode(self, session_id, codes):
        assert session_id == "audio-session"
        self.events.append(("decode", codes))
        pcm = array("h", [len(self.events)] * 1764).tobytes()
        return {
            "pcm16": base64.b64encode(pcm).decode(),
            "sample_rate": offline_inference.OUTPUT_RATE,
            "samples": 1764,
        }

    def close(self, session_id):
        assert session_id == "audio-session"
        self.events.append("close")


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(str(token) for token in tokens)


def make_runtime(*, capacity=16, steps=2):
    duplex = FakeEngine(
        [
            {
                "output_ids": [20 + index],
                "meta_info": {"function_tokens": [30 + index]},
            }
            for index in range(steps)
        ]
    )
    eartts = FakeEngine(
        [{"output_ids": [0]}]
        + [
            {
                "output_ids": [0],
                "meta_info": {"audio_codes": [[index, index + 1]]},
            }
            for index in range(steps)
        ]
    )
    runtime = SimpleNamespace(
        sidecar=FakeSidecar(),
        duplex=duplex,
        eartts=eartts,
        session_capacity=capacity,
        speaker=SimpleNamespace(shape=(2, 1152)),
        config=SimpleNamespace(pad_token_id=0),
        tokenizer=FakeTokenizer(),
        prompt_ids=lambda _prompt: [1, 2],
    )
    return runtime


def test_direct_offline_inference_runs_file_and_trailing_silence_frames():
    runtime = make_runtime()

    result = offline_inference.run_offline_inference(
        runtime,
        bytes(offline_inference.FRAME_BYTES),
        system_prompt="test prompt",
        trailing_silence=0.08,
    )

    assert result.text == "20 21"
    assert result.function_text == "30 31"
    assert result.text_tokens == [20, 21]
    assert result.function_tokens == [30, 31]
    assert result.frames == 2
    assert len(result.audio_pcm16) == 2 * 1764 * 2
    assert runtime.sidecar.events[0] == "start"
    assert runtime.sidecar.events[-1] == "close"
    assert runtime.sidecar.events.count("encode") == 2
    assert [event for event in runtime.sidecar.events if isinstance(event, tuple)] == [
        ("decode", [0, 1]),
        ("decode", [1, 2]),
    ]
    assert len(runtime.duplex.closed) == len(runtime.eartts.closed) == 1


def test_context_budget_failure_releases_both_session_types():
    runtime = make_runtime(capacity=4, steps=0)

    with pytest.raises(ValueError, match="requires 2 frames.*permits only 1"):
        offline_inference.run_offline_inference(
            runtime,
            bytes(offline_inference.FRAME_BYTES * 2),
            trailing_silence=0,
        )

    assert runtime.sidecar.events == ["start", "close"]
    assert len(runtime.duplex.closed) == len(runtime.eartts.closed) == 1


def test_pipeline_failure_cancels_a_blocked_input_producer():
    runtime = make_runtime(steps=10)
    runtime.sidecar.fail_encode = True

    async def run():
        return await asyncio.wait_for(
            offline_inference.async_run_offline_inference(
                runtime,
                bytes(offline_inference.FRAME_BYTES * 10),
                trailing_silence=0,
            ),
            timeout=1,
        )

    with pytest.raises(RuntimeError, match="encode failed"):
        asyncio.run(run())

    assert runtime.sidecar.events == ["start", "encode", "close"]
    assert len(runtime.duplex.closed) == len(runtime.eartts.closed) == 1


def test_model_input_pads_partial_frame_and_appends_whole_silence_frames():
    model_input = offline_inference._model_input(b"\x01\x02", 0.16)

    assert len(model_input) == 3 * offline_inference.FRAME_BYTES
    assert model_input[:2] == b"\x01\x02"
    assert set(model_input[2:]) == {0}


def test_save_outputs_match_reference_names_and_audio_layout(tmp_path):
    input_pcm = array("h", [100, -100] * 80).tobytes()
    output_pcm = array("h", [200, -200] * 100).tobytes()
    result = offline_inference.OfflineVoiceChatResult(
        text="hello",
        function_text="",
        text_tokens=[1],
        function_tokens=[0],
        audio_pcm16=output_pcm,
        frames=1,
    )

    paths = offline_inference.save_offline_outputs(
        result, input_pcm, tmp_path, tmp_path / "sample.wav"
    )

    assert {path.name for path in paths.values()} == {
        "sample_output.txt",
        "sample_output.wav",
        "sample_combined.wav",
        "sample_output.json",
    }
    assert paths["text"].read_text() == "hello"
    with wave.open(str(paths["output"]), "rb") as output:
        assert output.getnchannels() == 1
        assert output.getframerate() == offline_inference.OUTPUT_RATE
        assert output.readframes(output.getnframes()) == output_pcm
    with wave.open(str(paths["combined"]), "rb") as combined:
        assert combined.getnchannels() == 2
        assert combined.getframerate() == offline_inference.OUTPUT_RATE


def test_offline_parser_reuses_runtime_defaults():
    args = offline_inference.build_parser().parse_args(
        [
            "--duplex-model",
            "/models/duplex",
            "--eartts-model",
            "/models/eartts",
            "--wav",
            "sample.wav",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.audio_sidecar == "http://127.0.0.1:18081"
    assert args.context_length == 8192
    assert args.trailing_silence == 2.0
    assert args.skip_warmup
    assert not hasattr(args, "host")
    assert not hasattr(args, "max_audio_queue_frames")


def test_offline_parser_allows_explicit_warmup():
    args = offline_inference.build_parser().parse_args(
        [
            "--duplex-model",
            "/models/duplex",
            "--eartts-model",
            "/models/eartts",
            "--wav",
            "sample.wav",
            "--output-dir",
            "outputs",
            "--warmup",
        ]
    )

    assert not args.skip_warmup

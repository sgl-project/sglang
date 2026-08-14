import wave
from array import array

from examples.voicechat import client


def _write_wav(path, rate, samples):
    pcm = array("h", samples)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def test_read_wav_preserves_16khz_pcm(tmp_path):
    path = tmp_path / "input.wav"
    samples = [0, 100, -100, 32767, -32768]
    _write_wav(path, client.INPUT_RATE, samples)

    assert client.read_wav(path) == array("h", samples).tobytes()


def test_read_wav_resamples_24khz_to_16khz(tmp_path):
    path = tmp_path / "input.wav"
    _write_wav(path, 24_000, list(range(24)))

    output = array("h")
    output.frombytes(client.read_wav(path))

    assert len(output) == 16
    assert list(output[:4]) == [0, 2, 3, 4]


def test_resample_pcm16_uses_exact_80ms_device_frame_sizes():
    microphone = array("h", range(1_920)).tobytes()
    model_input = client.resample_pcm16(microphone, 24_000, client.INPUT_RATE)
    assert len(model_input) // 2 == client.FRAME_SAMPLES

    model_output = array("h", range(1_764)).tobytes()
    playback = client.resample_pcm16(model_output, client.OUTPUT_RATE, 48_000)
    assert len(playback) // 2 == 3_840


def test_complete_frames_pads_only_the_final_frame():
    complete = bytes(client.FRAME_BYTES)
    assert client._complete_frames(complete) is complete

    partial = bytes(client.FRAME_BYTES + 2)
    padded = client._complete_frames(partial)
    assert len(padded) == client.FRAME_BYTES * 2
    assert padded.startswith(partial)


def test_client_defaults_to_microphone_with_playback():
    args = client.build_parser().parse_args(
        ["--url", "ws://localhost:18080/v1/realtime"]
    )

    assert args.input_wav is None
    assert not args.no_playback
    assert args.trailing_silence == 2.0

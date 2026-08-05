import argparse
import asyncio
import base64
import io
import json
import time
from statistics import mean, median

import httpx
import librosa
import numpy as np
import soundfile
from datasets import load_dataset
from evaluate import load
from openai import AsyncOpenAI, OpenAI
from transformers import AutoTokenizer


def to_bytes(y, sr):
    buffer = io.BytesIO()
    soundfile.write(buffer, y, sr, format="WAV")
    buffer.seek(0)
    return buffer


async def run_asr_chat(client, model_name, y, sr):
    """Use chat completions API with audio_url for ASR."""
    with to_bytes(y, sr) as f:
        audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

    start_time = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"},
                    }
                ],
            }
        ],
        temperature=0.0,
    )
    end_time = time.perf_counter()

    asr_text = response.choices[0].message.content
    latency = end_time - start_time
    return latency, asr_text


def run_asr_transcription_sync(client, model_name, y, sr, language=None):
    """Use audio transcriptions API for ASR (sync version)."""
    audio_buffer = to_bytes(y, sr)
    audio_buffer.name = "audio.wav"  # OpenAI client needs a name attribute

    start_time = time.perf_counter()
    kwargs = {
        "model": model_name,
        "file": audio_buffer,
    }
    if language:
        kwargs["language"] = language

    transcription = client.audio.transcriptions.create(**kwargs)
    end_time = time.perf_counter()

    latency = end_time - start_time
    return latency, transcription.text


def run_asr_transcription_stream_sync(
    base_url, model_name, y, sr, language=None, show_stream=False
):
    """Use audio transcriptions API with streaming for ASR."""
    audio_buffer = to_bytes(y, sr)
    audio_bytes = audio_buffer.read()

    data = {
        "model": model_name,
        "response_format": "json",
        "stream": "true",
    }
    if language:
        data["language"] = language

    start_time = time.perf_counter()
    text_chunks = []

    if show_stream:
        print("[STREAM] ", end="", flush=True)

    with httpx.stream(
        "POST",
        f"{base_url}/v1/audio/transcriptions",
        data=data,
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        timeout=60.0,
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                try:
                    chunk = json.loads(line[6:])
                    if "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            text_chunks.append(content)
                            if show_stream:
                                print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass

    if show_stream:
        print()  # newline after stream

    end_time = time.perf_counter()
    latency = end_time - start_time
    return latency, "".join(text_chunks)


async def run_asr_transcription(
    client,
    model_name,
    y,
    sr,
    language=None,
    stream=False,
    base_url=None,
    show_stream=False,
):
    """Async wrapper for transcription API (runs sync call in executor)."""
    loop = asyncio.get_event_loop()
    if stream:
        return await loop.run_in_executor(
            None,
            run_asr_transcription_stream_sync,
            base_url,
            model_name,
            y,
            sr,
            language,
            show_stream,
        )
    return await loop.run_in_executor(
        None, run_asr_transcription_sync, client, model_name, y, sr, language
    )


async def bound_asr(
    sem,
    client,
    model_name,
    tokenizer,
    audio,
    reference,
    api_type="chat",
    language=None,
    stream=False,
    base_url=None,
    show_stream=False,
):
    async with sem:
        try:
            if api_type == "transcription":
                latency, text = await run_asr_transcription(
                    client,
                    model_name,
                    *audio,
                    language=language,
                    stream=stream,
                    base_url=base_url,
                    show_stream=show_stream,
                )
            else:
                latency, text = await run_asr_chat(client, model_name, *audio)

            # Calculate tokens for throughput metrics
            num_output_tokens = len(tokenizer(text, add_special_tokens=False).input_ids)

            # Normalize for WER evaluation
            # Whisper tokenizer has a normalize method
            if hasattr(tokenizer, "normalize"):
                out = tokenizer.normalize(text)
                ref = tokenizer.normalize(reference)
            else:
                out = text.lower().strip()
                ref = reference.lower().strip()

            return latency, num_output_tokens, out, ref
        except Exception as e:
            print(f"Error during ASR: {e}")
            return None


async def process_dataset(
    model_name,
    client,
    data,
    concurrent_request,
    api_type="chat",
    language=None,
    stream=False,
    base_url=None,
    show_predictions=False,
):
    sem = asyncio.Semaphore(concurrent_request)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Warmup
    print("Performing warmup...")
    audio_warmup, sr_warmup = (
        data[0]["audio"]["array"],
        data[0]["audio"]["sampling_rate"],
    )
    await bound_asr(
        sem,
        client,
        model_name,
        tokenizer,
        (audio_warmup, sr_warmup),
        "",
        api_type=api_type,
        language=language,
        stream=stream,
        base_url=base_url,
        show_stream=False,  # Don't show stream during warmup
    )

    tasks = []
    print(f"Processing {len(data)} samples...")
    for sample in data:
        audio, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        tasks.append(
            asyncio.create_task(
                bound_asr(
                    sem,
                    client,
                    model_name,
                    tokenizer,
                    (audio, sr),
                    sample["text"],
                    api_type=api_type,
                    language=language,
                    stream=stream,
                    base_url=base_url,
                    show_stream=show_predictions and stream,
                )
            )
        )

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def run_evaluation(args):
    # Use sync client for transcription API, async for chat API
    if args.api_type == "transcription":
        client = OpenAI(base_url=f"{args.base_url}/v1", api_key="None")
    else:
        client = AsyncOpenAI(base_url=f"{args.base_url}/v1", api_key="None")

    print(f"Loading dataset: {args.dataset}...")
    print(f"Using API type: {args.api_type}" + (f" (streaming)" if args.stream else ""))
    dataset = load_dataset(args.dataset, split=args.split)

    # Filter by duration if needed (Whisper max is 30s)
    def add_duration(sample):
        y, sr = sample["audio"]["array"], sample["audio"]["sampling_rate"]
        sample["duration_ms"] = librosa.get_duration(y=y, sr=sr) * 1000
        return sample

    if "duration_ms" not in dataset.column_names:
        dataset = dataset.map(add_duration)

    dataset = dataset.filter(lambda x: x["duration_ms"] < 30000)

    if args.n_examples > 0:
        dataset = dataset.select(range(min(args.n_examples, len(dataset))))

    start = time.perf_counter()
    results = asyncio.run(
        process_dataset(
            args.model,
            client,
            dataset,
            args.concurrency,
            api_type=args.api_type,
            language=args.language,
            stream=args.stream,
            base_url=args.base_url,
            show_predictions=args.show_predictions,
        )
    )
    total_test_time = time.perf_counter() - start

    if not results:
        print("No successful results to evaluate.")
        return

    # Metrics
    latencies = [res[0] for res in results]
    total_tokens = sum([res[1] for res in results])
    predictions = [res[2] for res in results]
    references = [res[3] for res in results]

    wer_metric = load("wer")
    wer_score = 100 * wer_metric.compute(references=references, predictions=predictions)

    print("-" * 30)
    print(f"Results for {args.model}:")
    print(f"Total Requests: {len(results)}")
    print(f"WER: {wer_score:.4f}")
    print(f"Average Latency: {mean(latencies):.4f}s")
    print(f"Median Latency: {median(latencies):.4f}s")
    print(f"95th Latency: {np.percentile(latencies, 95):.4f}s")
    print(f"Throughput: {len(results) / total_test_time:.2f} req/s")
    print(f"Token Throughput: {total_tokens / total_test_time:.2f} tok/s")
    print(f"Total Test Time: {total_test_time:.4f}s")
    print("-" * 30)

    if args.output:
        with open(args.output, "w") as f:
            import json

            json.dump(
                {
                    "model": args.model,
                    "dataset": args.dataset,
                    "wer": wer_score,
                    "avg_latency": mean(latencies),
                    "throughput": len(results) / total_test_time,
                    "token_throughput": total_tokens / total_test_time,
                },
                f,
                indent=2,
            )

    if args.show_predictions:
        print("\n" + "=" * 20 + " Sample Predictions " + "=" * 20)
        num_to_show = min(args.print_n, len(results))
        for i in range(num_to_show):
            print(f"Sample {i+1}:")
            print(f"  REF: {references[i]}")
            print(f"  PRED: {predictions[i]}")
            print("-" * 40)
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sGLang ASR performance.")
    parser.add_argument(
        "--base-url", default="http://localhost:30000", help="sGLang server base URL"
    )
    parser.add_argument(
        "--model", default="openai/whisper-large-v3", help="Model name on the server"
    )
    parser.add_argument(
        "--dataset",
        default="D4nt3/esb-datasets-earnings22-validation-tiny-filtered",
        help="HF dataset repo",
    )
    parser.add_argument("--split", default="validation", help="Dataset split")
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--n-examples",
        "-n",
        type=int,
        default=-1,
        help="Number of examples to test (-1 for all)",
    )
    parser.add_argument("--output", help="Path to save results in JSON")
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="Print sample predictions and references",
    )
    parser.add_argument(
        "--print-n", type=int, default=5, help="Number of sample predictions to print"
    )
    parser.add_argument(
        "--api-type",
        choices=["chat", "transcription"],
        default="chat",
        help="API type to use: 'chat' for chat completions with audio_url, 'transcription' for audio.transcriptions API",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for transcription API (e.g., 'en')",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode for transcription API",
    )
    args = parser.parse_args()

    run_evaluation(args)

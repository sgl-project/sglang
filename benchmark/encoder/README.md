# Encoder Benchmark

Benchmark tools for encoder performance in EPD separation mode.

> The benchmark hits the `/encode` endpoint exposed by `--encoder-only` servers, so the target model must support EPD disaggregation.

## Files

- `bench_encoder.py` - QPS-based benchmark script (image, audio & video)
- `mock_receiver.py` - ZMQ receiver for TCP send testing

## Usage

### Mooncake (Encode-Only)

#### Image Benchmark (Default)

```bash
# Start encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct --port 30000 --encoder-only \
  --encoder-transfer-backend mooncake

# Benchmark with random images
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --backend mooncake --qps 20 --duration 60

# Benchmark with URL image
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --backend mooncake --qps 20 \
  --image-url https://example.com/image.jpg
```

#### Audio Benchmark

```bash
# Benchmark with random audio (1s @ 24kHz)
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --modality audio --qps 10 --duration 60

# Custom audio duration and sample rate
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --modality audio --qps 10 \
  --audio-duration 2.0 --audio-sample-rate 24000 --num-audios 1

# Benchmark with URL audio
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --modality audio --qps 10 \
  --audio-url https://example.com/audio.wav
```

#### Video Benchmark

```bash
# Benchmark with a single video URL
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --modality video --qps 5 --duration 60 \
  --video-url https://example.com/video.mp4

# Benchmark with multiple video URLs
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --modality video --qps 5 --duration 60 \
  --video-url https://example.com/video1.mp4 \
  --video-url https://example.com/video2.mp4
```


### ZMQ (Encode + TCP Send)

```bash
# Terminal 1: Start receiver
python mock_receiver.py --port 12345

# Terminal 2: Start encoder
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-8B-Instruct --port 30000 --encoder-only \
  --encoder-transfer-backend zmq_to_tokenizer

# Terminal 3: Benchmark
python bench_encoder.py \
  --encoder-url http://localhost:30000 \
  --backend zmq --receiver-url tcp://127.0.0.1:12345 --qps 10 --duration 60
```

## Common Options

```
--encoder-url       Encoder server URL (required)
--qps               Target queries per second (required)
--modality          image | audio | video (default: image)
--duration          Test duration in seconds (default: 60)
--warmup            Warmup duration in seconds (default: 5)
--backend           mooncake | zmq (default: mooncake)
--receiver-url      Mock receiver URL for zmq (e.g., tcp://127.0.0.1:12345)
```

### Image Options

```
--image-size        Random image size (default: 448)
--num-images        Images per request (default: 1)
--image-url         Use specified URL instead of random (can repeat)
```

### Audio Options

```
--audio-duration    Duration of random audio in seconds (default: 1.0)
--audio-sample-rate Sample rate of random audio in Hz (default: 24000)
--num-audios        Audio clips per request (default: 1)
--audio-url         Use specified URL instead of random (can repeat)
```

### Video Options

```
--video-url         Video URL to use (required for video, can repeat)
```

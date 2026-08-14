# Prerequisites

Verify the host and client requirements before converting or serving NVIDIA
NemotronLabs VoiceChat with SGLang.

## Server hardware

- Linux on x86_64.
- One NVIDIA GPU with at least 80 GB of VRAM. The reference SGLang deployment
  has been validated on a single H100.
- At least 8 GB of shared memory available to each serving container.
- Enough local storage for the unified checkpoint, two converted stages,
  container images, and temporary conversion data. Plan for at least 100 GB.

The default single-GPU memory split reserves approximately 45% for Duplex and
20% for EarTTS. The remaining memory is used by NeMo perception, the codec,
MaskGIT, and runtime workspaces. Stop unrelated GPU workloads before launch.

Store converted safetensors on a local POSIX filesystem that Docker can read.
Some shared CIFS mounts allow directory listing but fail when safetensors tries
to open or memory-map a shard. Stage the converted directories onto local SSD
storage if a loader reports `No such file or directory` for a shard that exists.

## Server software

- A recent NVIDIA driver compatible with the selected container images.
- Docker Engine with permission to run containers.
- NVIDIA Container Toolkit configured for Docker.
- Git and Git LFS for the SGLang checkout.
- Hugging Face CLI for downloading the unified checkpoint.
- Access to NVIDIA's VoiceChat inference image for the NeMo sidecar and EarTTS
  conversion environment.

Verify Docker GPU access:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

Build an SGLang image from the branch containing VoiceChat support:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
# Check out the branch or commit that contains VoiceChat support.
docker build -t sglang-voicechat -f docker/Dockerfile .
```

The NVIDIA image name used in these instructions is:

```text
nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

Authenticate Docker to `nvcr.io` according to your NGC access policy before
pulling it. Do not place registry credentials in shell history, Dockerfiles,
checked-in environment files, or command-line arguments.

## Model artifacts

The deployment needs:

1. The unified
   [`nvidia/NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
   checkpoint, including `config.json` and `model.safetensors`.
2. A converted `duplex/` directory.
3. A converted `eartts/` directory containing at least one speaker latent.

Follow [Generate SGLang stages](generate-model-repo.md) if items 2 and 3 do
not exist yet.

## Client software

The client supports Linux and macOS. WAV-only operation requires `websockets`:

```bash
python -m pip install websockets
```

Microphone capture and speaker playback additionally require PortAudio and
PyAudio.

Ubuntu or Debian:

```bash
sudo apt-get install portaudio19-dev
python -m pip install pyaudio
```

macOS:

```bash
brew install portaudio
python -m pip install pyaudio
```

## Network and security

The example server listens on TCP port `18080`. Its WebSocket endpoint has no
built-in authentication or TLS. Bind it only to a trusted network, restrict the
port with a firewall, or place it behind an authenticated TLS reverse proxy.
Use `wss://` from remote clients when TLS is terminated by a proxy.

The NeMo sidecar listens on `127.0.0.1:18081` and should not be exposed outside
the server host.

Next: [Generate SGLang stages](generate-model-repo.md)

# SGLang Docker Environment Structure

This directory contains a restructured Docker setup for SGLang that provides:
1. **Shared base image** with common dependencies
2. **Environment-specific images** for different hardware/use cases  
3. **Build system** that supports environment tags

## Directory Structure

```
docker/
├── Dockerfile.base                 # Shared base image with common dependencies
├── envs/                           # Environment-specific Dockerfiles
│   ├── Dockerfile.default          # Standard CUDA environment
│   ├── Dockerfile.gb200            # Blackwell/GB200 optimized
│   ├── Dockerfile.npu             # Ascend NPU environment  
│   ├── Dockerfile.rocm            # AMD ROCm environment
│   ├── Dockerfile.router          # Lightweight router service
│   └── Dockerfile.xeon            # Intel Xeon CPU-only
├── build.sh                       # Build script with environment support
├── docker-compose.envs.yaml       # Docker Compose for all environments
├── docker-compose.prebuilt.yaml   # Docker Compose for pre-built images
├── k8s-sglang-*.yaml              # Kubernetes manifests
└── README.md                      # This documentation
```

## Quick Start

### Using the Build Script

The `build.sh` script provides a convenient way to build images for different environments:

```bash
# Build default environment
./build.sh

# Build GB200 environment with CUDA 12.9.1
./build.sh -e gb200 -c 12.9.1

# Build NPU environment
./build.sh -e npu

# Build ROCm environment with specific GPU architecture  
GPU_ARCH=gfx942 ./build.sh -e rocm

# Build router service only
./build.sh -e router

# Build and push to registry
./build.sh -e default --push
```

### Using Docker Compose

```bash
# Build and run default environment
docker-compose -f docker-compose.envs.yaml --profile default up --build

# Build and run GB200 environment
CUDA_VERSION=12.9.1 docker-compose -f docker-compose.envs.yaml --profile gb200 up --build

# Run router service
docker-compose -f docker-compose.envs.yaml --profile router up -d
```

### Direct Docker Commands

```bash
# Build base image first (for CUDA environments)
docker build -f Dockerfile.base -t sglang/base:12.6.1 --build-arg CUDA_VERSION=12.6.1 ..

# Build specific environment
docker build -f envs/Dockerfile.default -t sglang:default --build-arg CUDA_VERSION=12.6.1 ..
```

## Environment Descriptions

### Default Environment (`default`)
- **Base**: CUDA 12.6.1
- **Features**: Full SGLang with all features, NVSHMEM, DeepEP, sgl-router
- **Use Case**: General development and production workloads
- **GPU**: NVIDIA CUDA GPUs

### GB200 Environment (`gb200`) 
- **Base**: CUDA 12.9.1
- **Features**: Blackwell-optimized SGLang, CUDA architecture 10.0 and 12.0
- **Use Case**: NVIDIA GB200 and Blackwell GPU systems
- **GPU**: NVIDIA Blackwell architecture (GB200, B200, etc.)

### NPU Environment (`npu`)
- **Base**: Ascend CANN runtime
- **Features**: Ascend NPU support via torch_npu
- **Use Case**: Huawei Ascend NPU systems
- **Hardware**: Ascend 910, 310P, etc.

### ROCm Environment (`rocm`)
- **Base**: ROCm development images
- **Features**: AMD GPU support, Triton for ROCm
- **Use Case**: AMD GPU systems (MI200, MI300 series)
- **GPU**: AMD Instinct series

### Router Environment (`router`)
- **Base**: Ubuntu 24.04 minimal
- **Features**: Lightweight sgl-router service only
- **Use Case**: Load balancing and request routing
- **Resource**: CPU-only, minimal footprint

### Xeon Environment (`xeon`)
- **Base**: Ubuntu 22.04
- **Features**: CPU-only SGLang with Intel optimizations
- **Use Case**: Intel Xeon CPU inference
- **Hardware**: Intel Xeon processors

## Build Script Options

The `build.sh` script supports the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `-e, --environment` | Environment to build (default, gb200, npu, rocm, router, xeon) | `default` |
| `-c, --cuda-version` | CUDA version to use | `12.6.1` |
| `-b, --build-type` | Build type (all, blackwell, srt, etc.) | `all` |  
| `-t, --branch-type` | Branch type (local, remote) | `remote` |
| `-p, --tag-prefix` | Tag prefix for images | `sglang` |
| `--push` | Push images to registry after build | `false` |
| `-h, --help` | Show help message | - |

## Environment Variables

### Common Variables
- `CUDA_VERSION`: CUDA version for base image
- `BUILD_TYPE`: SGLang build type
- `BRANCH_TYPE`: Use local source (`local`) or clone from GitHub (`remote`)

### Environment-Specific Variables  

#### ROCm
- `GPU_ARCH`: GPU architecture (gfx942, gfx950, etc.)

#### NPU
- `CANN_VERSION`: CANN runtime version
- `DEVICE_TYPE`: Ascend device type

#### Docker Compose
- `HOST_WORKSPACE`: Host directory to mount as workspace
- `HOST_MODELS`: Host directory for model storage
- `SGL_PORT`: Port for SGLang service
- `CUDA_VISIBLE_DEVICES`: GPU visibility for CUDA environments

## Migration from Old Structure

The old monolithic Dockerfiles have been restructured as follows:

| Old File | New Structure | Status |
|----------|--------------|---------|
| `Dockerfile` | `Dockerfile.base` + `envs/Dockerfile.default` | ✅ Removed |
| `Dockerfile.gb200` | `Dockerfile.base` + `envs/Dockerfile.gb200` | ✅ Removed |
| `Dockerfile.npu` | `envs/Dockerfile.npu` (standalone) | ✅ Removed |
| `Dockerfile.rocm` | `envs/Dockerfile.rocm` (standalone) | ✅ Removed |
| `Dockerfile.router` | `envs/Dockerfile.router` (standalone) | ✅ Removed |
| `Dockerfile.xeon` | `envs/Dockerfile.xeon` (standalone) | ✅ Removed |
| `Dockerfile.sagemaker` | Not migrated (deprecated) | ✅ Removed |
| `compose.yaml` | `docker-compose.prebuilt.yaml` | ✅ Renamed |
| `serve` | Not needed (SageMaker specific) | ✅ Removed |

## Benefits of New Structure

1. **Shared Dependencies**: Common packages installed once in base image
2. **Faster Builds**: Environment images build on top of cached base
3. **Consistency**: All CUDA environments share same base configuration  
4. **Maintainability**: Environment-specific changes isolated to individual files
5. **Flexibility**: Easy to add new environments or modify existing ones
6. **Build Automation**: Single script handles all build scenarios

## Adding New Environments  

To add a new environment:

1. Create `envs/Dockerfile.newenv`
2. Add build logic to `build.sh` 
3. Add service definition to `docker-compose.envs.yaml`
4. Update `AVAILABLE_ENVS` in `build.sh`
5. Document in this README

## Troubleshooting

### Build Issues
- Ensure Docker BuildKit is enabled: `export DOCKER_BUILDKIT=1`
- Check available disk space for layer caching
- For local builds, ensure source code is in parent directory

### Runtime Issues  
- Verify GPU drivers and runtime (CUDA/ROCm/NPU)
- Check device permissions and group membership
- Ensure required environment variables are set

### Common Commands

```bash
# Check built images
docker images | grep sglang

# Clean up intermediate images
docker system prune -f

# View build logs
./build.sh -e default 2>&1 | tee build.log

# Test specific environment
docker run -it --gpus all sglang:default bash
```
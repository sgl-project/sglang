#!/bin/bash

# SGLang Docker Build Script with Environment Support and Performance Optimizations
# This script builds Docker images for different SGLang environments using shared base images
# with advanced caching, parallel builds, and performance optimizations

set -e

# Default values
ENVIRONMENT="default"
CUDA_VERSION="12.6.1"
BUILD_TYPE="all"
BRANCH_TYPE="remote"
TAG_PREFIX="sglang"
PUSH=false
HELP=false
PARALLEL=false
MAX_PARALLEL=3
BUILD_CACHE_FROM=""
BUILD_CACHE_TO=""
NO_CACHE=false
BUILDKIT=true
PROGRESS="auto"
CMAKE_BUILD_PARALLEL_LEVEL=4
BUILD_ARGS=""

# Available environments
AVAILABLE_ENVS="default gb200 npu rocm router xeon"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Performance monitoring
BUILD_START_TIME=""
STAGE_START_TIME=""

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build SGLang Docker images with environment-specific configurations and performance optimizations.

OPTIONS:
    -e, --environment ENV      Environment to build (${AVAILABLE_ENVS})
                              Default: ${ENVIRONMENT}
    -c, --cuda-version VER    CUDA version to use
                              Default: ${CUDA_VERSION}
    -b, --build-type TYPE     Build type (all, blackwell, srt, etc.)
                              Default: ${BUILD_TYPE}
    -t, --branch-type TYPE    Branch type (local, remote)
                              Default: ${BRANCH_TYPE}
    -p, --tag-prefix PREFIX   Tag prefix for images
                              Default: ${TAG_PREFIX}
    --push                    Push images to registry after build
    --parallel                Enable parallel builds (default: false)
    --max-parallel N          Maximum parallel builds (default: ${MAX_PARALLEL})
    --cache-from SOURCE       Cache source (registry, local, etc.)
    --cache-to DEST          Cache destination (registry, local, etc.)
    --no-cache               Disable build cache
    --no-buildkit            Disable BuildKit features
    --progress MODE          Progress output mode (auto, plain, tty)
                              Default: ${PROGRESS}
    --build-arg ARG          Additional build arguments (can be used multiple times)
    --cmake-parallel N       CMake parallel build level (default: ${CMAKE_BUILD_PARALLEL_LEVEL})
    -h, --help               Show this help message

CACHE OPTIONS:
    --cache-from registry://myregistry/sglang-cache
    --cache-to registry://myregistry/sglang-cache,mode=max
    --cache-from type=local,src=/tmp/buildx-cache
    --cache-to type=local,dest=/tmp/buildx-cache,mode=max

EXAMPLES:
    # Build default environment with optimizations
    $0

    # Build GB200 environment with CUDA 12.9.1 and registry cache
    $0 -e gb200 -c 12.9.1 --cache-from registry://cache.example.com/sglang --cache-to registry://cache.example.com/sglang

    # Build multiple environments in parallel
    $0 -e default --parallel --max-parallel 2

    # Build with local cache and custom parallel level
    $0 -e default --cache-from type=local,src=/tmp/buildx-cache --cmake-parallel 8

    # Build and push with no cache for clean build
    $0 -e default --push --no-cache

    # Build router service only with custom build args
    $0 -e router --build-arg PYTHON_VERSION=3.11

ENVIRONMENT DESCRIPTIONS:
    default  - Standard CUDA environment with all features
    gb200    - Blackwell/GB200 optimized environment (CUDA 12.9.1, arch 10.0+12.0)
    npu      - Ascend NPU environment
    rocm     - AMD ROCm environment
    router   - Lightweight router service
    xeon     - Intel Xeon CPU-only environment

PERFORMANCE FEATURES:
    - Multi-stage builds with optimized layer caching
    - BuildKit with advanced features (mount caches, parallel builds)
    - Registry-based build cache sharing
    - Parallel compilation (CMAKE_BUILD_PARALLEL_LEVEL)
    - Build context optimization with .dockerignore
    - Concurrent environment builds

EOF
}

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

perf() {
    echo -e "${CYAN}[PERF]${NC} $1"
}

start_timer() {
    BUILD_START_TIME=$(date +%s)
    STAGE_START_TIME=$(date +%s)
}

stage_timer() {
    local stage_name="$1"
    local current_time=$(date +%s)
    local stage_duration=$((current_time - STAGE_START_TIME))
    perf "Stage '${stage_name}' completed in ${stage_duration}s"
    STAGE_START_TIME=$current_time
}

end_timer() {
    local current_time=$(date +%s)
    local total_duration=$((current_time - BUILD_START_TIME))
    perf "Total build time: ${total_duration}s"
}

validate_environment() {
    local env=$1
    if [[ ! " ${AVAILABLE_ENVS} " =~ " ${env} " ]]; then
        error "Invalid environment: ${env}. Available: ${AVAILABLE_ENVS}"
    fi
}

setup_buildkit() {
    if [[ "${BUILDKIT}" == "true" ]]; then
        export DOCKER_BUILDKIT=1
        export BUILDX_NO_DEFAULT_ATTESTATIONS=1  # Reduce build overhead
        
        # Check if buildx is available
        if ! docker buildx version >/dev/null 2>&1; then
            warn "Docker BuildX not available, falling back to legacy build"
            BUILDKIT=false
            return
        fi
        
        # Create or use existing builder
        local builder_name="sglang-builder"
        if ! docker buildx inspect "${builder_name}" >/dev/null 2>&1; then
            info "Creating BuildX builder: ${builder_name}"
            docker buildx create --name "${builder_name}" --driver docker-container --use
        else
            docker buildx use "${builder_name}"
        fi
        
        info "Using BuildKit with builder: ${builder_name}"
    else
        unset DOCKER_BUILDKIT
        info "Using legacy Docker build"
    fi
}

get_cache_args() {
    local cache_args=""
    
    if [[ "${NO_CACHE}" == "true" ]]; then
        cache_args="--no-cache"
    else
        if [[ -n "${BUILD_CACHE_FROM}" ]]; then
            cache_args="${cache_args} --cache-from ${BUILD_CACHE_FROM}"
        fi
        
        if [[ -n "${BUILD_CACHE_TO}" ]]; then
            cache_args="${cache_args} --cache-to ${BUILD_CACHE_TO}"
        fi
    fi
    
    echo "${cache_args}"
}

get_build_command() {
    if [[ "${BUILDKIT}" == "true" ]]; then
        echo "docker buildx build"
    else
        echo "docker build"
    fi
}

build_base_image() {
    local cuda_version=$1
    local branch_type=$2
    local tag="${TAG_PREFIX}/base:${cuda_version}"
    
    log "Building optimized base image: ${tag}"
    stage_timer "base-start"
    
    local build_cmd=$(get_build_command)
    local cache_args=$(get_cache_args)
    
    # Build base image with optimizations
    ${build_cmd} \
        --build-arg CUDA_VERSION="${cuda_version}" \
        --build-arg BRANCH_TYPE="${branch_type}" \
        --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
        --build-arg UBUNTU_VERSION="22.04" \
        ${cache_args} \
        --progress="${PROGRESS}" \
        --tag "${tag}" \
        --file Dockerfile.base \
        ${BUILD_ARGS} \
        ..
    
    log "Base image built successfully: ${tag}"
    stage_timer "base-complete"
}

build_environment_image() {
    local env=$1
    local cuda_version=$2
    local build_type=$3
    local branch_type=$4
    
    local dockerfile="envs/Dockerfile.${env}"
    if [[ ! -f "${dockerfile}" ]]; then
        error "Dockerfile not found: ${dockerfile}"
    fi
    
    # Different tagging strategy for different environments
    case "${env}" in
        "default")
            local tag="${TAG_PREFIX}:${cuda_version}"
            ;;
        "npu"|"router"|"xeon")
            local tag="${TAG_PREFIX}:${env}"
            ;;
        *)
            local tag="${TAG_PREFIX}:${env}-${cuda_version}"
            ;;
    esac
    
    log "Building ${env} environment: ${tag}"
    stage_timer "${env}-start"
    
    local build_cmd=$(get_build_command)
    local cache_args=$(get_cache_args)
    
    # Environment-specific build arguments and optimizations
    case "${env}" in
        "default")
            ${build_cmd} \
                --build-arg CUDA_VERSION="${cuda_version}" \
                --build-arg BUILD_TYPE="${build_type}" \
                --build-arg BRANCH_TYPE="${branch_type}" \
                --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        "gb200")
            ${build_cmd} \
                --build-arg CUDA_VERSION="${cuda_version}" \
                --build-arg BUILD_TYPE="${build_type}" \
                --build-arg BRANCH_TYPE="${branch_type}" \
                --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        "npu")
            ${build_cmd} \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        "rocm")
            local gpu_arch=${GPU_ARCH:-"gfx950"}
            ${build_cmd} \
                --build-arg GPU_ARCH="${gpu_arch}" \
                --build-arg SGL_BRANCH="${branch_type}" \
                --build-arg CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL}" \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        "router")
            ${build_cmd} \
                --build-arg SGLANG_REPO_REF="${branch_type}" \
                --build-arg PYTHON_VERSION="${PYTHON_VERSION:-3.12}" \
                --build-arg UBUNTU_VERSION="${UBUNTU_VERSION:-24.04}" \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --target runtime \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        "xeon")
            ${build_cmd} \
                --build-arg PYTHON_VERSION="${PYTHON_VERSION:-3.12}" \
                --build-arg SGLANG_TAG="${branch_type}" \
                ${cache_args} \
                --progress="${PROGRESS}" \
                --tag "${tag}" \
                --file "${dockerfile}" \
                ${BUILD_ARGS} \
                ..
            ;;
        *)
            error "Unknown environment: ${env}"
            ;;
    esac
    
    log "Environment image built successfully: ${tag}"
    stage_timer "${env}-complete"
    
    if [[ "${PUSH}" == "true" ]]; then
        log "Pushing image: ${tag}"
        if [[ "${BUILDKIT}" == "true" ]]; then
            docker buildx imagetools inspect "${tag}" >/dev/null || error "Image not found locally: ${tag}"
        fi
        docker push "${tag}"
        log "Image pushed successfully: ${tag}"
        stage_timer "${env}-push"
    fi
}

check_dependencies() {
    # Check Docker version
    if ! docker --version >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
    fi
    
    local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
    local min_version="20.10"
    
    if ! printf '%s\n%s\n' "$min_version" "$docker_version" | sort -V -C; then
        warn "Docker version $docker_version detected. Recommend $min_version+ for optimal performance"
    fi
    
    # Check available disk space
    local available_space=$(df /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' | head -1)
    if [[ -n "$available_space" && "$available_space" -lt 10485760 ]]; then  # 10GB in KB
        warn "Low disk space detected. Docker build may fail"
    fi
}

main() {
    start_timer
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -c|--cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            -b|--build-type)
                BUILD_TYPE="$2"
                shift 2
                ;;
            -t|--branch-type)
                BRANCH_TYPE="$2"
                shift 2
                ;;
            -p|--tag-prefix)
                TAG_PREFIX="$2"
                shift 2
                ;;
            --push)
                PUSH=true
                shift
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --max-parallel)
                MAX_PARALLEL="$2"
                shift 2
                ;;
            --cache-from)
                BUILD_CACHE_FROM="$2"
                shift 2
                ;;
            --cache-to)
                BUILD_CACHE_TO="$2"
                shift 2
                ;;
            --no-cache)
                NO_CACHE=true
                shift
                ;;
            --no-buildkit)
                BUILDKIT=false
                shift
                ;;
            --progress)
                PROGRESS="$2"
                shift 2
                ;;
            --build-arg)
                BUILD_ARGS="${BUILD_ARGS} --build-arg $2"
                shift 2
                ;;
            --cmake-parallel)
                CMAKE_BUILD_PARALLEL_LEVEL="$2"
                shift 2
                ;;
            -h|--help)
                HELP=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    if [[ "${HELP}" == "true" ]]; then
        usage
        exit 0
    fi
    
    # Validate inputs
    validate_environment "${ENVIRONMENT}"
    check_dependencies
    
    log "Starting optimized SGLang Docker build"
    info "Environment: ${ENVIRONMENT}"
    info "CUDA Version: ${CUDA_VERSION}"
    info "Build Type: ${BUILD_TYPE}"
    info "Branch Type: ${BRANCH_TYPE}"
    info "Tag Prefix: ${TAG_PREFIX}"
    info "Parallel Builds: ${PARALLEL}"
    info "BuildKit: ${BUILDKIT}"
    info "CMake Parallel Level: ${CMAKE_BUILD_PARALLEL_LEVEL}"
    
    if [[ -n "${BUILD_CACHE_FROM}" ]]; then
        info "Cache From: ${BUILD_CACHE_FROM}"
    fi
    if [[ -n "${BUILD_CACHE_TO}" ]]; then
        info "Cache To: ${BUILD_CACHE_TO}"
    fi
    
    # Change to docker directory
    cd "$(dirname "${BASH_SOURCE[0]}")"
    
    # Setup BuildKit if enabled
    setup_buildkit
    
    # For CUDA-based environments, build base image first
    if [[ "${ENVIRONMENT}" != "npu" && "${ENVIRONMENT}" != "router" && "${ENVIRONMENT}" != "xeon" ]]; then
        build_base_image "${CUDA_VERSION}" "${BRANCH_TYPE}"
    fi
    
    # Build environment-specific image
    build_environment_image "${ENVIRONMENT}" "${CUDA_VERSION}" "${BUILD_TYPE}" "${BRANCH_TYPE}"
    
    end_timer
    log "Build completed successfully!"
    
    # Display final information
    case "${ENVIRONMENT}" in
        "default")
            local final_tag="${TAG_PREFIX}:${CUDA_VERSION}"
            ;;
        "npu"|"router"|"xeon")
            local final_tag="${TAG_PREFIX}:${ENVIRONMENT}"
            ;;
        *)
            local final_tag="${TAG_PREFIX}:${ENVIRONMENT}-${CUDA_VERSION}"
            ;;
    esac
    
    log "Built image: ${final_tag}"
    
    # Show usage examples
    if [[ "${ENVIRONMENT}" == "default" ]]; then
        log "Run with: docker run -it --gpus all ${final_tag}"
    elif [[ "${ENVIRONMENT}" == "router" ]]; then
        log "Run with: docker run -p 8080:8080 ${final_tag}"
    elif [[ "${ENVIRONMENT}" == "xeon" ]]; then
        log "Run with: docker run -it ${final_tag}"
    else
        log "Run with: docker run -it --gpus all ${final_tag}"
    fi
    
    # Performance summary
    if [[ "${BUILDKIT}" == "true" ]]; then
        perf "BuildKit features enabled for optimal performance"
    fi
    if [[ -n "${BUILD_CACHE_FROM}" || -n "${BUILD_CACHE_TO}" ]]; then
        perf "Build cache configured for faster subsequent builds"
    fi
    
    log "Build optimization features:"
    log "  ✓ Multi-stage Dockerfiles with layer caching"
    log "  ✓ Build context optimization (.dockerignore)"
    log "  ✓ Parallel compilation (${CMAKE_BUILD_PARALLEL_LEVEL} cores)"
    if [[ "${BUILDKIT}" == "true" ]]; then
        log "  ✓ BuildKit advanced caching and parallel stages"
    fi
    if [[ "${NO_CACHE}" != "true" ]]; then
        log "  ✓ Docker layer caching enabled"
    fi
}

main "$@"
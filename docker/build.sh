#!/bin/bash

# SGLang Docker Build Script with Environment Support
# This script builds Docker images for different SGLang environments using shared base images

set -e

# Default values
ENVIRONMENT="default"
CUDA_VERSION="12.6.1"
BUILD_TYPE="all"
BRANCH_TYPE="remote"
TAG_PREFIX="sglang"
PUSH=false
HELP=false

# Available environments
AVAILABLE_ENVS="default gb200 npu rocm router xeon"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build SGLang Docker images with environment-specific configurations.

OPTIONS:
    -e, --environment ENV    Environment to build (${AVAILABLE_ENVS})
                            Default: ${ENVIRONMENT}
    -c, --cuda-version VER  CUDA version to use
                            Default: ${CUDA_VERSION}
    -b, --build-type TYPE   Build type (all, blackwell, srt, etc.)
                            Default: ${BUILD_TYPE}
    -t, --branch-type TYPE  Branch type (local, remote)
                            Default: ${BRANCH_TYPE}
    -p, --tag-prefix PREFIX Tag prefix for images
                            Default: ${TAG_PREFIX}
    --push                  Push images to registry after build
    -h, --help              Show this help message

EXAMPLES:
    # Build default environment
    $0

    # Build GB200 environment with CUDA 12.9.1
    $0 -e gb200 -c 12.9.1

    # Build NPU environment
    $0 -e npu

    # Build ROCm environment with specific GPU architecture
    $0 -e rocm

    # Build router service only
    $0 -e router

    # Build and push to registry
    $0 -e default --push

ENVIRONMENT DESCRIPTIONS:
    default  - Standard CUDA environment with all features
    gb200    - Blackwell/GB200 optimized environment
    npu      - Ascend NPU environment
    rocm     - AMD ROCm environment
    router   - Lightweight router service
    xeon     - Intel Xeon CPU-only environment

EOF
}

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

validate_environment() {
    local env=$1
    if [[ ! " ${AVAILABLE_ENVS} " =~ " ${env} " ]]; then
        error "Invalid environment: ${env}. Available: ${AVAILABLE_ENVS}"
    fi
}

build_base_image() {
    local cuda_version=$1
    local branch_type=$2
    local tag="${TAG_PREFIX}/base:${cuda_version}"
    
    log "Building base image: ${tag}"
    
    docker build \
        --build-arg CUDA_VERSION="${cuda_version}" \
        --build-arg BRANCH_TYPE="${branch_type}" \
        -t "${tag}" \
        -f Dockerfile.base \
        ..
    
    log "Base image built successfully: ${tag}"
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
        "npu")
            local tag="${TAG_PREFIX}:${env}"
            ;;
        "router")
            local tag="${TAG_PREFIX}:${env}"
            ;;
        "xeon")
            local tag="${TAG_PREFIX}:${env}"
            ;;
        *)
            local tag="${TAG_PREFIX}:${env}-${cuda_version}"
            ;;
    esac
    
    log "Building ${env} environment: ${tag}"
    
    # Build arguments vary by environment
    case "${env}" in
        "default")
            docker build \
                --build-arg CUDA_VERSION="${cuda_version}" \
                --build-arg BUILD_TYPE="${build_type}" \
                --build-arg BRANCH_TYPE="${branch_type}" \
                -t "${tag}" \
                -f "${dockerfile}" \
                ..
            ;;
        "gb200")
            docker build \
                --build-arg CUDA_VERSION="${cuda_version}" \
                --build-arg BUILD_TYPE="${build_type}" \
                --build-arg BRANCH_TYPE="${branch_type}" \
                -t "${tag}" \
                -f "${dockerfile}" \
                ..
            ;;
        "npu")
            docker build \
                -t "${tag}" \
                -f "${dockerfile}" \
                ..
            ;;
        "rocm")
            # ROCm needs special handling for GPU architecture
            local gpu_arch=${GPU_ARCH:-"gfx950"}
            docker build \
                --build-arg GPU_ARCH="${gpu_arch}" \
                --build-arg SGL_BRANCH="${branch_type}" \
                -t "${tag}" \
                -f "${dockerfile}" \
                ..
            ;;
        "router")
            docker build \
                --build-arg SGLANG_REPO_REF="${branch_type}" \
                -t "${tag}" \
                --target router-image \
                -f "${dockerfile}" \
                ..
            ;;
        "xeon")
            docker build \
                -t "${tag}" \
                -f "${dockerfile}" \
                ..
            ;;
        *)
            error "Unknown environment: ${env}"
            ;;
    esac
    
    log "Environment image built successfully: ${tag}"
    
    if [[ "${PUSH}" == "true" ]]; then
        log "Pushing image: ${tag}"
        docker push "${tag}"
        log "Image pushed successfully: ${tag}"
    fi
}

main() {
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
    
    log "Starting SGLang Docker build"
    log "Environment: ${ENVIRONMENT}"
    log "CUDA Version: ${CUDA_VERSION}"
    log "Build Type: ${BUILD_TYPE}"
    log "Branch Type: ${BRANCH_TYPE}"
    log "Tag Prefix: ${TAG_PREFIX}"
    
    # Change to docker directory
    cd "$(dirname "${BASH_SOURCE[0]}")"
    
    # For CUDA-based environments, build base image first
    if [[ "${ENVIRONMENT}" != "npu" && "${ENVIRONMENT}" != "router" && "${ENVIRONMENT}" != "xeon" ]]; then
        build_base_image "${CUDA_VERSION}" "${BRANCH_TYPE}"
    fi
    
    # Build environment-specific image
    build_environment_image "${ENVIRONMENT}" "${CUDA_VERSION}" "${BUILD_TYPE}" "${BRANCH_TYPE}"
    
    log "Build completed successfully!"
    log "Built image: ${TAG_PREFIX}:${ENVIRONMENT}"
    
    if [[ "${ENVIRONMENT}" == "default" ]]; then
        log "Default environment built. You can run it with:"
        log "  docker run -it --gpus all ${TAG_PREFIX}:${CUDA_VERSION}"
    elif [[ "${ENVIRONMENT}" == "router" ]]; then
        log "Router service built. You can run it with:"
        log "  docker run -p 8080:8080 ${TAG_PREFIX}:router"
    else
        log "Environment built. You can run it with:"
        log "  docker run -it --gpus all ${TAG_PREFIX}:${ENVIRONMENT}"
    fi
}

main "$@"
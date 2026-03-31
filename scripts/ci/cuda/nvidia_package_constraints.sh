#!/bin/bash
# Generate a pip constraints file to pin nvidia packages that we override
# after installation. Without these constraints, pip downloads the versions
# pinned by torch (e.g. cudnn 9.10, ~700 MB) only for us to replace them
# immediately afterwards — wasting bandwidth and CI time.
#
# Usage:
#   source scripts/ci/cuda/nvidia_package_constraints.sh
#   pip install -e "python[dev]" -c "$CONSTRAINTS_FILE" ...
#   rm -f "$CONSTRAINTS_FILE"

CONSTRAINTS_FILE=$(mktemp)

# Versions must match what ci_install_dependency.sh / Dockerfile install later.
# cudnn 9.16: Conv3D performance regression with older versions
# nvshmem 3.4.5: required by DeepEP
# nccl 2.28.3: CUDA 12/13 compatibility patch
CUDA_MAJOR="${CUDA_VERSION%%.*}"
if [ -z "$CUDA_MAJOR" ]; then
    CUDA_MAJOR="12"
fi

if [ "$CUDA_MAJOR" = "12" ]; then
    cat > "$CONSTRAINTS_FILE" <<EOF
nvidia-cudnn-cu12==9.16.0.29
nvidia-nccl-cu12==2.28.3
nvidia-nvshmem-cu12==3.4.5
EOF
elif [ "$CUDA_MAJOR" = "13" ]; then
    cat > "$CONSTRAINTS_FILE" <<EOF
nvidia-cudnn-cu13==9.16.0.29
nvidia-nccl-cu13==2.28.3
EOF
else
    touch "$CONSTRAINTS_FILE"
fi

export CONSTRAINTS_FILE

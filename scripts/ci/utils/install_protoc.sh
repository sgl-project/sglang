#!/bin/bash
# Ensure protoc is installed for router build (gRPC protobuf compilation).
set -euxo pipefail

if command -v protoc >/dev/null 2>&1 && protoc --version >/dev/null 2>&1; then
    echo "protoc already installed: $(protoc --version)"
    exit 0
fi

if command -v protoc >/dev/null 2>&1; then
    echo "protoc found but not runnable, reinstalling..."
else
    echo "protoc not found, installing..."
fi

ARCH=$(uname -m)

if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian
    apt-get update || true # May fail due to unrelated broken packages
    PROTOC_APT_PACKAGES=(wget unzip gcc g++ perl make)
    apt-get install -y --no-install-recommends "${PROTOC_APT_PACKAGES[@]}" || {
        echo "Warning: apt-get install failed, checking if required packages are available..."
        for pkg in "${PROTOC_APT_PACKAGES[@]}"; do
            if ! dpkg -l "$pkg" 2>/dev/null | grep -q "^ii"; then
                echo "ERROR: Required package $pkg is not installed and apt-get failed"
                exit 1
            fi
        done
        echo "All required packages are already installed, continuing..."
    }
elif command -v yum &> /dev/null; then
    # RHEL/CentOS
    yum update -y
    yum install -y wget unzip gcc gcc-c++ perl-core make
else
    echo "ERROR: Neither apt-get nor yum found; cannot install protoc build deps"
    exit 1
fi

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    PROTOC_ARCH="aarch_64"
else
    PROTOC_ARCH="x86_64"
fi
PROTOC_ZIP="protoc-32.0-linux-${PROTOC_ARCH}.zip"
(
    cd /tmp
    wget "https://github.com/protocolbuffers/protobuf/releases/download/v32.0/${PROTOC_ZIP}"
    unzip -o "${PROTOC_ZIP}" -d /usr/local
    rm -f "${PROTOC_ZIP}"
)
protoc --version

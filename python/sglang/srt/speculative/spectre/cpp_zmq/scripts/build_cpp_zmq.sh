#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ZMQ_HPP_PATH="/usr/include/zmq.hpp"
pushd "$PROJECT_DIR"

pip install pybind11 msgpack --break-system-packages
apt-get update && apt-get install -y libmsgpack-dev libzmq3-dev

if [ ! -f "$ZMQ_HPP_PATH" ]; then
  echo "Missing ${ZMQ_HPP_PATH}. Download zmq.hpp from:" >&2
  echo "  https://github.com/zeromq/cppzmq/blob/master/zmq.hpp" >&2
  echo "and place it at ${ZMQ_HPP_PATH} before building spectre_zmq." >&2
  exit 1
fi

python3 setup.py build_ext --inplace
popd

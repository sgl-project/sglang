#!/bin/sh
set -eu

runfiles_dir="${RUNFILES_DIR:-${0}.runfiles}"
for protobuf_dir in \
    "${runfiles_dir}/protobuf+" \
    "${runfiles_dir}/_main/external/protobuf+"
do
    if [ -x "${protobuf_dir}/protoc" ] && [ -d "${protobuf_dir}/src" ]; then
        exec "${protobuf_dir}/protoc" "--proto_path=${protobuf_dir}/src" "$@"
    fi
done

echo "gateway_protoc: protobuf runfiles not found under ${runfiles_dir}" >&2
exit 1

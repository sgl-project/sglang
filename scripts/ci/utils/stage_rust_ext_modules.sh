#!/bin/bash
# Copy the built PyO3 extension modules into rust-ext-staging/<pkg>/ for
# upload-artifact. Shared by both jobs of _pr-test-rust-ext-build.yml, so the
# archive layout and the module-count check cannot drift between them.
#
# MAX_GLIBC (optional): also reject a module requiring a newer GLIBC symbol
# version than the test runners have. Only set where the modules were just
# compiled - on a cache hit these are the same bytes that already passed.
set -euo pipefail
shopt -s nullglob
# upload-artifact strips the longest common prefix it matched, so a missing
# module would silently shift the archive layout.
rm -rf rust-ext-staging
built=()
# Same suffix set across pkgs, or one ABI's Rust-server tests silently skip.
expected_suffixes=""
for pkg in server grpc multimodal; do
    found=(python/sglang/srt/"${pkg}"/_core*.so)
    if [ ${#found[@]} -eq 0 ]; then
        echo "::error::no extension module found for ${pkg}"
        exit 1
    fi
    suffixes=$(printf '%s\n' "${found[@]##*/_core}" | sort)
    if [ -z "${expected_suffixes}" ]; then
        expected_suffixes="${suffixes}"
    elif [ "${suffixes}" != "${expected_suffixes}" ]; then
        echo "::error::extension modules for ${pkg} do not match server's interpreter set"
        printf 'have:\n%s\nwant:\n%s\n' "${suffixes}" "${expected_suffixes}"
        exit 1
    fi
    mkdir -p "rust-ext-staging/${pkg}"
    cp "${found[@]}" "rust-ext-staging/${pkg}/"
    built+=("${found[@]}")
done
max_allowed="${MAX_GLIBC:-}"
[ -n "${max_allowed}" ] || exit 0

# Newer glibc than the test runners fails at import: "GLIBC_2.xx not found".
status=0
for so in "${built[@]}"; do
    # grep exits 1 with no match; without `|| true` pipefail kills the script.
    needed=$(objdump -T "$so" \
        | grep -oE 'GLIBC_2\.[0-9]+' \
        | sed 's/GLIBC_//' \
        | sort -V | tail -1 || true)
    echo "${so}: requires glibc <= ${needed:-none}"
    if [ -n "${needed}" ] \
       && [ "$(printf '%s\n%s\n' "${max_allowed}" "${needed}" | sort -V | tail -1)" != "${max_allowed}" ]; then
        echo "::error::${so} requires glibc ${needed} > ${max_allowed} supported by the test runners; build on an older image"
        status=1
    fi
done
exit $status

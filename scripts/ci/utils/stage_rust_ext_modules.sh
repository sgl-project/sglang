#!/bin/bash
# Copy the built PyO3 extension modules into their package-relative paths under
# rust-ext-staging/. Shared by both jobs of _pr-test-rust-ext-build.yml, so the
# archive layout and module-count checks cannot drift between them.
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
# Same suffix set across modules, or one ABI's Rust-server tests silently skip.
expected_suffixes=""
mkdir -p rust-ext-staging/rust_extensions
for module in server grpc multimodal; do
    found=(python/sglang/srt/rust_extensions/_"${module}"*.so)
    if [ ${#found[@]} -eq 0 ]; then
        echo "::error::no extension module found for ${module}"
        exit 1
    fi
    suffixes=$(printf '%s\n' "${found[@]##*/_${module}}" | sort)
    if [ -z "${expected_suffixes}" ]; then
        expected_suffixes="${suffixes}"
    elif [ "${suffixes}" != "${expected_suffixes}" ]; then
        echo "::error::extension modules for ${module} do not match server's interpreter set"
        printf 'have:\n%s\nwant:\n%s\n' "${suffixes}" "${expected_suffixes}"
        exit 1
    fi
    cp "${found[@]}" rust-ext-staging/rust_extensions/
    built+=("${found[@]}")
done

mkdir -p rust-ext-staging/mem_cache/rust_tree_core
for module in mem_cache mem_cache_inspection; do
    tree_core=(python/sglang/srt/mem_cache/rust_tree_core/"${module}".*.so)
    if [ ${#tree_core[@]} -eq 0 ]; then
        echo "::error::no Rust TreeCore ${module} extension module found"
        exit 1
    fi
    tree_core_suffixes=$(printf '%s\n' "${tree_core[@]##*/${module}}" | sort)
    if [ "${tree_core_suffixes}" != "${expected_suffixes}" ]; then
        echo "::error::Rust TreeCore ${module} extension does not match the interpreter set"
        printf 'have:\n%s\nwant:\n%s\n' "${tree_core_suffixes}" "${expected_suffixes}"
        exit 1
    fi
    cp "${tree_core[@]}" rust-ext-staging/mem_cache/rust_tree_core/
    built+=("${tree_core[@]}")
done
max_allowed="${MAX_GLIBC:-}"
[ -n "${max_allowed}" ] || exit 0

# Newer glibc than the test runners fails at import: "GLIBC_2.xx not found".
status=0
for so in "${built[@]}"; do
    # Its own invocation, not the head of a pipeline: there its failure is
    # invisible, and the empty symbol list a wrong-arch objdump leaves behind reads
    # exactly like "needs no glibc", passing this gate silently.
    if ! symbols=$(objdump -T "$so" 2>&1); then
        echo "::error::objdump could not read ${so}: ${symbols}"
        echo "::error::this gate must run on the same architecture as the modules"
        status=1
        continue
    fi
    # grep exits 1 with no match; without `|| true` pipefail kills the script. With
    # objdump known to have succeeded, no match really is no requirement.
    references=$(printf '%s\n' "${symbols}" | grep -coE 'GLIBC_2\.[0-9]+' || true)
    needed=$(printf '%s\n' "${symbols}" \
        | grep -oE 'GLIBC_2\.[0-9]+' \
        | sed 's/GLIBC_//' \
        | sort -V | tail -1 || true)
    echo "${so}: requires glibc <= ${needed:-none} (${references} GLIBC references)"
    if [ -n "${needed}" ] \
       && [ "$(printf '%s\n%s\n' "${max_allowed}" "${needed}" | sort -V | tail -1)" != "${max_allowed}" ]; then
        echo "::error::${so} requires glibc ${needed} > ${max_allowed} supported by the test runners; build on an older image"
        status=1
    fi
done
exit $status

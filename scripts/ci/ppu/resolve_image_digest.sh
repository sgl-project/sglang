#!/usr/bin/env bash
# Resolve the base image's manifest digest, and pin the run to it.
#
# A trend row records base_image_digest so that "throughput fell 8% on the 20th"
# can be told apart from "throughput fell 8% on the 20th, the night the image
# moved". On the K8s lines that field was null: the Answer bare-metal line fills it
# with `docker image inspect`, and this orchestration shell has no Docker daemon --
# nor python3, nor jq, which is why what follows is curl and sed. The registry will
# answer the question a daemon would have: a HEAD of the manifest returns the same
# digest a pull would resolve the tag to.
#
# Resolving is not enough on its own. The action submits pods with
# imagePullPolicy: IfNotPresent, so a node still holding an older image under this
# tag runs that one, and a digest merely read from the registry would name an image
# the measurement never used -- a field that looks authoritative and is not, which
# is worse for attribution than the null it replaced. So the digest is also handed
# back as a pinned reference for the pods to run: under IfNotPresent a digest
# reference matches only that image, so the node either already has it or pulls it,
# and both nodes of a multi-node group are then guaranteed the same one.
#
# Never fails the run. An unresolvable digest leaves the reference as the tag and
# the digest empty, which is exactly the behaviour of every run before this script
# existed; a nightly must not go red over provenance it cannot look up.
#
# Reads:
#   PPU_BASE_IMAGE      the image, in tag form
#   GITHUB_ENV          appended to, for the steps that follow
#
# Optional, so that one script serves both images a nightly run touches -- the
# platform base the pods run on, and the build product they are meant to test:
#   PPU_RESOLVE_INPUT   image to resolve instead of PPU_BASE_IMAGE
#   PPU_RESOLVE_PREFIX  variable name prefix to write, default PPU_BASE_IMAGE
#                       (the artifact image passes PPU_TEST_IMAGE, so the two
#                       digests land in separate fields and a trend row can tell
#                       "the platform moved" from "the product moved")
#   PPU_RESOLVE_STRICT  "1" turns every give-up below into a failure. The base
#                       image keeps the default: provenance it cannot look up is
#                       no reason to redden a nightly. The artifact image passes
#                       it, because there a missing digest means the pods are
#                       about to run something nobody named.
#
# Writes, as environment for later steps:
#   <PREFIX>_DIGEST  sha256:... , or empty when the lookup failed
#   <PREFIX>_REF     what the pods should run: image@digest, or the tag
#
# Usage: bash scripts/ci/ppu/resolve_image_digest.sh
set -uo pipefail

image="${PPU_RESOLVE_INPUT:-${PPU_BASE_IMAGE:-}}"
prefix="${PPU_RESOLVE_PREFIX:-PPU_BASE_IMAGE}"
strict="${PPU_RESOLVE_STRICT:-0}"
if [ -z "${image}" ]; then
  echo "::error::PPU_BASE_IMAGE (or PPU_RESOLVE_INPUT) is not set" >&2
  exit 1
fi

# Give up rather than fail, in one place, so that every unhappy path below leaves
# the run exactly as it was before this step existed -- unless the caller asked
# for strict, in which case giving up is itself the failure.
give_up() {
  if [ "${strict}" = "1" ]; then
    echo "::error::could not resolve the digest of ${image} (${1})" >&2
    exit 1
  fi
  echo "::warning::could not resolve the digest of ${image} (${1}); recording none"
  {
    echo "${prefix}_DIGEST="
    echo "${prefix}_REF=${image}"
  } >> "${GITHUB_ENV}"
  exit 0
}

# Split host from repository path and tag. The tag is what follows the last colon,
# but only when that piece is not a port number preceding a path.
registry="${image%%/*}"
remainder="${image#*/}"
case "${remainder##*:}" in
  */*) repository="${remainder}" ; tag="latest" ;;
  "${remainder}") repository="${remainder}" ; tag="latest" ;;
  *) repository="${remainder%:*}" ; tag="${remainder##*:}" ;;
esac

manifest_url="https://${registry}/v2/${repository}/manifests/${tag}"
# Every manifest media type a pull would accept. Asking for only one gets a 404
# from a registry holding the other, which would read as a missing image.
accept="application/vnd.docker.distribution.manifest.list.v2+json"
accept="${accept}, application/vnd.oci.image.index.v1+json"
accept="${accept}, application/vnd.docker.distribution.manifest.v2+json"
accept="${accept}, application/vnd.oci.image.manifest.v1+json"

echo "resolving ${repository}:${tag} against ${registry}"
# -I rather than -X HEAD: the latter sends a HEAD but still waits for a body, so
# curl ends every one of these requests with exit 18 and a successful lookup would
# be discarded as a failure.
challenge="$(curl -sS -m 30 -I -H "Accept: ${accept}" \
  "${manifest_url}" 2>/dev/null | tr -d '\r')" || give_up "registry unreachable"

read_digest() {
  printf '%s\n' "${1}" | sed -n 's/^[Dd]ocker-[Cc]ontent-[Dd]igest: *//p' | head -1
}

digest="$(read_digest "${challenge}")"
if [ -z "${digest}" ]; then
  # A token realm, which is how this registry answers an unauthenticated HEAD. The
  # realm, service and scope are taken from the challenge rather than assembled
  # here: Artifactory's scope names the image without the repository key in front
  # of it, and guessing that wrong is a 401 that looks like a missing image.
  authenticate="$(printf '%s\n' "${challenge}" |
    sed -n 's/^[Ww]ww-[Aa]uthenticate: *//p' | head -1)"
  field() { printf '%s\n' "${authenticate}" | sed -n "s/.*${1}=\"\([^\"]*\)\".*/\1/p"; }
  realm="$(field realm)"
  [ -n "${realm}" ] || give_up "no token realm offered"
  service="$(field service)"
  scope="$(field scope)"

  # -E, because the alternation below is not portable as a basic-regex \| : BSD sed
  # matches nothing at all and the token reads as never issued.
  token="$(curl -sS -m 30 --get \
    ${service:+--data-urlencode "service=${service}"} \
    ${scope:+--data-urlencode "scope=${scope}"} \
    "${realm}" 2>/dev/null |
    sed -n -E 's/.*"(access_token|token)":"([^"]*)".*/\2/p' | head -1)"
  [ -n "${token}" ] || give_up "the registry issued no token"

  digest="$(read_digest "$(curl -sS -m 30 -I \
    -H "Authorization: Bearer ${token}" -H "Accept: ${accept}" \
    "${manifest_url}" 2>/dev/null | tr -d '\r')")"
fi

case "${digest}" in
  sha256:*) ;;
  *) give_up "the registry returned no digest" ;;
esac

echo "${image} is ${digest}"
{
  echo "${prefix}_DIGEST=${digest}"
  echo "${prefix}_REF=${registry}/${repository}@${digest}"
} >> "${GITHUB_ENV}"

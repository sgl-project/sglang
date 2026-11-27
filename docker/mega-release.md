# NVIDIA megakernel versions in Docker images

Nightly/development image builds resolve the current Mega `main` commit at build
time. Official release builds read the full Mega SHA in
`docker/mega-release-ref.txt` from the corresponding SGLang release tag. Both
architectures receive the same resolved SGLang and Mega revisions.

## Preparing a release

1. Choose a Mega commit and put its full 40-character SHA in
   `docker/mega-release-ref.txt` on the SGLang release branch.
2. Build and validate that SGLang/Mega combination before tagging the release.
   For a local build, pass
   `--build-arg SGL_NVIDIA_MEGA_REF="$(cat docker/mega-release-ref.txt)"` together
   with the usual Docker build arguments.
3. Commit the pin before creating the SGLang release tag. The initial pin retains
   the revision previously hardcoded in the Dockerfiles; it does not replace
   validation for a new SGLang release.

The release workflow checks out `refs/tags/v<version>`, including on manual runs,
and reads the pin from that checkout. Rebuilding the same tag therefore selects
the same Mega commit even after Mega `main` or the pin on SGLang `main` changes.
A missing or malformed pin fails the workflow; it never falls back to `main`.
Older tags without the pin file cannot be built through this release-pin path.

Direct Docker builds select Mega in this order: an explicitly supplied
`SGL_NVIDIA_MEGA_REF`, the legacy `SGL_NVIDIA_MEGA_COMMIT` argument if nonempty,
then `main`. Existing scripts that pass `SGL_NVIDIA_MEGA_COMMIT` keep their pin;
new scripts should use `SGL_NVIDIA_MEGA_REF`. The selected Mega revision is logged
and saved in `/sgl-workspace/sgl-nvidia-mega.commit` inside the image. This fixes
Mega source selection; it does not make all other image dependencies reproducible.

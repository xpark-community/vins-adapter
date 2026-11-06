#!/usr/bin/env bash
# Build the CUDA-enabled OpenCV install that the GPU adapter (vins_adapter_gpu)
# links against, and package it as the release asset consumers install on their
# GPU workers.
#
# Why this exists
# ---------------
# `VINS-Fusion-gpu` links `${OpenCV_LIBS}` -- i.e. *every* module of the OpenCV it
# is configured against -- so the GPU binary hard-requires that exact module set
# at run time. The OpenCV is therefore a first-class, versioned artifact of this
# project (published as a release asset and pinned by URL + sha256 in
# `.github/workflows/release.yml`), not something each maintainer improvises.
#
# Why this image
# --------------
# The build runs inside `nvidia/cuda:<ver>-devel` (digest-pinned below), NOT the
# host's toolkit: one install must carry CUDA kernels for everyone's GPU, and
# `RTX PRO 5000` / Blackwell (`sm_120`) needs CUDA >= 12.8. The toolkit only has
# to *compile*; the worker supplies its own CUDA runtime, so a 12.9-built OpenCV
# still runs on 12.x drivers (minor version compatibility). No GPU is required on
# the build host -- nvcc emits fatbins for every arch in CUDA_ARCH_BIN.
#
# Usage
#   ./tools/build_opencv_cuda.sh                    # default: all archs, ./dist
#   ./tools/build_opencv_cuda.sh --arch-bin 7.5     # one arch (fast smoke build)
#   ./tools/build_opencv_cuda.sh --keep-build       # reuse the cmake build tree
#
#   Options
#     --out DIR          where the tarball/md5/sha256 land (default: <repo>/dist)
#     --cache DIR        work dir for sources + build tree (default: <repo>/.opencv-cuda-cache)
#     --opencv-version V OpenCV version to build (default: 4.13.0)
#     --arch-bin LIST    CUDA_ARCH_BIN, semicolon separated (default: see below)
#     --modules LIST     BUILD_LIST (default: exactly what the adapter needs)
#     --cuda-image REF   override the pinned devel image (testing only)
#     --jobs N           parallel build jobs inside the container (default: nproc)
#     --keep-build       reuse the previous build tree instead of starting clean
#     -h, --help         this text
#
# Output (in --out):
#   opencv-cuda-<ver>-linux-<pkgarch>.tar.gz   install prefix, unpack anywhere
#   ...tar.gz.md5  ...tar.gz.sha256            md5sum/sha256sum -c compatible
#
# Then publish it and pin it:
#   gh release upload <tag> dist/opencv-cuda-*.tar.gz* --clobber
#   # copy the printed sha256 into .github/workflows/release.yml
#   #   env.DEFAULT_OPENCV_CUDA_TARBALL / env.DEFAULT_OPENCV_CUDA_SHA256
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# Pinned build environment. The digest keeps this reproducible even if the tag is
# re-pointed; bump both together and record why in the commit message.
CUDA_IMAGE="nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8"
OPENCV_VERSION="4.13.0"
# sm_75 T4 | sm_80 A100 | sm_86 A10/A40 | sm_89 L20/L4/4090 | sm_90 H100 |
# sm_120 RTX PRO 5000 (Blackwell; needs CUDA >= 12.8).
ARCH_BIN="7.5;8.0;8.6;8.9;9.0;12.0"
# Exactly the module set the adapter uses; adding modules makes the GPU binary
# require them at run time (${OpenCV_LIBS}), so keep this list deliberate.
MODULES="core,imgproc,imgcodecs,calib3d,features2d,flann,highgui,video,photo,gapi,videostab,optflow,ximgproc,cudaarithm,cudafeatures2d,cudafilters,cudaimgproc,cudaoptflow,cudawarping,cudev"

OUT="$REPO/dist"
CACHE="$REPO/.opencv-cuda-cache"
JOBS=""
KEEP_BUILD=0

log() { printf '\033[1m[opencv-cuda]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[opencv-cuda] error:\033[0m %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)            OUT="$2"; shift 2 ;;
    --cache)          CACHE="$2"; shift 2 ;;
    --opencv-version) OPENCV_VERSION="$2"; shift 2 ;;
    --arch-bin)       ARCH_BIN="$2"; shift 2 ;;
    --modules)        MODULES="$2"; shift 2 ;;
    --cuda-image)     CUDA_IMAGE="$2"; shift 2 ;;
    --jobs)           JOBS="$2"; shift 2 ;;
    --keep-build)     KEEP_BUILD=1; shift ;;
    -h|--help)        awk 'NR==1{next} /^#/{sub(/^#+ ?/,""); print; next} {exit}' "$0"; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

command -v docker >/dev/null || die "docker is required"
command -v curl >/dev/null || die "curl is required"

case "$(uname -m)" in
  x86_64|amd64) PKG_ARCH=x86_64 ;;
  *) die "the CUDA devel image is amd64; build on an x86_64 host (or use an amd64 VM)" ;;
esac

mkdir -p "$CACHE" "$OUT"
mkdir -p "$OUT"

log "image:  $CUDA_IMAGE"
log "arch:   $ARCH_BIN (artifact arch $PKG_ARCH)"
log "modules:$MODULES"

# 1. Sources (cached; checksums recorded into BUILDINFO for provenance).
for pkg in opencv opencv_contrib; do
  tb="$CACHE/$pkg-$OPENCV_VERSION.tar.gz"
  if [[ ! -f "$tb" ]]; then
    log "fetching $pkg $OPENCV_VERSION"
    curl -fsSL -o "$tb" "https://github.com/opencv/$pkg/archive/refs/tags/$OPENCV_VERSION.tar.gz" \
      || die "cannot download $pkg $OPENCV_VERSION"
  fi
  [[ -d "$CACHE/$pkg-$OPENCV_VERSION" ]] || tar xzf "$tb" -C "$CACHE"
done
SRC_SHA_OPENCV="$(openssl dgst -sha256 "$CACHE/opencv-$OPENCV_VERSION.tar.gz" 2>/dev/null | awk '{print $NF}' || true)"
SRC_SHA_CONTRIB="$(openssl dgst -sha256 "$CACHE/opencv_contrib-$OPENCV_VERSION.tar.gz" 2>/dev/null | awk '{print $NF}' || true)"

# 2. Build inside the pinned image. No --gpus: nvcc compiles for the listed archs
#    whether or not the host has a device, which is what keeps this runnable on a
#    plain CI runner.
docker image inspect "$CUDA_IMAGE" >/dev/null 2>&1 || { log "pulling image"; docker pull "$CUDA_IMAGE"; }

if [[ "$KEEP_BUILD" -eq 0 ]]; then
  log "clean build tree"
  docker run --rm -v "$CACHE":/work ubuntu:22.04 rm -rf /work/build /work/install >/dev/null 2>&1 || true
fi

log "building OpenCV $OPENCV_VERSION (this takes a while)"
docker run --rm \
  -v "$CACHE":/work -w /work \
  -e BUILT_VERSION="$OPENCV_VERSION" \
  -e BUILT_ARCH_BIN="$ARCH_BIN" \
  -e BUILT_MODULES="$MODULES" \
  -e BUILT_JOBS="$JOBS" \
  "$CUDA_IMAGE" bash -c '
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq >/dev/null
    apt-get install -y -qq --no-install-recommends build-essential cmake pkg-config libssl-dev >/dev/null
    nvcc --version | tail -1
    cmake -S "opencv-$BUILT_VERSION" -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/work/install \
      -DWITH_CUDA=ON -DWITH_CUDNN=OFF \
      -DCUDA_ARCH_BIN="$BUILT_ARCH_BIN" -DCUDA_ARCH_PTX= \
      -DBUILD_LIST="$BUILT_MODULES" \
      -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF \
      -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_QT=OFF -DWITH_GTK=OFF -DWITH_VTK=OFF \
      -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=ON -DWITH_EIGEN=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_EXTRA_MODULES_PATH="/work/opencv_contrib-$BUILT_VERSION/modules"
    cmake --build build -j"${BUILT_JOBS:-$(nproc)}"
    cmake --install build
  '

# The container runs as root; hand the tree back to the caller before packaging.
docker run --rm -v "$CACHE":/work ubuntu:22.04 \
  chown -R "$(id -u):$(id -g)" /work/install /work/build >/dev/null 2>&1 || true

PREFIX_NAME="opencv-cuda-$OPENCV_VERSION"
PREFIX="$OUT/$PREFIX_NAME"
rm -rf "$PREFIX"
cp -a "$CACHE/install" "$PREFIX"

IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$CUDA_IMAGE")"
CUDA_VERSION="$(docker run --rm "$CUDA_IMAGE" nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1)"
INSTALLED_MODULES="$(ls "$PREFIX/lib" | grep -oE '^libopencv_[a-z0-9]+\.so\.[0-9]+$' \
  | sed -E 's/^libopencv_(.*)\.so\.[0-9]+$/\1/' | sort -u | tr '\n' ' ')"

# Provenance travels with the artifact: whoever installs it can see exactly which
# image and flags produced it.
{
  echo "OpenCV $OPENCV_VERSION CUDA install prefix for the VINS GPU adapter"
  echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(uname -s)/$(uname -m) $(hostname)"
  echo "opencv_cuda_image=$CUDA_IMAGE"
  echo "opencv_cuda_image_id=$IMAGE_ID"
  echo "cuda_version=$CUDA_VERSION"
  echo "arch_bin=$ARCH_BIN"
  echo "arch_ptx=(none)"
  echo "modules=$INSTALLED_MODULES"
  echo "cmake_args=BUILD_LIST=$MODULES; WITH_CUDA=ON; CUDA_ARCH_BIN=$ARCH_BIN; CUDA_ARCH_PTX=(none); WITH_FFMPEG/GSTREAMER/QT/GTK/VTK/OPENCL/IPP/TBB/EIGEN=OFF; WITH_OPENMP=ON"
  echo "source=opencv-$OPENCV_VERSION.tar.gz sha256=${SRC_SHA_OPENCV:-unknown}; opencv_contrib-$OPENCV_VERSION.tar.gz sha256=${SRC_SHA_CONTRIB:-unknown}"
} > "$PREFIX/BUILDINFO.txt"

TARBALL="$OUT/$PREFIX_NAME-linux-$PKG_ARCH.tar.gz"
rm -f "$TARBALL" "$TARBALL.md5" "$TARBALL.sha256"
tar -czf "$TARBALL" -C "$OUT" "$PREFIX_NAME"
if command -v sha256sum >/dev/null; then
  sha256sum "$TARBALL" > "$TARBALL.sha256"
  md5sum "$TARBALL" > "$TARBALL.md5"
else
  shasum -a 256 "$TARBALL" | awk '{print $1"  '"$(basename "$TARBALL")"'"}' > "$TARBALL.sha256"
  md5 -q "$TARBALL" | awk '{print $1"  '"$(basename "$TARBALL")"'"}' > "$TARBALL.md5"
fi
SHA="$(awk '{print $1}' "$TARBALL.sha256")"

log "bundle:  $TARBALL ($(du -h "$TARBALL" | cut -f1))"
log "sha256:  $SHA"
log "md5:     $(awk '{print $1}' "$TARBALL.md5")"
log ""
log "publish and pin:"
log "  gh release upload <tag> $(basename "$TARBALL") $(basename "$TARBALL").md5 $(basename "$TARBALL").sha256 --clobber"
log "  # .github/workflows/release.yml"
log "  DEFAULT_OPENCV_CUDA_TARBALL: <release asset URL>"
log "  DEFAULT_OPENCV_CUDA_SHA256: $SHA"

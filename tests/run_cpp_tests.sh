#!/usr/bin/env bash
#
# Build and run the adapter-spec unit tests (tests/test_adapter_spec.cpp).
#
#   ./tests/run_cpp_tests.sh
#
# The suite pins the vins_adapter in/out contract (config resolution, csv
# parsing, camodocal output, TUM line format, emission gate); it only needs
# C++17 + Eigen + yaml-cpp. Preference order:
#   1. local g++/clang++ with Eigen + yaml-cpp (paths probed for homebrew/apt)
#   2. the pinned vins-adapter build image, where both are guaranteed
# The Dockerfile also builds and runs the same suite on every adapter build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO/src/adapter_spec.h"
TEST="$REPO/tests/test_adapter_spec.cpp"
OUT="${TMPDIR:-/tmp}/vins_adapter_test_spec_$$"

for f in "$SRC" "$TEST"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 2; }
done
trap 'rm -rf "$OUT"' EXIT
mkdir -p "$OUT"

# --- 1. local toolchain -----------------------------------------------------
# Apple clang does not search homebrew paths by default, so probe them.
eigen_inc=""
for d in /opt/homebrew/include/eigen3 /usr/local/include/eigen3 \
         /usr/include/eigen3 /usr/include; do
  [[ -f "$d/Eigen/Geometry" ]] && { eigen_inc="$d"; break; }
done
extra_inc=""
for d in /opt/homebrew/include /usr/local/include; do
  [[ -f "$d/yaml-cpp/yaml.h" ]] && { extra_inc="$d"; break; }
done

if [[ -n "$eigen_inc" && -n "$extra_inc" ]] && command -v g++ >/dev/null 2>&1; then
  echo "== building adapter-spec tests locally (Eigen: $eigen_inc) =="
  lib_dirs=(-L/opt/homebrew/lib -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu)
  if g++ -std=c++17 -O1 -Wall -Wextra -I"$eigen_inc" -I"$extra_inc" \
        -I"$REPO/src" "${lib_dirs[@]}" \
        "$TEST" -o "$OUT/test_adapter_spec" -lyaml-cpp \
        2>"$OUT/build.log"; then
    "$OUT/test_adapter_spec"
    exit $?
  fi
  echo "local build failed; falling back to docker:" >&2
  sed 's/^/  /' "$OUT/build.log" >&2 || true
fi

# --- 2. pinned build image --------------------------------------------------
IMAGE="vins-adapter:noetic-jammy"
if command -v docker >/dev/null 2>&1 \
   && docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "== building adapter-spec tests in $IMAGE =="
  exec docker run --rm -v "$REPO":/repo:ro --workdir /tmp \
    "$IMAGE" bash -c '
      g++ -std=c++17 -O1 -Wall -Wextra \
          -I/opt/ros1/include/eigen3 -I/opt/ros1/include -I/repo/src \
          /repo/tests/test_adapter_spec.cpp -o /tmp/test_adapter_spec \
          -L/opt/ros1/lib -lyaml-cpp -Wl,-rpath,/opt/ros1/lib \
      && /tmp/test_adapter_spec'
fi

echo "no Eigen/yaml-cpp toolchain found:" >&2
echo "  - brew install eigen yaml-cpp   (macOS)  -- then rerun" >&2
echo "  - or build the pinned image first: ./build_vins_adapter.sh --skip-tests" >&2
exit 2

# thirdparty/

Bundled VINS-Fusion sources. These were previously git submodules; they are
now vendored as plain source so they can be edited directly (the adapter needs
code-level changes upstream cannot take as-is).

| Directory          | Upstream                                                                   | Pinned commit                               |
| ------------------ | -------------------------------------------------------------------------- | ------------------------------------------- |
| `VINS-Fusion/`     | https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git                   | `be55a937a57436548ddfb1bd324bc1e9a9e828e0`  |
| `VINS-Fusion-gpu/` | https://github.com/pjrambo/VINS-Fusion-gpu.git                             | `9ec9283d9d2348df05d99ed05bb8ecd8e4687efc`  |

Machine-readable copy of the pinning table: `upstream.txt` (consumed by
`build_vins_adapter.sh` for `BUILDINFO.json` provenance).

## Local changes on top of the pinned snapshots

The bundled trees already carry the adapter's local fixes (these used to be
uncommitted edits inside the submodule checkouts):

- `vins_estimator/CMakeLists.txt` (both forks; the GPU fork additionally wires
  the CUDA build)
- `vins_estimator/src/estimator/estimator.h` (both forks)
- `vins_estimator/src/utility/visualization.cpp` (both forks)

Toolchain-only fixes (c++14 for gcc-11/ceres 2.1, `CV_*` -> `cv::*` for
OpenCV 4.13) are still applied with `sed` **inside the Docker image**, so the
bundled source stays pristine; see the repo `Dockerfile`.

## License

Both forks are GPLv3. They are compiled locally and never redistributed; the
adapter binary carries the GPL with it.

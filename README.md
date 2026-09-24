# VINS-Fusion Adapter

Run [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion)'s
estimator offline: feed it frame directories plus an optional IMU csv, get a TUM
trajectory back — no ROS runtime, no `roscore`, no bag playback.

```
vins_adapter <config.yaml> <output.tum>
```

It is a self-contained Linux bundle: unpack it anywhere, write a config, run it.
With the GPU add-on the bundle also carries a CUDA feature-tracking binary; the
launcher uses it by default and falls back to the CPU binary when no usable GPU
is present.

## Quick start

```bash
tar xzf vins_adapter-linux-x86_64.tar.gz

cat > config.yaml <<'YAML'
left_dir: /data/ep001/cam0
T_cam0_body:
- [1.0, 0.0, 0.0, 0.0]
- [0.0, 1.0, 0.0, 0.0]
- [0.0, 0.0, 1.0, 0.0]
YAML

./vins_adapter/run_vins_adapter.sh config.yaml trajectory.tum     # GPU if usable
./vins_adapter/run_vins_adapter.sh --cpu config.yaml trajectory.tum
```

The launcher uses the **GPU binary by default**. It falls back to the CPU binary
(with a note on stderr) when `--cpu` is passed, when the bundle is CPU-only, when
no NVIDIA device is present, or when the worker's CUDA/OpenCV libraries cannot be
resolved. Point it at those libraries with `VINS_OPENCV_CUDA_DIR` (or expose them
via `LD_LIBRARY_PATH` / `ldconfig`); they are **not** bundled. The CPU binary
needs nothing but glibc.

## Config yaml

A **flat** yaml (not the upstream VINS config):

```yaml
imu: 1                      # 1 = use imu_csv
num_of_cam: 2               # 2 = stereo, requires right_dir
estimate_extrinsic: 0       # 0 trust, 1 refine around guess, 2 refine online
max_features: 200           # per-frame feature cap (floor 50)

# pinhole intrinsics + radial/tangential distortion
fx: 458.65
fy: 457.30
cx: 367.21
cy: 248.38
k1: -0.2834
k2: 0.0739
p1: 0.0002
p2: -0.0001

left_dir:  /data/ep001/cam0       # frames: .png/.jpg/.jpeg/.bmp/.tif/.tiff
right_dir: /data/ep001/cam1       # optional, stereo only
imu_csv:   /data/ep001/imu.csv    # optional
frame_times_csv: /data/ep001/times.csv   # optional

# 3x4 [R|t] row-major (three rows), camera pose in the body frame (RIC/TIC convention)
T_cam0_body: [[1,0,0, 0.0],
              [0,1,0, 0.0],
              [0,0,1, 0.0]]
T_cam1_body: [[1,0,0,-0.12],
              [0,1,0, 0.0],
              [0,0,1, 0.0]]      # optional, stereo only
```

| key | required | notes |
| --- | --- | --- |
| `left_dir` | yes | sorted by filename; extension-filtered |
| `right_dir` | stereo | paired with left by sort order; may not be shorter than left |
| `imu` / `imu_csv` | no | both needed for visual-inertial; header row, then `timestamp,gx,gy,gz,ax,ay,az` (seconds, gyro rad/s **first**, accel m/s²) |
| `frame_times_csv` | no | header row, then `index,timestamp_ns` (per-frame clock) |
| `num_of_cam` | no | `2` + `right_dir` = stereo (default 1) |
| `estimate_extrinsic` | no | default 2; **forced to 0** on vision-only runs, since online refinement needs IMU preintegration |
| `max_features` | no | default 200, min 50 |
| `fx fy cx cy k1 k2 p1 p2` | no | defaults 500/500/320/240 and zero distortion |
| `T_cam0_body` | yes | 3 rows of 4 numbers |
| `T_cam1_body` | stereo | same shape |

## Output

One TUM line per optimized frame:

```
timestamp tx ty tz qx qy qz qw
```

Seconds, body pose in world. Timestamps come from `frame_times_csv` when given;
otherwise a relative clock of `frame_index / 30 fps`.

As in upstream VINS-Fusion, poses are emitted only once the estimator reaches
`NON_LINEAR` (sliding window initialized) — **short or near-static episodes
legitimately produce fewer rows than input frames.** That is expected, not a
failure.

### Reading VINS-Fusion's own `vio.csv` (the two conversion traps)

The adapter does these conversions for you, but they bite anyone comparing
against upstream output:

```
vio.csv:  ts_ns, px,py,pz, qw,qx,qy,qz, vx,vy,vz     (nanoseconds, qw FIRST)
TUM:      ts_s,  tx,ty,tz, qx,qy,qz,qw               (seconds,     q  LAST)
```

## Exit codes

| code | meaning |
| --- | --- |
| `0` | ok |
| `1` | runtime failure (message on stderr: bad config, unreadable frames, stereo mismatch, …) |
| `2` | usage error (wrong argv) |

## CPU vs GPU benchmark

`benchmarks/benchmark_cpu_gpu.py` runs the CPU and CUDA binaries on the **same**
episode and reports the wall-clock gain, throughput, and a trajectory
cross-check. Each variant gets an untimed warmup (CUDA context/library init plus
warm page cache) and then `--runs` timed runs; the headline number is the
median-based speedup.

Because the GPU tracker computes in float on the device while the CPU path runs
the same algorithms on the host, the two trajectories are **not** bit-identical:
the tool reports matched-timestamp translation RMSE/max and max rotation
difference, and small nonzero values are expected and healthy. The GPU side is
skipped with a reason when the bundle has no GPU binary or the worker's
CUDA/OpenCV cannot load it (use `--require-gpu` to make that fatal);
`VINS_OPENCV_CUDA_DIR` is honored exactly like the launcher does.

```bash
# synthetic stereo episode (no dataset needed)
python3 benchmarks/benchmark_cpu_gpu.py

python3 benchmarks/benchmark_cpu_gpu.py --mono              # monocular
python3 benchmarks/benchmark_cpu_gpu.py --config real.yaml  # a real episode
python3 benchmarks/benchmark_cpu_gpu.py --json bench.json   # machine-readable
```

### Measured result — EuRoC MH_01_easy

Reproducible on real data with the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/euroc-mav/)
(Burri et al., IJRR 2016 — free for research use, please cite). It maps 1:1 onto
the adapter's contract: stereo 752×480 global-shutter PNG dirs, pinhole + radtan
distortion, 200 Hz IMU csv in gyro-first order, and `T_BS` in `sensor.yaml` is
already the `T_cam{i}_body` convention.

```bash
# download (~1-2.5 GB) + extract + generate the config in one command
python3 benchmarks/prepare_euroc.py --download MH_01_easy --datasets ~/datasets/euroc

# or prepare an already-downloaded sequence
python3 benchmarks/prepare_euroc.py ~/datasets/euroc/MH_01_easy --with-gt

# benchmark it
VINS_OPENCV_CUDA_DIR=<cuda-opencv-dir> \
python3 benchmarks/benchmark_cpu_gpu.py \
  --config ~/datasets/euroc/MH_01_easy/config.yaml --runs 3 --warmup 1
```

Reference run: `MH_01_easy` (3682 stereo frames + 200 Hz IMU), stereo + IMU,
`max_features: 200`, 1 warmup + 3 timed runs, wall clock per whole process,
single-threaded. Host: Intel Xeon Platinum 8255C (16 vCPU) + NVIDIA Tesla T4
(CUDA 12.4). The sequence was staged on local disk — benchmarking straight off a
network mount lets I/O dominate and destabilizes the numbers.

| variant | median wall | throughput |
| --- | --- | --- |
| CPU `vins_adapter` | 468.73 s | 7.86 frames/s |
| GPU `vins_adapter_gpu` | 245.12 s | 15.02 frames/s |

**Speedup ≈ 1.91x, wall time cut 47.7 %.** Trajectory cross-check over
3672/3672 matched poses: translation RMSE 29.6 mm (max 81.5 mm), max rotation
0.72°.

Only the feature tracker moves to the GPU; PNG decode, IMU preintegration, and
the Ceres sliding-window optimizer stay on the CPU, so ~1.9x is the expected
ceiling for this workload rather than a GPU-utilization figure. Full settings and
per-run numbers: [`benchmarks/MH_01_easy_cpu_vs_gpu.md`](benchmarks/MH_01_easy_cpu_vs_gpu.md)
and [`benchmarks/MH_01_easy_cpu_vs_gpu.json`](benchmarks/MH_01_easy_cpu_vs_gpu.json).

The tool is standard-library-only. Run it on the worker the bundle was shipped
to — it is a Linux ELF, and GPU numbers only exist where a CUDA device and the
matching CUDA/OpenCV build are present.

## Build (maintainers)

Requires only Docker on the build host; ROS1 Noetic, Ceres, OpenCV and
VINS-Fusion are all built inside the image.

```bash
./build_vins_adapter.sh                     # CPU-only bundle
./build_vins_adapter.sh --opencv-cuda DIR   # + CUDA binary (CUDA-enabled OpenCV at DIR)
./build_vins_adapter.sh --cpu-only          # explicitly skip the GPU variant
./build_vins_adapter.sh --remote user@host  # build on a native docker daemon over ssh
./build_vins_adapter.sh --help              # all options
```

GPU is off unless `--opencv-cuda` points at a directory containing a
CUDA-enabled OpenCV (`OpenCVConfig.cmake`, or
`lib/cmake/opencv4/OpenCVConfig.cmake`) whose major.minor matches the CPU
build's OpenCV. That directory is staged into the build context and is **not**
redistributed — the GPU worker must provide the same CUDA + OpenCV at runtime.
The build target follows the compiling machine's CPU arch; override with
`--platform linux/amd64|linux/arm64`. The regression suite runs automatically
(opt out with `--skip-tests`).

Artifacts land in `vins_adapter/`:

| path | what |
| --- | --- |
| `vins_adapter` | the CPU Linux ELF binary |
| `lib/` | CPU bundled shared-library closure (via `$ORIGIN/lib`) |
| `vins_adapter_gpu` | the CUDA binary (only with `--opencv-cuda`) |
| `lib_gpu/` | GPU binary's non-CUDA, non-OpenCV closure (via `$ORIGIN/lib_gpu`) |
| `run_vins_adapter.sh` | launcher — use this (picks GPU/CPU, honors `--cpu`) |
| `BUILDINFO.json` | provenance: source commit, image id, platform, arch, gpu, build time |

plus a shippable `vins_adapter-linux-<arch>.tar.gz` (+ `.md5`) beside it
(`md5sum -c vins_adapter-linux-x86_64.tar.gz.md5` to verify a transfer).

## Tests

```bash
tests/run_cpp_tests.sh          # C++ unit tests (needs Eigen + yaml-cpp)
python3 tests/run_tests.py      # regression + functional tests
python3 tests/run_tests.py --adapter /path/to/binary
python3 tests/run_tests.py --sources-only   # no build needed
```

## License

VINS-Fusion is **GPLv3**, and the adapter links against it, so this project is
GPLv3 too (see `LICENSE`, and `thirdparty/VINS-Fusion/LICENSE` for upstream). The
binary is built locally and never redistributed; it carries the GPL with it.

All state estimation is the work of **VINS-Fusion** by Tong Qin, Shaozu Cao, Jie
Pan, Peiliang Li and Shaojie Shen (HKUST Aerial Robotics Group). Please credit
and cite upstream, not this adapter:

```bibtex
@article{qin2017vins,
  title={VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator},
  author={Qin, Tong and Li, Peiliang and Shen, Shaojie},
  journal={IEEE Transactions on Robotics},
  year={2018},
  volume={34},
  number={4},
  pages={1004-1020}
}
```

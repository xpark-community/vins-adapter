#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
"""CPU vs GPU performance benchmark for the offline VINS adapter.

Runs the CPU binary (vins_adapter) and the CUDA binary (vins_adapter_gpu)
from a built bundle on the SAME episode and reports the wall-clock gain:

    python3 benchmarks/benchmark_cpu_gpu.py                     # synthetic stereo episode
    python3 benchmarks/benchmark_cpu_gpu.py --mono              # monocular
    python3 benchmarks/benchmark_cpu_gpu.py --config real.yaml  # benchmark a real episode
    python3 benchmarks/benchmark_cpu_gpu.py --json bench.json   # also write machine-readable results

What it measures
----------------
Each variant gets `--warmup` untimed runs (the GPU side needs one for CUDA
context + library init, the CPU side gets one too so page caches are warm),
then `--runs` timed runs. Reported per variant: min/median/mean/max wall time
and frames/s, plus the number of TUM rows produced. The headline number is
the median-based CPU->GPU speedup.

Because the GPU fork's feature tracker computes in float on the device while
the CPU path uses the same algorithms on the host, the two trajectories are
NOT bit-identical; the benchmark therefore also cross-checks them
(matched-timestamp translation RMSE/max and max rotation angle difference).
Small nonzero differences are expected and healthy.

GPU side availability
---------------------
The GPU binary's CUDA + CUDA-enabled OpenCV are worker-provided (not bundled,
see README). The benchmark mirrors the launcher's probes: no GPU binary in
the bundle, no NVIDIA device, or an unresolvable shared library all mean the
GPU side is skipped with a reason (pass --require-gpu to make that fatal).
VINS_OPENCV_CUDA_DIR is honored exactly like run_vins_adapter.sh does.

Synthetic episode
-----------------
A deterministic, richly textured "wallpaper" that pans horizontally: every
feature translates uniformly, so the tracker is exercised at full MAX_CNT
load and the estimator gets real parallax. Frames are generated with the
standard library only (hand-rolled PNG) and cached under the system temp dir
(--workdir / --regen to control), so repeated benchmarks do not pay the
generation cost again.

Standard library only. The bundle is a Linux ELF: run this on the worker the
bundle was shipped to, not on macOS.

Exit codes: 0 ok (GPU side may be skipped unless --require-gpu), 1 error,
2 usage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import NoReturn, TypedDict

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = REPO / "vins_adapter"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Synthetic episode defaults.
PAN_PX = 8            # texture scroll per frame: camera pans right
BASELINE_M = 0.12     # stereo baseline (T_cam1_body tx), README example value
SCENE_DEPTH_M = 2.0   # synthetic wall distance -> fixed stereo disparity


class Episode(TypedDict):
    source: str
    frames: int
    width: int
    height: int
    stereo: bool
    max_features: int
    workdir: str


class RunResult(TypedDict):
    binary: str
    runs: list[float]
    min_s: float
    median_s: float
    mean_s: float
    max_s: float
    fps_median: float | None
    tum_rows: int


class TrajResult(TypedDict):
    cpu_rows: int
    gpu_rows: int
    matched: int
    trans_rmse_m: float | None
    trans_max_m: float | None
    rot_max_deg: float | None


class HostInfo(TypedDict, total=False):
    node: str
    system: str
    python: str
    cpu_count: int | None
    cpu_model: str
    gpu_model: str
    buildinfo: dict[str, object]


class Comparison(TypedDict):
    speedup_median: float | None
    wall_time_reduction_pct: float | None
    trajectory: TrajResult


class BenchResult(TypedDict, total=False):
    benchmark: str
    timestamp: str
    episode: Episode | dict[str, object]
    environment: HostInfo
    cpu: RunResult
    gpu: RunResult | dict[str, str]
    comparison: Comparison | None


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> NoReturn:
    print(f"benchmark: error: {msg}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------- episode gen


def _png_chunk(typ: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)


def make_texture(width: int, height: int) -> bytes:
    """Deterministic corner-rich texture, one flat W*H byte buffer.

    Per-pixel integer hash (stable across runs/platforms) mixed with a coarse
    checker so goodFeaturesToTrack always finds a full MAX_CNT set.
    """
    tex = bytearray(width * height)
    for y in range(height):
        base = y * width
        for x in range(width):
            h = (x * 73856093) ^ (y * 19349663)
            h ^= h >> 13
            v = h & 0xFF
            if ((x // 24 + y // 24) & 1) == 0:
                v = (v + 90) & 0xFF
            tex[base + x] = v
    return bytes(tex)


def write_frame(path: Path, tex: bytes, width: int, height: int, offset: int) -> None:
    """One 8-bit grayscale PNG: the texture rows rolled by `offset` pixels.

    Rolling = the camera panning over a periodic wallpaper: every feature
    moves uniformly, full parallax, no per-pixel Python cost.
    """
    rows: list[bytes] = []
    off = offset % width
    for y in range(height):
        r = tex[y * width : (y + 1) * width]
        rows.append(b"\x00" + (r[off:] + r[:off]))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), 1))
        + _png_chunk(b"IEND", b"")
    )


def write_config(
    path: Path,
    left_dir: Path,
    right_dir: Path | None,
    width: int,
    height: int,
    max_features: int,
) -> float:
    """Flat adapter config for the synthetic episode; returns fx."""
    fx = 0.6 * width
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    lines = [
        "imu: 0",
        f"num_of_cam: {2 if right_dir else 1}",
        "estimate_extrinsic: 0",
        f"max_features: {max_features}",
        f"fx: {fx}",
        f"fy: {fy}",
        f"cx: {cx}",
        f"cy: {cy}",
        "k1: 0.0",
        "k2: 0.0",
        "p1: 0.0",
        "p2: 0.0",
        f"left_dir: {left_dir}",
    ]
    if right_dir:
        lines.append(f"right_dir: {right_dir}")
    lines += [
        "T_cam0_body:",
        "- [1.0, 0.0, 0.0, 0.0]",
        "- [0.0, 1.0, 0.0, 0.0]",
        "- [0.0, 0.0, 1.0, 0.0]",
    ]
    if right_dir:
        lines += [
            "T_cam1_body:",
            f"- [1.0, 0.0, 0.0, {-BASELINE_M}]",
            "- [0.0, 1.0, 0.0, 0.0]",
            "- [0.0, 0.0, 1.0, 0.0]",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fx


def prepare_synthetic_episode(
    frames: int,
    width: int,
    height: int,
    pan_px: int,
    mono: bool,
    max_features: int,
    workdir_opt: str,
    regen: bool,
) -> tuple[Path, Episode]:
    """Generate (or reuse the cached) synthetic episode; returns (config, info)."""
    stereo = not mono
    tag = f"{frames}f_{width}x{height}_mf{max_features}_pan{pan_px}" + ("_stereo" if stereo else "_mono")
    workdir = Path(workdir_opt) if workdir_opt else Path(tempfile.gettempdir()) / f"vins_adapter_bench_{tag}"
    config = workdir / "config.yaml"
    left = workdir / "left"
    right = workdir / "right" if stereo else None

    done = config.is_file() and left.is_dir() and len(list(left.iterdir())) == frames
    if stereo and done:
        done = right is not None and right.is_dir() and len(list(right.iterdir())) == frames

    if done and not regen:
        log(f"episode: reusing cached {workdir} ({frames} frames, pass --regen to rebuild)")
        return config, _episode_info(config, frames, width, height, stereo, max_features, workdir)

    if regen and workdir.exists():
        shutil.rmtree(workdir)
    log(f"episode: generating {frames} frames {width}x{height} "
        f"({'stereo' if stereo else 'mono'}, pan {pan_px}px/frame) in {workdir} ...")
    t0 = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    left.mkdir(parents=True, exist_ok=True)
    if right is not None:
        right.mkdir(parents=True, exist_ok=True)

    tex = make_texture(width, height)
    fx = write_config(config, left, right, width, height, max_features)
    # Right camera sees the wall shifted by the stereo disparity: a point at
    # u in the left image lands at u + fx*baseline/depth in the right one.
    disparity = max(1, round(fx * BASELINE_M / SCENE_DEPTH_M))
    for i in range(frames):
        off = (i * pan_px) % width
        write_frame(left / f"{i:06d}.png", tex, width, height, off)
        if right is not None:
            write_frame(right / f"{i:06d}.png", tex, width, height, off - disparity)
    log(f"episode: done in {time.perf_counter() - t0:.1f}s (stereo disparity {disparity}px "
        f"~ wall at {SCENE_DEPTH_M}m, baseline {BASELINE_M}m)")
    return config, _episode_info(config, frames, width, height, stereo, max_features, workdir)


def _episode_info(
    config: Path, frames: int, width: int, height: int, stereo: bool, max_features: int, workdir: Path
) -> Episode:
    return Episode(
        source=str(config),
        frames=frames,
        width=width,
        height=height,
        stereo=stereo,
        max_features=max_features,
        workdir=str(workdir),
    )


def episode_from_config(config_path: Path) -> tuple[Path, int]:
    """Use a real episode config; frame count = images under its left_dir."""
    if not config_path.is_file():
        die(f"--config: no such file: {config_path}")
    left_dir = ""
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("left_dir:"):
            left_dir = line.split(":", 1)[1].strip()
            break
    if not left_dir:
        die("--config: no left_dir key found in the config")
    p = Path(left_dir)
    if not p.is_dir():
        die(f"--config: left_dir does not exist: {p}")
    n = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
    if n == 0:
        die(f"--config: no image frames under {p}")
    return config_path, n


# ------------------------------------------------------------------ running


def gpu_env() -> dict[str, str]:
    """Environment for the GPU binary, honoring VINS_OPENCV_CUDA_DIR."""
    env = os.environ.copy()
    ocv = env.get("VINS_OPENCV_CUDA_DIR", "")
    if ocv:
        extra = [d for d in (str(Path(ocv) / "lib"), str(Path(ocv) / "lib64")) if Path(d).is_dir()]
        if extra:
            existing = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(extra + [existing]) if existing else ":".join(extra)
    return env


def nvidia_device_present() -> bool:
    if Path("/dev/nvidiactl").exists():
        return True
    return shutil.which("nvidia-smi") is not None


def run_once(
    binary: Path, config: Path, out_tum: Path, env: dict[str, str], timeout: float
) -> tuple[float, int | None, str]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [str(binary), str(config), str(out_tum)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return time.perf_counter() - t0, None, f"timed out after {timeout:.0f}s"
    dt = time.perf_counter() - t0
    return dt, proc.returncode, (proc.stderr or "").strip()


def measure(
    name: str,
    binary: Path,
    config: Path,
    n_frames: int,
    runs: int,
    warmup: int,
    env: dict[str, str],
    out_dir: Path,
    timeout: float,
) -> RunResult:
    """Warmup + timed runs of one binary; returns a result dict (or raises)."""
    for k in range(max(0, warmup)):
        _, rc, err = run_once(binary, config, out_dir / f"{name}_warmup{k}.tum", env, timeout)
        if rc is None or rc != 0:
            raise RuntimeError(f"{name} warmup run failed (exit {rc}): {err[-300:]}")
    times: list[float] = []
    last_tum: Path | None = None
    for k in range(runs):
        tum = out_dir / f"{name}_run{k}.tum"
        dt, rc, err = run_once(binary, config, tum, env, timeout)
        if rc is None or rc != 0:
            raise RuntimeError(f"{name} run {k} failed (exit {rc}): {err[-300:]}")
        times.append(dt)
        last_tum = tum
    rows = 0
    if last_tum is not None and last_tum.is_file():
        rows = sum(1 for ln in last_tum.read_text(encoding="utf-8").splitlines() if ln.strip())
    med = statistics.median(times)
    return RunResult(
        binary=str(binary),
        runs=times,
        min_s=min(times),
        median_s=med,
        mean_s=statistics.mean(times),
        max_s=max(times),
        fps_median=n_frames / med if med > 0 else None,
        tum_rows=rows,
    )


# ------------------------------------------------------------- tum compare


def parse_tum(path: Path) -> dict[float, list[float]]:
    poses: dict[float, list[float]] = {}
    if not path.is_file():
        return poses
    for ln in path.read_text(encoding="utf-8").splitlines():
        parts = ln.split()
        if len(parts) != 8:
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        poses[round(vals[0], 9)] = vals[1:]  # tx ty tz qx qy qz qw
    return poses


def quat_angle_deg(q1: list[float], q2: list[float]) -> float:
    dot = abs(sum(a * b for a, b in zip(q1[3:], q2[3:])))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def compare_trajectories(cpu_tum: Path, gpu_tum: Path) -> TrajResult:
    a, b = parse_tum(cpu_tum), parse_tum(gpu_tum)
    common = sorted(set(a) & set(b))
    diffs: list[float] = []
    rot_max = 0.0
    for t in common:
        pa, pb = a[t], b[t]
        diffs.append(math.dist(pa[:3], pb[:3]))
        rot_max = max(rot_max, quat_angle_deg(pa, pb))
    return TrajResult(
        cpu_rows=len(a),
        gpu_rows=len(b),
        matched=len(common),
        trans_rmse_m=math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else None,
        trans_max_m=max(diffs) if diffs else None,
        rot_max_deg=rot_max if diffs else None,
    )


# ------------------------------------------------------------------ report


def host_info(bundle: Path) -> HostInfo:
    info: HostInfo = {
        "node": platform.node(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    cpu_model = ""
    try:
        if sys.platform.startswith("linux"):
            for ln in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if ln.startswith("model name"):
                    cpu_model = ln.split(":", 1)[1].strip()
                    break
        elif sys.platform == "darwin":
            cpu_model = subprocess.run(  # noqa: S603 -- fixed argv
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
    except Exception:  # noqa: BLE001 -- best effort only
        pass
    if cpu_model:
        info["cpu_model"] = cpu_model
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(  # noqa: S603 -- fixed argv
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                info["gpu_model"] = out.stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            pass
    buildinfo = bundle / "BUILDINFO.json"
    if buildinfo.is_file():
        try:
            parsed = json.loads(buildinfo.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                info["buildinfo"] = parsed
        except (OSError, json.JSONDecodeError):
            pass
    return info


def fmt_variant(name: str, r: RunResult, n_frames: int) -> None:
    log(f"-- {name} --")
    log(f"  binary: {r['binary']}")
    log(f"  wall: min {r['min_s']:.3f}s | median {r['median_s']:.3f}s | "
        f"mean {r['mean_s']:.3f}s | max {r['max_s']:.3f}s")
    fps = r["fps_median"]
    if fps:
        log(f"  throughput: {fps:.1f} frames/s (median, {n_frames} frames)")
    else:
        log("  throughput: n/a")
    log(f"  trajectory: {r['tum_rows']} TUM rows")


def maybe_json(path_opt: str, result: BenchResult) -> None:
    if not path_opt:
        return
    path = Path(path_opt)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log(f"results written: {path}")


# -------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE),
                    help="built bundle dir (default: vins_adapter/ next to this repo)")
    ap.add_argument("--config", metavar="YAML",
                    help="benchmark a real episode from this adapter config (ignores the synthetic episode)")
    ap.add_argument("--frames", type=int, default=300, help="synthetic episode frame count (default 300)")
    ap.add_argument("--size", nargs=2, type=int, metavar=("W", "H"), default=[752, 480],
                    help="synthetic frame size (default 752 480)")
    ap.add_argument("--pan-px", type=int, default=PAN_PX,
                    help="texture scroll per frame in px = camera pan speed (default 8)")
    ap.add_argument("--mono", action="store_true", help="monocular episode (default stereo)")
    ap.add_argument("--max-features", type=int, default=200,
                    help="max_features for the synthetic config (default 200)")
    ap.add_argument("--runs", type=int, default=5, help="timed runs per variant (default 5)")
    ap.add_argument("--warmup", type=int, default=1, help="untimed warmup runs per variant (default 1)")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-run timeout in seconds (default 900)")
    ap.add_argument("--workdir", default="", help="episode cache dir (default: system temp)")
    ap.add_argument("--regen", action="store_true", help="force episode regeneration")
    ap.add_argument("--json", default="", metavar="PATH", help="also write results as JSON")
    ap.add_argument("--require-gpu", action="store_true",
                    help="fail instead of skipping when the GPU side cannot run")
    ap.add_argument("--prepare-only", action="store_true",
                    help="generate/verify the episode and exit (no binaries needed)")
    args = ap.parse_args()

    if args.runs < 1:
        ap.error("--runs must be >= 1")
    if args.frames < 2:
        ap.error("--frames must be >= 2")

    bundle = Path(args.bundle)
    if args.config:
        config, n_frames = episode_from_config(Path(args.config))
        episode: Episode | dict[str, object] = {"source": args.config, "frames": n_frames}
    else:
        config, ep = prepare_synthetic_episode(
            frames=args.frames,
            width=args.size[0],
            height=args.size[1],
            pan_px=args.pan_px,
            mono=args.mono,
            max_features=args.max_features,
            workdir_opt=args.workdir,
            regen=args.regen,
        )
        n_frames = ep["frames"]
        episode = ep

    if args.prepare_only:
        log(f"episode ready: {config} ({n_frames} frames)")
        return 0

    cpu_bin = bundle / "vins_adapter"
    gpu_bin = bundle / "vins_adapter_gpu"
    if not cpu_bin.is_file():
        die(f"no CPU binary at {cpu_bin} -- build the bundle first (build_vins_adapter.sh)")

    log("== vins-adapter CPU vs GPU benchmark ==")
    log(f"bundle: {bundle} (gpu binary: {'yes' if gpu_bin.is_file() else 'no'})")
    log(f"episode: {episode['source']} ({n_frames} frames)")
    info = host_info(bundle)
    host_line = info.get("cpu_model", info.get("system", "unknown host"))
    if gpu_bin.is_file():
        host_line += f" | gpu: {info.get('gpu_model', 'n/a')}"
    log(f"host: {host_line}")

    # Synthetic episodes keep their run outputs beside the frames; real
    # episodes (--config) get a fresh temp dir so nothing is written next to
    # the user's data.
    if args.config:
        out_dir = Path(tempfile.mkdtemp(prefix="vins_adapter_bench_"))
    else:
        out_dir = Path(str(episode["workdir"])) / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CPU side
    try:
        cpu = measure("cpu", cpu_bin, config, n_frames, args.runs, args.warmup,
                      os.environ.copy(), out_dir, args.timeout)
    except RuntimeError as e:
        die(str(e))
    fmt_variant("cpu", cpu, n_frames)

    result: BenchResult = {
        "benchmark": "vins_adapter_cpu_vs_gpu",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episode": episode,
        "environment": info,
        "cpu": cpu,
    }

    # ---- GPU side
    gpu: RunResult | None = None
    skip_reason = ""
    if not gpu_bin.is_file():
        skip_reason = "no GPU binary in this bundle (built without --opencv-cuda)"
    elif not nvidia_device_present():
        skip_reason = "no NVIDIA device/driver detected (nvidia-smi / /dev/nvidiactl)"
    if not skip_reason:
        try:
            gpu = measure("gpu", gpu_bin, config, n_frames, args.runs, args.warmup,
                          gpu_env(), out_dir, args.timeout)
        except RuntimeError as e:
            skip_reason = str(e)
    if gpu is None:
        result["gpu"] = {"skipped": skip_reason}
        result["comparison"] = None
        maybe_json(args.json, result)
        if args.require_gpu:
            die(f"GPU side unavailable: {skip_reason}")
        log(f"-- gpu -- SKIPPED: {skip_reason}")
        return 0

    fmt_variant("gpu", gpu, n_frames)
    result["gpu"] = gpu

    # ---- comparison
    speedup = cpu["median_s"] / gpu["median_s"] if gpu["median_s"] > 0 else None
    reduction = (1.0 - gpu["median_s"] / cpu["median_s"]) * 100.0 if cpu["median_s"] > 0 else None
    traj = compare_trajectories(out_dir / f"cpu_run{args.runs - 1}.tum",
                                out_dir / f"gpu_run{args.runs - 1}.tum")
    log("-- comparison --")
    if speedup:
        log(f"  speedup (median): {speedup:.2f}x  (CPU {cpu['median_s']:.3f}s -> GPU {gpu['median_s']:.3f}s)")
    if reduction is not None:
        log(f"  wall-time reduction: {reduction:.1f}%")
    if traj["matched"]:
        rmse, tmax, rmax = traj["trans_rmse_m"], traj["trans_max_m"], traj["rot_max_deg"]
        if rmse is not None and tmax is not None and rmax is not None:
            log(f"  trajectory: {traj['matched']}/{max(traj['cpu_rows'], traj['gpu_rows'])} poses matched | "
                f"trans RMSE {rmse * 1000:.2f}mm max {tmax * 1000:.2f}mm | "
                f"rot max {rmax:.3f} deg")
    else:
        log(f"  trajectory: no common poses (cpu {traj['cpu_rows']} rows, gpu {traj['gpu_rows']} rows) "
            "-- timing comparison only")
    result["comparison"] = Comparison(
        speedup_median=speedup,
        wall_time_reduction_pct=reduction,
        trajectory=traj,
    )

    maybe_json(args.json, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

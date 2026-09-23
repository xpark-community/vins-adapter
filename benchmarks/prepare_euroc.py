#!/usr/bin/env python3
# pyright: reportAny=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
"""Prepare a EuRoC MAV sequence for the vins_adapter / CPU-vs-GPU benchmark.

The EuRoC MAV dataset (Burri et al., IJRR 2016) is the canonical
VINS-Fusion benchmark and a perfect fit for the adapter's input contract:

  - stereo 752x480 global-shutter grayscale PNG dirs (mav0/cam0, mav0/cam1),
    hardware-synchronized, filenames = ns timestamps
  - pinhole + radtan distortion (exactly the adapter's camodocal model)
  - 200 Hz IMU csv, gyro-first column order (exactly the adapter's imu_csv)
  - T_BS in sensor.yaml = camera pose in the body frame (= T_cam{i}_body)

This script turns an extracted sequence directory into a ready-to-run
adapter episode, converting only what needs converting (IMU ns->s, frame
times csv, ground truth to TUM). Images are NOT copied: the config points at
the original frame directories.

Two modes:

  1. download + prepare (one sequence, ~1-2.5 GB):

       python3 benchmarks/prepare_euroc.py --download MH_01_easy --datasets ~/datasets/euroc

     Downloads from the ETH Research Collection (the official host since the
     ASL server retired). The collection rate-limits bursts hard (HTTP 429):
     download one sequence at a time; the script retries with backoff.

  2. prepare an already-downloaded/extracted sequence:

       python3 benchmarks/prepare_euroc.py ~/datasets/euroc/MH_01_easy [options]

     The directory must contain mav0/ (cam0, cam1, imu). Outputs, written
     next to the sequence root (or --out):

       config.yaml        adapter flat config (stereo + IMU by default)
       imu_seconds.csv    timestamp[s],gx,gy,gz,ax,ay,az  (converted from ns)
       frame_times.csv    index,timestamp_ns              (from cam0 filenames)
       gt.tum             --with-gt: ground truth as TUM (qx qy qz qw, seconds)

Then benchmark:

  python3 benchmarks/benchmark_cpu_gpu.py --config <seq>/config.yaml --runs 3

Sequences (11): Machine Hall MH_01_easy MH_02_easy MH_03_medium MH_04_difficult
MH_05_difficult; Vicon Room 1 V1_01_easy V1_02_medium V1_03_difficult; Vicon
Room 2 V2_01_easy V2_02_medium V2_03_difficult. MH_* are texture-rich (best
tracker load for the GPU benchmark); V*_easy are good smoke tests.

License: free for research use; cite Burri et al., "The EuRoC micro aerial
vehicle datasets", IJRR 35(10), 2016.

Alternatives considered:
  - TUM-VI (CC BY 4.0, EuRoC-format exports): wide-FOV lenses with an
    equidistant/omni distortion model -- NOT compatible with the adapter's
    pinhole+radtan camodocal model; do not use for this benchmark.
  - KITTI Odometry (stereo PNG dirs, no IMU, vision-only): workable via a
    hand-written config but requires a registration to download.

Standard library only. Exit codes: 0 ok, 1 error, 2 usage.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import NoReturn

# All 11 sequences live in this ETH Research Collection item (handle
# 20.500.11850/690084, DOI 10.3929/ethz-b-000690084 -- the official host the
# ASL dataset page points to since robotics.ethz.ch retired).
COLLECTION_URL = "https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/690084"

SEQUENCES = (
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str) -> NoReturn:
    print(f"prepare_euroc: error: {msg}", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------- download


def download(sequence: str, datasets_dir: Path) -> Path:
    """Download and extract one sequence zip; returns the sequence root dir."""
    if sequence not in SEQUENCES:
        die(f"unknown sequence {sequence!r}; one of: {', '.join(SEQUENCES)}")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = datasets_dir / sequence
    if (seq_dir / "mav0").is_dir():
        log(f"download: {seq_dir} already extracted, reusing")
        return seq_dir

    zip_path = datasets_dir / f"{sequence}.zip"
    if not zip_path.is_file():
        url = f"{COLLECTION_URL}/{sequence}.zip"
        log(f"download: {url}")
        log("download: the collection rate-limits bursts (HTTP 429) -- one sequence at a time")
        # ~1-2.5 GB streamed to disk; retry with backoff on 429/5xx.
        for attempt in range(1, 5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "vins-adapter-benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp, zip_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = 30 * attempt
                    log(f"download: HTTP {e.code}, retrying in {wait}s (attempt {attempt}/4)")
                    time.sleep(wait)
                    continue
                zip_path.unlink(missing_ok=True)
                die(f"download failed: HTTP {e.code} for {url}")
            except (urllib.error.URLError, OSError) as e:
                zip_path.unlink(missing_ok=True)
                die(f"download failed: {e}")

    log(f"download: extracting {zip_path.name} ...")
    # EuRoC ASL zips carry mav0/ at their ROOT (unzip yields mav0/..., not
    # <sequence>/mav0/...), so extract into the sequence dir to get
    # <seq>/mav0/.
    seq_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(seq_dir)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)
        die(f"{zip_path} is not a valid zip (truncated download?) -- removed, retry")
    if not (seq_dir / "mav0").is_dir():
        die(f"extracted {seq_dir} but it has no mav0/ -- unexpected zip layout")
    log(f"download: extracted to {seq_dir}")
    return seq_dir


# ------------------------------------------------------------ sensor.yaml IO

_YAML_SCALAR = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")


def _parse_value(raw: str) -> object:
    """Scalar or [a, b, c] list from the sensor.yaml subset (stdlib only)."""
    raw = raw.split("#", 1)[0].strip()
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].replace("\n", " ")
        items = [t.strip() for t in inner.split(",") if t.strip()]
        out = []
        for t in items:
            try:
                out.append(float(t))
            except ValueError:
                out.append(t)
        return out
    try:
        return float(raw)
    except ValueError:
        return raw


def load_sensor_yaml(path: Path) -> dict[str, object]:
    """Parse the ASL sensor.yaml subset: flat scalars, flat lists, and

    T_BS:
      rows: 4
      cols: 4
      data: [ ...16 values, possibly wrapped over several lines... ]
    """
    text = path.read_text(encoding="utf-8")
    # Join wrapped list values (continuation lines start with whitespace while
    # the previous line still has an unclosed '[').
    joined: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if joined and line[:1] in (" ", "\t") and "[" in joined[-1] and "]" not in joined[-1]:
            joined[-1] += " " + stripped
        else:
            joined.append(line)
    out: dict[str, object] = {}
    for line in joined:
        m = _YAML_SCALAR.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        if raw == "":
            continue  # block keys (T_BS) -- rebuilt from rows/cols/data below
        out[key] = _parse_value(raw)
    if "data" in out and "rows" in out:
        out["T_BS"] = {"rows": out["rows"], "cols": out["cols"], "data": out["data"]}
    return out


def t_bs_rows(sensor: dict[str, object]) -> list[list[float]]:
    """T_BS 4x4 as four [r00 r01 r02 tx] rows -- the adapter's T_cam_body."""
    t = sensor.get("T_BS")
    if not isinstance(t, dict):
        die("sensor.yaml has no T_BS block")
    data, rows = t["data"], t["rows"]
    if not isinstance(data, list) or not isinstance(rows, (int, float)) or int(rows) != 4 or len(data) != 16:
        die("sensor.yaml T_BS is not a 4x4 matrix")
    try:
        return [[float(v) for v in data[i * 4:(i + 1) * 4]] for i in range(4)]  # type: ignore[index,call-arg]
    except (TypeError, ValueError):
        die("sensor.yaml T_BS data contains non-numeric values")
        raise  # unreachable; for type checkers


# ----------------------------------------------------------------- conversion


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def convert_imu(imu_csv: Path, out_csv: Path) -> int:
    """EuRoC imu/data.csv (ns, gyro xyz, accel xyz) -> seconds, same column order."""
    n = 0
    with imu_csv.open(encoding="utf-8") as src, out_csv.open("w", encoding="utf-8") as dst:
        dst.write("timestamp,gx,gy,gz,ax,ay,az\n")
        for k, line in enumerate(src):
            if k == 0 or not line.strip():
                continue  # header
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                t_ns = float(parts[0])
                vals = [float(x) for x in parts[1:7]]
            except ValueError:
                continue
            dst.write(f"{t_ns / 1e9:.9f},{vals[0]:.9f},{vals[1]:.9f},{vals[2]:.9f},"
                      f"{vals[3]:.9f},{vals[4]:.9f},{vals[5]:.9f}\n")
            n += 1
    return n


def write_frame_times(frames: list[Path], out_csv: Path) -> int:
    """index,timestamp_ns from the cam0 filenames (EuRoC names are ns stamps)."""
    with out_csv.open("w", encoding="utf-8") as dst:
        dst.write("index,timestamp_ns\n")
        for i, f in enumerate(frames):
            stamp = f.stem
            if not stamp.isdigit():
                die(f"frame filename {f.name} is not a nanosecond timestamp -- "
                    "not an unmodified EuRoC sequence?")
            dst.write(f"{i},{stamp}\n")
    return len(frames)


def write_gt_tum(gt_csv: Path, out_tum: Path) -> int:
    """EuRoC ground truth (ns, px py pz, qw qx qy qz, ...) -> TUM (s, q last)."""
    n = 0
    with gt_csv.open(encoding="utf-8") as src, out_tum.open("w", encoding="utf-8") as dst:
        for k, line in enumerate(src):
            if k == 0 or not line.strip():
                continue  # header
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                t_ns = float(parts[0])
                px, py, pz = (float(x) for x in parts[1:4])
                qw, qx, qy, qz = (float(x) for x in parts[4:8])
            except ValueError:
                continue
            dst.write(f"{t_ns / 1e9:.9f} {px:.9f} {py:.9f} {pz:.9f} "
                      f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
            n += 1
    return n


def write_config(
    path: Path,
    left_dir: Path,
    right_dir: Path | None,
    intrinsics: list[float],
    distortion: list[float],
    t_cam0: list[list[float]],
    t_cam1: list[list[float]] | None,
    imu_csv: Path | None,
    frame_times_csv: Path,
    estimate_extrinsic: int,
    max_features: int,
) -> None:
    fx, fy, cx, cy = intrinsics
    k1, k2, p1, p2 = distortion
    lines = [
        f"imu: {1 if imu_csv else 0}",
        f"num_of_cam: {2 if right_dir else 1}",
        f"estimate_extrinsic: {estimate_extrinsic}",
        f"max_features: {max_features}",
        f"fx: {fx}",
        f"fy: {fy}",
        f"cx: {cx}",
        f"cy: {cy}",
        f"k1: {k1}",
        f"k2: {k2}",
        f"p1: {p1}",
        f"p2: {p2}",
        f"left_dir: {left_dir}",
    ]
    if right_dir:
        lines.append(f"right_dir: {right_dir}")
    if imu_csv:
        lines.append(f"imu_csv: {imu_csv}")
    lines.append(f"frame_times_csv: {frame_times_csv}")
    lines.append("T_cam0_body:")
    lines += [f"- [{r[0]}, {r[1]}, {r[2]}, {r[3]}]" for r in t_cam0]
    if right_dir and t_cam1:
        lines.append("T_cam1_body:")
        lines += [f"- [{r[0]}, {r[1]}, {r[2]}, {r[3]}]" for r in t_cam1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------- main


def prepare(seq_dir: Path, out_opt: str, mono: bool, no_imu: bool,
            estimate_extrinsic: int, max_features: int, with_gt: bool) -> Path:
    mav0 = seq_dir / "mav0"
    if not mav0.is_dir():
        die(f"{seq_dir} does not contain mav0/ (pass the sequence root, e.g. .../MH_01_easy)")
    out_dir = Path(out_opt) if out_opt else seq_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cam0_yaml = mav0 / "cam0" / "sensor.yaml"
    if not cam0_yaml.is_file():
        die(f"missing {cam0_yaml}")
    cam0 = load_sensor_yaml(cam0_yaml)

    model = cam0.get("camera_model")
    if model not in (None, "pinhole"):
        die(f"cam0 camera_model is {model!r}, not pinhole -- the adapter only supports pinhole+radtan")
    dmodel = cam0.get("distortion_model")
    # "radtan" (Kalibr spelling) and "radial-tangential" (EuRoC sensor.yaml
    # spelling) are the same k1, k2, p1, p2 model the adapter feeds to
    # camodocal's PINHOLE model.
    if dmodel not in (None, "radtan", "radial-tangential"):
        die(f"cam0 distortion_model is {dmodel!r}, not radtan -- the adapter only supports pinhole+radtan")

    intrinsics = cam0.get("intrinsics")
    distortion = cam0.get("distortion_coefficients")
    if not isinstance(intrinsics, list) or len(intrinsics) != 4:
        die("cam0 sensor.yaml intrinsics is not [fx, fy, cx, cy]")
    if not isinstance(distortion, list) or len(distortion) != 4:
        die("cam0 sensor.yaml distortion_coefficients is not [k1, k2, p1, p2]")

    left = mav0 / "cam0" / "data"
    left_frames = list_images(left)
    if not left_frames:
        die(f"no frames under {left}")
    log(f"cam0: {len(left_frames)} frames")

    right_frames: list[Path] | None = None
    t_cam1: list[list[float]] | None = None
    if not mono:
        right = mav0 / "cam1" / "data"
        cam1_yaml = mav0 / "cam1" / "sensor.yaml"
        if not right.is_dir() or not cam1_yaml.is_file():
            die("stereo requested but mav0/cam1 is missing (pass --mono for monocular)")
        right_frames = list_images(right)
        if len(right_frames) != len(left_frames):
            die(f"stereo frame count mismatch: cam0 {len(left_frames)} vs cam1 {len(right_frames)} "
                "(EuRoC stereo is hardware-synced and must match)")
        t_cam1 = t_bs_rows(load_sensor_yaml(cam1_yaml))
        log(f"cam1: {len(right_frames)} frames")

    imu_csv: Path | None = None
    if not no_imu:
        src_imu = mav0 / "imu" / "data.csv"
        if not src_imu.is_file():
            src_imu = mav0 / "imu0" / "data.csv"  # TUM-VI EuRoC-format naming
        if not src_imu.is_file():
            die("no mav0/imu/data.csv (pass --no-imu for vision-only)")
        imu_csv = out_dir / "imu_seconds.csv"
        n = convert_imu(src_imu, imu_csv)
        if n == 0:
            die(f"no IMU samples parsed from {src_imu}")
        log(f"imu: {n} samples -> {imu_csv}")

    frame_times_csv = out_dir / "frame_times.csv"
    write_frame_times(left_frames, frame_times_csv)
    log(f"frame times: {len(left_frames)} -> {frame_times_csv}")

    if with_gt:
        gt_src = mav0 / "state_groundtruth_estimate0" / "data.csv"
        if gt_src.is_file():
            n = write_gt_tum(gt_src, out_dir / "gt.tum")
            log(f"ground truth: {n} poses -> {out_dir / 'gt.tum'} (TUM, qw->q-last converted)")
        else:
            log("ground truth: none in this sequence, skipped")

    config = out_dir / "config.yaml"
    write_config(config, left, right_frames[0].parent if right_frames else None,
                 [float(v) for v in intrinsics], [float(v) for v in distortion],
                 t_bs_rows(cam0), t_cam1, imu_csv, frame_times_csv,
                 estimate_extrinsic, max_features)

    stereo_s = "stereo" if right_frames else "mono"
    imu_s = "+IMU" if imu_csv else "vision-only"
    log(f"config: {config} ({stereo_s}, {imu_s}, estimate_extrinsic={estimate_extrinsic}, "
        f"max_features={max_features})")
    return config


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("sequence_dir", nargs="?", help="extracted EuRoC sequence root (contains mav0/)")
    ap.add_argument("--download", metavar="SEQ", choices=SEQUENCES,
                    help="download SEQ from the ETH Research Collection first (needs --datasets DIR)")
    ap.add_argument("--datasets", default=".", metavar="DIR",
                    help="where downloads are stored/extracted (default: current dir)")
    ap.add_argument("--out", default="", help="output dir for config/csvs (default: the sequence root)")
    ap.add_argument("--mono", action="store_true", help="monocular (cam0 only; default stereo)")
    ap.add_argument("--no-imu", action="store_true", help="vision-only (default: use IMU)")
    ap.add_argument("--estimate-extrinsic", type=int, default=2, choices=(0, 1, 2),
                    help="0 trust calibration, 1 refine around it, 2 refine online (default 2)")
    ap.add_argument("--max-features", type=int, default=200,
                    help="max_features in the generated config (default 200)")
    ap.add_argument("--with-gt", action="store_true",
                    help="also convert the ground truth to TUM (gt.tum)")
    args = ap.parse_args()

    if not args.download and not args.sequence_dir:
        ap.error("pass either a sequence dir or --download SEQ")

    if args.download:
        seq_dir = download(args.download, Path(args.datasets))
    elif args.sequence_dir:
        seq_dir = Path(args.sequence_dir)

    config = prepare(seq_dir, args.out, args.mono, args.no_imu,
                     args.estimate_extrinsic, args.max_features, args.with_gt)

    log("")
    log("ready. benchmark with:")
    log(f"  python3 benchmarks/benchmark_cpu_gpu.py --config {config} --runs 3")
    return 0


if __name__ == "__main__":
    sys.exit(main())

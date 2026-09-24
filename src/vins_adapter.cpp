// Offline VINS-Fusion adapter.
//
// License note: this file links against VINS-Fusion (GPLv3, HKUST Aerial
// Robotics Group) and is therefore distributed under GPLv3 as well. It is
// built locally inside the vins-adapter Docker image; see the LICENSE file.
//
// Contract:
//
//     vins_adapter <config.yaml> <output.tum>
//
// The flat config yaml contains:
//
//     imu: 0|1                     # imu_csv present
//     num_of_cam: 1|2              # right_dir present
//     estimate_extrinsic: 0|1|2    # 2 = refine camera/IMU extrinsics online
//     max_features: int
//     fx, fy, cx, cy, k1, k2, p1, p2: float
//     left_dir: <dir of left frames>
//     right_dir: <dir of right frames>          (optional)
//     imu_csv: <timestamp,gx,gy,gz,ax,ay,az csv>  (optional; gyro first)
//     frame_times_csv: <index,timestamp_ns csv>   (optional; per-frame clock)
//     T_cam0_body: [[R00 R01 R02 t0], [R10 R11 R12 t1], [R20 R21 R22 t2]]
//     T_cam1_body: ...                          (optional, stereo)
//
// On success writes one TUM line per optimized frame to <output.tum>:
//
//     timestamp tx ty tz qx qy qz qw        (seconds, body pose in world)
//
// TUM timestamps are the frame timestamps when frame_times_csv is given,
// otherwise a relative clock (frame_index / 30 fps). As in
// upstream VINS-Fusion, poses are only emitted once the estimator reaches the
// NON_LINEAR state (sliding window initialized) -- short or near-static
// episodes legitimately produce fewer poses than frames.
//
// Exit codes: 0 ok, 2 usage error, 1 runtime failure (message on stderr).
//
// The input/output contract itself (config resolution, csv parsing, TUM line
// formatting, the pose emission gate) lives in adapter_spec.h so it can be
// unit tested without ROS/OpenCV: tests/test_adapter_spec.cpp locks it down.

#include <ros/ros.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include "adapter_spec.h"  // the in/out spec (unit tested)
#include "estimator/estimator.h"  // Estimator + the parameters.h globals

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <cstdio>
#include <iostream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Left/right frames are paired by sort order (the caller materializes them in
// lockstep), the same convention as the KITTI sequences VINS consumes.
bool image_at(const std::vector<fs::path> &files, size_t i, cv::Mat *out) {
  *out = cv::imread(files[i].string(), cv::IMREAD_GRAYSCALE);
  return !out->empty();
}

}  // namespace

int main(int argc, char **argv) {
  using namespace vins_adapter;
  // ROS plumbing is compiled into the estimator's visualization layer; the
  // publishers are never registered, so publish() calls no-op and no master
  // is required. ros::init only seeds the node name for logging.
  ros::init(argc, argv, "vins_adapter",
            ros::init_options::AnonymousName | ros::init_options::NoRosout);

  if (argc != 3) {
    std::cerr << "usage: vins_adapter <config.yaml> <output.tum>\n";
    return 2;
  }
  const std::string config_path = argv[1];
  const std::string output_path = argv[2];

  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_path);
  } catch (const std::exception &e) {
    std::cerr << "vins_adapter: cannot read config " << config_path << ": "
              << e.what() << "\n";
    return 1;
  }

  // In-spec: resolve the flat yaml into the adapter's runtime choices
  // (defaults, gates, clamps -- unit tested in tests/test_adapter_spec.cpp).
  const AdapterConfig spec = resolve_config(cfg);

  std::string err;
  std::vector<fs::path> left_images = list_images(spec.left_dir, &err);
  if (left_images.empty()) {
    std::cerr << "vins_adapter: " << err << "\n";
    return 1;
  }
  std::vector<fs::path> right_images;
  if (spec.stereo) {
    right_images = list_images(spec.right_dir, &err);
    if (right_images.empty()) {
      std::cerr << "vins_adapter: " << err << "\n";
      return 1;
    }
    if (right_images.size() < left_images.size()) {
      std::cerr << "vins_adapter: stereo mismatch: " << left_images.size()
                << " left frames but " << right_images.size()
                << " right frames\n";
      return 1;
    }
  }
  const size_t n_frames = left_images.size();

  std::vector<ImuSample> imu;
  if (spec.use_imu) imu = read_imu_csv(spec.imu_csv);

  const std::vector<double> times =
      resolve_frame_times(spec.frame_times_csv, n_frames);

  // First frame fixes the image size the camodocal calib files are written
  // with (the feature tracker also re-derives row/col from each image).
  cv::Mat probe;
  if (!image_at(left_images, 0, &probe)) {
    std::cerr << "vins_adapter: cannot decode first left frame "
              << left_images[0].string() << "\n";
    return 1;
  }
  const int width = probe.cols;
  const int height = probe.rows;

  fs::path tmp_dir = fs::temp_directory_path() /
                     ("vins_adapter_" + std::to_string(static_cast<long>(getpid())));
  std::error_code ec;
  fs::create_directories(tmp_dir, ec);
  if (ec) {
    std::cerr << "vins_adapter: cannot create temp dir " << tmp_dir.string()
              << ": " << ec.message() << "\n";
    return 1;
  }

  fs::path cam0_yaml = tmp_dir / "cam0.yaml";
  fs::path cam1_yaml = tmp_dir / "cam1.yaml";
  write_camodocal_pinhole(cam0_yaml, "cam0", width, height, spec.fx, spec.fy,
                          spec.cx, spec.cy, spec.k1, spec.k2, spec.p1,
                          spec.p2);
  if (spec.stereo)
    write_camodocal_pinhole(cam1_yaml, "cam1", width, height, spec.fx, spec.fy,
                            spec.cx, spec.cy, spec.k1, spec.k2, spec.p1,
                            spec.p2);

  // Populate the estimator's global parameters directly (no ROS parameter
  // server). RIC/TIC follow the VINS convention: the camera pose in the body
  // frame, i.e. the config's T_cam{i}_body read as-is.
  MAX_CNT = spec.max_features;  // clamped to >= 50 in resolve_config()
  MIN_DIST = 20;
  F_THRESHOLD = 1.0;
  FLOW_BACK = 1;
  SHOW_TRACK = 0;
  MULTIPLE_THREAD = 0;  // synchronous inputImage(): deterministic offline runs
#ifdef VINS_GPU
  // GPU-fork globals (absent from the CPU fork's parameters.h): upstream they
  // are read from the ROS yaml in readParameters(), which never runs here.
  // Without this the CUDA build silently takes the CPU optical-flow and
  // feature-detection paths (both default to 0).
  USE_GPU = 1;
  USE_GPU_ACC_FLOW = 1;
#endif
  USE_IMU = spec.use_imu ? 1 : 0;
  STEREO = spec.stereo ? 1 : 0;
  NUM_OF_CAM = spec.num_of_cam;
  // ESTIMATE_EXTRINSIC already carries the vision-only gate from
  // resolve_config(): online extrinsic refinement dereferences null
  // pre_integrations without IMU.
  ESTIMATE_EXTRINSIC = spec.estimate_extrinsic;
  ACC_N = 0.1;
  ACC_W = 0.001;
  GYR_N = 0.01;
  GYR_W = 0.0001;
  G = Eigen::Vector3d(0.0, 0.0, 9.8);
  TD = 0.0;
  ESTIMATE_TD = 0;
  INIT_DEPTH = 5.0;
  MIN_PARALLAX = 10.0 / FOCAL_LENGTH;
  SOLVER_TIME = 0.04;
  NUM_ITERATIONS = 8;
  ROLLING_SHUTTER = 0;
  ROW = height - 1;
  COL = width - 1;
  VINS_RESULT_PATH = "";
  OUTPUT_FOLDER = "";
  FISHEYE_MASK = "";

  bool ok = true;
  Eigen::Matrix3d R0;
  Eigen::Vector3d t0;
  read_cam_transform(cfg["T_cam0_body"], &R0, &t0, &ok);
  if (!ok) {
    std::cerr << "vins_adapter: bad or missing T_cam0_body in config\n";
    return 1;
  }
  RIC.assign(spec.num_of_cam, Eigen::Matrix3d::Identity());
  TIC.assign(spec.num_of_cam, Eigen::Vector3d::Zero());
  RIC[0] = R0;
  TIC[0] = t0;
  if (spec.stereo) {
    Eigen::Matrix3d R1;
    Eigen::Vector3d t1;
    read_cam_transform(cfg["T_cam1_body"], &R1, &t1, &ok);
    if (!ok) {
      std::cerr << "vins_adapter: bad or missing T_cam1_body in config\n";
      return 1;
    }
    RIC[1] = R1;
    TIC[1] = t1;
  }
  CAM_NAMES.assign(1, cam0_yaml.string());
  if (spec.stereo) CAM_NAMES.push_back(cam1_yaml.string());

  FILE *out = std::fopen(output_path.c_str(), "w");
  if (out == nullptr) {
    std::cerr << "vins_adapter: cannot open output " << output_path << ": "
              << std::strerror(errno) << "\n";
    return 1;
  }

  Estimator estimator;
  estimator.setParameter();

  const char *run_error = nullptr;
  size_t imu_idx = 0;
  for (size_t i = 0; i < n_frames; ++i) {
    const double t = times[i];

    cv::Mat left;
    if (!image_at(left_images, i, &left)) {
      run_error = "cannot decode left frame";
      break;
    }
    if (spec.stereo) {
      cv::Mat right;
      if (!image_at(right_images, i, &right)) {
        run_error = "cannot decode right frame";
        break;
      }
      if (USE_IMU) {
        while (imu_idx < imu.size() && imu[imu_idx].t <= t + 1e-9) {
          estimator.inputIMU(imu[imu_idx].t, imu[imu_idx].acc,
                             imu[imu_idx].gyro);
          ++imu_idx;
        }
      }
      estimator.inputImage(t, left, right);
    } else {
      if (USE_IMU) {
        while (imu_idx < imu.size() && imu[imu_idx].t <= t + 1e-9) {
          estimator.inputIMU(imu[imu_idx].t, imu[imu_idx].acc,
                             imu[imu_idx].gyro);
          ++imu_idx;
        }
      }
      estimator.inputImage(t, left);
    }

    // inputImage() ran processMeasurements() synchronously (MULTIPLE_THREAD=0),
    // so the newest window frame carries this image's pose. Out-spec: one TUM
    // line per NON_LINEAR full-window frame (byte format locked by the unit
    // tests).
    estimator.mProcess.lock();
    if (emits_pose(estimator.solver_flag == Estimator::NON_LINEAR,
                   estimator.frame_count, WINDOW_SIZE)) {
      const Eigen::Quaterniond q(estimator.Rs[estimator.frame_count]);
      const Eigen::Vector3d &p = estimator.Ps[estimator.frame_count];
      std::fputs(format_tum_line(t, p, q).c_str(), out);
    }
    estimator.mProcess.unlock();
  }

  std::fclose(out);
  fs::remove_all(tmp_dir, ec);

  if (run_error != nullptr) {
    std::cerr << "vins_adapter: " << run_error << "\n";
    return 1;
  }
  return 0;
}

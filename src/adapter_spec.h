// The vins_adapter input/output specification, as a dependency-light library.
//
// Extracted verbatim from vins_adapter.cpp (which keeps the ROS/OpenCV/
// estimator plumbing) so the contract can be unit tested without ROS or
// OpenCV: tests/test_adapter_spec.cpp locks it down and the Dockerfile builds
// and runs that suite on every adapter build. tests/run_cpp_tests.sh runs it
// on the host. Changes here are API changes -- update the tests with intent.
//
// IN  : the flat config yaml (schema in vins_adapter.cpp's header comment),
//       optional imu csv (header row, then timestamp,gx,gy,gz,ax,ay,az --
//       gyro first), optional frame-times csv (header row, then
//       index,timestamp_ns) and the image directories.
// OUT : one TUM line per optimized frame,
//         timestamp tx ty tz qx qy qz qw      (%.9f, seconds, body in world)
//       emitted only while the estimator is NON_LINEAR with a full sliding
//       window; timestamps come from frame_times_csv or the relative clock
//       frame_index / kSyntheticFps.

#pragma once

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace vins_adapter {

namespace fs = std::filesystem;

// Relative frame clock when no frame_times_csv is given.
inline constexpr double kSyntheticFps = 30.0;

inline bool is_image_file(const fs::path &p) {
  std::string ext = p.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
         ext == ".tif" || ext == ".tiff";
}

inline std::vector<fs::path> list_images(const std::string &dir,
                                         std::string *err) {
  std::vector<fs::path> images;
  std::error_code ec;
  fs::directory_iterator it(dir, ec), end;
  if (ec) {
    *err = "cannot read image directory " + dir + ": " + ec.message();
    return images;
  }
  for (; it != end; it.increment(ec)) {
    if (ec) break;
    if (it->is_regular_file() && is_image_file(it->path()))
      images.push_back(it->path());
  }
  std::sort(images.begin(), images.end());
  if (images.empty()) *err = "no images under " + dir;
  return images;
}

// The %YAML:1.0 header is required: OpenCV >= 4.13 no longer auto-detects
// bare YAML in FileStorage::FORMAT_AUTO and throws "Input file is invalid".
inline void write_camodocal_pinhole(const fs::path &path,
                                    const std::string &name, int width,
                                    int height, double fx, double fy, double cx,
                                    double cy, double k1, double k2, double p1,
                                    double p2) {
  std::ofstream f(path);
  f << "%YAML:1.0\n---\n"
    << "model_type: PINHOLE\n"
    << "camera_name: " << name << "\n"
    << "image_width: " << width << "\n"
    << "image_height: " << height << "\n"
    << "distortion_parameters:\n"
    << "   k1: " << k1 << "\n"
    << "   k2: " << k2 << "\n"
    << "   p1: " << p1 << "\n"
    << "   p2: " << p2 << "\n"
    << "projection_parameters:\n"
    << "   fx: " << fx << "\n"
    << "   fy: " << fy << "\n"
    << "   cx: " << cx << "\n"
    << "   cy: " << cy << "\n";
}

inline void read_cam_transform(const YAML::Node &node, Eigen::Matrix3d *R,
                               Eigen::Vector3d *t, bool *ok) {
  *R = Eigen::Matrix3d::Identity();
  *t = Eigen::Vector3d::Zero();
  if (!node || !node.IsSequence() || node.size() != 3) {
    *ok = false;
    return;
  }
  for (int i = 0; i < 3; ++i) {
    const YAML::Node &row = node[i];
    if (!row.IsSequence() || row.size() != 4) {
      *ok = false;
      return;
    }
    for (int j = 0; j < 3; ++j) (*R)(i, j) = row[j].as<double>();
    (*t)(i) = row[3].as<double>();
  }
}

struct ImuSample {
  double t;
  Eigen::Vector3d acc;
  Eigen::Vector3d gyro;
};

// IMU csv convention: header line, then
// timestamp,gx,gy,gz,ax,ay,az -- seconds, gyro rad/s, accel m/s^2.
inline std::vector<ImuSample> read_imu_csv(const std::string &path) {
  std::vector<ImuSample> samples;
  std::ifstream f(path);
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (first) {  // skip the header row
      first = false;
      continue;
    }
    if (line.empty()) continue;
    std::stringstream ss(line);
    double gx, gy, gz, ax, ay, az, ts;
    char comma;
    if (ss >> ts >> comma >> gx >> comma >> gy >> comma >> gz >> comma >> ax >>
        comma >> ay >> comma >> az)
      samples.push_back({ts, Eigen::Vector3d(ax, ay, az),
                         Eigen::Vector3d(gx, gy, gz)});
  }
  return samples;
}

// frame_times_csv: header line, then index,timestamp_ns. Indices may be
// sparse; missing entries stay negative so resolve_frame_times() can fall
// back to the relative clock. Rows with a negative index are out of spec and
// skipped (they used to be an out-of-bounds vector write).
inline std::vector<double> read_frame_times_csv(const std::string &path,
                                                size_t n_frames) {
  std::vector<double> times;
  std::ifstream f(path);
  std::string line;
  bool first = true;
  while (std::getline(f, line)) {
    if (first) {
      first = false;
      continue;
    }
    if (line.empty()) continue;
    std::stringstream ss(line);
    long long index, ts_ns;
    char comma;
    if (ss >> index >> comma >> ts_ns) {
      if (index < 0) continue;
      if (static_cast<size_t>(index) >= times.size())
        times.resize(static_cast<size_t>(index) + 1, -1.0);
      times[static_cast<size_t>(index)] = ts_ns / 1e9;
    }
  }
  if (times.size() < n_frames) times.resize(n_frames, -1.0);
  return times;
}

// Frame timestamps: from frame_times_csv when given, else the relative clock
// frame_index / kSyntheticFps. Entries that are missing or out of spec
// (< 0) fall back to the relative clock one by one.
inline std::vector<double> resolve_frame_times(
    const std::string &frame_times_csv, size_t n_frames) {
  std::vector<double> times(n_frames, -1.0);
  if (!frame_times_csv.empty())
    times = read_frame_times_csv(frame_times_csv, n_frames);
  for (size_t i = 0; i < n_frames; ++i)
    if (times[i] < 0.0) times[i] = static_cast<double>(i) / kSyntheticFps;
  return times;
}

// One TUM line: timestamp tx ty tz qx qy qz qw, %.9f each. Byte-identical to
// the fprintf() the adapter used to inline in its frame loop.
inline std::string format_tum_line(double t, const Eigen::Vector3d &p,
                                   const Eigen::Quaterniond &q) {
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n", t, p.x(), p.y(),
                p.z(), q.x(), q.y(), q.z(), q.w());
  return std::string(buf);
}

// Pose emission gate (upstream VINS-Fusion behavior): only frames seen by a
// NON_LINEAR estimator with a full sliding window are written, so short or
// near-static episodes legitimately produce fewer poses than frames.
inline bool emits_pose(bool solver_nonlinear, int frame_count,
                       int window_size) {
  return solver_nonlinear && frame_count == window_size;
}

// The flat config yaml resolved into the adapter's runtime choices. The
// optional string members (imu_csv, right_dir) are only populated when the
// corresponding gate is open, mirroring when main() reads them.
struct AdapterConfig {
  bool use_imu = false;        // imu==1 and a non-empty imu_csv
  bool stereo = false;         // num_of_cam==2 and right_dir present
  int num_of_cam = 1;          // 1 or 2
  int estimate_extrinsic = 0;  // post-gate (see below); 0|1|2
  int max_features = 200;      // post-clamp: >= 50
  double fx = 500.0;
  double fy = 500.0;
  double cx = 320.0;
  double cy = 240.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  std::string left_dir;         // "" when absent (main reports the error)
  std::string right_dir;        // valid only when stereo
  std::string imu_csv;          // valid only when use_imu
  std::string frame_times_csv;  // "" when absent
};

inline AdapterConfig resolve_config(const YAML::Node &cfg) {
  AdapterConfig c;
  c.use_imu = cfg["imu"] && cfg["imu"].as<int>(0) == 1 && cfg["imu_csv"] &&
              !cfg["imu_csv"].as<std::string>().empty();
  c.stereo = cfg["num_of_cam"] && cfg["num_of_cam"].as<int>(1) == 2 &&
             cfg["right_dir"];
  c.num_of_cam = c.stereo ? 2 : 1;
  const int estimate_extrinsic =
      cfg["estimate_extrinsic"] ? cfg["estimate_extrinsic"].as<int>(2) : 2;
  // Online camera/IMU extrinsic refinement needs IMU preintegration: without
  // IMU, pre_integrations[] stay null and processImage() would dereference
  // pre_integrations[frame_count]->delta_q (estimator.cpp, ESTIMATE_EXTRINSIC
  // == 2 branch). Force it off for vision-only runs.
  c.estimate_extrinsic = c.use_imu ? estimate_extrinsic : 0;
  c.fx = cfg["fx"] ? cfg["fx"].as<double>() : 500.0;
  c.fy = cfg["fy"] ? cfg["fy"].as<double>() : 500.0;
  c.cx = cfg["cx"] ? cfg["cx"].as<double>() : 320.0;
  c.cy = cfg["cy"] ? cfg["cy"].as<double>() : 240.0;
  c.k1 = cfg["k1"] ? cfg["k1"].as<double>() : 0.0;
  c.k2 = cfg["k2"] ? cfg["k2"].as<double>() : 0.0;
  c.p1 = cfg["p1"] ? cfg["p1"].as<double>() : 0.0;
  c.p2 = cfg["p2"] ? cfg["p2"].as<double>() : 0.0;
  c.left_dir = cfg["left_dir"] ? cfg["left_dir"].as<std::string>() : "";
  if (c.use_imu) c.imu_csv = cfg["imu_csv"].as<std::string>();
  if (c.stereo) c.right_dir = cfg["right_dir"].as<std::string>();
  if (cfg["frame_times_csv"])
    c.frame_times_csv = cfg["frame_times_csv"].as<std::string>();
  const int max_features =
      cfg["max_features"] ? cfg["max_features"].as<int>() : 200;
  c.max_features = std::max(50, max_features);
  return c;
}

}  // namespace vins_adapter

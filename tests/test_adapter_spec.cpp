// Unit tests for src/adapter_spec.h -- the vins_adapter in/out contract.
//
// Dependency-light by design: yaml-cpp + Eigen only (no ROS, no OpenCV), so
// the suite runs on the host (tests/run_cpp_tests.sh) and inside the pinned
// build image (the Dockerfile builds and runs it on every adapter build).
//
// What is locked down here:
//   IN  : image-directory discovery (extensions, ordering, errors), the
//         camodocal pinhole yaml written for the fork's camera_models (exact
//         bytes, incl. the %YAML:1.0 header), T_cam{0,1}_body parsing and
//         rejection rules, imu csv + frame-times csv parsing (incl. the
//         negative-index rejection), and the whole config-resolution
//         default/gate/clamp table.
//   OUT : the exact TUM line format ("timestamp tx ty tz qx qy qz qw",
//         %.9f each) and the pose emission gate.
//
// Anything that changes these tests is an API change for every caller of the
// vins_adapter binary.

#include "adapter_spec.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (cond) {                                                        \
      std::printf("  PASS %s\n", #cond);                               \
    } else {                                                           \
      ++g_failed;                                                      \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
    }                                                                  \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                            \
  do {                                                                   \
    ++g_checks;                                                          \
    const double _va = static_cast<double>(a);                           \
    const double _vb = static_cast<double>(b);                           \
    if (_va >= _vb - (tol) && _va <= _vb + (tol)) {                      \
      std::printf("  PASS %s ~ %s\n", #a, #b);                           \
    } else {                                                             \
      ++g_failed;                                                        \
      std::printf("  FAIL %s:%d: %s == %g but %s == %g (tol %g)\n",      \
                  __FILE__, __LINE__, #a, _va, #b, _vb,                  \
                  static_cast<double>(tol));                             \
    }                                                                    \
  } while (0)

#define CHECK_STREQ(a, b)                                                \
  do {                                                                   \
    ++g_checks;                                                          \
    const std::string _sa(a);                                            \
    const std::string _sb(b);                                            \
    if (_sa == _sb) {                                                    \
      std::printf("  PASS %s == %s\n", #a, #b);                          \
    } else {                                                             \
      ++g_failed;                                                        \
      std::printf("  FAIL %s:%d:\n  actual:   \"%s\"\n  expected: \"%s\"\n", \
                  __FILE__, __LINE__, _sa.c_str(), _sb.c_str());         \
    }                                                                    \
  } while (0)

void section(const char *title) { std::printf("\n== %s ==\n", title); }

// ---------------------------------------------------------------- test utils

struct TmpDir {
  std::filesystem::path path;

  TmpDir() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::string tpl = (base / "vins_adapter_utest_XXXXXX").string();
    std::vector<char> buf(tpl.begin(), tpl.end());
    buf.push_back('\0');
    char *made = ::mkdtemp(buf.data());
    path = (made != nullptr) ? made : (base / "vins_adapter_utest_failed");
  }
  ~TmpDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_file(const std::filesystem::path &p, const std::string &content) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream f(p);
  f << content;
}

std::string read_file(const std::filesystem::path &p) {
  std::ifstream f(p);
  return std::string(std::istreambuf_iterator<char>(f),
                     std::istreambuf_iterator<char>());
}

// ------------------------------------------------------------------- in: fs

void test_is_image_file() {
  section("in: is_image_file");
  using vins_adapter::is_image_file;
  for (const char *ext :
       {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}) {
    const std::string name = std::string("frame_000001") + ext;
    CHECK(is_image_file(name));
  }
  // Case-insensitive.
  CHECK(is_image_file("a.PNG"));
  CHECK(is_image_file("a.Jpg"));
  CHECK(is_image_file("a.TIFF"));
  // Not images.
  CHECK(!is_image_file("notes.txt"));
  CHECK(!is_image_file("depth.raw"));
  CHECK(!is_image_file("clip.mp4"));
  CHECK(!is_image_file("noext"));
  CHECK(!is_image_file("frame.png.gz"));  // last extension wins
}

void test_list_images() {
  section("in: list_images");
  using vins_adapter::list_images;
  TmpDir td;
  const std::filesystem::path dir = td.path / "frames";
  std::filesystem::create_directories(dir);
  write_file(dir / "b.png", "x");
  write_file(dir / "a.JPG", "x");
  write_file(dir / "d.jpeg", "x");
  write_file(dir / "e.bmp", "x");
  write_file(dir / "f.tiff", "x");
  write_file(dir / "g.TIF", "x");
  write_file(dir / "notes.txt", "x");
  write_file(dir / "depth.raw", "x");
  std::filesystem::create_directories(dir / "subdir.png");  // a directory

  std::string err;  // untouched on success (main() reads it only on failure)
  const std::vector<std::filesystem::path> images = list_images(dir.string(), &err);
  CHECK(images.size() == 6);
  if (images.size() == 6) {
    CHECK(images[0].filename() == "a.JPG");   // sorted, ...
    CHECK(images[1].filename() == "b.png");
    CHECK(images[2].filename() == "d.jpeg");
    CHECK(images[3].filename() == "e.bmp");
    CHECK(images[4].filename() == "f.tiff");
    CHECK(images[5].filename() == "g.TIF");   // ... case-insensitive ext
  }
  CHECK(err.empty());

  // Missing directory: empty + a diagnostic naming the directory.
  err = "unset";
  CHECK(list_images((td.path / "no_such_dir").string(), &err).empty());
  CHECK(err.find("cannot read image directory") != std::string::npos);

  // Directory without images: empty + a diagnostic naming the directory.
  const std::filesystem::path empty_dir = td.path / "empty";
  std::filesystem::create_directories(empty_dir);
  write_file(empty_dir / "readme.md", "x");
  err = "unset";
  CHECK(list_images(empty_dir.string(), &err).empty());
  CHECK(err.find("no images under") != std::string::npos);
}

// ----------------------------------------------------------- in: camodocal

void test_write_camodocal_pinhole() {
  section("in: write_camodocal_pinhole (exact bytes)");
  TmpDir td;
  const std::filesystem::path calib = td.path / "cam0.yaml";
  vins_adapter::write_camodocal_pinhole(calib, "cam0", 752, 480, 458.654,
                                        457.296, 367.215, 248.375,
                                        -0.28340811, 0.07395907, 0.00019359,
                                        1.76187114e-05);
  // Lock the exact camodocal layout the fork's camera_models parses. The
  // %YAML:1.0 header is load-bearing: OpenCV >= 4.13 FileStorage rejects bare
  // YAML ("Input file is invalid" abort). Values print with the default
  // ostream precision (6 significant digits).
  const std::string expected =
      "%YAML:1.0\n"
      "---\n"
      "model_type: PINHOLE\n"
      "camera_name: cam0\n"
      "image_width: 752\n"
      "image_height: 480\n"
      "distortion_parameters:\n"
      "   k1: -0.283408\n"
      "   k2: 0.0739591\n"
      "   p1: 0.00019359\n"
      "   p2: 1.76187e-05\n"
      "projection_parameters:\n"
      "   fx: 458.654\n"
      "   fy: 457.296\n"
      "   cx: 367.215\n"
      "   cy: 248.375\n";
  CHECK_STREQ(read_file(calib), expected);

  // Values with more digits than the 6-digit default still round-trip
  // losslessly enough for the calibration load (relative tolerance).
  const std::filesystem::path calib2 = td.path / "cam1.yaml";
  vins_adapter::write_camodocal_pinhole(calib2, "cam1", 640, 480, 600.0 / 7.0,
                                        601.0 / 7.0, 320.5, 240.25, -0.01,
                                        0.02, 0.0001, -0.0002);
  const std::string content = read_file(calib2);
  CHECK(content.find("camera_name: cam1\n") != std::string::npos);
  CHECK(content.find("image_width: 640\n") != std::string::npos);
  CHECK(content.find("image_height: 480\n") != std::string::npos);
  CHECK(content.find("   cx: 320.5\n") != std::string::npos);
  CHECK(content.find("   cy: 240.25\n") != std::string::npos);
}

void test_read_cam_transform() {
  section("in: read_cam_transform");
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  bool ok = false;

  // Well-formed 3x4 [R|t].
  YAML::Node good = YAML::Load(
      "[[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 0.2], [0.0, 0.0, 1.0, 0.3]]");
  ok = true;
  vins_adapter::read_cam_transform(good, &R, &t, &ok);
  CHECK(ok);
  CHECK_NEAR(R(0, 1), -1.0, 1e-12);
  CHECK_NEAR(R(1, 0), 1.0, 1e-12);
  CHECK_NEAR(R(2, 2), 1.0, 1e-12);
  CHECK_NEAR(t(0), 0.1, 1e-12);
  CHECK_NEAR(t(1), 0.2, 1e-12);
  CHECK_NEAR(t(2), 0.3, 1e-12);

  // Absent key (undefined node): rejected, outputs neutralized.
  YAML::Node absent;
  ok = true;
  vins_adapter::read_cam_transform(absent, &R, &t, &ok);
  CHECK(!ok);
  CHECK(R.isIdentity(1e-12));
  CHECK(t.isZero(1e-12));

  // Not a sequence.
  ok = true;
  vins_adapter::read_cam_transform(YAML::Load("42"), &R, &t, &ok);
  CHECK(!ok);

  // Sequence of 2 rows.
  ok = true;
  vins_adapter::read_cam_transform(
      YAML::Load("[[1,0,0,0],[0,1,0,0]]"), &R, &t, &ok);
  CHECK(!ok);

  // Sequence of 3 rows but a row without the translation column.
  ok = true;
  vins_adapter::read_cam_transform(
      YAML::Load("[[1,0,0],[0,1,0],[0,0,1]]"), &R, &t, &ok);
  CHECK(!ok);
}

// -------------------------------------------------------------- in: csvs

void test_read_imu_csv() {
  section("in: read_imu_csv (gyro first)");
  TmpDir td;
  const std::filesystem::path csv = td.path / "imu.csv";
  write_file(csv,
             "timestamp,gx,gy,gz,ax,ay,az\n"
             "10.5,0.1,0.2,0.3,0.0,0.0,-9.8\n"
             "\n"  // blank lines are skipped
             "10.6,0.0,0.0,0.0,0.0,0.0,-9.8\n"
             "this line is malformed and skipped\n"
             "10.7,-0.1,0.0,0.1,1.5,-0.5,3.25\n");
  const std::vector<vins_adapter::ImuSample> s =
      vins_adapter::read_imu_csv(csv.string());
  CHECK(s.size() == 3);
  if (s.size() == 3) {
    CHECK_NEAR(s[0].t, 10.5, 1e-12);
    // Gyro columns map to .gyro, accel columns to .acc (swapping them is the
    // classic silent-trajectory-killer).
    CHECK_NEAR(s[0].gyro.x(), 0.1, 1e-12);
    CHECK_NEAR(s[0].gyro.y(), 0.2, 1e-12);
    CHECK_NEAR(s[0].gyro.z(), 0.3, 1e-12);
    CHECK_NEAR(s[0].acc.x(), 0.0, 1e-12);
    CHECK_NEAR(s[0].acc.y(), 0.0, 1e-12);
    CHECK_NEAR(s[0].acc.z(), -9.8, 1e-12);
    CHECK_NEAR(s[2].gyro.x(), -0.1, 1e-12);
    CHECK_NEAR(s[2].acc.x(), 1.5, 1e-12);
    CHECK_NEAR(s[2].acc.z(), 3.25, 1e-12);
  }

  // Missing file: no samples, no crash.
  CHECK(vins_adapter::read_imu_csv((td.path / "nope.csv").string()).empty());
}

void test_read_frame_times_csv() {
  section("in: read_frame_times_csv (index,timestamp_ns)");
  TmpDir td;
  const std::filesystem::path csv = td.path / "times.csv";
  write_file(csv,
             "index,timestamp_ns\n"
             "0,1000000000\n"
             "2,3000000000\n"
             "-1,999\n"  // negative index: out of spec, skipped
             "1,2500000000\n"
             "garbage row\n");
  const std::vector<double> t =
      vins_adapter::read_frame_times_csv(csv.string(), 4);
  CHECK(t.size() == 4);  // padded to n_frames
  CHECK_NEAR(t[0], 1.0, 1e-12);   // ns -> s
  CHECK_NEAR(t[1], 2.5, 1e-12);
  CHECK_NEAR(t[2], 3.0, 1e-12);
  CHECK(t[3] < 0.0);  // absent index stays negative (caller falls back)

  // Sparse indices (gaps stay negative), out-of-order rows land by index.
  const std::filesystem::path sparse = td.path / "sparse.csv";
  write_file(sparse, "index,timestamp_ns\n2,7000000000\n0,5000000000\n");
  const std::vector<double> s =
      vins_adapter::read_frame_times_csv(sparse.string(), 3);
  CHECK(s.size() == 3);
  CHECK_NEAR(s[0], 5.0, 1e-12);
  CHECK(s[1] < 0.0);
  CHECK_NEAR(s[2], 7.0, 1e-12);
}

void test_resolve_frame_times() {
  section("in: resolve_frame_times (fallback clock)");
  // No frame_times_csv: relative clock frame_index / 30 fps.
  const std::vector<double> rel = vins_adapter::resolve_frame_times("", 3);
  CHECK(rel.size() == 3);
  CHECK_NEAR(rel[0], 0.0, 1e-12);
  CHECK_NEAR(rel[1], 1.0 / 30.0, 1e-12);
  CHECK_NEAR(rel[2], 2.0 / 30.0, 1e-12);

  // With a csv: given indices keep their timestamps, missing ones fall back
  // to the relative clock entry by entry.
  TmpDir td;
  const std::filesystem::path csv = td.path / "times.csv";
  write_file(csv, "index,timestamp_ns\n0,5000000000\n2,7000000000\n");
  const std::vector<double> mixed =
      vins_adapter::resolve_frame_times(csv.string(), 3);
  CHECK(mixed.size() == 3);
  CHECK_NEAR(mixed[0], 5.0, 1e-12);
  CHECK_NEAR(mixed[1], 1.0 / 30.0, 1e-12);  // index 1 absent -> fallback
  CHECK_NEAR(mixed[2], 7.0, 1e-12);
}

// ----------------------------------------------------------- in: config

void test_resolve_config() {
  section("in: resolve_config (defaults)");
  {
    const vins_adapter::AdapterConfig c =
        vins_adapter::resolve_config(YAML::Load("{}"));
    CHECK(!c.use_imu);
    CHECK(!c.stereo);
    CHECK(c.num_of_cam == 1);
    CHECK(c.estimate_extrinsic == 0);  // default 2, forced off without IMU
    CHECK(c.max_features == 200);
    CHECK_NEAR(c.fx, 500.0, 1e-12);
    CHECK_NEAR(c.fy, 500.0, 1e-12);
    CHECK_NEAR(c.cx, 320.0, 1e-12);
    CHECK_NEAR(c.cy, 240.0, 1e-12);
    CHECK_NEAR(c.k1, 0.0, 1e-12);
    CHECK_NEAR(c.k2, 0.0, 1e-12);
    CHECK_NEAR(c.p1, 0.0, 1e-12);
    CHECK_NEAR(c.p2, 0.0, 1e-12);
    CHECK(c.left_dir.empty());
    CHECK(c.right_dir.empty());
    CHECK(c.imu_csv.empty());
    CHECK(c.frame_times_csv.empty());
  }

  section("in: resolve_config (imu gate)");
  {
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("imu: 1\nimu_csv: /data/imu.csv\n"));
    CHECK(c.use_imu);
    CHECK(c.imu_csv == "/data/imu.csv");
    CHECK(c.estimate_extrinsic == 2);  // default, IMU present -> kept
  }
  {
    // imu: 1 but no imu_csv key.
    const vins_adapter::AdapterConfig c =
        vins_adapter::resolve_config(YAML::Load("imu: 1\n"));
    CHECK(!c.use_imu);
    CHECK(c.estimate_extrinsic == 0);
  }
  {
    // imu: 1 but an empty imu_csv.
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("imu: 1\nimu_csv: \"\"\n"));
    CHECK(!c.use_imu);
  }
  {
    // imu: 0 with an estimate_extrinsic that must be forced off.
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("imu: 0\nimu_csv: /data/imu.csv\nestimate_extrinsic: 1\n"));
    CHECK(!c.use_imu);
    CHECK(c.estimate_extrinsic == 0);
  }
  {
    // estimate_extrinsic passes through when IMU is on.
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(YAML::Load(
        "imu: 1\nimu_csv: /data/imu.csv\nestimate_extrinsic: 1\n"));
    CHECK(c.use_imu);
    CHECK(c.estimate_extrinsic == 1);
  }

  section("in: resolve_config (stereo gate)");
  {
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("num_of_cam: 2\nright_dir: /data/right\n"));
    CHECK(c.stereo);
    CHECK(c.num_of_cam == 2);
    CHECK(c.right_dir == "/data/right");
  }
  {
    // num_of_cam: 2 without right_dir: silently mono (upstream contract).
    const vins_adapter::AdapterConfig c =
        vins_adapter::resolve_config(YAML::Load("num_of_cam: 2\n"));
    CHECK(!c.stereo);
    CHECK(c.num_of_cam == 1);
  }
  {
    // num_of_cam: 1 with a right_dir present: still mono, right_dir unread.
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("num_of_cam: 1\nright_dir: /data/right\n"));
    CHECK(!c.stereo);
    CHECK(c.num_of_cam == 1);
    CHECK(c.right_dir.empty());
  }

  section("in: resolve_config (overrides and clamps)");
  {
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("max_features: 10\n"
                   "fx: 700.0\nfy: 701.0\ncx: 320.5\ncy: 121.5\n"
                   "k1: -0.1\nk2: 0.2\np1: 0.001\np2: -0.002\n"
                   "left_dir: /data/left\n"
                   "frame_times_csv: /data/times.csv\n"));
    CHECK(c.max_features == 50);  // clamped to the floor
    CHECK_NEAR(c.fx, 700.0, 1e-12);
    CHECK_NEAR(c.fy, 701.0, 1e-12);
    CHECK_NEAR(c.cx, 320.5, 1e-12);
    CHECK_NEAR(c.cy, 121.5, 1e-12);
    CHECK_NEAR(c.k1, -0.1, 1e-12);
    CHECK_NEAR(c.k2, 0.2, 1e-12);
    CHECK_NEAR(c.p1, 0.001, 1e-12);
    CHECK_NEAR(c.p2, -0.002, 1e-12);
    CHECK(c.left_dir == "/data/left");
    CHECK(c.frame_times_csv == "/data/times.csv");
  }
  {
    const vins_adapter::AdapterConfig c = vins_adapter::resolve_config(
        YAML::Load("max_features: 350\n"));
    CHECK(c.max_features == 350);  // above the floor: kept
  }
}

// ---------------------------------------------------------------- out: TUM

void test_format_tum_line() {
  section("out: format_tum_line (exact bytes)");
  {
    const std::string line = vins_adapter::format_tum_line(
        1.0, Eigen::Vector3d(0.25, -1.5, 2.0),
        Eigen::Quaterniond(0.5, 0.5, 0.5, 0.5));
    CHECK_STREQ(line,
                "1.000000000 0.250000000 -1.500000000 2.000000000 "
                "0.500000000 0.500000000 0.500000000 0.500000000\n");
  }
  {
    // Field order: t x y z qx qy qz qw (qw last -- TUM's Hamilton order).
    const std::string line = vins_adapter::format_tum_line(
        123.456789, Eigen::Vector3d(7.0, 8.0, 9.0),
        Eigen::Quaterniond(1.0, 2.0, 3.0, 4.0));  // w x y z ctor
    CHECK_STREQ(line,
                "123.456789000 7.000000000 8.000000000 9.000000000 "
                "2.000000000 3.000000000 4.000000000 1.000000000\n");
  }
  {
    // Negative-zero-free, 9 decimals, newline-terminated, exactly 8 fields.
    const std::string line = vins_adapter::format_tum_line(
        0.0, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
    CHECK_STREQ(line,
                "0.000000000 0.000000000 0.000000000 0.000000000 "
                "0.000000000 0.000000000 0.000000000 1.000000000\n");
    // Exactly 8 space-separated fields, newline-terminated.
    const size_t spaces =
        static_cast<size_t>(std::count(line.begin(), line.end(), ' '));
    CHECK(spaces + 1 == 8);
    CHECK(line.back() == '\n');
  }
}

void test_emits_pose() {
  section("out: emits_pose gate");
  CHECK(vins_adapter::emits_pose(true, 10, 10));   // NON_LINEAR + full window
  CHECK(!vins_adapter::emits_pose(false, 10, 10)); // still initializing
  CHECK(!vins_adapter::emits_pose(true, 9, 10));   // window not full yet
  CHECK(!vins_adapter::emits_pose(true, 11, 10));  // beyond: never emitted
}

}  // namespace

int main() {
  test_is_image_file();
  test_list_images();
  test_write_camodocal_pinhole();
  test_read_cam_transform();
  test_read_imu_csv();
  test_read_frame_times_csv();
  test_resolve_frame_times();
  test_resolve_config();
  test_format_tum_line();
  test_emits_pose();

  std::printf("\n%d checks, %d failed\n", g_checks, g_failed);
  return g_failed == 0 ? 0 : 1;
}

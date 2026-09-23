# Appended to the fork's vins_estimator/CMakeLists.txt by the Dockerfile.
# Runs in that package's scope: catkin_* / OpenCV / Ceres variables and the
# vins_lib target are already defined there.
#
# Built in two workspaces: the CPU fork builds vins_adapter, the CUDA fork
# builds vins_adapter_gpu (VINS_GPU=ON, which also defines VINS_GPU so the
# shared adapter_main.cpp can set the fork's GPU globals).

find_package(yaml-cpp REQUIRED)

# conda-forge's yaml-cpp 0.8 exports the namespaced target yaml-cpp::yaml-cpp,
# while the legacy YAML_CPP_LIBRARIES variable holds the bare name "yaml-cpp";
# passing that to the linker fails because the conda lib lives in
# /opt/ros1/lib, not a default ld search path. Prefer the real target, fall
# back to locating the library file explicitly.
if(TARGET yaml-cpp::yaml-cpp)
  set(VINS_YAML_CPP yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(VINS_YAML_CPP yaml-cpp)
else()
  find_library(VINS_YAML_CPP NAMES yaml-cpp
    HINTS ${CMAKE_PREFIX_PATH} /opt/ros1
    PATH_SUFFIXES lib lib64
    REQUIRED)
endif()

if(VINS_GPU)
  set(VINS_ADAPTER_TARGET vins_adapter_gpu)
else()
  set(VINS_ADAPTER_TARGET vins_adapter)
endif()

add_executable(${VINS_ADAPTER_TARGET} src/adapter_main.cpp)
if(VINS_GPU)
  target_compile_definitions(${VINS_ADAPTER_TARGET} PRIVATE VINS_GPU=1)
endif()
set_property(TARGET ${VINS_ADAPTER_TARGET} PROPERTY CXX_STANDARD 17)
set_property(TARGET ${VINS_ADAPTER_TARGET} PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(${VINS_ADAPTER_TARGET}
  vins_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${VINS_YAML_CPP}
)

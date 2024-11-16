cmake_minimum_required(VERSION 3.14)
project(EfficientNetTensorRT)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# # Use vcpkg toolchain
set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_PREFIX_PATH "C:/tools/vcpkg/installed/x64-windows")

# Include TensorRT and CUDA directories
include_directories(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.6.0.26/include"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include"
)

# Link TensorRT and CUDA directories
link_directories(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.6.0.26/lib"
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/lib/x64"
)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(main main.cpp)

# Link TensorRT, CUDA, OpenCV, and Protobuf libraries
target_link_libraries(main
    nvinfer_10.lib
    nvinfer_plugin_10.lib
    cudart.lib
    ${OpenCV_LIBS}
)

cmake_minimum_required(VERSION 3.14)
project(EfficientNetTensorRT)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)

# Define vcpkg path
set(CMAKE_TOOLCHAIN_FILE "C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Include directories
include_directories(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.6.0.26/include"
    "C:/tools/vcpkg/packages/opencv4_x64-windows/include"
)

# Link directories
link_directories(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT-10.6.0.26/lib"
    "C:/tools/vcpkg/packages/opencv4_x64-windows/lib"
)

# Add the executable
add_executable(main main.cpp)

# Link TensorRT libraries
target_link_libraries(main
    nvinfer.lib
    nvinfer_plugin.lib
    cudart.lib
)

# Link OpenCV libraries explicitly
target_link_libraries(main
    opencv_core4.lib
    opencv_imgproc4.lib
    opencv_imgcodecs4.lib
    opencv_highgui4.lib
)

# Find CUDA (adjust CUDA path if necessary)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(main ${CUDA_LIBRARIES})

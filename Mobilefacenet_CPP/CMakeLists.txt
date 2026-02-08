cmake_minimum_required(VERSION 3.10)
project(MobileFaceNet_Test)

set(CMAKE_CXX_STANDARD 11)

# Find OpenMP (REQUIRED for ncnn)
find_package(OpenMP REQUIRED)

# Find OpenCV (this is the important addition)
find_package(OpenCV REQUIRED)

# ncnn (system-wide)
set(NCNN_INCLUDE_DIR /usr/local/include/ncnn)
set(NCNN_LIB /usr/local/lib/ncnn/libncnn.a)

# glslang libraries (only needed if you're actually using Vulkan)
set(GLSLANG_LIBS
    /usr/local/lib/libglslang.a
    /usr/local/lib/libSPIRV.a
    /usr/local/lib/libMachineIndependent.a
    /usr/local/lib/libGenericCodeGen.a
    /usr/local/lib/libOSDependent.a
    /usr/local/lib/libglslang-default-resource-limits.a
)

# Include directories
include_directories(${NCNN_INCLUDE_DIR})
# OpenCV include dirs are automatically added by find_package

add_executable(test_model src/main.cpp)

target_link_libraries(test_model
    ${NCNN_LIB}
    ${GLSLANG_LIBS}
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}          # <-- this links all OpenCV modules you need
    pthread
    dl
)

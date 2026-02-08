cmake_minimum_required(VERSION 3.10)
project(FaceEngineRPi)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags for RPi4 (Cortex-A72)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=armv8-a+crc -mtune=cortex-a72 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=armv8-a+crc -mtune=cortex-a72 -DNDEBUG")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# ncnn
set(NCNN_INCLUDE_DIR /usr/local/include/ncnn)
set(NCNN_LIB /usr/local/lib/ncnn/libncnn.a)

# glslang (if needed for your ncnn build)
set(GLSLANG_LIBS "")
if(EXISTS /usr/local/lib/libglslang.a)
    set(GLSLANG_LIBS
        /usr/local/lib/libglslang.a
        /usr/local/lib/libSPIRV.a
        /usr/local/lib/libMachineIndependent.a
        /usr/local/lib/libGenericCodeGen.a
        /usr/local/lib/libOSDependent.a
        /usr/local/lib/libglslang-default-resource-limits.a
    )
endif()

include_directories(${NCNN_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src)

# ====== Shared library for Python ======
add_library(face_engine SHARED
    src/face_detector.cpp
    src/mobilefacenet.cpp
    src/face_engine.cpp
)

target_link_libraries(face_engine
    ${NCNN_LIB}
    ${GLSLANG_LIBS}
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    pthread
    dl
)

# ====== Standalone C++ executable (Option B) ======
add_executable(face_system_cpp
    src/face_detector.cpp
    src/mobilefacenet.cpp
    src/face_engine.cpp
    src/main_standalone.cpp
)

target_link_libraries(face_system_cpp
    ${NCNN_LIB}
    ${GLSLANG_LIBS}
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
    pthread
    dl
)

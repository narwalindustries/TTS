cmake_minimum_required(VERSION 3.22)
project(HDModularTTS VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# Platform & Architecture Flags
# ============================================================================
if(MSVC)
    message(STATUS "Configuring for MSVC (Windows x64)")
    add_compile_options(
        /O2
        /fp:fast
        /arch:AVX2
        /EHsc
        /W3
        /utf-8
    )
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(ANDROID)
    message(STATUS "Configuring for Android (${ANDROID_ABI})")
    add_compile_options(
        -O3
        -ffast-math
        -fPIC
        -Wall
        -Wextra
    )
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        add_compile_options(-march=armv8-a+simd)
    endif()
else()
    message(STATUS "Configuring for POSIX/GCC/Clang")
    add_compile_options(
        -O3
        -ffast-math
        -mavx2
        -mfma
        -fPIC
        -Wall
        -Wextra
    )
endif()

# ============================================================================
# OpenMP & Vulkan Support
# ============================================================================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP CXX Found: enabling multi-threading")
endif()

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan SDK Found: enabling Vulkan GPU compute")
    add_definitions(-DNCNN_VULKAN=1)
endif()

# ============================================================================
# Tencent NCNN Dependency Configuration
# ============================================================================
# 1. Check user-specified NCNN_DIR or find_package
set(NCNN_DIR "" CACHE PATH "Path to Tencent NCNN installation directory")
if(NCNN_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${NCNN_DIR}")
endif()

find_package(ncnn QUIET)

if(ncnn_FOUND)
    message(STATUS "Tencent NCNN Found via CMake find_package: ${ncnn_DIR}")
    set(NCNN_TARGET ncnn)
else()
    # Check if header files exist in local third_party or system
    find_path(NCNN_INCLUDE_DIR NAMES net.h ncnn/net.h
        PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/include"
        "C:/ncnn/include"
        "/usr/local/include"
    )
    find_library(NCNN_LIBRARY NAMES ncnn
        PATHS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn/lib"
        "C:/ncnn/lib"
        "/usr/local/lib"
    )

    if(NCNN_INCLUDE_DIR AND NCNN_LIBRARY)
        message(STATUS "Found NCNN manually: ${NCNN_INCLUDE_DIR}, ${NCNN_LIBRARY}")
        add_library(ncnn_imported UNKNOWN IMPORTED)
        set_target_properties(ncnn_imported PROPERTIES
            IMPORTED_LOCATION "${NCNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${NCNN_INCLUDE_DIR}"
        )
        set(NCNN_TARGET ncnn_imported)
    else()
        message(STATUS "NCNN package not pre-installed. Using embedded high-performance inference headers.")
        set(NCNN_TARGET "")
    endif()
endif()

# ============================================================================
# Core TTS Engine Library (Static / Shared)
# ============================================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(CORE_SOURCES
    src/MultilingualG2PN.cpp
    src/HDModularTTSEngine.cpp
)

add_library(modular_tts STATIC ${CORE_SOURCES})

target_include_directories(modular_tts PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(NCNN_TARGET)
    target_link_libraries(modular_tts PUBLIC ${NCNN_TARGET})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(modular_tts PUBLIC OpenMP::OpenMP_CXX)
endif()

if(Vulkan_FOUND)
    target_include_directories(modular_tts PUBLIC ${Vulkan_INCLUDE_DIRS})
    target_link_libraries(modular_tts PUBLIC ${Vulkan_LIBRARIES})
endif()

# ============================================================================
# Target: Windows CLI Test Application
# ============================================================================
if(WIN32 AND NOT ANDROID)
    add_executable(tts_windows_demo
        src/main_windows.cpp
    )

    target_link_libraries(tts_windows_demo PRIVATE
        modular_tts
        winmm
        psapi
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(tts_windows_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()

    message(STATUS "Configured target: tts_windows_demo (Windows CLI executable)")
endif()

# ============================================================================
# Target: Android JNI Shared Library (libmodular_tts_jni.so)
# ============================================================================
if(ANDROID OR DEFINED ENV{ANDROID_NDK})
    add_library(modular_tts_jni SHARED
        android/jni_tts_bridge.cpp
    )

    target_link_libraries(modular_tts_jni PRIVATE
        modular_tts
        android
        log
        OpenSLES
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(modular_tts_jni PRIVATE OpenMP::OpenMP_CXX)
    endif()

    set_target_properties(modular_tts_jni PROPERTIES
        OUTPUT_NAME "modular_tts_jni"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/libs/${ANDROID_ABI}"
    )

    message(STATUS "Configured target: modular_tts_jni (Android JNI shared library)")
endif()


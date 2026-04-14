#!/bin/bash

set -e

missing=()
if ! command -v cmake &> /dev/null; then
    missing+=("cmake")
fi

if ! command -v make &> /dev/null; then
    missing+=("make")
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    missing+=("g++")
fi

if [ ${#missing[@]} -gt 0 ]; then
    echo "Error: Missing required build tools: ${missing[*]}"
    echo ""
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Install with: sudo apt-get install cmake build-essential"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Install with: xcode-select --install && brew install cmake"
    else
        echo "Please install cmake and a C++ compiler for your platform."
    fi
    exit 1
fi

echo "Building Cactus library..."

cd "$(dirname "$0")/../cactus"

rm -rf build

mkdir -p build
cd build

cmake_args=(
    -DCMAKE_RULE_MESSAGES=OFF
    -DCMAKE_VERBOSE_MAKEFILE=OFF
)

detect_arm_profile_args() {
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
        return 0
    fi

    local features_line
    features_line="$(grep -m1 '^Features' /proc/cpuinfo 2>/dev/null || true)"
    local features
    features="$(printf '%s' "${features_line#*:}" | tr '\t' ' ' | sed -E 's/[[:space:]]+/ /g')"
    features=" ${features} "

    has_feature() {
        [[ "$features" == *" $1 "* ]]
    }

    if [[ "${features}" == "  " ]]; then
        echo "Warning: unable to read CPU feature flags from /proc/cpuinfo." >&2
        echo "- Falling back to conservative compatibility profile." >&2
        echo "-DCACTUS_ARM_BASELINE=armv8-a+simd+fp16 -DCACTUS_ENABLE_I8MM=OFF -DENABLE_SME2=OFF"
        return 0
    fi

    if ! has_feature asimdhp; then
        echo "Error: ARM64 CPU is missing 'asimdhp' (FP16 vector arithmetic)." >&2
        echo "Cactus local kernels require this feature. Use a newer CPU/device." >&2
        exit 1
    fi

    if ! has_feature asimddp; then
        echo "Detected ARM64 CPU without DOTPROD (Pi 4 class)." >&2
        echo "- Applying compatibility profile: armv8-a+simd+fp16, I8MM OFF, SME2 OFF" >&2
        echo "-DCACTUS_ARM_BASELINE=armv8-a+simd+fp16 -DCACTUS_ENABLE_I8MM=OFF -DENABLE_SME2=OFF"
        return 0
    fi

    if has_feature i8mm; then
        echo "Detected ARM64 CPU with DOTPROD+I8MM (Pi 5/newer class)." >&2
    else
        echo "Detected ARM64 CPU with DOTPROD (Pi 5 class)." >&2
    fi
    echo "- Using default optimized CMake profile." >&2
    return 0
}

if [ -n "${CACTUS_CMAKE_ARGS:-}" ]; then
    read -r -a extra_cmake_args <<< "${CACTUS_CMAKE_ARGS}"
    cmake_args+=("${extra_cmake_args[@]}")
    echo "Using custom CMake args: ${CACTUS_CMAKE_ARGS}"
else
    auto_profile_args="$(detect_arm_profile_args)"
    if [ -n "${auto_profile_args}" ]; then
        read -r -a auto_cmake_args <<< "${auto_profile_args}"
        cmake_args+=("${auto_cmake_args[@]}")
    fi
fi

cmake .. "${cmake_args[@]}" > /dev/null 2>&1
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "Cactus library built successfully!"
echo "Library location: $(pwd)/libcactus.a"

cmake_minimum_required(VERSION 3.26)
project(wemm_native LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # Ampere through Blackwell (CUDA 13 dropped pre-Turing).
  set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90 100 120)
endif()

# LibTorch: point CMAKE_PREFIX_PATH (or Torch_DIR) at an unpacked libtorch
# (cu130 build) or at `python -c 'import torch; print(torch.utils.cmake_prefix_path)'`.
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(nlohmann_json 3.10 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil libswscale)

# ---- Hugging Face tokenizers (Rust) -------------------------------------
find_program(CARGO cargo HINTS $ENV{HOME}/.cargo/bin REQUIRED)
set(TOK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tokenizers-ffi)
set(TOK_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/cargo)
set(TOK_LIB ${TOK_TARGET_DIR}/release/libwemm_tokenizers_ffi.a)
add_custom_command(
  OUTPUT ${TOK_LIB}
  COMMAND ${CARGO} build --release --manifest-path ${TOK_DIR}/Cargo.toml --target-dir ${TOK_TARGET_DIR}
  DEPENDS ${TOK_DIR}/Cargo.toml ${TOK_DIR}/src/lib.rs
  COMMENT "Building tokenizers FFI (cargo)")
add_custom_target(wemm_tokenizers_build DEPENDS ${TOK_LIB})
add_library(wemm_tokenizers STATIC IMPORTED)
set_target_properties(wemm_tokenizers PROPERTIES IMPORTED_LOCATION ${TOK_LIB})
add_dependencies(wemm_tokenizers wemm_tokenizers_build)

# ---- core library ---------------------------------------------------------
add_library(wemm SHARED
  src/safetensors.cpp
  src/config.cpp
  src/tokenizer.cpp
  src/media_image.cpp
  src/media_video.cpp
  src/processor.cpp
  src/pixels.cpp
  src/model.cpp
  src/embedder.cpp
  src/kernels/gdn.cu
  src/kernels/fused.cu
)
target_include_directories(wemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(wemm
  PUBLIC ${TORCH_LIBRARIES}
  PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS} PkgConfig::FFMPEG wemm_tokenizers dl pthread)
target_compile_options(wemm PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-unused-function>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -lineinfo>)
set_target_properties(wemm PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)

add_executable(wemm_embed apps/wemm_embed.cpp)
target_link_libraries(wemm_embed PRIVATE wemm nlohmann_json::nlohmann_json)

add_executable(wemm_bench apps/wemm_bench.cpp)
target_link_libraries(wemm_bench PRIVATE wemm)

add_executable(wemm_test_reference tests/test_reference.cpp)
target_link_libraries(wemm_test_reference PRIVATE wemm nlohmann_json::nlohmann_json)

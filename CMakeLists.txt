cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libalgebra C CXX)
set (CMAKE_BUILD_TYPE Release)
set (CMAKE_VERBOSE_MAKEFILE on) # Debug verbosity

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_C_STANDARD 99)
  endif ()
endmacro(use_cxx11)

use_cxx11()

option(STORM_ENABLE_SIMD_AVX512 "Enable AVX512 optimizations" OFF)
option(STORM_ENABLE_SIMD_AVX2 "Enable AVX2 optimizations" OFF)
option(STORM_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" OFF)
option(STORM_DISABLE_NATIVE "Force disable native compilaton" OFF)

if(STORM_ENABLE_SIMD_AVX512)
	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx512)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX512)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX512)
	endif()
	message(STATUS "STORM: AVX2 instruction set")
elseif(STORM_ENABLE_SIMD_AVX2)
	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-mavx2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxAVX2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
		add_compile_options(/arch:AVX2)
	endif()
	message(STATUS "STORM: AVX2 instruction set")
elseif(STORM_ENABLE_SIMD_SSE4_2)
	if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
		add_compile_options(-msse4.2)
	elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		add_compile_options(/QxSSE4.2)
	elseif((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
		# VC doesn't support SSE4.2
    # See https://docs.microsoft.com/en-us/cpp/build/reference/arch-x64
    add_compile_options(/arch:AVX)
	endif()
	message(STATUS "STORM: SSE4.2 instruction set")
endif()

message(STATUS "CXX_FLAGS=${CMAKE_CXX_FLAGS}")
message(STATUS "C_FLAGS=${CMAKE_C_FLAGS}")

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    SET(CMAKE_FIND_LIBRARY_PREFIXES "lib")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".so" ".a" ".dylib")
    message(STATUS "Searching for .so and .a")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    SET(CMAKE_FIND_LIBRARY_PREFIXES "")
    SET(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
    message(STATUS "Searching for .lib and .dll")
endif()

if(NOT MSVC)
if(NOT STORM_DISABLE_NATIVE)
add_compile_options(-march=native)
endif()
endif()

if (MSVC)
add_compile_options(/O2)
endif()

find_path(LM_ROARING_INCLUDE_DIR NAMES REQUIRED roaring/roaring.h)
find_library(LM_ROARING_LIBRARY NAMES REQUIRED libroaring roaring)

message("LM_ROARING include dir = ${LM_ROARING_INCLUDE_DIR}")
message("LM_ROARING lib = ${LM_ROARING_LIBRARY}")

add_executable(benchmark storm.c benchmark.cpp)
target_link_libraries(benchmark PUBLIC ${LM_ROARING_LIBRARY})
target_include_directories(benchmark PUBLIC ${LM_ROARING_INCLUDE_DIR} "${PROJECT_SOURCE_DIR}")
message("target_include_directories = ${LM_ROARING_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}")
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(libalgebra C CXX)
set (CMAKE_BUILD_TYPE Release)

macro(use_cxx11)
  if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
  else ()
    set (CMAKE_CXX_STANDARD 11)
  endif ()
endmacro(use_cxx11)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

set (CMAKE_C_STANDARD 99)

#set (CMAKE_CXX_FLAGS "-std=c++11")
#set (CMAKE_C_FLAGS "-std=c99")
include_directories(.)

find_path(LM_ROARING_INCLUDE_DIR NAMES REQUIRED roaring/roaring.h)
find_library(LM_ROARING_LIBRARY NAMES REQUIRED libroaring roaring)

message("LM_ROARING include dir = ${LM_ROARING_INCLUDE_DIR}")
message("LM_ROARING lib = ${LM_ROARING_LIBRARY}")

add_executable(benchmark benchmark.cpp storm.c experimental.c)
target_link_libraries(benchmark PUBLIC ${LM_ROARING_LIBRARY})
target_include_directories(benchmark PUBLIC ${LM_ROARING_INCLUDE_DIR})
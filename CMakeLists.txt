cmake_minimum_required(VERSION 3.20)
project(ntc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)

add_executable(ntc main.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ntc PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
  target_compile_options(ntc PRIVATE /W3 /O2 /fp:fast /arch:AVX2)
else()
  target_compile_options(ntc PRIVATE -Wall -O3 -ffast-math -march=native)
endif()

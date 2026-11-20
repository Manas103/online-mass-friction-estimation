cmake_minimum_required(VERSION 3.15)
project(property_model CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

option(ENABLE_ASAN "Build with AddressSanitizer + UBSan" OFF)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Eigen3 REQUIRED)

add_executable(property_model_main property_model_main.cpp)
target_link_libraries(property_model_main PRIVATE Eigen3::Eigen)

enable_testing()
add_executable(check_property_model tests/test_property_model.cpp)
target_link_libraries(check_property_model PRIVATE Eigen3::Eigen)
add_test(NAME check_property_model COMMAND check_property_model)

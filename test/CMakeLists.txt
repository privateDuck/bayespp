cmake_minimum_required(VERSION 3.15)
project(Test LANGUAGES CXX)

# 1. Set C++ Standard (LBFGS++ usually requires C++11 or higher)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3. Define the Executable
add_executable(test_bench main.cpp
        ../include/bayespp/vgrad.hpp)

# 4. Include your local library headers (root/include)
target_include_directories(test_bench PRIVATE "../include")
target_include_directories(test_bench PRIVATE "../deps")

# 5. Link Dependencies
# This handles both include paths and necessary compile definitions
#[[
target_link_libraries(test_bench
        PRIVATE
        Eigen3::Eigen
        LBFGSpp::LBFGSpp
)]]

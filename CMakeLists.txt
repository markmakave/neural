cmake_minimum_required(VERSION 3.0)

project(neural)

set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++17")

add_executable(neural ./src/main.cpp)
target_include_directories(neural PUBLIC include)

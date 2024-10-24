cmake_minimum_required(VERSION 3.29)
project(CollisionBasedGasSimulator C)

set(CMAKE_C_FLAGS_DEBUG "-Og -g3")
set(CMAKE_C_STANDARD 23)

include(cmake/CPM.cmake)
CPMAddPackage("gh:raysan5/raylib#5.0")

add_executable(CollisionBasedGasSimulator main.c simulator.c)
target_link_libraries(CollisionBasedGasSimulator OpenCL m raylib)

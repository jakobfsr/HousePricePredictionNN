cmake_minimum_required(VERSION 3.10)
project(NNExample C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")

find_package(OpenMP REQUIRED)
if (OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

add_executable(nn_example main.c nn.c)
target_link_libraries(nn_example PRIVATE OpenMP::OpenMP_C m)

# Optionale Ausgabe zur Kontrolle:
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")

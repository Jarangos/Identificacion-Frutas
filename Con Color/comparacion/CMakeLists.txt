cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(comparacion)

set(PCL_DIR "C:/Program Files/PCL 1.9.1/cmake/PCLConfig.cmake")

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (comparacion comparacion.cpp)
target_link_libraries (comparacion ${PCL_LIBRARIES})

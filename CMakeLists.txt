cmake_minimum_required(VERSION 3.16)
project(tp_oranges)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED)

add_executable(tp_oranges origem.cpp vc.cpp)
target_include_directories(tp_oranges PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tp_oranges PRIVATE ${OpenCV_LIBS})

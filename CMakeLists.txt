cmake_minimum_required(VERSION 3.8)
project(PageRank)  # This should match your actual project's main file without the .cpp

set(CMAKE_CXX_STANDARD 17)  # Use C++17 standard

find_package(OpenMP REQUIRED)  # Find OpenMP

if(OPENMP_FOUND)
    message(STATUS "OpenMP is found on this system.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Adjust the name and path as per your project files
add_executable(PageRank PageRank.cpp)  # Ensure this is the correct C++ file you want to compile
if(OPENMP_FOUND)
    target_link_libraries(PageRank PUBLIC OpenMP::OpenMP_CXX)
endif()

cmake_minimum_required(VERSION 3.2)
project(mpi_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 --std=c++11")
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include_directories(/home/robertse/Code/phiprof-master/include)
link_directories(/home/robertse/Code/phiprof-master/lib)

set(SOURCE_FILES main.cpp)

add_executable(mpi_io_test ${SOURCE_FILES})

target_link_libraries(mpi_io_test ${MPI_LIBRARIES})
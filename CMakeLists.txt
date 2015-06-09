cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project (PPFMap)

set (BUILD_TYPE Release)

find_package (CUDA REQUIRED)
set(CUDA_NVCC_FLAGS " -arch=sm_10 ")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# Point Cloud Library (PCL)
find_package (PCL 1.7 REQUIRED)
include_directories (${PCL_INCLUDE_DIRS})


##############################################################################
#   Check for C++11 compiler support                                         #
##############################################################################

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR CMAKE_COMPILER_IS_GNUCXX)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(--std=c++11 SUPPORTS_STD_CXX11)
    check_cxx_compiler_flag(--std=c++0x SUPPORTS_STD_CXX01)
    if(SUPPORTS_STD_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --std=c++11")
    elseif(SUPPORTS_STD_CXX01)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++0x")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --std=c++0x")
    else()
        message(ERROR "Compiler does not support --std=c++11 or --std=c++0x.")
    endif()
endif()


set (DEMO_FILES src/demo.cpp) 

set (CPP_FILES 
    src/Map.cpp)

set (HPP_FILES 
    include/PPFMap/Map.h
    include/PPFMap/PPFMatch.h)

set (CU_FILES
    src/Map.cu
    src/ppf_cuda_calls.cu)


include_directories(${CMAKE_SOURCE_DIR}/include)
cuda_include_directories (${CMAKE_SOURCE_DIR}/include)

cuda_add_library(ppfmap SHARED ${CPP_FILES} ${HPP_FILES} ${CU_FILES})

cuda_add_executable(demo ${DEMO_FILES})
target_link_libraries (demo ppfmap ${PCL_LIBRARIES})


##############################################################################
#   Doxygen documentation                                                    #
##############################################################################

# adds custom command 'make doc' to generate the documentation.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

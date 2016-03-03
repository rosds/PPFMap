cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)

project (PPFMap CXX)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")


##############################################################################
#   Check for C++11 compiler support                                         #
##############################################################################

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-std=c++11 SUPPORTS_STD_CXX11)
check_cxx_compiler_flag(-std=c++0x SUPPORTS_STD_CXX01)
if(SUPPORTS_STD_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
elseif(SUPPORTS_STD_CXX01)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++0x")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++0x")
else()
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()



##############################################################################
#   Dependencies                                                             #
##############################################################################

set (CUDA_ARCH "sm_35" CACHE STRING "Cuda Architechture") 

find_package (CUDA REQUIRED)

set (DEMO_FILES src/demo.cpp) 

set (CPP_FILES "")

set (HPP_FILES 
    include/PPFMap/cuda_map.h
    include/PPFMap/CudaPPFMatch.h)


set (CU_SRC_FILES
    src/Map.cu
    src/ppf_cuda_calls.cu)
    

# CUDA
FIND_PACKAGE(CUDA REQUIRED)
message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY})
list(APPEND CUDA_NVCC_FLAGS "-arch=${CUDA_ARCH};-O2;-Xcompiler;-fPIC;-DVERBOSE")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
add_definitions(-DGPU)
cuda_include_directories (${CMAKE_SOURCE_DIR}/include)


set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

include(CudaComputeTargetFlags)
APPEND_TARGET_ARCH_FLAGS()


include_directories(${CMAKE_SOURCE_DIR}/include)

CUDA_COMPILE(cuda_objs ${CU_SRC_FILES})


cuda_add_library(ppfmap SHARED ${CPP_FILES} ${HPP_FILES} ${cuda_objs})


# Point Cloud Library (PCL)
find_package (PCL 1.7 REQUIRED)
include_directories (${PCL_INCLUDE_DIRS})

cuda_add_executable(demo ${DEMO_FILES})
target_link_libraries (demo ppfmap ${PCL_LIBRARIES} ${LIBRARIES})


##############################################################################
#   Installation rules                                                       #
##############################################################################

install (TARGETS ppfmap DESTINATION lib)
install (DIRECTORY include/PPFMap DESTINATION include)

##############################################################################
#   Doxygen documentation                                                    #
##############################################################################

# adds custom command 'make doc' to generate the documentation.
find_package(Doxygen)
if(DOXYGEN_FOUND)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
        ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

# BSD 2-Clause License
#
# Copyright (c) 2022, Eijiro SHIBUSAWA
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

project(VFSPython CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# CUDA Toolkit
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
	message(STATUS "CUDA Toolkit enabled")
endif()

# Python / Anaconda
find_package(Python COMPONENTS Interpreter Development)
set(pybind11_DIR "${Python_SITELIB}/pybind11/share/cmake/pybind11")
find_package(pybind11 CONFIG REQUIRED)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${Python_SITELIB}" CACHE PATH "..." FORCE)
endif()
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

# OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_INCLUDE_DIRS)
	message(STATUS "OpenCV enabled")
endif()

add_library(vfs SHARED
	../stereotgv/stereotgv.cu
	../stereotgv/thresholdingL1.cu
	../stereotgv/updateDualVariables.cu
	../stereotgv/updatePrimalVariables.cu
	../stereotgv/upscale.cu
	../stereotgv/warpImage.cu
	../stereotgv/calcTensor.cu
	../stereotgv/computeDerivatives.cu
	../stereotgv/conversion.cu
	../stereotgv/copyClone.cu
	../stereotgv/downscale.cu
	../stereotgv/flowToHsv.cu
	../stereotgv/imageConverter.cu
	../stereotgv/limitRange.cu
	../stereotgv/medianFilter.cu
	../stereotgv/preconditioning.cu
	../stereotgv/scalar.cu
	)
target_include_directories(vfs PRIVATE
	../stereotgv/
	${CUDAToolkit_INCLUDE_DIR}
	${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc/
	${OpenCV_INCLUDE_DIRS}
	)
target_link_libraries(vfs ${CUDA_CUDART} ${OpenCV_LIBRARIES})
set_target_properties(vfs PROPERTIES CUDA_ARCHITECTURES "61")
install(TARGETS vfs DESTINATION .)

add_executable(vfs_main
	../test_vfs/main.cpp
	)
target_include_directories(vfs_main PRIVATE
	../
	${CUDAToolkit_INCLUDE_DIR}
	${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc/
	${OpenCV_INCLUDE_DIRS}
	)
target_link_libraries(vfs_main vfs ${OpenCV_LIBRARIES})
install(TARGETS vfs_main DESTINATION .)

pybind11_add_module(pyvfs pyvfs.cpp)
target_link_libraries(pyvfs PRIVATE vfs ${OpenCV_LIBRARIES})
target_include_directories(pyvfs PRIVATE
	../
	${CUDAToolkit_INCLUDE_DIR}
	${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc/
	${OpenCV_INCLUDE_DIRS}
	)
set_target_properties(pyvfs PROPERTIES PREFIX "")
install(TARGETS pyvfs DESTINATION .)

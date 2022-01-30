cmake_minimum_required(VERSION 3.18)

project(VFSPython CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE 1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
	CACHE PATH "Output directory for libraries.")

# CUDA Toolkit
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
	message(STATUS "CUDA Toolkit enabled")
endif()

# Python / Anaconda
set(PYTHON_VERSION 3.6)
string(REPLACE "." "" PYTHON_VERSION_REP ${PYTHON_VERSION})
set(ANACONDA_ROOT "/path/to/anaconda3/envs/your-env")

set(BOOST_PYTHON_LIBS python${PYTHON_VERSION}m boost_python${PYTHON_VERSION_REP} boost_numpy${PYTHON_VERSION_REP})
foreach(LIB IN LISTS BOOST_PYTHON_LIBS)
	set(FOUND_LIB "FOUND_LIB-NOTFOUND")
	find_library(FOUND_LIB ${LIB} PATHS ${ANACONDA_ROOT}/lib)
	LIST(APPEND BOOST_PYTHON_LIBRARIES ${FOUND_LIB})
endforeach()
set(BOOST_PYTHON_INCLUDES ${ANACONDA_ROOT}/include ${ANACONDA_ROOT}/include/python${PYTHON_VERSION}m)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(CMAKE_INSTALL_PREFIX "${ANACONDA_ROOT}/lib/python3.6/site-packages" CACHE PATH "..." FORCE)
endif()
set(CMAKE_INSTALL_RPATH "${ANACONDA_ROOT}/lib" "${CMAKE_INSTALL_PREFIX}")

# OpenCV
set(OpenCV_DIR "${ANACONDA_ROOT}/lib/cmake/opencv4/")
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

add_library(pyvfs SHARED
	pyvfs.cpp
	)
target_include_directories(pyvfs PRIVATE
	../
	${BOOST_PYTHON_INCLUDES}
	${CUDAToolkit_INCLUDE_DIR}
	${CUDAToolkit_LIBRARY_ROOT}/samples/common/inc/
	${OpenCV_INCLUDE_DIRS}
	)
target_link_libraries(pyvfs vfs ${BOOST_PYTHON_LIBRARIES} ${OpenCV_LIBRARIES})
set_target_properties(pyvfs PROPERTIES PREFIX "")
install(TARGETS pyvfs DESTINATION .)

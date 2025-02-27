cmake_minimum_required(VERSION 3.15...3.29)
project(main_build LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# setting torch and eigen install locations depending on system 
message("System / OS: ${CMAKE_SYSTEM_NAME}")
IF(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	# TODO change these
	set(TORCH_INSTALL_PREFIX "/opt/venv/lib/python3.12/site-packages/torch")
	set(EIGEN3_INCLUDE_DIR "${HOME}/opt/Eigen")
	set(Eigen3_DIR "${HOME}/opt/Eigen/share/eigen3/cmake")
	set (pybind11_DIR "/opt/venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

ELSE()
	set(TORCH_INSTALL_PREFIX "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/torch")
	set(EIGEN3_INCLUDE_DIR "${HOME}/opt/Eigen")
	set(Eigen3_DIR "${HOME}/opt/Eigen/share/eigen3/cmake")
	set (pybind11_DIR "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
ENDIF()

# set(TORCH_INSTALL_PREFIX "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/torch")
# set(EIGEN3_INCLUDE_DIR "${HOME}/opt/Eigen")
# set(Eigen3_DIR "${HOME}/opt/Eigen/share/eigen3/cmake")

list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})

# TEMP Deact
# list(APPEND CMAKE_PREFIX_PATH "/usr/include/python3.12")
# set(PYTHON_INCLUDE_DIRS "/usr/include/python3.12")


set(PYBIND11_FINDPYTHON ON)
find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

message("PYthon include dirs: ${PYTHON_INCLUDE_DIRS}")
message("TORCH include dirs: ${TORCH_INCLUDE_DIRS}")
message("Pybind include dirs: ${pybind11_DIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python "${TORCH_INSTALL_PREFIX}/lib")
message(TORCH_PYTHON_LIBRARY="${TORCH_PYTHON_LIBRARY}")

# set (pybind11_DIR "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
# find_package(pybind11 CONFIG REQUIRED)


set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_VISIBILITY_PRESET hidden) 

pybind11_add_module(main_build main.cpp test.cpp)
# add_executable(main_build ../main.cpp ../test.cpp)
target_link_libraries(main_build PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Eigen3::Eigen)
set_property(TARGET main_build PROPERTY CXX_STANDARD 17)


cmake_minimum_required(VERSION 3.15...3.29)
project(np_interop LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TORCH_INSTALL_PREFIX "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/torch")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})
find_package(Torch REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_library(TORCH_PYTHON_LIBRARY torch_python "${TORCH_INSTALL_PREFIX}/lib")
message(TORCH_PYTHON_LIBRARY="${TORCH_PYTHON_LIBRARY}")

set (pybind11_DIR "/opt/homebrew/Caskroom/miniconda/base/envs/ML-practice/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


set (CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_VISIBILITY_PRESET hidden) 

pybind11_add_module(np_interop np_interop.cpp)
target_link_libraries(np_interop PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
install(TARGETS np_interop DESTINATION .)

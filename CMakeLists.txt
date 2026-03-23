cmake_minimum_required(VERSION 3.15)
project(ShapeAndContourAnalyzer VERSION 1.0 LANGUAGES CXX)
cmake_policy(SET CMP0148 NEW)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==============================================================================
# 1. DEPENDENCIES
# ==============================================================================

set(PYBIND11_FINDPYTHON ON)

# Path to your OpenCV and Pybind11 installations (update these if needed)
set(OpenCV_DIR "D:/opencv/build")
set(pybind11_DIR "D:/Python11/Lib/site-packages/pybind11/share/cmake/pybind11")

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ==============================================================================
# 2. CREATE THE PYTHON MODULE
# ==============================================================================

pybind11_add_module(cv_backend
    Backend/Main_bindings.cpp
    Backend/Canny.cpp
    Backend/Snake.cpp
    Backend/Hough.cpp
)

target_include_directories(cv_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Backend
    ${CMAKE_CURRENT_SOURCE_DIR}/Helpers
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(cv_backend PRIVATE ${OpenCV_LIBS})

# ==============================================================================
# 3. OUTPUT DIRECTORY (to frontend/, matching your previous project)
# ==============================================================================

set_target_properties(cv_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/frontend"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/frontend"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/frontend"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/frontend"
)

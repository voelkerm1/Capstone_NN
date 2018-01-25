# Specify the minimum version for CMake

cmake_minimum_required(VERSION 2.8)

# Project's name

project(NeuralNetwork)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

find_package( OpenCV REQUIRED )


# The following folder will be included
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/OpenNN-master/opennn)
include_directories(${PROJECT_SOURCE_DIR}/OpenNN-master/eigen)
include_directories(${PROJECT_SOURCE_DIR}/Include)


# The following folders are searched for libraries (since OpenNN is non-standard)
find_library(OPENNN opennn HINTS ${PROJECT_SOURCE_DIR}/OpenNN-master/Build/opennn)

# Define executable and link libraries, source code to it
add_executable(NeuralNetwork ${PROJECT_SOURCE_DIR}/NNmain.cpp)
add_executable(ImageReader ${PROJECT_SOURCE_DIR}/imageReaderMain.cpp)
target_link_libraries( NeuralNetwork ${OPENNN})
target_link_libraries( ImageReader ${OpenCV_LIBS})
target_sources( NeuralNetwork
        PUBLIC
            "${PROJECT_SOURCE_DIR}/Link/ConfigFileParser.cpp"
        )
target_sources( ImageReader
        PUBLIC
            "${PROJECT_SOURCE_DIR}/Link/ImageReader.cpp"
        )



cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

# Find Packages
find_package(OpenCV REQUIRED)

# Executable
add_library(rbv src/Camera.cpp src/StereoPair.cpp src/Threshold.cpp)

# Linked Libraries
target_link_libraries(rbv ${OpenCV_LIBS})

# Directories to include
target_include_directories(rbv PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
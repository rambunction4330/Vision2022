
# Find Packages
find_package(OpenCV REQUIRED)

# Executabl 
add_executable(hubDetection hub.cpp)
add_executable(ballDetection hub.cpp)

# Link Libraries
target_link_libraries(hubDetection ${OpenCV_LIBS} rbv)
target_link_libraries(ballDetection ${OpenCV_LIBS} rbv)

# Directories to include
target_include_directories(hubDetection PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(ballDetection PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
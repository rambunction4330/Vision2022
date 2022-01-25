
# Find Packages
find_package(OpenCV REQUIRED)

# Executabl 
add_executable(hsvTunning hsvTunning.cpp)
add_executable(cameraSettings cameraSettings.cpp)
add_executable(camerCalibration cameraCalibration.cpp)
add_executable(stereoCalibration cameraCalibration.cpp)

# Link Libraries
target_link_libraries(hsvTunning ${OpenCV_LIBS} rbv)
target_link_libraries(cameraSettings ${OpenCV_LIBS} rbv)
target_link_libraries(camerCalibration ${OpenCV_LIBS} rbv)
target_link_libraries(stereoCalibration ${OpenCV_LIBS} rbv)

# Directories to include
target_include_directories(hsvTunning PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(cameraSettings PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(camerCalibration PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(stereoCalibration PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
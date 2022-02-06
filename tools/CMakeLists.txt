
# Find Packages
find_package(OpenCV REQUIRED)

# Executabl 
add_executable(hsvTunning hsvTunning.cpp)
add_executable(calibrationViewer calibrationViewer.cpp)
add_executable(calibrateCamera calibrateCamera.cpp)
add_executable(calibrateStereo calibrateStereo.cpp)

# Link Libraries
target_link_libraries(hsvTunning ${OpenCV_LIBS} rbv)
target_link_libraries(calibrationViewer ${OpenCV_LIBS} rbv)
target_link_libraries(calibrateCamera ${OpenCV_LIBS} rbv)
target_link_libraries(calibrateStereo ${OpenCV_LIBS} rbv)

# Directories to include
target_include_directories(hsvTunning PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrationViewer PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrateCamera PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrateStereo PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
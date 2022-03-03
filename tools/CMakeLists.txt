
# Find Packages
find_package(OpenCV REQUIRED)

# Executabl 
add_executable(hsvTunning hsvTunning.cpp)
add_executable(calibrationViewer calibrationViewer.cpp)
add_executable(calibrateCamera calibrateCamera.cpp)
add_executable(calibrateStereo calibrateStereo.cpp)
add_executable(rectDetection rectDetection.cpp)
add_executable(circleDetection circleDetection.cpp)
add_executable(networkTest networkTest.cpp)

# Link Libraries
target_link_libraries(hsvTunning ${OpenCV_LIBS} rbv)
target_link_libraries(calibrationViewer ${OpenCV_LIBS} rbv)
target_link_libraries(calibrateCamera ${OpenCV_LIBS} rbv)
target_link_libraries(calibrateStereo ${OpenCV_LIBS} rbv)
target_link_libraries(rectDetection ${OpenCV_LIBS} rbv)
target_link_libraries(circleDetection ${OpenCV_LIBS} rbv)
target_link_libraries(networkTest ${OpenCV_LIBS} rbv)

# Directories to include
target_include_directories(hsvTunning PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrationViewer PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrateCamera PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(calibrateStereo PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(rectDetection PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(circleDetection PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
target_include_directories(networkTest PUBLIC ${PROJECT_SOURCE_DIR}/lib/include)
cmake_minimum_required(VERSION 3.20.0)

project("test-grpc" VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")
set(CMAKE_BUILD_TYPE Debug)

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GRPCPP REQUIRED grpc++)
pkg_search_module(OPENCV REQUIRED opencv4)

include_directories(
    ${OPENCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_SOURCE_DIR}/PredictionClient
    ${CMAKE_SOURCE_DIR}/include)

add_subdirectory(proto)
add_subdirectory(PredictionClient)

# Client
add_executable(grpc-client client.cc)
target_link_libraries(grpc-client PRIVATE
    ${OPENCV_LIBRARIES}
    ${Protobuf_LIBRARIES} 
    ${GRPCPP_LIBRARIES}
    proto_tensorflow
    prediction_client
)
install(TARGETS grpc-client
    RUNTIME DESTINATION bin
)

install(FILES cat.jpg DESTINATION bin)
install(FILES 11866.png DESTINATION bin)
install(FILES 358.png DESTINATION bin)

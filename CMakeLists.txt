cmake_minimum_required(VERSION 3.20.0)

project("test-grpc" VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GRPCPP REQUIRED grpc++)

include_directories(
    ${CMAKE_SOURCE_DIR}/proto
    ${CMAKE_SOURCE_DIR}/PredictionClient)

add_subdirectory(proto)
add_subdirectory(PredictionClient)

# Client
add_executable(grpc-client client.cc)
target_link_libraries(grpc-client PRIVATE
    ${Protobuf_LIBRARIES} 
    ${GRPCPP_LIBRARIES}
    proto_tensorflow
    prediction_client
)
install(TARGETS grpc-client
    RUNTIME DESTINATION bin
)

# Server
#add_executable(grpc-server server.cc)
#target_link_libraries(grpc-server PRIVATE
    #${Protobuf_LIBRARIES} 
    #${GRPC_LIBRARIES}
    #proto_route_guide
#)
#install(TARGETS grpc-server
    #RUNTIME DESTINATION bin
#)

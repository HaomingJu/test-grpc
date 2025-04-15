project(proto_tensorflow)

file(GLOB_RECURSE PROTO_SRC_FILES "*.cc")

add_library(${PROJECT_NAME} SHARED ${PROTO_SRC_FILES})

target_link_libraries(${PROJECT_NAME}
    ${Protobuf_LIBRARIES})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)

project(proto_route_guide)

file(GLOB_RECURSE PROTO_SRC_FILES "*.cc")

add_library(${PROJECT_NAME} SHARED ${PROTO_SRC_FILES})

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
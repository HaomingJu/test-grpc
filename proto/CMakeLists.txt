project(proto_route_guide)

add_library(${PROJECT_NAME} SHARED route_guide.pb.cc)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
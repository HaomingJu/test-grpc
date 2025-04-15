project(prediction_client)

add_library(${PROJECT_NAME} SHARED PredictionClient.cc)
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OPENCV_LIBRARIES}
    ${Protobuf_LIBRARIES} 
    ${GRPCPP_LIBRARIES}
    proto_tensorflow
)
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

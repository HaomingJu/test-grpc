#pragma once

#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include <array>
#include <cstddef>
#include <grpc++/grpc++.h>

#include <grpcpp/channel.h>
#include <memory>
#include <tensorflow_serving/apis/prediction_service.grpc.pb.h>
#include <tensorflow_serving/apis/prediction_service.pb.h>

#include <opencv2/opencv.hpp>

using tensorflow::serving::GetModelMetadataRequest;
using tensorflow::serving::GetModelMetadataResponse;
using tensorflow::serving::PredictionService;
using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;

class PredictionClient {
public:
  int Init(const std::string &target,
           const ::tensorflow::TensorProto &&tensor_input,
           const ::tensorflow::TensorProto &&tensor_output,
           const std::string &model_name = "dogcat");

  void Final();

public:
  int GetModelMetadata(GetModelMetadataResponse *response);
  int Predict(cv::Mat &&image);
  int Post(); // 后处理

private:
  std::unique_ptr<PredictionService::Stub> stub_{};
  std::shared_ptr<::grpc::Channel> channel_;
  std::string model_name_;

private:
  void letterbox_preprocess(const cv::Mat &src, cv::Mat &dest, float *scale,
                            int *padding_top, int *padding_left,
                            int target_size = 640);

  int drawResult(const tensorflow::TensorProto &result, float scale,
                 int padding_top, int padding_left, cv::Mat &origin_image,
                 const std::string &output_image = {});

private:
  ::tensorflow::TensorProto tensor_input_;
  ::tensorflow::TensorProto tensor_output_;
};

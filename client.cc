#include "opencv2/imgcodecs.hpp"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include <PredictionClient.h>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  std::cout << "Hello from client!" << std::endl;
  PredictionClient client;
  int ret = client.Init("123.57.18.145:8500");
  if (ret) {
    std::cerr << "client init failed" << std::endl;
    return 0;
  }

  // 测试模型可用性
  ::tensorflow::serving::GetModelMetadataResponse response_meta;
  client.GetModelMetadata(&response_meta);

  // 读取测试图片
  auto image = cv::imread(argv[1]);
  if (image.empty()) {
    std::cerr << "image is empty" << std::endl;
    return 0;
  }

  // 推理图片
  ::tensorflow::serving::PredictResponse response_predict;
  auto status = client.Predict(std::move(image), &response_predict);

  if (status) {
    std::cout << "Predict end, return true" << std::endl;
    const auto &result = response_predict.outputs().at("output0");
    for (int i = 0; i < result.tensor_shape().dim_size(); ++i) {
      std::cout << "name: " << result.tensor_shape().dim(i).name()
                << " | sz: " << result.tensor_shape().dim(i).size()
                << std::endl;
    }

    std::cout << "float_val_size: " << result.float_val_size() << std::endl;
    // TODO: 根据阈值进行筛选

    // TODO: NMS算法

    // TODO: 绘制图像

  } else {
    std::cout << "Predict end, return false" << std::endl;
  }

  return 0;
}

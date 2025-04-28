#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include <PredictionClient.h>
#include <argparse.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  std::cout << "Hello from client!" << std::endl;
  // 解析参数
  ::argparse::ArgumentParser program("tensorflow");
  program.add_argument("-i", "--input").required().help("input image path");
  program.add_argument("-o", "--output").required().help("output image path");
  program.add_argument("-m", "--model-name")
      .required()
      .help("yolo11 model name");
  program.add_argument("-t", "--target")
      .required()
      .help("model server: ip:port");

  // 1x640x640x3
  program.add_argument("--tensor-input-dim")
      .nargs(4)
      .default_value(std::vector<int>{1, 640, 640, 3})
      .scan<'i', int>();

  // 1x15x8400
  program.add_argument("--tensor-output-dim")
      .nargs(3)
      .default_value(std::vector<int>{1, 15, 840})
      .scan<'i', int>();

  ::tensorflow::TensorProto tensor_input, tensor_output;
  std::string target, input_image, output_image, model_name;
  std::vector<int> tensor_input_dim, tensor_output_dim;

  try {
    program.parse_args(argc, argv);

    target = program.get("target");
    input_image = program.get("input");
    output_image = program.get("output");
    model_name = program.get("model-name");
    tensor_input_dim = program.get<std::vector<int>>("--tensor-input-dim");
    tensor_output_dim = program.get<std::vector<int>>("--tensor-output-dim");

    std::cout << "model target: " << target << std::endl;
    std::cout << "input image: " << input_image << std::endl;
    std::cout << "output image: " << output_image << std::endl;
    std::cout << "model name: " << model_name << std::endl;
    std::cout << "tensor_input_dim: " << tensor_input_dim[0] << "x"
              << tensor_input_dim[1] << "x" << tensor_input_dim[2] << "x"
              << tensor_input_dim[3] << std::endl;
    std::cout << "tensor_output_dim: " << tensor_output_dim[0] << "x"
              << tensor_output_dim[1] << "x" << tensor_output_dim[2]
              << std::endl;

  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  tensor_input.set_dtype(tensorflow::DataType::DT_FLOAT);
  tensor_input.mutable_tensor_shape()->add_dim()->set_size(tensor_input_dim[0]);
  tensor_input.mutable_tensor_shape()->add_dim()->set_size(tensor_input_dim[1]);
  tensor_input.mutable_tensor_shape()->add_dim()->set_size(tensor_input_dim[2]);
  tensor_input.mutable_tensor_shape()->add_dim()->set_size(tensor_input_dim[3]);

  tensor_output.set_dtype(tensorflow::DataType::DT_FLOAT);
  tensor_output.mutable_tensor_shape()->add_dim()->set_size(
      tensor_output_dim[0]);
  tensor_output.mutable_tensor_shape()->add_dim()->set_size(
      tensor_output_dim[1]);
  tensor_output.mutable_tensor_shape()->add_dim()->set_size(
      tensor_output_dim[2]);

  PredictionClient client;
  int ret = client.Init(target, std::move(tensor_input),
                        std::move(tensor_output), model_name);
  if (ret) {
    std::cerr << "client init failed" << std::endl;
    return 0;
  }

  // 测试模型可用性
  ::tensorflow::serving::GetModelMetadataResponse response_meta;
  ret = client.GetModelMetadata(&response_meta);
  if (ret) {
    std::cerr << "get model meta info failed" << std::endl;
    return 0;
  }

  // 读取测试图片
  auto image = cv::imread(input_image);
  if (image.empty()) {
    std::cerr << "image is empty" << std::endl;
    return 0;
  }

  // 推理图片
  ret = client.Predict(std::move(image), 0.5, output_image);
  if (ret) {
    std::cout << "Predict end, return -1" << std::endl;
  } else {
    std::cout << "Predict end, return 0" << std::endl;
  }
  return 0;
}

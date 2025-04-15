#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include <PredictionClient.h>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << "Hello from client!" << std::endl;
  PredictionClient client;
  int ret = client.Init("123.57.18.145:8500");
  if (ret) {
    std::cerr << "client init failed" << std::endl;
    return 0;
  }

  ::tensorflow::serving::GetModelMetadataResponse response;
  client.GetModelMetadata(&response);

  return 0;
}

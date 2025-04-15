#include "PredictionClient.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/support/status.h>

int PredictionClient::Init(const std::string &target,
                           const std::string &model_name /* = "dogcat" */) {

  // Step.0: Assign to value
  model_name_ = model_name;

  // Step.1: Create channel
  channel_ = ::grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
  if (!channel_) {
    std::cerr << "Channel create failed" << std::endl;
    return -1;
  }

  // Step.2: Create stub
  stub_ = ::tensorflow::serving::PredictionService::NewStub(channel_);

  return 0;
}

bool PredictionClient::GetModelMetadata(GetModelMetadataResponse *response) {

  ::grpc::ClientContext context;
  ::tensorflow::serving::GetModelMetadataRequest request;
  request.mutable_model_spec()->set_name(model_name_);
  request.mutable_metadata_field()->Add("signature_def");

  auto status = stub_->GetModelMetadata(&context, request, response);
  if (!status.ok()) {
    std::cerr << "GetModelMetadata failed" << std::endl;
    return false;
  }

  std::cout << "model.name: " << response->model_spec().name() << std::endl
            << "model.version: " << response->model_spec().version().value()
            << std::endl
            << "model.signature_name: "
            << response->model_spec().signature_name() << std::endl;
  return true;
}

void PredictionClient::Final() {}

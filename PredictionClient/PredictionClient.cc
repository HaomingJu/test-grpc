#include "PredictionClient.h"
#include "opencv2/core/types.hpp"
#include "opencv2/opencv.hpp"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow_serving/apis/get_model_metadata.pb.h"
#include "tensorflow_serving/apis/predict.pb.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"
#include <algorithm>
#include <cassert>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/support/status.h>
#include <iterator>
#include <numeric>
#include <vector>

int PredictionClient::Init(const std::string &target,
                           const ::tensorflow::TensorProto &&tensor_input,
                           const ::tensorflow::TensorProto &&tensor_output,
                           const std::string &model_name /* = "dogcat" */) {

  assert(tensor_input.tensor_shape().dim_size() == 4);
  assert(tensor_output.tensor_shape().dim_size() == 3);

  // Step.0: Assign to value
  model_name_ = model_name;
  tensor_input_.CopyFrom(tensor_input);
  tensor_output_.CopyFrom(tensor_output);

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

int PredictionClient::GetModelMetadata(GetModelMetadataResponse *response) {

  // Step.1: 构建必要上下文
  ::grpc::ClientContext context;
  ::tensorflow::serving::GetModelMetadataRequest request;
  request.mutable_model_spec()->set_name(model_name_);
  request.mutable_metadata_field()->Add("signature_def");

  // Step.2: 调用stub函数
  auto status = stub_->GetModelMetadata(&context, request, response);
  if (!status.ok()) {
    std::cerr << "GetModelMetadata failed" << std::endl;
    return -1;
  }

  // Step.3: 输出模型信息
  std::cout << "model.name: " << response->model_spec().name() << std::endl
            << "model.version: " << response->model_spec().version().value()
            << std::endl
            << "model.signature_name: "
            << response->model_spec().signature_name() << std::endl;
  return 0;
}

int PredictionClient::Predict(cv::Mat &&image, float score /* = 0.5 */,
                              const std::string &save_path /* = {} */) {

  // TODO: 确定1和2对应的是w和h,分别是哪一个
  const int image_w = tensor_input_.tensor_shape().dim(1).size();
  const int image_h = tensor_input_.tensor_shape().dim(2).size();
  const int result_class = tensor_output_.tensor_shape().dim(1).size();
  const int result_num = tensor_output_.tensor_shape().dim(2).size();

  std::cout << "image_w: " << image_w << std::endl
            << "image_h: " << image_h << std::endl
            << "result_class: " << result_class << std::endl
            << "result_num: " << result_num << std::endl;

  // Step.1: 构造必要上下文
  ::grpc::ClientContext context;
  PredictRequest request;

  // Step.2: 构造请求数据
  request.mutable_model_spec()->set_name(model_name_);
  request.mutable_model_spec()->set_signature_name("serving_default");

  // Step.3 处理输入图像
  std::cout << "image.dims: " << image.dims << std::endl
            << "image.channels: " << image.channels() << std::endl
            << "image.cols|width: " << image.cols << std::endl
            << "image.rows|height: " << image.rows << std::endl
            << "image.depth: " << image.depth() << std::endl
            << "image.elemSize: " << image.elemSize() << std::endl;
  assert(image.dims == 2);       // 维度为二维
  assert(image.channels() == 3); // 通道为3
  assert(image_w == image_h);

  cv::Mat convert_mat(image_w, image_h, CV_32FC3);
  float scale = 0.0;
  int padding_top = 0, padding_left = 0;
  letterbox_preprocess(image, convert_mat, &scale, &padding_top, &padding_left,
                       image_w);

  std::cout << "convert_mat.dims: " << convert_mat.dims << std::endl
            << "convert_mat.channels: " << convert_mat.channels() << std::endl
            << "convert_mat.cols|width: " << convert_mat.cols << std::endl
            << "convert_mat.rows|height: " << convert_mat.rows << std::endl
            << "convert_mat.depth: " << convert_mat.depth() << std::endl
            << "convert_mat.total: " << convert_mat.total() << std::endl
            << "convert_mat.elemSize: " << convert_mat.elemSize() << std::endl;

  // Step.4: 填充数据
  ::tensorflow::TensorProto tensor_proto(tensor_input_);
  const size_t data_size = convert_mat.total() * convert_mat.elemSize();
  tensor_proto.set_tensor_content((char *)(convert_mat.data), data_size);
  auto &inputs = *request.mutable_inputs();
  inputs["images"].CopyFrom(tensor_proto);

  // Step.5: 调用远程并返回
  ::tensorflow::serving::PredictResponse response;
  auto status = stub_->Predict(&context, request, &response);

  // Step.6: 处理返回结果
  if (!status.ok()) {
    std::cerr << "Predict failed." << std::endl;
    return -1;
  }

  const auto &result = response.outputs().at("output0");

  auto filter_box =
      this->filterBoxByScores(result, scale, padding_top, padding_left, score);

  auto obj_box = this->nms(filter_box);

  this->drawResult(obj_box, image, save_path);

  return 0;
}

void PredictionClient::Final() {}

void PredictionClient::letterbox_preprocess(const cv::Mat &src, cv::Mat &dest,
                                            float *p_scale, int *p_padding_top,
                                            int *p_padding_left,
                                            int target_size /* = 640 */,
                                            bool normalization /* = true */
) {

  // 原始图像尺寸
  int src_h = src.rows;
  int src_w = src.cols;

  // 计算缩放比例
  float scale = std::min(static_cast<float>(target_size) / src_h,
                         static_cast<float>(target_size) / src_w);

  // 计算缩放后的新尺寸
  int new_w = static_cast<int>(src_w * scale);
  int new_h = static_cast<int>(src_h * scale);

  // 执行缩放
  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

  // 计算填充量
  int padding_w = target_size - new_w;
  int padding_h = target_size - new_h;

  // 均分填充到两侧（右侧/底部可能有1像素的余数）
  int padding_left = padding_w / 2;
  int padding_right = padding_w - padding_left;
  int padding_top = padding_h / 2;
  int padding_bottom = padding_h - padding_top;

  // 创建目标图像并填充
  cv::Mat padded = cv::Mat::zeros(target_size, target_size, src.type());
  padded.setTo(cv::Scalar(114, 114, 114)); // 填充灰色

  // 将缩放后的图像复制到中心
  resized.copyTo(padded(cv::Rect(padding_left, padding_top, new_w, new_h)));

  // 转换为RGB格式（如果需要）
  cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

  if (normalization) {
    padded.convertTo(dest, CV_32FC3, 1.0f / 255.0);
  } else {
    padded.convertTo(dest, CV_32FC3, 1.0f);
  }

  *p_scale = scale;
  *p_padding_top = padding_top;
  *p_padding_left = padding_left;
}

int PredictionClient::drawResult(std::vector<BoxInfo> &boxs_info,
                                 cv::Mat &origin_image,
                                 const std::string &output_image /* = {} */) {

  const int image_w = tensor_input_.tensor_shape().dim(1).size();
  const int image_h = tensor_input_.tensor_shape().dim(2).size();

  cv::Mat convert_mat(image_w, image_h, origin_image.type());
  float scale = 0.0;
  int padding_top = 0, padding_left = 0;

  letterbox_preprocess(origin_image, convert_mat, &scale, &padding_top,
                       &padding_left, image_w, false);

  std::sort(boxs_info.begin(), boxs_info.end(),
            [this](const BoxInfo &l, const BoxInfo &r) {
              return (l.x1 + l.x2) < (r.x1 + r.x2);
            });

  int pos_y = 20;
  for (const auto &box : boxs_info) {
    char s[64] = {'\0'};
    sprintf(s, "Label: %d | %10.3f", box.label, box.score);
    cv::putText(convert_mat, s, cv::Point(50, pos_y += 20),
                cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(255, 0, 0));
    cv::rectangle(convert_mat, cv::Point(box.x1, box.y1),
                  cv::Point(box.x2, box.y2), cv::Scalar(255, 0, 0));
  }

  if (!output_image.empty()) {
    cv::imwrite(output_image, convert_mat);
    std::cout << "Write to image: " << output_image << std::endl;
  }
  return 0;
}

std::vector<BoxInfo> PredictionClient::nms(std::vector<BoxInfo> &boxes,
                                           float nms_threshold /* = 0.5 */
) {

  std::vector<BoxInfo> result;
  std::vector<size_t> indices(boxes.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(boxes.begin(), boxes.end(),
            [this](const BoxInfo &lhs, const BoxInfo &rhs) {
              return lhs.score > rhs.score;
            });

  for (auto &box : boxes) {
    while (!indices.empty()) {
      size_t best_idx = indices[0];
      result.push_back(boxes[best_idx]);
      // 移除已选中的框
      indices.erase(indices.begin());
      indices.erase(std::remove_if(indices.begin(), indices.end(),
                                   [&](size_t i) {
                                     return iou(boxes[best_idx], boxes[i]) >
                                            nms_threshold;
                                   }),
                    indices.end());
    }
  }

  return result;
}

std::vector<BoxInfo> PredictionClient::filterBoxByScores(
    const tensorflow::TensorProto &result, float scale, int padding_top,
    int padding_left, float scores /* = 0.5 */) {
  std::vector<BoxInfo> ret;

  const int result_class = tensor_output_.tensor_shape().dim(1).size();
  const int result_num = tensor_output_.tensor_shape().dim(2).size();
  const int offset_cx = result_num * 0;
  const int offset_cy = result_num * 1;
  const int offset_w = result_num * 2;
  const int offset_h = result_num * 3;

  for (int i = 0; i < result_num; ++i) {
    std::vector<float> class_ele;
    class_ele.reserve(32);

    for (int j = 4; j < result_class; ++j) {
      class_ele.push_back(result.float_val(result_num * j + i));
    }

    auto biggest_iter = std::max_element(class_ele.begin(), class_ele.end());

    if (biggest_iter == class_ele.end()) {
      std::cerr << "Can find biggest iterator" << std::endl;
      return {};
    }

    int biggest_pos = std::distance(class_ele.begin(), biggest_iter);
    float biggest_value = class_ele[biggest_pos];

    if (biggest_value > scores) {
      float cx = result.float_val(offset_cx + i);
      float cy = result.float_val(offset_cy + i);
      float w = result.float_val(offset_w + i);
      float h = result.float_val(offset_h + i);
      int x1 = int(cx - w / 2);
      int x2 = int(cx + w / 2);
      int y1 = int(cy - h / 2);
      int y2 = int(cy + h / 2);
      int areas = (y2 - y1) * (x2 - x1);
      ret.emplace_back(x1, y1, x2, y2, biggest_pos, biggest_value, areas);
    } else {
      continue;
    }
  }

  return ret;
}

float PredictionClient::iou(const BoxInfo &box1, const BoxInfo &box2) {
  // Calculate intersection area
  float x1 = std::max(box1.x1, box2.x1);
  float y1 = std::max(box1.y1, box2.y1);
  float x2 = std::min(box1.x2, box2.x2);
  float y2 = std::min(box1.y2, box2.y2);

  float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);

  // Calculate union area
  float box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
  float box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
  float union_area = box1_area + box2_area - intersection_area;

  // Avoid division by zero
  if (union_area == 0.0f)
    return 0.0f;

  return intersection_area / union_area;
}

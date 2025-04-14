#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import numpy as np
import cv2


def load_image(image_path):
    """
    加载并预处理图像
    :param image_path: 图像文件路径
    :return: 预处理后的图像数组
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img.astype(np.float32)


def check_model_availability(channel, model_name):
    """
    检查模型是否可用
    :param channel: gRPC 通道
    :param model_name: 模型名称
    :return: 模型是否可用的布尔值
    """
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    from tensorflow_serving.apis import get_model_metadata_pb2
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = model_name
    request.metadata_field.append("signature_def")
    try:
        result = stub.GetModelMetadata(request, 10.0)
        print(f"模型 {model_name} 可用。")
        return True
    except grpc.RpcError as e:
        print(f"模型 {model_name} 不可用: {e}")
        return False

def postprocess_output(output, original_image, confidence_threshold=0.5):
    """
    对模型输出进行后处理，绘制检测框
    :param output: 模型输出
    :param original_image: 原始图像
    :param confidence_threshold: 置信度阈值
    :return: 绘制了检测框的图像
    """
    # 6 = cx, cy, w, h, confidence_0, confidence_1

    height, width, _ = original_image.shape

    scale_x = width / 640.0
    scale_y = height / 640.0

    for i in range(output.shape[2]):
        detection = output[0, :, i]
        cx = int(detection[0] * scale_x)
        cy = int(detection[1] * scale_y)
        w = int(detection[2]* scale_x)
        h = int(detection[3]* scale_y)
        confidence_0 = detection[4]
        confidence_1 = detection[5]

        if confidence_0 > confidence_threshold:
            x1 = cx - int(w / 2)
            x2 = cx + int(w / 2)
            y1 = cy - int(h / 2)
            y2 = cy + int(h / 2)
            cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            print("Class: 0")
            # label = f'{class_id}: {confidence:.2f}'
            # print("x_center: {}, detection[0]: {}, detection[1]: {}, width: {}, height: {}", x_center, detection[0], detection[1], width, height)
            # print(">>>> {} ({} {}) ({}, {})", label, x1, y1, x2, y2)
            # cv2.putText(original_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # if confidence_1 > confidence_threshold:
            # cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # print("Class 1")
    return original_image


def main():
    # 连接到 TensorFlow Serving 服务
    channel = grpc.insecure_channel('123.57.18.145:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    model_name = 'dogcat'

    # 检查模型可用性
    if not check_model_availability(channel, model_name):
        return

    # 加载测试图像
    image_path = './cat.jpg'
    input_image = load_image(image_path)
    original_image = cv2.imread(image_path)


    # 创建预测请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'

    # 设置输入张量
    request.inputs['images'].CopyFrom(
        tf.make_tensor_proto(input_image, shape=input_image.shape))

    try:
        # 发送预测请求
        result = stub.Predict(request, 60.0)
        # 获取输出张量
        output = result.outputs['output0']
        output_tensor = tf.make_ndarray(output)
        print("推理结果形状:", output_tensor.shape)
        result_image = postprocess_output(output_tensor, original_image)
        cv2.imwrite("result.jpg", result_image)
    except grpc.RpcError as e:
        print(f"推理过程中出现错误: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash

protoc -I proto --cpp_out=proto proto/tensorflow/core/example/*.proto
protoc -I proto --cpp_out=proto proto/tensorflow/core/framework/*.proto
protoc -I proto --cpp_out=proto proto/tensorflow/core/protobuf/*.proto
protoc -I proto --cpp_out=proto proto/tensorflow_serving/apis/*.proto
protoc -I proto --cpp_out=proto proto/tsl/protobuf/*.proto
protoc -I proto --cpp_out=proto proto/xla/*.proto
protoc -I proto --cpp_out=proto proto/xla/tsl/protobuf/*.proto
protoc -I proto --cpp_out=proto proto/tensorflow_serving/config/*.proto
protoc -I proto --cpp_out=proto proto/xla/service/*.proto


protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tensorflow/core/example/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tensorflow/core/framework/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tensorflow/core/protobuf/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tensorflow_serving/apis/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tsl/protobuf/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/xla/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/xla/tsl/protobuf/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/tensorflow_serving/config/*.proto
protoc -I proto --grpc_out=proto --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` proto/xla/service/*.proto

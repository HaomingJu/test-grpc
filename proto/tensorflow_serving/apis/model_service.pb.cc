// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/apis/model_service.proto

#include "tensorflow_serving/apis/model_service.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>

PROTOBUF_PRAGMA_INIT_SEG

namespace _pb = ::PROTOBUF_NAMESPACE_ID;
namespace _pbi = _pb::internal;

namespace tensorflow {
namespace serving {
}  // namespace serving
}  // namespace tensorflow
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto = nullptr;
const uint32_t TableStruct_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto::offsets[1] = {};
static constexpr ::_pbi::MigrationSchema* schemas = nullptr;
static constexpr ::_pb::Message* const* file_default_instances = nullptr;

const char descriptor_table_protodef_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n+tensorflow_serving/apis/model_service."
  "proto\022\022tensorflow.serving\032.tensorflow_se"
  "rving/apis/get_model_status.proto\032.tenso"
  "rflow_serving/apis/model_management.prot"
  "o2\347\001\n\014ModelService\022g\n\016GetModelStatus\022).t"
  "ensorflow.serving.GetModelStatusRequest\032"
  "*.tensorflow.serving.GetModelStatusRespo"
  "nse\022n\n\031HandleReloadConfigRequest\022\'.tenso"
  "rflow.serving.ReloadConfigRequest\032(.tens"
  "orflow.serving.ReloadConfigResponseB\003\370\001\001"
  "b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto_deps[2] = {
  &::descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto,
  &::descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fmanagement_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto = {
    false, false, 408, descriptor_table_protodef_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto,
    "tensorflow_serving/apis/model_service.proto",
    &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto_once, descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto_deps, 2, 0,
    schemas, file_default_instances, TableStruct_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto::offsets,
    nullptr, file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto,
    file_level_service_descriptors_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto_getter() {
  return &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto(&descriptor_table_tensorflow_5fserving_2fapis_2fmodel_5fservice_2eproto);
namespace tensorflow {
namespace serving {

// @@protoc_insertion_point(namespace_scope)
}  // namespace serving
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

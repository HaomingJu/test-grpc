// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/master_service.proto

#include "tensorflow/core/protobuf/master_service.pb.h"

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
namespace grpc {
}  // namespace grpc
}  // namespace tensorflow
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto = nullptr;
const uint32_t TableStruct_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto::offsets[1] = {};
static constexpr ::_pbi::MigrationSchema* schemas = nullptr;
static constexpr ::_pb::Message* const* file_default_instances = nullptr;

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n-tensorflow/core/protobuf/master_servic"
  "e.proto\022\017tensorflow.grpc\032%tensorflow/cor"
  "e/protobuf/master.proto2\273\006\n\rMasterServic"
  "e\022T\n\rCreateSession\022 .tensorflow.CreateSe"
  "ssionRequest\032!.tensorflow.CreateSessionR"
  "esponse\022T\n\rExtendSession\022 .tensorflow.Ex"
  "tendSessionRequest\032!.tensorflow.ExtendSe"
  "ssionResponse\022Z\n\017PartialRunSetup\022\".tenso"
  "rflow.PartialRunSetupRequest\032#.tensorflo"
  "w.PartialRunSetupResponse\022B\n\007RunStep\022\032.t"
  "ensorflow.RunStepRequest\032\033.tensorflow.Ru"
  "nStepResponse\022Q\n\014CloseSession\022\037.tensorfl"
  "ow.CloseSessionRequest\032 .tensorflow.Clos"
  "eSessionResponse\022N\n\013ListDevices\022\036.tensor"
  "flow.ListDevicesRequest\032\037.tensorflow.Lis"
  "tDevicesResponse\022<\n\005Reset\022\030.tensorflow.R"
  "esetRequest\032\031.tensorflow.ResetResponse\022Q"
  "\n\014MakeCallable\022\037.tensorflow.MakeCallable"
  "Request\032 .tensorflow.MakeCallableRespons"
  "e\022N\n\013RunCallable\022\036.tensorflow.RunCallabl"
  "eRequest\032\037.tensorflow.RunCallableRespons"
  "e\022Z\n\017ReleaseCallable\022\".tensorflow.Releas"
  "eCallableRequest\032#.tensorflow.ReleaseCal"
  "lableResponseB\212\001\n\032org.tensorflow.distrun"
  "timeB\023MasterServiceProtosP\001ZUgithub.com/"
  "tensorflow/tensorflow/tensorflow/go/core"
  "/protobuf/for_core_protos_go_protob\006prot"
  "o3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto = {
    false, false, 1082, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto,
    "tensorflow/core/protobuf/master_service.proto",
    &descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto_once, descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto_deps, 1, 0,
    schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto::offsets,
    nullptr, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto,
    file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto_getter() {
  return &descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fmaster_5fservice_2eproto);
namespace tensorflow {
namespace grpc {

// @@protoc_insertion_point(namespace_scope)
}  // namespace grpc
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

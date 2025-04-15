// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/apis/get_model_status.proto

#include "tensorflow_serving/apis/get_model_status.pb.h"

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
PROTOBUF_CONSTEXPR GetModelStatusRequest::GetModelStatusRequest(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.model_spec_)*/nullptr
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct GetModelStatusRequestDefaultTypeInternal {
  PROTOBUF_CONSTEXPR GetModelStatusRequestDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~GetModelStatusRequestDefaultTypeInternal() {}
  union {
    GetModelStatusRequest _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GetModelStatusRequestDefaultTypeInternal _GetModelStatusRequest_default_instance_;
PROTOBUF_CONSTEXPR ModelVersionStatus::ModelVersionStatus(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.status_)*/nullptr
  , /*decltype(_impl_.version_)*/int64_t{0}
  , /*decltype(_impl_.state_)*/0
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ModelVersionStatusDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ModelVersionStatusDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ModelVersionStatusDefaultTypeInternal() {}
  union {
    ModelVersionStatus _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ModelVersionStatusDefaultTypeInternal _ModelVersionStatus_default_instance_;
PROTOBUF_CONSTEXPR GetModelStatusResponse::GetModelStatusResponse(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.model_version_status_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct GetModelStatusResponseDefaultTypeInternal {
  PROTOBUF_CONSTEXPR GetModelStatusResponseDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~GetModelStatusResponseDefaultTypeInternal() {}
  union {
    GetModelStatusResponse _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 GetModelStatusResponseDefaultTypeInternal _GetModelStatusResponse_default_instance_;
}  // namespace serving
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[3];
static const ::_pb::EnumDescriptor* file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[1];
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::GetModelStatusRequest, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::GetModelStatusRequest, _impl_.model_spec_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelVersionStatus, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelVersionStatus, _impl_.version_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelVersionStatus, _impl_.state_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelVersionStatus, _impl_.status_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::GetModelStatusResponse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::GetModelStatusResponse, _impl_.model_version_status_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::serving::GetModelStatusRequest)},
  { 7, -1, -1, sizeof(::tensorflow::serving::ModelVersionStatus)},
  { 16, -1, -1, sizeof(::tensorflow::serving::GetModelStatusResponse)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::serving::_GetModelStatusRequest_default_instance_._instance,
  &::tensorflow::serving::_ModelVersionStatus_default_instance_._instance,
  &::tensorflow::serving::_GetModelStatusResponse_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n.tensorflow_serving/apis/get_model_stat"
  "us.proto\022\022tensorflow.serving\032#tensorflow"
  "_serving/apis/model.proto\032$tensorflow_se"
  "rving/apis/status.proto\"J\n\025GetModelStatu"
  "sRequest\0221\n\nmodel_spec\030\001 \001(\0132\035.tensorflo"
  "w.serving.ModelSpec\"\350\001\n\022ModelVersionStat"
  "us\022\017\n\007version\030\001 \001(\003\022;\n\005state\030\002 \001(\0162,.ten"
  "sorflow.serving.ModelVersionStatus.State"
  "\022/\n\006status\030\003 \001(\0132\037.tensorflow.serving.St"
  "atusProto\"S\n\005State\022\013\n\007UNKNOWN\020\000\022\t\n\005START"
  "\020\n\022\013\n\007LOADING\020\024\022\r\n\tAVAILABLE\020\036\022\r\n\tUNLOAD"
  "ING\020(\022\007\n\003END\0202\"t\n\026GetModelStatusResponse"
  "\022Z\n\024model_version_status\030\001 \003(\0132&.tensorf"
  "low.serving.ModelVersionStatusR\024model_ve"
  "rsion_statusB\003\370\001\001b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_deps[2] = {
  &::descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto,
  &::descriptor_table_tensorflow_5fserving_2fapis_2fstatus_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto = {
    false, false, 585, descriptor_table_protodef_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto,
    "tensorflow_serving/apis/get_model_status.proto",
    &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_once, descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_deps, 2, 3,
    schemas, file_default_instances, TableStruct_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto::offsets,
    file_level_metadata_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto, file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto,
    file_level_service_descriptors_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_getter() {
  return &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto(&descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto);
namespace tensorflow {
namespace serving {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* ModelVersionStatus_State_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto);
  return file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[0];
}
bool ModelVersionStatus_State_IsValid(int value) {
  switch (value) {
    case 0:
    case 10:
    case 20:
    case 30:
    case 40:
    case 50:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))
constexpr ModelVersionStatus_State ModelVersionStatus::UNKNOWN;
constexpr ModelVersionStatus_State ModelVersionStatus::START;
constexpr ModelVersionStatus_State ModelVersionStatus::LOADING;
constexpr ModelVersionStatus_State ModelVersionStatus::AVAILABLE;
constexpr ModelVersionStatus_State ModelVersionStatus::UNLOADING;
constexpr ModelVersionStatus_State ModelVersionStatus::END;
constexpr ModelVersionStatus_State ModelVersionStatus::State_MIN;
constexpr ModelVersionStatus_State ModelVersionStatus::State_MAX;
constexpr int ModelVersionStatus::State_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || (_MSC_VER >= 1900 && _MSC_VER < 1912))

// ===================================================================

class GetModelStatusRequest::_Internal {
 public:
  static const ::tensorflow::serving::ModelSpec& model_spec(const GetModelStatusRequest* msg);
};

const ::tensorflow::serving::ModelSpec&
GetModelStatusRequest::_Internal::model_spec(const GetModelStatusRequest* msg) {
  return *msg->_impl_.model_spec_;
}
void GetModelStatusRequest::clear_model_spec() {
  if (GetArenaForAllocation() == nullptr && _impl_.model_spec_ != nullptr) {
    delete _impl_.model_spec_;
  }
  _impl_.model_spec_ = nullptr;
}
GetModelStatusRequest::GetModelStatusRequest(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.GetModelStatusRequest)
}
GetModelStatusRequest::GetModelStatusRequest(const GetModelStatusRequest& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  GetModelStatusRequest* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.model_spec_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_model_spec()) {
    _this->_impl_.model_spec_ = new ::tensorflow::serving::ModelSpec(*from._impl_.model_spec_);
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.GetModelStatusRequest)
}

inline void GetModelStatusRequest::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.model_spec_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

GetModelStatusRequest::~GetModelStatusRequest() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.GetModelStatusRequest)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void GetModelStatusRequest::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (this != internal_default_instance()) delete _impl_.model_spec_;
}

void GetModelStatusRequest::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void GetModelStatusRequest::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.GetModelStatusRequest)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaForAllocation() == nullptr && _impl_.model_spec_ != nullptr) {
    delete _impl_.model_spec_;
  }
  _impl_.model_spec_ = nullptr;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* GetModelStatusRequest::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // .tensorflow.serving.ModelSpec model_spec = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_model_spec(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* GetModelStatusRequest::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.GetModelStatusRequest)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.serving.ModelSpec model_spec = 1;
  if (this->_internal_has_model_spec()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, _Internal::model_spec(this),
        _Internal::model_spec(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.GetModelStatusRequest)
  return target;
}

size_t GetModelStatusRequest::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.GetModelStatusRequest)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .tensorflow.serving.ModelSpec model_spec = 1;
  if (this->_internal_has_model_spec()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *_impl_.model_spec_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData GetModelStatusRequest::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    GetModelStatusRequest::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetModelStatusRequest::GetClassData() const { return &_class_data_; }


void GetModelStatusRequest::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<GetModelStatusRequest*>(&to_msg);
  auto& from = static_cast<const GetModelStatusRequest&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.GetModelStatusRequest)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_model_spec()) {
    _this->_internal_mutable_model_spec()->::tensorflow::serving::ModelSpec::MergeFrom(
        from._internal_model_spec());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void GetModelStatusRequest::CopyFrom(const GetModelStatusRequest& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.GetModelStatusRequest)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool GetModelStatusRequest::IsInitialized() const {
  return true;
}

void GetModelStatusRequest::InternalSwap(GetModelStatusRequest* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_impl_.model_spec_, other->_impl_.model_spec_);
}

::PROTOBUF_NAMESPACE_ID::Metadata GetModelStatusRequest::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[0]);
}

// ===================================================================

class ModelVersionStatus::_Internal {
 public:
  static const ::tensorflow::serving::StatusProto& status(const ModelVersionStatus* msg);
};

const ::tensorflow::serving::StatusProto&
ModelVersionStatus::_Internal::status(const ModelVersionStatus* msg) {
  return *msg->_impl_.status_;
}
void ModelVersionStatus::clear_status() {
  if (GetArenaForAllocation() == nullptr && _impl_.status_ != nullptr) {
    delete _impl_.status_;
  }
  _impl_.status_ = nullptr;
}
ModelVersionStatus::ModelVersionStatus(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.ModelVersionStatus)
}
ModelVersionStatus::ModelVersionStatus(const ModelVersionStatus& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ModelVersionStatus* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.status_){nullptr}
    , decltype(_impl_.version_){}
    , decltype(_impl_.state_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_status()) {
    _this->_impl_.status_ = new ::tensorflow::serving::StatusProto(*from._impl_.status_);
  }
  ::memcpy(&_impl_.version_, &from._impl_.version_,
    static_cast<size_t>(reinterpret_cast<char*>(&_impl_.state_) -
    reinterpret_cast<char*>(&_impl_.version_)) + sizeof(_impl_.state_));
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.ModelVersionStatus)
}

inline void ModelVersionStatus::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.status_){nullptr}
    , decltype(_impl_.version_){int64_t{0}}
    , decltype(_impl_.state_){0}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

ModelVersionStatus::~ModelVersionStatus() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.ModelVersionStatus)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ModelVersionStatus::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (this != internal_default_instance()) delete _impl_.status_;
}

void ModelVersionStatus::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ModelVersionStatus::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.ModelVersionStatus)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  if (GetArenaForAllocation() == nullptr && _impl_.status_ != nullptr) {
    delete _impl_.status_;
  }
  _impl_.status_ = nullptr;
  ::memset(&_impl_.version_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&_impl_.state_) -
      reinterpret_cast<char*>(&_impl_.version_)) + sizeof(_impl_.state_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ModelVersionStatus::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // int64 version = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _impl_.version_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // .tensorflow.serving.ModelVersionStatus.State state = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 16)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          _internal_set_state(static_cast<::tensorflow::serving::ModelVersionStatus_State>(val));
        } else
          goto handle_unusual;
        continue;
      // .tensorflow.serving.StatusProto status = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_status(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* ModelVersionStatus::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.ModelVersionStatus)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 version = 1;
  if (this->_internal_version() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt64ToArray(1, this->_internal_version(), target);
  }

  // .tensorflow.serving.ModelVersionStatus.State state = 2;
  if (this->_internal_state() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteEnumToArray(
      2, this->_internal_state(), target);
  }

  // .tensorflow.serving.StatusProto status = 3;
  if (this->_internal_has_status()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(3, _Internal::status(this),
        _Internal::status(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.ModelVersionStatus)
  return target;
}

size_t ModelVersionStatus::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.ModelVersionStatus)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .tensorflow.serving.StatusProto status = 3;
  if (this->_internal_has_status()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *_impl_.status_);
  }

  // int64 version = 1;
  if (this->_internal_version() != 0) {
    total_size += ::_pbi::WireFormatLite::Int64SizePlusOne(this->_internal_version());
  }

  // .tensorflow.serving.ModelVersionStatus.State state = 2;
  if (this->_internal_state() != 0) {
    total_size += 1 +
      ::_pbi::WireFormatLite::EnumSize(this->_internal_state());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ModelVersionStatus::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ModelVersionStatus::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ModelVersionStatus::GetClassData() const { return &_class_data_; }


void ModelVersionStatus::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ModelVersionStatus*>(&to_msg);
  auto& from = static_cast<const ModelVersionStatus&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.ModelVersionStatus)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_has_status()) {
    _this->_internal_mutable_status()->::tensorflow::serving::StatusProto::MergeFrom(
        from._internal_status());
  }
  if (from._internal_version() != 0) {
    _this->_internal_set_version(from._internal_version());
  }
  if (from._internal_state() != 0) {
    _this->_internal_set_state(from._internal_state());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ModelVersionStatus::CopyFrom(const ModelVersionStatus& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.ModelVersionStatus)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ModelVersionStatus::IsInitialized() const {
  return true;
}

void ModelVersionStatus::InternalSwap(ModelVersionStatus* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(ModelVersionStatus, _impl_.state_)
      + sizeof(ModelVersionStatus::_impl_.state_)
      - PROTOBUF_FIELD_OFFSET(ModelVersionStatus, _impl_.status_)>(
          reinterpret_cast<char*>(&_impl_.status_),
          reinterpret_cast<char*>(&other->_impl_.status_));
}

::PROTOBUF_NAMESPACE_ID::Metadata ModelVersionStatus::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[1]);
}

// ===================================================================

class GetModelStatusResponse::_Internal {
 public:
};

GetModelStatusResponse::GetModelStatusResponse(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.GetModelStatusResponse)
}
GetModelStatusResponse::GetModelStatusResponse(const GetModelStatusResponse& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  GetModelStatusResponse* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.model_version_status_){from._impl_.model_version_status_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.GetModelStatusResponse)
}

inline void GetModelStatusResponse::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.model_version_status_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

GetModelStatusResponse::~GetModelStatusResponse() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.GetModelStatusResponse)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void GetModelStatusResponse::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.model_version_status_.~RepeatedPtrField();
}

void GetModelStatusResponse::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void GetModelStatusResponse::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.GetModelStatusResponse)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.model_version_status_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* GetModelStatusResponse::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .tensorflow.serving.ModelVersionStatus model_version_status = 1 [json_name = "model_version_status"];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_model_version_status(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      default:
        goto handle_unusual;
    }  // switch
  handle_unusual:
    if ((tag == 0) || ((tag & 7) == 4)) {
      CHK_(ptr);
      ctx->SetLastTag(tag);
      goto message_done;
    }
    ptr = UnknownFieldParse(
        tag,
        _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
        ptr, ctx);
    CHK_(ptr != nullptr);
  }  // while
message_done:
  return ptr;
failure:
  ptr = nullptr;
  goto message_done;
#undef CHK_
}

uint8_t* GetModelStatusResponse::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.GetModelStatusResponse)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.serving.ModelVersionStatus model_version_status = 1 [json_name = "model_version_status"];
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_model_version_status_size()); i < n; i++) {
    const auto& repfield = this->_internal_model_version_status(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.GetModelStatusResponse)
  return target;
}

size_t GetModelStatusResponse::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.GetModelStatusResponse)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.serving.ModelVersionStatus model_version_status = 1 [json_name = "model_version_status"];
  total_size += 1UL * this->_internal_model_version_status_size();
  for (const auto& msg : this->_impl_.model_version_status_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData GetModelStatusResponse::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    GetModelStatusResponse::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetModelStatusResponse::GetClassData() const { return &_class_data_; }


void GetModelStatusResponse::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<GetModelStatusResponse*>(&to_msg);
  auto& from = static_cast<const GetModelStatusResponse&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.GetModelStatusResponse)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.model_version_status_.MergeFrom(from._impl_.model_version_status_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void GetModelStatusResponse::CopyFrom(const GetModelStatusResponse& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.GetModelStatusResponse)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool GetModelStatusResponse::IsInitialized() const {
  return true;
}

void GetModelStatusResponse::InternalSwap(GetModelStatusResponse* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.model_version_status_.InternalSwap(&other->_impl_.model_version_status_);
}

::PROTOBUF_NAMESPACE_ID::Metadata GetModelStatusResponse::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2fget_5fmodel_5fstatus_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace serving
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::serving::GetModelStatusRequest*
Arena::CreateMaybeMessage< ::tensorflow::serving::GetModelStatusRequest >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::GetModelStatusRequest >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::serving::ModelVersionStatus*
Arena::CreateMaybeMessage< ::tensorflow::serving::ModelVersionStatus >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::ModelVersionStatus >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::serving::GetModelStatusResponse*
Arena::CreateMaybeMessage< ::tensorflow::serving::GetModelStatusResponse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::GetModelStatusResponse >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

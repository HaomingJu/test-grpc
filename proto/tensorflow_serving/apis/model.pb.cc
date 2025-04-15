// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/apis/model.proto

#include "tensorflow_serving/apis/model.pb.h"

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
PROTOBUF_CONSTEXPR ModelSpec::ModelSpec(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.signature_name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.version_choice_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}
  , /*decltype(_impl_._oneof_case_)*/{}} {}
struct ModelSpecDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ModelSpecDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ModelSpecDefaultTypeInternal() {}
  union {
    ModelSpec _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ModelSpecDefaultTypeInternal _ModelSpec_default_instance_;
}  // namespace serving
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_5fserving_2fapis_2fmodel_2eproto[1];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fmodel_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_5fserving_2fapis_2fmodel_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_5fserving_2fapis_2fmodel_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelSpec, _internal_metadata_),
  ~0u,  // no _extensions_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelSpec, _impl_._oneof_case_[0]),
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelSpec, _impl_.name_),
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelSpec, _impl_.signature_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ModelSpec, _impl_.version_choice_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::serving::ModelSpec)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::serving::_ModelSpec_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_5fserving_2fapis_2fmodel_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n#tensorflow_serving/apis/model.proto\022\022t"
  "ensorflow.serving\032\036google/protobuf/wrapp"
  "ers.proto\"\214\001\n\tModelSpec\022\014\n\004name\030\001 \001(\t\022.\n"
  "\007version\030\002 \001(\0132\033.google.protobuf.Int64Va"
  "lueH\000\022\027\n\rversion_label\030\004 \001(\tH\000\022\026\n\016signat"
  "ure_name\030\003 \001(\tB\020\n\016version_choiceB\003\370\001\001b\006p"
  "roto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_deps[1] = {
  &::descriptor_table_google_2fprotobuf_2fwrappers_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto = {
    false, false, 245, descriptor_table_protodef_tensorflow_5fserving_2fapis_2fmodel_2eproto,
    "tensorflow_serving/apis/model.proto",
    &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_once, descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_deps, 1, 1,
    schemas, file_default_instances, TableStruct_tensorflow_5fserving_2fapis_2fmodel_2eproto::offsets,
    file_level_metadata_tensorflow_5fserving_2fapis_2fmodel_2eproto, file_level_enum_descriptors_tensorflow_5fserving_2fapis_2fmodel_2eproto,
    file_level_service_descriptors_tensorflow_5fserving_2fapis_2fmodel_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_getter() {
  return &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_5fserving_2fapis_2fmodel_2eproto(&descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto);
namespace tensorflow {
namespace serving {

// ===================================================================

class ModelSpec::_Internal {
 public:
  static const ::PROTOBUF_NAMESPACE_ID::Int64Value& version(const ModelSpec* msg);
};

const ::PROTOBUF_NAMESPACE_ID::Int64Value&
ModelSpec::_Internal::version(const ModelSpec* msg) {
  return *msg->_impl_.version_choice_.version_;
}
void ModelSpec::set_allocated_version(::PROTOBUF_NAMESPACE_ID::Int64Value* version) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  clear_version_choice();
  if (version) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(version));
    if (message_arena != submessage_arena) {
      version = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, version, submessage_arena);
    }
    set_has_version();
    _impl_.version_choice_.version_ = version;
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.ModelSpec.version)
}
void ModelSpec::clear_version() {
  if (_internal_has_version()) {
    if (GetArenaForAllocation() == nullptr) {
      delete _impl_.version_choice_.version_;
    }
    clear_has_version_choice();
  }
}
ModelSpec::ModelSpec(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.ModelSpec)
}
ModelSpec::ModelSpec(const ModelSpec& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ModelSpec* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.name_){}
    , decltype(_impl_.signature_name_){}
    , decltype(_impl_.version_choice_){}
    , /*decltype(_impl_._cached_size_)*/{}
    , /*decltype(_impl_._oneof_case_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _impl_.name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_name().empty()) {
    _this->_impl_.name_.Set(from._internal_name(), 
      _this->GetArenaForAllocation());
  }
  _impl_.signature_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.signature_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_signature_name().empty()) {
    _this->_impl_.signature_name_.Set(from._internal_signature_name(), 
      _this->GetArenaForAllocation());
  }
  clear_has_version_choice();
  switch (from.version_choice_case()) {
    case kVersion: {
      _this->_internal_mutable_version()->::PROTOBUF_NAMESPACE_ID::Int64Value::MergeFrom(
          from._internal_version());
      break;
    }
    case kVersionLabel: {
      _this->_internal_set_version_label(from._internal_version_label());
      break;
    }
    case VERSION_CHOICE_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.ModelSpec)
}

inline void ModelSpec::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.name_){}
    , decltype(_impl_.signature_name_){}
    , decltype(_impl_.version_choice_){}
    , /*decltype(_impl_._cached_size_)*/{}
    , /*decltype(_impl_._oneof_case_)*/{}
  };
  _impl_.name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.signature_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.signature_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  clear_has_version_choice();
}

ModelSpec::~ModelSpec() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.ModelSpec)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ModelSpec::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.name_.Destroy();
  _impl_.signature_name_.Destroy();
  if (has_version_choice()) {
    clear_version_choice();
  }
}

void ModelSpec::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ModelSpec::clear_version_choice() {
// @@protoc_insertion_point(one_of_clear_start:tensorflow.serving.ModelSpec)
  switch (version_choice_case()) {
    case kVersion: {
      if (GetArenaForAllocation() == nullptr) {
        delete _impl_.version_choice_.version_;
      }
      break;
    }
    case kVersionLabel: {
      _impl_.version_choice_.version_label_.Destroy();
      break;
    }
    case VERSION_CHOICE_NOT_SET: {
      break;
    }
  }
  _impl_._oneof_case_[0] = VERSION_CHOICE_NOT_SET;
}


void ModelSpec::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.ModelSpec)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.name_.ClearToEmpty();
  _impl_.signature_name_.ClearToEmpty();
  clear_version_choice();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ModelSpec::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.ModelSpec.name"));
        } else
          goto handle_unusual;
        continue;
      // .google.protobuf.Int64Value version = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_version(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // string signature_name = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          auto str = _internal_mutable_signature_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.ModelSpec.signature_name"));
        } else
          goto handle_unusual;
        continue;
      // string version_label = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 34)) {
          auto str = _internal_mutable_version_label();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.ModelSpec.version_label"));
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

uint8_t* ModelSpec::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.ModelSpec)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (!this->_internal_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_name().data(), static_cast<int>(this->_internal_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.ModelSpec.name");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_name(), target);
  }

  // .google.protobuf.Int64Value version = 2;
  if (_internal_has_version()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(2, _Internal::version(this),
        _Internal::version(this).GetCachedSize(), target, stream);
  }

  // string signature_name = 3;
  if (!this->_internal_signature_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_signature_name().data(), static_cast<int>(this->_internal_signature_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.ModelSpec.signature_name");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_signature_name(), target);
  }

  // string version_label = 4;
  if (_internal_has_version_label()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_version_label().data(), static_cast<int>(this->_internal_version_label().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.ModelSpec.version_label");
    target = stream->WriteStringMaybeAliased(
        4, this->_internal_version_label(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.ModelSpec)
  return target;
}

size_t ModelSpec::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.ModelSpec)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string name = 1;
  if (!this->_internal_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_name());
  }

  // string signature_name = 3;
  if (!this->_internal_signature_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_signature_name());
  }

  switch (version_choice_case()) {
    // .google.protobuf.Int64Value version = 2;
    case kVersion: {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *_impl_.version_choice_.version_);
      break;
    }
    // string version_label = 4;
    case kVersionLabel: {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
          this->_internal_version_label());
      break;
    }
    case VERSION_CHOICE_NOT_SET: {
      break;
    }
  }
  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ModelSpec::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ModelSpec::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ModelSpec::GetClassData() const { return &_class_data_; }


void ModelSpec::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ModelSpec*>(&to_msg);
  auto& from = static_cast<const ModelSpec&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.ModelSpec)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_name().empty()) {
    _this->_internal_set_name(from._internal_name());
  }
  if (!from._internal_signature_name().empty()) {
    _this->_internal_set_signature_name(from._internal_signature_name());
  }
  switch (from.version_choice_case()) {
    case kVersion: {
      _this->_internal_mutable_version()->::PROTOBUF_NAMESPACE_ID::Int64Value::MergeFrom(
          from._internal_version());
      break;
    }
    case kVersionLabel: {
      _this->_internal_set_version_label(from._internal_version_label());
      break;
    }
    case VERSION_CHOICE_NOT_SET: {
      break;
    }
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ModelSpec::CopyFrom(const ModelSpec& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.ModelSpec)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ModelSpec::IsInitialized() const {
  return true;
}

void ModelSpec::InternalSwap(ModelSpec* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.name_, lhs_arena,
      &other->_impl_.name_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.signature_name_, lhs_arena,
      &other->_impl_.signature_name_, rhs_arena
  );
  swap(_impl_.version_choice_, other->_impl_.version_choice_);
  swap(_impl_._oneof_case_[0], other->_impl_._oneof_case_[0]);
}

::PROTOBUF_NAMESPACE_ID::Metadata ModelSpec::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2fmodel_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2fmodel_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace serving
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::serving::ModelSpec*
Arena::CreateMaybeMessage< ::tensorflow::serving::ModelSpec >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::ModelSpec >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/config/ssl_config.proto

#include "tensorflow_serving/config/ssl_config.pb.h"

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
PROTOBUF_CONSTEXPR SSLConfig::SSLConfig(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.server_key_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.server_cert_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.custom_ca_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.client_verify_)*/false
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct SSLConfigDefaultTypeInternal {
  PROTOBUF_CONSTEXPR SSLConfigDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~SSLConfigDefaultTypeInternal() {}
  union {
    SSLConfig _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 SSLConfigDefaultTypeInternal _SSLConfig_default_instance_;
}  // namespace serving
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto[1];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::SSLConfig, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::SSLConfig, _impl_.server_key_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::SSLConfig, _impl_.server_cert_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::SSLConfig, _impl_.custom_ca_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::SSLConfig, _impl_.client_verify_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::serving::SSLConfig)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::serving::_SSLConfig_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n*tensorflow_serving/config/ssl_config.p"
  "roto\022\022tensorflow.serving\"^\n\tSSLConfig\022\022\n"
  "\nserver_key\030\001 \001(\t\022\023\n\013server_cert\030\002 \001(\t\022\021"
  "\n\tcustom_ca\030\003 \001(\t\022\025\n\rclient_verify\030\004 \001(\010"
  "B\003\370\001\001b\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto = {
    false, false, 173, descriptor_table_protodef_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto,
    "tensorflow_serving/config/ssl_config.proto",
    &descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto_once, nullptr, 0, 1,
    schemas, file_default_instances, TableStruct_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto::offsets,
    file_level_metadata_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto, file_level_enum_descriptors_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto,
    file_level_service_descriptors_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto_getter() {
  return &descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto(&descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto);
namespace tensorflow {
namespace serving {

// ===================================================================

class SSLConfig::_Internal {
 public:
};

SSLConfig::SSLConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.SSLConfig)
}
SSLConfig::SSLConfig(const SSLConfig& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  SSLConfig* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.server_key_){}
    , decltype(_impl_.server_cert_){}
    , decltype(_impl_.custom_ca_){}
    , decltype(_impl_.client_verify_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _impl_.server_key_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.server_key_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_server_key().empty()) {
    _this->_impl_.server_key_.Set(from._internal_server_key(), 
      _this->GetArenaForAllocation());
  }
  _impl_.server_cert_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.server_cert_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_server_cert().empty()) {
    _this->_impl_.server_cert_.Set(from._internal_server_cert(), 
      _this->GetArenaForAllocation());
  }
  _impl_.custom_ca_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.custom_ca_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_custom_ca().empty()) {
    _this->_impl_.custom_ca_.Set(from._internal_custom_ca(), 
      _this->GetArenaForAllocation());
  }
  _this->_impl_.client_verify_ = from._impl_.client_verify_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.SSLConfig)
}

inline void SSLConfig::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.server_key_){}
    , decltype(_impl_.server_cert_){}
    , decltype(_impl_.custom_ca_){}
    , decltype(_impl_.client_verify_){false}
    , /*decltype(_impl_._cached_size_)*/{}
  };
  _impl_.server_key_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.server_key_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.server_cert_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.server_cert_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.custom_ca_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.custom_ca_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

SSLConfig::~SSLConfig() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.SSLConfig)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void SSLConfig::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.server_key_.Destroy();
  _impl_.server_cert_.Destroy();
  _impl_.custom_ca_.Destroy();
}

void SSLConfig::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void SSLConfig::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.SSLConfig)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.server_key_.ClearToEmpty();
  _impl_.server_cert_.ClearToEmpty();
  _impl_.custom_ca_.ClearToEmpty();
  _impl_.client_verify_ = false;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SSLConfig::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string server_key = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_server_key();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.SSLConfig.server_key"));
        } else
          goto handle_unusual;
        continue;
      // string server_cert = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_server_cert();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.SSLConfig.server_cert"));
        } else
          goto handle_unusual;
        continue;
      // string custom_ca = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          auto str = _internal_mutable_custom_ca();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.serving.SSLConfig.custom_ca"));
        } else
          goto handle_unusual;
        continue;
      // bool client_verify = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 32)) {
          _impl_.client_verify_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
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

uint8_t* SSLConfig::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.SSLConfig)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string server_key = 1;
  if (!this->_internal_server_key().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_server_key().data(), static_cast<int>(this->_internal_server_key().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.SSLConfig.server_key");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_server_key(), target);
  }

  // string server_cert = 2;
  if (!this->_internal_server_cert().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_server_cert().data(), static_cast<int>(this->_internal_server_cert().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.SSLConfig.server_cert");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_server_cert(), target);
  }

  // string custom_ca = 3;
  if (!this->_internal_custom_ca().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_custom_ca().data(), static_cast<int>(this->_internal_custom_ca().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.serving.SSLConfig.custom_ca");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_custom_ca(), target);
  }

  // bool client_verify = 4;
  if (this->_internal_client_verify() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteBoolToArray(4, this->_internal_client_verify(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.SSLConfig)
  return target;
}

size_t SSLConfig::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.SSLConfig)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string server_key = 1;
  if (!this->_internal_server_key().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_server_key());
  }

  // string server_cert = 2;
  if (!this->_internal_server_cert().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_server_cert());
  }

  // string custom_ca = 3;
  if (!this->_internal_custom_ca().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_custom_ca());
  }

  // bool client_verify = 4;
  if (this->_internal_client_verify() != 0) {
    total_size += 1 + 1;
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SSLConfig::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    SSLConfig::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SSLConfig::GetClassData() const { return &_class_data_; }


void SSLConfig::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<SSLConfig*>(&to_msg);
  auto& from = static_cast<const SSLConfig&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.SSLConfig)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_server_key().empty()) {
    _this->_internal_set_server_key(from._internal_server_key());
  }
  if (!from._internal_server_cert().empty()) {
    _this->_internal_set_server_cert(from._internal_server_cert());
  }
  if (!from._internal_custom_ca().empty()) {
    _this->_internal_set_custom_ca(from._internal_custom_ca());
  }
  if (from._internal_client_verify() != 0) {
    _this->_internal_set_client_verify(from._internal_client_verify());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SSLConfig::CopyFrom(const SSLConfig& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.SSLConfig)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SSLConfig::IsInitialized() const {
  return true;
}

void SSLConfig::InternalSwap(SSLConfig* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.server_key_, lhs_arena,
      &other->_impl_.server_key_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.server_cert_, lhs_arena,
      &other->_impl_.server_cert_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.custom_ca_, lhs_arena,
      &other->_impl_.custom_ca_, rhs_arena
  );
  swap(_impl_.client_verify_, other->_impl_.client_verify_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SSLConfig::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fconfig_2fssl_5fconfig_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace serving
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::serving::SSLConfig*
Arena::CreateMaybeMessage< ::tensorflow::serving::SSLConfig >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::SSLConfig >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

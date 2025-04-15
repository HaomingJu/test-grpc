// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/queue_runner.proto

#include "tensorflow/core/protobuf/queue_runner.pb.h"

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
PROTOBUF_CONSTEXPR QueueRunnerDef::QueueRunnerDef(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.enqueue_op_name_)*/{}
  , /*decltype(_impl_.queue_closed_exception_types_)*/{}
  , /*decltype(_impl_._queue_closed_exception_types_cached_byte_size_)*/{0}
  , /*decltype(_impl_.queue_name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.close_op_name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.cancel_op_name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct QueueRunnerDefDefaultTypeInternal {
  PROTOBUF_CONSTEXPR QueueRunnerDefDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~QueueRunnerDefDefaultTypeInternal() {}
  union {
    QueueRunnerDef _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 QueueRunnerDefDefaultTypeInternal _QueueRunnerDef_default_instance_;
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto[1];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _impl_.queue_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _impl_.enqueue_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _impl_.close_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _impl_.cancel_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _impl_.queue_closed_exception_types_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::QueueRunnerDef)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::_QueueRunnerDef_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n+tensorflow/core/protobuf/queue_runner."
  "proto\022\ntensorflow\032*tensorflow/core/proto"
  "buf/error_codes.proto\"\252\001\n\016QueueRunnerDef"
  "\022\022\n\nqueue_name\030\001 \001(\t\022\027\n\017enqueue_op_name\030"
  "\002 \003(\t\022\025\n\rclose_op_name\030\003 \001(\t\022\026\n\016cancel_o"
  "p_name\030\004 \001(\t\022<\n\034queue_closed_exception_t"
  "ypes\030\005 \003(\0162\026.tensorflow.error.CodeB\211\001\n\030o"
  "rg.tensorflow.frameworkB\021QueueRunnerProt"
  "osP\001ZUgithub.com/tensorflow/tensorflow/t"
  "ensorflow/go/core/protobuf/for_core_prot"
  "os_go_proto\370\001\001b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2fcore_2fprotobuf_2ferror_5fcodes_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = {
    false, false, 422, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto,
    "tensorflow/core/protobuf/queue_runner.proto",
    &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_once, descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_deps, 1, 1,
    schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto::offsets,
    file_level_metadata_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto,
    file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_getter() {
  return &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto);
namespace tensorflow {

// ===================================================================

class QueueRunnerDef::_Internal {
 public:
};

QueueRunnerDef::QueueRunnerDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.QueueRunnerDef)
}
QueueRunnerDef::QueueRunnerDef(const QueueRunnerDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  QueueRunnerDef* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.enqueue_op_name_){from._impl_.enqueue_op_name_}
    , decltype(_impl_.queue_closed_exception_types_){from._impl_.queue_closed_exception_types_}
    , /*decltype(_impl_._queue_closed_exception_types_cached_byte_size_)*/{0}
    , decltype(_impl_.queue_name_){}
    , decltype(_impl_.close_op_name_){}
    , decltype(_impl_.cancel_op_name_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _impl_.queue_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.queue_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_queue_name().empty()) {
    _this->_impl_.queue_name_.Set(from._internal_queue_name(), 
      _this->GetArenaForAllocation());
  }
  _impl_.close_op_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.close_op_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_close_op_name().empty()) {
    _this->_impl_.close_op_name_.Set(from._internal_close_op_name(), 
      _this->GetArenaForAllocation());
  }
  _impl_.cancel_op_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.cancel_op_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_cancel_op_name().empty()) {
    _this->_impl_.cancel_op_name_.Set(from._internal_cancel_op_name(), 
      _this->GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.QueueRunnerDef)
}

inline void QueueRunnerDef::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.enqueue_op_name_){arena}
    , decltype(_impl_.queue_closed_exception_types_){arena}
    , /*decltype(_impl_._queue_closed_exception_types_cached_byte_size_)*/{0}
    , decltype(_impl_.queue_name_){}
    , decltype(_impl_.close_op_name_){}
    , decltype(_impl_.cancel_op_name_){}
    , /*decltype(_impl_._cached_size_)*/{}
  };
  _impl_.queue_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.queue_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.close_op_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.close_op_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.cancel_op_name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.cancel_op_name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

QueueRunnerDef::~QueueRunnerDef() {
  // @@protoc_insertion_point(destructor:tensorflow.QueueRunnerDef)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void QueueRunnerDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.enqueue_op_name_.~RepeatedPtrField();
  _impl_.queue_closed_exception_types_.~RepeatedField();
  _impl_.queue_name_.Destroy();
  _impl_.close_op_name_.Destroy();
  _impl_.cancel_op_name_.Destroy();
}

void QueueRunnerDef::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void QueueRunnerDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.QueueRunnerDef)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.enqueue_op_name_.Clear();
  _impl_.queue_closed_exception_types_.Clear();
  _impl_.queue_name_.ClearToEmpty();
  _impl_.close_op_name_.ClearToEmpty();
  _impl_.cancel_op_name_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* QueueRunnerDef::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string queue_name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_queue_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.QueueRunnerDef.queue_name"));
        } else
          goto handle_unusual;
        continue;
      // repeated string enqueue_op_name = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            auto str = _internal_add_enqueue_op_name();
            ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
            CHK_(ptr);
            CHK_(::_pbi::VerifyUTF8(str, "tensorflow.QueueRunnerDef.enqueue_op_name"));
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
        } else
          goto handle_unusual;
        continue;
      // string close_op_name = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          auto str = _internal_mutable_close_op_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.QueueRunnerDef.close_op_name"));
        } else
          goto handle_unusual;
        continue;
      // string cancel_op_name = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 34)) {
          auto str = _internal_mutable_cancel_op_name();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.QueueRunnerDef.cancel_op_name"));
        } else
          goto handle_unusual;
        continue;
      // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 42)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedEnumParser(_internal_mutable_queue_closed_exception_types(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<uint8_t>(tag) == 40) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          _internal_add_queue_closed_exception_types(static_cast<::tensorflow::error::Code>(val));
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

uint8_t* QueueRunnerDef::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.QueueRunnerDef)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string queue_name = 1;
  if (!this->_internal_queue_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_queue_name().data(), static_cast<int>(this->_internal_queue_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.queue_name");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_queue_name(), target);
  }

  // repeated string enqueue_op_name = 2;
  for (int i = 0, n = this->_internal_enqueue_op_name_size(); i < n; i++) {
    const auto& s = this->_internal_enqueue_op_name(i);
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      s.data(), static_cast<int>(s.length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.enqueue_op_name");
    target = stream->WriteString(2, s, target);
  }

  // string close_op_name = 3;
  if (!this->_internal_close_op_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_close_op_name().data(), static_cast<int>(this->_internal_close_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.close_op_name");
    target = stream->WriteStringMaybeAliased(
        3, this->_internal_close_op_name(), target);
  }

  // string cancel_op_name = 4;
  if (!this->_internal_cancel_op_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_cancel_op_name().data(), static_cast<int>(this->_internal_cancel_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.cancel_op_name");
    target = stream->WriteStringMaybeAliased(
        4, this->_internal_cancel_op_name(), target);
  }

  // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
  {
    int byte_size = _impl_._queue_closed_exception_types_cached_byte_size_.load(std::memory_order_relaxed);
    if (byte_size > 0) {
      target = stream->WriteEnumPacked(
          5, _impl_.queue_closed_exception_types_, byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.QueueRunnerDef)
  return target;
}

size_t QueueRunnerDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.QueueRunnerDef)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated string enqueue_op_name = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(_impl_.enqueue_op_name_.size());
  for (int i = 0, n = _impl_.enqueue_op_name_.size(); i < n; i++) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
      _impl_.enqueue_op_name_.Get(i));
  }

  // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
  {
    size_t data_size = 0;
    unsigned int count = static_cast<unsigned int>(this->_internal_queue_closed_exception_types_size());for (unsigned int i = 0; i < count; i++) {
      data_size += ::_pbi::WireFormatLite::EnumSize(
        this->_internal_queue_closed_exception_types(static_cast<int>(i)));
    }
    if (data_size > 0) {
      total_size += 1 +
        ::_pbi::WireFormatLite::Int32Size(static_cast<int32_t>(data_size));
    }
    int cached_size = ::_pbi::ToCachedSize(data_size);
    _impl_._queue_closed_exception_types_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // string queue_name = 1;
  if (!this->_internal_queue_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_queue_name());
  }

  // string close_op_name = 3;
  if (!this->_internal_close_op_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_close_op_name());
  }

  // string cancel_op_name = 4;
  if (!this->_internal_cancel_op_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_cancel_op_name());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData QueueRunnerDef::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    QueueRunnerDef::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*QueueRunnerDef::GetClassData() const { return &_class_data_; }


void QueueRunnerDef::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<QueueRunnerDef*>(&to_msg);
  auto& from = static_cast<const QueueRunnerDef&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.QueueRunnerDef)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.enqueue_op_name_.MergeFrom(from._impl_.enqueue_op_name_);
  _this->_impl_.queue_closed_exception_types_.MergeFrom(from._impl_.queue_closed_exception_types_);
  if (!from._internal_queue_name().empty()) {
    _this->_internal_set_queue_name(from._internal_queue_name());
  }
  if (!from._internal_close_op_name().empty()) {
    _this->_internal_set_close_op_name(from._internal_close_op_name());
  }
  if (!from._internal_cancel_op_name().empty()) {
    _this->_internal_set_cancel_op_name(from._internal_cancel_op_name());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void QueueRunnerDef::CopyFrom(const QueueRunnerDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.QueueRunnerDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool QueueRunnerDef::IsInitialized() const {
  return true;
}

void QueueRunnerDef::InternalSwap(QueueRunnerDef* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.enqueue_op_name_.InternalSwap(&other->_impl_.enqueue_op_name_);
  _impl_.queue_closed_exception_types_.InternalSwap(&other->_impl_.queue_closed_exception_types_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.queue_name_, lhs_arena,
      &other->_impl_.queue_name_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.close_op_name_, lhs_arena,
      &other->_impl_.close_op_name_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.cancel_op_name_, lhs_arena,
      &other->_impl_.cancel_op_name_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata QueueRunnerDef::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::QueueRunnerDef*
Arena::CreateMaybeMessage< ::tensorflow::QueueRunnerDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::QueueRunnerDef >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

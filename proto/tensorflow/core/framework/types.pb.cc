// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/types.proto

#include "tensorflow/core/framework/types.pb.h"

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
PROTOBUF_CONSTEXPR SerializedDType::SerializedDType(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.datatype_)*/0
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct SerializedDTypeDefaultTypeInternal {
  PROTOBUF_CONSTEXPR SerializedDTypeDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~SerializedDTypeDefaultTypeInternal() {}
  union {
    SerializedDType _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 SerializedDTypeDefaultTypeInternal _SerializedDType_default_instance_;
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_2fcore_2fframework_2ftypes_2eproto[1];
static const ::_pb::EnumDescriptor* file_level_enum_descriptors_tensorflow_2fcore_2fframework_2ftypes_2eproto[1];
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fframework_2ftypes_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_2fcore_2fframework_2ftypes_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SerializedDType, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SerializedDType, _impl_.datatype_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::SerializedDType)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::_SerializedDType_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_2fcore_2fframework_2ftypes_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n%tensorflow/core/framework/types.proto\022"
  "\ntensorflow\"9\n\017SerializedDType\022&\n\010dataty"
  "pe\030\001 \001(\0162\024.tensorflow.DataType*\306\007\n\010DataT"
  "ype\022\016\n\nDT_INVALID\020\000\022\014\n\010DT_FLOAT\020\001\022\r\n\tDT_"
  "DOUBLE\020\002\022\014\n\010DT_INT32\020\003\022\014\n\010DT_UINT8\020\004\022\014\n\010"
  "DT_INT16\020\005\022\013\n\007DT_INT8\020\006\022\r\n\tDT_STRING\020\007\022\020"
  "\n\014DT_COMPLEX64\020\010\022\014\n\010DT_INT64\020\t\022\013\n\007DT_BOO"
  "L\020\n\022\014\n\010DT_QINT8\020\013\022\r\n\tDT_QUINT8\020\014\022\r\n\tDT_Q"
  "INT32\020\r\022\017\n\013DT_BFLOAT16\020\016\022\r\n\tDT_QINT16\020\017\022"
  "\016\n\nDT_QUINT16\020\020\022\r\n\tDT_UINT16\020\021\022\021\n\rDT_COM"
  "PLEX128\020\022\022\013\n\007DT_HALF\020\023\022\017\n\013DT_RESOURCE\020\024\022"
  "\016\n\nDT_VARIANT\020\025\022\r\n\tDT_UINT32\020\026\022\r\n\tDT_UIN"
  "T64\020\027\022\022\n\016DT_FLOAT8_E5M2\020\030\022\024\n\020DT_FLOAT8_E"
  "4M3FN\020\031\022\013\n\007DT_INT4\020\035\022\014\n\010DT_UINT4\020\036\022\020\n\014DT"
  "_FLOAT_REF\020e\022\021\n\rDT_DOUBLE_REF\020f\022\020\n\014DT_IN"
  "T32_REF\020g\022\020\n\014DT_UINT8_REF\020h\022\020\n\014DT_INT16_"
  "REF\020i\022\017\n\013DT_INT8_REF\020j\022\021\n\rDT_STRING_REF\020"
  "k\022\024\n\020DT_COMPLEX64_REF\020l\022\020\n\014DT_INT64_REF\020"
  "m\022\017\n\013DT_BOOL_REF\020n\022\020\n\014DT_QINT8_REF\020o\022\021\n\r"
  "DT_QUINT8_REF\020p\022\021\n\rDT_QINT32_REF\020q\022\023\n\017DT"
  "_BFLOAT16_REF\020r\022\021\n\rDT_QINT16_REF\020s\022\022\n\016DT"
  "_QUINT16_REF\020t\022\021\n\rDT_UINT16_REF\020u\022\025\n\021DT_"
  "COMPLEX128_REF\020v\022\017\n\013DT_HALF_REF\020w\022\023\n\017DT_"
  "RESOURCE_REF\020x\022\022\n\016DT_VARIANT_REF\020y\022\021\n\rDT"
  "_UINT32_REF\020z\022\021\n\rDT_UINT64_REF\020{\022\026\n\022DT_F"
  "LOAT8_E5M2_REF\020|\022\030\n\024DT_FLOAT8_E4M3FN_REF"
  "\020}\022\020\n\013DT_INT4_REF\020\201\001\022\021\n\014DT_UINT4_REF\020\202\001B"
  "z\n\030org.tensorflow.frameworkB\013TypesProtos"
  "P\001ZLgithub.com/tensorflow/tensorflow/ten"
  "sorflow/go/core/framework/types_go_proto"
  "\370\001\001b\006proto3"
  ;
static ::_pbi::once_flag descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto = {
    false, false, 1211, descriptor_table_protodef_tensorflow_2fcore_2fframework_2ftypes_2eproto,
    "tensorflow/core/framework/types.proto",
    &descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto_once, nullptr, 0, 1,
    schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fframework_2ftypes_2eproto::offsets,
    file_level_metadata_tensorflow_2fcore_2fframework_2ftypes_2eproto, file_level_enum_descriptors_tensorflow_2fcore_2fframework_2ftypes_2eproto,
    file_level_service_descriptors_tensorflow_2fcore_2fframework_2ftypes_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto_getter() {
  return &descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_2fcore_2fframework_2ftypes_2eproto(&descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto);
namespace tensorflow {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* DataType_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto);
  return file_level_enum_descriptors_tensorflow_2fcore_2fframework_2ftypes_2eproto[0];
}
bool DataType_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 24:
    case 25:
    case 29:
    case 30:
    case 101:
    case 102:
    case 103:
    case 104:
    case 105:
    case 106:
    case 107:
    case 108:
    case 109:
    case 110:
    case 111:
    case 112:
    case 113:
    case 114:
    case 115:
    case 116:
    case 117:
    case 118:
    case 119:
    case 120:
    case 121:
    case 122:
    case 123:
    case 124:
    case 125:
    case 129:
    case 130:
      return true;
    default:
      return false;
  }
}


// ===================================================================

class SerializedDType::_Internal {
 public:
};

SerializedDType::SerializedDType(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SerializedDType)
}
SerializedDType::SerializedDType(const SerializedDType& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  SerializedDType* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.datatype_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _this->_impl_.datatype_ = from._impl_.datatype_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.SerializedDType)
}

inline void SerializedDType::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.datatype_){0}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

SerializedDType::~SerializedDType() {
  // @@protoc_insertion_point(destructor:tensorflow.SerializedDType)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void SerializedDType::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
}

void SerializedDType::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void SerializedDType::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SerializedDType)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.datatype_ = 0;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SerializedDType::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // .tensorflow.DataType datatype = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          uint64_t val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          _internal_set_datatype(static_cast<::tensorflow::DataType>(val));
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

uint8_t* SerializedDType::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SerializedDType)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.DataType datatype = 1;
  if (this->_internal_datatype() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteEnumToArray(
      1, this->_internal_datatype(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SerializedDType)
  return target;
}

size_t SerializedDType::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SerializedDType)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // .tensorflow.DataType datatype = 1;
  if (this->_internal_datatype() != 0) {
    total_size += 1 +
      ::_pbi::WireFormatLite::EnumSize(this->_internal_datatype());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SerializedDType::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    SerializedDType::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SerializedDType::GetClassData() const { return &_class_data_; }


void SerializedDType::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<SerializedDType*>(&to_msg);
  auto& from = static_cast<const SerializedDType&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SerializedDType)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (from._internal_datatype() != 0) {
    _this->_internal_set_datatype(from._internal_datatype());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SerializedDType::CopyFrom(const SerializedDType& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SerializedDType)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SerializedDType::IsInitialized() const {
  return true;
}

void SerializedDType::InternalSwap(SerializedDType* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_impl_.datatype_, other->_impl_.datatype_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SerializedDType::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fframework_2ftypes_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fframework_2ftypes_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::SerializedDType*
Arena::CreateMaybeMessage< ::tensorflow::SerializedDType >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::SerializedDType >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

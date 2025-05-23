// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/saved_model.proto

#include "tensorflow/core/protobuf/saved_model.pb.h"

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
PROTOBUF_CONSTEXPR SavedModel::SavedModel(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.meta_graphs_)*/{}
  , /*decltype(_impl_.saved_model_schema_version_)*/int64_t{0}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct SavedModelDefaultTypeInternal {
  PROTOBUF_CONSTEXPR SavedModelDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~SavedModelDefaultTypeInternal() {}
  union {
    SavedModel _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 SavedModelDefaultTypeInternal _SavedModel_default_instance_;
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto[1];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SavedModel, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SavedModel, _impl_.saved_model_schema_version_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SavedModel, _impl_.meta_graphs_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::SavedModel)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::_SavedModel_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n*tensorflow/core/protobuf/saved_model.p"
  "roto\022\ntensorflow\032)tensorflow/core/protob"
  "uf/meta_graph.proto\"_\n\nSavedModel\022\"\n\032sav"
  "ed_model_schema_version\030\001 \001(\003\022-\n\013meta_gr"
  "aphs\030\002 \003(\0132\030.tensorflow.MetaGraphDefB\210\001\n"
  "\030org.tensorflow.frameworkB\020SavedModelPro"
  "tosP\001ZUgithub.com/tensorflow/tensorflow/"
  "tensorflow/go/core/protobuf/for_core_pro"
  "tos_go_proto\370\001\001b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2fcore_2fprotobuf_2fmeta_5fgraph_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto = {
    false, false, 343, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto,
    "tensorflow/core/protobuf/saved_model.proto",
    &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_once, descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_deps, 1, 1,
    schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto::offsets,
    file_level_metadata_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto,
    file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_getter() {
  return &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto);
namespace tensorflow {

// ===================================================================

class SavedModel::_Internal {
 public:
};

void SavedModel::clear_meta_graphs() {
  _impl_.meta_graphs_.Clear();
}
SavedModel::SavedModel(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SavedModel)
}
SavedModel::SavedModel(const SavedModel& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  SavedModel* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.meta_graphs_){from._impl_.meta_graphs_}
    , decltype(_impl_.saved_model_schema_version_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _this->_impl_.saved_model_schema_version_ = from._impl_.saved_model_schema_version_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.SavedModel)
}

inline void SavedModel::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.meta_graphs_){arena}
    , decltype(_impl_.saved_model_schema_version_){int64_t{0}}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

SavedModel::~SavedModel() {
  // @@protoc_insertion_point(destructor:tensorflow.SavedModel)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void SavedModel::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.meta_graphs_.~RepeatedPtrField();
}

void SavedModel::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void SavedModel::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SavedModel)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.meta_graphs_.Clear();
  _impl_.saved_model_schema_version_ = int64_t{0};
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* SavedModel::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // int64 saved_model_schema_version = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _impl_.saved_model_schema_version_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_meta_graphs(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<18>(ptr));
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

uint8_t* SavedModel::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SavedModel)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 saved_model_schema_version = 1;
  if (this->_internal_saved_model_schema_version() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt64ToArray(1, this->_internal_saved_model_schema_version(), target);
  }

  // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_meta_graphs_size()); i < n; i++) {
    const auto& repfield = this->_internal_meta_graphs(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(2, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SavedModel)
  return target;
}

size_t SavedModel::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SavedModel)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.MetaGraphDef meta_graphs = 2;
  total_size += 1UL * this->_internal_meta_graphs_size();
  for (const auto& msg : this->_impl_.meta_graphs_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // int64 saved_model_schema_version = 1;
  if (this->_internal_saved_model_schema_version() != 0) {
    total_size += ::_pbi::WireFormatLite::Int64SizePlusOne(this->_internal_saved_model_schema_version());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData SavedModel::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    SavedModel::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*SavedModel::GetClassData() const { return &_class_data_; }


void SavedModel::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<SavedModel*>(&to_msg);
  auto& from = static_cast<const SavedModel&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SavedModel)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.meta_graphs_.MergeFrom(from._impl_.meta_graphs_);
  if (from._internal_saved_model_schema_version() != 0) {
    _this->_internal_set_saved_model_schema_version(from._internal_saved_model_schema_version());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void SavedModel::CopyFrom(const SavedModel& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SavedModel)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SavedModel::IsInitialized() const {
  return true;
}

void SavedModel::InternalSwap(SavedModel* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.meta_graphs_.InternalSwap(&other->_impl_.meta_graphs_);
  swap(_impl_.saved_model_schema_version_, other->_impl_.saved_model_schema_version_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SavedModel::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fprotobuf_2fsaved_5fmodel_2eproto[0]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::SavedModel*
Arena::CreateMaybeMessage< ::tensorflow::SavedModel >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::SavedModel >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

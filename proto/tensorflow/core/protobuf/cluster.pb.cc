// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/cluster.proto

#include "tensorflow/core/protobuf/cluster.pb.h"

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
PROTOBUF_CONSTEXPR JobDef_TasksEntry_DoNotUse::JobDef_TasksEntry_DoNotUse(
    ::_pbi::ConstantInitialized) {}
struct JobDef_TasksEntry_DoNotUseDefaultTypeInternal {
  PROTOBUF_CONSTEXPR JobDef_TasksEntry_DoNotUseDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~JobDef_TasksEntry_DoNotUseDefaultTypeInternal() {}
  union {
    JobDef_TasksEntry_DoNotUse _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 JobDef_TasksEntry_DoNotUseDefaultTypeInternal _JobDef_TasksEntry_DoNotUse_default_instance_;
PROTOBUF_CONSTEXPR JobDef::JobDef(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.tasks_)*/{::_pbi::ConstantInitialized()}
  , /*decltype(_impl_.name_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct JobDefDefaultTypeInternal {
  PROTOBUF_CONSTEXPR JobDefDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~JobDefDefaultTypeInternal() {}
  union {
    JobDef _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 JobDefDefaultTypeInternal _JobDef_default_instance_;
PROTOBUF_CONSTEXPR ClusterDef::ClusterDef(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.job_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ClusterDefDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ClusterDefDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ClusterDefDefaultTypeInternal() {}
  union {
    ClusterDef _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ClusterDefDefaultTypeInternal _ClusterDef_default_instance_;
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef_TasksEntry_DoNotUse, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef_TasksEntry_DoNotUse, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef_TasksEntry_DoNotUse, key_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef_TasksEntry_DoNotUse, value_),
  0,
  1,
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef, _impl_.name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::JobDef, _impl_.tasks_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::ClusterDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::ClusterDef, _impl_.job_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 8, -1, sizeof(::tensorflow::JobDef_TasksEntry_DoNotUse)},
  { 10, -1, -1, sizeof(::tensorflow::JobDef)},
  { 18, -1, -1, sizeof(::tensorflow::ClusterDef)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::_JobDef_TasksEntry_DoNotUse_default_instance_._instance,
  &::tensorflow::_JobDef_default_instance_._instance,
  &::tensorflow::_ClusterDef_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n&tensorflow/core/protobuf/cluster.proto"
  "\022\ntensorflow\"r\n\006JobDef\022\014\n\004name\030\001 \001(\t\022,\n\005"
  "tasks\030\002 \003(\0132\035.tensorflow.JobDef.TasksEnt"
  "ry\032,\n\nTasksEntry\022\013\n\003key\030\001 \001(\005\022\r\n\005value\030\002"
  " \001(\t:\0028\001\"-\n\nClusterDef\022\037\n\003job\030\001 \003(\0132\022.te"
  "nsorflow.JobDefB\207\001\n\032org.tensorflow.distr"
  "untimeB\rClusterProtosP\001ZUgithub.com/tens"
  "orflow/tensorflow/tensorflow/go/core/pro"
  "tobuf/for_core_protos_go_proto\370\001\001b\006proto"
  "3"
  ;
static ::_pbi::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto = {
    false, false, 361, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto,
    "tensorflow/core/protobuf/cluster.proto",
    &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_once, nullptr, 0, 3,
    schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto::offsets,
    file_level_metadata_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto,
    file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_getter() {
  return &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto);
namespace tensorflow {

// ===================================================================

JobDef_TasksEntry_DoNotUse::JobDef_TasksEntry_DoNotUse() {}
JobDef_TasksEntry_DoNotUse::JobDef_TasksEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena)
    : SuperType(arena) {}
void JobDef_TasksEntry_DoNotUse::MergeFrom(const JobDef_TasksEntry_DoNotUse& other) {
  MergeFromInternal(other);
}
::PROTOBUF_NAMESPACE_ID::Metadata JobDef_TasksEntry_DoNotUse::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto[0]);
}

// ===================================================================

class JobDef::_Internal {
 public:
};

JobDef::JobDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  if (arena != nullptr && !is_message_owned) {
    arena->OwnCustomDestructor(this, &JobDef::ArenaDtor);
  }
  // @@protoc_insertion_point(arena_constructor:tensorflow.JobDef)
}
JobDef::JobDef(const JobDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  JobDef* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      /*decltype(_impl_.tasks_)*/{}
    , decltype(_impl_.name_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _this->_impl_.tasks_.MergeFrom(from._impl_.tasks_);
  _impl_.name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_name().empty()) {
    _this->_impl_.name_.Set(from._internal_name(), 
      _this->GetArenaForAllocation());
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.JobDef)
}

inline void JobDef::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      /*decltype(_impl_.tasks_)*/{::_pbi::ArenaInitialized(), arena}
    , decltype(_impl_.name_){}
    , /*decltype(_impl_._cached_size_)*/{}
  };
  _impl_.name_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.name_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

JobDef::~JobDef() {
  // @@protoc_insertion_point(destructor:tensorflow.JobDef)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    ArenaDtor(this);
    return;
  }
  SharedDtor();
}

inline void JobDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.tasks_.Destruct();
  _impl_.tasks_.~MapField();
  _impl_.name_.Destroy();
}

void JobDef::ArenaDtor(void* object) {
  JobDef* _this = reinterpret_cast< JobDef* >(object);
  _this->_impl_.tasks_.Destruct();
}
void JobDef::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void JobDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.JobDef)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.tasks_.Clear();
  _impl_.name_.ClearToEmpty();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* JobDef::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
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
          CHK_(::_pbi::VerifyUTF8(str, "tensorflow.JobDef.name"));
        } else
          goto handle_unusual;
        continue;
      // map<int32, string> tasks = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(&_impl_.tasks_, ptr);
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

uint8_t* JobDef::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.JobDef)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string name = 1;
  if (!this->_internal_name().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_name().data(), static_cast<int>(this->_internal_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.JobDef.name");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_name(), target);
  }

  // map<int32, string> tasks = 2;
  if (!this->_internal_tasks().empty()) {
    using MapType = ::_pb::Map<int32_t, std::string>;
    using WireHelper = JobDef_TasksEntry_DoNotUse::Funcs;
    const auto& map_field = this->_internal_tasks();
    auto check_utf8 = [](const MapType::value_type& entry) {
      (void)entry;
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
        entry.second.data(), static_cast<int>(entry.second.length()),
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
        "tensorflow.JobDef.TasksEntry.value");
    };

    if (stream->IsSerializationDeterministic() && map_field.size() > 1) {
      for (const auto& entry : ::_pbi::MapSorterFlat<MapType>(map_field)) {
        target = WireHelper::InternalSerialize(2, entry.first, entry.second, target, stream);
        check_utf8(entry);
      }
    } else {
      for (const auto& entry : map_field) {
        target = WireHelper::InternalSerialize(2, entry.first, entry.second, target, stream);
        check_utf8(entry);
      }
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.JobDef)
  return target;
}

size_t JobDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.JobDef)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // map<int32, string> tasks = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->_internal_tasks_size());
  for (::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >::const_iterator
      it = this->_internal_tasks().begin();
      it != this->_internal_tasks().end(); ++it) {
    total_size += JobDef_TasksEntry_DoNotUse::Funcs::ByteSizeLong(it->first, it->second);
  }

  // string name = 1;
  if (!this->_internal_name().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_name());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData JobDef::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    JobDef::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*JobDef::GetClassData() const { return &_class_data_; }


void JobDef::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<JobDef*>(&to_msg);
  auto& from = static_cast<const JobDef&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.JobDef)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.tasks_.MergeFrom(from._impl_.tasks_);
  if (!from._internal_name().empty()) {
    _this->_internal_set_name(from._internal_name());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void JobDef::CopyFrom(const JobDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.JobDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool JobDef::IsInitialized() const {
  return true;
}

void JobDef::InternalSwap(JobDef* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.tasks_.InternalSwap(&other->_impl_.tasks_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.name_, lhs_arena,
      &other->_impl_.name_, rhs_arena
  );
}

::PROTOBUF_NAMESPACE_ID::Metadata JobDef::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto[1]);
}

// ===================================================================

class ClusterDef::_Internal {
 public:
};

ClusterDef::ClusterDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.ClusterDef)
}
ClusterDef::ClusterDef(const ClusterDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ClusterDef* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.job_){from._impl_.job_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:tensorflow.ClusterDef)
}

inline void ClusterDef::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.job_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

ClusterDef::~ClusterDef() {
  // @@protoc_insertion_point(destructor:tensorflow.ClusterDef)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ClusterDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.job_.~RepeatedPtrField();
}

void ClusterDef::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ClusterDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.ClusterDef)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.job_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ClusterDef::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .tensorflow.JobDef job = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_job(), ptr);
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

uint8_t* ClusterDef::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.ClusterDef)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.JobDef job = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_job_size()); i < n; i++) {
    const auto& repfield = this->_internal_job(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.ClusterDef)
  return target;
}

size_t ClusterDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.ClusterDef)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.JobDef job = 1;
  total_size += 1UL * this->_internal_job_size();
  for (const auto& msg : this->_impl_.job_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ClusterDef::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ClusterDef::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ClusterDef::GetClassData() const { return &_class_data_; }


void ClusterDef::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ClusterDef*>(&to_msg);
  auto& from = static_cast<const ClusterDef&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.ClusterDef)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.job_.MergeFrom(from._impl_.job_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ClusterDef::CopyFrom(const ClusterDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.ClusterDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ClusterDef::IsInitialized() const {
  return true;
}

void ClusterDef::InternalSwap(ClusterDef* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.job_.InternalSwap(&other->_impl_.job_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ClusterDef::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_getter, &descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto_once,
      file_level_metadata_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::JobDef_TasksEntry_DoNotUse*
Arena::CreateMaybeMessage< ::tensorflow::JobDef_TasksEntry_DoNotUse >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::JobDef_TasksEntry_DoNotUse >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::JobDef*
Arena::CreateMaybeMessage< ::tensorflow::JobDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::JobDef >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::ClusterDef*
Arena::CreateMaybeMessage< ::tensorflow::ClusterDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::ClusterDef >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: xla/autotune_results.proto

#include "xla/autotune_results.pb.h"

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

namespace xla {
PROTOBUF_CONSTEXPR AutotuneResults_Entry::AutotuneResults_Entry(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.device_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.hlo_)*/{&::_pbi::fixed_address_empty_string, ::_pbi::ConstantInitialized{}}
  , /*decltype(_impl_.result_)*/nullptr
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct AutotuneResults_EntryDefaultTypeInternal {
  PROTOBUF_CONSTEXPR AutotuneResults_EntryDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~AutotuneResults_EntryDefaultTypeInternal() {}
  union {
    AutotuneResults_Entry _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 AutotuneResults_EntryDefaultTypeInternal _AutotuneResults_Entry_default_instance_;
PROTOBUF_CONSTEXPR AutotuneResults::AutotuneResults(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.results_)*/{}
  , /*decltype(_impl_.version_)*/0
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct AutotuneResultsDefaultTypeInternal {
  PROTOBUF_CONSTEXPR AutotuneResultsDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~AutotuneResultsDefaultTypeInternal() {}
  union {
    AutotuneResults _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 AutotuneResultsDefaultTypeInternal _AutotuneResults_default_instance_;
PROTOBUF_CONSTEXPR AutotuningLogs::AutotuningLogs(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.logs_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct AutotuningLogsDefaultTypeInternal {
  PROTOBUF_CONSTEXPR AutotuningLogsDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~AutotuningLogsDefaultTypeInternal() {}
  union {
    AutotuningLogs _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 AutotuningLogsDefaultTypeInternal _AutotuningLogs_default_instance_;
}  // namespace xla
static ::_pb::Metadata file_level_metadata_xla_2fautotune_5fresults_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_xla_2fautotune_5fresults_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_xla_2fautotune_5fresults_2eproto = nullptr;

const uint32_t TableStruct_xla_2fautotune_5fresults_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults_Entry, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults_Entry, _impl_.device_),
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults_Entry, _impl_.hlo_),
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults_Entry, _impl_.result_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults, _impl_.version_),
  PROTOBUF_FIELD_OFFSET(::xla::AutotuneResults, _impl_.results_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuningLogs, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::xla::AutotuningLogs, _impl_.logs_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::xla::AutotuneResults_Entry)},
  { 9, -1, -1, sizeof(::xla::AutotuneResults)},
  { 17, -1, -1, sizeof(::xla::AutotuningLogs)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::xla::_AutotuneResults_Entry_default_instance_._instance,
  &::xla::_AutotuneResults_default_instance_._instance,
  &::xla::_AutotuningLogs_default_instance_._instance,
};

const char descriptor_table_protodef_xla_2fautotune_5fresults_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\032xla/autotune_results.proto\022\003xla\032\024xla/a"
  "utotuning.proto\"\246\001\n\017AutotuneResults\022\017\n\007v"
  "ersion\030\001 \001(\005\022+\n\007results\030\004 \003(\0132\032.xla.Auto"
  "tuneResults.Entry\032I\n\005Entry\022\016\n\006device\030\001 \001"
  "(\t\022\013\n\003hlo\030\002 \001(\t\022#\n\006result\030\003 \001(\0132\023.xla.Au"
  "totuneResultJ\004\010\002\020\003J\004\010\003\020\004\"2\n\016AutotuningLo"
  "gs\022 \n\004logs\030\001 \003(\0132\022.xla.AutotuningLogb\006pr"
  "oto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_xla_2fautotune_5fresults_2eproto_deps[1] = {
  &::descriptor_table_xla_2fautotuning_2eproto,
};
static ::_pbi::once_flag descriptor_table_xla_2fautotune_5fresults_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_xla_2fautotune_5fresults_2eproto = {
    false, false, 284, descriptor_table_protodef_xla_2fautotune_5fresults_2eproto,
    "xla/autotune_results.proto",
    &descriptor_table_xla_2fautotune_5fresults_2eproto_once, descriptor_table_xla_2fautotune_5fresults_2eproto_deps, 1, 3,
    schemas, file_default_instances, TableStruct_xla_2fautotune_5fresults_2eproto::offsets,
    file_level_metadata_xla_2fautotune_5fresults_2eproto, file_level_enum_descriptors_xla_2fautotune_5fresults_2eproto,
    file_level_service_descriptors_xla_2fautotune_5fresults_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_xla_2fautotune_5fresults_2eproto_getter() {
  return &descriptor_table_xla_2fautotune_5fresults_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_xla_2fautotune_5fresults_2eproto(&descriptor_table_xla_2fautotune_5fresults_2eproto);
namespace xla {

// ===================================================================

class AutotuneResults_Entry::_Internal {
 public:
  static const ::xla::AutotuneResult& result(const AutotuneResults_Entry* msg);
};

const ::xla::AutotuneResult&
AutotuneResults_Entry::_Internal::result(const AutotuneResults_Entry* msg) {
  return *msg->_impl_.result_;
}
void AutotuneResults_Entry::clear_result() {
  if (GetArenaForAllocation() == nullptr && _impl_.result_ != nullptr) {
    delete _impl_.result_;
  }
  _impl_.result_ = nullptr;
}
AutotuneResults_Entry::AutotuneResults_Entry(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:xla.AutotuneResults.Entry)
}
AutotuneResults_Entry::AutotuneResults_Entry(const AutotuneResults_Entry& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  AutotuneResults_Entry* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.device_){}
    , decltype(_impl_.hlo_){}
    , decltype(_impl_.result_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _impl_.device_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.device_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_device().empty()) {
    _this->_impl_.device_.Set(from._internal_device(), 
      _this->GetArenaForAllocation());
  }
  _impl_.hlo_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.hlo_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (!from._internal_hlo().empty()) {
    _this->_impl_.hlo_.Set(from._internal_hlo(), 
      _this->GetArenaForAllocation());
  }
  if (from._internal_has_result()) {
    _this->_impl_.result_ = new ::xla::AutotuneResult(*from._impl_.result_);
  }
  // @@protoc_insertion_point(copy_constructor:xla.AutotuneResults.Entry)
}

inline void AutotuneResults_Entry::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.device_){}
    , decltype(_impl_.hlo_){}
    , decltype(_impl_.result_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}
  };
  _impl_.device_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.device_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  _impl_.hlo_.InitDefault();
  #ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
    _impl_.hlo_.Set("", GetArenaForAllocation());
  #endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
}

AutotuneResults_Entry::~AutotuneResults_Entry() {
  // @@protoc_insertion_point(destructor:xla.AutotuneResults.Entry)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void AutotuneResults_Entry::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.device_.Destroy();
  _impl_.hlo_.Destroy();
  if (this != internal_default_instance()) delete _impl_.result_;
}

void AutotuneResults_Entry::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void AutotuneResults_Entry::Clear() {
// @@protoc_insertion_point(message_clear_start:xla.AutotuneResults.Entry)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.device_.ClearToEmpty();
  _impl_.hlo_.ClearToEmpty();
  if (GetArenaForAllocation() == nullptr && _impl_.result_ != nullptr) {
    delete _impl_.result_;
  }
  _impl_.result_ = nullptr;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* AutotuneResults_Entry::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // string device = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          auto str = _internal_mutable_device();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "xla.AutotuneResults.Entry.device"));
        } else
          goto handle_unusual;
        continue;
      // string hlo = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          auto str = _internal_mutable_hlo();
          ptr = ::_pbi::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(ptr);
          CHK_(::_pbi::VerifyUTF8(str, "xla.AutotuneResults.Entry.hlo"));
        } else
          goto handle_unusual;
        continue;
      // .xla.AutotuneResult result = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 26)) {
          ptr = ctx->ParseMessage(_internal_mutable_result(), ptr);
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

uint8_t* AutotuneResults_Entry::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:xla.AutotuneResults.Entry)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // string device = 1;
  if (!this->_internal_device().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_device().data(), static_cast<int>(this->_internal_device().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "xla.AutotuneResults.Entry.device");
    target = stream->WriteStringMaybeAliased(
        1, this->_internal_device(), target);
  }

  // string hlo = 2;
  if (!this->_internal_hlo().empty()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_hlo().data(), static_cast<int>(this->_internal_hlo().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "xla.AutotuneResults.Entry.hlo");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_hlo(), target);
  }

  // .xla.AutotuneResult result = 3;
  if (this->_internal_has_result()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(3, _Internal::result(this),
        _Internal::result(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:xla.AutotuneResults.Entry)
  return target;
}

size_t AutotuneResults_Entry::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:xla.AutotuneResults.Entry)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string device = 1;
  if (!this->_internal_device().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_device());
  }

  // string hlo = 2;
  if (!this->_internal_hlo().empty()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_hlo());
  }

  // .xla.AutotuneResult result = 3;
  if (this->_internal_has_result()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *_impl_.result_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData AutotuneResults_Entry::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    AutotuneResults_Entry::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*AutotuneResults_Entry::GetClassData() const { return &_class_data_; }


void AutotuneResults_Entry::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<AutotuneResults_Entry*>(&to_msg);
  auto& from = static_cast<const AutotuneResults_Entry&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:xla.AutotuneResults.Entry)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  if (!from._internal_device().empty()) {
    _this->_internal_set_device(from._internal_device());
  }
  if (!from._internal_hlo().empty()) {
    _this->_internal_set_hlo(from._internal_hlo());
  }
  if (from._internal_has_result()) {
    _this->_internal_mutable_result()->::xla::AutotuneResult::MergeFrom(
        from._internal_result());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void AutotuneResults_Entry::CopyFrom(const AutotuneResults_Entry& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:xla.AutotuneResults.Entry)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool AutotuneResults_Entry::IsInitialized() const {
  return true;
}

void AutotuneResults_Entry::InternalSwap(AutotuneResults_Entry* other) {
  using std::swap;
  auto* lhs_arena = GetArenaForAllocation();
  auto* rhs_arena = other->GetArenaForAllocation();
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.device_, lhs_arena,
      &other->_impl_.device_, rhs_arena
  );
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr::InternalSwap(
      &_impl_.hlo_, lhs_arena,
      &other->_impl_.hlo_, rhs_arena
  );
  swap(_impl_.result_, other->_impl_.result_);
}

::PROTOBUF_NAMESPACE_ID::Metadata AutotuneResults_Entry::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_xla_2fautotune_5fresults_2eproto_getter, &descriptor_table_xla_2fautotune_5fresults_2eproto_once,
      file_level_metadata_xla_2fautotune_5fresults_2eproto[0]);
}

// ===================================================================

class AutotuneResults::_Internal {
 public:
};

AutotuneResults::AutotuneResults(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:xla.AutotuneResults)
}
AutotuneResults::AutotuneResults(const AutotuneResults& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  AutotuneResults* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.results_){from._impl_.results_}
    , decltype(_impl_.version_){}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  _this->_impl_.version_ = from._impl_.version_;
  // @@protoc_insertion_point(copy_constructor:xla.AutotuneResults)
}

inline void AutotuneResults::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.results_){arena}
    , decltype(_impl_.version_){0}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

AutotuneResults::~AutotuneResults() {
  // @@protoc_insertion_point(destructor:xla.AutotuneResults)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void AutotuneResults::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.results_.~RepeatedPtrField();
}

void AutotuneResults::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void AutotuneResults::Clear() {
// @@protoc_insertion_point(message_clear_start:xla.AutotuneResults)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.results_.Clear();
  _impl_.version_ = 0;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* AutotuneResults::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // int32 version = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 8)) {
          _impl_.version_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint32(&ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // repeated .xla.AutotuneResults.Entry results = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 34)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_results(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<34>(ptr));
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

uint8_t* AutotuneResults::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:xla.AutotuneResults)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // int32 version = 1;
  if (this->_internal_version() != 0) {
    target = stream->EnsureSpace(target);
    target = ::_pbi::WireFormatLite::WriteInt32ToArray(1, this->_internal_version(), target);
  }

  // repeated .xla.AutotuneResults.Entry results = 4;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_results_size()); i < n; i++) {
    const auto& repfield = this->_internal_results(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(4, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:xla.AutotuneResults)
  return target;
}

size_t AutotuneResults::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:xla.AutotuneResults)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .xla.AutotuneResults.Entry results = 4;
  total_size += 1UL * this->_internal_results_size();
  for (const auto& msg : this->_impl_.results_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // int32 version = 1;
  if (this->_internal_version() != 0) {
    total_size += ::_pbi::WireFormatLite::Int32SizePlusOne(this->_internal_version());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData AutotuneResults::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    AutotuneResults::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*AutotuneResults::GetClassData() const { return &_class_data_; }


void AutotuneResults::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<AutotuneResults*>(&to_msg);
  auto& from = static_cast<const AutotuneResults&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:xla.AutotuneResults)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.results_.MergeFrom(from._impl_.results_);
  if (from._internal_version() != 0) {
    _this->_internal_set_version(from._internal_version());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void AutotuneResults::CopyFrom(const AutotuneResults& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:xla.AutotuneResults)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool AutotuneResults::IsInitialized() const {
  return true;
}

void AutotuneResults::InternalSwap(AutotuneResults* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.results_.InternalSwap(&other->_impl_.results_);
  swap(_impl_.version_, other->_impl_.version_);
}

::PROTOBUF_NAMESPACE_ID::Metadata AutotuneResults::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_xla_2fautotune_5fresults_2eproto_getter, &descriptor_table_xla_2fautotune_5fresults_2eproto_once,
      file_level_metadata_xla_2fautotune_5fresults_2eproto[1]);
}

// ===================================================================

class AutotuningLogs::_Internal {
 public:
};

void AutotuningLogs::clear_logs() {
  _impl_.logs_.Clear();
}
AutotuningLogs::AutotuningLogs(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:xla.AutotuningLogs)
}
AutotuningLogs::AutotuningLogs(const AutotuningLogs& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  AutotuningLogs* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.logs_){from._impl_.logs_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:xla.AutotuningLogs)
}

inline void AutotuningLogs::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.logs_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

AutotuningLogs::~AutotuningLogs() {
  // @@protoc_insertion_point(destructor:xla.AutotuningLogs)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void AutotuningLogs::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.logs_.~RepeatedPtrField();
}

void AutotuningLogs::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void AutotuningLogs::Clear() {
// @@protoc_insertion_point(message_clear_start:xla.AutotuningLogs)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.logs_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* AutotuningLogs::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .xla.AutotuningLog logs = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_logs(), ptr);
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

uint8_t* AutotuningLogs::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:xla.AutotuningLogs)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .xla.AutotuningLog logs = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_logs_size()); i < n; i++) {
    const auto& repfield = this->_internal_logs(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:xla.AutotuningLogs)
  return target;
}

size_t AutotuningLogs::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:xla.AutotuningLogs)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .xla.AutotuningLog logs = 1;
  total_size += 1UL * this->_internal_logs_size();
  for (const auto& msg : this->_impl_.logs_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData AutotuningLogs::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    AutotuningLogs::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*AutotuningLogs::GetClassData() const { return &_class_data_; }


void AutotuningLogs::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<AutotuningLogs*>(&to_msg);
  auto& from = static_cast<const AutotuningLogs&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:xla.AutotuningLogs)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.logs_.MergeFrom(from._impl_.logs_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void AutotuningLogs::CopyFrom(const AutotuningLogs& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:xla.AutotuningLogs)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool AutotuningLogs::IsInitialized() const {
  return true;
}

void AutotuningLogs::InternalSwap(AutotuningLogs* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.logs_.InternalSwap(&other->_impl_.logs_);
}

::PROTOBUF_NAMESPACE_ID::Metadata AutotuningLogs::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_xla_2fautotune_5fresults_2eproto_getter, &descriptor_table_xla_2fautotune_5fresults_2eproto_once,
      file_level_metadata_xla_2fautotune_5fresults_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace xla
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::xla::AutotuneResults_Entry*
Arena::CreateMaybeMessage< ::xla::AutotuneResults_Entry >(Arena* arena) {
  return Arena::CreateMessageInternal< ::xla::AutotuneResults_Entry >(arena);
}
template<> PROTOBUF_NOINLINE ::xla::AutotuneResults*
Arena::CreateMaybeMessage< ::xla::AutotuneResults >(Arena* arena) {
  return Arena::CreateMessageInternal< ::xla::AutotuneResults >(arena);
}
template<> PROTOBUF_NOINLINE ::xla::AutotuningLogs*
Arena::CreateMaybeMessage< ::xla::AutotuningLogs >(Arena* arena) {
  return Arena::CreateMessageInternal< ::xla::AutotuningLogs >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

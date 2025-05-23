// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow_serving/apis/input.proto

#include "tensorflow_serving/apis/input.pb.h"

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
PROTOBUF_CONSTEXPR ExampleList::ExampleList(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.examples_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ExampleListDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ExampleListDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ExampleListDefaultTypeInternal() {}
  union {
    ExampleList _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ExampleListDefaultTypeInternal _ExampleList_default_instance_;
PROTOBUF_CONSTEXPR ExampleListWithContext::ExampleListWithContext(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.examples_)*/{}
  , /*decltype(_impl_.context_)*/nullptr
  , /*decltype(_impl_._cached_size_)*/{}} {}
struct ExampleListWithContextDefaultTypeInternal {
  PROTOBUF_CONSTEXPR ExampleListWithContextDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~ExampleListWithContextDefaultTypeInternal() {}
  union {
    ExampleListWithContext _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 ExampleListWithContextDefaultTypeInternal _ExampleListWithContext_default_instance_;
PROTOBUF_CONSTEXPR Input::Input(
    ::_pbi::ConstantInitialized): _impl_{
    /*decltype(_impl_.kind_)*/{}
  , /*decltype(_impl_._cached_size_)*/{}
  , /*decltype(_impl_._oneof_case_)*/{}} {}
struct InputDefaultTypeInternal {
  PROTOBUF_CONSTEXPR InputDefaultTypeInternal()
      : _instance(::_pbi::ConstantInitialized{}) {}
  ~InputDefaultTypeInternal() {}
  union {
    Input _instance;
  };
};
PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 InputDefaultTypeInternal _Input_default_instance_;
}  // namespace serving
}  // namespace tensorflow
static ::_pb::Metadata file_level_metadata_tensorflow_5fserving_2fapis_2finput_2eproto[3];
static constexpr ::_pb::EnumDescriptor const** file_level_enum_descriptors_tensorflow_5fserving_2fapis_2finput_2eproto = nullptr;
static constexpr ::_pb::ServiceDescriptor const** file_level_service_descriptors_tensorflow_5fserving_2fapis_2finput_2eproto = nullptr;

const uint32_t TableStruct_tensorflow_5fserving_2fapis_2finput_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ExampleList, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ExampleList, _impl_.examples_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ExampleListWithContext, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ExampleListWithContext, _impl_.examples_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::ExampleListWithContext, _impl_.context_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::Input, _internal_metadata_),
  ~0u,  // no _extensions_
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::Input, _impl_._oneof_case_[0]),
  ~0u,  // no _weak_field_map_
  ~0u,  // no _inlined_string_donated_
  ::_pbi::kInvalidFieldOffsetTag,
  ::_pbi::kInvalidFieldOffsetTag,
  PROTOBUF_FIELD_OFFSET(::tensorflow::serving::Input, _impl_.kind_),
};
static const ::_pbi::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, -1, sizeof(::tensorflow::serving::ExampleList)},
  { 7, -1, -1, sizeof(::tensorflow::serving::ExampleListWithContext)},
  { 15, -1, -1, sizeof(::tensorflow::serving::Input)},
};

static const ::_pb::Message* const file_default_instances[] = {
  &::tensorflow::serving::_ExampleList_default_instance_._instance,
  &::tensorflow::serving::_ExampleListWithContext_default_instance_._instance,
  &::tensorflow::serving::_Input_default_instance_._instance,
};

const char descriptor_table_protodef_tensorflow_5fserving_2fapis_2finput_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n#tensorflow_serving/apis/input.proto\022\022t"
  "ensorflow.serving\032%tensorflow/core/examp"
  "le/example.proto\"4\n\013ExampleList\022%\n\010examp"
  "les\030\001 \003(\0132\023.tensorflow.Example\"e\n\026Exampl"
  "eListWithContext\022%\n\010examples\030\001 \003(\0132\023.ten"
  "sorflow.Example\022$\n\007context\030\002 \001(\0132\023.tenso"
  "rflow.Example\"\241\001\n\005Input\022;\n\014example_list\030"
  "\001 \001(\0132\037.tensorflow.serving.ExampleListB\002"
  "(\001H\000\022S\n\031example_list_with_context\030\002 \001(\0132"
  "*.tensorflow.serving.ExampleListWithCont"
  "extB\002(\001H\000B\006\n\004kindB\003\370\001\001b\006proto3"
  ;
static const ::_pbi::DescriptorTable* const descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2fcore_2fexample_2fexample_2eproto,
};
static ::_pbi::once_flag descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto = {
    false, false, 430, descriptor_table_protodef_tensorflow_5fserving_2fapis_2finput_2eproto,
    "tensorflow_serving/apis/input.proto",
    &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_once, descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_deps, 1, 3,
    schemas, file_default_instances, TableStruct_tensorflow_5fserving_2fapis_2finput_2eproto::offsets,
    file_level_metadata_tensorflow_5fserving_2fapis_2finput_2eproto, file_level_enum_descriptors_tensorflow_5fserving_2fapis_2finput_2eproto,
    file_level_service_descriptors_tensorflow_5fserving_2fapis_2finput_2eproto,
};
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_getter() {
  return &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto;
}

// Force running AddDescriptors() at dynamic initialization time.
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2 static ::_pbi::AddDescriptorsRunner dynamic_init_dummy_tensorflow_5fserving_2fapis_2finput_2eproto(&descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto);
namespace tensorflow {
namespace serving {

// ===================================================================

class ExampleList::_Internal {
 public:
};

void ExampleList::clear_examples() {
  _impl_.examples_.Clear();
}
ExampleList::ExampleList(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.ExampleList)
}
ExampleList::ExampleList(const ExampleList& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ExampleList* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.examples_){from._impl_.examples_}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.ExampleList)
}

inline void ExampleList::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.examples_){arena}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

ExampleList::~ExampleList() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.ExampleList)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ExampleList::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.examples_.~RepeatedPtrField();
}

void ExampleList::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ExampleList::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.ExampleList)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.examples_.Clear();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ExampleList::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .tensorflow.Example examples = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_examples(), ptr);
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

uint8_t* ExampleList::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.ExampleList)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.Example examples = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_examples_size()); i < n; i++) {
    const auto& repfield = this->_internal_examples(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.ExampleList)
  return target;
}

size_t ExampleList::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.ExampleList)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.Example examples = 1;
  total_size += 1UL * this->_internal_examples_size();
  for (const auto& msg : this->_impl_.examples_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ExampleList::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ExampleList::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ExampleList::GetClassData() const { return &_class_data_; }


void ExampleList::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ExampleList*>(&to_msg);
  auto& from = static_cast<const ExampleList&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.ExampleList)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.examples_.MergeFrom(from._impl_.examples_);
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ExampleList::CopyFrom(const ExampleList& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.ExampleList)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ExampleList::IsInitialized() const {
  return true;
}

void ExampleList::InternalSwap(ExampleList* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.examples_.InternalSwap(&other->_impl_.examples_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ExampleList::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2finput_2eproto[0]);
}

// ===================================================================

class ExampleListWithContext::_Internal {
 public:
  static const ::tensorflow::Example& context(const ExampleListWithContext* msg);
};

const ::tensorflow::Example&
ExampleListWithContext::_Internal::context(const ExampleListWithContext* msg) {
  return *msg->_impl_.context_;
}
void ExampleListWithContext::clear_examples() {
  _impl_.examples_.Clear();
}
void ExampleListWithContext::clear_context() {
  if (GetArenaForAllocation() == nullptr && _impl_.context_ != nullptr) {
    delete _impl_.context_;
  }
  _impl_.context_ = nullptr;
}
ExampleListWithContext::ExampleListWithContext(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.ExampleListWithContext)
}
ExampleListWithContext::ExampleListWithContext(const ExampleListWithContext& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  ExampleListWithContext* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.examples_){from._impl_.examples_}
    , decltype(_impl_.context_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  if (from._internal_has_context()) {
    _this->_impl_.context_ = new ::tensorflow::Example(*from._impl_.context_);
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.ExampleListWithContext)
}

inline void ExampleListWithContext::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.examples_){arena}
    , decltype(_impl_.context_){nullptr}
    , /*decltype(_impl_._cached_size_)*/{}
  };
}

ExampleListWithContext::~ExampleListWithContext() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.ExampleListWithContext)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void ExampleListWithContext::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  _impl_.examples_.~RepeatedPtrField();
  if (this != internal_default_instance()) delete _impl_.context_;
}

void ExampleListWithContext::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void ExampleListWithContext::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.ExampleListWithContext)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.examples_.Clear();
  if (GetArenaForAllocation() == nullptr && _impl_.context_ != nullptr) {
    delete _impl_.context_;
  }
  _impl_.context_ = nullptr;
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* ExampleListWithContext::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // repeated .tensorflow.Example examples = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_examples(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<10>(ptr));
        } else
          goto handle_unusual;
        continue;
      // .tensorflow.Example context = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_context(), ptr);
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

uint8_t* ExampleListWithContext::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.ExampleListWithContext)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.Example examples = 1;
  for (unsigned i = 0,
      n = static_cast<unsigned>(this->_internal_examples_size()); i < n; i++) {
    const auto& repfield = this->_internal_examples(i);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
        InternalWriteMessage(1, repfield, repfield.GetCachedSize(), target, stream);
  }

  // .tensorflow.Example context = 2;
  if (this->_internal_has_context()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(2, _Internal::context(this),
        _Internal::context(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.ExampleListWithContext)
  return target;
}

size_t ExampleListWithContext::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.ExampleListWithContext)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.Example examples = 1;
  total_size += 1UL * this->_internal_examples_size();
  for (const auto& msg : this->_impl_.examples_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // .tensorflow.Example context = 2;
  if (this->_internal_has_context()) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *_impl_.context_);
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData ExampleListWithContext::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    ExampleListWithContext::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*ExampleListWithContext::GetClassData() const { return &_class_data_; }


void ExampleListWithContext::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<ExampleListWithContext*>(&to_msg);
  auto& from = static_cast<const ExampleListWithContext&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.ExampleListWithContext)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_impl_.examples_.MergeFrom(from._impl_.examples_);
  if (from._internal_has_context()) {
    _this->_internal_mutable_context()->::tensorflow::Example::MergeFrom(
        from._internal_context());
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void ExampleListWithContext::CopyFrom(const ExampleListWithContext& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.ExampleListWithContext)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool ExampleListWithContext::IsInitialized() const {
  return true;
}

void ExampleListWithContext::InternalSwap(ExampleListWithContext* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.examples_.InternalSwap(&other->_impl_.examples_);
  swap(_impl_.context_, other->_impl_.context_);
}

::PROTOBUF_NAMESPACE_ID::Metadata ExampleListWithContext::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2finput_2eproto[1]);
}

// ===================================================================

class Input::_Internal {
 public:
  static const ::tensorflow::serving::ExampleList& example_list(const Input* msg);
  static const ::tensorflow::serving::ExampleListWithContext& example_list_with_context(const Input* msg);
};

const ::tensorflow::serving::ExampleList&
Input::_Internal::example_list(const Input* msg) {
  return *msg->_impl_.kind_.example_list_;
}
const ::tensorflow::serving::ExampleListWithContext&
Input::_Internal::example_list_with_context(const Input* msg) {
  return *msg->_impl_.kind_.example_list_with_context_;
}
void Input::set_allocated_example_list(::tensorflow::serving::ExampleList* example_list) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  clear_kind();
  if (example_list) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(example_list);
    if (message_arena != submessage_arena) {
      example_list = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, example_list, submessage_arena);
    }
    set_has_example_list();
    _impl_.kind_.example_list_ = example_list;
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.Input.example_list)
}
void Input::set_allocated_example_list_with_context(::tensorflow::serving::ExampleListWithContext* example_list_with_context) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  clear_kind();
  if (example_list_with_context) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
      ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(example_list_with_context);
    if (message_arena != submessage_arena) {
      example_list_with_context = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, example_list_with_context, submessage_arena);
    }
    set_has_example_list_with_context();
    _impl_.kind_.example_list_with_context_ = example_list_with_context;
  }
  // @@protoc_insertion_point(field_set_allocated:tensorflow.serving.Input.example_list_with_context)
}
Input::Input(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                         bool is_message_owned)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena, is_message_owned) {
  SharedCtor(arena, is_message_owned);
  // @@protoc_insertion_point(arena_constructor:tensorflow.serving.Input)
}
Input::Input(const Input& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  Input* const _this = this; (void)_this;
  new (&_impl_) Impl_{
      decltype(_impl_.kind_){}
    , /*decltype(_impl_._cached_size_)*/{}
    , /*decltype(_impl_._oneof_case_)*/{}};

  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  clear_has_kind();
  switch (from.kind_case()) {
    case kExampleList: {
      _this->_internal_mutable_example_list()->::tensorflow::serving::ExampleList::MergeFrom(
          from._internal_example_list());
      break;
    }
    case kExampleListWithContext: {
      _this->_internal_mutable_example_list_with_context()->::tensorflow::serving::ExampleListWithContext::MergeFrom(
          from._internal_example_list_with_context());
      break;
    }
    case KIND_NOT_SET: {
      break;
    }
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.serving.Input)
}

inline void Input::SharedCtor(
    ::_pb::Arena* arena, bool is_message_owned) {
  (void)arena;
  (void)is_message_owned;
  new (&_impl_) Impl_{
      decltype(_impl_.kind_){}
    , /*decltype(_impl_._cached_size_)*/{}
    , /*decltype(_impl_._oneof_case_)*/{}
  };
  clear_has_kind();
}

Input::~Input() {
  // @@protoc_insertion_point(destructor:tensorflow.serving.Input)
  if (auto *arena = _internal_metadata_.DeleteReturnArena<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>()) {
  (void)arena;
    return;
  }
  SharedDtor();
}

inline void Input::SharedDtor() {
  GOOGLE_DCHECK(GetArenaForAllocation() == nullptr);
  if (has_kind()) {
    clear_kind();
  }
}

void Input::SetCachedSize(int size) const {
  _impl_._cached_size_.Set(size);
}

void Input::clear_kind() {
// @@protoc_insertion_point(one_of_clear_start:tensorflow.serving.Input)
  switch (kind_case()) {
    case kExampleList: {
      if (GetArenaForAllocation() == nullptr) {
        delete _impl_.kind_.example_list_;
      }
      break;
    }
    case kExampleListWithContext: {
      if (GetArenaForAllocation() == nullptr) {
        delete _impl_.kind_.example_list_with_context_;
      }
      break;
    }
    case KIND_NOT_SET: {
      break;
    }
  }
  _impl_._oneof_case_[0] = KIND_NOT_SET;
}


void Input::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.serving.Input)
  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  clear_kind();
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Input::_InternalParse(const char* ptr, ::_pbi::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    uint32_t tag;
    ptr = ::_pbi::ReadTag(ptr, &tag);
    switch (tag >> 3) {
      // .tensorflow.serving.ExampleList example_list = 1 [lazy = true];
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 10)) {
          ptr = ctx->ParseMessage(_internal_mutable_example_list(), ptr);
          CHK_(ptr);
        } else
          goto handle_unusual;
        continue;
      // .tensorflow.serving.ExampleListWithContext example_list_with_context = 2 [lazy = true];
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<uint8_t>(tag) == 18)) {
          ptr = ctx->ParseMessage(_internal_mutable_example_list_with_context(), ptr);
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

uint8_t* Input::_InternalSerialize(
    uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.serving.Input)
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  // .tensorflow.serving.ExampleList example_list = 1 [lazy = true];
  if (_internal_has_example_list()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(1, _Internal::example_list(this),
        _Internal::example_list(this).GetCachedSize(), target, stream);
  }

  // .tensorflow.serving.ExampleListWithContext example_list_with_context = 2 [lazy = true];
  if (_internal_has_example_list_with_context()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(2, _Internal::example_list_with_context(this),
        _Internal::example_list_with_context(this).GetCachedSize(), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.serving.Input)
  return target;
}

size_t Input::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.serving.Input)
  size_t total_size = 0;

  uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  switch (kind_case()) {
    // .tensorflow.serving.ExampleList example_list = 1 [lazy = true];
    case kExampleList: {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *_impl_.kind_.example_list_);
      break;
    }
    // .tensorflow.serving.ExampleListWithContext example_list_with_context = 2 [lazy = true];
    case kExampleListWithContext: {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          *_impl_.kind_.example_list_with_context_);
      break;
    }
    case KIND_NOT_SET: {
      break;
    }
  }
  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}

const ::PROTOBUF_NAMESPACE_ID::Message::ClassData Input::_class_data_ = {
    ::PROTOBUF_NAMESPACE_ID::Message::CopyWithSourceCheck,
    Input::MergeImpl
};
const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*Input::GetClassData() const { return &_class_data_; }


void Input::MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg) {
  auto* const _this = static_cast<Input*>(&to_msg);
  auto& from = static_cast<const Input&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.serving.Input)
  GOOGLE_DCHECK_NE(&from, _this);
  uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  switch (from.kind_case()) {
    case kExampleList: {
      _this->_internal_mutable_example_list()->::tensorflow::serving::ExampleList::MergeFrom(
          from._internal_example_list());
      break;
    }
    case kExampleListWithContext: {
      _this->_internal_mutable_example_list_with_context()->::tensorflow::serving::ExampleListWithContext::MergeFrom(
          from._internal_example_list_with_context());
      break;
    }
    case KIND_NOT_SET: {
      break;
    }
  }
  _this->_internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
}

void Input::CopyFrom(const Input& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.serving.Input)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Input::IsInitialized() const {
  return true;
}

void Input::InternalSwap(Input* other) {
  using std::swap;
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  swap(_impl_.kind_, other->_impl_.kind_);
  swap(_impl_._oneof_case_[0], other->_impl_._oneof_case_[0]);
}

::PROTOBUF_NAMESPACE_ID::Metadata Input::GetMetadata() const {
  return ::_pbi::AssignDescriptors(
      &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_getter, &descriptor_table_tensorflow_5fserving_2fapis_2finput_2eproto_once,
      file_level_metadata_tensorflow_5fserving_2fapis_2finput_2eproto[2]);
}

// @@protoc_insertion_point(namespace_scope)
}  // namespace serving
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::serving::ExampleList*
Arena::CreateMaybeMessage< ::tensorflow::serving::ExampleList >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::ExampleList >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::serving::ExampleListWithContext*
Arena::CreateMaybeMessage< ::tensorflow::serving::ExampleListWithContext >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::ExampleListWithContext >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::serving::Input*
Arena::CreateMaybeMessage< ::tensorflow::serving::Input >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::serving::Input >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

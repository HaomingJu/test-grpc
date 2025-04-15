// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/fingerprint.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3021000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3021012 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata_lite.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
#include "tensorflow/core/framework/versions.pb.h"
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto;
namespace tensorflow {
class FingerprintDef;
struct FingerprintDefDefaultTypeInternal;
extern FingerprintDefDefaultTypeInternal _FingerprintDef_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::FingerprintDef* Arena::CreateMaybeMessage<::tensorflow::FingerprintDef>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class FingerprintDef final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.FingerprintDef) */ {
 public:
  inline FingerprintDef() : FingerprintDef(nullptr) {}
  ~FingerprintDef() override;
  explicit PROTOBUF_CONSTEXPR FingerprintDef(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  FingerprintDef(const FingerprintDef& from);
  FingerprintDef(FingerprintDef&& from) noexcept
    : FingerprintDef() {
    *this = ::std::move(from);
  }

  inline FingerprintDef& operator=(const FingerprintDef& from) {
    CopyFrom(from);
    return *this;
  }
  inline FingerprintDef& operator=(FingerprintDef&& from) noexcept {
    if (this == &from) return *this;
    if (GetOwningArena() == from.GetOwningArena()
  #ifdef PROTOBUF_FORCE_COPY_IN_MOVE
        && GetOwningArena() != nullptr
  #endif  // !PROTOBUF_FORCE_COPY_IN_MOVE
    ) {
      InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return default_instance().GetMetadata().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return default_instance().GetMetadata().reflection;
  }
  static const FingerprintDef& default_instance() {
    return *internal_default_instance();
  }
  static inline const FingerprintDef* internal_default_instance() {
    return reinterpret_cast<const FingerprintDef*>(
               &_FingerprintDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(FingerprintDef& a, FingerprintDef& b) {
    a.Swap(&b);
  }
  inline void Swap(FingerprintDef* other) {
    if (other == this) return;
  #ifdef PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() != nullptr &&
        GetOwningArena() == other->GetOwningArena()) {
   #else  // PROTOBUF_FORCE_COPY_IN_SWAP
    if (GetOwningArena() == other->GetOwningArena()) {
  #endif  // !PROTOBUF_FORCE_COPY_IN_SWAP
      InternalSwap(other);
    } else {
      ::PROTOBUF_NAMESPACE_ID::internal::GenericSwap(this, other);
    }
  }
  void UnsafeArenaSwap(FingerprintDef* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  FingerprintDef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<FingerprintDef>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const FingerprintDef& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const FingerprintDef& from) {
    FingerprintDef::MergeImpl(*this, from);
  }
  private:
  static void MergeImpl(::PROTOBUF_NAMESPACE_ID::Message& to_msg, const ::PROTOBUF_NAMESPACE_ID::Message& from_msg);
  public:
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  uint8_t* _InternalSerialize(
      uint8_t* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _impl_._cached_size_.Get(); }

  private:
  void SharedCtor(::PROTOBUF_NAMESPACE_ID::Arena* arena, bool is_message_owned);
  void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(FingerprintDef* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.FingerprintDef";
  }
  protected:
  explicit FingerprintDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kVersionFieldNumber = 6,
    kSavedModelChecksumFieldNumber = 1,
    kGraphDefProgramHashFieldNumber = 2,
    kSignatureDefHashFieldNumber = 3,
    kSavedObjectGraphHashFieldNumber = 4,
    kCheckpointHashFieldNumber = 5,
  };
  // .tensorflow.VersionDef version = 6;
  bool has_version() const;
  private:
  bool _internal_has_version() const;
  public:
  void clear_version();
  const ::tensorflow::VersionDef& version() const;
  PROTOBUF_NODISCARD ::tensorflow::VersionDef* release_version();
  ::tensorflow::VersionDef* mutable_version();
  void set_allocated_version(::tensorflow::VersionDef* version);
  private:
  const ::tensorflow::VersionDef& _internal_version() const;
  ::tensorflow::VersionDef* _internal_mutable_version();
  public:
  void unsafe_arena_set_allocated_version(
      ::tensorflow::VersionDef* version);
  ::tensorflow::VersionDef* unsafe_arena_release_version();

  // uint64 saved_model_checksum = 1;
  void clear_saved_model_checksum();
  uint64_t saved_model_checksum() const;
  void set_saved_model_checksum(uint64_t value);
  private:
  uint64_t _internal_saved_model_checksum() const;
  void _internal_set_saved_model_checksum(uint64_t value);
  public:

  // uint64 graph_def_program_hash = 2;
  void clear_graph_def_program_hash();
  uint64_t graph_def_program_hash() const;
  void set_graph_def_program_hash(uint64_t value);
  private:
  uint64_t _internal_graph_def_program_hash() const;
  void _internal_set_graph_def_program_hash(uint64_t value);
  public:

  // uint64 signature_def_hash = 3;
  void clear_signature_def_hash();
  uint64_t signature_def_hash() const;
  void set_signature_def_hash(uint64_t value);
  private:
  uint64_t _internal_signature_def_hash() const;
  void _internal_set_signature_def_hash(uint64_t value);
  public:

  // uint64 saved_object_graph_hash = 4;
  void clear_saved_object_graph_hash();
  uint64_t saved_object_graph_hash() const;
  void set_saved_object_graph_hash(uint64_t value);
  private:
  uint64_t _internal_saved_object_graph_hash() const;
  void _internal_set_saved_object_graph_hash(uint64_t value);
  public:

  // uint64 checkpoint_hash = 5;
  void clear_checkpoint_hash();
  uint64_t checkpoint_hash() const;
  void set_checkpoint_hash(uint64_t value);
  private:
  uint64_t _internal_checkpoint_hash() const;
  void _internal_set_checkpoint_hash(uint64_t value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.FingerprintDef)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::tensorflow::VersionDef* version_;
    uint64_t saved_model_checksum_;
    uint64_t graph_def_program_hash_;
    uint64_t signature_def_hash_;
    uint64_t saved_object_graph_hash_;
    uint64_t checkpoint_hash_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// FingerprintDef

// uint64 saved_model_checksum = 1;
inline void FingerprintDef::clear_saved_model_checksum() {
  _impl_.saved_model_checksum_ = uint64_t{0u};
}
inline uint64_t FingerprintDef::_internal_saved_model_checksum() const {
  return _impl_.saved_model_checksum_;
}
inline uint64_t FingerprintDef::saved_model_checksum() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.saved_model_checksum)
  return _internal_saved_model_checksum();
}
inline void FingerprintDef::_internal_set_saved_model_checksum(uint64_t value) {
  
  _impl_.saved_model_checksum_ = value;
}
inline void FingerprintDef::set_saved_model_checksum(uint64_t value) {
  _internal_set_saved_model_checksum(value);
  // @@protoc_insertion_point(field_set:tensorflow.FingerprintDef.saved_model_checksum)
}

// uint64 graph_def_program_hash = 2;
inline void FingerprintDef::clear_graph_def_program_hash() {
  _impl_.graph_def_program_hash_ = uint64_t{0u};
}
inline uint64_t FingerprintDef::_internal_graph_def_program_hash() const {
  return _impl_.graph_def_program_hash_;
}
inline uint64_t FingerprintDef::graph_def_program_hash() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.graph_def_program_hash)
  return _internal_graph_def_program_hash();
}
inline void FingerprintDef::_internal_set_graph_def_program_hash(uint64_t value) {
  
  _impl_.graph_def_program_hash_ = value;
}
inline void FingerprintDef::set_graph_def_program_hash(uint64_t value) {
  _internal_set_graph_def_program_hash(value);
  // @@protoc_insertion_point(field_set:tensorflow.FingerprintDef.graph_def_program_hash)
}

// uint64 signature_def_hash = 3;
inline void FingerprintDef::clear_signature_def_hash() {
  _impl_.signature_def_hash_ = uint64_t{0u};
}
inline uint64_t FingerprintDef::_internal_signature_def_hash() const {
  return _impl_.signature_def_hash_;
}
inline uint64_t FingerprintDef::signature_def_hash() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.signature_def_hash)
  return _internal_signature_def_hash();
}
inline void FingerprintDef::_internal_set_signature_def_hash(uint64_t value) {
  
  _impl_.signature_def_hash_ = value;
}
inline void FingerprintDef::set_signature_def_hash(uint64_t value) {
  _internal_set_signature_def_hash(value);
  // @@protoc_insertion_point(field_set:tensorflow.FingerprintDef.signature_def_hash)
}

// uint64 saved_object_graph_hash = 4;
inline void FingerprintDef::clear_saved_object_graph_hash() {
  _impl_.saved_object_graph_hash_ = uint64_t{0u};
}
inline uint64_t FingerprintDef::_internal_saved_object_graph_hash() const {
  return _impl_.saved_object_graph_hash_;
}
inline uint64_t FingerprintDef::saved_object_graph_hash() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.saved_object_graph_hash)
  return _internal_saved_object_graph_hash();
}
inline void FingerprintDef::_internal_set_saved_object_graph_hash(uint64_t value) {
  
  _impl_.saved_object_graph_hash_ = value;
}
inline void FingerprintDef::set_saved_object_graph_hash(uint64_t value) {
  _internal_set_saved_object_graph_hash(value);
  // @@protoc_insertion_point(field_set:tensorflow.FingerprintDef.saved_object_graph_hash)
}

// uint64 checkpoint_hash = 5;
inline void FingerprintDef::clear_checkpoint_hash() {
  _impl_.checkpoint_hash_ = uint64_t{0u};
}
inline uint64_t FingerprintDef::_internal_checkpoint_hash() const {
  return _impl_.checkpoint_hash_;
}
inline uint64_t FingerprintDef::checkpoint_hash() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.checkpoint_hash)
  return _internal_checkpoint_hash();
}
inline void FingerprintDef::_internal_set_checkpoint_hash(uint64_t value) {
  
  _impl_.checkpoint_hash_ = value;
}
inline void FingerprintDef::set_checkpoint_hash(uint64_t value) {
  _internal_set_checkpoint_hash(value);
  // @@protoc_insertion_point(field_set:tensorflow.FingerprintDef.checkpoint_hash)
}

// .tensorflow.VersionDef version = 6;
inline bool FingerprintDef::_internal_has_version() const {
  return this != internal_default_instance() && _impl_.version_ != nullptr;
}
inline bool FingerprintDef::has_version() const {
  return _internal_has_version();
}
inline const ::tensorflow::VersionDef& FingerprintDef::_internal_version() const {
  const ::tensorflow::VersionDef* p = _impl_.version_;
  return p != nullptr ? *p : reinterpret_cast<const ::tensorflow::VersionDef&>(
      ::tensorflow::_VersionDef_default_instance_);
}
inline const ::tensorflow::VersionDef& FingerprintDef::version() const {
  // @@protoc_insertion_point(field_get:tensorflow.FingerprintDef.version)
  return _internal_version();
}
inline void FingerprintDef::unsafe_arena_set_allocated_version(
    ::tensorflow::VersionDef* version) {
  if (GetArenaForAllocation() == nullptr) {
    delete reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(_impl_.version_);
  }
  _impl_.version_ = version;
  if (version) {
    
  } else {
    
  }
  // @@protoc_insertion_point(field_unsafe_arena_set_allocated:tensorflow.FingerprintDef.version)
}
inline ::tensorflow::VersionDef* FingerprintDef::release_version() {
  
  ::tensorflow::VersionDef* temp = _impl_.version_;
  _impl_.version_ = nullptr;
#ifdef PROTOBUF_FORCE_COPY_IN_RELEASE
  auto* old =  reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(temp);
  temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  if (GetArenaForAllocation() == nullptr) { delete old; }
#else  // PROTOBUF_FORCE_COPY_IN_RELEASE
  if (GetArenaForAllocation() != nullptr) {
    temp = ::PROTOBUF_NAMESPACE_ID::internal::DuplicateIfNonNull(temp);
  }
#endif  // !PROTOBUF_FORCE_COPY_IN_RELEASE
  return temp;
}
inline ::tensorflow::VersionDef* FingerprintDef::unsafe_arena_release_version() {
  // @@protoc_insertion_point(field_release:tensorflow.FingerprintDef.version)
  
  ::tensorflow::VersionDef* temp = _impl_.version_;
  _impl_.version_ = nullptr;
  return temp;
}
inline ::tensorflow::VersionDef* FingerprintDef::_internal_mutable_version() {
  
  if (_impl_.version_ == nullptr) {
    auto* p = CreateMaybeMessage<::tensorflow::VersionDef>(GetArenaForAllocation());
    _impl_.version_ = p;
  }
  return _impl_.version_;
}
inline ::tensorflow::VersionDef* FingerprintDef::mutable_version() {
  ::tensorflow::VersionDef* _msg = _internal_mutable_version();
  // @@protoc_insertion_point(field_mutable:tensorflow.FingerprintDef.version)
  return _msg;
}
inline void FingerprintDef::set_allocated_version(::tensorflow::VersionDef* version) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaForAllocation();
  if (message_arena == nullptr) {
    delete reinterpret_cast< ::PROTOBUF_NAMESPACE_ID::MessageLite*>(_impl_.version_);
  }
  if (version) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena =
        ::PROTOBUF_NAMESPACE_ID::Arena::InternalGetOwningArena(
                reinterpret_cast<::PROTOBUF_NAMESPACE_ID::MessageLite*>(version));
    if (message_arena != submessage_arena) {
      version = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, version, submessage_arena);
    }
    
  } else {
    
  }
  _impl_.version_ = version;
  // @@protoc_insertion_point(field_set_allocated:tensorflow.FingerprintDef.version)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2ffingerprint_2eproto

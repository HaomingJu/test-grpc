// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: xla/service/buffer_assignment.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_xla_2fservice_2fbuffer_5fassignment_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_xla_2fservice_2fbuffer_5fassignment_2eproto

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
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_xla_2fservice_2fbuffer_5fassignment_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_xla_2fservice_2fbuffer_5fassignment_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_xla_2fservice_2fbuffer_5fassignment_2eproto;
namespace xla {
namespace buffer_assignment {
class BufferIsolationConfig;
struct BufferIsolationConfigDefaultTypeInternal;
extern BufferIsolationConfigDefaultTypeInternal _BufferIsolationConfig_default_instance_;
}  // namespace buffer_assignment
}  // namespace xla
PROTOBUF_NAMESPACE_OPEN
template<> ::xla::buffer_assignment::BufferIsolationConfig* Arena::CreateMaybeMessage<::xla::buffer_assignment::BufferIsolationConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace xla {
namespace buffer_assignment {

// ===================================================================

class BufferIsolationConfig final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:xla.buffer_assignment.BufferIsolationConfig) */ {
 public:
  inline BufferIsolationConfig() : BufferIsolationConfig(nullptr) {}
  ~BufferIsolationConfig() override;
  explicit PROTOBUF_CONSTEXPR BufferIsolationConfig(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  BufferIsolationConfig(const BufferIsolationConfig& from);
  BufferIsolationConfig(BufferIsolationConfig&& from) noexcept
    : BufferIsolationConfig() {
    *this = ::std::move(from);
  }

  inline BufferIsolationConfig& operator=(const BufferIsolationConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline BufferIsolationConfig& operator=(BufferIsolationConfig&& from) noexcept {
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
  static const BufferIsolationConfig& default_instance() {
    return *internal_default_instance();
  }
  static inline const BufferIsolationConfig* internal_default_instance() {
    return reinterpret_cast<const BufferIsolationConfig*>(
               &_BufferIsolationConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(BufferIsolationConfig& a, BufferIsolationConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(BufferIsolationConfig* other) {
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
  void UnsafeArenaSwap(BufferIsolationConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  BufferIsolationConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<BufferIsolationConfig>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const BufferIsolationConfig& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const BufferIsolationConfig& from) {
    BufferIsolationConfig::MergeImpl(*this, from);
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
  void InternalSwap(BufferIsolationConfig* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "xla.buffer_assignment.BufferIsolationConfig";
  }
  protected:
  explicit BufferIsolationConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kIsolationColorsFieldNumber = 5,
    kBaseOffsetBytesFieldNumber = 1,
    kIsolationFuelFieldNumber = 2,
    kIsolationPaddingBytesFieldNumber = 3,
    kIsolationOrderSaltFieldNumber = 4,
  };
  // repeated int32 isolation_colors = 5;
  int isolation_colors_size() const;
  private:
  int _internal_isolation_colors_size() const;
  public:
  void clear_isolation_colors();
  private:
  int32_t _internal_isolation_colors(int index) const;
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >&
      _internal_isolation_colors() const;
  void _internal_add_isolation_colors(int32_t value);
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >*
      _internal_mutable_isolation_colors();
  public:
  int32_t isolation_colors(int index) const;
  void set_isolation_colors(int index, int32_t value);
  void add_isolation_colors(int32_t value);
  const ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >&
      isolation_colors() const;
  ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >*
      mutable_isolation_colors();

  // int64 base_offset_bytes = 1;
  void clear_base_offset_bytes();
  int64_t base_offset_bytes() const;
  void set_base_offset_bytes(int64_t value);
  private:
  int64_t _internal_base_offset_bytes() const;
  void _internal_set_base_offset_bytes(int64_t value);
  public:

  // int64 isolation_fuel = 2;
  void clear_isolation_fuel();
  int64_t isolation_fuel() const;
  void set_isolation_fuel(int64_t value);
  private:
  int64_t _internal_isolation_fuel() const;
  void _internal_set_isolation_fuel(int64_t value);
  public:

  // int64 isolation_padding_bytes = 3;
  void clear_isolation_padding_bytes();
  int64_t isolation_padding_bytes() const;
  void set_isolation_padding_bytes(int64_t value);
  private:
  int64_t _internal_isolation_padding_bytes() const;
  void _internal_set_isolation_padding_bytes(int64_t value);
  public:

  // uint64 isolation_order_salt = 4;
  void clear_isolation_order_salt();
  uint64_t isolation_order_salt() const;
  void set_isolation_order_salt(uint64_t value);
  private:
  uint64_t _internal_isolation_order_salt() const;
  void _internal_set_isolation_order_salt(uint64_t value);
  public:

  // @@protoc_insertion_point(class_scope:xla.buffer_assignment.BufferIsolationConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t > isolation_colors_;
    mutable std::atomic<int> _isolation_colors_cached_byte_size_;
    int64_t base_offset_bytes_;
    int64_t isolation_fuel_;
    int64_t isolation_padding_bytes_;
    uint64_t isolation_order_salt_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_xla_2fservice_2fbuffer_5fassignment_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// BufferIsolationConfig

// int64 base_offset_bytes = 1;
inline void BufferIsolationConfig::clear_base_offset_bytes() {
  _impl_.base_offset_bytes_ = int64_t{0};
}
inline int64_t BufferIsolationConfig::_internal_base_offset_bytes() const {
  return _impl_.base_offset_bytes_;
}
inline int64_t BufferIsolationConfig::base_offset_bytes() const {
  // @@protoc_insertion_point(field_get:xla.buffer_assignment.BufferIsolationConfig.base_offset_bytes)
  return _internal_base_offset_bytes();
}
inline void BufferIsolationConfig::_internal_set_base_offset_bytes(int64_t value) {
  
  _impl_.base_offset_bytes_ = value;
}
inline void BufferIsolationConfig::set_base_offset_bytes(int64_t value) {
  _internal_set_base_offset_bytes(value);
  // @@protoc_insertion_point(field_set:xla.buffer_assignment.BufferIsolationConfig.base_offset_bytes)
}

// int64 isolation_fuel = 2;
inline void BufferIsolationConfig::clear_isolation_fuel() {
  _impl_.isolation_fuel_ = int64_t{0};
}
inline int64_t BufferIsolationConfig::_internal_isolation_fuel() const {
  return _impl_.isolation_fuel_;
}
inline int64_t BufferIsolationConfig::isolation_fuel() const {
  // @@protoc_insertion_point(field_get:xla.buffer_assignment.BufferIsolationConfig.isolation_fuel)
  return _internal_isolation_fuel();
}
inline void BufferIsolationConfig::_internal_set_isolation_fuel(int64_t value) {
  
  _impl_.isolation_fuel_ = value;
}
inline void BufferIsolationConfig::set_isolation_fuel(int64_t value) {
  _internal_set_isolation_fuel(value);
  // @@protoc_insertion_point(field_set:xla.buffer_assignment.BufferIsolationConfig.isolation_fuel)
}

// int64 isolation_padding_bytes = 3;
inline void BufferIsolationConfig::clear_isolation_padding_bytes() {
  _impl_.isolation_padding_bytes_ = int64_t{0};
}
inline int64_t BufferIsolationConfig::_internal_isolation_padding_bytes() const {
  return _impl_.isolation_padding_bytes_;
}
inline int64_t BufferIsolationConfig::isolation_padding_bytes() const {
  // @@protoc_insertion_point(field_get:xla.buffer_assignment.BufferIsolationConfig.isolation_padding_bytes)
  return _internal_isolation_padding_bytes();
}
inline void BufferIsolationConfig::_internal_set_isolation_padding_bytes(int64_t value) {
  
  _impl_.isolation_padding_bytes_ = value;
}
inline void BufferIsolationConfig::set_isolation_padding_bytes(int64_t value) {
  _internal_set_isolation_padding_bytes(value);
  // @@protoc_insertion_point(field_set:xla.buffer_assignment.BufferIsolationConfig.isolation_padding_bytes)
}

// uint64 isolation_order_salt = 4;
inline void BufferIsolationConfig::clear_isolation_order_salt() {
  _impl_.isolation_order_salt_ = uint64_t{0u};
}
inline uint64_t BufferIsolationConfig::_internal_isolation_order_salt() const {
  return _impl_.isolation_order_salt_;
}
inline uint64_t BufferIsolationConfig::isolation_order_salt() const {
  // @@protoc_insertion_point(field_get:xla.buffer_assignment.BufferIsolationConfig.isolation_order_salt)
  return _internal_isolation_order_salt();
}
inline void BufferIsolationConfig::_internal_set_isolation_order_salt(uint64_t value) {
  
  _impl_.isolation_order_salt_ = value;
}
inline void BufferIsolationConfig::set_isolation_order_salt(uint64_t value) {
  _internal_set_isolation_order_salt(value);
  // @@protoc_insertion_point(field_set:xla.buffer_assignment.BufferIsolationConfig.isolation_order_salt)
}

// repeated int32 isolation_colors = 5;
inline int BufferIsolationConfig::_internal_isolation_colors_size() const {
  return _impl_.isolation_colors_.size();
}
inline int BufferIsolationConfig::isolation_colors_size() const {
  return _internal_isolation_colors_size();
}
inline void BufferIsolationConfig::clear_isolation_colors() {
  _impl_.isolation_colors_.Clear();
}
inline int32_t BufferIsolationConfig::_internal_isolation_colors(int index) const {
  return _impl_.isolation_colors_.Get(index);
}
inline int32_t BufferIsolationConfig::isolation_colors(int index) const {
  // @@protoc_insertion_point(field_get:xla.buffer_assignment.BufferIsolationConfig.isolation_colors)
  return _internal_isolation_colors(index);
}
inline void BufferIsolationConfig::set_isolation_colors(int index, int32_t value) {
  _impl_.isolation_colors_.Set(index, value);
  // @@protoc_insertion_point(field_set:xla.buffer_assignment.BufferIsolationConfig.isolation_colors)
}
inline void BufferIsolationConfig::_internal_add_isolation_colors(int32_t value) {
  _impl_.isolation_colors_.Add(value);
}
inline void BufferIsolationConfig::add_isolation_colors(int32_t value) {
  _internal_add_isolation_colors(value);
  // @@protoc_insertion_point(field_add:xla.buffer_assignment.BufferIsolationConfig.isolation_colors)
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >&
BufferIsolationConfig::_internal_isolation_colors() const {
  return _impl_.isolation_colors_;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >&
BufferIsolationConfig::isolation_colors() const {
  // @@protoc_insertion_point(field_list:xla.buffer_assignment.BufferIsolationConfig.isolation_colors)
  return _internal_isolation_colors();
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >*
BufferIsolationConfig::_internal_mutable_isolation_colors() {
  return &_impl_.isolation_colors_;
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedField< int32_t >*
BufferIsolationConfig::mutable_isolation_colors() {
  // @@protoc_insertion_point(field_mutable_list:xla.buffer_assignment.BufferIsolationConfig.isolation_colors)
  return _internal_mutable_isolation_colors();
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace buffer_assignment
}  // namespace xla

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_xla_2fservice_2fbuffer_5fassignment_2eproto

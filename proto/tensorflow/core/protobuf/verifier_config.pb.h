// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/verifier_config.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto

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
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto;
namespace tensorflow {
class VerifierConfig;
struct VerifierConfigDefaultTypeInternal;
extern VerifierConfigDefaultTypeInternal _VerifierConfig_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::VerifierConfig* Arena::CreateMaybeMessage<::tensorflow::VerifierConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

enum VerifierConfig_Toggle : int {
  VerifierConfig_Toggle_DEFAULT = 0,
  VerifierConfig_Toggle_ON = 1,
  VerifierConfig_Toggle_OFF = 2,
  VerifierConfig_Toggle_VerifierConfig_Toggle_INT_MIN_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::min(),
  VerifierConfig_Toggle_VerifierConfig_Toggle_INT_MAX_SENTINEL_DO_NOT_USE_ = std::numeric_limits<int32_t>::max()
};
bool VerifierConfig_Toggle_IsValid(int value);
constexpr VerifierConfig_Toggle VerifierConfig_Toggle_Toggle_MIN = VerifierConfig_Toggle_DEFAULT;
constexpr VerifierConfig_Toggle VerifierConfig_Toggle_Toggle_MAX = VerifierConfig_Toggle_OFF;
constexpr int VerifierConfig_Toggle_Toggle_ARRAYSIZE = VerifierConfig_Toggle_Toggle_MAX + 1;

const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* VerifierConfig_Toggle_descriptor();
template<typename T>
inline const std::string& VerifierConfig_Toggle_Name(T enum_t_value) {
  static_assert(::std::is_same<T, VerifierConfig_Toggle>::value ||
    ::std::is_integral<T>::value,
    "Incorrect type passed to function VerifierConfig_Toggle_Name.");
  return ::PROTOBUF_NAMESPACE_ID::internal::NameOfEnum(
    VerifierConfig_Toggle_descriptor(), enum_t_value);
}
inline bool VerifierConfig_Toggle_Parse(
    ::PROTOBUF_NAMESPACE_ID::ConstStringParam name, VerifierConfig_Toggle* value) {
  return ::PROTOBUF_NAMESPACE_ID::internal::ParseNamedEnum<VerifierConfig_Toggle>(
    VerifierConfig_Toggle_descriptor(), name, value);
}
// ===================================================================

class VerifierConfig final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.VerifierConfig) */ {
 public:
  inline VerifierConfig() : VerifierConfig(nullptr) {}
  ~VerifierConfig() override;
  explicit PROTOBUF_CONSTEXPR VerifierConfig(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  VerifierConfig(const VerifierConfig& from);
  VerifierConfig(VerifierConfig&& from) noexcept
    : VerifierConfig() {
    *this = ::std::move(from);
  }

  inline VerifierConfig& operator=(const VerifierConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline VerifierConfig& operator=(VerifierConfig&& from) noexcept {
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
  static const VerifierConfig& default_instance() {
    return *internal_default_instance();
  }
  static inline const VerifierConfig* internal_default_instance() {
    return reinterpret_cast<const VerifierConfig*>(
               &_VerifierConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(VerifierConfig& a, VerifierConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(VerifierConfig* other) {
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
  void UnsafeArenaSwap(VerifierConfig* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  VerifierConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<VerifierConfig>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const VerifierConfig& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const VerifierConfig& from) {
    VerifierConfig::MergeImpl(*this, from);
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
  void InternalSwap(VerifierConfig* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.VerifierConfig";
  }
  protected:
  explicit VerifierConfig(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  typedef VerifierConfig_Toggle Toggle;
  static constexpr Toggle DEFAULT =
    VerifierConfig_Toggle_DEFAULT;
  static constexpr Toggle ON =
    VerifierConfig_Toggle_ON;
  static constexpr Toggle OFF =
    VerifierConfig_Toggle_OFF;
  static inline bool Toggle_IsValid(int value) {
    return VerifierConfig_Toggle_IsValid(value);
  }
  static constexpr Toggle Toggle_MIN =
    VerifierConfig_Toggle_Toggle_MIN;
  static constexpr Toggle Toggle_MAX =
    VerifierConfig_Toggle_Toggle_MAX;
  static constexpr int Toggle_ARRAYSIZE =
    VerifierConfig_Toggle_Toggle_ARRAYSIZE;
  static inline const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor*
  Toggle_descriptor() {
    return VerifierConfig_Toggle_descriptor();
  }
  template<typename T>
  static inline const std::string& Toggle_Name(T enum_t_value) {
    static_assert(::std::is_same<T, Toggle>::value ||
      ::std::is_integral<T>::value,
      "Incorrect type passed to function Toggle_Name.");
    return VerifierConfig_Toggle_Name(enum_t_value);
  }
  static inline bool Toggle_Parse(::PROTOBUF_NAMESPACE_ID::ConstStringParam name,
      Toggle* value) {
    return VerifierConfig_Toggle_Parse(name, value);
  }

  // accessors -------------------------------------------------------

  enum : int {
    kVerificationTimeoutInMsFieldNumber = 1,
    kStructureVerifierFieldNumber = 2,
  };
  // int64 verification_timeout_in_ms = 1;
  void clear_verification_timeout_in_ms();
  int64_t verification_timeout_in_ms() const;
  void set_verification_timeout_in_ms(int64_t value);
  private:
  int64_t _internal_verification_timeout_in_ms() const;
  void _internal_set_verification_timeout_in_ms(int64_t value);
  public:

  // .tensorflow.VerifierConfig.Toggle structure_verifier = 2;
  void clear_structure_verifier();
  ::tensorflow::VerifierConfig_Toggle structure_verifier() const;
  void set_structure_verifier(::tensorflow::VerifierConfig_Toggle value);
  private:
  ::tensorflow::VerifierConfig_Toggle _internal_structure_verifier() const;
  void _internal_set_structure_verifier(::tensorflow::VerifierConfig_Toggle value);
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.VerifierConfig)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    int64_t verification_timeout_in_ms_;
    int structure_verifier_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// VerifierConfig

// int64 verification_timeout_in_ms = 1;
inline void VerifierConfig::clear_verification_timeout_in_ms() {
  _impl_.verification_timeout_in_ms_ = int64_t{0};
}
inline int64_t VerifierConfig::_internal_verification_timeout_in_ms() const {
  return _impl_.verification_timeout_in_ms_;
}
inline int64_t VerifierConfig::verification_timeout_in_ms() const {
  // @@protoc_insertion_point(field_get:tensorflow.VerifierConfig.verification_timeout_in_ms)
  return _internal_verification_timeout_in_ms();
}
inline void VerifierConfig::_internal_set_verification_timeout_in_ms(int64_t value) {
  
  _impl_.verification_timeout_in_ms_ = value;
}
inline void VerifierConfig::set_verification_timeout_in_ms(int64_t value) {
  _internal_set_verification_timeout_in_ms(value);
  // @@protoc_insertion_point(field_set:tensorflow.VerifierConfig.verification_timeout_in_ms)
}

// .tensorflow.VerifierConfig.Toggle structure_verifier = 2;
inline void VerifierConfig::clear_structure_verifier() {
  _impl_.structure_verifier_ = 0;
}
inline ::tensorflow::VerifierConfig_Toggle VerifierConfig::_internal_structure_verifier() const {
  return static_cast< ::tensorflow::VerifierConfig_Toggle >(_impl_.structure_verifier_);
}
inline ::tensorflow::VerifierConfig_Toggle VerifierConfig::structure_verifier() const {
  // @@protoc_insertion_point(field_get:tensorflow.VerifierConfig.structure_verifier)
  return _internal_structure_verifier();
}
inline void VerifierConfig::_internal_set_structure_verifier(::tensorflow::VerifierConfig_Toggle value) {
  
  _impl_.structure_verifier_ = value;
}
inline void VerifierConfig::set_structure_verifier(::tensorflow::VerifierConfig_Toggle value) {
  _internal_set_structure_verifier(value);
  // @@protoc_insertion_point(field_set:tensorflow.VerifierConfig.structure_verifier)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__

// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

PROTOBUF_NAMESPACE_OPEN

template <> struct is_proto_enum< ::tensorflow::VerifierConfig_Toggle> : ::std::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::tensorflow::VerifierConfig_Toggle>() {
  return ::tensorflow::VerifierConfig_Toggle_descriptor();
}

PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fverifier_5fconfig_2eproto

// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/cluster.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto

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
#include <google/protobuf/map.h>  // IWYU pragma: export
#include <google/protobuf/map_entry.h>
#include <google/protobuf/map_field_inl.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto {
  static const uint32_t offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto;
namespace tensorflow {
class ClusterDef;
struct ClusterDefDefaultTypeInternal;
extern ClusterDefDefaultTypeInternal _ClusterDef_default_instance_;
class JobDef;
struct JobDefDefaultTypeInternal;
extern JobDefDefaultTypeInternal _JobDef_default_instance_;
class JobDef_TasksEntry_DoNotUse;
struct JobDef_TasksEntry_DoNotUseDefaultTypeInternal;
extern JobDef_TasksEntry_DoNotUseDefaultTypeInternal _JobDef_TasksEntry_DoNotUse_default_instance_;
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> ::tensorflow::ClusterDef* Arena::CreateMaybeMessage<::tensorflow::ClusterDef>(Arena*);
template<> ::tensorflow::JobDef* Arena::CreateMaybeMessage<::tensorflow::JobDef>(Arena*);
template<> ::tensorflow::JobDef_TasksEntry_DoNotUse* Arena::CreateMaybeMessage<::tensorflow::JobDef_TasksEntry_DoNotUse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace tensorflow {

// ===================================================================

class JobDef_TasksEntry_DoNotUse : public ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<JobDef_TasksEntry_DoNotUse, 
    int32_t, std::string,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING> {
public:
  typedef ::PROTOBUF_NAMESPACE_ID::internal::MapEntry<JobDef_TasksEntry_DoNotUse, 
    int32_t, std::string,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING> SuperType;
  JobDef_TasksEntry_DoNotUse();
  explicit PROTOBUF_CONSTEXPR JobDef_TasksEntry_DoNotUse(
      ::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);
  explicit JobDef_TasksEntry_DoNotUse(::PROTOBUF_NAMESPACE_ID::Arena* arena);
  void MergeFrom(const JobDef_TasksEntry_DoNotUse& other);
  static const JobDef_TasksEntry_DoNotUse* internal_default_instance() { return reinterpret_cast<const JobDef_TasksEntry_DoNotUse*>(&_JobDef_TasksEntry_DoNotUse_default_instance_); }
  static bool ValidateKey(void*) { return true; }
  static bool ValidateValue(std::string* s) {
    return ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(s->data(), static_cast<int>(s->size()), ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE, "tensorflow.JobDef.TasksEntry.value");
 }
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto;
};

// -------------------------------------------------------------------

class JobDef final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.JobDef) */ {
 public:
  inline JobDef() : JobDef(nullptr) {}
  ~JobDef() override;
  explicit PROTOBUF_CONSTEXPR JobDef(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  JobDef(const JobDef& from);
  JobDef(JobDef&& from) noexcept
    : JobDef() {
    *this = ::std::move(from);
  }

  inline JobDef& operator=(const JobDef& from) {
    CopyFrom(from);
    return *this;
  }
  inline JobDef& operator=(JobDef&& from) noexcept {
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
  static const JobDef& default_instance() {
    return *internal_default_instance();
  }
  static inline const JobDef* internal_default_instance() {
    return reinterpret_cast<const JobDef*>(
               &_JobDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(JobDef& a, JobDef& b) {
    a.Swap(&b);
  }
  inline void Swap(JobDef* other) {
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
  void UnsafeArenaSwap(JobDef* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  JobDef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<JobDef>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const JobDef& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const JobDef& from) {
    JobDef::MergeImpl(*this, from);
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
  void InternalSwap(JobDef* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.JobDef";
  }
  protected:
  explicit JobDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  private:
  static void ArenaDtor(void* object);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------


  // accessors -------------------------------------------------------

  enum : int {
    kTasksFieldNumber = 2,
    kNameFieldNumber = 1,
  };
  // map<int32, string> tasks = 2;
  int tasks_size() const;
  private:
  int _internal_tasks_size() const;
  public:
  void clear_tasks();
  private:
  const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >&
      _internal_tasks() const;
  ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >*
      _internal_mutable_tasks();
  public:
  const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >&
      tasks() const;
  ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >*
      mutable_tasks();

  // string name = 1;
  void clear_name();
  const std::string& name() const;
  template <typename ArgT0 = const std::string&, typename... ArgT>
  void set_name(ArgT0&& arg0, ArgT... args);
  std::string* mutable_name();
  PROTOBUF_NODISCARD std::string* release_name();
  void set_allocated_name(std::string* name);
  private:
  const std::string& _internal_name() const;
  inline PROTOBUF_ALWAYS_INLINE void _internal_set_name(const std::string& value);
  std::string* _internal_mutable_name();
  public:

  // @@protoc_insertion_point(class_scope:tensorflow.JobDef)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::internal::MapField<
        JobDef_TasksEntry_DoNotUse,
        int32_t, std::string,
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32,
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_STRING> tasks_;
    ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto;
};
// -------------------------------------------------------------------

class ClusterDef final :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:tensorflow.ClusterDef) */ {
 public:
  inline ClusterDef() : ClusterDef(nullptr) {}
  ~ClusterDef() override;
  explicit PROTOBUF_CONSTEXPR ClusterDef(::PROTOBUF_NAMESPACE_ID::internal::ConstantInitialized);

  ClusterDef(const ClusterDef& from);
  ClusterDef(ClusterDef&& from) noexcept
    : ClusterDef() {
    *this = ::std::move(from);
  }

  inline ClusterDef& operator=(const ClusterDef& from) {
    CopyFrom(from);
    return *this;
  }
  inline ClusterDef& operator=(ClusterDef&& from) noexcept {
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
  static const ClusterDef& default_instance() {
    return *internal_default_instance();
  }
  static inline const ClusterDef* internal_default_instance() {
    return reinterpret_cast<const ClusterDef*>(
               &_ClusterDef_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(ClusterDef& a, ClusterDef& b) {
    a.Swap(&b);
  }
  inline void Swap(ClusterDef* other) {
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
  void UnsafeArenaSwap(ClusterDef* other) {
    if (other == this) return;
    GOOGLE_DCHECK(GetOwningArena() == other->GetOwningArena());
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  ClusterDef* New(::PROTOBUF_NAMESPACE_ID::Arena* arena = nullptr) const final {
    return CreateMaybeMessage<ClusterDef>(arena);
  }
  using ::PROTOBUF_NAMESPACE_ID::Message::CopyFrom;
  void CopyFrom(const ClusterDef& from);
  using ::PROTOBUF_NAMESPACE_ID::Message::MergeFrom;
  void MergeFrom( const ClusterDef& from) {
    ClusterDef::MergeImpl(*this, from);
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
  void InternalSwap(ClusterDef* other);

  private:
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "tensorflow.ClusterDef";
  }
  protected:
  explicit ClusterDef(::PROTOBUF_NAMESPACE_ID::Arena* arena,
                       bool is_message_owned = false);
  public:

  static const ClassData _class_data_;
  const ::PROTOBUF_NAMESPACE_ID::Message::ClassData*GetClassData() const final;

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kJobFieldNumber = 1,
  };
  // repeated .tensorflow.JobDef job = 1;
  int job_size() const;
  private:
  int _internal_job_size() const;
  public:
  void clear_job();
  ::tensorflow::JobDef* mutable_job(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::JobDef >*
      mutable_job();
  private:
  const ::tensorflow::JobDef& _internal_job(int index) const;
  ::tensorflow::JobDef* _internal_add_job();
  public:
  const ::tensorflow::JobDef& job(int index) const;
  ::tensorflow::JobDef* add_job();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::JobDef >&
      job() const;

  // @@protoc_insertion_point(class_scope:tensorflow.ClusterDef)
 private:
  class _Internal;

  template <typename T> friend class ::PROTOBUF_NAMESPACE_ID::Arena::InternalHelper;
  typedef void InternalArenaConstructable_;
  typedef void DestructorSkippable_;
  struct Impl_ {
    ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::JobDef > job_;
    mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  };
  union { Impl_ _impl_; };
  friend struct ::TableStruct_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// -------------------------------------------------------------------

// JobDef

// string name = 1;
inline void JobDef::clear_name() {
  _impl_.name_.ClearToEmpty();
}
inline const std::string& JobDef::name() const {
  // @@protoc_insertion_point(field_get:tensorflow.JobDef.name)
  return _internal_name();
}
template <typename ArgT0, typename... ArgT>
inline PROTOBUF_ALWAYS_INLINE
void JobDef::set_name(ArgT0&& arg0, ArgT... args) {
 
 _impl_.name_.Set(static_cast<ArgT0 &&>(arg0), args..., GetArenaForAllocation());
  // @@protoc_insertion_point(field_set:tensorflow.JobDef.name)
}
inline std::string* JobDef::mutable_name() {
  std::string* _s = _internal_mutable_name();
  // @@protoc_insertion_point(field_mutable:tensorflow.JobDef.name)
  return _s;
}
inline const std::string& JobDef::_internal_name() const {
  return _impl_.name_.Get();
}
inline void JobDef::_internal_set_name(const std::string& value) {
  
  _impl_.name_.Set(value, GetArenaForAllocation());
}
inline std::string* JobDef::_internal_mutable_name() {
  
  return _impl_.name_.Mutable(GetArenaForAllocation());
}
inline std::string* JobDef::release_name() {
  // @@protoc_insertion_point(field_release:tensorflow.JobDef.name)
  return _impl_.name_.Release();
}
inline void JobDef::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  _impl_.name_.SetAllocated(name, GetArenaForAllocation());
#ifdef PROTOBUF_FORCE_COPY_DEFAULT_STRING
  if (_impl_.name_.IsDefault()) {
    _impl_.name_.Set("", GetArenaForAllocation());
  }
#endif // PROTOBUF_FORCE_COPY_DEFAULT_STRING
  // @@protoc_insertion_point(field_set_allocated:tensorflow.JobDef.name)
}

// map<int32, string> tasks = 2;
inline int JobDef::_internal_tasks_size() const {
  return _impl_.tasks_.size();
}
inline int JobDef::tasks_size() const {
  return _internal_tasks_size();
}
inline void JobDef::clear_tasks() {
  _impl_.tasks_.Clear();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >&
JobDef::_internal_tasks() const {
  return _impl_.tasks_.GetMap();
}
inline const ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >&
JobDef::tasks() const {
  // @@protoc_insertion_point(field_map:tensorflow.JobDef.tasks)
  return _internal_tasks();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >*
JobDef::_internal_mutable_tasks() {
  return _impl_.tasks_.MutableMap();
}
inline ::PROTOBUF_NAMESPACE_ID::Map< int32_t, std::string >*
JobDef::mutable_tasks() {
  // @@protoc_insertion_point(field_mutable_map:tensorflow.JobDef.tasks)
  return _internal_mutable_tasks();
}

// -------------------------------------------------------------------

// ClusterDef

// repeated .tensorflow.JobDef job = 1;
inline int ClusterDef::_internal_job_size() const {
  return _impl_.job_.size();
}
inline int ClusterDef::job_size() const {
  return _internal_job_size();
}
inline void ClusterDef::clear_job() {
  _impl_.job_.Clear();
}
inline ::tensorflow::JobDef* ClusterDef::mutable_job(int index) {
  // @@protoc_insertion_point(field_mutable:tensorflow.ClusterDef.job)
  return _impl_.job_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::JobDef >*
ClusterDef::mutable_job() {
  // @@protoc_insertion_point(field_mutable_list:tensorflow.ClusterDef.job)
  return &_impl_.job_;
}
inline const ::tensorflow::JobDef& ClusterDef::_internal_job(int index) const {
  return _impl_.job_.Get(index);
}
inline const ::tensorflow::JobDef& ClusterDef::job(int index) const {
  // @@protoc_insertion_point(field_get:tensorflow.ClusterDef.job)
  return _internal_job(index);
}
inline ::tensorflow::JobDef* ClusterDef::_internal_add_job() {
  return _impl_.job_.Add();
}
inline ::tensorflow::JobDef* ClusterDef::add_job() {
  ::tensorflow::JobDef* _add = _internal_add_job();
  // @@protoc_insertion_point(field_add:tensorflow.ClusterDef.job)
  return _add;
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::tensorflow::JobDef >&
ClusterDef::job() const {
  // @@protoc_insertion_point(field_list:tensorflow.ClusterDef.job)
  return _impl_.job_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tensorflow_2fcore_2fprotobuf_2fcluster_2eproto

#define SLANG_PRELUDE_EXPORT

#ifdef __CUDACC_RTC__
#define SLANG_CUDA_RTC 1
#else
#define SLANG_CUDA_RTC 0
#endif

#if SLANG_CUDA_RTC

#else

#include <cstdint>
#include <stdio.h>

#endif

// Define SLANG_CUDA_ENABLE_HALF to use the cuda_fp16 include to add half support.
// For this to work NVRTC needs to have the path to the CUDA SDK.
//
// As it stands the includes paths defined for Slang are passed down to NVRTC. Similarly defines
// defined for the Slang compile are passed down.

#ifdef SLANG_CUDA_ENABLE_HALF
// We don't want half2 operators, because it will implement comparison operators that return a
// bool(!). We want to generate those functions. Doing so means that we will have to define all
// the other half2 operators.
#define __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#endif

#ifdef SLANG_CUDA_ENABLE_OPTIX
#include <optix.h>
#endif

// Define slang offsetof implementation
#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(type, member) (size_t)((char*)&(((type*)0)->member) - (char*)0)
#endif

#ifndef SLANG_ALIGN_OF
#define SLANG_ALIGN_OF(type) __alignof__(type)
#endif

// Must be large enough to cause overflow and therefore infinity
#ifndef SLANG_INFINITY
#define SLANG_INFINITY ((float)(1e+300 * 1e+300))
#endif

// For now we'll disable any asserts in this prelude
#define SLANG_PRELUDE_ASSERT(x)

#ifndef SLANG_CUDA_WARP_SIZE
#define SLANG_CUDA_WARP_SIZE 32
#endif

#define SLANG_CUDA_WARP_MASK \
    (SLANG_CUDA_WARP_SIZE - 1) // Used for masking threadIdx.x to the warp lane index
#define SLANG_CUDA_WARP_BITMASK (~int(0))

//
#define SLANG_FORCE_INLINE inline

#define SLANG_CUDA_CALL __device__

#define SLANG_FORCE_INLINE inline
#define SLANG_INLINE inline


// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ZERO_INDEX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

// This macro handles how out-of-range surface coordinates are handled;
// I can equal
// cudaBoundaryModeClamp, in which case out-of-range coordinates are clamped to the valid range
// cudaBoundaryModeZero, in which case out-of-range reads return zero and out-of-range writes are
// ignored cudaBoundaryModeTrap, in which case out-of-range accesses cause the kernel execution to
// fail.

#ifndef SLANG_CUDA_BOUNDARY_MODE
#define SLANG_CUDA_BOUNDARY_MODE cudaBoundaryModeZero

// Can be one of SLANG_CUDA_PTX_BOUNDARY_MODE. Only applies *PTX* emitted CUDA operations
// which currently is just RWTextureRW format writes
//
// .trap         causes an execution trap on out-of-bounds addresses
// .clamp        stores data at the nearest surface location (sized appropriately)
// .zero         drops stores to out-of-bounds addresses

#define SLANG_PTX_BOUNDARY_MODE "zero"
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    SLANG_CUDA_CALL T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

// Typically defined in cuda.h, but we can't ship/rely on that, so just define here
typedef unsigned long long CUtexObject;
typedef unsigned long long CUsurfObject;

// On CUDA sampler state is actually bound up with the texture object. We have a SamplerState type,
// backed as a pointer, to simplify code generation, with the downside that such a binding will take
// up uniform space, even though it will have no effect.
// TODO(JS): Consider ways to strip use of variables of this type so have no binding,
struct SamplerStateUnused;
typedef SamplerStateUnused* SamplerState;


// TODO(JS): Not clear yet if this can be handled on CUDA, by just ignoring.
// For now, just map to the index type.
typedef size_t NonUniformResourceIndex;

// Code generator will generate the specific type
template<typename T, int ROWS, int COLS>
struct Matrix;

typedef int1 bool1;
typedef int2 bool2;
typedef int3 bool3;
typedef int4 bool4;

#if SLANG_CUDA_RTC

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#endif

typedef long long longlong;
typedef unsigned long long ulonglong;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL float make_float(T val)
{
    return (float)val;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float _slang_fmod(float x, float y)
{
    return ::fmodf(x, y);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double _slang_fmod(double x, double y)
{
    return ::fmod(x, y);
}

#if SLANG_CUDA_ENABLE_HALF

// Add the other vector half types
struct __half1
{
    __half x;
};
struct __align__(4) __half3
{
    __half x, y, z;
};
struct __align__(4) __half4
{
    __half x, y, z, w;
};
#endif

#define SLANG_VECTOR_GET_ELEMENT(T)                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##1 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##2 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##3 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }                                                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_vector_get_element(T##4 x, int index) \
    {                                                                                 \
        return ((T*)(&x))[index];                                                     \
    }
SLANG_VECTOR_GET_ELEMENT(int)
SLANG_VECTOR_GET_ELEMENT(uint)
SLANG_VECTOR_GET_ELEMENT(short)
SLANG_VECTOR_GET_ELEMENT(ushort)
SLANG_VECTOR_GET_ELEMENT(char)
SLANG_VECTOR_GET_ELEMENT(uchar)
SLANG_VECTOR_GET_ELEMENT(longlong)
SLANG_VECTOR_GET_ELEMENT(ulonglong)
SLANG_VECTOR_GET_ELEMENT(float)
SLANG_VECTOR_GET_ELEMENT(double)

#define SLANG_VECTOR_GET_ELEMENT_PTR(T)                                                      \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##1 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##2 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##3 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }                                                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T* _slang_vector_get_element_ptr(T##4 * x, int index) \
    {                                                                                        \
        return ((T*)(x)) + index;                                                            \
    }
SLANG_VECTOR_GET_ELEMENT_PTR(int)
SLANG_VECTOR_GET_ELEMENT_PTR(uint)
SLANG_VECTOR_GET_ELEMENT_PTR(short)
SLANG_VECTOR_GET_ELEMENT_PTR(ushort)
SLANG_VECTOR_GET_ELEMENT_PTR(char)
SLANG_VECTOR_GET_ELEMENT_PTR(uchar)
SLANG_VECTOR_GET_ELEMENT_PTR(longlong)
SLANG_VECTOR_GET_ELEMENT_PTR(ulonglong)
SLANG_VECTOR_GET_ELEMENT_PTR(float)
SLANG_VECTOR_GET_ELEMENT_PTR(double)

#if SLANG_CUDA_ENABLE_HALF
SLANG_VECTOR_GET_ELEMENT(__half)
SLANG_VECTOR_GET_ELEMENT_PTR(__half)
#endif

#define SLANG_CUDA_VECTOR_BINARY_OP(T, n, op)                                                 \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal, T##n other)             \
    {                                                                                         \
        T##n result;                                                                          \
        for (int i = 0; i < n; i++)                                                           \
            *_slang_vector_get_element_ptr(&result, i) =                                      \
                _slang_vector_get_element(thisVal, i) op _slang_vector_get_element(other, i); \
        return result;                                                                        \
    }
#define SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, op)                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL bool##n operator op(T##n thisVal, T##n other) \
    {                                                                                \
        bool##n result;                                                              \
        for (int i = 0; i < n; i++)                                                  \
            *_slang_vector_get_element_ptr(&result, i) =                             \
                (int)(_slang_vector_get_element(thisVal, i)                          \
                          op _slang_vector_get_element(other, i));                   \
        return result;                                                               \
    }
#define SLANG_CUDA_VECTOR_UNARY_OP(T, n, op)                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator op(T##n thisVal)                              \
    {                                                                                              \
        T##n result;                                                                               \
        for (int i = 0; i < n; i++)                                                                \
            *_slang_vector_get_element_ptr(&result, i) = op _slang_vector_get_element(thisVal, i); \
        return result;                                                                             \
    }

#define SLANG_CUDA_VECTOR_INT_OP(T, n)            \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, %)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ^)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, |)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, >>)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, <<)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, !)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)           \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, ~)

#define SLANG_CUDA_VECTOR_INT_OPS(T) \
    SLANG_CUDA_VECTOR_INT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_INT_OP(T, 4)

SLANG_CUDA_VECTOR_INT_OPS(int)
SLANG_CUDA_VECTOR_INT_OPS(uint)
SLANG_CUDA_VECTOR_INT_OPS(ushort)
SLANG_CUDA_VECTOR_INT_OPS(short)
SLANG_CUDA_VECTOR_INT_OPS(char)
SLANG_CUDA_VECTOR_INT_OPS(uchar)
SLANG_CUDA_VECTOR_INT_OPS(longlong)
SLANG_CUDA_VECTOR_INT_OPS(ulonglong)

#define SLANG_CUDA_VECTOR_FLOAT_OP(T, n)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, +)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, -)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, *)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, /)          \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, &&)         \
    SLANG_CUDA_VECTOR_BINARY_OP(T, n, ||)         \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <)  \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, >=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, <=) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, ==) \
    SLANG_CUDA_VECTOR_BINARY_COMPARE_OP(T, n, !=) \
    SLANG_CUDA_VECTOR_UNARY_OP(T, n, -)
#define SLANG_CUDA_VECTOR_FLOAT_OPS(T) \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 2)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 3)   \
    SLANG_CUDA_VECTOR_FLOAT_OP(T, 4)

SLANG_CUDA_VECTOR_FLOAT_OPS(float)
SLANG_CUDA_VECTOR_FLOAT_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_CUDA_VECTOR_FLOAT_OPS(__half)
#endif
#define SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, n)                                             \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n operator%(const T##n& left, const T##n& right) \
    {                                                                                      \
        T##n result;                                                                       \
        for (int i = 0; i < n; i++)                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_fmod(                      \
                _slang_vector_get_element(left, i),                                        \
                _slang_vector_get_element(right, i));                                      \
        return result;                                                                     \
    }
#define SLANG_CUDA_FLOAT_VECTOR_MOD(T)     \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 2) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 3) \
    SLANG_CUDA_FLOAT_VECTOR_MOD_IMPL(T, 4)

SLANG_CUDA_FLOAT_VECTOR_MOD(float)
SLANG_CUDA_FLOAT_VECTOR_MOD(double)

#if SLANG_CUDA_RTC || SLANG_CUDA_ENABLE_HALF
#define SLANG_MAKE_VECTOR(T)                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x, T y)           \
    {                                                                       \
        return T##2 {x, y};                                                 \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x, T y, T z)      \
    {                                                                       \
        return T##3 {x, y, z};                                              \
    }                                                                       \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x, T y, T z, T w) \
    {                                                                       \
        return T##4 {x, y, z, w};                                           \
    }
#endif

#if SLANG_CUDA_RTC
SLANG_MAKE_VECTOR(int)
SLANG_MAKE_VECTOR(uint)
SLANG_MAKE_VECTOR(short)
SLANG_MAKE_VECTOR(ushort)
SLANG_MAKE_VECTOR(char)
SLANG_MAKE_VECTOR(uchar)
SLANG_MAKE_VECTOR(float)
SLANG_MAKE_VECTOR(double)
SLANG_MAKE_VECTOR(longlong)
SLANG_MAKE_VECTOR(ulonglong)
#endif

#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR(__half)
#endif

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool1 make_bool1(bool x)
{
    return bool1{x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x, bool y)
{
    return bool2{x, y};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x, bool y, bool z)
{
    return bool3{x, y, z};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x, bool y, bool z, bool w)
{
    return bool4{x, y, z, w};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool2 make_bool2(bool x)
{
    return bool2{x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool3 make_bool3(bool x)
{
    return bool3{x, x, x};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool4 make_bool4(bool x)
{
    return bool4{x, x, x, x};
}

#if SLANG_CUDA_RTC
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##1 make_##T##1(T x) \
    {                                                        \
        return T##1 {x};                                     \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#else
#define SLANG_MAKE_VECTOR_FROM_SCALAR(T)                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##2 make_##T##2(T x) \
    {                                                        \
        return make_##T##2(x, x);                            \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##3 make_##T##3(T x) \
    {                                                        \
        return make_##T##3(x, x, x);                         \
    }                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##4 make_##T##4(T x) \
    {                                                        \
        return make_##T##4(x, x, x, x);                      \
    }
#endif
SLANG_MAKE_VECTOR_FROM_SCALAR(int)
SLANG_MAKE_VECTOR_FROM_SCALAR(uint)
SLANG_MAKE_VECTOR_FROM_SCALAR(short)
SLANG_MAKE_VECTOR_FROM_SCALAR(ushort)
SLANG_MAKE_VECTOR_FROM_SCALAR(char)
SLANG_MAKE_VECTOR_FROM_SCALAR(uchar)
SLANG_MAKE_VECTOR_FROM_SCALAR(longlong)
SLANG_MAKE_VECTOR_FROM_SCALAR(ulonglong)
SLANG_MAKE_VECTOR_FROM_SCALAR(float)
SLANG_MAKE_VECTOR_FROM_SCALAR(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_MAKE_VECTOR_FROM_SCALAR(__half)
#if !SLANG_CUDA_RTC
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half1 make___half1(__half x)
{
    return __half1{x};
}
#endif
#endif

#define SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(Fn, T, N)                                            \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T##N Fn(T##N* address, T##N val)                           \
    {                                                                                             \
        T##N result;                                                                              \
        for (int i = 0; i < N; i++)                                                               \
            *_slang_vector_get_element_ptr(&result, i) =                                          \
                Fn(_slang_vector_get_element_ptr(address, i), _slang_vector_get_element(val, i)); \
        return result;                                                                            \
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 4)
#endif
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, float, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, int, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, uint, 4)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 2)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 3)
SLANG_CUDA_VECTOR_ATOMIC_BINARY_IMPL(atomicAdd, ulonglong, 4)

template<typename T, int n>
struct GetVectorTypeImpl
{
};

#define GET_VECTOR_TYPE_IMPL(T, n)                                     \
    template<>                                                         \
    struct GetVectorTypeImpl<T, n>                                     \
    {                                                                  \
        typedef T##n type;                                             \
        static SLANG_FORCE_INLINE SLANG_CUDA_CALL T##n fromScalar(T v) \
        {                                                              \
            return make_##T##n(v);                                     \
        }                                                              \
    };
#define GET_VECTOR_TYPE_IMPL_N(T) \
    GET_VECTOR_TYPE_IMPL(T, 1)    \
    GET_VECTOR_TYPE_IMPL(T, 2)    \
    GET_VECTOR_TYPE_IMPL(T, 3)    \
    GET_VECTOR_TYPE_IMPL(T, 4)

GET_VECTOR_TYPE_IMPL_N(int)
GET_VECTOR_TYPE_IMPL_N(uint)
GET_VECTOR_TYPE_IMPL_N(short)
GET_VECTOR_TYPE_IMPL_N(ushort)
GET_VECTOR_TYPE_IMPL_N(char)
GET_VECTOR_TYPE_IMPL_N(uchar)
GET_VECTOR_TYPE_IMPL_N(longlong)
GET_VECTOR_TYPE_IMPL_N(ulonglong)
GET_VECTOR_TYPE_IMPL_N(float)
GET_VECTOR_TYPE_IMPL_N(double)
#if SLANG_CUDA_ENABLE_HALF
GET_VECTOR_TYPE_IMPL_N(__half)
#endif
template<typename T, int n>
using Vector = typename GetVectorTypeImpl<T, n>::type;

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, COLS>& operator[](size_t index)
    {
        return rows[index];
    }
};


template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T scalar)
{
    Matrix<T, ROWS, COLS> result;
    for (int i = 0; i < ROWS; i++)
        result.rows[i] = GetVectorTypeImpl<T, COLS>::fromScalar(scalar);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(const Vector<T, COLS>& row0)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Vector<T, COLS>& row0,
    const Vector<T, COLS>& row1,
    const Vector<T, COLS>& row2,
    const Vector<T, COLS>& row3)
{
    Matrix<T, ROWS, COLS> result;
    result.rows[0] = row0;
    result.rows[1] = row1;
    result.rows[2] = row2;
    result.rows[3] = row3;
    return result;
}

template<typename T, int ROWS, int COLS, typename U, int otherRow, int otherCol>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    const Matrix<U, otherRow, otherCol>& other)
{
    Matrix<T, ROWS, COLS> result;
    int minRow = ROWS;
    int minCol = COLS;
    if (minRow > otherRow)
        minRow = otherRow;
    if (minCol > otherCol)
        minCol = otherCol;
    for (int i = 0; i < minRow; i++)
        for (int j = 0; j < minCol; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) =
                (T)_slang_vector_get_element(other.rows[i], j);
    return result;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(T v0, T v1, T v2, T v3)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[1].x = v2;
    rs.rows[1].y = v3;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 3)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v5;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[1].x = v2;
        rs.rows[1].y = v3;
        rs.rows[2].x = v4;
        rs.rows[2].y = v5;
        rs.rows[3].x = v6;
        rs.rows[3].y = v7;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[1].x = v3;
    rs.rows[1].y = v4;
    rs.rows[1].z = v5;
    rs.rows[2].x = v6;
    rs.rows[2].y = v7;
    rs.rows[2].z = v8;
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11)
{
    Matrix<T, ROWS, COLS> rs;
    if (COLS == 4)
    {
        *_slang_vector_get_element_ptr(&rs.rows[0], 0) = v0;
        *_slang_vector_get_element_ptr(&rs.rows[0], 1) = v1;
        *_slang_vector_get_element_ptr(&rs.rows[0], 2) = v2;
        *_slang_vector_get_element_ptr(&rs.rows[0], 3) = v3;
        *_slang_vector_get_element_ptr(&rs.rows[1], 0) = v4;
        *_slang_vector_get_element_ptr(&rs.rows[1], 1) = v5;
        *_slang_vector_get_element_ptr(&rs.rows[1], 2) = v6;
        *_slang_vector_get_element_ptr(&rs.rows[1], 3) = v7;
        *_slang_vector_get_element_ptr(&rs.rows[2], 0) = v8;
        *_slang_vector_get_element_ptr(&rs.rows[2], 1) = v9;
        *_slang_vector_get_element_ptr(&rs.rows[2], 2) = v10;
        *_slang_vector_get_element_ptr(&rs.rows[2], 3) = v11;
    }
    else
    {
        rs.rows[0].x = v0;
        rs.rows[0].y = v1;
        rs.rows[0].z = v2;
        rs.rows[1].x = v3;
        rs.rows[1].y = v4;
        rs.rows[1].z = v5;
        rs.rows[2].x = v6;
        rs.rows[2].y = v7;
        rs.rows[2].z = v8;
        rs.rows[3].x = v9;
        rs.rows[3].y = v10;
        rs.rows[3].z = v11;
    }
    return rs;
}

template<typename T, int ROWS, int COLS>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, ROWS, COLS> makeMatrix(
    T v0,
    T v1,
    T v2,
    T v3,
    T v4,
    T v5,
    T v6,
    T v7,
    T v8,
    T v9,
    T v10,
    T v11,
    T v12,
    T v13,
    T v14,
    T v15)
{
    Matrix<T, ROWS, COLS> rs;
    rs.rows[0].x = v0;
    rs.rows[0].y = v1;
    rs.rows[0].z = v2;
    rs.rows[0].w = v3;
    rs.rows[1].x = v4;
    rs.rows[1].y = v5;
    rs.rows[1].z = v6;
    rs.rows[1].w = v7;
    rs.rows[2].x = v8;
    rs.rows[2].y = v9;
    rs.rows[2].z = v10;
    rs.rows[2].w = v11;
    rs.rows[3].x = v12;
    rs.rows[3].y = v13;
    rs.rows[3].z = v14;
    rs.rows[3].w = v15;
    return rs;
}

#define SLANG_MATRIX_BINARY_OP(T, op)                                   \
    template<int R, int C>                                              \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(     \
        const Matrix<T, R, C>& thisVal,                                 \
        const Matrix<T, R, C>& other)                                   \
    {                                                                   \
        Matrix<T, R, C> result;                                         \
        for (int i = 0; i < R; i++)                                     \
            for (int j = 0; j < C; j++)                                 \
                *_slang_vector_get_element_ptr(result.rows + i, j) =    \
                    _slang_vector_get_element(thisVal.rows[i], j)       \
                        op _slang_vector_get_element(other.rows[i], j); \
        return result;                                                  \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                                                               \
    template<int R, int C>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                                                              \
        Matrix<T, R, C> result;                                                                    \
        for (int i = 0; i < R; i++)                                                                \
            for (int j = 0; j < C; j++)                                                            \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                               \
                    op _slang_vector_get_element(thisVal.rows[i], j);                              \
        return result;                                                                             \
    }
#define SLANG_INT_MATRIX_OPS(T)   \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_BINARY_OP(T, &)  \
    SLANG_MATRIX_BINARY_OP(T, |)  \
    SLANG_MATRIX_BINARY_OP(T, &&) \
    SLANG_MATRIX_BINARY_OP(T, ||) \
    SLANG_MATRIX_BINARY_OP(T, ^)  \
    SLANG_MATRIX_BINARY_OP(T, %)  \
    SLANG_MATRIX_UNARY_OP(T, !)   \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T) \
    SLANG_MATRIX_BINARY_OP(T, +)  \
    SLANG_MATRIX_BINARY_OP(T, -)  \
    SLANG_MATRIX_BINARY_OP(T, *)  \
    SLANG_MATRIX_BINARY_OP(T, /)  \
    SLANG_MATRIX_UNARY_OP(T, -)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(short)
SLANG_INT_MATRIX_OPS(ushort)
SLANG_INT_MATRIX_OPS(char)
SLANG_INT_MATRIX_OPS(uchar)
SLANG_INT_MATRIX_OPS(longlong)
SLANG_INT_MATRIX_OPS(ulonglong)
SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)
#if SLANG_CUDA_ENABLE_HALF
SLANG_FLOAT_MATRIX_OPS(__half)
#endif
#define SLANG_MATRIX_INT_NEG_OP(T)                                                        \
    template<int R, int C>                                                                \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                                     \
        Matrix<T, R, C> result;                                                           \
        for (int i = 0; i < R; i++)                                                       \
            for (int j = 0; j < C; j++)                                                   \
                *_slang_vector_get_element_ptr(result.rows + i, j) =                      \
                    0 - _slang_vector_get_element(thisVal.rows[i], j);                    \
        return result;                                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(short)
SLANG_MATRIX_INT_NEG_OP(ushort)
SLANG_MATRIX_INT_NEG_OP(char)
SLANG_MATRIX_INT_NEG_OP(uchar)
SLANG_MATRIX_INT_NEG_OP(longlong)
SLANG_MATRIX_INT_NEG_OP(ulonglong)

#define SLANG_FLOAT_MATRIX_MOD(T)                                                 \
    template<int R, int C>                                                        \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<T, R, C> operator%(                 \
        Matrix<T, R, C> left,                                                     \
        Matrix<T, R, C> right)                                                    \
    {                                                                             \
        Matrix<T, R, C> result;                                                   \
        for (int i = 0; i < R; i++)                                               \
            for (int j = 0; j < C; j++)                                           \
                *_slang_vector_get_element_ptr(result.rows + i, j) = _slang_fmod( \
                    _slang_vector_get_element(left.rows[i], j),                   \
                    _slang_vector_get_element(right.rows[i], j));                 \
        return result;                                                            \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#if SLANG_CUDA_ENABLE_HALF
template<int R, int C>
SLANG_FORCE_INLINE SLANG_CUDA_CALL Matrix<__half, R, C> operator%(
    Matrix<__half, R, C> left,
    Matrix<__half, R, C> right)
{
    Matrix<__half, R, C> result;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            *_slang_vector_get_element_ptr(result.rows + i, j) = __float2half(_slang_fmod(
                __half2float(_slang_vector_get_element(left.rows[i], j)),
                __half2float(_slang_vector_get_element(right.rows[i], j))));
    return result;
}
#endif
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

#define SLANG_SELECT_IMPL(T, N)                                                                  \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL Vector<T, N> _slang_select(                               \
        bool##N condition,                                                                       \
        Vector<T, N> v0,                                                                         \
        Vector<T, N> v1)                                                                         \
    {                                                                                            \
        Vector<T, N> result;                                                                     \
        for (int i = 0; i < N; i++)                                                              \
        {                                                                                        \
            *_slang_vector_get_element_ptr(&result, i) = _slang_vector_get_element(condition, i) \
                                                             ? _slang_vector_get_element(v0, i)  \
                                                             : _slang_vector_get_element(v1, i); \
        }                                                                                        \
        return result;                                                                           \
    }
#define SLANG_SELECT_T(T)   \
    SLANG_SELECT_IMPL(T, 2) \
    SLANG_SELECT_IMPL(T, 3) \
    SLANG_SELECT_IMPL(T, 4)

SLANG_SELECT_T(int)
SLANG_SELECT_T(uint)
SLANG_SELECT_T(short)
SLANG_SELECT_T(ushort)
SLANG_SELECT_T(char)
SLANG_SELECT_T(uchar)
SLANG_SELECT_T(float)
SLANG_SELECT_T(double)

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

//
// Half support
//

#if SLANG_CUDA_ENABLE_HALF
SLANG_SELECT_T(__half)

// Convenience functions ushort -> half

SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 __ushort_as_half(const ushort2& i)
{
    return __halves2half2(__ushort_as_half(i.x), __ushort_as_half(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half3 __ushort_as_half(const ushort3& i)
{
    return __half3{__ushort_as_half(i.x), __ushort_as_half(i.y), __ushort_as_half(i.z)};
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 __ushort_as_half(const ushort4& i)
{
    return __half4{
        __ushort_as_half(i.x),
        __ushort_as_half(i.y),
        __ushort_as_half(i.z),
        __ushort_as_half(i.w)};
}

// Convenience functions half -> ushort

SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort2 __half_as_ushort(const __half2& i)
{
    return make_ushort2(__half_as_ushort(i.x), __half_as_ushort(i.y));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort3 __half_as_ushort(const __half3& i)
{
    return make_ushort3(__half_as_ushort(i.x), __half_as_ushort(i.y), __half_as_ushort(i.z));
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL ushort4 __half_as_ushort(const __half4& i)
{
    return make_ushort4(
        __half_as_ushort(i.x),
        __half_as_ushort(i.y),
        __half_as_ushort(i.z),
        __half_as_ushort(i.w));
}

// This is a little bit of a hack. Fortunately CUDA has the definitions of the templated types in
// include/surface_indirect_functions.h
// Here we find the template definition requires a specialization of __nv_isurf_trait to allow
// a specialization of the surface write functions.
// This *isn't* a problem on the read functions as they don't have a return type that uses this
// mechanism

template<>
struct __nv_isurf_trait<__half>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half2>
{
    typedef void type;
};
template<>
struct __nv_isurf_trait<__half4>
{
    typedef void type;
};

#define SLANG_DROP_PARENS(...) __VA_ARGS__

#define SLANG_SURFACE_READ(FUNC_NAME, TYPE_ARGS, ARGS)                                             \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half FUNC_NAME<__half>(                                   \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(FUNC_NAME<ushort>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half2 FUNC_NAME<__half2>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL __half4 FUNC_NAME<__half4>(                                 \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        return __ushort_as_half(                                                                   \
            FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                    \
    }

SLANG_SURFACE_READ(surf1Dread, (int x), (x))
SLANG_SURFACE_READ(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ(surf3Dread, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_READ(surf1DLayeredread, (int x, int layer), (x, layer))
SLANG_SURFACE_READ(surf2DLayeredread, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_READ(surfCubemapread, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_READ(surfCubemapLayeredread, (int x, int y, int layerFace), (x, y, layerFace))

#define SLANG_SURFACE_WRITE(FUNC_NAME, TYPE_ARGS, ARGS)                                            \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half>(                                     \
        __half data,                                                                               \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode);  \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half2>(                                    \
        __half2 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort2>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }                                                                                              \
                                                                                                   \
    template<>                                                                                     \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL void FUNC_NAME<__half4>(                                    \
        __half4 data,                                                                              \
        cudaSurfaceObject_t surfObj,                                                               \
        SLANG_DROP_PARENS TYPE_ARGS,                                                               \
        cudaSurfaceBoundaryMode boundaryMode)                                                      \
    {                                                                                              \
        FUNC_NAME<ushort4>(__half_as_ushort(data), surfObj, SLANG_DROP_PARENS ARGS, boundaryMode); \
    }

SLANG_SURFACE_WRITE(surf1Dwrite, (int x), (x))
SLANG_SURFACE_WRITE(surf2Dwrite, (int x, int y), (x, y))
SLANG_SURFACE_WRITE(surf3Dwrite, (int x, int y, int z), (x, y, z))
SLANG_SURFACE_WRITE(surf1DLayeredwrite, (int x, int layer), (x, layer))
SLANG_SURFACE_WRITE(surf2DLayeredwrite, (int x, int y, int layer), (x, y, layer))
SLANG_SURFACE_WRITE(surfCubemapwrite, (int x, int y, int face), (x, y, face))
SLANG_SURFACE_WRITE(surfCubemapLayeredwrite, (int x, int y, int layerFace), (x, y, layerFace))

// ! Hack to test out reading !!!
// Only works converting *from* half

// template <typename T>
// SLANG_FORCE_INLINE SLANG_CUDA_CALL T surf2Dread_convert(cudaSurfaceObject_t surfObj, int x, int
// y, cudaSurfaceBoundaryMode boundaryMode);

#define SLANG_SURFACE_READ_HALF_CONVERT(FUNC_NAME, TYPE_ARGS, ARGS)                              \
                                                                                                 \
    template<typename T>                                                                         \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL T FUNC_NAME##_convert(                                    \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode);                                                   \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float FUNC_NAME##_convert<float>(                         \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        return __ushort_as_half(                                                                 \
            FUNC_NAME<uint16_t>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode));                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float2 FUNC_NAME##_convert<float2>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half2 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort2>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float2{v.x, v.y};                                                                 \
    }                                                                                            \
                                                                                                 \
    template<>                                                                                   \
    SLANG_FORCE_INLINE SLANG_CUDA_CALL float4 FUNC_NAME##_convert<float4>(                       \
        cudaSurfaceObject_t surfObj,                                                             \
        SLANG_DROP_PARENS TYPE_ARGS,                                                             \
        cudaSurfaceBoundaryMode boundaryMode)                                                    \
    {                                                                                            \
        const __half4 v =                                                                        \
            __ushort_as_half(FUNC_NAME<ushort4>(surfObj, SLANG_DROP_PARENS ARGS, boundaryMode)); \
        return float4{v.x, v.y, v.z, v.w};                                                       \
    }

SLANG_SURFACE_READ_HALF_CONVERT(surf1Dread, (int x), (x))
SLANG_SURFACE_READ_HALF_CONVERT(surf2Dread, (int x, int y), (x, y))
SLANG_SURFACE_READ_HALF_CONVERT(surf3Dread, (int x, int y, int z), (x, y, z))

#endif

// Support for doing format conversion when writing to a surface/RWTexture

// NOTE! For normal surface access x values are *byte* addressed.
// For the _convert versions they are *not*. They don't need to be because sust.p does not require
// it.

template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode);
template<typename T>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert(
    T,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode);

// https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust

// Float

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(v));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float>(
    float v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(v));
}

// Float2

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2}], {%3,%4};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float2>(
    float2 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1,%2,%3}], {%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy));
}

// Float4
template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf1Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.1d.b32." SLANG_PTX_BOUNDARY_MODE " [%0, {%1}], {%2,%3,%4,%5};}\n\t" ::"l"(surfObj),
        "r"(x),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf2Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2}], {%3,%4,%5,%6};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

template<>
SLANG_FORCE_INLINE SLANG_CUDA_CALL void surf3Dwrite_convert<float4>(
    float4 v,
    cudaSurfaceObject_t surfObj,
    int x,
    int y,
    int z,
    cudaSurfaceBoundaryMode boundaryMode)
{
    const float vx = v.x, vy = v.y, vz = v.z, vw = v.w;
    asm volatile(
        "{sust.p.2d.b32." SLANG_PTX_BOUNDARY_MODE
        " [%0, {%1,%2,%3}], {%4,%5,%6,%7};}\n\t" ::"l"(surfObj),
        "r"(x),
        "r"(y),
        "r"(z),
        "f"(vx),
        "f"(vy),
        "f"(vz),
        "f"(vw));
}

// ----------------------------- F32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F32_sincos(float f, float* s, float* c)
{
    ::sincosf(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sqrt(float f)
{
    return ::sqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_rsqrt(float f)
{
    return ::rsqrtf(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_sign(float f)
{
    return (f == 0.0f) ? f : ((f < 0.0f) ? -1.0f : 1.0f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isnan(float f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isfinite(float f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F32_isinf(float f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_frexp(float x, int* e)
{
    return frexpf(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}


// ----------------------------- F64 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_sincos(double f, double* s, double* c)
{
    ::sincos(f, s, c);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sqrt(double f)
{
    return ::sqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_rsqrt(double f)
{
    return ::rsqrt(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_sign(double f)
{
    return (f == 0.0) ? f : ((f < 0.0) ? -1.0 : 1.0);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isnan(double f)
{
    return isnan(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isfinite(double f)
{
    return isfinite(f);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL bool F64_isinf(double f)
{
    return isinf(f);
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

// Ternary
SLANG_FORCE_INLINE SLANG_CUDA_CALL double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

// ----------------------------- I32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

// ----------------------------- U32 -----------------------------------------

// Unary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_abs(uint32_t f)
{
    return f;
}

// Binary
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U32_countbits(uint32_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popc(v);
}


// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE SLANG_CUDA_CALL int64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE SLANG_CUDA_CALL uint32_t U64_countbits(uint64_t v)
{
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html#group__CUDA__MATH__INTRINSIC__INT_1g43c9c7d2b9ebf202ff1ef5769989be46
    return __popcll(v);
}


// ----------------------------- ResourceType -----------------------------------------


// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct StructuredBuffer
{
    SLANG_CUDA_CALL const T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

    SLANG_CUDA_CALL const T& Load(size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, count);
#endif
        return data[index];
    }

#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }
#endif

    T* data;
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
    size_t count;
#endif
};

template<typename T>
struct RWStructuredBuffer : StructuredBuffer<T>
{
    SLANG_CUDA_CALL T& operator[](size_t index) const
    {
#ifndef SLANG_CUDA_STRUCTURED_BUFFER_NO_COUNT
        SLANG_BOUND_CHECK(index, this->count);
#endif
        return this->data[index];
    }
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }
    template<typename T>
    SLANG_CUDA_CALL StructuredBuffer<T> asStructuredBuffer() const
    {
        StructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    SLANG_CUDA_CALL void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    SLANG_CUDA_CALL uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    SLANG_CUDA_CALL uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    SLANG_CUDA_CALL uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    SLANG_CUDA_CALL uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    SLANG_CUDA_CALL T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        T data;
        memcpy(&data, ((const char*)this->data) + index, sizeof(T));
        return data;
    }

    SLANG_CUDA_CALL void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    SLANG_CUDA_CALL void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    SLANG_CUDA_CALL void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    SLANG_CUDA_CALL void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    SLANG_CUDA_CALL void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        memcpy((char*)data + index, &value, sizeof(T));
    }

    /// Can be used in the core module to gain access
    template<typename T>
    SLANG_CUDA_CALL T* _getPtrAt(size_t index)
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return (T*)(((char*)data) + index);
    }
    template<typename T>
    SLANG_CUDA_CALL RWStructuredBuffer<T> asStructuredBuffer() const
    {
        RWStructuredBuffer<T> rs;
        rs.data = (T*)data;
        rs.count = sizeInBytes / sizeof(T);
        return rs;
    }
    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};


// ---------------------- Wave --------------------------------------

// TODO(JS): It appears that cuda does not have a simple way to get a lane index.
//
// Another approach could be...
// laneId = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) &
// SLANG_CUDA_WARP_MASK If that is really true another way to do this, would be for code generator
// to add this function with the [numthreads] baked in.
//
// For now I'll just assume you have a launch that makes the following correct if the kernel uses
// WaveGetLaneIndex()
#ifndef SLANG_USE_ASM_LANE_ID
__forceinline__ __device__ uint32_t _getLaneId()
{
    // If the launch is (or I guess some multiple of the warp size)
    // we try this mechanism, which is apparently faster.
    return threadIdx.x & SLANG_CUDA_WARP_MASK;
}
#else
__forceinline__ __device__ uint32_t _getLaneId()
{
    // https://stackoverflow.com/questions/44337309/whats-the-most-efficient-way-to-calculate-the-warp-id-lane-id-in-a-1-d-grid#
    // This mechanism is not the fastest way to do it, and that is why the other mechanism
    // is the default. But the other mechanism relies on a launch that makes the assumption
    // true.
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}
#endif

typedef int WarpMask;

// It appears that the __activemask() cannot always be used because
// threads need to be converged.
//
// For CUDA the article claims mask has to be used carefully
// https://devblogs.nvidia.com/using-cuda-warp-level-primitives/
// With the Warp intrinsics there is no mask, and it's just the 'active lanes'.
// __activemask() though does not require there is convergence, so that doesn't work.
//
// '__ballot_sync' produces a convergance.
//
// From the CUDA docs:
// ```For __all_sync, __any_sync, and __ballot_sync, a mask must be passed that specifies the
// threads participating in the call. A bit, representing the thread's lane ID, must be set for each
// participating thread to ensure they are properly converged before the intrinsic is executed by
// the hardware. All active threads named in mask must execute the same intrinsic with the same
// mask, or the result is undefined.```
//
// Currently there isn't a mechanism to correctly get the mask without it being passed through.
// Doing so will most likely require some changes to slang code generation to track masks, for now
// then we use _getActiveMask.

// Return mask of all the lanes less than the current lane
__forceinline__ __device__ WarpMask _getLaneLtMask()
{
    return (int(1) << _getLaneId()) - 1;
}

// TODO(JS):
// THIS IS NOT CORRECT! That determining the appropriate active mask requires appropriate
// mask tracking.
__forceinline__ __device__ WarpMask _getActiveMask()
{
    return __ballot_sync(__activemask(), true);
}

// Return a mask suitable for the 'MultiPrefix' style functions
__forceinline__ __device__ WarpMask _getMultiPrefixMask(int mask)
{
    return mask;
}

// Note! Note will return true if mask is 0, but thats okay, because there must be one
// lane active to execute anything
__inline__ __device__ bool _waveIsSingleLane(WarpMask mask)
{
    return (mask & (mask - 1)) == 0;
}

// Returns the power of 2 size of run of set bits. Returns 0 if not a suitable run.
// Examples:
// 0b00000000'00000000'00000000'11111111 -> 8
// 0b11111111'11111111'11111111'11111111 -> 32
// 0b00000000'00000000'00000000'00011111 -> 0 (since 5 is not a power of 2)
// 0b00000000'00000000'00000000'11110000 -> 0 (since the run of bits does not start at the LSB)
// 0b00000000'00000000'00000000'00100111 -> 0 (since it is not a single contiguous run)
__inline__ __device__ int _waveCalcPow2Offset(WarpMask mask)
{
    // This should be the most common case, so fast path it
    if (mask == SLANG_CUDA_WARP_BITMASK)
    {
        return SLANG_CUDA_WARP_SIZE;
    }
    // Is it a contiguous run of bits?
    if ((mask & (mask + 1)) == 0)
    {
        // const int offsetSize = __ffs(mask + 1) - 1;
        const int offset = 32 - __clz(mask);
        // Is it a power of 2 size
        if ((offset & (offset - 1)) == 0)
        {
            return offset;
        }
    }
    return 0;
}

__inline__ __device__ bool _waveIsFirstLane()
{
    const WarpMask mask = __activemask();
    // We special case bit 0, as that most warps are expected to be fully active.

    // mask & -mask, isolates the lowest set bit.
    // return (mask & 1 ) || ((mask & -mask) == (1 << _getLaneId()));

    // This mechanism is most similar to what was in an nVidia post, so assume it is prefered.
    return (mask & 1) || ((__ffs(mask) - 1) == _getLaneId());
}

template<typename T>
struct WaveOpOr
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a | b; }
};

template<typename T>
struct WaveOpAnd
{
    __inline__ __device__ static T getInitial(T a) { return ~T(0); }
    __inline__ __device__ static T doOp(T a, T b) { return a & b; }
};

template<typename T>
struct WaveOpXor
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a ^ b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a ^ b; }
};

template<typename T>
struct WaveOpAdd
{
    __inline__ __device__ static T getInitial(T a) { return 0; }
    __inline__ __device__ static T doOp(T a, T b) { return a + b; }
    __inline__ __device__ static T doInverse(T a, T b) { return a - b; }
};

template<typename T>
struct WaveOpMul
{
    __inline__ __device__ static T getInitial(T a) { return T(1); }
    __inline__ __device__ static T doOp(T a, T b) { return a * b; }
    // Using this inverse for int is probably undesirable - because in general it requires T to have
    // more precision There is also a performance aspect to it, where divides are generally
    // significantly slower
    __inline__ __device__ static T doInverse(T a, T b) { return a / b; }
};

template<typename T>
struct WaveOpMax
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a > b ? a : b; }
};

template<typename T>
struct WaveOpMin
{
    __inline__ __device__ static T getInitial(T a) { return a; }
    __inline__ __device__ static T doOp(T a, T b) { return a < b ? a : b; }
};

template<typename T>
struct ElementTypeTrait;

// Scalar
template<>
struct ElementTypeTrait<int>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<uint>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<float>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<double>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<uint64_t>
{
    typedef uint64_t Type;
};
template<>
struct ElementTypeTrait<int64_t>
{
    typedef int64_t Type;
};

// Vector
template<>
struct ElementTypeTrait<int1>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int2>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int3>
{
    typedef int Type;
};
template<>
struct ElementTypeTrait<int4>
{
    typedef int Type;
};

template<>
struct ElementTypeTrait<uint1>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint2>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint3>
{
    typedef uint Type;
};
template<>
struct ElementTypeTrait<uint4>
{
    typedef uint Type;
};

template<>
struct ElementTypeTrait<float1>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float2>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float3>
{
    typedef float Type;
};
template<>
struct ElementTypeTrait<float4>
{
    typedef float Type;
};

template<>
struct ElementTypeTrait<double1>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double2>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double3>
{
    typedef double Type;
};
template<>
struct ElementTypeTrait<double4>
{
    typedef double Type;
};

// Matrix
template<typename T, int ROWS, int COLS>
struct ElementTypeTrait<Matrix<T, ROWS, COLS>>
{
    typedef T Type;
};

// Scalar
template<typename INTF, typename T>
__device__ T _waveReduceScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            val = INTF::doOp(val, __shfl_xor_sync(mask, val, offset));
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        T result = INTF::getInitial(val);
        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            result = INTF::doOp(result, __shfl_sync(mask, val, srcLane));
            remaining &= ~laneBit;
        }
        return result;
    }
    return val;
}


// Multiple values
template<typename INTF, typename T, size_t COUNT>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);
    if (offsetSize > 0)
    {
        // Fast path O(log2(activeLanes))
        for (int offset = offsetSize >> 1; offset > 0; offset >>= 1)
        {
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_xor_sync(mask, val[i], offset));
            }
        }
    }
    else if (!_waveIsSingleLane(mask))
    {
        // Copy the original
        T originalVal[COUNT];
        for (size_t i = 0; i < COUNT; ++i)
        {
            const T v = val[i];
            originalVal[i] = v;
            val[i] = INTF::getInitial(v);
        }

        int remaining = mask;
        while (remaining)
        {
            const int laneBit = remaining & -remaining;
            // Get the sourceLane
            const int srcLane = __ffs(laneBit) - 1;
            // Broadcast (can also broadcast to self)
            for (size_t i = 0; i < COUNT; ++i)
            {
                val[i] = INTF::doOp(val[i], __shfl_sync(mask, originalVal[i], srcLane));
            }
            remaining &= ~laneBit;
        }
    }
}

template<typename INTF, typename T>
__device__ void _waveReduceMultiple(WarpMask mask, T* val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<INTF, ElemType, sizeof(T) / sizeof(ElemType)>(mask, (ElemType*)val);
}

template<typename T>
__inline__ __device__ T _waveOr(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveAnd(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAnd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveXor(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveProduct(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveSum(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMin(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMin<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _waveMax(WarpMask mask, T val)
{
    return _waveReduceScalar<WaveOpMax<T>, T>(mask, val);
}

// Fast-path specializations when CUDA warp reduce operators are available
#if __CUDA_ARCH__ >= 800 // 8.x or higher
template<>
__inline__ __device__ unsigned _waveOr<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_or_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveAnd<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_and_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveXor<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_xor_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveSum<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ int _waveSum<int>(WarpMask mask, int val)
{
    return __reduce_add_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMin<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMin<int>(WarpMask mask, int val)
{
    return __reduce_min_sync(mask, val);
}

template<>
__inline__ __device__ unsigned _waveMax<unsigned>(WarpMask mask, unsigned val)
{
    return __reduce_max_sync(mask, val);
}

template<>
__inline__ __device__ int _waveMax<int>(WarpMask mask, int val)
{
    return __reduce_max_sync(mask, val);
}
#endif


// Multiple

template<typename T>
__inline__ __device__ T _waveOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpOr<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAnd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpXor<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMul<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpAdd<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMinMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMin<ElemType>>(mask, &val);
    return val;
}

template<typename T>
__inline__ __device__ T _waveMaxMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _waveReduceMultiple<WaveOpMax<ElemType>>(mask, &val);
    return val;
}


template<typename T>
__inline__ __device__ bool _waveAllEqual(WarpMask mask, T val)
{
    int pred;
    __match_all_sync(mask, val, &pred);
    return pred != 0;
}

template<typename T>
__inline__ __device__ bool _waveAllEqualMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    for (size_t i = 0; i < count; ++i)
    {
        __match_all_sync(mask, src[i], &pred);
        if (pred == 0)
        {
            return false;
        }
    }
    return true;
}

template<typename T>
__inline__ __device__ T _waveReadFirst(WarpMask mask, T val)
{
    const int lowestLaneId = __ffs(mask) - 1;
    return __shfl_sync(mask, val, lowestLaneId);
}

template<typename T>
__inline__ __device__ T _waveReadFirstMultiple(WarpMask mask, T inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    const int lowestLaneId = __ffs(mask) - 1;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lowestLaneId);
    }
    return outVal;
}

template<typename T>
__inline__ __device__ T _waveShuffleMultiple(WarpMask mask, T inVal, int lane)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    T outVal;
    const ElemType* src = (const ElemType*)&inVal;
    ElemType* dst = (ElemType*)&outVal;
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __shfl_sync(mask, src[i], lane);
    }
    return outVal;
}

// Scalar

// Invertable means that when we get to the end of the reduce, we can remove val (to make
// exclusive), using the inverse of the op.
template<typename INTF, typename T>
__device__ T _wavePrefixInvertableScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result;
    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        result = val;
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, result, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
            }
        }
        // Remove val from the result, by applyin inverse
        result = INTF::doInverse(result, val);
    }
    else
    {
        result = INTF::getInitial(val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


// This implementation separately tracks the value to be propogated, and the value
// that is the final result
template<typename INTF, typename T>
__device__ T _wavePrefixScalar(WarpMask mask, T val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T result = INTF::getInitial(val);
    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra multiply for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            const T readVal = __shfl_up_sync(mask, val, i, offsetSize);
            if (laneId >= i)
            {
                result = INTF::doOp(result, readVal);
                val = INTF::doOp(val, readVal);
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;
                // Broadcast (can also broadcast to self)
                const T readValue = __shfl_sync(mask, val, srcLane);
                // Only accumulate if srcLane is less than this lane
                if (srcLane < laneId)
                {
                    result = INTF::doOp(result, readValue);
                }
                remaining &= ~laneBit;
            }
        }
    }
    return result;
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpCopy(T* dst, const T* src)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        dst[j] = src[j];
    }
}


template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpDoInverse(T* inOut, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        inOut[j] = INTF::doInverse(inOut[j], val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _waveOpSetInitial(T* out, const T* val)
{
    for (size_t j = 0; j < COUNT; ++j)
    {
        out[j] = INTF::getInitial(val[j]);
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixInvertableMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();
    T originalVal[COUNT];
    _waveOpCopy<INTF, T, COUNT>(originalVal, val);

    if (offsetSize > 0)
    {
        // Sum is calculated inclusive of this lanes value
        for (int i = 1; i < offsetSize; i += i)
        {
            // TODO(JS): Note that here I don't split the laneId outside so it's only tested once.
            // This may be better but it would also mean that there would be shfl between lanes
            // that are on different (albeit identical) instructions. So this seems more likely to
            // work as expected with everything in lock step.
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, val[j], i, offsetSize);
                if (laneId >= i)
                {
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
        // Remove originalVal from the result, by applyin inverse
        _waveOpDoInverse<INTF, T, COUNT>(val, originalVal);
    }
    else
    {
        _waveOpSetInitial<INTF, T, COUNT>(val, val);
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, originalVal[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                    remaining &= ~laneBit;
                }
            }
        }
    }
}

template<typename INTF, typename T, size_t COUNT>
__device__ T _wavePrefixMultiple(WarpMask mask, T* val)
{
    const int offsetSize = _waveCalcPow2Offset(mask);

    const int laneId = _getLaneId();

    T work[COUNT];
    _waveOpCopy<INTF, T, COUNT>(work, val);
    _waveOpSetInitial<INTF, T, COUNT>(val, val);

    if (offsetSize > 0)
    {
        // For transmitted value we will do it inclusively with this lanes value
        // For the result we do not include the lanes value. This means an extra op for each
        // iteration but means we don't need to have a divide at the end and also removes overflow
        // issues in that scenario.
        for (int i = 1; i < offsetSize; i += i)
        {
            for (size_t j = 0; j < COUNT; ++j)
            {
                const T readVal = __shfl_up_sync(mask, work[j], i, offsetSize);
                if (laneId >= i)
                {
                    work[j] = INTF::doOp(work[j], readVal);
                    val[j] = INTF::doOp(val[j], readVal);
                }
            }
        }
    }
    else
    {
        if (!_waveIsSingleLane(mask))
        {
            int remaining = mask;
            while (remaining)
            {
                const int laneBit = remaining & -remaining;
                // Get the sourceLane
                const int srcLane = __ffs(laneBit) - 1;

                for (size_t j = 0; j < COUNT; ++j)
                {
                    // Broadcast (can also broadcast to self)
                    const T readValue = __shfl_sync(mask, work[j], srcLane);
                    // Only accumulate if srcLane is less than this lane
                    if (srcLane < laneId)
                    {
                        val[j] = INTF::doOp(val[j], readValue);
                    }
                }
                remaining &= ~laneBit;
            }
        }
    }
}

template<typename T>
__inline__ __device__ T _wavePrefixProduct(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpMul<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixSum(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpAdd<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixXor(WarpMask mask, T val)
{
    return _wavePrefixInvertableScalar<WaveOpXor<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixOr(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpOr<T>, T>(mask, val);
}

template<typename T>
__inline__ __device__ T _wavePrefixAnd(WarpMask mask, T val)
{
    return _wavePrefixScalar<WaveOpAnd<T>, T>(mask, val);
}


template<typename T>
__inline__ __device__ T _wavePrefixProductMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpMul<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixSumMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpAdd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixXorMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixInvertableMultiple<WaveOpXor<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixOrMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpOr<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ T _wavePrefixAndMultiple(WarpMask mask, T val)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    _wavePrefixMultiple<WaveOpAnd<ElemType>, ElemType, sizeof(T) / sizeof(ElemType)>(
        mask,
        (ElemType*)&val);
    return val;
}

template<typename T>
__inline__ __device__ uint4 _waveMatchScalar(WarpMask mask, T val)
{
    int pred;
    return make_uint4(__match_all_sync(mask, val, &pred), 0, 0, 0);
}

template<typename T>
__inline__ __device__ uint4 _waveMatchMultiple(WarpMask mask, const T& inVal)
{
    typedef typename ElementTypeTrait<T>::Type ElemType;
    const size_t count = sizeof(T) / sizeof(ElemType);
    int pred;
    const ElemType* src = (const ElemType*)&inVal;
    uint matchBits = 0xffffffff;
    for (size_t i = 0; i < count && matchBits; ++i)
    {
        matchBits = matchBits & __match_all_sync(mask, src[i], &pred);
    }
    return make_uint4(matchBits, 0, 0, 0);
}

__device__ uint getAt(dim3 a, int b)
{
    SLANG_PRELUDE_ASSERT(b >= 0 && b < 3);
    return (&a.x)[b];
}
__device__ uint3 operator*(uint3 a, dim3 b)
{
    uint3 r;
    r.x = a.x * b.x;
    r.y = a.y * b.y;
    r.z = a.z * b.z;
    return r;
}

template<typename TResult, typename TInput>
__inline__ __device__ TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */


/* Type that defines the uniform entry point params. The actual content of this type is dependent on
the entry point parameters, and can be found via reflection or defined such that it matches the
shader appropriately.
*/
struct UniformEntryPointParams;
struct UniformState;

// ---------------------- OptiX Ray Payload --------------------------------------
#ifdef SLANG_CUDA_ENABLE_OPTIX
struct RayDesc
{
    float3 Origin;
    float TMin;
    float3 Direction;
    float TMax;
};

static __forceinline__ __device__ void* unpackOptiXRayPayloadPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void* ptr = reinterpret_cast<void*>(uptr);
    return ptr;
}

static __forceinline__ __device__ void packOptiXRayPayloadPointer(
    void* ptr,
    uint32_t& i0,
    uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* getOptiXRayPayloadPtr()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackOptiXRayPayloadPointer(u0, u1);
}

template<typename T>
__forceinline__ __device__ void* traceOptiXRay(
    OptixTraversableHandle AccelerationStructure,
    uint32_t RayFlags,
    uint32_t InstanceInclusionMask,
    uint32_t RayContributionToHitGroupIndex,
    uint32_t MultiplierForGeometryContributionToHitGroupIndex,
    uint32_t MissShaderIndex,
    RayDesc Ray,
    T* Payload)
{
    uint32_t r0, r1;
    packOptiXRayPayloadPointer((void*)Payload, r0, r1);
    optixTrace(
        AccelerationStructure,
        Ray.Origin,
        Ray.Direction,
        Ray.TMin,
        Ray.TMax,
        0.f, /* Time for motion blur, currently unsupported in slang */
        InstanceInclusionMask,
        RayFlags,
        RayContributionToHitGroupIndex,
        MultiplierForGeometryContributionToHitGroupIndex,
        MissShaderIndex,
        r0,
        r1);
}

#endif

static const int kSlangTorchTensorMaxDim = 5;

// TensorView
struct TensorView
{
    uint8_t* data;
    uint32_t strides[kSlangTorchTensorMaxDim];
    uint32_t sizes[kSlangTorchTensorMaxDim];
    uint32_t dimensionCount;

    template<typename T>
    __device__ T* data_ptr()
    {
        return reinterpret_cast<T*>(data);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint32_t index)
    {
        uint64_t offset = strides[0] * index;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint2 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint3 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T* data_ptr_at(uint4 index)
    {
        uint64_t offset = strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
                          strides[3] * index.w;
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T, unsigned int N>
    __device__ T* data_ptr_at(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ T& load(uint32_t x)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y);
    }
    template<typename T>
    __device__ T& load(uint2 index)
    {
        return *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z)
    {
        return *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z);
    }
    template<typename T>
    __device__ T& load(uint3 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z);
    }
    template<typename T>
    __device__ T& load(uint32_t x, uint32_t y, uint32_t z, uint32_t w)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w);
    }
    template<typename T>
    __device__ T& load(uint4 index)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w);
    }
    template<typename T>
    __device__ T& load(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4)
    {
        return *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4);
    }

    // Generic version of load
    template<typename T, unsigned int N>
    __device__ T& load(uint index[N])
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        return *reinterpret_cast<T*>(data + offset);
    }

    template<typename T>
    __device__ void store(uint32_t x, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y) = val;
    }
    template<typename T>
    __device__ void store(uint2 index, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * index.x + strides[1] * index.y) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, T val)
    {
        *reinterpret_cast<T*>(data + strides[0] * x + strides[1] * y + strides[2] * z) = val;
    }
    template<typename T>
    __device__ void store(uint3 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z) = val;
    }
    template<typename T>
    __device__ void store(uint32_t x, uint32_t y, uint32_t z, uint32_t w, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * x + strides[1] * y + strides[2] * z + strides[3] * w) = val;
    }
    template<typename T>
    __device__ void store(uint4 index, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * index.x + strides[1] * index.y + strides[2] * index.z +
            strides[3] * index.w) = val;
    }
    template<typename T>
    __device__ void store(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t i4, T val)
    {
        *reinterpret_cast<T*>(
            data + strides[0] * i0 + strides[1] * i1 + strides[2] * i2 + strides[3] * i3 +
            strides[4] * i4) = val;
    }

    // Generic version
    template<typename T, unsigned int N>
    __device__ void store(uint index[N], T val)
    {
        uint64_t offset = 0;
        for (unsigned int i = 0; i < N; ++i)
        {
            offset += strides[i] * index[i];
        }
        *reinterpret_cast<T*>(data + offset) = val;
    }
};


#line 694 "diff.meta.slang"
struct AtomicAdd_0
{
    TensorView diff_0;
};


#line 707
__device__ float AtomicAdd_load_forward_0(AtomicAdd_0 this_0, uint3  i_0)
{
    float _S1 = ((this_0.diff_0).load<float>((i_0)));

#line 709
    return _S1;
}


#line 707
__device__ float AtomicAdd_load_forward_1(AtomicAdd_0 this_1, uint2  i_1)
{
    float _S2 = ((this_1.diff_0).load<float>((i_1)));

#line 709
    return _S2;
}


#line 720
__device__ void AtomicAdd_load_backward_0(AtomicAdd_0 this_2, uint3  i_2, float dOut_0)
{
    float oldVal_0;
    *((&oldVal_0)) = atomicAdd((this_2.diff_0).data_ptr_at<float>((i_2)), (dOut_0));
    return;
}


#line 720
__device__ void AtomicAdd_load_backward_1(AtomicAdd_0 this_3, uint2  i_3, float dOut_1)
{
    float oldVal_1;
    *((&oldVal_1)) = atomicAdd((this_3.diff_0).data_ptr_at<float>((i_3)), (dOut_1));
    return;
}


#line 790
__device__ void AtomicAdd_storeOnce_forward_0(AtomicAdd_0 this_4, uint3  i_4, float dx_0)
{
    (this_4.diff_0).store<float>((i_4), (dx_0));
    return;
}


#line 802
__device__ float AtomicAdd_storeOnce_backward_0(AtomicAdd_0 this_5, uint3  i_5)
{
    float _S3 = ((this_5.diff_0).load<float>((i_5)));

#line 804
    return _S3;
}



struct DiffTensorView_0
{
    TensorView primal_0;
    AtomicAdd_0 diff_1;
};


#line 814
__device__ uint DiffTensorView_size_0(DiffTensorView_0 this_6, uint i_6)
{
    uint _S4 = ((this_6.primal_0).sizes[(i_6)]);

#line 816
    return _S4;
}


#line 234 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
struct Splat_2D_AlphaBlend_0
{
    float3  xyz_vs_0;
    float3  rgb_0;
    float opacity_0;
    Matrix<float, 2, 2>  inv_cov_vs_0;
};


#line 234
__device__ Splat_2D_AlphaBlend_0 Splat_2D_AlphaBlend_x24_syn_dzero_0()
{

#line 234
    Splat_2D_AlphaBlend_0 result_0;

#line 1751 "core.meta.slang"
    float3  _S5 = make_float3 (0.0f);

#line 1751
    (&result_0)->xyz_vs_0 = _S5;

#line 1751
    (&result_0)->rgb_0 = _S5;

#line 1751
    (&result_0)->opacity_0 = 0.0f;

#line 1751
    (&result_0)->inv_cov_vs_0 = makeMatrix<float, 2, 2> (0.0f);

#line 1751
    return result_0;
}


#line 850 "diff.meta.slang"
__device__ float DiffTensorView_load_0(DiffTensorView_0 this_7, uint3  i_7)
{

#line 850
    float _S6 = ((this_7.primal_0).load<float>((i_7)));

#line 850
    return _S6;
}


#line 850
__device__ float DiffTensorView_load_1(DiffTensorView_0 this_8, uint2  i_8)
{

#line 850
    float _S7 = ((this_8.primal_0).load<float>((i_8)));

#line 850
    return _S7;
}


#line 23 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
__device__ __shared__ FixedArray<uint, 256>  collected_idx_0;


#line 22
__device__ __shared__ FixedArray<Splat_2D_AlphaBlend_0, 256>  collected_splats_0;


#line 26 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ float3  read_t3_float3_0(uint idx_0, DiffTensorView_0 t3_0)
{
    return make_float3 (DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 0U)), DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 1U)), DiffTensorView_load_1(t3_0, make_uint2 (idx_0, 2U)));
}


#line 20
__device__ float read_t1_float_0(uint idx_1, DiffTensorView_0 t1_0)
{
    return DiffTensorView_load_1(t1_0, make_uint2 (idx_1, 0U));
}


#line 52
__device__ Matrix<float, 2, 2>  read_t2x2_float2x2_0(uint idx_2, DiffTensorView_0 t2x2_0)
{
    return makeMatrix<float, 2, 2> (DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 0U, 0U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 1U, 0U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 0U, 1U)), DiffTensorView_load_0(t2x2_0, make_uint3 (idx_2, 1U, 1U)));
}


#line 243
__device__ Splat_2D_AlphaBlend_0 load_splat_alphablend_0(int g_idx_0, DiffTensorView_0 xyz_vs_1, DiffTensorView_0 inv_cov_vs_1, DiffTensorView_0 opacity_1, DiffTensorView_0 rgb_1)
{

#line 249
    uint _S8 = uint(g_idx_0);

#line 254
    Splat_2D_AlphaBlend_0 _S9 = { read_t3_float3_0(_S8, xyz_vs_1), read_t3_float3_0(_S8, rgb_1), read_t1_float_0(_S8, opacity_1), read_t2x2_float2x2_0(_S8, inv_cov_vs_1) };

#line 254
    return _S9;
}


#line 61
__device__ float ndc2pix_0(float v_0, int S_0)
{
    return ((v_0 + 1.0f) * float(S_0) - 1.0f) * 0.5f;
}


#line 63
struct DiffPair_float_0
{
    float primal_1;
    float differential_0;
};


#line 1 "token paste"
__device__ void _d_exp_0(DiffPair_float_0 * dpx_0, float dOut_2)
{

#line 1719 "diff.meta.slang"
    float _S10 = (F32_exp(((*dpx_0).primal_1))) * dOut_2;

#line 1719
    dpx_0->primal_1 = (*dpx_0).primal_1;

#line 1719
    dpx_0->differential_0 = _S10;



    return;
}


#line 1 "token paste"
__device__ DiffPair_float_0 _d_exp_1(DiffPair_float_0 dpx_1)
{

#line 1692 "diff.meta.slang"
    float _S11 = (F32_exp((dpx_1.primal_1)));

#line 1692
    DiffPair_float_0 _S12 = { _S11, _S11 * dpx_1.differential_0 };

#line 1692
    return _S12;
}


#line 1960
__device__ void _d_min_0(DiffPair_float_0 * dpx_2, DiffPair_float_0 * dpy_0, float dOut_3)
{
    DiffPair_float_0 _S13 = *dpx_2;

#line 1962
    float _S14;

#line 1962
    if((*dpx_2).primal_1 < (*dpy_0).primal_1)
    {

#line 1962
        _S14 = dOut_3;

#line 1962
    }
    else
    {

#line 1962
        _S14 = 0.0f;

#line 1962
    }

#line 1962
    dpx_2->primal_1 = _S13.primal_1;

#line 1962
    dpx_2->differential_0 = _S14;
    DiffPair_float_0 _S15 = *dpy_0;

#line 1963
    if((*dpy_0).primal_1 < _S13.primal_1)
    {

#line 1963
        _S14 = dOut_3;

#line 1963
    }
    else
    {

#line 1963
        _S14 = 0.0f;

#line 1963
    }

#line 1963
    dpy_0->primal_1 = _S15.primal_1;

#line 1963
    dpy_0->differential_0 = _S14;
    return;
}


#line 1948
__device__ DiffPair_float_0 _d_min_1(DiffPair_float_0 dpx_3, DiffPair_float_0 dpy_1)
{

    float _S16 = (F32_min((dpx_3.primal_1), (dpy_1.primal_1)));

#line 1951
    float _S17;
    if(dpx_3.primal_1 < dpy_1.primal_1)
    {

#line 1952
        _S17 = dpx_3.differential_0;

#line 1952
    }
    else
    {

#line 1952
        _S17 = dpy_1.differential_0;

#line 1952
    }

#line 1952
    DiffPair_float_0 _S18 = { _S16, _S17 };

#line 1950
    return _S18;
}


#line 258 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ float4  evaluate_splat_0(Splat_2D_AlphaBlend_0 g_0, float2  pix_coord_0, uint H_0, uint W_0)
{

#line 269
    float _S19 = pix_coord_0.x - ndc2pix_0(g_0.xyz_vs_0.x, int(W_0));
    float _S20 = pix_coord_0.y - ndc2pix_0(g_0.xyz_vs_0.y, int(H_0));


    float alpha_0 = (F32_min((0.99000000953674316f), (g_0.opacity_0 * (F32_exp((-0.5f * (g_0.inv_cov_vs_0.rows[int(0)].x * _S19 * _S19 + g_0.inv_cov_vs_0.rows[int(1)].y * _S20 * _S20 + (g_0.inv_cov_vs_0.rows[int(0)].y + g_0.inv_cov_vs_0.rows[int(1)].x) * _S19 * _S20)))))));


    return make_float4 ((g_0.rgb_0 * make_float3 (alpha_0)).x, (g_0.rgb_0 * make_float3 (alpha_0)).y, (g_0.rgb_0 * make_float3 (alpha_0)).z, alpha_0);
}


#line 33 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
__device__ float4  undo_pixel_state_0(float4  pixel_state_t_n_0, float4  gauss_rgba_t_n_0)
{
    float transmittance_t_nm1_0 = pixel_state_t_n_0.w / (1.0f - gauss_rgba_t_n_0.w);

    return make_float4 ((float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).x, (float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).y, (float3 {pixel_state_t_n_0.x, pixel_state_t_n_0.y, pixel_state_t_n_0.z} - float3 {gauss_rgba_t_n_0.x, gauss_rgba_t_n_0.y, gauss_rgba_t_n_0.z} * make_float3 (transmittance_t_nm1_0)).z, transmittance_t_nm1_0);
}


#line 26
__device__ float4  update_pixel_state_0(float4  pixel_state_t_nm1_0, float4  gauss_rgba_t_n_1)
{
    float _S21 = pixel_state_t_nm1_0.w;

    return make_float4 ((float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S21)).x, (float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S21)).y, (float3 {pixel_state_t_nm1_0.x, pixel_state_t_nm1_0.y, pixel_state_t_nm1_0.z} + float3 {gauss_rgba_t_n_1.x, gauss_rgba_t_n_1.y, gauss_rgba_t_n_1.z} * make_float3 (_S21)).z, _S21 * (1.0f - gauss_rgba_t_n_1.w));
}


#line 216
struct DiffPair_vectorx3Cfloatx2C2x3E_0
{
    float2  primal_1;
    float2  differential_0;
};


#line 77
struct DiffPair_Splat_2D_AlphaBlend_0
{
    Splat_2D_AlphaBlend_0 primal_1;
    Splat_2D_AlphaBlend_0 differential_0;
};


#line 180
struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_1;
    float4  differential_0;
};


#line 26
__device__ void s_bwd_prop_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dppixel_state_t_nm1_0, DiffPair_vectorx3Cfloatx2C4x3E_0 * dpgauss_rgba_t_n_0, float4  _s_dOut_0)
{
    float _S22 = (*dppixel_state_t_nm1_0).primal_1.w;

    float3  s_diff_color_t_n_T_0 = float3 {_s_dOut_0.x, _s_dOut_0.y, _s_dOut_0.z};

#line 28
    float3  _S23 = float3 {(*dpgauss_rgba_t_n_0).primal_1.x, (*dpgauss_rgba_t_n_0).primal_1.y, (*dpgauss_rgba_t_n_0).primal_1.z} * s_diff_color_t_n_T_0;

#line 28
    float3  _S24 = make_float3 (_S22) * s_diff_color_t_n_T_0;

#line 28
    float _S25 = (1.0f - (*dpgauss_rgba_t_n_0).primal_1.w) * _s_dOut_0.w + _S23.x + _S23.y + _S23.z;

#line 28
    float4  _S26 = make_float4 (_S24.x, _S24.y, _S24.z, - (_S22 * _s_dOut_0.w));

#line 28
    dpgauss_rgba_t_n_0->primal_1 = (*dpgauss_rgba_t_n_0).primal_1;

#line 28
    dpgauss_rgba_t_n_0->differential_0 = _S26;

#line 28
    float4  _S27 = make_float4 (s_diff_color_t_n_T_0.x, s_diff_color_t_n_T_0.y, s_diff_color_t_n_T_0.z, _S25);

#line 28
    dppixel_state_t_nm1_0->primal_1 = (*dppixel_state_t_nm1_0).primal_1;

#line 28
    dppixel_state_t_nm1_0->differential_0 = _S27;

#line 26
    return;
}


#line 26
__device__ void s_bwd_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * _S28, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S29, float4  _S30)
{

#line 26
    s_bwd_prop_update_pixel_state_0(_S28, _S29, _S30);

#line 26
    return;
}


#line 186
__device__ float s_primal_ctx_ndc2pix_0(float dpv_0, int S_1)
{

#line 61 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    return ((dpv_0 + 1.0f) * float(S_1) - 1.0f) * 0.5f;
}


#line 61
__device__ float s_primal_ctx_exp_0(float _S31)
{

#line 61
    return (F32_exp((_S31)));
}


#line 61
__device__ float s_primal_ctx_min_0(float _S32, float _S33)
{

#line 61
    return (F32_min((_S32), (_S33)));
}


#line 239 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S34, DiffPair_float_0 * _S35, float _S36)
{

#line 239
    _d_min_0(_S34, _S35, _S36);

#line 239
    return;
}


#line 239
__device__ void s_bwd_prop_exp_0(DiffPair_float_0 * _S37, float _S38)
{

#line 239
    _d_exp_0(_S37, _S38);

#line 239
    return;
}


#line 61 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ void s_bwd_prop_ndc2pix_0(DiffPair_float_0 * dpv_1, int S_2, float _s_dOut_1)
{
    float _S39 = float(S_2) * (0.5f * _s_dOut_1);

#line 63
    dpv_1->primal_1 = (*dpv_1).primal_1;

#line 63
    dpv_1->differential_0 = _S39;

#line 61
    return;
}


#line 258
__device__ void s_bwd_prop_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 * dpg_0, DiffPair_vectorx3Cfloatx2C2x3E_0 * dppix_coord_0, uint H_1, uint W_1, float4  _s_dOut_2)
{

#line 269
    float _S40 = (*dpg_0).primal_1.xyz_vs_0.x;

#line 269
    int _S41 = int(W_1);

#line 269
    float _S42 = (*dppix_coord_0).primal_1.x - s_primal_ctx_ndc2pix_0(_S40, _S41);
    float _S43 = (*dpg_0).primal_1.xyz_vs_0.y;

#line 270
    int _S44 = int(H_1);

#line 270
    float _S45 = (*dppix_coord_0).primal_1.y - s_primal_ctx_ndc2pix_0(_S43, _S44);
    float _S46 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].x * _S42;
    float _S47 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].y * _S45;

#line 272
    float _S48 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].y + (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].x;

#line 272
    float _S49 = _S48 * _S42;

#line 271
    float power_0 = -0.5f * (_S46 * _S42 + _S47 * _S45 + _S49 * _S45);

#line 271
    float _S50 = s_primal_ctx_exp_0(power_0);

    float _S51 = (*dpg_0).primal_1.opacity_0 * _S50;


    float3  s_diff_premult_rgb_T_0 = float3 {_s_dOut_2.x, _s_dOut_2.y, _s_dOut_2.z};

#line 274
    float3  _S52 = (*dpg_0).primal_1.rgb_0 * s_diff_premult_rgb_T_0;

#line 274
    float3  _S53 = make_float3 (s_primal_ctx_min_0(0.99000000953674316f, _S51)) * s_diff_premult_rgb_T_0;

#line 273
    float _S54 = _s_dOut_2.w + _S52.x + _S52.y + _S52.z;

#line 273
    DiffPair_float_0 _S55;

#line 273
    (&_S55)->primal_1 = 0.99000000953674316f;

#line 273
    (&_S55)->differential_0 = 0.0f;

#line 273
    DiffPair_float_0 _S56;

#line 273
    (&_S56)->primal_1 = _S51;

#line 273
    (&_S56)->differential_0 = 0.0f;

#line 273
    s_bwd_prop_min_0(&_S55, &_S56, _S54);

#line 273
    float _S57 = (*dpg_0).primal_1.opacity_0 * _S56.differential_0;

#line 273
    float _S58 = _S50 * _S56.differential_0;

#line 273
    DiffPair_float_0 _S59;

#line 273
    (&_S59)->primal_1 = power_0;

#line 273
    (&_S59)->differential_0 = 0.0f;

#line 273
    s_bwd_prop_exp_0(&_S59, _S57);

#line 271
    float _S60 = -0.5f * _S59.differential_0;
    float _S61 = _S49 * _S60;

#line 272
    float _S62 = _S45 * _S60;

#line 272
    float _S63 = _S48 * _S62;

#line 272
    float _S64 = _S42 * _S62;

#line 272
    float _S65 = _S47 * _S60;

#line 272
    float _S66 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(1)].y * _S62;

#line 272
    float _S67 = _S45 * _S62;

#line 1751 "core.meta.slang"
    float2  _S68 = make_float2 (0.0f);

#line 1751
    float2  _S69 = _S68;

#line 1751
    *&((&_S69)->x) = _S64;

#line 1751
    *&((&_S69)->y) = _S67;

#line 271 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    float _S70 = _S46 * _S60;

#line 271
    float _S71 = _S42 * _S60;

#line 271
    float _S72 = (*dpg_0).primal_1.inv_cov_vs_0.rows[int(0)].x * _S71;

#line 271
    float _S73 = _S42 * _S71;

#line 271
    float2  _S74 = _S68;

#line 271
    *&((&_S74)->y) = _S64;

#line 271
    *&((&_S74)->x) = _S73;

#line 270
    float _S75 = _S61 + _S65 + _S66;

#line 270
    float _S76 = - _S75;

#line 270
    DiffPair_float_0 _S77;

#line 270
    (&_S77)->primal_1 = _S43;

#line 270
    (&_S77)->differential_0 = 0.0f;

#line 270
    s_bwd_prop_ndc2pix_0(&_S77, _S44, _S76);

#line 269
    float _S78 = _S63 + _S70 + _S72;

#line 269
    float _S79 = - _S78;

#line 269
    DiffPair_float_0 _S80;

#line 269
    (&_S80)->primal_1 = _S40;

#line 269
    (&_S80)->differential_0 = 0.0f;

#line 269
    s_bwd_prop_ndc2pix_0(&_S80, _S41, _S79);

#line 269
    Matrix<float, 2, 2>  _S81 = makeMatrix<float, 2, 2> (0.0f);

#line 269
    _S81[int(1)] = _S69;

#line 269
    _S81[int(0)] = _S74;

#line 269
    float3  _S82 = make_float3 (_S80.differential_0, _S77.differential_0, 0.0f);

#line 269
    float2  _S83 = make_float2 (_S78, _S75);

#line 269
    dppix_coord_0->primal_1 = (*dppix_coord_0).primal_1;

#line 269
    dppix_coord_0->differential_0 = _S83;

#line 269
    Splat_2D_AlphaBlend_0 _S84 = Splat_2D_AlphaBlend_x24_syn_dzero_0();

#line 269
    (&_S84)->inv_cov_vs_0 = _S81;

#line 269
    (&_S84)->opacity_0 = _S58;

#line 269
    (&_S84)->rgb_0 = _S53;

#line 269
    (&_S84)->xyz_vs_0 = _S82;

#line 269
    dpg_0->primal_1 = (*dpg_0).primal_1;

#line 269
    dpg_0->differential_0 = _S84;

#line 258
    return;
}


#line 258
__device__ void s_bwd_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 * _S85, DiffPair_vectorx3Cfloatx2C2x3E_0 * _S86, uint _S87, uint _S88, float4  _S89)
{

    s_bwd_prop_evaluate_splat_0(_S85, _S86, _S87, _S88, _S89);

#line 261
    return;
}


#line 52
__device__ void s_bwd_prop_read_t2x2_float2x2_0(uint idx_3, DiffTensorView_0 t2x2_1, Matrix<float, 2, 2>  _s_dOut_3)
{
    uint3  _S90 = make_uint3 (idx_3, 0U, 0U);
    uint3  _S91 = make_uint3 (idx_3, 1U, 0U);
    uint3  _S92 = make_uint3 (idx_3, 0U, 1U);

#line 54
    AtomicAdd_load_backward_0(t2x2_1.diff_1, make_uint3 (idx_3, 1U, 1U), _s_dOut_3.rows[int(1)].y);

#line 54
    AtomicAdd_load_backward_0(t2x2_1.diff_1, _S92, _s_dOut_3.rows[int(1)].x);

#line 54
    AtomicAdd_load_backward_0(t2x2_1.diff_1, _S91, _s_dOut_3.rows[int(0)].y);

#line 54
    AtomicAdd_load_backward_0(t2x2_1.diff_1, _S90, _s_dOut_3.rows[int(0)].x);

#line 52
    return;
}


#line 20
__device__ void s_bwd_prop_read_t1_float_0(uint idx_4, DiffTensorView_0 t1_1, float _s_dOut_4)
{
    AtomicAdd_load_backward_1(t1_1.diff_1, make_uint2 (idx_4, 0U), _s_dOut_4);

#line 20
    return;
}




__device__ void s_bwd_prop_read_t3_float3_0(uint idx_5, DiffTensorView_0 t3_1, float3  _s_dOut_5)
{
    uint2  _S93 = make_uint2 (idx_5, 0U);
    uint2  _S94 = make_uint2 (idx_5, 1U);

#line 28
    AtomicAdd_load_backward_1(t3_1.diff_1, make_uint2 (idx_5, 2U), _s_dOut_5.z);

#line 28
    AtomicAdd_load_backward_1(t3_1.diff_1, _S94, _s_dOut_5.y);

#line 28
    AtomicAdd_load_backward_1(t3_1.diff_1, _S93, _s_dOut_5.x);

#line 26
    return;
}


#line 243
__device__ void s_bwd_prop_load_splat_alphablend_0(int g_idx_1, DiffTensorView_0 xyz_vs_2, DiffTensorView_0 inv_cov_vs_2, DiffTensorView_0 opacity_2, DiffTensorView_0 rgb_2, Splat_2D_AlphaBlend_0 _s_dOut_6)
{

#line 249
    uint _S95 = uint(g_idx_1);

#line 249
    s_bwd_prop_read_t2x2_float2x2_0(_S95, inv_cov_vs_2, _s_dOut_6.inv_cov_vs_0);

#line 249
    s_bwd_prop_read_t1_float_0(_S95, opacity_2, _s_dOut_6.opacity_0);

#line 249
    s_bwd_prop_read_t3_float3_0(_S95, rgb_2, _s_dOut_6.rgb_0);

#line 249
    s_bwd_prop_read_t3_float3_0(_S95, xyz_vs_2, _s_dOut_6.xyz_vs_0);

#line 243
    return;
}


#line 243
__device__ void s_bwd_load_splat_alphablend_0(int _S96, DiffTensorView_0 _S97, DiffTensorView_0 _S98, DiffTensorView_0 _S99, DiffTensorView_0 _S100, Splat_2D_AlphaBlend_0 _S101)
{


    s_bwd_prop_load_splat_alphablend_0(_S96, _S97, _S98, _S99, _S100, _S101);

#line 247
    return;
}


#line 110 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
__device__ void bwd_alpha_blend_0(TensorView sorted_gauss_idx_0, DiffTensorView_0 xyz_vs_3, DiffTensorView_0 inv_cov_vs_3, DiffTensorView_0 opacity_3, DiffTensorView_0 rgb_3, DiffTensorView_0 final_pixel_state_0, TensorView n_contributors_0, uint2  pix_coord_1, uint tile_idx_start_0, uint tile_idx_end_0, uint tile_height_0, uint tile_width_0, uint H_2, uint W_2, float4  d_current_pixel_state_0)
{

#line 127
    uint _S102 = pix_coord_1.x;

#line 127
    bool is_inside_0;

#line 127
    if(_S102 < W_2)
    {

#line 127
        is_inside_0 = pix_coord_1.y < H_2;

#line 127
    }
    else
    {

#line 127
        is_inside_0 = false;

#line 127
    }
    uint block_size_0 = tile_height_0 * tile_width_0;
    uint _S103 = tile_idx_end_0 - tile_idx_start_0;

#line 129
    uint _S104 = (_S103 + block_size_0 - 1U) / block_size_0;

#line 129
    int _S105 = int(_S104);

    int _S106 = int(_S103);

#line 131
    int n_contrib_fwd_0;

#line 131
    float4  current_pixel_state_0;

#line 136
    if(is_inside_0)
    {

#line 137
        uint _S107 = pix_coord_1.y;

#line 137
        float4  _S108 = make_float4 (DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S107, _S102, 0U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S107, _S102, 1U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S107, _S102, 2U)), DiffTensorView_load_0(final_pixel_state_0, make_uint3 (_S107, _S102, 3U)));



        int _S109 = ((n_contributors_0).load<int>((_S107), (_S102), (0U)));

#line 141
        n_contrib_fwd_0 = _S109;

#line 141
        current_pixel_state_0 = _S108;

#line 136
    }

#line 144
    float2  center_pix_coord_0 = make_float2 ((float)pix_coord_1.x, (float)pix_coord_1.y);

    float2  _S110 = make_float2 (0.0f);

#line 146
    DiffPair_vectorx3Cfloatx2C2x3E_0 dp_center_pix_coord_0;

#line 146
    (&dp_center_pix_coord_0)->primal_1 = center_pix_coord_0;

#line 146
    (&dp_center_pix_coord_0)->differential_0 = _S110;


    uint3  _S111 = ((threadIdx));

#line 149
    uint _S112 = _S111.y * ((blockDim)).x + _S111.x;

#line 149
    float4  _S113 = d_current_pixel_state_0;

#line 149
    int i_9 = int(0);

#line 149
    int splats_left_to_process_0 = _S106;

#line 149
    uint current_splat_offset_0 = _S103;

#line 190
    int _S114 = int(block_size_0);

#line 180
    Splat_2D_AlphaBlend_0 _S115 = Splat_2D_AlphaBlend_x24_syn_dzero_0();
    float4  _S116 = make_float4 (0.0f);

#line 150
    for(;;)
    {

#line 150
        if(i_9 < _S105)
        {
        }
        else
        {

#line 150
            break;
        }

        __syncthreads();

        uint _S117 = uint(int(uint(i_9) * block_size_0 + _S112));

#line 155
        if(tile_idx_start_0 + _S117 < tile_idx_end_0)
        {
            int _S118 = ((sorted_gauss_idx_0).load<int>((tile_idx_end_0 - _S117 - 1U)));

#line 157
            uint coll_id_0 = uint(_S118);
            (*&collected_idx_0)[_S112] = coll_id_0;
            (*&collected_splats_0)[_S112] = load_splat_alphablend_0(int(coll_id_0), xyz_vs_3, inv_cov_vs_3, opacity_3, rgb_3);

#line 155
        }

#line 161
        __syncthreads();
        if(is_inside_0)
        {

#line 162
            float4  current_pixel_state_1 = current_pixel_state_0;

#line 162
            float4  _S119 = _S113;

#line 162
            int j_0 = int(0);

#line 162
            uint current_splat_offset_1 = current_splat_offset_0;
            for(;;)
            {

#line 163
                if(uint(j_0) < (U32_min((block_size_0), (uint(splats_left_to_process_0)))))
                {
                }
                else
                {

#line 163
                    break;
                }
                uint current_splat_offset_2 = current_splat_offset_1 - 1U;
                if(current_splat_offset_2 >= uint(n_contrib_fwd_0))
                {

#line 167
                    j_0 = j_0 + int(1);

#line 167
                    current_splat_offset_1 = current_splat_offset_2;

#line 163
                    continue;
                }



                uint g_idx_2 = (*&collected_idx_0)[j_0];
                Splat_2D_AlphaBlend_0 g_1 = (*&collected_splats_0)[j_0];

                float4  gauss_rgba_0 = evaluate_splat_0((*&collected_splats_0)[j_0], center_pix_coord_0, H_2, W_2);

                if(gauss_rgba_0.w < 0.00392156885936856f)
                {

#line 174
                    j_0 = j_0 + int(1);

#line 174
                    current_splat_offset_1 = current_splat_offset_2;

#line 163
                    continue;
                }

#line 177
                float4  current_pixel_state_2 = undo_pixel_state_0(current_pixel_state_1, gauss_rgba_0);


                DiffPair_Splat_2D_AlphaBlend_0 dp_g_0;

#line 180
                (&dp_g_0)->primal_1 = g_1;

#line 180
                (&dp_g_0)->differential_0 = _S115;
                DiffPair_vectorx3Cfloatx2C4x3E_0 dp_gauss_rgba_0;

#line 181
                (&dp_gauss_rgba_0)->primal_1 = gauss_rgba_0;

#line 181
                (&dp_gauss_rgba_0)->differential_0 = _S116;
                DiffPair_vectorx3Cfloatx2C4x3E_0 dp_current_pixel_state_0;

#line 182
                (&dp_current_pixel_state_0)->primal_1 = current_pixel_state_2;

#line 182
                (&dp_current_pixel_state_0)->differential_0 = _S116;

                s_bwd_update_pixel_state_0(&dp_current_pixel_state_0, &dp_gauss_rgba_0, _S119);

                s_bwd_evaluate_splat_0(&dp_g_0, &dp_center_pix_coord_0, H_2, W_2, dp_gauss_rgba_0.differential_0);
                s_bwd_load_splat_alphablend_0(int(g_idx_2), xyz_vs_3, inv_cov_vs_3, opacity_3, rgb_3, dp_g_0.differential_0);

#line 187
                current_pixel_state_1 = current_pixel_state_2;

#line 187
                _S119 = dp_current_pixel_state_0.differential_0;

#line 163
                j_0 = j_0 + int(1);

#line 163
                current_splat_offset_1 = current_splat_offset_2;

#line 163
            }

#line 163
            current_pixel_state_0 = current_pixel_state_1;

#line 163
            _S113 = _S119;

#line 163
            current_splat_offset_0 = current_splat_offset_1;

#line 162
        }

#line 190
        int splats_left_to_process_1 = splats_left_to_process_0 - _S114;

#line 150
        i_9 = i_9 + int(1);

#line 150
        splats_left_to_process_0 = splats_left_to_process_1;

#line 150
    }

#line 192
    return;
}


#line 41
__device__ float4  alpha_blend_0(TensorView sorted_gauss_idx_1, DiffTensorView_0 xyz_vs_4, DiffTensorView_0 inv_cov_vs_4, DiffTensorView_0 opacity_4, DiffTensorView_0 rgb_4, DiffTensorView_0 final_pixel_state_1, TensorView n_contributors_1, uint2  pix_coord_2, uint tile_idx_start_1, uint tile_idx_end_1, uint tile_height_1, uint tile_width_1, uint H_3, uint W_3)
{

#line 56
    float2  _S120 = make_float2 ((float)pix_coord_2.x, (float)pix_coord_2.y);
    float4  _S121 = make_float4 (0.0f, 0.0f, 0.0f, 1.0f);
    uint block_size_1 = tile_height_1 * tile_width_1;
    uint _S122 = pix_coord_2.x;

#line 59
    bool is_inside_1;

#line 59
    if(_S122 < W_3)
    {

#line 59
        is_inside_1 = pix_coord_2.y < H_3;

#line 59
    }
    else
    {

#line 59
        is_inside_1 = false;

#line 59
    }


    uint _S123 = tile_idx_end_1 - tile_idx_start_1;

#line 62
    uint _S124 = (_S123 + block_size_1 - 1U) / block_size_1;

#line 62
    int _S125 = int(_S124);

    uint3  _S126 = ((threadIdx));

#line 64
    uint _S127 = _S126.y * ((blockDim)).x + _S126.x;


    int _S128 = int(_S123);

#line 67
    bool thread_active_0 = is_inside_1;

#line 67
    float4  curr_pixel_state_0 = _S121;

#line 67
    int i_10 = int(0);

#line 67
    int splats_left_to_process_2 = _S128;

#line 67
    int local_n_contrib_0 = int(0);

#line 101
    int _S129 = int(block_size_1);

#line 68
    for(;;)
    {

#line 68
        if(i_10 < _S125)
        {
        }
        else
        {

#line 68
            break;
        }

        __syncthreads();

        uint _S130 = tile_idx_start_1 + uint(int(uint(i_10) * block_size_1 + _S127));

#line 73
        if(_S130 < tile_idx_end_1)
        {

            int _S131 = ((sorted_gauss_idx_1).load<int>((_S130)));
            (*&collected_splats_0)[_S127] = load_splat_alphablend_0(int(uint(_S131)), xyz_vs_4, inv_cov_vs_4, opacity_4, rgb_4);

#line 73
        }

#line 79
        __syncthreads();

#line 79
        float4  curr_pixel_state_1;
        if(thread_active_0)
        {

#line 80
            int local_n_contrib_1;

#line 80
            bool thread_active_1;

#line 80
            curr_pixel_state_1 = curr_pixel_state_0;

#line 80
            int j_1 = int(0);

#line 80
            int local_n_contrib_2 = local_n_contrib_0;
            for(;;)
            {

#line 81
                if(uint(j_1) < (U32_min((block_size_1), (uint(splats_left_to_process_2)))))
                {
                }
                else
                {

#line 81
                    thread_active_1 = thread_active_0;

#line 81
                    local_n_contrib_1 = local_n_contrib_2;

#line 81
                    break;
                }
                int local_n_contrib_3 = local_n_contrib_2 + int(1);

                float4  gauss_rgba_1 = evaluate_splat_0((*&collected_splats_0)[j_1], _S120, H_3, W_3);


                if(gauss_rgba_1.w < 0.00392156885936856f)
                {

#line 89
                    j_1 = j_1 + int(1);

#line 89
                    local_n_contrib_2 = local_n_contrib_3;

#line 81
                    continue;
                }

#line 91
                float4  new_pixel_state_0 = update_pixel_state_0(curr_pixel_state_1, gauss_rgba_1);
                if(new_pixel_state_0.w < 0.00009999999747379f)
                {
                    int _S132 = local_n_contrib_3 - int(1);

#line 94
                    thread_active_1 = false;

#line 94
                    local_n_contrib_1 = _S132;

                    break;
                }

#line 96
                curr_pixel_state_1 = new_pixel_state_0;

#line 81
                j_1 = j_1 + int(1);

#line 81
                local_n_contrib_2 = local_n_contrib_3;

#line 81
            }

#line 81
            thread_active_0 = thread_active_1;

#line 81
            local_n_contrib_0 = local_n_contrib_1;

#line 80
        }
        else
        {

#line 80
            curr_pixel_state_1 = curr_pixel_state_0;

#line 80
        }

#line 101
        int splats_left_to_process_3 = splats_left_to_process_2 - _S129;

#line 68
        int _S133 = i_10 + int(1);

#line 68
        curr_pixel_state_0 = curr_pixel_state_1;

#line 68
        i_10 = _S133;

#line 68
        splats_left_to_process_2 = splats_left_to_process_3;

#line 68
    }

#line 104
    if(is_inside_1)
    {

#line 105
        (n_contributors_1).store<int>((pix_coord_2.y), (_S122), (0U), (local_n_contrib_0));

#line 104
    }


    return curr_pixel_state_0;
}


#line 1035 "diff.meta.slang"
__device__ void DiffTensorView_storeOnce_forward_0(DiffTensorView_0 this_9, uint3  x_0, DiffPair_float_0 dpval_0)
{
    (this_9.primal_0).store<float>((x_0), (dpval_0.primal_1));
    AtomicAdd_storeOnce_forward_0(this_9.diff_1, x_0, dpval_0.differential_0);
    return;
}


#line 1026
__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_10, uint3  x_1, float val_0)
{

#line 1026
    (this_10.primal_0).store<float>((x_1), (val_0));

#line 1026
    return;
}


#line 197 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
struct s_bwd_prop_splat_tiled_Intermediates_0
{
    int _S134;
    int _S135;
};


#line 197
__device__ float4  s_primal_ctx_alpha_blend_0(TensorView _S136, DiffTensorView_0 _S137, DiffTensorView_0 _S138, DiffTensorView_0 _S139, DiffTensorView_0 _S140, DiffTensorView_0 _S141, TensorView _S142, uint2  _S143, uint _S144, uint _S145, uint _S146, uint _S147, uint _S148, uint _S149)
{

#line 197
    float4  _S150 = alpha_blend_0(_S136, _S137, _S138, _S139, _S140, _S141, _S142, _S143, _S144, _S145, _S146, _S147, _S148, _S149);

#line 197
    return _S150;
}


#line 197
__device__ void s_primal_ctx_splat_tiled_0(TensorView sorted_gauss_idx_2, TensorView tile_ranges_0, DiffTensorView_0 xyz_vs_5, DiffTensorView_0 inv_cov_vs_5, DiffTensorView_0 opacity_5, DiffTensorView_0 rgb_5, DiffTensorView_0 output_img_0, TensorView n_contributors_2, int grid_height_0, int grid_width_0, int tile_height_2, int tile_width_2, s_bwd_prop_splat_tiled_Intermediates_0 * _s_diff_ctx_0)
{

#line 208
    _s_diff_ctx_0->_S134 = int(0);

#line 208
    _s_diff_ctx_0->_S135 = int(0);

#line 215
    _s_diff_ctx_0->_S134 = int(0);
    _s_diff_ctx_0->_S135 = int(0);

#line 210
    uint3  _S151 = ((blockIdx));

    uint2  pix_coord_3 = uint2 {(_S151 * ((blockDim)) + ((threadIdx))).x, (_S151 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_0 = _S151.y * uint(grid_width_0) + _S151.x;
    int _S152 = ((tile_ranges_0).load<int>((tile_idx_0), (0U)));

#line 215
    _s_diff_ctx_0->_S134 = _S152;

#line 215
    uint tile_idx_start_2 = uint(_S152);
    int _S153 = ((tile_ranges_0).load<int>((tile_idx_0), (1U)));

#line 216
    _s_diff_ctx_0->_S135 = _S153;

#line 216
    uint tile_idx_end_2 = uint(_S153);

    uint _S154 = pix_coord_3.x;

#line 218
    uint _S155 = DiffTensorView_size_0(output_img_0, 1U);

#line 218
    bool is_inside_2;

#line 218
    if(_S154 < _S155)
    {

#line 218
        is_inside_2 = pix_coord_3.y < DiffTensorView_size_0(output_img_0, 0U);

#line 218
    }
    else
    {

#line 218
        is_inside_2 = false;

#line 218
    }

#line 218
    float4  _S156 = s_primal_ctx_alpha_blend_0(sorted_gauss_idx_2, xyz_vs_5, inv_cov_vs_5, opacity_5, rgb_5, output_img_0, n_contributors_2, pix_coord_3, tile_idx_start_2, tile_idx_end_2, uint(tile_height_2), uint(tile_width_2), DiffTensorView_size_0(output_img_0, 0U), _S155);

#line 235
    if(is_inside_2)
    {

#line 236
        uint _S157 = pix_coord_3.y;

#line 236
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S157, _S154, 0U), _S156.x);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S157, _S154, 1U), _S156.y);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S157, _S154, 2U), _S156.z);
        DiffTensorView_storeOnce_0(output_img_0, make_uint3 (_S157, _S154, 3U), _S156.w);

#line 235
    }

#line 235
    return;
}


#line 235
__device__ void s_bwd_prop_alpha_blend_0(TensorView _S158, DiffTensorView_0 _S159, DiffTensorView_0 _S160, DiffTensorView_0 _S161, DiffTensorView_0 _S162, DiffTensorView_0 _S163, TensorView _S164, uint2  _S165, uint _S166, uint _S167, uint _S168, uint _S169, uint _S170, uint _S171, float4  _S172)
{

#line 235
    bwd_alpha_blend_0(_S158, _S159, _S160, _S161, _S162, _S163, _S164, _S165, _S166, _S167, _S168, _S169, _S170, _S171, _S172);

#line 235
    return;
}


#line 197
__device__ void s_bwd_prop_splat_tiled_0(TensorView sorted_gauss_idx_3, TensorView tile_ranges_1, DiffTensorView_0 xyz_vs_6, DiffTensorView_0 inv_cov_vs_6, DiffTensorView_0 opacity_6, DiffTensorView_0 rgb_6, DiffTensorView_0 output_img_1, TensorView n_contributors_3, int grid_height_1, int grid_width_1, int tile_height_3, int tile_width_3, s_bwd_prop_splat_tiled_Intermediates_0 _s_diff_ctx_1)
{

#line 236
    uint3  _S173 = make_uint3 (0U);

#line 212
    uint2  pix_coord_4 = uint2 {(((blockIdx)) * ((blockDim)) + ((threadIdx))).x, (((blockIdx)) * ((blockDim)) + ((threadIdx))).y};


    uint tile_idx_start_3 = uint(_s_diff_ctx_1._S134);
    uint tile_idx_end_3 = uint(_s_diff_ctx_1._S135);

    uint _S174 = pix_coord_4.x;

#line 218
    uint _S175 = DiffTensorView_size_0(output_img_1, 1U);

#line 218
    bool is_inside_3;

#line 218
    if(_S174 < _S175)
    {

#line 218
        is_inside_3 = pix_coord_4.y < DiffTensorView_size_0(output_img_1, 0U);

#line 218
    }
    else
    {

#line 218
        is_inside_3 = false;

#line 218
    }

#line 230
    uint _S176 = uint(tile_height_3);
    uint _S177 = uint(tile_width_3);
    uint _S178 = DiffTensorView_size_0(output_img_1, 0U);

#line 232
    uint3  _S179;

#line 232
    uint3  _S180;

#line 232
    uint3  _S181;

#line 232
    uint3  _S182;


    if(is_inside_3)
    {

#line 236
        uint _S183 = pix_coord_4.y;

#line 236
        uint3  _S184 = make_uint3 (_S183, _S174, 0U);
        uint3  _S185 = make_uint3 (_S183, _S174, 1U);
        uint3  _S186 = make_uint3 (_S183, _S174, 2U);

#line 238
        _S179 = make_uint3 (_S183, _S174, 3U);

#line 238
        _S180 = _S186;

#line 238
        _S181 = _S185;

#line 238
        _S182 = _S184;

#line 238
    }
    else
    {

#line 238
        _S179 = _S173;

#line 238
        _S180 = _S173;

#line 238
        _S181 = _S173;

#line 238
        _S182 = _S173;

#line 238
    }

#line 220
    float4  _S187 = make_float4 (0.0f);

#line 220
    float4  _S188;

#line 220
    if(is_inside_3)
    {

#line 220
        _S188 = make_float4 (AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S182), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S181), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S180), AtomicAdd_storeOnce_backward_0(output_img_1.diff_1, _S179));

#line 220
    }
    else
    {

#line 220
        _S188 = _S187;

#line 220
    }

#line 220
    s_bwd_prop_alpha_blend_0(sorted_gauss_idx_3, xyz_vs_6, inv_cov_vs_6, opacity_6, rgb_6, output_img_1, n_contributors_3, pix_coord_4, tile_idx_start_3, tile_idx_end_3, _S176, _S177, _S178, _S175, _S188);

#line 197
    return;
}


#line 197
__device__ void s_bwd_splat_tiled_0(TensorView _S189, TensorView _S190, DiffTensorView_0 _S191, DiffTensorView_0 _S192, DiffTensorView_0 _S193, DiffTensorView_0 _S194, DiffTensorView_0 _S195, TensorView _S196, int _S197, int _S198, int _S199, int _S200)
{

#line 208
    s_bwd_prop_splat_tiled_Intermediates_0 _S201;

#line 208
    s_primal_ctx_splat_tiled_0(_S189, _S190, _S191, _S192, _S193, _S194, _S195, _S196, _S197, _S198, _S199, _S200, &_S201);

#line 208
    s_bwd_prop_splat_tiled_0(_S189, _S190, _S191, _S192, _S193, _S194, _S195, _S196, _S197, _S198, _S199, _S200, _S201);

#line 208
    return;
}


#line 208
extern "C" {
__global__ void __kernel__splat_tiled_bwd_diff(TensorView sorted_gauss_idx_4, TensorView tile_ranges_2, DiffTensorView_0 xyz_vs_7, DiffTensorView_0 inv_cov_vs_7, DiffTensorView_0 opacity_7, DiffTensorView_0 rgb_7, DiffTensorView_0 output_img_2, TensorView n_contributors_4, int grid_height_2, int grid_width_2, int tile_height_4, int tile_width_4)
{

#line 208
    s_bwd_splat_tiled_0(sorted_gauss_idx_4, tile_ranges_2, xyz_vs_7, inv_cov_vs_7, opacity_7, rgb_7, output_img_2, n_contributors_4, grid_height_2, grid_width_2, tile_height_4, tile_width_4);

#line 208
    return;
}

}

#line 77
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_1;
    float3  differential_0;
};


#line 249 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_read_t3_float3_0(uint idx_6, DiffTensorView_0 t3_2)
{

#line 28
    uint2  _S202 = make_uint2 (idx_6, 0U);

#line 28
    float _S203 = ((t3_2.primal_0).load<float>((_S202)));

#line 28
    float _S204 = AtomicAdd_load_forward_1(t3_2.diff_1, _S202);
    uint2  _S205 = make_uint2 (idx_6, 1U);

#line 28
    float _S206 = ((t3_2.primal_0).load<float>((_S205)));

#line 28
    float _S207 = AtomicAdd_load_forward_1(t3_2.diff_1, _S205);

    uint2  _S208 = make_uint2 (idx_6, 2U);

#line 28
    float _S209 = ((t3_2.primal_0).load<float>((_S208)));

#line 28
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S210 = { make_float3 (_S203, _S206, _S209), make_float3 (_S204, _S207, AtomicAdd_load_forward_1(t3_2.diff_1, _S208)) };

#line 28
    return _S210;
}


#line 251
__device__ DiffPair_float_0 s_fwd_read_t1_float_0(uint idx_7, DiffTensorView_0 t1_2)
{

#line 22
    uint2  _S211 = make_uint2 (idx_7, 0U);

#line 22
    float _S212 = ((t1_2.primal_0).load<float>((_S211)));

#line 22
    DiffPair_float_0 _S213 = { _S212, AtomicAdd_load_forward_1(t1_2.diff_1, _S211) };

#line 22
    return _S213;
}


#line 22
struct DiffPair_matrixx3Cfloatx2C2x2C2x3E_0
{
    Matrix<float, 2, 2>  primal_1;
    Matrix<float, 2, 2>  differential_0;
};


#line 252
__device__ DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 s_fwd_read_t2x2_float2x2_0(uint idx_8, DiffTensorView_0 t2x2_2)
{

#line 54
    uint3  _S214 = make_uint3 (idx_8, 0U, 0U);

#line 54
    float _S215 = ((t2x2_2.primal_0).load<float>((_S214)));

#line 54
    float _S216 = AtomicAdd_load_forward_0(t2x2_2.diff_1, _S214);
    uint3  _S217 = make_uint3 (idx_8, 1U, 0U);

#line 54
    float _S218 = ((t2x2_2.primal_0).load<float>((_S217)));

#line 54
    float _S219 = AtomicAdd_load_forward_0(t2x2_2.diff_1, _S217);

    uint3  _S220 = make_uint3 (idx_8, 0U, 1U);

#line 54
    float _S221 = ((t2x2_2.primal_0).load<float>((_S220)));

#line 54
    float _S222 = AtomicAdd_load_forward_0(t2x2_2.diff_1, _S220);


    uint3  _S223 = make_uint3 (idx_8, 1U, 1U);

#line 54
    float _S224 = ((t2x2_2.primal_0).load<float>((_S223)));

#line 54
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S225 = { makeMatrix<float, 2, 2> (_S215, _S218, _S221, _S224), makeMatrix<float, 2, 2> (_S216, _S219, _S222, AtomicAdd_load_forward_0(t2x2_2.diff_1, _S223)) };

#line 54
    return _S225;
}


#line 54
__device__ DiffPair_Splat_2D_AlphaBlend_0 s_fwd_load_splat_alphablend_0(int g_idx_3, DiffTensorView_0 xyz_vs_8, DiffTensorView_0 inv_cov_vs_8, DiffTensorView_0 opacity_8, DiffTensorView_0 rgb_8)
{

#line 249
    uint _S226 = uint(g_idx_3);

#line 249
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S227 = s_fwd_read_t3_float3_0(_S226, xyz_vs_8);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S228 = s_fwd_read_t3_float3_0(_S226, rgb_8);
    DiffPair_float_0 _S229 = s_fwd_read_t1_float_0(_S226, opacity_8);
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S230 = s_fwd_read_t2x2_float2x2_0(_S226, inv_cov_vs_8);

    Splat_2D_AlphaBlend_0 _S231 = { _S227.primal_1, _S228.primal_1, _S229.primal_1, _S230.primal_1 };

#line 254
    Splat_2D_AlphaBlend_0 _S232 = { _S227.differential_0, _S228.differential_0, _S229.differential_0, _S230.differential_0 };

#line 254
    DiffPair_Splat_2D_AlphaBlend_0 _S233 = { _S231, _S232 };

#line 254
    return _S233;
}


#line 269
__device__ DiffPair_float_0 s_fwd_ndc2pix_0(DiffPair_float_0 dpv_2, int S_3)
{

#line 63
    float _S234 = float(S_3);

#line 63
    DiffPair_float_0 _S235 = { ((dpv_2.primal_1 + 1.0f) * _S234 - 1.0f) * 0.5f, dpv_2.differential_0 * _S234 * 0.5f };

#line 63
    return _S235;
}


#line 63
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_evaluate_splat_0(DiffPair_Splat_2D_AlphaBlend_0 dpg_1, DiffPair_vectorx3Cfloatx2C2x3E_0 dppix_coord_1, uint H_4, uint W_4)
{

#line 261
    DiffPair_float_0 _S236 = { dpg_1.primal_1.xyz_vs_0.x, dpg_1.differential_0.xyz_vs_0.x };

#line 269
    DiffPair_float_0 _S237 = s_fwd_ndc2pix_0(_S236, int(W_4));

#line 269
    float _S238 = dppix_coord_1.primal_1.x - _S237.primal_1;

#line 269
    float _S239 = dppix_coord_1.differential_0.x - _S237.differential_0;

#line 269
    DiffPair_float_0 _S240 = { dpg_1.primal_1.xyz_vs_0.y, dpg_1.differential_0.xyz_vs_0.y };
    DiffPair_float_0 _S241 = s_fwd_ndc2pix_0(_S240, int(H_4));

#line 270
    float _S242 = dppix_coord_1.primal_1.y - _S241.primal_1;

#line 270
    float _S243 = dppix_coord_1.differential_0.y - _S241.differential_0;
    float _S244 = dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].x * _S238;
    float _S245 = dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].y * _S242;

#line 272
    float _S246 = dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].y + dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].x;

#line 272
    float _S247 = _S246 * _S238;

#line 272
    DiffPair_float_0 _S248 = { -0.5f * (_S244 * _S238 + _S245 * _S242 + _S247 * _S242), ((dpg_1.differential_0.inv_cov_vs_0.rows[int(0)].x * _S238 + _S239 * dpg_1.primal_1.inv_cov_vs_0.rows[int(0)].x) * _S238 + _S239 * _S244 + ((dpg_1.differential_0.inv_cov_vs_0.rows[int(1)].y * _S242 + _S243 * dpg_1.primal_1.inv_cov_vs_0.rows[int(1)].y) * _S242 + _S243 * _S245) + (((dpg_1.differential_0.inv_cov_vs_0.rows[int(0)].y + dpg_1.differential_0.inv_cov_vs_0.rows[int(1)].x) * _S238 + _S239 * _S246) * _S242 + _S243 * _S247)) * -0.5f };
    DiffPair_float_0 _S249 = _d_exp_1(_S248);

#line 273
    DiffPair_float_0 _S250 = { 0.99000000953674316f, 0.0f };

#line 273
    DiffPair_float_0 _S251 = { dpg_1.primal_1.opacity_0 * _S249.primal_1, dpg_1.differential_0.opacity_0 * _S249.primal_1 + _S249.differential_0 * dpg_1.primal_1.opacity_0 };

#line 273
    DiffPair_float_0 _S252 = _d_min_1(_S250, _S251);

#line 273
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S253 = { make_float4 ((dpg_1.primal_1.rgb_0 * make_float3 (_S252.primal_1)).x, (dpg_1.primal_1.rgb_0 * make_float3 (_S252.primal_1)).y, (dpg_1.primal_1.rgb_0 * make_float3 (_S252.primal_1)).z, _S252.primal_1), make_float4 ((dpg_1.differential_0.rgb_0 * make_float3 (_S252.primal_1) + make_float3 (_S252.differential_0) * dpg_1.primal_1.rgb_0).x, (dpg_1.differential_0.rgb_0 * make_float3 (_S252.primal_1) + make_float3 (_S252.differential_0) * dpg_1.primal_1.rgb_0).y, (dpg_1.differential_0.rgb_0 * make_float3 (_S252.primal_1) + make_float3 (_S252.differential_0) * dpg_1.primal_1.rgb_0).z, _S252.differential_0) };


    return _S253;
}


#line 91 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/alphablend_shader.slang"
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_update_pixel_state_0(DiffPair_vectorx3Cfloatx2C4x3E_0 dppixel_state_t_nm1_1, DiffPair_vectorx3Cfloatx2C4x3E_0 dpgauss_rgba_t_n_1)
{

#line 28
    float3  _S254 = float3 {dpgauss_rgba_t_n_1.primal_1.x, dpgauss_rgba_t_n_1.primal_1.y, dpgauss_rgba_t_n_1.primal_1.z};

#line 28
    float _S255 = dppixel_state_t_nm1_1.primal_1.w;

#line 28
    float _S256 = dppixel_state_t_nm1_1.differential_0.w;
    float _S257 = 1.0f - dpgauss_rgba_t_n_1.primal_1.w;

#line 29
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S258 = { make_float4 ((float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S254 * make_float3 (_S255)).x, (float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S254 * make_float3 (_S255)).y, (float3 {dppixel_state_t_nm1_1.primal_1.x, dppixel_state_t_nm1_1.primal_1.y, dppixel_state_t_nm1_1.primal_1.z} + _S254 * make_float3 (_S255)).z, _S255 * _S257), make_float4 ((float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S255) + make_float3 (_S256) * _S254)).x, (float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S255) + make_float3 (_S256) * _S254)).y, (float3 {dppixel_state_t_nm1_1.differential_0.x, dppixel_state_t_nm1_1.differential_0.y, dppixel_state_t_nm1_1.differential_0.z} + (float3 {dpgauss_rgba_t_n_1.differential_0.x, dpgauss_rgba_t_n_1.differential_0.y, dpgauss_rgba_t_n_1.differential_0.z} * make_float3 (_S255) + make_float3 (_S256) * _S254)).z, _S256 * _S257 + (0.0f - dpgauss_rgba_t_n_1.differential_0.w) * _S255) };
    return _S258;
}


#line 30
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_alpha_blend_0(TensorView sorted_gauss_idx_5, DiffTensorView_0 xyz_vs_9, DiffTensorView_0 inv_cov_vs_9, DiffTensorView_0 opacity_9, DiffTensorView_0 rgb_9, DiffTensorView_0 final_pixel_state_2, TensorView n_contributors_5, uint2  pix_coord_5, uint tile_idx_start_4, uint tile_idx_end_4, uint tile_height_5, uint tile_width_5, uint H_5, uint W_5)
{

#line 56
    float2  _S259 = make_float2 ((float)pix_coord_5.x, (float)pix_coord_5.y);
    float4  _S260 = make_float4 (0.0f, 0.0f, 0.0f, 1.0f);

#line 57
    float4  _S261 = make_float4 (0.0f, 0.0f, 0.0f, 0.0f);
    uint block_size_2 = tile_height_5 * tile_width_5;
    uint _S262 = pix_coord_5.x;

#line 59
    bool is_inside_4;

#line 59
    if(_S262 < W_5)
    {

#line 59
        is_inside_4 = pix_coord_5.y < H_5;

#line 59
    }
    else
    {

#line 59
        is_inside_4 = false;

#line 59
    }


    uint _S263 = tile_idx_end_4 - tile_idx_start_4;

#line 62
    uint _S264 = (_S263 + block_size_2 - 1U) / block_size_2;

#line 62
    int _S265 = int(_S264);

    uint3  _S266 = ((threadIdx));

#line 64
    uint _S267 = _S266.y * ((blockDim)).x + _S266.x;


    int _S268 = int(_S263);

#line 101
    int _S269 = int(block_size_2);

#line 101
    bool thread_active_2 = is_inside_4;

#line 101
    float4  curr_pixel_state_2 = _S260;

#line 101
    float4  s_diff_curr_pixel_state_0 = _S261;

#line 101
    int i_11 = int(0);

#line 101
    int splats_left_to_process_4 = _S268;

#line 101
    int local_n_contrib_4 = int(0);

#line 85
    Splat_2D_AlphaBlend_0 _S270 = Splat_2D_AlphaBlend_x24_syn_dzero_0();

#line 85
    DiffPair_vectorx3Cfloatx2C2x3E_0 _S271 = { _S259, make_float2 (0.0f) };

#line 68
    for(;;)
    {

#line 68
        if(i_11 < _S265)
        {
        }
        else
        {

#line 68
            break;
        }

        __syncthreads();

        uint _S272 = tile_idx_start_4 + uint(int(uint(i_11) * block_size_2 + _S267));

#line 73
        if(_S272 < tile_idx_end_4)
        {

            int _S273 = ((sorted_gauss_idx_5).load<int>((_S272)));
            DiffPair_Splat_2D_AlphaBlend_0 _S274 = s_fwd_load_splat_alphablend_0(int(uint(_S273)), xyz_vs_9, inv_cov_vs_9, opacity_9, rgb_9);

#line 77
            FixedArray<Splat_2D_AlphaBlend_0, 256>  _S275 = *&collected_splats_0;

#line 77
            _S275[_S267] = _S274.primal_1;

#line 77
            *&collected_splats_0 = _S275;

#line 73
        }

#line 79
        __syncthreads();

#line 79
        float4  curr_pixel_state_3;

#line 79
        float4  s_diff_curr_pixel_state_1;
        if(thread_active_2)
        {

#line 80
            int local_n_contrib_5;

#line 80
            bool thread_active_3;
            uint _S276 = (U32_min((block_size_2), (uint(splats_left_to_process_4))));

#line 81
            curr_pixel_state_3 = curr_pixel_state_2;

#line 81
            s_diff_curr_pixel_state_1 = s_diff_curr_pixel_state_0;

#line 81
            int j_2 = int(0);

#line 81
            int local_n_contrib_6 = local_n_contrib_4;

#line 81
            for(;;)
            {

#line 81
                if(uint(j_2) < _S276)
                {
                }
                else
                {

#line 81
                    thread_active_3 = thread_active_2;

#line 81
                    local_n_contrib_5 = local_n_contrib_6;

#line 81
                    break;
                }
                int local_n_contrib_7 = local_n_contrib_6 + int(1);

#line 83
                DiffPair_Splat_2D_AlphaBlend_0 _S277 = { (*&collected_splats_0)[j_2], _S270 };

                DiffPair_vectorx3Cfloatx2C4x3E_0 _S278 = s_fwd_evaluate_splat_0(_S277, _S271, H_5, W_5);


                if(_S278.primal_1.w < 0.00392156885936856f)
                {

#line 89
                    j_2 = j_2 + int(1);

#line 89
                    local_n_contrib_6 = local_n_contrib_7;

#line 81
                    continue;
                }

#line 81
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S279 = { curr_pixel_state_3, s_diff_curr_pixel_state_1 };

#line 81
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S280 = { _S278.primal_1, _S278.differential_0 };

#line 91
                DiffPair_vectorx3Cfloatx2C4x3E_0 _S281 = s_fwd_update_pixel_state_0(_S279, _S280);
                if(_S281.primal_1.w < 0.00009999999747379f)
                {
                    int _S282 = local_n_contrib_7 - int(1);

#line 94
                    thread_active_3 = false;

#line 94
                    local_n_contrib_5 = _S282;

                    break;
                }

#line 96
                curr_pixel_state_3 = _S281.primal_1;

#line 96
                s_diff_curr_pixel_state_1 = _S281.differential_0;

#line 81
                j_2 = j_2 + int(1);

#line 81
                local_n_contrib_6 = local_n_contrib_7;

#line 81
            }

#line 81
            thread_active_2 = thread_active_3;

#line 81
            local_n_contrib_4 = local_n_contrib_5;

#line 80
        }
        else
        {

#line 80
            curr_pixel_state_3 = curr_pixel_state_2;

#line 80
            s_diff_curr_pixel_state_1 = s_diff_curr_pixel_state_0;

#line 80
        }

#line 101
        int splats_left_to_process_5 = splats_left_to_process_4 - _S269;

#line 68
        int _S283 = i_11 + int(1);

#line 68
        curr_pixel_state_2 = curr_pixel_state_3;

#line 68
        s_diff_curr_pixel_state_0 = s_diff_curr_pixel_state_1;

#line 68
        i_11 = _S283;

#line 68
        splats_left_to_process_4 = splats_left_to_process_5;

#line 68
    }

#line 104
    if(is_inside_4)
    {

#line 105
        (n_contributors_5).store<int>((pix_coord_5.y), (_S262), (0U), (local_n_contrib_4));

#line 104
    }

#line 104
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S284 = { curr_pixel_state_2, s_diff_curr_pixel_state_0 };


    return _S284;
}


#line 107
__device__ void s_fwd_splat_tiled_0(TensorView sorted_gauss_idx_6, TensorView tile_ranges_3, DiffTensorView_0 xyz_vs_10, DiffTensorView_0 inv_cov_vs_10, DiffTensorView_0 opacity_10, DiffTensorView_0 rgb_10, DiffTensorView_0 output_img_3, TensorView n_contributors_6, int grid_height_3, int grid_width_3, int tile_height_6, int tile_width_6)
{

#line 210
    uint3  _S285 = ((blockIdx));

    uint2  pix_coord_6 = uint2 {(_S285 * ((blockDim)) + ((threadIdx))).x, (_S285 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_1 = _S285.y * uint(grid_width_3) + _S285.x;
    int _S286 = ((tile_ranges_3).load<int>((tile_idx_1), (0U)));

#line 215
    uint tile_idx_start_5 = uint(_S286);
    int _S287 = ((tile_ranges_3).load<int>((tile_idx_1), (1U)));

#line 216
    uint tile_idx_end_5 = uint(_S287);

    uint _S288 = pix_coord_6.x;

#line 218
    uint _S289 = DiffTensorView_size_0(output_img_3, 1U);

#line 218
    bool is_inside_5;

#line 218
    if(_S288 < _S289)
    {

#line 218
        is_inside_5 = pix_coord_6.y < DiffTensorView_size_0(output_img_3, 0U);

#line 218
    }
    else
    {

#line 218
        is_inside_5 = false;

#line 218
    }

    DiffPair_vectorx3Cfloatx2C4x3E_0 _S290 = s_fwd_alpha_blend_0(sorted_gauss_idx_6, xyz_vs_10, inv_cov_vs_10, opacity_10, rgb_10, output_img_3, n_contributors_6, pix_coord_6, tile_idx_start_5, tile_idx_end_5, uint(tile_height_6), uint(tile_width_6), DiffTensorView_size_0(output_img_3, 0U), _S289);

#line 235
    if(is_inside_5)
    {

#line 236
        uint _S291 = pix_coord_6.y;

#line 236
        DiffPair_float_0 _S292 = { _S290.primal_1.x, _S290.differential_0.x };

#line 236
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S291, _S288, 0U), _S292);

#line 236
        DiffPair_float_0 _S293 = { _S290.primal_1.y, _S290.differential_0.y };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S291, _S288, 1U), _S293);

#line 237
        DiffPair_float_0 _S294 = { _S290.primal_1.z, _S290.differential_0.z };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S291, _S288, 2U), _S294);

#line 238
        DiffPair_float_0 _S295 = { _S290.primal_1.w, _S290.differential_0.w };
        DiffTensorView_storeOnce_forward_0(output_img_3, make_uint3 (_S291, _S288, 3U), _S295);

#line 235
    }

#line 241
    return;
}


#line 241
extern "C" {
__global__ void __kernel__splat_tiled_fwd_diff(TensorView sorted_gauss_idx_7, TensorView tile_ranges_4, DiffTensorView_0 xyz_vs_11, DiffTensorView_0 inv_cov_vs_11, DiffTensorView_0 opacity_11, DiffTensorView_0 rgb_11, DiffTensorView_0 output_img_4, TensorView n_contributors_7, int grid_height_4, int grid_width_4, int tile_height_7, int tile_width_7)
{

#line 241
    s_fwd_splat_tiled_0(sorted_gauss_idx_7, tile_ranges_4, xyz_vs_11, inv_cov_vs_11, opacity_11, rgb_11, output_img_4, n_contributors_7, grid_height_4, grid_width_4, tile_height_7, tile_width_7);

#line 241
    return;
}

}

#line 197
__global__ void __kernel__splat_tiled(TensorView sorted_gauss_idx_8, TensorView tile_ranges_5, DiffTensorView_0 xyz_vs_12, DiffTensorView_0 inv_cov_vs_12, DiffTensorView_0 opacity_12, DiffTensorView_0 rgb_12, DiffTensorView_0 output_img_5, TensorView n_contributors_8, int grid_height_5, int grid_width_5, int tile_height_8, int tile_width_8)
{

#line 210
    uint3  _S296 = ((blockIdx));

    uint2  pix_coord_7 = uint2 {(_S296 * ((blockDim)) + ((threadIdx))).x, (_S296 * ((blockDim)) + ((threadIdx))).y};

    uint tile_idx_2 = _S296.y * uint(grid_width_5) + _S296.x;
    int _S297 = ((tile_ranges_5).load<int>((tile_idx_2), (0U)));

#line 215
    uint tile_idx_start_6 = uint(_S297);
    int _S298 = ((tile_ranges_5).load<int>((tile_idx_2), (1U)));

#line 216
    uint tile_idx_end_6 = uint(_S298);

    uint _S299 = pix_coord_7.x;

#line 218
    uint _S300 = DiffTensorView_size_0(output_img_5, 1U);

#line 218
    bool is_inside_6;

#line 218
    if(_S299 < _S300)
    {

#line 218
        is_inside_6 = pix_coord_7.y < DiffTensorView_size_0(output_img_5, 0U);

#line 218
    }
    else
    {

#line 218
        is_inside_6 = false;

#line 218
    }

    float4  pixel_state_0 = alpha_blend_0(sorted_gauss_idx_8, xyz_vs_12, inv_cov_vs_12, opacity_12, rgb_12, output_img_5, n_contributors_8, pix_coord_7, tile_idx_start_6, tile_idx_end_6, uint(tile_height_8), uint(tile_width_8), DiffTensorView_size_0(output_img_5, 0U), _S300);

#line 235
    if(is_inside_6)
    {

#line 236
        uint _S301 = pix_coord_7.y;

#line 236
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S301, _S299, 0U), pixel_state_0.x);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S301, _S299, 1U), pixel_state_0.y);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S301, _S299, 2U), pixel_state_0.z);
        DiffTensorView_storeOnce_0(output_img_5, make_uint3 (_S301, _S299, 3U), pixel_state_0.w);

#line 235
    }

#line 241
    return;
}


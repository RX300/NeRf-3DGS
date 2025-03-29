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
__device__ float AtomicAdd_load_forward_0(AtomicAdd_0 this_0, uint2  i_0)
{
    float _S1 = ((this_0.diff_0).load<float>((i_0)));

#line 709
    return _S1;
}


#line 707
__device__ float AtomicAdd_load_forward_1(AtomicAdd_0 this_1, uint3  i_1)
{
    float _S2 = ((this_1.diff_0).load<float>((i_1)));

#line 709
    return _S2;
}


#line 720
__device__ void AtomicAdd_load_backward_0(AtomicAdd_0 this_2, uint2  i_2, float dOut_0)
{
    float oldVal_0;
    *((&oldVal_0)) = atomicAdd((this_2.diff_0).data_ptr_at<float>((i_2)), (dOut_0));
    return;
}


#line 720
__device__ void AtomicAdd_load_backward_1(AtomicAdd_0 this_3, uint3  i_3, float dOut_1)
{
    float oldVal_1;
    *((&oldVal_1)) = atomicAdd((this_3.diff_0).data_ptr_at<float>((i_3)), (dOut_1));
    return;
}


#line 790
__device__ void AtomicAdd_storeOnce_forward_0(AtomicAdd_0 this_4, uint2  i_4, float dx_0)
{
    (this_4.diff_0).store<float>((i_4), (dx_0));
    return;
}


#line 790
__device__ void AtomicAdd_storeOnce_forward_1(AtomicAdd_0 this_5, uint3  i_5, float dx_1)
{
    (this_5.diff_0).store<float>((i_5), (dx_1));
    return;
}


#line 802
__device__ float AtomicAdd_storeOnce_backward_0(AtomicAdd_0 this_6, uint2  i_6)
{
    float _S3 = ((this_6.diff_0).load<float>((i_6)));

#line 804
    return _S3;
}


#line 802
__device__ float AtomicAdd_storeOnce_backward_1(AtomicAdd_0 this_7, uint3  i_7)
{
    float _S4 = ((this_7.diff_0).load<float>((i_7)));

#line 804
    return _S4;
}


#line 78 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
struct Camera_Differential_0
{
    Matrix<float, 4, 4>  world_view_transform_0;
    Matrix<float, 4, 4>  proj_mat_0;
    float3  position_0;
    float fovy_0;
    float fovx_0;
};


#line 1 "token paste"
__device__ Camera_Differential_0 Camera_x24_syn_dzero_0()
{

#line 1
    Camera_Differential_0 result_0;

#line 1805 "core.meta.slang"
    Matrix<float, 4, 4>  _S5 = makeMatrix<float, 4, 4> (0.0f);

#line 1805
    (&result_0)->world_view_transform_0 = _S5;

#line 1805
    (&result_0)->proj_mat_0 = _S5;

#line 1805
    (&result_0)->position_0 = make_float3 (0.0f);

#line 1805
    (&result_0)->fovy_0 = 0.0f;

#line 1805
    (&result_0)->fovx_0 = 0.0f;

#line 1805
    return result_0;
}


#line 1805
__device__ Camera_Differential_0 Camera_x24_syn_dadd_0(Camera_Differential_0 SLANG_anonymous_0_0, Camera_Differential_0 SLANG_anonymous_1_0)
{

#line 1805
    Camera_Differential_0 result_1;

#line 1805
    (&result_1)->world_view_transform_0 = SLANG_anonymous_0_0.world_view_transform_0 + SLANG_anonymous_1_0.world_view_transform_0;

#line 1805
    (&result_1)->proj_mat_0 = SLANG_anonymous_0_0.proj_mat_0 + SLANG_anonymous_1_0.proj_mat_0;

#line 1805
    (&result_1)->position_0 = SLANG_anonymous_0_0.position_0 + SLANG_anonymous_1_0.position_0;

#line 1805
    (&result_1)->fovy_0 = SLANG_anonymous_0_0.fovy_0 + SLANG_anonymous_1_0.fovy_0;

#line 1805
    (&result_1)->fovx_0 = SLANG_anonymous_0_0.fovx_0 + SLANG_anonymous_1_0.fovx_0;

#line 1805
    return result_1;
}


#line 34 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
struct SpherHarmCoeffs_0
{
    float3  coeff0_0;
    float3  coeff1_0;
    float3  coeff2_0;
    float3  coeff3_0;
    float3  coeff4_0;
    float3  coeff5_0;
    float3  coeff6_0;
    float3  coeff7_0;
    float3  coeff8_0;
    float3  coeff9_0;
    float3  coeff10_0;
    float3  coeff11_0;
    float3  coeff12_0;
    float3  coeff13_0;
    float3  coeff14_0;
    float3  coeff15_0;
};


#line 34
__device__ SpherHarmCoeffs_0 SpherHarmCoeffs_x24_syn_dzero_0()
{

#line 34
    SpherHarmCoeffs_0 result_2;

#line 1751 "core.meta.slang"
    float3  _S6 = make_float3 (0.0f);

#line 1751
    (&result_2)->coeff0_0 = _S6;

#line 1751
    (&result_2)->coeff1_0 = _S6;

#line 1751
    (&result_2)->coeff2_0 = _S6;

#line 1751
    (&result_2)->coeff3_0 = _S6;

#line 1751
    (&result_2)->coeff4_0 = _S6;

#line 1751
    (&result_2)->coeff5_0 = _S6;

#line 1751
    (&result_2)->coeff6_0 = _S6;

#line 1751
    (&result_2)->coeff7_0 = _S6;

#line 1751
    (&result_2)->coeff8_0 = _S6;

#line 1751
    (&result_2)->coeff9_0 = _S6;

#line 1751
    (&result_2)->coeff10_0 = _S6;

#line 1751
    (&result_2)->coeff11_0 = _S6;

#line 1751
    (&result_2)->coeff12_0 = _S6;

#line 1751
    (&result_2)->coeff13_0 = _S6;

#line 1751
    (&result_2)->coeff14_0 = _S6;

#line 1751
    (&result_2)->coeff15_0 = _S6;

#line 1751
    return result_2;
}


#line 1751
__device__ SpherHarmCoeffs_0 SpherHarmCoeffs_x24_syn_dadd_0(SpherHarmCoeffs_0 SLANG_anonymous_0_1, SpherHarmCoeffs_0 SLANG_anonymous_1_1)
{

#line 1751
    SpherHarmCoeffs_0 result_3;

#line 1751
    (&result_3)->coeff0_0 = SLANG_anonymous_0_1.coeff0_0 + SLANG_anonymous_1_1.coeff0_0;

#line 1751
    (&result_3)->coeff1_0 = SLANG_anonymous_0_1.coeff1_0 + SLANG_anonymous_1_1.coeff1_0;

#line 1751
    (&result_3)->coeff2_0 = SLANG_anonymous_0_1.coeff2_0 + SLANG_anonymous_1_1.coeff2_0;

#line 1751
    (&result_3)->coeff3_0 = SLANG_anonymous_0_1.coeff3_0 + SLANG_anonymous_1_1.coeff3_0;

#line 1751
    (&result_3)->coeff4_0 = SLANG_anonymous_0_1.coeff4_0 + SLANG_anonymous_1_1.coeff4_0;

#line 1751
    (&result_3)->coeff5_0 = SLANG_anonymous_0_1.coeff5_0 + SLANG_anonymous_1_1.coeff5_0;

#line 1751
    (&result_3)->coeff6_0 = SLANG_anonymous_0_1.coeff6_0 + SLANG_anonymous_1_1.coeff6_0;

#line 1751
    (&result_3)->coeff7_0 = SLANG_anonymous_0_1.coeff7_0 + SLANG_anonymous_1_1.coeff7_0;

#line 1751
    (&result_3)->coeff8_0 = SLANG_anonymous_0_1.coeff8_0 + SLANG_anonymous_1_1.coeff8_0;

#line 1751
    (&result_3)->coeff9_0 = SLANG_anonymous_0_1.coeff9_0 + SLANG_anonymous_1_1.coeff9_0;

#line 1751
    (&result_3)->coeff10_0 = SLANG_anonymous_0_1.coeff10_0 + SLANG_anonymous_1_1.coeff10_0;

#line 1751
    (&result_3)->coeff11_0 = SLANG_anonymous_0_1.coeff11_0 + SLANG_anonymous_1_1.coeff11_0;

#line 1751
    (&result_3)->coeff12_0 = SLANG_anonymous_0_1.coeff12_0 + SLANG_anonymous_1_1.coeff12_0;

#line 1751
    (&result_3)->coeff13_0 = SLANG_anonymous_0_1.coeff13_0 + SLANG_anonymous_1_1.coeff13_0;

#line 1751
    (&result_3)->coeff14_0 = SLANG_anonymous_0_1.coeff14_0 + SLANG_anonymous_1_1.coeff14_0;

#line 1751
    (&result_3)->coeff15_0 = SLANG_anonymous_0_1.coeff15_0 + SLANG_anonymous_1_1.coeff15_0;

#line 1751
    return result_3;
}


#line 161 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
struct Gaussian_3D_0
{
    float3  xyz_ws_0;
    SpherHarmCoeffs_0 sh_coeffs_0;
    float4  rotations_0;
    float3  scales_0;
};


#line 1352 "diff.meta.slang"
__device__ Gaussian_3D_0 Gaussian_3D_x24_syn_dzero_0()
{

#line 1352
    Gaussian_3D_0 result_4;

#line 1751 "core.meta.slang"
    float3  _S7 = make_float3 (0.0f);

#line 1751
    (&result_4)->xyz_ws_0 = _S7;

#line 1751
    (&result_4)->sh_coeffs_0 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 1751
    (&result_4)->rotations_0 = make_float4 (0.0f);

#line 1751
    (&result_4)->scales_0 = _S7;

#line 1751
    return result_4;
}


#line 1751
__device__ Gaussian_3D_0 Gaussian_3D_x24_syn_dadd_0(Gaussian_3D_0 SLANG_anonymous_0_2, Gaussian_3D_0 SLANG_anonymous_1_2)
{

#line 1751
    Gaussian_3D_0 result_5;

#line 1751
    (&result_5)->xyz_ws_0 = SLANG_anonymous_0_2.xyz_ws_0 + SLANG_anonymous_1_2.xyz_ws_0;

#line 1751
    (&result_5)->sh_coeffs_0 = SpherHarmCoeffs_x24_syn_dadd_0(SLANG_anonymous_0_2.sh_coeffs_0, SLANG_anonymous_1_2.sh_coeffs_0);

#line 1751
    (&result_5)->rotations_0 = SLANG_anonymous_0_2.rotations_0 + SLANG_anonymous_1_2.rotations_0;

#line 1751
    (&result_5)->scales_0 = SLANG_anonymous_0_2.scales_0 + SLANG_anonymous_1_2.scales_0;

#line 1751
    return result_5;
}


#line 186 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
struct Splat_2D_Vertex_0
{
    float3  xyz_vs_0;
    float3  rgb_0;
    Matrix<float, 2, 2>  cov_vs_0;
};


#line 186
__device__ Splat_2D_Vertex_0 Splat_2D_Vertex_x24_syn_dzero_0()
{

#line 186
    Splat_2D_Vertex_0 result_6;

#line 1751 "core.meta.slang"
    float3  _S8 = make_float3 (0.0f);

#line 1751
    (&result_6)->xyz_vs_0 = _S8;

#line 1751
    (&result_6)->rgb_0 = _S8;

#line 1751
    (&result_6)->cov_vs_0 = makeMatrix<float, 2, 2> (0.0f);

#line 1751
    return result_6;
}


#line 1751
__device__ Splat_2D_Vertex_0 Splat_2D_Vertex_x24_syn_dadd_0(Splat_2D_Vertex_0 SLANG_anonymous_0_3, Splat_2D_Vertex_0 SLANG_anonymous_1_3)
{

#line 1751
    Splat_2D_Vertex_0 result_7;

#line 1751
    (&result_7)->xyz_vs_0 = SLANG_anonymous_0_3.xyz_vs_0 + SLANG_anonymous_1_3.xyz_vs_0;

#line 1751
    (&result_7)->rgb_0 = SLANG_anonymous_0_3.rgb_0 + SLANG_anonymous_1_3.rgb_0;

#line 1751
    (&result_7)->cov_vs_0 = SLANG_anonymous_0_3.cov_vs_0 + SLANG_anonymous_1_3.cov_vs_0;

#line 1751
    return result_7;
}


#line 809 "diff.meta.slang"
struct DiffTensorView_0
{
    TensorView primal_0;
    AtomicAdd_0 diff_1;
};


#line 814
__device__ uint DiffTensorView_size_0(DiffTensorView_0 this_8, uint i_8)
{
    uint _S9 = ((this_8.primal_0).sizes[(i_8)]);

#line 816
    return _S9;
}


#line 78 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
struct Camera_0
{
    Matrix<float, 4, 4>  world_view_transform_1;
    Matrix<float, 4, 4>  proj_mat_1;
    float3  position_1;
    float fovy_1;
    float fovx_1;
    int H_0;
    int W_0;
};

__device__ Camera_0 load_camera_0(TensorView world_view_transform_t_0, TensorView proj_mat_t_0, TensorView position_t_0, float fovy_2, float fovx_2, uint H_1, uint W_1)
{

#line 90
    float _S10 = ((world_view_transform_t_0).load<float>((0U), (0U)));

#line 90
    float _S11 = ((world_view_transform_t_0).load<float>((0U), (1U)));

#line 90
    float _S12 = ((world_view_transform_t_0).load<float>((0U), (2U)));

#line 90
    float _S13 = ((world_view_transform_t_0).load<float>((0U), (3U)));

#line 90
    float _S14 = ((world_view_transform_t_0).load<float>((1U), (0U)));

#line 90
    float _S15 = ((world_view_transform_t_0).load<float>((1U), (1U)));

#line 90
    float _S16 = ((world_view_transform_t_0).load<float>((1U), (2U)));

#line 90
    float _S17 = ((world_view_transform_t_0).load<float>((1U), (3U)));

#line 90
    float _S18 = ((world_view_transform_t_0).load<float>((2U), (0U)));

#line 90
    float _S19 = ((world_view_transform_t_0).load<float>((2U), (1U)));

#line 90
    float _S20 = ((world_view_transform_t_0).load<float>((2U), (2U)));

#line 90
    float _S21 = ((world_view_transform_t_0).load<float>((2U), (3U)));

#line 90
    float _S22 = ((world_view_transform_t_0).load<float>((3U), (0U)));

#line 90
    float _S23 = ((world_view_transform_t_0).load<float>((3U), (1U)));

#line 90
    float _S24 = ((world_view_transform_t_0).load<float>((3U), (2U)));

#line 90
    float _S25 = ((world_view_transform_t_0).load<float>((3U), (3U)));

#line 90
    Matrix<float, 4, 4>  world_view_transform_2 = makeMatrix<float, 4, 4> (_S10, _S11, _S12, _S13, _S14, _S15, _S16, _S17, _S18, _S19, _S20, _S21, _S22, _S23, _S24, _S25);

#line 95
    float _S26 = ((proj_mat_t_0).load<float>((0U), (0U)));

#line 95
    float _S27 = ((proj_mat_t_0).load<float>((0U), (1U)));

#line 95
    float _S28 = ((proj_mat_t_0).load<float>((0U), (2U)));

#line 95
    float _S29 = ((proj_mat_t_0).load<float>((0U), (3U)));

#line 95
    float _S30 = ((proj_mat_t_0).load<float>((1U), (0U)));

#line 95
    float _S31 = ((proj_mat_t_0).load<float>((1U), (1U)));

#line 95
    float _S32 = ((proj_mat_t_0).load<float>((1U), (2U)));

#line 95
    float _S33 = ((proj_mat_t_0).load<float>((1U), (3U)));

#line 95
    float _S34 = ((proj_mat_t_0).load<float>((2U), (0U)));

#line 95
    float _S35 = ((proj_mat_t_0).load<float>((2U), (1U)));

#line 95
    float _S36 = ((proj_mat_t_0).load<float>((2U), (2U)));

#line 95
    float _S37 = ((proj_mat_t_0).load<float>((2U), (3U)));

#line 95
    float _S38 = ((proj_mat_t_0).load<float>((3U), (0U)));

#line 95
    float _S39 = ((proj_mat_t_0).load<float>((3U), (1U)));

#line 95
    float _S40 = ((proj_mat_t_0).load<float>((3U), (2U)));

#line 95
    float _S41 = ((proj_mat_t_0).load<float>((3U), (3U)));

#line 95
    Matrix<float, 4, 4>  proj_mat_2 = makeMatrix<float, 4, 4> (_S26, _S27, _S28, _S29, _S30, _S31, _S32, _S33, _S34, _S35, _S36, _S37, _S38, _S39, _S40, _S41);



    float _S42 = ((position_t_0).load<float>((0U)));

#line 99
    float _S43 = ((position_t_0).load<float>((1U)));

#line 99
    float _S44 = ((position_t_0).load<float>((2U)));

    Camera_0 _S45 = { world_view_transform_2, proj_mat_2, make_float3 (_S42, _S43, _S44), fovy_2, fovx_2, int(H_1), int(W_1) };

#line 101
    return _S45;
}


#line 850 "diff.meta.slang"
__device__ float DiffTensorView_load_0(DiffTensorView_0 this_9, uint2  i_9)
{

#line 850
    float _S46 = ((this_9.primal_0).load<float>((i_9)));

#line 850
    return _S46;
}


#line 850
__device__ float DiffTensorView_load_1(DiffTensorView_0 this_10, uint3  i_10)
{

#line 850
    float _S47 = ((this_10.primal_0).load<float>((i_10)));

#line 850
    return _S47;
}


#line 26 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ float3  read_t3_float3_0(uint idx_0, DiffTensorView_0 t3_0)
{
    return make_float3 (DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 0U)), DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 1U)), DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 2U)));
}


#line 62 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
__device__ SpherHarmCoeffs_0 read_spherical_harmonics_coeffs_0(uint g_idx_0, DiffTensorView_0 sh_coeffs_1, uint active_sh_0)
{
    SpherHarmCoeffs_0 g_sh_coeffs_0;
    (&g_sh_coeffs_0)->coeff0_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 0U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 0U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 0U, 2U)));

    if(active_sh_0 > 0U)
    {

#line 68
        (&g_sh_coeffs_0)->coeff1_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 1U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 1U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 1U, 2U)));
        (&g_sh_coeffs_0)->coeff2_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 2U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 2U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 2U, 2U)));
        (&g_sh_coeffs_0)->coeff3_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 3U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 3U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 3U, 2U)));

        if(active_sh_0 > 1U)
        {

#line 73
            (&g_sh_coeffs_0)->coeff4_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 4U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 4U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 4U, 2U)));
            (&g_sh_coeffs_0)->coeff5_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 5U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 5U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 5U, 2U)));
            (&g_sh_coeffs_0)->coeff6_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 6U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 6U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 6U, 2U)));
            (&g_sh_coeffs_0)->coeff7_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 7U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 7U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 7U, 2U)));
            (&g_sh_coeffs_0)->coeff8_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 8U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 8U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 8U, 2U)));

            if(active_sh_0 > 2U)
            {

#line 80
                (&g_sh_coeffs_0)->coeff9_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 9U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 9U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 9U, 2U)));
                (&g_sh_coeffs_0)->coeff10_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 10U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 10U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 10U, 2U)));
                (&g_sh_coeffs_0)->coeff11_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 11U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 11U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 11U, 2U)));
                (&g_sh_coeffs_0)->coeff12_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 12U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 12U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 12U, 2U)));
                (&g_sh_coeffs_0)->coeff13_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 13U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 13U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 13U, 2U)));
                (&g_sh_coeffs_0)->coeff14_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 14U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 14U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 14U, 2U)));
                (&g_sh_coeffs_0)->coeff15_0 = make_float3 (DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 15U, 0U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 15U, 1U)), DiffTensorView_load_1(sh_coeffs_1, make_uint3 (g_idx_0, 15U, 2U)));

#line 79
            }

#line 72
        }

#line 67
    }

#line 90
    return g_sh_coeffs_0;
}


#line 34 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ float4  read_t4_float4_0(uint idx_1, DiffTensorView_0 t4_0)
{
    return make_float4 (DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 0U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 1U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 2U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 3U)));
}


#line 170
__device__ Gaussian_3D_0 load_gaussian_0(int g_idx_1, DiffTensorView_0 xyz_ws_1, DiffTensorView_0 sh_coeffs_2, DiffTensorView_0 rotations_1, DiffTensorView_0 scales_1, uint active_sh_1)
{

#line 177
    uint _S48 = uint(g_idx_1);

#line 182
    Gaussian_3D_0 _S49 = { read_t3_float3_0(_S48, xyz_ws_1), read_spherical_harmonics_coeffs_0(_S48, sh_coeffs_2, active_sh_1), read_t4_float4_0(_S48, rotations_1), read_t3_float3_0(_S48, scales_1) };

#line 182
    return _S49;
}


#line 182
struct DiffPair_matrixx3Cfloatx2C4x2C4x3E_0
{
    Matrix<float, 4, 4>  primal_1;
    Matrix<float, 4, 4>  differential_0;
};


#line 1386 "diff.meta.slang"
__device__ void mul_0(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * left_0, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * right_0, Matrix<float, 4, 4>  dOut_2)
{
    Matrix<float, 4, 4>  left_d_result_0;

#line 1393
    *&(((&left_d_result_0)->rows + (int(0)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(0)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(0)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(0)))->w) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(1)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(1)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(1)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(1)))->w) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(2)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(2)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(2)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(2)))->w) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(3)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(3)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(3)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_0)->rows + (int(3)))->w) = 0.0f;

    Matrix<float, 4, 4>  right_d_result_0;

#line 1400
    *&(((&right_d_result_0)->rows + (int(0)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(0)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(0)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(0)))->w) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(1)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(1)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(1)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(1)))->w) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(2)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(2)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(2)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(2)))->w) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(3)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(3)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(3)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_0)->rows + (int(3)))->w) = 0.0f;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->x) = *&(((&left_d_result_0)->rows + (int(0)))->x) + (*right_0).primal_1.rows[int(0)].x * dOut_2.rows[int(0)].x;
    *&(((&right_d_result_0)->rows + (int(0)))->x) = *&(((&right_d_result_0)->rows + (int(0)))->x) + (*left_0).primal_1.rows[int(0)].x * dOut_2.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->y) = *&(((&left_d_result_0)->rows + (int(0)))->y) + (*right_0).primal_1.rows[int(1)].x * dOut_2.rows[int(0)].x;
    *&(((&right_d_result_0)->rows + (int(1)))->x) = *&(((&right_d_result_0)->rows + (int(1)))->x) + (*left_0).primal_1.rows[int(0)].y * dOut_2.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->z) = *&(((&left_d_result_0)->rows + (int(0)))->z) + (*right_0).primal_1.rows[int(2)].x * dOut_2.rows[int(0)].x;
    *&(((&right_d_result_0)->rows + (int(2)))->x) = *&(((&right_d_result_0)->rows + (int(2)))->x) + (*left_0).primal_1.rows[int(0)].z * dOut_2.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->w) = *&(((&left_d_result_0)->rows + (int(0)))->w) + (*right_0).primal_1.rows[int(3)].x * dOut_2.rows[int(0)].x;
    *&(((&right_d_result_0)->rows + (int(3)))->x) = *&(((&right_d_result_0)->rows + (int(3)))->x) + (*left_0).primal_1.rows[int(0)].w * dOut_2.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->x) = *&(((&left_d_result_0)->rows + (int(0)))->x) + (*right_0).primal_1.rows[int(0)].y * dOut_2.rows[int(0)].y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = *&(((&right_d_result_0)->rows + (int(0)))->y) + (*left_0).primal_1.rows[int(0)].x * dOut_2.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->y) = *&(((&left_d_result_0)->rows + (int(0)))->y) + (*right_0).primal_1.rows[int(1)].y * dOut_2.rows[int(0)].y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = *&(((&right_d_result_0)->rows + (int(1)))->y) + (*left_0).primal_1.rows[int(0)].y * dOut_2.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->z) = *&(((&left_d_result_0)->rows + (int(0)))->z) + (*right_0).primal_1.rows[int(2)].y * dOut_2.rows[int(0)].y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = *&(((&right_d_result_0)->rows + (int(2)))->y) + (*left_0).primal_1.rows[int(0)].z * dOut_2.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->w) = *&(((&left_d_result_0)->rows + (int(0)))->w) + (*right_0).primal_1.rows[int(3)].y * dOut_2.rows[int(0)].y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = *&(((&right_d_result_0)->rows + (int(3)))->y) + (*left_0).primal_1.rows[int(0)].w * dOut_2.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->x) = *&(((&left_d_result_0)->rows + (int(0)))->x) + (*right_0).primal_1.rows[int(0)].z * dOut_2.rows[int(0)].z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = *&(((&right_d_result_0)->rows + (int(0)))->z) + (*left_0).primal_1.rows[int(0)].x * dOut_2.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->y) = *&(((&left_d_result_0)->rows + (int(0)))->y) + (*right_0).primal_1.rows[int(1)].z * dOut_2.rows[int(0)].z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = *&(((&right_d_result_0)->rows + (int(1)))->z) + (*left_0).primal_1.rows[int(0)].y * dOut_2.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->z) = *&(((&left_d_result_0)->rows + (int(0)))->z) + (*right_0).primal_1.rows[int(2)].z * dOut_2.rows[int(0)].z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = *&(((&right_d_result_0)->rows + (int(2)))->z) + (*left_0).primal_1.rows[int(0)].z * dOut_2.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->w) = *&(((&left_d_result_0)->rows + (int(0)))->w) + (*right_0).primal_1.rows[int(3)].z * dOut_2.rows[int(0)].z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = *&(((&right_d_result_0)->rows + (int(3)))->z) + (*left_0).primal_1.rows[int(0)].w * dOut_2.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->x) = *&(((&left_d_result_0)->rows + (int(0)))->x) + (*right_0).primal_1.rows[int(0)].w * dOut_2.rows[int(0)].w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = *&(((&right_d_result_0)->rows + (int(0)))->w) + (*left_0).primal_1.rows[int(0)].x * dOut_2.rows[int(0)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->y) = *&(((&left_d_result_0)->rows + (int(0)))->y) + (*right_0).primal_1.rows[int(1)].w * dOut_2.rows[int(0)].w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = *&(((&right_d_result_0)->rows + (int(1)))->w) + (*left_0).primal_1.rows[int(0)].y * dOut_2.rows[int(0)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->z) = *&(((&left_d_result_0)->rows + (int(0)))->z) + (*right_0).primal_1.rows[int(2)].w * dOut_2.rows[int(0)].w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = *&(((&right_d_result_0)->rows + (int(2)))->w) + (*left_0).primal_1.rows[int(0)].z * dOut_2.rows[int(0)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(0)))->w) = *&(((&left_d_result_0)->rows + (int(0)))->w) + (*right_0).primal_1.rows[int(3)].w * dOut_2.rows[int(0)].w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = *&(((&right_d_result_0)->rows + (int(3)))->w) + (*left_0).primal_1.rows[int(0)].w * dOut_2.rows[int(0)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->x) = *&(((&left_d_result_0)->rows + (int(1)))->x) + (*right_0).primal_1.rows[int(0)].x * dOut_2.rows[int(1)].x;
    *&(((&right_d_result_0)->rows + (int(0)))->x) = *&(((&right_d_result_0)->rows + (int(0)))->x) + (*left_0).primal_1.rows[int(1)].x * dOut_2.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->y) = *&(((&left_d_result_0)->rows + (int(1)))->y) + (*right_0).primal_1.rows[int(1)].x * dOut_2.rows[int(1)].x;
    *&(((&right_d_result_0)->rows + (int(1)))->x) = *&(((&right_d_result_0)->rows + (int(1)))->x) + (*left_0).primal_1.rows[int(1)].y * dOut_2.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->z) = *&(((&left_d_result_0)->rows + (int(1)))->z) + (*right_0).primal_1.rows[int(2)].x * dOut_2.rows[int(1)].x;
    *&(((&right_d_result_0)->rows + (int(2)))->x) = *&(((&right_d_result_0)->rows + (int(2)))->x) + (*left_0).primal_1.rows[int(1)].z * dOut_2.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->w) = *&(((&left_d_result_0)->rows + (int(1)))->w) + (*right_0).primal_1.rows[int(3)].x * dOut_2.rows[int(1)].x;
    *&(((&right_d_result_0)->rows + (int(3)))->x) = *&(((&right_d_result_0)->rows + (int(3)))->x) + (*left_0).primal_1.rows[int(1)].w * dOut_2.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->x) = *&(((&left_d_result_0)->rows + (int(1)))->x) + (*right_0).primal_1.rows[int(0)].y * dOut_2.rows[int(1)].y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = *&(((&right_d_result_0)->rows + (int(0)))->y) + (*left_0).primal_1.rows[int(1)].x * dOut_2.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->y) = *&(((&left_d_result_0)->rows + (int(1)))->y) + (*right_0).primal_1.rows[int(1)].y * dOut_2.rows[int(1)].y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = *&(((&right_d_result_0)->rows + (int(1)))->y) + (*left_0).primal_1.rows[int(1)].y * dOut_2.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->z) = *&(((&left_d_result_0)->rows + (int(1)))->z) + (*right_0).primal_1.rows[int(2)].y * dOut_2.rows[int(1)].y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = *&(((&right_d_result_0)->rows + (int(2)))->y) + (*left_0).primal_1.rows[int(1)].z * dOut_2.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->w) = *&(((&left_d_result_0)->rows + (int(1)))->w) + (*right_0).primal_1.rows[int(3)].y * dOut_2.rows[int(1)].y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = *&(((&right_d_result_0)->rows + (int(3)))->y) + (*left_0).primal_1.rows[int(1)].w * dOut_2.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->x) = *&(((&left_d_result_0)->rows + (int(1)))->x) + (*right_0).primal_1.rows[int(0)].z * dOut_2.rows[int(1)].z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = *&(((&right_d_result_0)->rows + (int(0)))->z) + (*left_0).primal_1.rows[int(1)].x * dOut_2.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->y) = *&(((&left_d_result_0)->rows + (int(1)))->y) + (*right_0).primal_1.rows[int(1)].z * dOut_2.rows[int(1)].z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = *&(((&right_d_result_0)->rows + (int(1)))->z) + (*left_0).primal_1.rows[int(1)].y * dOut_2.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->z) = *&(((&left_d_result_0)->rows + (int(1)))->z) + (*right_0).primal_1.rows[int(2)].z * dOut_2.rows[int(1)].z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = *&(((&right_d_result_0)->rows + (int(2)))->z) + (*left_0).primal_1.rows[int(1)].z * dOut_2.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->w) = *&(((&left_d_result_0)->rows + (int(1)))->w) + (*right_0).primal_1.rows[int(3)].z * dOut_2.rows[int(1)].z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = *&(((&right_d_result_0)->rows + (int(3)))->z) + (*left_0).primal_1.rows[int(1)].w * dOut_2.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->x) = *&(((&left_d_result_0)->rows + (int(1)))->x) + (*right_0).primal_1.rows[int(0)].w * dOut_2.rows[int(1)].w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = *&(((&right_d_result_0)->rows + (int(0)))->w) + (*left_0).primal_1.rows[int(1)].x * dOut_2.rows[int(1)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->y) = *&(((&left_d_result_0)->rows + (int(1)))->y) + (*right_0).primal_1.rows[int(1)].w * dOut_2.rows[int(1)].w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = *&(((&right_d_result_0)->rows + (int(1)))->w) + (*left_0).primal_1.rows[int(1)].y * dOut_2.rows[int(1)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->z) = *&(((&left_d_result_0)->rows + (int(1)))->z) + (*right_0).primal_1.rows[int(2)].w * dOut_2.rows[int(1)].w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = *&(((&right_d_result_0)->rows + (int(2)))->w) + (*left_0).primal_1.rows[int(1)].z * dOut_2.rows[int(1)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(1)))->w) = *&(((&left_d_result_0)->rows + (int(1)))->w) + (*right_0).primal_1.rows[int(3)].w * dOut_2.rows[int(1)].w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = *&(((&right_d_result_0)->rows + (int(3)))->w) + (*left_0).primal_1.rows[int(1)].w * dOut_2.rows[int(1)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->x) = *&(((&left_d_result_0)->rows + (int(2)))->x) + (*right_0).primal_1.rows[int(0)].x * dOut_2.rows[int(2)].x;
    *&(((&right_d_result_0)->rows + (int(0)))->x) = *&(((&right_d_result_0)->rows + (int(0)))->x) + (*left_0).primal_1.rows[int(2)].x * dOut_2.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->y) = *&(((&left_d_result_0)->rows + (int(2)))->y) + (*right_0).primal_1.rows[int(1)].x * dOut_2.rows[int(2)].x;
    *&(((&right_d_result_0)->rows + (int(1)))->x) = *&(((&right_d_result_0)->rows + (int(1)))->x) + (*left_0).primal_1.rows[int(2)].y * dOut_2.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->z) = *&(((&left_d_result_0)->rows + (int(2)))->z) + (*right_0).primal_1.rows[int(2)].x * dOut_2.rows[int(2)].x;
    *&(((&right_d_result_0)->rows + (int(2)))->x) = *&(((&right_d_result_0)->rows + (int(2)))->x) + (*left_0).primal_1.rows[int(2)].z * dOut_2.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->w) = *&(((&left_d_result_0)->rows + (int(2)))->w) + (*right_0).primal_1.rows[int(3)].x * dOut_2.rows[int(2)].x;
    *&(((&right_d_result_0)->rows + (int(3)))->x) = *&(((&right_d_result_0)->rows + (int(3)))->x) + (*left_0).primal_1.rows[int(2)].w * dOut_2.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->x) = *&(((&left_d_result_0)->rows + (int(2)))->x) + (*right_0).primal_1.rows[int(0)].y * dOut_2.rows[int(2)].y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = *&(((&right_d_result_0)->rows + (int(0)))->y) + (*left_0).primal_1.rows[int(2)].x * dOut_2.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->y) = *&(((&left_d_result_0)->rows + (int(2)))->y) + (*right_0).primal_1.rows[int(1)].y * dOut_2.rows[int(2)].y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = *&(((&right_d_result_0)->rows + (int(1)))->y) + (*left_0).primal_1.rows[int(2)].y * dOut_2.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->z) = *&(((&left_d_result_0)->rows + (int(2)))->z) + (*right_0).primal_1.rows[int(2)].y * dOut_2.rows[int(2)].y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = *&(((&right_d_result_0)->rows + (int(2)))->y) + (*left_0).primal_1.rows[int(2)].z * dOut_2.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->w) = *&(((&left_d_result_0)->rows + (int(2)))->w) + (*right_0).primal_1.rows[int(3)].y * dOut_2.rows[int(2)].y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = *&(((&right_d_result_0)->rows + (int(3)))->y) + (*left_0).primal_1.rows[int(2)].w * dOut_2.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->x) = *&(((&left_d_result_0)->rows + (int(2)))->x) + (*right_0).primal_1.rows[int(0)].z * dOut_2.rows[int(2)].z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = *&(((&right_d_result_0)->rows + (int(0)))->z) + (*left_0).primal_1.rows[int(2)].x * dOut_2.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->y) = *&(((&left_d_result_0)->rows + (int(2)))->y) + (*right_0).primal_1.rows[int(1)].z * dOut_2.rows[int(2)].z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = *&(((&right_d_result_0)->rows + (int(1)))->z) + (*left_0).primal_1.rows[int(2)].y * dOut_2.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->z) = *&(((&left_d_result_0)->rows + (int(2)))->z) + (*right_0).primal_1.rows[int(2)].z * dOut_2.rows[int(2)].z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = *&(((&right_d_result_0)->rows + (int(2)))->z) + (*left_0).primal_1.rows[int(2)].z * dOut_2.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->w) = *&(((&left_d_result_0)->rows + (int(2)))->w) + (*right_0).primal_1.rows[int(3)].z * dOut_2.rows[int(2)].z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = *&(((&right_d_result_0)->rows + (int(3)))->z) + (*left_0).primal_1.rows[int(2)].w * dOut_2.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->x) = *&(((&left_d_result_0)->rows + (int(2)))->x) + (*right_0).primal_1.rows[int(0)].w * dOut_2.rows[int(2)].w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = *&(((&right_d_result_0)->rows + (int(0)))->w) + (*left_0).primal_1.rows[int(2)].x * dOut_2.rows[int(2)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->y) = *&(((&left_d_result_0)->rows + (int(2)))->y) + (*right_0).primal_1.rows[int(1)].w * dOut_2.rows[int(2)].w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = *&(((&right_d_result_0)->rows + (int(1)))->w) + (*left_0).primal_1.rows[int(2)].y * dOut_2.rows[int(2)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->z) = *&(((&left_d_result_0)->rows + (int(2)))->z) + (*right_0).primal_1.rows[int(2)].w * dOut_2.rows[int(2)].w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = *&(((&right_d_result_0)->rows + (int(2)))->w) + (*left_0).primal_1.rows[int(2)].z * dOut_2.rows[int(2)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(2)))->w) = *&(((&left_d_result_0)->rows + (int(2)))->w) + (*right_0).primal_1.rows[int(3)].w * dOut_2.rows[int(2)].w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = *&(((&right_d_result_0)->rows + (int(3)))->w) + (*left_0).primal_1.rows[int(2)].w * dOut_2.rows[int(2)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->x) = *&(((&left_d_result_0)->rows + (int(3)))->x) + (*right_0).primal_1.rows[int(0)].x * dOut_2.rows[int(3)].x;
    *&(((&right_d_result_0)->rows + (int(0)))->x) = *&(((&right_d_result_0)->rows + (int(0)))->x) + (*left_0).primal_1.rows[int(3)].x * dOut_2.rows[int(3)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->y) = *&(((&left_d_result_0)->rows + (int(3)))->y) + (*right_0).primal_1.rows[int(1)].x * dOut_2.rows[int(3)].x;
    *&(((&right_d_result_0)->rows + (int(1)))->x) = *&(((&right_d_result_0)->rows + (int(1)))->x) + (*left_0).primal_1.rows[int(3)].y * dOut_2.rows[int(3)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->z) = *&(((&left_d_result_0)->rows + (int(3)))->z) + (*right_0).primal_1.rows[int(2)].x * dOut_2.rows[int(3)].x;
    *&(((&right_d_result_0)->rows + (int(2)))->x) = *&(((&right_d_result_0)->rows + (int(2)))->x) + (*left_0).primal_1.rows[int(3)].z * dOut_2.rows[int(3)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->w) = *&(((&left_d_result_0)->rows + (int(3)))->w) + (*right_0).primal_1.rows[int(3)].x * dOut_2.rows[int(3)].x;
    *&(((&right_d_result_0)->rows + (int(3)))->x) = *&(((&right_d_result_0)->rows + (int(3)))->x) + (*left_0).primal_1.rows[int(3)].w * dOut_2.rows[int(3)].x;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->x) = *&(((&left_d_result_0)->rows + (int(3)))->x) + (*right_0).primal_1.rows[int(0)].y * dOut_2.rows[int(3)].y;
    *&(((&right_d_result_0)->rows + (int(0)))->y) = *&(((&right_d_result_0)->rows + (int(0)))->y) + (*left_0).primal_1.rows[int(3)].x * dOut_2.rows[int(3)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->y) = *&(((&left_d_result_0)->rows + (int(3)))->y) + (*right_0).primal_1.rows[int(1)].y * dOut_2.rows[int(3)].y;
    *&(((&right_d_result_0)->rows + (int(1)))->y) = *&(((&right_d_result_0)->rows + (int(1)))->y) + (*left_0).primal_1.rows[int(3)].y * dOut_2.rows[int(3)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->z) = *&(((&left_d_result_0)->rows + (int(3)))->z) + (*right_0).primal_1.rows[int(2)].y * dOut_2.rows[int(3)].y;
    *&(((&right_d_result_0)->rows + (int(2)))->y) = *&(((&right_d_result_0)->rows + (int(2)))->y) + (*left_0).primal_1.rows[int(3)].z * dOut_2.rows[int(3)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->w) = *&(((&left_d_result_0)->rows + (int(3)))->w) + (*right_0).primal_1.rows[int(3)].y * dOut_2.rows[int(3)].y;
    *&(((&right_d_result_0)->rows + (int(3)))->y) = *&(((&right_d_result_0)->rows + (int(3)))->y) + (*left_0).primal_1.rows[int(3)].w * dOut_2.rows[int(3)].y;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->x) = *&(((&left_d_result_0)->rows + (int(3)))->x) + (*right_0).primal_1.rows[int(0)].z * dOut_2.rows[int(3)].z;
    *&(((&right_d_result_0)->rows + (int(0)))->z) = *&(((&right_d_result_0)->rows + (int(0)))->z) + (*left_0).primal_1.rows[int(3)].x * dOut_2.rows[int(3)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->y) = *&(((&left_d_result_0)->rows + (int(3)))->y) + (*right_0).primal_1.rows[int(1)].z * dOut_2.rows[int(3)].z;
    *&(((&right_d_result_0)->rows + (int(1)))->z) = *&(((&right_d_result_0)->rows + (int(1)))->z) + (*left_0).primal_1.rows[int(3)].y * dOut_2.rows[int(3)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->z) = *&(((&left_d_result_0)->rows + (int(3)))->z) + (*right_0).primal_1.rows[int(2)].z * dOut_2.rows[int(3)].z;
    *&(((&right_d_result_0)->rows + (int(2)))->z) = *&(((&right_d_result_0)->rows + (int(2)))->z) + (*left_0).primal_1.rows[int(3)].z * dOut_2.rows[int(3)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->w) = *&(((&left_d_result_0)->rows + (int(3)))->w) + (*right_0).primal_1.rows[int(3)].z * dOut_2.rows[int(3)].z;
    *&(((&right_d_result_0)->rows + (int(3)))->z) = *&(((&right_d_result_0)->rows + (int(3)))->z) + (*left_0).primal_1.rows[int(3)].w * dOut_2.rows[int(3)].z;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->x) = *&(((&left_d_result_0)->rows + (int(3)))->x) + (*right_0).primal_1.rows[int(0)].w * dOut_2.rows[int(3)].w;
    *&(((&right_d_result_0)->rows + (int(0)))->w) = *&(((&right_d_result_0)->rows + (int(0)))->w) + (*left_0).primal_1.rows[int(3)].x * dOut_2.rows[int(3)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->y) = *&(((&left_d_result_0)->rows + (int(3)))->y) + (*right_0).primal_1.rows[int(1)].w * dOut_2.rows[int(3)].w;
    *&(((&right_d_result_0)->rows + (int(1)))->w) = *&(((&right_d_result_0)->rows + (int(1)))->w) + (*left_0).primal_1.rows[int(3)].y * dOut_2.rows[int(3)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->z) = *&(((&left_d_result_0)->rows + (int(3)))->z) + (*right_0).primal_1.rows[int(2)].w * dOut_2.rows[int(3)].w;
    *&(((&right_d_result_0)->rows + (int(2)))->w) = *&(((&right_d_result_0)->rows + (int(2)))->w) + (*left_0).primal_1.rows[int(3)].z * dOut_2.rows[int(3)].w;

#line 1411
    *&(((&left_d_result_0)->rows + (int(3)))->w) = *&(((&left_d_result_0)->rows + (int(3)))->w) + (*right_0).primal_1.rows[int(3)].w * dOut_2.rows[int(3)].w;
    *&(((&right_d_result_0)->rows + (int(3)))->w) = *&(((&right_d_result_0)->rows + (int(3)))->w) + (*left_0).primal_1.rows[int(3)].w * dOut_2.rows[int(3)].w;

#line 1412
    left_0->primal_1 = (*left_0).primal_1;

#line 1412
    left_0->differential_0 = left_d_result_0;

#line 1412
    right_0->primal_1 = (*right_0).primal_1;

#line 1412
    right_0->differential_0 = right_d_result_0;

#line 1418
    return;
}


#line 1418
struct DiffPair_matrixx3Cfloatx2C3x2C3x3E_0
{
    Matrix<float, 3, 3>  primal_1;
    Matrix<float, 3, 3>  differential_0;
};


#line 1386
__device__ void mul_1(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * left_1, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * right_1, Matrix<float, 3, 3>  dOut_3)
{
    Matrix<float, 3, 3>  left_d_result_1;

#line 1393
    *&(((&left_d_result_1)->rows + (int(0)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(0)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(0)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(1)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(1)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(1)))->z) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(2)))->x) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(2)))->y) = 0.0f;

#line 1393
    *&(((&left_d_result_1)->rows + (int(2)))->z) = 0.0f;

    Matrix<float, 3, 3>  right_d_result_1;

#line 1400
    *&(((&right_d_result_1)->rows + (int(0)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(0)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(0)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(1)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(1)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(1)))->z) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(2)))->x) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(2)))->y) = 0.0f;

#line 1400
    *&(((&right_d_result_1)->rows + (int(2)))->z) = 0.0f;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->x) = *&(((&left_d_result_1)->rows + (int(0)))->x) + (*right_1).primal_1.rows[int(0)].x * dOut_3.rows[int(0)].x;
    *&(((&right_d_result_1)->rows + (int(0)))->x) = *&(((&right_d_result_1)->rows + (int(0)))->x) + (*left_1).primal_1.rows[int(0)].x * dOut_3.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->y) = *&(((&left_d_result_1)->rows + (int(0)))->y) + (*right_1).primal_1.rows[int(1)].x * dOut_3.rows[int(0)].x;
    *&(((&right_d_result_1)->rows + (int(1)))->x) = *&(((&right_d_result_1)->rows + (int(1)))->x) + (*left_1).primal_1.rows[int(0)].y * dOut_3.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->z) = *&(((&left_d_result_1)->rows + (int(0)))->z) + (*right_1).primal_1.rows[int(2)].x * dOut_3.rows[int(0)].x;
    *&(((&right_d_result_1)->rows + (int(2)))->x) = *&(((&right_d_result_1)->rows + (int(2)))->x) + (*left_1).primal_1.rows[int(0)].z * dOut_3.rows[int(0)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->x) = *&(((&left_d_result_1)->rows + (int(0)))->x) + (*right_1).primal_1.rows[int(0)].y * dOut_3.rows[int(0)].y;
    *&(((&right_d_result_1)->rows + (int(0)))->y) = *&(((&right_d_result_1)->rows + (int(0)))->y) + (*left_1).primal_1.rows[int(0)].x * dOut_3.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->y) = *&(((&left_d_result_1)->rows + (int(0)))->y) + (*right_1).primal_1.rows[int(1)].y * dOut_3.rows[int(0)].y;
    *&(((&right_d_result_1)->rows + (int(1)))->y) = *&(((&right_d_result_1)->rows + (int(1)))->y) + (*left_1).primal_1.rows[int(0)].y * dOut_3.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->z) = *&(((&left_d_result_1)->rows + (int(0)))->z) + (*right_1).primal_1.rows[int(2)].y * dOut_3.rows[int(0)].y;
    *&(((&right_d_result_1)->rows + (int(2)))->y) = *&(((&right_d_result_1)->rows + (int(2)))->y) + (*left_1).primal_1.rows[int(0)].z * dOut_3.rows[int(0)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->x) = *&(((&left_d_result_1)->rows + (int(0)))->x) + (*right_1).primal_1.rows[int(0)].z * dOut_3.rows[int(0)].z;
    *&(((&right_d_result_1)->rows + (int(0)))->z) = *&(((&right_d_result_1)->rows + (int(0)))->z) + (*left_1).primal_1.rows[int(0)].x * dOut_3.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->y) = *&(((&left_d_result_1)->rows + (int(0)))->y) + (*right_1).primal_1.rows[int(1)].z * dOut_3.rows[int(0)].z;
    *&(((&right_d_result_1)->rows + (int(1)))->z) = *&(((&right_d_result_1)->rows + (int(1)))->z) + (*left_1).primal_1.rows[int(0)].y * dOut_3.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(0)))->z) = *&(((&left_d_result_1)->rows + (int(0)))->z) + (*right_1).primal_1.rows[int(2)].z * dOut_3.rows[int(0)].z;
    *&(((&right_d_result_1)->rows + (int(2)))->z) = *&(((&right_d_result_1)->rows + (int(2)))->z) + (*left_1).primal_1.rows[int(0)].z * dOut_3.rows[int(0)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->x) = *&(((&left_d_result_1)->rows + (int(1)))->x) + (*right_1).primal_1.rows[int(0)].x * dOut_3.rows[int(1)].x;
    *&(((&right_d_result_1)->rows + (int(0)))->x) = *&(((&right_d_result_1)->rows + (int(0)))->x) + (*left_1).primal_1.rows[int(1)].x * dOut_3.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->y) = *&(((&left_d_result_1)->rows + (int(1)))->y) + (*right_1).primal_1.rows[int(1)].x * dOut_3.rows[int(1)].x;
    *&(((&right_d_result_1)->rows + (int(1)))->x) = *&(((&right_d_result_1)->rows + (int(1)))->x) + (*left_1).primal_1.rows[int(1)].y * dOut_3.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->z) = *&(((&left_d_result_1)->rows + (int(1)))->z) + (*right_1).primal_1.rows[int(2)].x * dOut_3.rows[int(1)].x;
    *&(((&right_d_result_1)->rows + (int(2)))->x) = *&(((&right_d_result_1)->rows + (int(2)))->x) + (*left_1).primal_1.rows[int(1)].z * dOut_3.rows[int(1)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->x) = *&(((&left_d_result_1)->rows + (int(1)))->x) + (*right_1).primal_1.rows[int(0)].y * dOut_3.rows[int(1)].y;
    *&(((&right_d_result_1)->rows + (int(0)))->y) = *&(((&right_d_result_1)->rows + (int(0)))->y) + (*left_1).primal_1.rows[int(1)].x * dOut_3.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->y) = *&(((&left_d_result_1)->rows + (int(1)))->y) + (*right_1).primal_1.rows[int(1)].y * dOut_3.rows[int(1)].y;
    *&(((&right_d_result_1)->rows + (int(1)))->y) = *&(((&right_d_result_1)->rows + (int(1)))->y) + (*left_1).primal_1.rows[int(1)].y * dOut_3.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->z) = *&(((&left_d_result_1)->rows + (int(1)))->z) + (*right_1).primal_1.rows[int(2)].y * dOut_3.rows[int(1)].y;
    *&(((&right_d_result_1)->rows + (int(2)))->y) = *&(((&right_d_result_1)->rows + (int(2)))->y) + (*left_1).primal_1.rows[int(1)].z * dOut_3.rows[int(1)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->x) = *&(((&left_d_result_1)->rows + (int(1)))->x) + (*right_1).primal_1.rows[int(0)].z * dOut_3.rows[int(1)].z;
    *&(((&right_d_result_1)->rows + (int(0)))->z) = *&(((&right_d_result_1)->rows + (int(0)))->z) + (*left_1).primal_1.rows[int(1)].x * dOut_3.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->y) = *&(((&left_d_result_1)->rows + (int(1)))->y) + (*right_1).primal_1.rows[int(1)].z * dOut_3.rows[int(1)].z;
    *&(((&right_d_result_1)->rows + (int(1)))->z) = *&(((&right_d_result_1)->rows + (int(1)))->z) + (*left_1).primal_1.rows[int(1)].y * dOut_3.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(1)))->z) = *&(((&left_d_result_1)->rows + (int(1)))->z) + (*right_1).primal_1.rows[int(2)].z * dOut_3.rows[int(1)].z;
    *&(((&right_d_result_1)->rows + (int(2)))->z) = *&(((&right_d_result_1)->rows + (int(2)))->z) + (*left_1).primal_1.rows[int(1)].z * dOut_3.rows[int(1)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->x) = *&(((&left_d_result_1)->rows + (int(2)))->x) + (*right_1).primal_1.rows[int(0)].x * dOut_3.rows[int(2)].x;
    *&(((&right_d_result_1)->rows + (int(0)))->x) = *&(((&right_d_result_1)->rows + (int(0)))->x) + (*left_1).primal_1.rows[int(2)].x * dOut_3.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->y) = *&(((&left_d_result_1)->rows + (int(2)))->y) + (*right_1).primal_1.rows[int(1)].x * dOut_3.rows[int(2)].x;
    *&(((&right_d_result_1)->rows + (int(1)))->x) = *&(((&right_d_result_1)->rows + (int(1)))->x) + (*left_1).primal_1.rows[int(2)].y * dOut_3.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->z) = *&(((&left_d_result_1)->rows + (int(2)))->z) + (*right_1).primal_1.rows[int(2)].x * dOut_3.rows[int(2)].x;
    *&(((&right_d_result_1)->rows + (int(2)))->x) = *&(((&right_d_result_1)->rows + (int(2)))->x) + (*left_1).primal_1.rows[int(2)].z * dOut_3.rows[int(2)].x;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->x) = *&(((&left_d_result_1)->rows + (int(2)))->x) + (*right_1).primal_1.rows[int(0)].y * dOut_3.rows[int(2)].y;
    *&(((&right_d_result_1)->rows + (int(0)))->y) = *&(((&right_d_result_1)->rows + (int(0)))->y) + (*left_1).primal_1.rows[int(2)].x * dOut_3.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->y) = *&(((&left_d_result_1)->rows + (int(2)))->y) + (*right_1).primal_1.rows[int(1)].y * dOut_3.rows[int(2)].y;
    *&(((&right_d_result_1)->rows + (int(1)))->y) = *&(((&right_d_result_1)->rows + (int(1)))->y) + (*left_1).primal_1.rows[int(2)].y * dOut_3.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->z) = *&(((&left_d_result_1)->rows + (int(2)))->z) + (*right_1).primal_1.rows[int(2)].y * dOut_3.rows[int(2)].y;
    *&(((&right_d_result_1)->rows + (int(2)))->y) = *&(((&right_d_result_1)->rows + (int(2)))->y) + (*left_1).primal_1.rows[int(2)].z * dOut_3.rows[int(2)].y;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->x) = *&(((&left_d_result_1)->rows + (int(2)))->x) + (*right_1).primal_1.rows[int(0)].z * dOut_3.rows[int(2)].z;
    *&(((&right_d_result_1)->rows + (int(0)))->z) = *&(((&right_d_result_1)->rows + (int(0)))->z) + (*left_1).primal_1.rows[int(2)].x * dOut_3.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->y) = *&(((&left_d_result_1)->rows + (int(2)))->y) + (*right_1).primal_1.rows[int(1)].z * dOut_3.rows[int(2)].z;
    *&(((&right_d_result_1)->rows + (int(1)))->z) = *&(((&right_d_result_1)->rows + (int(1)))->z) + (*left_1).primal_1.rows[int(2)].y * dOut_3.rows[int(2)].z;

#line 1411
    *&(((&left_d_result_1)->rows + (int(2)))->z) = *&(((&left_d_result_1)->rows + (int(2)))->z) + (*right_1).primal_1.rows[int(2)].z * dOut_3.rows[int(2)].z;
    *&(((&right_d_result_1)->rows + (int(2)))->z) = *&(((&right_d_result_1)->rows + (int(2)))->z) + (*left_1).primal_1.rows[int(2)].z * dOut_3.rows[int(2)].z;

#line 1412
    left_1->primal_1 = (*left_1).primal_1;

#line 1412
    left_1->differential_0 = left_d_result_1;

#line 1412
    right_1->primal_1 = (*right_1).primal_1;

#line 1412
    right_1->differential_0 = right_d_result_1;

#line 1418
    return;
}


#line 11019 "hlsl.meta.slang"
__device__ Matrix<float, 4, 4>  mul_2(Matrix<float, 4, 4>  left_2, Matrix<float, 4, 4>  right_2)
{

#line 11031
    Matrix<float, 4, 4>  result_8;

#line 11031
    int r_0 = int(0);
    for(;;)
    {

#line 11032
        if(r_0 < int(4))
        {
        }
        else
        {

#line 11032
            break;
        }

#line 11032
        int _S50 = r_0;

#line 11032
        int c_0 = int(0);
        for(;;)
        {

#line 11033
            if(c_0 < int(4))
            {
            }
            else
            {

#line 11033
                break;
            }

#line 11033
            int i_11 = int(0);

#line 11033
            float sum_0 = 0.0f;


            for(;;)
            {

#line 11036
                if(i_11 < int(4))
                {
                }
                else
                {

#line 11036
                    break;
                }
                float sum_1 = sum_0 + _slang_vector_get_element(left_2.rows[_S50], i_11) * _slang_vector_get_element(right_2.rows[i_11], c_0);

#line 11036
                i_11 = i_11 + int(1);

#line 11036
                sum_0 = sum_1;

#line 11036
            }



            *_slang_vector_get_element_ptr(((&result_8)->rows + (r_0)), c_0) = sum_0;

#line 11033
            c_0 = c_0 + int(1);

#line 11033
        }

#line 11032
        r_0 = r_0 + int(1);

#line 11032
    }

#line 11042
    return result_8;
}


#line 11019
__device__ Matrix<float, 3, 3>  mul_3(Matrix<float, 3, 3>  left_3, Matrix<float, 3, 3>  right_3)
{

#line 11031
    Matrix<float, 3, 3>  result_9;

#line 11031
    int r_1 = int(0);
    for(;;)
    {

#line 11032
        if(r_1 < int(3))
        {
        }
        else
        {

#line 11032
            break;
        }

#line 11032
        int _S51 = r_1;

#line 11032
        int c_1 = int(0);
        for(;;)
        {

#line 11033
            if(c_1 < int(3))
            {
            }
            else
            {

#line 11033
                break;
            }

#line 11033
            int i_12 = int(0);

#line 11033
            float sum_2 = 0.0f;


            for(;;)
            {

#line 11036
                if(i_12 < int(3))
                {
                }
                else
                {

#line 11036
                    break;
                }
                float sum_3 = sum_2 + _slang_vector_get_element(left_3.rows[_S51], i_12) * _slang_vector_get_element(right_3.rows[i_12], c_1);

#line 11036
                i_12 = i_12 + int(1);

#line 11036
                sum_2 = sum_3;

#line 11036
            }



            *_slang_vector_get_element_ptr(((&result_9)->rows + (r_1)), c_1) = sum_2;

#line 11033
            c_1 = c_1 + int(1);

#line 11033
        }

#line 11032
        r_1 = r_1 + int(1);

#line 11032
    }

#line 11042
    return result_9;
}


#line 11042
struct DiffPair_vectorx3Cfloatx2C4x3E_0
{
    float4  primal_1;
    float4  differential_0;
};


#line 1349 "diff.meta.slang"
__device__ void _d_mul_0(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * left_4, DiffPair_vectorx3Cfloatx2C4x3E_0 * right_4, float4  dOut_4)
{

    float4  right_d_result_2;

#line 1352
    float _S52 = (*left_4).primal_1.rows[int(0)].x * dOut_4.x;

#line 1351
    Matrix<float, 4, 4>  left_d_result_2;

#line 1361
    *&(((&left_d_result_2)->rows + (int(0)))->x) = (*right_4).primal_1.x * dOut_4.x;

#line 1360
    float sum_4 = _S52 + (*left_4).primal_1.rows[int(1)].x * dOut_4.y;
    *&(((&left_d_result_2)->rows + (int(1)))->x) = (*right_4).primal_1.x * dOut_4.y;

#line 1360
    float sum_5 = sum_4 + (*left_4).primal_1.rows[int(2)].x * dOut_4.z;
    *&(((&left_d_result_2)->rows + (int(2)))->x) = (*right_4).primal_1.x * dOut_4.z;

#line 1360
    float sum_6 = sum_5 + (*left_4).primal_1.rows[int(3)].x * dOut_4.w;
    *&(((&left_d_result_2)->rows + (int(3)))->x) = (*right_4).primal_1.x * dOut_4.w;

    *&((&right_d_result_2)->x) = sum_6;

#line 1363
    float _S53 = (*left_4).primal_1.rows[int(0)].y * dOut_4.x;

#line 1361
    *&(((&left_d_result_2)->rows + (int(0)))->y) = (*right_4).primal_1.y * dOut_4.x;

#line 1360
    float sum_7 = _S53 + (*left_4).primal_1.rows[int(1)].y * dOut_4.y;
    *&(((&left_d_result_2)->rows + (int(1)))->y) = (*right_4).primal_1.y * dOut_4.y;

#line 1360
    float sum_8 = sum_7 + (*left_4).primal_1.rows[int(2)].y * dOut_4.z;
    *&(((&left_d_result_2)->rows + (int(2)))->y) = (*right_4).primal_1.y * dOut_4.z;

#line 1360
    float sum_9 = sum_8 + (*left_4).primal_1.rows[int(3)].y * dOut_4.w;
    *&(((&left_d_result_2)->rows + (int(3)))->y) = (*right_4).primal_1.y * dOut_4.w;

    *&((&right_d_result_2)->y) = sum_9;

#line 1363
    float _S54 = (*left_4).primal_1.rows[int(0)].z * dOut_4.x;

#line 1361
    *&(((&left_d_result_2)->rows + (int(0)))->z) = (*right_4).primal_1.z * dOut_4.x;

#line 1360
    float sum_10 = _S54 + (*left_4).primal_1.rows[int(1)].z * dOut_4.y;
    *&(((&left_d_result_2)->rows + (int(1)))->z) = (*right_4).primal_1.z * dOut_4.y;

#line 1360
    float sum_11 = sum_10 + (*left_4).primal_1.rows[int(2)].z * dOut_4.z;
    *&(((&left_d_result_2)->rows + (int(2)))->z) = (*right_4).primal_1.z * dOut_4.z;

#line 1360
    float sum_12 = sum_11 + (*left_4).primal_1.rows[int(3)].z * dOut_4.w;
    *&(((&left_d_result_2)->rows + (int(3)))->z) = (*right_4).primal_1.z * dOut_4.w;

    *&((&right_d_result_2)->z) = sum_12;

#line 1363
    float _S55 = (*left_4).primal_1.rows[int(0)].w * dOut_4.x;

#line 1361
    *&(((&left_d_result_2)->rows + (int(0)))->w) = (*right_4).primal_1.w * dOut_4.x;

#line 1360
    float sum_13 = _S55 + (*left_4).primal_1.rows[int(1)].w * dOut_4.y;
    *&(((&left_d_result_2)->rows + (int(1)))->w) = (*right_4).primal_1.w * dOut_4.y;

#line 1360
    float sum_14 = sum_13 + (*left_4).primal_1.rows[int(2)].w * dOut_4.z;
    *&(((&left_d_result_2)->rows + (int(2)))->w) = (*right_4).primal_1.w * dOut_4.z;

#line 1360
    float sum_15 = sum_14 + (*left_4).primal_1.rows[int(3)].w * dOut_4.w;
    *&(((&left_d_result_2)->rows + (int(3)))->w) = (*right_4).primal_1.w * dOut_4.w;

    *&((&right_d_result_2)->w) = sum_15;

#line 1363
    left_4->primal_1 = (*left_4).primal_1;

#line 1363
    left_4->differential_0 = left_d_result_2;

#line 1363
    right_4->primal_1 = (*right_4).primal_1;

#line 1363
    right_4->differential_0 = right_d_result_2;



    return;
}


#line 10938 "hlsl.meta.slang"
__device__ float4  mul_4(Matrix<float, 4, 4>  left_5, float4  right_5)
{

#line 10950
    float4  result_10;

#line 10950
    int i_13 = int(0);
    for(;;)
    {

#line 10951
        if(i_13 < int(4))
        {
        }
        else
        {

#line 10951
            break;
        }

#line 10951
        int _S56 = i_13;

#line 10951
        int j_0 = int(0);

#line 10951
        float sum_16 = 0.0f;


        for(;;)
        {

#line 10954
            if(j_0 < int(4))
            {
            }
            else
            {

#line 10954
                break;
            }
            float sum_17 = sum_16 + _slang_vector_get_element(left_5.rows[_S56], j_0) * _slang_vector_get_element(right_5, j_0);

#line 10954
            j_0 = j_0 + int(1);

#line 10954
            sum_16 = sum_17;

#line 10954
        }



        *_slang_vector_get_element_ptr(&result_10, i_13) = sum_16;

#line 10951
        i_13 = i_13 + int(1);

#line 10951
    }

#line 10960
    return result_10;
}


#line 105 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ float3  geom_transform_points_0(float3  point_0, Matrix<float, 4, 4>  transf_matrix_0)
{
    float4  p_out_0 = mul_4(transf_matrix_0, make_float4 (point_0.x, point_0.y, point_0.z, 1.0f));
    return float3 {p_out_0.x, p_out_0.y, p_out_0.z} / make_float3 (p_out_0.w + 1.00000001168609742e-07f);
}


__device__ float3  geom_transform_points2_0(float3  point_1, Matrix<float, 4, 4>  transf_matrix_1)
{

    return float3 {mul_4(transf_matrix_1, make_float4 (point_1.x, point_1.y, point_1.z, 1.0f)).x, mul_4(transf_matrix_1, make_float4 (point_1.x, point_1.y, point_1.z, 1.0f)).y, mul_4(transf_matrix_1, make_float4 (point_1.x, point_1.y, point_1.z, 1.0f)).z};
}


__device__ float3  project_point_0(float3  point_2, Camera_0 cam_0)
{

#line 120
    float3  proj_point_0 = geom_transform_points_0(point_2, mul_2(cam_0.proj_mat_1, cam_0.world_view_transform_1));

    *&((&proj_point_0)->z) = geom_transform_points2_0(point_2, cam_0.world_view_transform_1).z;
    return proj_point_0;
}


#line 123
struct DiffPair_float_0
{
    float primal_1;
    float differential_0;
};


#line 1935 "diff.meta.slang"
__device__ void _d_max_0(DiffPair_float_0 * dpx_0, DiffPair_float_0 * dpy_0, float dOut_5)
{
    DiffPair_float_0 _S57 = *dpx_0;

#line 1937
    float _S58;

#line 1937
    if((*dpx_0).primal_1 > (*dpy_0).primal_1)
    {

#line 1937
        _S58 = dOut_5;

#line 1937
    }
    else
    {

#line 1937
        _S58 = 0.0f;

#line 1937
    }

#line 1937
    dpx_0->primal_1 = _S57.primal_1;

#line 1937
    dpx_0->differential_0 = _S58;
    DiffPair_float_0 _S59 = *dpy_0;

#line 1938
    if((*dpy_0).primal_1 > _S57.primal_1)
    {

#line 1938
        _S58 = dOut_5;

#line 1938
    }
    else
    {

#line 1938
        _S58 = 0.0f;

#line 1938
    }

#line 1938
    dpy_0->primal_1 = _S59.primal_1;

#line 1938
    dpy_0->differential_0 = _S58;
    return;
}


#line 1923
__device__ DiffPair_float_0 _d_max_1(DiffPair_float_0 dpx_1, DiffPair_float_0 dpy_1)
{

    float _S60 = (F32_max((dpx_1.primal_1), (dpy_1.primal_1)));

#line 1926
    float _S61;
    if(dpx_1.primal_1 > dpy_1.primal_1)
    {

#line 1927
        _S61 = dpx_1.differential_0;

#line 1927
    }
    else
    {

#line 1927
        _S61 = dpy_1.differential_0;

#line 1927
    }

#line 1927
    DiffPair_float_0 _S62 = { _S60, _S61 };

#line 1925
    return _S62;
}


#line 1 "token paste"
__device__ void _d_sqrt_0(DiffPair_float_0 * dpx_2, float dOut_6)
{

#line 1719 "diff.meta.slang"
    float _S63 = 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), ((*dpx_2).primal_1)))))) * dOut_6;

#line 1719
    dpx_2->primal_1 = (*dpx_2).primal_1;

#line 1719
    dpx_2->differential_0 = _S63;



    return;
}


#line 1 "token paste"
__device__ DiffPair_float_0 _d_sqrt_1(DiffPair_float_0 dpx_3)
{

#line 1689 "diff.meta.slang"
    DiffPair_float_0 _S64 = { (F32_sqrt((dpx_3.primal_1))), 0.5f / (F32_sqrt(((F32_max((1.00000001168609742e-07f), (dpx_3.primal_1)))))) * dpx_3.differential_0 };


    return _S64;
}


#line 7891 "hlsl.meta.slang"
__device__ float dot_0(float3  x_0, float3  y_0)
{

#line 7891
    int i_14 = int(0);

#line 7891
    float result_11 = 0.0f;

#line 7904
    for(;;)
    {

#line 7904
        if(i_14 < int(3))
        {
        }
        else
        {

#line 7904
            break;
        }

#line 7905
        float result_12 = result_11 + _slang_vector_get_element(x_0, i_14) * _slang_vector_get_element(y_0, i_14);

#line 7904
        i_14 = i_14 + int(1);

#line 7904
        result_11 = result_12;

#line 7904
    }

    return result_11;
}


#line 9729
__device__ float length_0(float3  x_1)
{

#line 9741
    return (F32_sqrt((dot_0(x_1, x_1))));
}


#line 11211
__device__ float3  normalize_0(float3  x_2)
{

#line 11223
    return x_2 / make_float3 (length_0(x_2));
}


#line 11223
struct DiffPair_vectorx3Cfloatx2C3x3E_0
{
    float3  primal_1;
    float3  differential_0;
};


#line 1 "token paste"
__device__ void _d_max_vector_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_4, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpy_2, float3  dOut_7)
{

#line 1558 "diff.meta.slang"
    DiffPair_float_0 left_dp_0;

#line 1558
    (&left_dp_0)->primal_1 = (*dpx_4).primal_1.x;

#line 1558
    (&left_dp_0)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_0;

#line 1559
    (&right_dp_0)->primal_1 = (*dpy_2).primal_1.x;

#line 1559
    (&right_dp_0)->differential_0 = 0.0f;
    _d_max_0(&left_dp_0, &right_dp_0, dOut_7.x);

#line 1555
    float3  left_d_result_3;

#line 1561
    *&((&left_d_result_3)->x) = left_dp_0.differential_0;

#line 1555
    float3  right_d_result_3;

#line 1562
    *&((&right_d_result_3)->x) = right_dp_0.differential_0;

#line 1558
    DiffPair_float_0 left_dp_1;

#line 1558
    (&left_dp_1)->primal_1 = (*dpx_4).primal_1.y;

#line 1558
    (&left_dp_1)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_1;

#line 1559
    (&right_dp_1)->primal_1 = (*dpy_2).primal_1.y;

#line 1559
    (&right_dp_1)->differential_0 = 0.0f;
    _d_max_0(&left_dp_1, &right_dp_1, dOut_7.y);
    *&((&left_d_result_3)->y) = left_dp_1.differential_0;
    *&((&right_d_result_3)->y) = right_dp_1.differential_0;

#line 1558
    DiffPair_float_0 left_dp_2;

#line 1558
    (&left_dp_2)->primal_1 = (*dpx_4).primal_1.z;

#line 1558
    (&left_dp_2)->differential_0 = 0.0f;
    DiffPair_float_0 right_dp_2;

#line 1559
    (&right_dp_2)->primal_1 = (*dpy_2).primal_1.z;

#line 1559
    (&right_dp_2)->differential_0 = 0.0f;
    _d_max_0(&left_dp_2, &right_dp_2, dOut_7.z);
    *&((&left_d_result_3)->z) = left_dp_2.differential_0;
    *&((&right_d_result_3)->z) = right_dp_2.differential_0;

#line 1562
    dpx_4->primal_1 = (*dpx_4).primal_1;

#line 1562
    dpx_4->differential_0 = left_d_result_3;

#line 1562
    dpy_2->primal_1 = (*dpy_2).primal_1;

#line 1562
    dpy_2->differential_0 = right_d_result_3;



    return;
}


#line 1 "token paste"
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 _d_max_vector_1(DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_5, DiffPair_vectorx3Cfloatx2C3x3E_0 dpy_3)
{

#line 1514 "diff.meta.slang"
    DiffPair_float_0 _S65 = { dpx_5.primal_1.x, dpx_5.differential_0.x };

#line 1514
    DiffPair_float_0 _S66 = { dpy_3.primal_1.x, dpy_3.differential_0.x };

#line 1520
    DiffPair_float_0 dp_elem_0 = _d_max_1(_S65, _S66);

#line 1516
    float3  result_13;

#line 1523
    *&((&result_13)->x) = dp_elem_0.primal_1;

#line 1517
    float3  d_result_0;

#line 1524
    *&((&d_result_0)->x) = dp_elem_0.differential_0;

#line 1524
    DiffPair_float_0 _S67 = { dpx_5.primal_1.y, dpx_5.differential_0.y };

#line 1524
    DiffPair_float_0 _S68 = { dpy_3.primal_1.y, dpy_3.differential_0.y };

#line 1520
    DiffPair_float_0 dp_elem_1 = _d_max_1(_S67, _S68);


    *&((&result_13)->y) = dp_elem_1.primal_1;
    *&((&d_result_0)->y) = dp_elem_1.differential_0;

#line 1524
    DiffPair_float_0 _S69 = { dpx_5.primal_1.z, dpx_5.differential_0.z };

#line 1524
    DiffPair_float_0 _S70 = { dpy_3.primal_1.z, dpy_3.differential_0.z };

#line 1520
    DiffPair_float_0 dp_elem_2 = _d_max_1(_S69, _S70);


    *&((&result_13)->z) = dp_elem_2.primal_1;
    *&((&d_result_0)->z) = dp_elem_2.differential_0;

#line 1524
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S71 = { result_13, d_result_0 };

    return _S71;
}


#line 10224 "hlsl.meta.slang"
__device__ float3  max_0(float3  x_3, float3  y_1)
{

#line 5510
    float3  result_14;

#line 5510
    int i_15 = int(0);

#line 5510
    for(;;)
    {

#line 5510
        if(i_15 < int(3))
        {
        }
        else
        {

#line 5510
            break;
        }

#line 5510
        *_slang_vector_get_element_ptr(&result_14, i_15) = (F32_max((_slang_vector_get_element(x_3, i_15)), (_slang_vector_get_element(y_1, i_15))));

#line 5510
        i_15 = i_15 + int(1);

#line 5510
    }

#line 5510
    return result_14;
}


#line 94 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
__device__ float3  compute_color_from_sh_coeffs_0(SpherHarmCoeffs_0 sh_0, float3  g_xyz_ws_0, float3  cam_pos_0, uint active_sh_2)
{
    float3  dir_0 = normalize_0(g_xyz_ws_0 - cam_pos_0);

    float3  rgb_1 = make_float3 (0.282094806432724f) * sh_0.coeff0_0;

#line 98
    float3  rgb_2;
    if(active_sh_2 > 0U)
    {

#line 100
        float _S72 = dir_0.y;

#line 100
        float _S73 = dir_0.z;

#line 100
        float _S74 = dir_0.x;

#line 100
        float3  rgb_3 = rgb_1 - make_float3 (0.48860251903533936f * _S72) * sh_0.coeff1_0 + make_float3 (0.48860251903533936f * _S73) * sh_0.coeff2_0 - make_float3 (0.48860251903533936f * _S74) * sh_0.coeff3_0;
        if(active_sh_2 > 1U)
        {
            float xx_0 = _S74 * _S74;

#line 103
            float yy_0 = _S72 * _S72;

#line 103
            float zz_0 = _S73 * _S73;
            float xy_0 = _S74 * _S72;



            float _S75 = 2.0f * zz_0;

            float _S76 = xx_0 - yy_0;

#line 109
            float3  rgb_4 = rgb_3 + make_float3 (1.09254848957061768f * xy_0) * sh_0.coeff4_0 + make_float3 (-1.09254848957061768f * (_S72 * _S73)) * sh_0.coeff5_0 + make_float3 (0.31539157032966614f * (_S75 - xx_0 - yy_0)) * sh_0.coeff6_0 + make_float3 (-1.09254848957061768f * (_S74 * _S73)) * sh_0.coeff7_0 + make_float3 (0.54627424478530884f * _S76) * sh_0.coeff8_0;


            if(active_sh_2 > 2U)
            {

                float _S77 = 3.0f * xx_0;

                float _S78 = 4.0f * zz_0 - xx_0 - yy_0;
                float _S79 = 3.0f * yy_0;

#line 118
                rgb_2 = rgb_4 + make_float3 (-0.59004360437393188f * _S72 * (_S77 - yy_0)) * sh_0.coeff9_0 + make_float3 (2.89061141014099121f * xy_0 * _S73) * sh_0.coeff10_0 + make_float3 (-0.4570457935333252f * _S72 * _S78) * sh_0.coeff11_0 + make_float3 (0.37317633628845215f * _S73 * (_S75 - _S77 - _S79)) * sh_0.coeff12_0 + make_float3 (-0.4570457935333252f * _S74 * _S78) * sh_0.coeff13_0 + make_float3 (1.44530570507049561f * _S73 * _S76) * sh_0.coeff14_0 + make_float3 (-0.59004360437393188f * _S74 * (xx_0 - _S79)) * sh_0.coeff15_0;

#line 112
            }
            else
            {

#line 112
                rgb_2 = rgb_4;

#line 112
            }

#line 101
        }
        else
        {

#line 101
            rgb_2 = rgb_3;

#line 101
        }

#line 99
    }
    else
    {

#line 99
        rgb_2 = rgb_1;

#line 99
    }

#line 128
    return max_0(rgb_2 + make_float3 (0.5f), make_float3 (0.0f));
}


#line 12514 "hlsl.meta.slang"
__device__ Matrix<float, 3, 3>  transpose_0(Matrix<float, 3, 3>  x_4)
{

#line 12525
    Matrix<float, 3, 3>  result_15;

#line 12525
    int r_2 = int(0);
    for(;;)
    {

#line 12526
        if(r_2 < int(3))
        {
        }
        else
        {

#line 12526
            break;
        }

#line 12526
        int c_2 = int(0);
        for(;;)
        {

#line 12527
            if(c_2 < int(3))
            {
            }
            else
            {

#line 12527
                break;
            }

#line 12528
            *_slang_vector_get_element_ptr(((&result_15)->rows + (r_2)), c_2) = _slang_vector_get_element(x_4.rows[c_2], r_2);

#line 12527
            c_2 = c_2 + int(1);

#line 12527
        }

#line 12526
        r_2 = r_2 + int(1);

#line 12526
    }


    return result_15;
}


#line 280 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ Matrix<float, 3, 3>  get_covariance_from_quat_scales_0(float4  q_0, float3  s_0)
{

#line 280
    float y_2 = q_0.z;



    float _S80 = y_2 * y_2;

#line 284
    float _S81 = q_0.w * q_0.w;

#line 284
    float _S82 = q_0.y * q_0.z;

#line 284
    float _S83 = q_0.x * q_0.w;

#line 284
    float _S84 = q_0.y * q_0.w;

#line 284
    float _S85 = q_0.x * q_0.z;
    float _S86 = q_0.y * q_0.y;

#line 285
    float _S87 = q_0.z * q_0.w;

#line 285
    float _S88 = q_0.x * q_0.y;

#line 292
    Matrix<float, 3, 3>  L_0 = mul_3(makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S80 + _S81), 2.0f * (_S82 - _S83), 2.0f * (_S84 + _S85), 2.0f * (_S82 + _S83), 1.0f - 2.0f * (_S86 + _S81), 2.0f * (_S87 - _S88), 2.0f * (_S84 - _S85), 2.0f * (_S87 + _S88), 1.0f - 2.0f * (_S86 + _S80)), makeMatrix<float, 3, 3> (s_0.x, 0.0f, 0.0f, 0.0f, s_0.y, 0.0f, 0.0f, 0.0f, s_0.z));

    return mul_3(L_0, transpose_0(L_0));
}


#line 1 "token paste"
__device__ void _d_tan_0(DiffPair_float_0 * dpx_6, float dOut_8)
{

#line 1719 "diff.meta.slang"
    float _S89 = 1.0f / ((F32_cos(((*dpx_6).primal_1))) * (F32_cos(((*dpx_6).primal_1)))) * dOut_8;

#line 1719
    dpx_6->primal_1 = (*dpx_6).primal_1;

#line 1719
    dpx_6->differential_0 = _S89;



    return;
}


#line 1 "token paste"
__device__ DiffPair_float_0 _d_tan_1(DiffPair_float_0 dpx_7)
{

#line 1805 "diff.meta.slang"
    float _S90 = (F32_cos((dpx_7.primal_1)));

#line 1805
    DiffPair_float_0 _S91 = { (F32_tan((dpx_7.primal_1))), 1.0f / (_S90 * _S90) * dpx_7.differential_0 };

#line 1692
    return _S91;
}


#line 1960
__device__ void _d_min_0(DiffPair_float_0 * dpx_8, DiffPair_float_0 * dpy_4, float dOut_9)
{
    DiffPair_float_0 _S92 = *dpx_8;

#line 1962
    float _S93;

#line 1962
    if((*dpx_8).primal_1 < (*dpy_4).primal_1)
    {

#line 1962
        _S93 = dOut_9;

#line 1962
    }
    else
    {

#line 1962
        _S93 = 0.0f;

#line 1962
    }

#line 1962
    dpx_8->primal_1 = _S92.primal_1;

#line 1962
    dpx_8->differential_0 = _S93;
    DiffPair_float_0 _S94 = *dpy_4;

#line 1963
    if((*dpy_4).primal_1 < _S92.primal_1)
    {

#line 1963
        _S93 = dOut_9;

#line 1963
    }
    else
    {

#line 1963
        _S93 = 0.0f;

#line 1963
    }

#line 1963
    dpy_4->primal_1 = _S94.primal_1;

#line 1963
    dpy_4->differential_0 = _S93;
    return;
}


#line 1948
__device__ DiffPair_float_0 _d_min_1(DiffPair_float_0 dpx_9, DiffPair_float_0 dpy_5)
{

    float _S95 = (F32_min((dpx_9.primal_1), (dpy_5.primal_1)));

#line 1951
    float _S96;
    if(dpx_9.primal_1 < dpy_5.primal_1)
    {

#line 1952
        _S96 = dpx_9.differential_0;

#line 1952
    }
    else
    {

#line 1952
        _S96 = dpy_5.differential_0;

#line 1952
    }

#line 1952
    DiffPair_float_0 _S97 = { _S95, _S96 };

#line 1950
    return _S97;
}


#line 127 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ Matrix<float, 3, 3>  compute_jacobian_0(float3  xyz_ws_2, Camera_0 cam_1)
{

#line 128
    float tan_half_fovx_0 = (F32_tan((cam_1.fovx_1 / 2.0f)));
    float tan_half_fovy_0 = (F32_tan((cam_1.fovy_1 / 2.0f)));
    float h_x_0 = float(cam_1.W_0) / (2.0f * tan_half_fovx_0);
    float h_y_0 = float(cam_1.H_0) / (2.0f * tan_half_fovy_0);

    float3  _S98 = geom_transform_points_0(xyz_ws_2, cam_1.world_view_transform_1);

#line 133
    float3  t_0 = _S98;


    float limx_0 = 1.29999995231628418f * tan_half_fovx_0;
    float limy_0 = 1.29999995231628418f * tan_half_fovy_0;
    float _S99 = _S98.z;
    float tytz_0 = _S98.y / _S99;
    *&((&t_0)->x) = (F32_min((limx_0), ((F32_max((- limx_0), (_S98.x / _S99)))))) * _S99;
    *&((&t_0)->y) = (F32_min((limy_0), ((F32_max((- limy_0), (tytz_0)))))) * t_0.z;

#line 147
    return makeMatrix<float, 3, 3> (h_x_0 / t_0.z, 0.0f, - (h_x_0 * t_0.x) / (t_0.z * t_0.z), 0.0f, h_y_0 / t_0.z, - (h_y_0 * t_0.y) / (t_0.z * t_0.z), 0.0f, 0.0f, 0.0f);
}


__device__ Matrix<float, 2, 2>  covariance_3d_to_2d_0(Camera_0 cam_2, float3  xyz_ws_3, Matrix<float, 3, 3>  cov_ws_0)
{

#line 151
    Matrix<float, 3, 3>  _S100 = makeMatrix<float, 3, 3> (float3 {cam_2.world_view_transform_1.rows[int(0)].x, cam_2.world_view_transform_1.rows[int(0)].y, cam_2.world_view_transform_1.rows[int(0)].z}, float3 {cam_2.world_view_transform_1.rows[int(1)].x, cam_2.world_view_transform_1.rows[int(1)].y, cam_2.world_view_transform_1.rows[int(1)].z}, float3 {cam_2.world_view_transform_1.rows[int(2)].x, cam_2.world_view_transform_1.rows[int(2)].y, cam_2.world_view_transform_1.rows[int(2)].z});

    Matrix<float, 3, 3>  J_0 = compute_jacobian_0(xyz_ws_3, cam_2);
    Matrix<float, 3, 3>  cov_vs_1 = mul_3(J_0, mul_3(_S100, mul_3(cov_ws_0, mul_3(transpose_0(_S100), transpose_0(J_0)))));
    *&(((&cov_vs_1)->rows + (int(0)))->x) = *&(((&cov_vs_1)->rows + (int(0)))->x) + 0.30000001192092896f;
    *&(((&cov_vs_1)->rows + (int(1)))->y) = *&(((&cov_vs_1)->rows + (int(1)))->y) + 0.30000001192092896f;

    return makeMatrix<float, 2, 2> (float2 {cov_vs_1.rows[int(0)].x, cov_vs_1.rows[int(0)].y}, float2 {cov_vs_1.rows[int(1)].x, cov_vs_1.rows[int(1)].y});
}


#line 222
__device__ Splat_2D_Vertex_0 project_gaussian_to_camera_0(Gaussian_3D_0 g_0, Camera_0 cam_3, uint active_sh_3)
{

#line 223
    float3  xyz_vs_1 = project_point_0(g_0.xyz_ws_0, cam_3);
    if(xyz_vs_1.z <= 0.20000000298023224f)
    {

#line 225
        float3  _S101 = make_float3 (0.0f);

#line 225
        Splat_2D_Vertex_0 _S102 = { _S101, _S101, makeMatrix<float, 2, 2> (0.0f) };

#line 225
        return _S102;
    }

#line 231
    Splat_2D_Vertex_0 _S103 = { xyz_vs_1, compute_color_from_sh_coeffs_0(g_0.sh_coeffs_0, g_0.xyz_ws_0, cam_3.position_1, active_sh_3), covariance_3d_to_2d_0(cam_3, g_0.xyz_ws_0, get_covariance_from_quat_scales_0(g_0.rotations_0, g_0.scales_0)) };

#line 231
    return _S103;
}


#line 203
__device__ float compute_det_0(Matrix<float, 2, 2>  M_0)
{

#line 204
    return M_0.rows[int(0)].x * M_0.rows[int(1)].y - M_0.rows[int(0)].y * M_0.rows[int(1)].x;
}


#line 193
__device__ float splat_radius_0(Matrix<float, 2, 2>  cov_vs_2, float det_0)
{

#line 194
    float mid_0 = 0.5f * (cov_vs_2.rows[int(0)].x + cov_vs_2.rows[int(1)].y);
    float _S104 = (F32_sqrt(((F32_max((0.10000000149011612f), (mid_0 * mid_0 - det_0))))));



    return (F32_ceil((3.0f * (F32_sqrt(((F32_max((mid_0 + _S104), (mid_0 - _S104)))))))));
}


#line 61
__device__ float ndc2pix_0(float v_0, int S_0)
{
    return ((v_0 + 1.0f) * float(S_0) - 1.0f) * 0.5f;
}


#line 72
__device__ float clip_0(float val_0, float min_val_0, float max_val_0)
{
    return (F32_max((min_val_0), ((F32_min((max_val_0), (val_0))))));
}


#line 4 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
struct rectangle_0
{
    int min_x_0;
    int min_y_0;
    int max_x_0;
    int max_y_0;
};

__device__ rectangle_0 get_rectangle_tile_space_0(float2  ndc_xy_0, float radius_0, uint grid_height_0, uint grid_width_0, uint tile_height_0, uint tile_width_0)
{

#line 20
    rectangle_0 rect_tile_space_0;

    float _S105 = ndc_xy_0.x;

#line 22
    float _S106 = float(tile_width_0);

#line 22
    float _S107 = float(grid_width_0);

#line 22
    (&rect_tile_space_0)->min_x_0 = int((F32_floor((clip_0((_S105 - radius_0) / _S106, 0.0f, _S107)))));
    float _S108 = ndc_xy_0.y;

#line 23
    float _S109 = float(tile_height_0);

#line 23
    float _S110 = float(grid_height_0);

#line 23
    (&rect_tile_space_0)->min_y_0 = int((F32_floor((clip_0((_S108 - radius_0) / _S109, 0.0f, _S110)))));
    (&rect_tile_space_0)->max_x_0 = int((F32_ceil((clip_0((_S105 + radius_0) / _S106, 0.0f, _S107)))));
    (&rect_tile_space_0)->max_y_0 = int((F32_ceil((clip_0((_S108 + radius_0) / _S109, 0.0f, _S110)))));

#line 31
    return rect_tile_space_0;
}


#line 1035 "diff.meta.slang"
__device__ void DiffTensorView_storeOnce_forward_0(DiffTensorView_0 this_11, uint2  x_5, DiffPair_float_0 dpval_0)
{
    (this_11.primal_0).store<float>((x_5), (dpval_0.primal_1));
    AtomicAdd_storeOnce_forward_0(this_11.diff_1, x_5, dpval_0.differential_0);
    return;
}


#line 1035
__device__ void DiffTensorView_storeOnce_forward_1(DiffTensorView_0 this_12, uint3  x_6, DiffPair_float_0 dpval_1)
{
    (this_12.primal_0).store<float>((x_6), (dpval_1.primal_1));
    AtomicAdd_storeOnce_forward_1(this_12.diff_1, x_6, dpval_1.differential_0);
    return;
}


#line 1026
__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_13, uint2  x_7, float val_1)
{

#line 1026
    (this_13.primal_0).store<float>((x_7), (val_1));

#line 1026
    return;
}


#line 1026
__device__ void DiffTensorView_storeOnce_1(DiffTensorView_0 this_14, uint3  x_8, float val_2)
{

#line 1026
    (this_14.primal_0).store<float>((x_8), (val_2));

#line 1026
    return;
}


#line 37 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
struct s_bwd_prop_vertex_shader_Intermediates_0
{
    Camera_0 _S111;
    Gaussian_3D_0 _S112;
};


#line 65
__device__ float3  s_primal_ctx_read_t3_float3_0(uint idx_2, DiffTensorView_0 t3_1)
{

#line 26 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    return make_float3 (DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 0U)), DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 1U)), DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 2U)));
}


#line 26
__device__ SpherHarmCoeffs_0 s_primal_ctx_read_spherical_harmonics_coeffs_0(uint g_idx_2, DiffTensorView_0 sh_coeffs_3, uint active_sh_4)
{

#line 64 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
    float3  _S113 = make_float3 (0.0f);
    float3  _S114 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 2U)));

#line 65
    SpherHarmCoeffs_0 g_sh_coeffs_1;

    if(active_sh_4 > 0U)
    {

#line 68
        float3  _S115 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 2U)));
        float3  _S116 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 2U)));
        float3  _S117 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 2U)));

        if(active_sh_4 > 1U)
        {

#line 73
            float3  _S118 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 2U)));
            float3  _S119 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 2U)));
            float3  _S120 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 2U)));
            float3  _S121 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 2U)));
            float3  _S122 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 2U)));

            if(active_sh_4 > 2U)
            {

#line 80
                float3  _S123 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 2U)));
                float3  _S124 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 2U)));
                float3  _S125 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 2U)));
                float3  _S126 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 2U)));
                float3  _S127 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 2U)));
                float3  _S128 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 2U)));
                float3  _S129 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 2U)));

#line 86
                (&g_sh_coeffs_1)->coeff0_0 = _S114;

#line 86
                (&g_sh_coeffs_1)->coeff1_0 = _S115;

#line 86
                (&g_sh_coeffs_1)->coeff2_0 = _S116;

#line 86
                (&g_sh_coeffs_1)->coeff3_0 = _S117;

#line 86
                (&g_sh_coeffs_1)->coeff4_0 = _S118;

#line 86
                (&g_sh_coeffs_1)->coeff5_0 = _S119;

#line 86
                (&g_sh_coeffs_1)->coeff6_0 = _S120;

#line 86
                (&g_sh_coeffs_1)->coeff7_0 = _S121;

#line 86
                (&g_sh_coeffs_1)->coeff8_0 = _S122;

#line 86
                (&g_sh_coeffs_1)->coeff9_0 = _S123;

#line 86
                (&g_sh_coeffs_1)->coeff10_0 = _S124;

#line 86
                (&g_sh_coeffs_1)->coeff11_0 = _S125;

#line 86
                (&g_sh_coeffs_1)->coeff12_0 = _S126;

#line 86
                (&g_sh_coeffs_1)->coeff13_0 = _S127;

#line 86
                (&g_sh_coeffs_1)->coeff14_0 = _S128;

#line 86
                (&g_sh_coeffs_1)->coeff15_0 = _S129;

#line 79
            }
            else
            {

#line 79
                (&g_sh_coeffs_1)->coeff0_0 = _S114;

#line 79
                (&g_sh_coeffs_1)->coeff1_0 = _S115;

#line 79
                (&g_sh_coeffs_1)->coeff2_0 = _S116;

#line 79
                (&g_sh_coeffs_1)->coeff3_0 = _S117;

#line 79
                (&g_sh_coeffs_1)->coeff4_0 = _S118;

#line 79
                (&g_sh_coeffs_1)->coeff5_0 = _S119;

#line 79
                (&g_sh_coeffs_1)->coeff6_0 = _S120;

#line 79
                (&g_sh_coeffs_1)->coeff7_0 = _S121;

#line 79
                (&g_sh_coeffs_1)->coeff8_0 = _S122;

#line 79
                (&g_sh_coeffs_1)->coeff9_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff10_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff11_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff12_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff13_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff14_0 = _S113;

#line 79
                (&g_sh_coeffs_1)->coeff15_0 = _S113;

#line 79
            }

#line 72
        }
        else
        {

#line 72
            (&g_sh_coeffs_1)->coeff0_0 = _S114;

#line 72
            (&g_sh_coeffs_1)->coeff1_0 = _S115;

#line 72
            (&g_sh_coeffs_1)->coeff2_0 = _S116;

#line 72
            (&g_sh_coeffs_1)->coeff3_0 = _S117;

#line 72
            (&g_sh_coeffs_1)->coeff4_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff5_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff6_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff7_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff8_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff9_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff10_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff11_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff12_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff13_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff14_0 = _S113;

#line 72
            (&g_sh_coeffs_1)->coeff15_0 = _S113;

#line 72
        }

#line 67
    }
    else
    {

#line 67
        (&g_sh_coeffs_1)->coeff0_0 = _S114;

#line 67
        (&g_sh_coeffs_1)->coeff1_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff2_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff3_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff4_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff5_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff6_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff7_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff8_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff9_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff10_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff11_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff12_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff13_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff14_0 = _S113;

#line 67
        (&g_sh_coeffs_1)->coeff15_0 = _S113;

#line 67
    }

#line 67
    return g_sh_coeffs_1;
}


#line 67
__device__ float4  s_primal_ctx_read_t4_float4_0(uint idx_3, DiffTensorView_0 t4_1)
{

#line 34 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    return make_float4 (DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 0U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 1U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 2U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 3U)));
}


#line 34
__device__ Gaussian_3D_0 s_primal_ctx_load_gaussian_0(int g_idx_3, DiffTensorView_0 xyz_ws_4, DiffTensorView_0 sh_coeffs_4, DiffTensorView_0 rotations_2, DiffTensorView_0 scales_2, uint active_sh_5)
{

#line 177
    uint _S130 = uint(g_idx_3);

#line 182
    Gaussian_3D_0 _S131 = { s_primal_ctx_read_t3_float3_0(_S130, xyz_ws_4), s_primal_ctx_read_spherical_harmonics_coeffs_0(_S130, sh_coeffs_4, active_sh_5), s_primal_ctx_read_t4_float4_0(_S130, rotations_2), s_primal_ctx_read_t3_float3_0(_S130, scales_2) };

#line 182
    return _S131;
}


#line 182
__device__ Matrix<float, 4, 4>  s_primal_ctx_mul_0(Matrix<float, 4, 4>  _S132, Matrix<float, 4, 4>  _S133)
{

#line 182
    return mul_2(_S132, _S133);
}


#line 182
__device__ float4  s_primal_ctx_mul_1(Matrix<float, 4, 4>  _S134, float4  _S135)
{

#line 182
    return mul_4(_S134, _S135);
}


#line 182
__device__ float3  s_primal_ctx_geom_transform_points_0(float3  dppoint_0, Matrix<float, 4, 4>  dptransf_matrix_0)
{

#line 105
    float4  _S136 = s_primal_ctx_mul_1(dptransf_matrix_0, make_float4 (dppoint_0.x, dppoint_0.y, dppoint_0.z, 1.0f));

#line 105
    return float3 {_S136.x, _S136.y, _S136.z} / make_float3 (_S136.w + 1.00000001168609742e-07f);
}


#line 105
__device__ float3  s_primal_ctx_geom_transform_points2_0(float3  dppoint_1, Matrix<float, 4, 4>  dptransf_matrix_1)
{

#line 112
    return float3 {s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).x, s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).y, s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).z};
}


#line 112
__device__ float3  s_primal_ctx_project_point_0(float3  dppoint_2, Camera_0 dpcam_0)
{

#line 122
    float _S137 = s_primal_ctx_geom_transform_points2_0(dppoint_2, dpcam_0.world_view_transform_1).z;

#line 122
    float3  _S138 = s_primal_ctx_geom_transform_points_0(dppoint_2, s_primal_ctx_mul_0(dpcam_0.proj_mat_1, dpcam_0.world_view_transform_1));

#line 122
    *&((&_S138)->z) = _S137;

#line 122
    return _S138;
}


#line 122
__device__ float3  s_primal_ctx_max_0(float3  _S139, float3  _S140)
{

#line 122
    return max_0(_S139, _S140);
}


#line 122
__device__ float3  s_primal_ctx_compute_color_from_sh_coeffs_0(SpherHarmCoeffs_0 dpsh_0, float3  dpg_xyz_ws_0, float3  dpcam_pos_0, uint active_sh_6)
{

#line 96 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
    float3  _S141 = normalize_0(dpg_xyz_ws_0 - dpcam_pos_0);

    float3  rgb_5 = make_float3 (0.282094806432724f) * dpsh_0.coeff0_0;

#line 98
    float3  rgb_6;
    if(active_sh_6 > 0U)
    {

#line 100
        float _S142 = _S141.y;

#line 100
        float _S143 = _S141.z;

#line 100
        float _S144 = _S141.x;

#line 100
        float3  rgb_7 = rgb_5 - make_float3 (0.48860251903533936f * _S142) * dpsh_0.coeff1_0 + make_float3 (0.48860251903533936f * _S143) * dpsh_0.coeff2_0 - make_float3 (0.48860251903533936f * _S144) * dpsh_0.coeff3_0;
        if(active_sh_6 > 1U)
        {
            float xx_1 = _S144 * _S144;

#line 103
            float yy_1 = _S142 * _S142;

#line 103
            float zz_1 = _S143 * _S143;
            float xy_1 = _S144 * _S142;



            float _S145 = 2.0f * zz_1;

            float _S146 = xx_1 - yy_1;

#line 109
            float3  rgb_8 = rgb_7 + make_float3 (1.09254848957061768f * xy_1) * dpsh_0.coeff4_0 + make_float3 (-1.09254848957061768f * (_S142 * _S143)) * dpsh_0.coeff5_0 + make_float3 (0.31539157032966614f * (_S145 - xx_1 - yy_1)) * dpsh_0.coeff6_0 + make_float3 (-1.09254848957061768f * (_S144 * _S143)) * dpsh_0.coeff7_0 + make_float3 (0.54627424478530884f * _S146) * dpsh_0.coeff8_0;


            if(active_sh_6 > 2U)
            {

                float _S147 = 3.0f * xx_1;

                float _S148 = 4.0f * zz_1 - xx_1 - yy_1;
                float _S149 = 3.0f * yy_1;

#line 118
                rgb_6 = rgb_8 + make_float3 (-0.59004360437393188f * _S142 * (_S147 - yy_1)) * dpsh_0.coeff9_0 + make_float3 (2.89061141014099121f * xy_1 * _S143) * dpsh_0.coeff10_0 + make_float3 (-0.4570457935333252f * _S142 * _S148) * dpsh_0.coeff11_0 + make_float3 (0.37317633628845215f * _S143 * (_S145 - _S147 - _S149)) * dpsh_0.coeff12_0 + make_float3 (-0.4570457935333252f * _S144 * _S148) * dpsh_0.coeff13_0 + make_float3 (1.44530570507049561f * _S143 * _S146) * dpsh_0.coeff14_0 + make_float3 (-0.59004360437393188f * _S144 * (xx_1 - _S149)) * dpsh_0.coeff15_0;

#line 112
            }
            else
            {

#line 112
                rgb_6 = rgb_8;

#line 112
            }

#line 101
        }
        else
        {

#line 101
            rgb_6 = rgb_7;

#line 101
        }

#line 99
    }
    else
    {

#line 99
        rgb_6 = rgb_5;

#line 99
    }

#line 99
    return s_primal_ctx_max_0(rgb_6 + make_float3 (0.5f), make_float3 (0.0f));
}


#line 99
__device__ Matrix<float, 3, 3>  s_primal_ctx_mul_2(Matrix<float, 3, 3>  _S150, Matrix<float, 3, 3>  _S151)
{

#line 99
    return mul_3(_S150, _S151);
}


#line 99
__device__ Matrix<float, 3, 3>  s_primal_ctx_get_covariance_from_quat_scales_0(float4  dpq_0, float3  dps_0)
{

#line 280 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    float _S152 = dpq_0.z;



    float _S153 = _S152 * _S152;

#line 284
    float _S154 = dpq_0.w * dpq_0.w;

#line 284
    float _S155 = dpq_0.y * dpq_0.z;

#line 284
    float _S156 = dpq_0.x * dpq_0.w;

#line 284
    float _S157 = dpq_0.y * dpq_0.w;

#line 284
    float _S158 = dpq_0.x * dpq_0.z;
    float _S159 = dpq_0.y * dpq_0.y;

#line 285
    float _S160 = dpq_0.z * dpq_0.w;

#line 285
    float _S161 = dpq_0.x * dpq_0.y;

#line 285
    Matrix<float, 3, 3>  _S162 = s_primal_ctx_mul_2(makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S153 + _S154), 2.0f * (_S155 - _S156), 2.0f * (_S157 + _S158), 2.0f * (_S155 + _S156), 1.0f - 2.0f * (_S159 + _S154), 2.0f * (_S160 - _S161), 2.0f * (_S157 - _S158), 2.0f * (_S160 + _S161), 1.0f - 2.0f * (_S159 + _S153)), makeMatrix<float, 3, 3> (dps_0.x, 0.0f, 0.0f, 0.0f, dps_0.y, 0.0f, 0.0f, 0.0f, dps_0.z));

#line 285
    return s_primal_ctx_mul_2(_S162, transpose_0(_S162));
}


#line 285
__device__ float s_primal_ctx_tan_0(float _S163)
{

#line 285
    return (F32_tan((_S163)));
}


#line 285
__device__ float s_primal_ctx_max_1(float _S164, float _S165)
{

#line 285
    return (F32_max((_S164), (_S165)));
}


#line 285
__device__ float s_primal_ctx_min_0(float _S166, float _S167)
{

#line 285
    return (F32_min((_S166), (_S167)));
}


#line 285
__device__ Matrix<float, 3, 3>  s_primal_ctx_compute_jacobian_0(float3  dpxyz_ws_0, Camera_0 dpcam_1)
{

#line 127
    float _S168 = s_primal_ctx_tan_0(dpcam_1.fovx_1 / 2.0f);

#line 127
    float _S169 = s_primal_ctx_tan_0(dpcam_1.fovy_1 / 2.0f);


    float h_x_1 = float(dpcam_1.W_0) / (2.0f * _S168);
    float h_y_1 = float(dpcam_1.H_0) / (2.0f * _S169);

#line 131
    float3  _S170 = s_primal_ctx_geom_transform_points_0(dpxyz_ws_0, dpcam_1.world_view_transform_1);

#line 136
    float limx_1 = 1.29999995231628418f * _S168;
    float limy_1 = 1.29999995231628418f * _S169;
    float _S171 = _S170.z;
    float tytz_1 = _S170.y / _S171;
    float _S172 = s_primal_ctx_min_0(limx_1, s_primal_ctx_max_1(- limx_1, _S170.x / _S171)) * _S171;

#line 140
    float3  _S173 = _S170;

#line 140
    *&((&_S173)->x) = _S172;

#line 140
    *&((&_S173)->y) = s_primal_ctx_min_0(limy_1, s_primal_ctx_max_1(- limy_1, tytz_1)) * _S173.z;


    float _S174 = _S173.z;

#line 143
    float _S175 = _S174 * _S174;

#line 143
    return makeMatrix<float, 3, 3> (h_x_1 / _S174, 0.0f, - (h_x_1 * _S173.x) / _S175, 0.0f, h_y_1 / _S174, - (h_y_1 * _S173.y) / _S175, 0.0f, 0.0f, 0.0f);
}


#line 143
__device__ Matrix<float, 2, 2>  s_primal_ctx_covariance_3d_to_2d_0(Camera_0 dpcam_2, float3  dpxyz_ws_1, Matrix<float, 3, 3>  dpcov_ws_0)
{

#line 151
    Matrix<float, 3, 3>  _S176 = makeMatrix<float, 3, 3> (float3 {dpcam_2.world_view_transform_1.rows[int(0)].x, dpcam_2.world_view_transform_1.rows[int(0)].y, dpcam_2.world_view_transform_1.rows[int(0)].z}, float3 {dpcam_2.world_view_transform_1.rows[int(1)].x, dpcam_2.world_view_transform_1.rows[int(1)].y, dpcam_2.world_view_transform_1.rows[int(1)].z}, float3 {dpcam_2.world_view_transform_1.rows[int(2)].x, dpcam_2.world_view_transform_1.rows[int(2)].y, dpcam_2.world_view_transform_1.rows[int(2)].z});

#line 151
    Matrix<float, 3, 3>  _S177 = s_primal_ctx_compute_jacobian_0(dpxyz_ws_1, dpcam_2);

#line 151
    Matrix<float, 3, 3>  _S178 = s_primal_ctx_mul_2(_S177, s_primal_ctx_mul_2(_S176, s_primal_ctx_mul_2(dpcov_ws_0, s_primal_ctx_mul_2(transpose_0(_S176), transpose_0(_S177)))));



    float _S179 = _S178.rows[int(0)].x + 0.30000001192092896f;

#line 155
    Matrix<float, 3, 3>  _S180 = _S178;

#line 155
    *&(((&_S180)->rows + (int(0)))->x) = _S179;

#line 155
    *&(((&_S180)->rows + (int(1)))->y) = _S178.rows[int(1)].y + 0.30000001192092896f;

#line 155
    return makeMatrix<float, 2, 2> (float2 {_S180.rows[int(0)].x, _S180.rows[int(0)].y}, float2 {_S180.rows[int(1)].x, _S180.rows[int(1)].y});
}


#line 155
__device__ Splat_2D_Vertex_0 s_primal_ctx_project_gaussian_to_camera_0(Gaussian_3D_0 dpg_0, Camera_0 dpcam_3, uint active_sh_7)
{

#line 222
    float3  _S181 = s_primal_ctx_project_point_0(dpg_0.xyz_ws_0, dpcam_3);

    bool _S182 = _S181.z <= 0.20000000298023224f;

#line 224
    Splat_2D_Vertex_0 _S183;

#line 224
    if(_S182)
    {

#line 225
        float3  _S184 = make_float3 (0.0f);

#line 225
        Matrix<float, 2, 2>  _S185 = makeMatrix<float, 2, 2> (0.0f);

#line 225
        (&_S183)->xyz_vs_0 = _S184;

#line 225
        (&_S183)->rgb_0 = _S184;

#line 225
        (&_S183)->cov_vs_0 = _S185;

#line 225
    }

#line 225
    bool _S186 = !_S182;

#line 225
    if(_S186)
    {

#line 225
        float3  _S187 = s_primal_ctx_compute_color_from_sh_coeffs_0(dpg_0.sh_coeffs_0, dpg_0.xyz_ws_0, dpcam_3.position_1, active_sh_7);

#line 225
        Matrix<float, 2, 2>  _S188 = s_primal_ctx_covariance_3d_to_2d_0(dpcam_3, dpg_0.xyz_ws_0, s_primal_ctx_get_covariance_from_quat_scales_0(dpg_0.rotations_0, dpg_0.scales_0));

#line 225
        (&_S183)->xyz_vs_0 = _S181;

#line 225
        (&_S183)->rgb_0 = _S187;

#line 225
        (&_S183)->cov_vs_0 = _S188;

#line 225
    }

#line 225
    return _S183;
}


#line 225
__device__ float s_primal_ctx_compute_det_0(Matrix<float, 2, 2>  dpM_0)
{

#line 203
    return dpM_0.rows[int(0)].x * dpM_0.rows[int(1)].y - dpM_0.rows[int(0)].y * dpM_0.rows[int(1)].x;
}


#line 203
__device__ float s_primal_ctx_ndc2pix_0(float dpv_0, int S_1)
{

#line 61
    return ((dpv_0 + 1.0f) * float(S_1) - 1.0f) * 0.5f;
}


#line 61
__device__ void s_primal_ctx_vertex_shader_0(DiffTensorView_0 xyz_ws_5, DiffTensorView_0 sh_coeffs_5, DiffTensorView_0 rotations_3, DiffTensorView_0 scales_3, uint active_sh_8, TensorView world_view_transform_3, TensorView proj_mat_3, TensorView cam_pos_1, TensorView out_tiles_touched_0, TensorView out_rect_tile_space_0, TensorView out_radii_0, DiffTensorView_0 out_xyz_vs_0, DiffTensorView_0 out_inv_cov_vs_0, DiffTensorView_0 out_rgb_0, float fovy_3, float fovx_3, uint image_height_0, uint image_width_0, uint grid_height_1, uint grid_width_1, uint tile_height_1, uint tile_width_1, s_bwd_prop_vertex_shader_Intermediates_0 * _s_diff_ctx_0)
{

#line 58 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
    Matrix<float, 4, 4>  _S189 = makeMatrix<float, 4, 4> (0.0f);

#line 58
    float3  _S190 = make_float3 (0.0f);

#line 58
    Camera_0 _S191 = { _S189, _S189, _S190, 0.0f, 0.0f, int(0), int(0) };

#line 58
    SpherHarmCoeffs_0 _S192 = { _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190, _S190 };

#line 58
    float4  _S193 = make_float4 (0.0f);

#line 58
    Gaussian_3D_0 _S194 = { _S190, _S192, _S193, _S190 };

#line 58
    _s_diff_ctx_0->_S111 = _S191;

#line 58
    _s_diff_ctx_0->_S112 = _S194;

#line 58
    (&_s_diff_ctx_0->_S111)->world_view_transform_1 = _S189;

#line 58
    (&_s_diff_ctx_0->_S111)->proj_mat_1 = _S189;

#line 58
    (&_s_diff_ctx_0->_S111)->position_1 = _S190;

#line 58
    (&_s_diff_ctx_0->_S111)->fovy_1 = 0.0f;

#line 58
    (&_s_diff_ctx_0->_S111)->fovx_1 = 0.0f;

#line 58
    (&_s_diff_ctx_0->_S111)->H_0 = int(0);

#line 58
    (&_s_diff_ctx_0->_S111)->W_0 = int(0);

#line 58
    (&_s_diff_ctx_0->_S112)->xyz_ws_0 = _S190;

#line 58
    (&_s_diff_ctx_0->_S112)->sh_coeffs_0 = _S192;

#line 58
    (&_s_diff_ctx_0->_S112)->rotations_0 = _S193;

#line 58
    (&_s_diff_ctx_0->_S112)->scales_0 = _S190;

    uint g_idx_4 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

#line 60
    bool _S195 = !(g_idx_4 >= DiffTensorView_size_0(xyz_ws_5, 0U));

#line 60
    if(_S195)
    {



        Camera_0 cam_4 = load_camera_0(world_view_transform_3, proj_mat_3, cam_pos_1, fovy_3, fovx_3, image_height_0, image_width_0);

#line 65
        _s_diff_ctx_0->_S111 = cam_4;

#line 65
        Gaussian_3D_0 _S196 = s_primal_ctx_load_gaussian_0(int(g_idx_4), xyz_ws_5, sh_coeffs_5, rotations_3, scales_3, active_sh_8);
        _s_diff_ctx_0->_S112 = _S196;

#line 66
        Splat_2D_Vertex_0 _S197 = s_primal_ctx_project_gaussian_to_camera_0(_S196, cam_4, active_sh_8);

        float _S198 = _S197.xyz_vs_0.z;

#line 68
        bool _bflag_0;

#line 68
        if(_S198 <= 0.20000000298023224f)
        {

#line 68
            _bflag_0 = false;

#line 68
        }
        else
        {

#line 68
            _bflag_0 = _S195;

#line 68
        }

#line 68
        if(_bflag_0)
        {

#line 68
            float _S199 = s_primal_ctx_compute_det_0(_S197.cov_vs_0);

#line 74
            if(_S199 == 0.0f)
            {

#line 74
                _bflag_0 = false;

#line 74
            }

#line 74
            if(_bflag_0)
            {
                float radius_1 = splat_radius_0(_S197.cov_vs_0, _S199);


                float _S200 = _S197.xyz_vs_0.x;

#line 79
                float _S201 = _S197.xyz_vs_0.y;
                rectangle_0 rect_tile_space_1 = get_rectangle_tile_space_0(make_float2 (s_primal_ctx_ndc2pix_0(_S200, int(image_width_0)), s_primal_ctx_ndc2pix_0(_S201, int(image_height_0))), radius_1, grid_height_1, grid_width_1, tile_height_1, tile_width_1);

                int n_tiles_0 = (rect_tile_space_1.max_x_0 - rect_tile_space_1.min_x_0) * (rect_tile_space_1.max_y_0 - rect_tile_space_1.min_y_0);

                if(n_tiles_0 == int(0))
                {

#line 84
                    _bflag_0 = false;

#line 84
                }

#line 84
                if(_bflag_0)
                {


                    Matrix<float, 2, 2>  g_inv_cov_vs_0 = makeMatrix<float, 2, 2> (_S197.cov_vs_0.rows[int(1)].y, - _S197.cov_vs_0.rows[int(0)].y, - _S197.cov_vs_0.rows[int(1)].x, _S197.cov_vs_0.rows[int(0)].x) / makeMatrix<float, 2, 2> (_S199);

                    (out_radii_0).store<int>((g_idx_4), (int(uint(radius_1))));
                    (out_tiles_touched_0).store<int>((g_idx_4), (n_tiles_0));
                    uint2  _S202 = make_uint2 (g_idx_4, 0U);

#line 92
                    (out_rect_tile_space_0).store<int>((g_idx_4), (0U), (rect_tile_space_1.min_x_0));
                    uint2  _S203 = make_uint2 (g_idx_4, 1U);

#line 93
                    (out_rect_tile_space_0).store<int>((g_idx_4), (1U), (rect_tile_space_1.min_y_0));
                    uint2  _S204 = make_uint2 (g_idx_4, 2U);

#line 94
                    (out_rect_tile_space_0).store<int>((g_idx_4), (2U), (rect_tile_space_1.max_x_0));
                    (out_rect_tile_space_0).store<int>((g_idx_4), (3U), (rect_tile_space_1.max_y_0));

                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S202, _S200);
                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S203, _S201);
                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S204, _S198);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 0U, 0U), g_inv_cov_vs_0.rows[int(0)].x);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 0U, 1U), g_inv_cov_vs_0.rows[int(0)].y);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 1U, 0U), g_inv_cov_vs_0.rows[int(1)].x);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 1U, 1U), g_inv_cov_vs_0.rows[int(1)].y);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S202, _S197.rgb_0.x);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S203, _S197.rgb_0.y);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S204, _S197.rgb_0.z);

#line 106
                }

#line 106
            }

#line 106
        }

#line 106
    }

#line 106
    return;
}


#line 61 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ void s_bwd_prop_ndc2pix_0(DiffPair_float_0 * dpv_1, int S_2, float _s_dOut_0)
{
    float _S205 = float(S_2) * (0.5f * _s_dOut_0);

#line 63
    dpv_1->primal_1 = (*dpv_1).primal_1;

#line 63
    dpv_1->differential_0 = _S205;

#line 61
    return;
}


#line 61
struct DiffPair_matrixx3Cfloatx2C2x2C2x3E_0
{
    Matrix<float, 2, 2>  primal_1;
    Matrix<float, 2, 2>  differential_0;
};


#line 203
__device__ void s_bwd_prop_compute_det_0(DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 * dpM_1, float _s_dOut_1)
{

#line 204
    float _S206 = - _s_dOut_1;

#line 204
    float _S207 = (*dpM_1).primal_1.rows[int(0)].y * _S206;

#line 204
    float _S208 = (*dpM_1).primal_1.rows[int(1)].x * _S206;

#line 204
    float _S209 = (*dpM_1).primal_1.rows[int(0)].x * _s_dOut_1;

#line 204
    float _S210 = (*dpM_1).primal_1.rows[int(1)].y * _s_dOut_1;

#line 1751 "core.meta.slang"
    float2  _S211 = make_float2 (0.0f);

#line 1751
    float2  _S212 = _S211;

#line 1751
    *&((&_S212)->x) = _S207;

#line 1751
    *&((&_S212)->y) = _S209;

#line 1751
    float2  _S213 = _S211;

#line 1751
    *&((&_S213)->y) = _S208;

#line 1751
    *&((&_S213)->x) = _S210;

#line 1751
    Matrix<float, 2, 2>  _S214 = makeMatrix<float, 2, 2> (0.0f);

#line 1751
    _S214[int(1)] = _S212;

#line 1751
    _S214[int(0)] = _S213;

#line 1751
    dpM_1->primal_1 = (*dpM_1).primal_1;

#line 1751
    dpM_1->differential_0 = _S214;

#line 203 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    return;
}


#line 203
struct DiffPair_Gaussian_3D_0
{
    Gaussian_3D_0 primal_1;
    Gaussian_3D_0 differential_0;
};


#line 67 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
struct DiffPair_Camera_0
{
    Camera_0 primal_1;
    Camera_Differential_0 differential_0;
};


#line 229 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ void s_bwd_prop_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S215, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S216, Matrix<float, 3, 3>  _S217)
{

#line 229
    mul_1(_S215, _S216, _S217);

#line 229
    return;
}


#line 229
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S218, DiffPair_float_0 * _S219, float _S220)
{

#line 229
    _d_min_0(_S218, _S219, _S220);

#line 229
    return;
}


#line 229
__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S221, DiffPair_float_0 * _S222, float _S223)
{

#line 229
    _d_max_0(_S221, _S222, _S223);

#line 229
    return;
}


#line 228
__device__ void s_bwd_prop_mul_1(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S224, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S225, float4  _S226)
{

#line 228
    _d_mul_0(_S224, _S225, _S226);

#line 228
    return;
}


#line 105
__device__ void s_bwd_prop_geom_transform_points_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_3, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dptransf_matrix_2, float3  _s_dOut_2)
{
    float4  _S227 = make_float4 ((*dppoint_3).primal_1.x, (*dppoint_3).primal_1.y, (*dppoint_3).primal_1.z, 1.0f);

#line 107
    float4  _S228 = s_primal_ctx_mul_1((*dptransf_matrix_2).primal_1, _S227);
    float _S229 = _S228.w + 1.00000001168609742e-07f;

#line 108
    float3  _S230 = _s_dOut_2 / make_float3 (_S229 * _S229);

#line 108
    float3  _S231 = float3 {_S228.x, _S228.y, _S228.z} * - _S230;

#line 108
    float3  _S232 = make_float3 (_S229) * _S230;

#line 107
    float4  _S233 = make_float4 (_S232.x, _S232.y, _S232.z, _S231.x + _S231.y + _S231.z);

#line 107
    Matrix<float, 4, 4>  _S234 = makeMatrix<float, 4, 4> (0.0f);

#line 107
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S235;

#line 107
    (&_S235)->primal_1 = (*dptransf_matrix_2).primal_1;

#line 107
    (&_S235)->differential_0 = _S234;

#line 107
    float4  _S236 = make_float4 (0.0f);

#line 107
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S237;

#line 107
    (&_S237)->primal_1 = _S227;

#line 107
    (&_S237)->differential_0 = _S236;

#line 107
    s_bwd_prop_mul_1(&_S235, &_S237, _S233);

#line 107
    float3  _S238 = float3 {_S237.differential_0.x, _S237.differential_0.y, _S237.differential_0.z};

#line 107
    dptransf_matrix_2->primal_1 = (*dptransf_matrix_2).primal_1;

#line 107
    dptransf_matrix_2->differential_0 = _S235.differential_0;

#line 107
    dppoint_3->primal_1 = (*dppoint_3).primal_1;

#line 107
    dppoint_3->differential_0 = _S238;

#line 105
    return;
}


#line 105
__device__ void s_bwd_prop_tan_0(DiffPair_float_0 * _S239, float _S240)
{

#line 105
    _d_tan_0(_S239, _S240);

#line 105
    return;
}


#line 127
__device__ void s_bwd_prop_compute_jacobian_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpxyz_ws_2, DiffPair_Camera_0 * dpcam_4, Matrix<float, 3, 3>  s_diff_J_T_0)
{

#line 128
    float _S241 = (*dpcam_4).primal_1.fovx_1 / 2.0f;

#line 128
    float _S242 = s_primal_ctx_tan_0(_S241);
    float _S243 = (*dpcam_4).primal_1.fovy_1 / 2.0f;

#line 129
    float _S244 = s_primal_ctx_tan_0(_S243);
    float _S245 = float((*dpcam_4).primal_1.W_0);

#line 130
    float _S246 = 2.0f * _S242;

#line 130
    float h_x_2 = _S245 / _S246;

#line 130
    float _S247 = _S246 * _S246;
    float _S248 = float((*dpcam_4).primal_1.H_0);

#line 131
    float _S249 = 2.0f * _S244;

#line 131
    float h_y_2 = _S248 / _S249;

#line 131
    float _S250 = _S249 * _S249;

#line 131
    float3  _S251 = s_primal_ctx_geom_transform_points_0((*dpxyz_ws_2).primal_1, (*dpcam_4).primal_1.world_view_transform_1);

#line 136
    float limx_2 = 1.29999995231628418f * _S242;
    float limy_2 = 1.29999995231628418f * _S244;
    float _S252 = _S251.x;

#line 138
    float _S253 = _S251.z;

#line 138
    float txtz_0 = _S252 / _S253;

#line 138
    float _S254 = _S253 * _S253;
    float _S255 = _S251.y;

#line 139
    float tytz_2 = _S255 / _S253;
    float _S256 = - limx_2;

#line 140
    float _S257 = s_primal_ctx_max_1(_S256, txtz_0);

#line 140
    float _S258 = s_primal_ctx_min_0(limx_2, _S257);

#line 140
    float _S259 = _S258 * _S253;

#line 140
    float3  _S260 = _S251;

#line 140
    *&((&_S260)->x) = _S259;
    float _S261 = - limy_2;

#line 141
    float _S262 = s_primal_ctx_max_1(_S261, tytz_2);

#line 141
    float _S263 = s_primal_ctx_min_0(limy_2, _S262);

#line 141
    float _S264 = _S260.z;

#line 141
    *&((&_S260)->y) = _S263 * _S264;

    float _S265 = _S260.z;

#line 143
    float _S266 = _S265 * _S265;

#line 143
    float _S267 = _S260.x;

#line 143
    float _S268 = _S266 * _S266;
    float _S269 = _S260.y;

#line 144
    float _S270 = s_diff_J_T_0.rows[int(1)].z / _S268;

#line 144
    float _S271 = - (_S266 * _S270);

#line 144
    float _S272 = h_y_2 * _S271;

#line 144
    float _S273 = _S269 * _S271;

#line 144
    float _S274 = s_diff_J_T_0.rows[int(1)].y / _S266;

#line 144
    float _S275 = _S265 * _S274;

#line 143
    float _S276 = s_diff_J_T_0.rows[int(0)].z / _S268;

#line 143
    float _S277 = _S265 * (- (h_y_2 * _S269) * - _S270 + - (h_x_2 * _S267) * - _S276);

#line 143
    float _S278 = - (_S266 * _S276);

#line 143
    float _S279 = _S267 * _S278;

#line 143
    float _S280 = s_diff_J_T_0.rows[int(0)].x / _S266;

#line 143
    float _S281 = _S265 * _S280;

#line 143
    _S260 = make_float3 (h_x_2 * _S278, _S272, h_y_2 * - _S274 + _S277 + _S277 + h_x_2 * - _S280);

#line 143
    *&((&_S260)->y) = 0.0f;

#line 141
    float _S282 = _S263 * _S272;

#line 141
    float _S283 = _S264 * _S272;

#line 141
    DiffPair_float_0 _S284;

#line 141
    (&_S284)->primal_1 = limy_2;

#line 141
    (&_S284)->differential_0 = 0.0f;

#line 141
    DiffPair_float_0 _S285;

#line 141
    (&_S285)->primal_1 = _S262;

#line 141
    (&_S285)->differential_0 = 0.0f;

#line 141
    s_bwd_prop_min_0(&_S284, &_S285, _S283);

#line 141
    DiffPair_float_0 _S286;

#line 141
    (&_S286)->primal_1 = _S261;

#line 141
    (&_S286)->differential_0 = 0.0f;

#line 141
    DiffPair_float_0 _S287;

#line 141
    (&_S287)->primal_1 = tytz_2;

#line 141
    (&_S287)->differential_0 = 0.0f;

#line 141
    s_bwd_prop_max_0(&_S286, &_S287, _S285.differential_0);

#line 141
    float _S288 = - _S286.differential_0;

#line 140
    float3  _S289 = _S260 + make_float3 (0.0f, 0.0f, _S282);

#line 140
    _S260 = _S289;

#line 140
    *&((&_S260)->x) = 0.0f;

#line 140
    float _S290 = _S258 * _S289.x;

#line 140
    float _S291 = _S253 * _S289.x;

#line 140
    DiffPair_float_0 _S292;

#line 140
    (&_S292)->primal_1 = limx_2;

#line 140
    (&_S292)->differential_0 = 0.0f;

#line 140
    DiffPair_float_0 _S293;

#line 140
    (&_S293)->primal_1 = _S257;

#line 140
    (&_S293)->differential_0 = 0.0f;

#line 140
    s_bwd_prop_min_0(&_S292, &_S293, _S291);

#line 140
    DiffPair_float_0 _S294;

#line 140
    (&_S294)->primal_1 = _S256;

#line 140
    (&_S294)->differential_0 = 0.0f;

#line 140
    DiffPair_float_0 _S295;

#line 140
    (&_S295)->primal_1 = txtz_0;

#line 140
    (&_S295)->differential_0 = 0.0f;

#line 140
    s_bwd_prop_max_0(&_S294, &_S295, _S293.differential_0);

#line 139
    float _S296 = _S287.differential_0 / _S254;

#line 138
    float _S297 = _S295.differential_0 / _S254;

#line 137
    float _S298 = 1.29999995231628418f * (_S284.differential_0 + _S288);

#line 136
    float _S299 = 1.29999995231628418f * (_S292.differential_0 + - _S294.differential_0);

#line 133
    float3  _S300 = _S260 + make_float3 (_S253 * _S297, _S253 * _S296, _S290 + _S255 * - _S296 + _S252 * - _S297);

#line 133
    float3  _S301 = make_float3 (0.0f);

#line 133
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S302;

#line 133
    (&_S302)->primal_1 = (*dpxyz_ws_2).primal_1;

#line 133
    (&_S302)->differential_0 = _S301;

#line 133
    Matrix<float, 4, 4>  _S303 = makeMatrix<float, 4, 4> (0.0f);

#line 133
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S304;

#line 133
    (&_S304)->primal_1 = (*dpcam_4).primal_1.world_view_transform_1;

#line 133
    (&_S304)->differential_0 = _S303;

#line 133
    s_bwd_prop_geom_transform_points_0(&_S302, &_S304, _S300);

#line 130
    float _S305 = 2.0f * (_S245 * - ((_S279 + _S281) / _S247));

#line 129
    float _S306 = _S298 + 2.0f * (_S248 * - ((_S273 + _S275) / _S250));

#line 129
    DiffPair_float_0 _S307;

#line 129
    (&_S307)->primal_1 = _S243;

#line 129
    (&_S307)->differential_0 = 0.0f;

#line 129
    s_bwd_prop_tan_0(&_S307, _S306);

#line 129
    float _S308 = 0.5f * _S307.differential_0;

#line 128
    float _S309 = _S299 + _S305;

#line 128
    DiffPair_float_0 _S310;

#line 128
    (&_S310)->primal_1 = _S241;

#line 128
    (&_S310)->differential_0 = 0.0f;

#line 128
    s_bwd_prop_tan_0(&_S310, _S309);

#line 128
    float _S311 = 0.5f * _S310.differential_0;

#line 128
    Camera_Differential_0 _S312 = Camera_x24_syn_dzero_0();

#line 128
    (&_S312)->world_view_transform_0 = _S304.differential_0;

#line 128
    (&_S312)->fovy_0 = _S308;

#line 128
    (&_S312)->fovx_0 = _S311;

#line 128
    dpcam_4->primal_1 = (*dpcam_4).primal_1;

#line 128
    dpcam_4->differential_0 = _S312;

#line 128
    dpxyz_ws_2->primal_1 = (*dpxyz_ws_2).primal_1;

#line 128
    dpxyz_ws_2->differential_0 = _S302.differential_0;

#line 127
    return;
}


#line 151
__device__ void s_bwd_prop_covariance_3d_to_2d_0(DiffPair_Camera_0 * dpcam_5, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpxyz_ws_3, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * dpcov_ws_1, Matrix<float, 2, 2>  _s_dOut_3)
{

#line 151
    Matrix<float, 3, 3>  _S313 = makeMatrix<float, 3, 3> (float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].z}, float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].z}, float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].z});

#line 151
    Matrix<float, 3, 3>  _S314 = s_primal_ctx_compute_jacobian_0((*dpxyz_ws_3).primal_1, (*dpcam_5).primal_1);


    Matrix<float, 3, 3>  _S315 = transpose_0(_S313);

#line 154
    Matrix<float, 3, 3>  _S316 = transpose_0(_S314);

#line 154
    Matrix<float, 3, 3>  _S317 = s_primal_ctx_mul_2(_S315, _S316);

#line 154
    Matrix<float, 3, 3>  _S318 = s_primal_ctx_mul_2((*dpcov_ws_1).primal_1, _S317);

#line 154
    Matrix<float, 3, 3>  _S319 = s_primal_ctx_mul_2(_S313, _S318);

#line 154
    float3  _S320 = make_float3 (_s_dOut_3.rows[int(1)].x, _s_dOut_3.rows[int(1)].y, 0.0f);

#line 154
    float3  _S321 = make_float3 (_s_dOut_3.rows[int(0)].x, _s_dOut_3.rows[int(0)].y, 0.0f);

    Matrix<float, 3, 3>  _S322 = makeMatrix<float, 3, 3> (0.0f);

#line 156
    Matrix<float, 3, 3>  _S323 = _S322;

#line 156
    _S323[int(1)] = _S320;

#line 156
    _S323[int(0)] = _S321;

#line 156
    Matrix<float, 3, 3>  _S324 = _S323;

#line 156
    *&(((&_S324)->rows + (int(1)))->y) = 0.0f;

#line 1751 "core.meta.slang"
    float3  _S325 = make_float3 (0.0f);

#line 1751
    float3  _S326 = _S325;

#line 1751
    *&((&_S326)->y) = _S323.rows[int(1)].y;

#line 1751
    *&(((&_S324)->rows + (int(0)))->x) = 0.0f;

#line 155 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    float3  _S327 = _S325;

#line 155
    *&((&_S327)->x) = _S323.rows[int(0)].x;

#line 154
    Matrix<float, 3, 3>  _S328 = _S322;

#line 154
    _S328[int(1)] = _S326;

#line 154
    _S328[int(0)] = _S327;

#line 154
    Matrix<float, 3, 3>  _S329 = _S324 + _S328;

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S330;

#line 154
    (&_S330)->primal_1 = _S314;

#line 154
    (&_S330)->differential_0 = _S322;

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S331;

#line 154
    (&_S331)->primal_1 = _S319;

#line 154
    (&_S331)->differential_0 = _S322;

#line 154
    s_bwd_prop_mul_0(&_S330, &_S331, _S329);

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S332;

#line 154
    (&_S332)->primal_1 = _S313;

#line 154
    (&_S332)->differential_0 = _S322;

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S333;

#line 154
    (&_S333)->primal_1 = _S318;

#line 154
    (&_S333)->differential_0 = _S322;

#line 154
    s_bwd_prop_mul_0(&_S332, &_S333, _S331.differential_0);

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S334;

#line 154
    (&_S334)->primal_1 = (*dpcov_ws_1).primal_1;

#line 154
    (&_S334)->differential_0 = _S322;

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S335;

#line 154
    (&_S335)->primal_1 = _S317;

#line 154
    (&_S335)->differential_0 = _S322;

#line 154
    s_bwd_prop_mul_0(&_S334, &_S335, _S333.differential_0);

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S336;

#line 154
    (&_S336)->primal_1 = _S315;

#line 154
    (&_S336)->differential_0 = _S322;

#line 154
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S337;

#line 154
    (&_S337)->primal_1 = _S316;

#line 154
    (&_S337)->differential_0 = _S322;

#line 154
    s_bwd_prop_mul_0(&_S336, &_S337, _S335.differential_0);

#line 154
    Matrix<float, 3, 3>  _S338 = transpose_0(_S336.differential_0);

#line 153
    Matrix<float, 3, 3>  _S339 = _S330.differential_0 + transpose_0(_S337.differential_0);

#line 153
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S340;

#line 153
    (&_S340)->primal_1 = (*dpxyz_ws_3).primal_1;

#line 153
    (&_S340)->differential_0 = _S325;

#line 153
    Camera_Differential_0 _S341 = Camera_x24_syn_dzero_0();

#line 153
    DiffPair_Camera_0 _S342;

#line 153
    (&_S342)->primal_1 = (*dpcam_5).primal_1;

#line 153
    (&_S342)->differential_0 = _S341;

#line 153
    s_bwd_prop_compute_jacobian_0(&_S340, &_S342, _S339);

#line 153
    Matrix<float, 3, 3>  _S343 = _S332.differential_0 + _S338;

#line 153
    float4  _S344 = make_float4 (_S343.rows[int(2)].x, _S343.rows[int(2)].y, _S343.rows[int(2)].z, 0.0f);

#line 153
    float4  _S345 = make_float4 (_S343.rows[int(1)].x, _S343.rows[int(1)].y, _S343.rows[int(1)].z, 0.0f);

#line 153
    float4  _S346 = make_float4 (_S343.rows[int(0)].x, _S343.rows[int(0)].y, _S343.rows[int(0)].z, 0.0f);

#line 153
    Matrix<float, 4, 4>  _S347 = makeMatrix<float, 4, 4> (0.0f);

#line 153
    _S347[int(2)] = _S344;

#line 153
    _S347[int(1)] = _S345;

#line 153
    _S347[int(0)] = _S346;

#line 153
    dpcov_ws_1->primal_1 = (*dpcov_ws_1).primal_1;

#line 153
    dpcov_ws_1->differential_0 = _S334.differential_0;

#line 153
    dpxyz_ws_3->primal_1 = (*dpxyz_ws_3).primal_1;

#line 153
    dpxyz_ws_3->differential_0 = _S340.differential_0;

#line 153
    Camera_Differential_0 _S348 = _S341;

#line 153
    (&_S348)->world_view_transform_0 = _S347;

#line 153
    Camera_Differential_0 _S349 = Camera_x24_syn_dadd_0(_S342.differential_0, _S348);

#line 153
    dpcam_5->primal_1 = (*dpcam_5).primal_1;

#line 153
    dpcam_5->differential_0 = _S349;

#line 151
    return;
}


#line 280
__device__ void s_bwd_prop_get_covariance_from_quat_scales_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dpq_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dps_1, Matrix<float, 3, 3>  _s_dOut_4)
{

#line 280
    float _S350 = (*dpq_1).primal_1.z;



    float _S351 = _S350 * _S350;

#line 284
    float _S352 = (*dpq_1).primal_1.w * (*dpq_1).primal_1.w;

#line 284
    float _S353 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.z;

#line 284
    float _S354 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.w;

#line 284
    float _S355 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.w;

#line 284
    float _S356 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.z;
    float _S357 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.y;

#line 285
    float _S358 = (*dpq_1).primal_1.z * (*dpq_1).primal_1.w;

#line 285
    float _S359 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.y;

#line 283
    Matrix<float, 3, 3>  rotation_matrix_0 = makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S351 + _S352), 2.0f * (_S353 - _S354), 2.0f * (_S355 + _S356), 2.0f * (_S353 + _S354), 1.0f - 2.0f * (_S357 + _S352), 2.0f * (_S358 - _S359), 2.0f * (_S355 - _S356), 2.0f * (_S358 + _S359), 1.0f - 2.0f * (_S357 + _S351));

#line 288
    Matrix<float, 3, 3>  scales_matrix_0 = makeMatrix<float, 3, 3> ((*dps_1).primal_1.x, 0.0f, 0.0f, 0.0f, (*dps_1).primal_1.y, 0.0f, 0.0f, 0.0f, (*dps_1).primal_1.z);

#line 288
    Matrix<float, 3, 3>  _S360 = s_primal_ctx_mul_2(rotation_matrix_0, scales_matrix_0);

#line 294
    Matrix<float, 3, 3>  _S361 = transpose_0(_S360);

#line 294
    Matrix<float, 3, 3>  _S362 = makeMatrix<float, 3, 3> (0.0f);

#line 294
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S363;

#line 294
    (&_S363)->primal_1 = _S360;

#line 294
    (&_S363)->differential_0 = _S362;

#line 294
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S364;

#line 294
    (&_S364)->primal_1 = _S361;

#line 294
    (&_S364)->differential_0 = _S362;

#line 294
    s_bwd_prop_mul_0(&_S363, &_S364, _s_dOut_4);

#line 292
    Matrix<float, 3, 3>  _S365 = _S363.differential_0 + transpose_0(_S364.differential_0);

#line 292
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S366;

#line 292
    (&_S366)->primal_1 = rotation_matrix_0;

#line 292
    (&_S366)->differential_0 = _S362;

#line 292
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S367;

#line 292
    (&_S367)->primal_1 = scales_matrix_0;

#line 292
    (&_S367)->differential_0 = _S362;

#line 292
    s_bwd_prop_mul_0(&_S366, &_S367, _S365);

#line 286
    float _S368 = 2.0f * - _S366.differential_0.rows[int(2)].z;

#line 286
    float _S369 = 2.0f * _S366.differential_0.rows[int(2)].y;

#line 286
    float _S370 = 2.0f * _S366.differential_0.rows[int(2)].x;

#line 285
    float _S371 = 2.0f * _S366.differential_0.rows[int(1)].z;

#line 285
    float _S372 = _S369 + - _S371;

#line 285
    float _S373 = _S369 + _S371;

#line 285
    float _S374 = 2.0f * - _S366.differential_0.rows[int(1)].y;

#line 285
    float _S375 = (*dpq_1).primal_1.y * (_S368 + _S374);

#line 285
    float _S376 = 2.0f * _S366.differential_0.rows[int(1)].x;

#line 284
    float _S377 = 2.0f * _S366.differential_0.rows[int(0)].z;

#line 284
    float _S378 = - _S370 + _S377;

#line 284
    float _S379 = _S370 + _S377;

#line 284
    float _S380 = 2.0f * _S366.differential_0.rows[int(0)].y;

#line 284
    float _S381 = _S376 + - _S380;

#line 284
    float _S382 = _S376 + _S380;

#line 284
    float _S383 = 2.0f * - _S366.differential_0.rows[int(0)].x;

#line 284
    float _S384 = (*dpq_1).primal_1.w * (_S374 + _S383);

#line 284
    float _S385 = (*dpq_1).primal_1.z * (_S368 + _S383);

#line 958 "core.meta.slang"
    float _S386 = (*dpq_1).primal_1.z * _S373 + (*dpq_1).primal_1.y * _S379 + (*dpq_1).primal_1.x * _S381 + _S384 + _S384;

#line 958
    float _S387 = (*dpq_1).primal_1.w * _S373 + (*dpq_1).primal_1.x * _S378 + (*dpq_1).primal_1.y * _S382 + _S385 + _S385;

#line 958
    float _S388 = (*dpq_1).primal_1.x * _S372 + _S375 + _S375 + (*dpq_1).primal_1.w * _S379 + (*dpq_1).primal_1.z * _S382;

#line 958
    float _S389 = (*dpq_1).primal_1.y * _S372 + (*dpq_1).primal_1.z * _S378 + (*dpq_1).primal_1.w * _S381;

#line 958
    float3  _S390 = make_float3 (0.0f);

#line 958
    *&((&_S390)->z) = _S367.differential_0.rows[int(2)].z;

#line 958
    *&((&_S390)->y) = _S367.differential_0.rows[int(1)].y;

#line 958
    *&((&_S390)->x) = _S367.differential_0.rows[int(0)].x;

#line 958
    dps_1->primal_1 = (*dps_1).primal_1;

#line 958
    dps_1->differential_0 = _S390;

#line 958
    float4  _S391 = make_float4 (0.0f);

#line 958
    *&((&_S391)->w) = _S386;

#line 958
    *&((&_S391)->z) = _S387;

#line 958
    *&((&_S391)->y) = _S388;

#line 958
    *&((&_S391)->x) = _S389;

#line 958
    dpq_1->primal_1 = (*dpq_1).primal_1;

#line 958
    dpq_1->differential_0 = _S391;

#line 280 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    return;
}


#line 280
struct DiffPair_SpherHarmCoeffs_0
{
    SpherHarmCoeffs_0 primal_1;
    SpherHarmCoeffs_0 differential_0;
};


#line 227
__device__ void s_bwd_prop_max_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S392, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S393, float3  _S394)
{

#line 227
    _d_max_vector_0(_S392, _S393, _S394);

#line 227
    return;
}


#line 2117 "diff.meta.slang"
__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S395, float _S396)
{

#line 2117
    _d_sqrt_0(_S395, _S396);

#line 2117
    return;
}


#line 2092
__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_10, float _s_dOut_5)
{

#line 2092
    float _S397 = (*dpx_10).primal_1.x;

#line 2092
    float _S398 = (*dpx_10).primal_1.y;

#line 2092
    float _S399 = (*dpx_10).primal_1.z;

#line 2099
    DiffPair_float_0 _S400;

#line 2099
    (&_S400)->primal_1 = _S397 * _S397 + _S398 * _S398 + _S399 * _S399;

#line 2099
    (&_S400)->differential_0 = 0.0f;

#line 2099
    s_bwd_prop_sqrt_0(&_S400, _s_dOut_5);

#line 2099
    float _S401 = (*dpx_10).primal_1.z * _S400.differential_0;

#line 958 "core.meta.slang"
    float _S402 = _S401 + _S401;

#line 958
    float _S403 = (*dpx_10).primal_1.y * _S400.differential_0;

#line 958
    float _S404 = _S403 + _S403;

#line 958
    float _S405 = (*dpx_10).primal_1.x * _S400.differential_0;

#line 958
    float _S406 = _S405 + _S405;

#line 958
    float3  _S407 = make_float3 (0.0f);

#line 958
    *&((&_S407)->z) = _S402;

#line 958
    *&((&_S407)->y) = _S404;

#line 958
    *&((&_S407)->x) = _S406;

#line 958
    dpx_10->primal_1 = (*dpx_10).primal_1;

#line 958
    dpx_10->differential_0 = _S407;

#line 2092 "diff.meta.slang"
    return;
}


#line 2092
__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S408, float _S409)
{

#line 2092
    s_bwd_prop_length_impl_0(_S408, _S409);

#line 2092
    return;
}


#line 2154
__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_11, float3  _s_dOut_6)
{
    float _S410 = length_0((*dpx_11).primal_1);
    float3  _S411 = (*dpx_11).primal_1 * _s_dOut_6;

#line 2157
    float3  _S412 = make_float3 (1.0f / _S410) * _s_dOut_6;

#line 2157
    float _S413 = - ((_S411.x + _S411.y + _S411.z) / (_S410 * _S410));

#line 2156
    float3  _S414 = make_float3 (0.0f);

#line 2156
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S415;

#line 2156
    (&_S415)->primal_1 = (*dpx_11).primal_1;

#line 2156
    (&_S415)->differential_0 = _S414;

#line 2156
    s_bwd_length_impl_0(&_S415, _S413);

#line 2156
    float3  _S416 = _S412 + _S415.differential_0;

#line 2156
    dpx_11->primal_1 = (*dpx_11).primal_1;

#line 2156
    dpx_11->differential_0 = _S416;

#line 2154
    return;
}


#line 2154
__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S417, float3  _S418)
{

#line 2154
    s_bwd_prop_normalize_impl_0(_S417, _S418);

#line 2154
    return;
}


#line 94 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
__device__ void s_bwd_prop_compute_color_from_sh_coeffs_0(DiffPair_SpherHarmCoeffs_0 * dpsh_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpg_xyz_ws_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcam_pos_1, uint active_sh_9, float3  _s_dOut_7)
{

#line 94
    DiffPair_SpherHarmCoeffs_0 _S419 = *dpsh_1;

#line 100
    float3  _S420 = make_float3 (0.0f);

#line 95
    float3  dir_1 = (*dpg_xyz_ws_1).primal_1 - (*dpcam_pos_1).primal_1;
    float3  _S421 = normalize_0(dir_1);

    float3  rgb_9 = make_float3 (0.282094806432724f) * (*dpsh_1).primal_1.coeff0_0;
    bool _S422 = active_sh_9 > 0U;

#line 99
    float3  rgb_10;

#line 99
    float3  _S423;

#line 99
    float3  _S424;

#line 99
    float3  _S425;

#line 99
    float3  _S426;

#line 99
    float3  _S427;

#line 99
    float3  _S428;

#line 99
    float3  _S429;

#line 99
    float3  _S430;

#line 99
    float3  _S431;

#line 99
    float3  _S432;

#line 99
    float3  _S433;

#line 99
    float3  _S434;

#line 99
    float3  _S435;

#line 99
    float3  _S436;

#line 99
    float3  _S437;

#line 99
    float3  _S438;

#line 99
    float3  _S439;

#line 99
    float3  _S440;

#line 99
    float3  _S441;

#line 99
    float3  _S442;

#line 99
    float3  _S443;

#line 99
    float3  _S444;

#line 99
    float3  _S445;

#line 99
    float3  _S446;

#line 99
    float3  _S447;

#line 99
    float3  _S448;

#line 99
    float3  _S449;

#line 99
    float3  _S450;

#line 99
    float3  _S451;

#line 99
    float3  _S452;

#line 99
    float _S453;

#line 99
    float _S454;

#line 99
    float _S455;

#line 99
    float _S456;

#line 99
    float _S457;

#line 99
    float _S458;

#line 99
    float _S459;

#line 99
    float _S460;

#line 99
    float _S461;

#line 99
    float _S462;

#line 99
    float _S463;

#line 99
    float _S464;

#line 99
    float _S465;

#line 99
    float _S466;

#line 99
    float _S467;

#line 99
    bool _S468;

#line 99
    bool _S469;

#line 99
    if(_S422)
    {

#line 100
        float _S470 = _S421.y;

#line 100
        float _S471 = 0.48860251903533936f * _S470;

#line 100
        float3  _S472 = make_float3 (_S471);

#line 100
        float _S473 = _S421.z;

#line 100
        float _S474 = 0.48860251903533936f * _S473;

#line 100
        float3  _S475 = make_float3 (_S474);

#line 100
        float _S476 = _S421.x;

#line 100
        float _S477 = 0.48860251903533936f * _S476;

#line 100
        float3  _S478 = make_float3 (_S477);

#line 100
        float3  rgb_11 = rgb_9 - make_float3 (_S471) * _S419.primal_1.coeff1_0 + make_float3 (_S474) * _S419.primal_1.coeff2_0 - make_float3 (_S477) * _S419.primal_1.coeff3_0;
        bool _S479 = active_sh_9 > 1U;

#line 101
        if(_S479)
        {
            float xx_2 = _S476 * _S476;

#line 103
            float yy_2 = _S470 * _S470;

#line 103
            float zz_2 = _S473 * _S473;
            float xy_2 = _S476 * _S470;

            float _S480 = 1.09254848957061768f * xy_2;

#line 106
            float3  _S481 = make_float3 (_S480);
            float _S482 = -1.09254848957061768f * (_S470 * _S473);

#line 107
            float3  _S483 = make_float3 (_S482);
            float _S484 = 2.0f * zz_2;

#line 108
            float _S485 = 0.31539157032966614f * (_S484 - xx_2 - yy_2);

#line 108
            float3  _S486 = make_float3 (_S485);
            float _S487 = -1.09254848957061768f * (_S476 * _S473);

#line 109
            float3  _S488 = make_float3 (_S487);
            float _S489 = xx_2 - yy_2;

#line 110
            float _S490 = 0.54627424478530884f * _S489;

#line 110
            float3  _S491 = make_float3 (_S490);

#line 109
            float3  rgb_12 = rgb_11 + make_float3 (_S480) * _S419.primal_1.coeff4_0 + make_float3 (_S482) * _S419.primal_1.coeff5_0 + make_float3 (_S485) * _S419.primal_1.coeff6_0 + make_float3 (_S487) * _S419.primal_1.coeff7_0 + make_float3 (_S490) * _S419.primal_1.coeff8_0;


            bool _S492 = active_sh_9 > 2U;

#line 112
            if(_S492)
            {

                float _S493 = -0.59004360437393188f * _S470;

#line 115
                float _S494 = 3.0f * xx_2;

#line 115
                float _S495 = _S494 - yy_2;

#line 115
                float _S496 = _S493 * _S495;

#line 115
                float3  _S497 = make_float3 (_S496);
                float _S498 = 2.89061141014099121f * xy_2;

#line 116
                float _S499 = _S498 * _S473;

#line 116
                float3  _S500 = make_float3 (_S499);
                float _S501 = -0.4570457935333252f * _S470;

#line 117
                float _S502 = 4.0f * zz_2 - xx_2 - yy_2;

#line 117
                float _S503 = _S501 * _S502;

#line 117
                float3  _S504 = make_float3 (_S503);
                float _S505 = 0.37317633628845215f * _S473;

#line 118
                float _S506 = 3.0f * yy_2;

#line 118
                float _S507 = _S484 - _S494 - _S506;

#line 118
                float _S508 = _S505 * _S507;

#line 118
                float3  _S509 = make_float3 (_S508);
                float _S510 = -0.4570457935333252f * _S476;

#line 119
                float _S511 = _S510 * _S502;

#line 119
                float3  _S512 = make_float3 (_S511);
                float _S513 = 1.44530570507049561f * _S473;

#line 120
                float _S514 = _S513 * _S489;

#line 120
                float3  _S515 = make_float3 (_S514);
                float _S516 = -0.59004360437393188f * _S476;

#line 121
                float _S517 = xx_2 - _S506;

#line 121
                float _S518 = _S516 * _S517;

#line 121
                float3  _S519 = make_float3 (_S518);

#line 121
                rgb_10 = rgb_12 + make_float3 (_S496) * _S419.primal_1.coeff9_0 + make_float3 (_S499) * _S419.primal_1.coeff10_0 + make_float3 (_S503) * _S419.primal_1.coeff11_0 + make_float3 (_S508) * _S419.primal_1.coeff12_0 + make_float3 (_S511) * _S419.primal_1.coeff13_0 + make_float3 (_S514) * _S419.primal_1.coeff14_0 + make_float3 (_S518) * _S419.primal_1.coeff15_0;

#line 121
                _S423 = _S519;

#line 121
                _S424 = _S419.primal_1.coeff15_0;

#line 121
                _S453 = _S516;

#line 121
                _S454 = _S517;

#line 121
                _S425 = _S515;

#line 121
                _S426 = _S419.primal_1.coeff14_0;

#line 121
                _S455 = _S513;

#line 121
                _S427 = _S512;

#line 121
                _S428 = _S419.primal_1.coeff13_0;

#line 121
                _S456 = _S510;

#line 121
                _S457 = _S502;

#line 121
                _S429 = _S509;

#line 121
                _S430 = _S419.primal_1.coeff12_0;

#line 121
                _S458 = _S505;

#line 121
                _S459 = _S507;

#line 121
                _S431 = _S504;

#line 121
                _S432 = _S419.primal_1.coeff11_0;

#line 121
                _S460 = _S501;

#line 121
                _S433 = _S500;

#line 121
                _S434 = _S419.primal_1.coeff10_0;

#line 121
                _S461 = _S498;

#line 121
                _S435 = _S497;

#line 121
                _S436 = _S419.primal_1.coeff9_0;

#line 121
                _S462 = _S493;

#line 121
                _S463 = _S495;

#line 121
            }
            else
            {

#line 121
                rgb_10 = rgb_12;

#line 121
                _S423 = _S420;

#line 121
                _S424 = _S420;

#line 121
                _S453 = 0.0f;

#line 121
                _S454 = 0.0f;

#line 121
                _S425 = _S420;

#line 121
                _S426 = _S420;

#line 121
                _S455 = 0.0f;

#line 121
                _S427 = _S420;

#line 121
                _S428 = _S420;

#line 121
                _S456 = 0.0f;

#line 121
                _S457 = 0.0f;

#line 121
                _S429 = _S420;

#line 121
                _S430 = _S420;

#line 121
                _S458 = 0.0f;

#line 121
                _S459 = 0.0f;

#line 121
                _S431 = _S420;

#line 121
                _S432 = _S420;

#line 121
                _S460 = 0.0f;

#line 121
                _S433 = _S420;

#line 121
                _S434 = _S420;

#line 121
                _S461 = 0.0f;

#line 121
                _S435 = _S420;

#line 121
                _S436 = _S420;

#line 121
                _S462 = 0.0f;

#line 121
                _S463 = 0.0f;

#line 121
            }

#line 119
            float _S520 = _S456;

#line 117
            float _S521 = _S457;
            float _S522 = _S458;

#line 118
            float _S523 = _S459;

#line 117
            float _S524 = _S460;

#line 116
            float _S525 = _S461;

#line 115
            float _S526 = _S462;

#line 115
            float _S527 = _S463;

#line 115
            _S468 = _S492;

#line 115
            _S456 = _S489;

#line 115
            _S457 = _S520;

#line 115
            _S458 = _S521;

#line 115
            _S459 = _S522;

#line 115
            _S460 = _S523;

#line 115
            _S461 = _S524;

#line 115
            _S462 = _S525;

#line 115
            _S463 = _S526;

#line 115
            _S464 = _S527;

#line 115
            _S437 = _S491;

#line 115
            _S438 = _S419.primal_1.coeff8_0;

#line 115
            _S439 = _S488;

#line 115
            _S440 = _S419.primal_1.coeff7_0;

#line 115
            _S441 = _S486;

#line 115
            _S442 = _S419.primal_1.coeff6_0;

#line 115
            _S443 = _S483;

#line 115
            _S444 = _S419.primal_1.coeff5_0;

#line 115
            _S445 = _S481;

#line 115
            _S446 = _S419.primal_1.coeff4_0;

#line 115
        }
        else
        {

#line 115
            rgb_10 = rgb_11;

#line 115
            _S468 = false;

#line 115
            _S423 = _S420;

#line 115
            _S424 = _S420;

#line 115
            _S453 = 0.0f;

#line 115
            _S454 = 0.0f;

#line 115
            _S425 = _S420;

#line 115
            _S426 = _S420;

#line 115
            _S455 = 0.0f;

#line 115
            _S456 = 0.0f;

#line 115
            _S427 = _S420;

#line 115
            _S428 = _S420;

#line 115
            _S457 = 0.0f;

#line 115
            _S458 = 0.0f;

#line 115
            _S429 = _S420;

#line 115
            _S430 = _S420;

#line 115
            _S459 = 0.0f;

#line 115
            _S460 = 0.0f;

#line 115
            _S431 = _S420;

#line 115
            _S432 = _S420;

#line 115
            _S461 = 0.0f;

#line 115
            _S433 = _S420;

#line 115
            _S434 = _S420;

#line 115
            _S462 = 0.0f;

#line 115
            _S435 = _S420;

#line 115
            _S436 = _S420;

#line 115
            _S463 = 0.0f;

#line 115
            _S464 = 0.0f;

#line 115
            _S437 = _S420;

#line 115
            _S438 = _S420;

#line 115
            _S439 = _S420;

#line 115
            _S440 = _S420;

#line 115
            _S441 = _S420;

#line 115
            _S442 = _S420;

#line 115
            _S443 = _S420;

#line 115
            _S444 = _S420;

#line 115
            _S445 = _S420;

#line 115
            _S446 = _S420;

#line 115
        }

#line 112
        bool _S528 = _S468;


        float _S529 = _S463;

#line 115
        float _S530 = _S464;

#line 115
        _S468 = _S479;

#line 115
        _S469 = _S528;

#line 115
        _S463 = _S473;

#line 115
        _S464 = _S529;

#line 115
        _S465 = _S530;

#line 115
        _S466 = _S476;

#line 115
        _S467 = _S470;

#line 115
        _S447 = _S478;

#line 115
        _S448 = _S419.primal_1.coeff3_0;

#line 115
        _S449 = _S475;

#line 115
        _S450 = _S419.primal_1.coeff2_0;

#line 115
        _S451 = _S472;

#line 115
        _S452 = _S419.primal_1.coeff1_0;

#line 115
    }
    else
    {

#line 115
        rgb_10 = rgb_9;

#line 115
        _S468 = false;

#line 115
        _S469 = false;

#line 115
        _S423 = _S420;

#line 115
        _S424 = _S420;

#line 115
        _S453 = 0.0f;

#line 115
        _S454 = 0.0f;

#line 115
        _S425 = _S420;

#line 115
        _S426 = _S420;

#line 115
        _S455 = 0.0f;

#line 115
        _S456 = 0.0f;

#line 115
        _S427 = _S420;

#line 115
        _S428 = _S420;

#line 115
        _S457 = 0.0f;

#line 115
        _S458 = 0.0f;

#line 115
        _S429 = _S420;

#line 115
        _S430 = _S420;

#line 115
        _S459 = 0.0f;

#line 115
        _S460 = 0.0f;

#line 115
        _S431 = _S420;

#line 115
        _S432 = _S420;

#line 115
        _S461 = 0.0f;

#line 115
        _S433 = _S420;

#line 115
        _S434 = _S420;

#line 115
        _S462 = 0.0f;

#line 115
        _S463 = 0.0f;

#line 115
        _S435 = _S420;

#line 115
        _S436 = _S420;

#line 115
        _S464 = 0.0f;

#line 115
        _S465 = 0.0f;

#line 115
        _S437 = _S420;

#line 115
        _S438 = _S420;

#line 115
        _S439 = _S420;

#line 115
        _S440 = _S420;

#line 115
        _S441 = _S420;

#line 115
        _S442 = _S420;

#line 115
        _S443 = _S420;

#line 115
        _S444 = _S420;

#line 115
        _S445 = _S420;

#line 115
        _S446 = _S420;

#line 115
        _S466 = 0.0f;

#line 115
        _S467 = 0.0f;

#line 115
        _S447 = _S420;

#line 115
        _S448 = _S420;

#line 115
        _S449 = _S420;

#line 115
        _S450 = _S420;

#line 115
        _S451 = _S420;

#line 115
        _S452 = _S420;

#line 115
    }

#line 126
    float3  rgb_13 = rgb_10 + make_float3 (0.5f);

    float3  _S531 = make_float3 (0.0f);

#line 128
    SpherHarmCoeffs_0 _S532 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S533;

#line 128
    (&_S533)->primal_1 = rgb_13;

#line 128
    (&_S533)->differential_0 = _S420;

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S534;

#line 128
    (&_S534)->primal_1 = _S531;

#line 128
    (&_S534)->differential_0 = _S420;

#line 128
    s_bwd_prop_max_1(&_S533, &_S534, _s_dOut_7);

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S535 = _S533;

#line 128
    SpherHarmCoeffs_0 _S536;

#line 128
    if(_S422)
    {

#line 128
        if(_S468)
        {

#line 128
            if(_S469)
            {

#line 121
                float3  _S537 = _S423 * _S535.differential_0;

#line 121
                float3  _S538 = _S424 * _S535.differential_0;

#line 121
                float _S539 = _S538.x + _S538.y + _S538.z;

#line 121
                float _S540 = _S453 * _S539;

#line 120
                float3  _S541 = _S425 * _S535.differential_0;

#line 120
                float3  _S542 = _S426 * _S535.differential_0;

#line 120
                float _S543 = _S542.x + _S542.y + _S542.z;

#line 120
                float _S544 = _S455 * _S543;

#line 120
                float _S545 = 1.44530570507049561f * (_S456 * _S543);

#line 119
                float3  _S546 = _S427 * _S535.differential_0;

#line 119
                float3  _S547 = _S428 * _S535.differential_0;

#line 119
                float _S548 = _S547.x + _S547.y + _S547.z;

#line 118
                float3  _S549 = _S429 * _S535.differential_0;

#line 118
                float3  _S550 = _S430 * _S535.differential_0;

#line 118
                float _S551 = _S550.x + _S550.y + _S550.z;

#line 118
                float _S552 = _S459 * _S551;

#line 118
                float _S553 = - _S552;

#line 118
                float _S554 = 3.0f * (- _S540 + _S553);

#line 118
                float _S555 = 0.37317633628845215f * (_S460 * _S551);

#line 117
                float3  _S556 = _S431 * _S535.differential_0;

#line 117
                float3  _S557 = _S432 * _S535.differential_0;

#line 117
                float _S558 = _S557.x + _S557.y + _S557.z;

#line 117
                float _S559 = _S457 * _S548 + _S461 * _S558;

#line 117
                float _S560 = - _S559;

#line 117
                float _S561 = 4.0f * _S559;

#line 117
                float _S562 = -0.4570457935333252f * (_S458 * _S558);

#line 116
                float3  _S563 = _S433 * _S535.differential_0;

#line 116
                float3  _S564 = _S434 * _S535.differential_0;

#line 116
                float _S565 = _S564.x + _S564.y + _S564.z;

#line 116
                float _S566 = _S462 * _S565;

#line 116
                float _S567 = 2.89061141014099121f * (_S463 * _S565);

#line 115
                float3  _S568 = _S435 * _S535.differential_0;

#line 115
                float3  _S569 = _S436 * _S535.differential_0;

#line 115
                float _S570 = _S569.x + _S569.y + _S569.z;

#line 115
                float _S571 = _S464 * _S570;

#line 115
                float _S572 = - _S571;

#line 115
                float _S573 = 3.0f * (_S553 + _S571);

#line 115
                float _S574 = -0.59004360437393188f * (_S465 * _S570);

#line 100
                float _S575 = -0.59004360437393188f * (_S454 * _S539) + -0.4570457935333252f * (_S458 * _S548);

#line 100
                SpherHarmCoeffs_0 _S576 = _S532;

#line 100
                (&_S576)->coeff15_0 = _S537;

#line 100
                (&_S576)->coeff14_0 = _S541;

#line 100
                (&_S576)->coeff13_0 = _S546;

#line 100
                (&_S576)->coeff12_0 = _S549;

#line 100
                (&_S576)->coeff11_0 = _S556;

#line 100
                (&_S576)->coeff10_0 = _S563;

#line 100
                (&_S576)->coeff9_0 = _S568;

#line 100
                SpherHarmCoeffs_0 _S577 = SpherHarmCoeffs_x24_syn_dadd_0(_S532, _S576);


                float _S578 = _S554 + _S560 + _S572;

#line 103
                float _S579 = _S540 + _S560 + _S573;

#line 100
                float _S580 = _S545 + _S555 + _S566;

#line 100
                float _S581 = _S562 + _S574;

#line 100
                _S453 = _S544;

#line 100
                _S454 = _S552;

#line 100
                _S455 = _S567;

#line 100
                _S456 = _S561;

#line 100
                _S457 = _S578;

#line 100
                _S458 = _S579;

#line 100
                _S536 = _S577;

#line 100
                _S459 = _S575;

#line 100
                _S460 = _S581;

#line 100
                _S461 = _S580;

#line 100
            }
            else
            {

#line 100
                _S453 = 0.0f;

#line 100
                _S454 = 0.0f;

#line 100
                _S455 = 0.0f;

#line 100
                _S456 = 0.0f;

#line 100
                _S457 = 0.0f;

#line 100
                _S458 = 0.0f;

#line 100
                _S536 = _S532;

#line 100
                _S459 = 0.0f;

#line 100
                _S460 = 0.0f;

#line 100
                _S461 = 0.0f;

#line 100
            }

#line 110
            float3  _S582 = _S437 * _S535.differential_0;

#line 110
            float3  _S583 = _S438 * _S535.differential_0;

#line 110
            float _S584 = 0.54627424478530884f * (_S583.x + _S583.y + _S583.z) + _S453;

#line 109
            float3  _S585 = _S439 * _S535.differential_0;

#line 109
            float3  _S586 = _S440 * _S535.differential_0;

#line 109
            float s_diff_xz_T_0 = -1.09254848957061768f * (_S586.x + _S586.y + _S586.z);

#line 108
            float3  _S587 = _S441 * _S535.differential_0;

#line 108
            float3  _S588 = _S442 * _S535.differential_0;

#line 108
            float _S589 = 0.31539157032966614f * (_S588.x + _S588.y + _S588.z);

#line 108
            float _S590 = - _S589;

#line 107
            float3  _S591 = _S443 * _S535.differential_0;

#line 107
            float3  _S592 = _S444 * _S535.differential_0;

#line 107
            float s_diff_yz_T_0 = -1.09254848957061768f * (_S592.x + _S592.y + _S592.z);

#line 106
            float3  _S593 = _S445 * _S535.differential_0;

#line 106
            float3  _S594 = _S446 * _S535.differential_0;

#line 104
            float _S595 = _S466 * s_diff_xz_T_0;

#line 104
            float _S596 = _S463 * s_diff_xz_T_0;

#line 104
            float _S597 = _S467 * s_diff_yz_T_0;

#line 104
            float _S598 = _S463 * s_diff_yz_T_0;

#line 104
            float _S599 = 1.09254848957061768f * (_S594.x + _S594.y + _S594.z) + _S455;

#line 104
            float _S600 = _S466 * _S599;

#line 104
            float _S601 = _S467 * _S599;

#line 103
            float _S602 = 2.0f * (_S589 + _S454) + _S456;

#line 103
            float _S603 = _S463 * _S602;

#line 103
            float _S604 = _S463 * _S602;

#line 103
            float _S605 = - _S584 + _S590 + _S457;

#line 103
            float _S606 = _S467 * _S605;

#line 103
            float _S607 = _S467 * _S605;

#line 103
            float _S608 = _S584 + _S590 + _S458;

#line 103
            float _S609 = _S466 * _S608;

#line 103
            float _S610 = _S466 * _S608;

#line 103
            SpherHarmCoeffs_0 _S611 = _S532;

#line 103
            (&_S611)->coeff8_0 = _S582;

#line 103
            (&_S611)->coeff7_0 = _S585;

#line 103
            (&_S611)->coeff6_0 = _S587;

#line 103
            (&_S611)->coeff5_0 = _S591;

#line 103
            (&_S611)->coeff4_0 = _S593;

#line 103
            SpherHarmCoeffs_0 _S612 = SpherHarmCoeffs_x24_syn_dadd_0(_S536, _S611);

#line 100
            float _S613 = _S598 + _S600 + _S606 + _S607 + _S460;

#line 100
            float _S614 = _S595 + _S597 + _S603 + _S604 + _S461;

#line 100
            _S453 = _S596 + _S601 + _S609 + _S610 + _S459;

#line 100
            _S454 = _S614;

#line 100
            _S455 = _S613;

#line 100
            _S536 = _S612;

#line 100
        }
        else
        {

#line 100
            _S453 = 0.0f;

#line 100
            _S454 = 0.0f;

#line 100
            _S455 = 0.0f;

#line 100
            _S536 = _S532;

#line 100
        }

#line 100
        float3  _S615 = - _S535.differential_0;

#line 100
        float3  _S616 = _S447 * _S615;

#line 100
        float3  _S617 = _S448 * _S615;

#line 100
        float3  _S618 = _S449 * _S535.differential_0;

#line 100
        float3  _S619 = _S450 * _S535.differential_0;

#line 100
        float3  _S620 = _S451 * _S615;

#line 100
        float3  _S621 = _S452 * _S615;

#line 96
        float3  _S622 = make_float3 (0.48860251903533936f * (_S617.x + _S617.y + _S617.z) + _S453, 0.48860251903533936f * (_S621.x + _S621.y + _S621.z) + _S455, 0.48860251903533936f * (_S619.x + _S619.y + _S619.z) + _S454);

#line 96
        SpherHarmCoeffs_0 _S623 = _S532;

#line 96
        (&_S623)->coeff3_0 = _S616;

#line 96
        (&_S623)->coeff2_0 = _S618;

#line 96
        (&_S623)->coeff1_0 = _S620;

#line 96
        SpherHarmCoeffs_0 _S624 = SpherHarmCoeffs_x24_syn_dadd_0(_S536, _S623);

#line 96
        rgb_10 = _S622;

#line 96
        _S536 = _S624;

#line 96
    }
    else
    {

#line 96
        rgb_10 = _S420;

#line 96
        _S536 = _S532;

#line 96
    }

    float3  _S625 = make_float3 (0.282094806432724f) * _S535.differential_0;

#line 96
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S626;

#line 96
    (&_S626)->primal_1 = dir_1;

#line 96
    (&_S626)->differential_0 = _S420;

#line 96
    s_bwd_normalize_impl_0(&_S626, rgb_10);

#line 95
    float3  _S627 = - _S626.differential_0;

#line 95
    dpcam_pos_1->primal_1 = (*dpcam_pos_1).primal_1;

#line 95
    dpcam_pos_1->differential_0 = _S627;

#line 95
    dpg_xyz_ws_1->primal_1 = (*dpg_xyz_ws_1).primal_1;

#line 95
    dpg_xyz_ws_1->differential_0 = _S626.differential_0;

#line 95
    SpherHarmCoeffs_0 _S628 = _S532;

#line 95
    (&_S628)->coeff0_0 = _S625;

#line 95
    SpherHarmCoeffs_0 _S629 = SpherHarmCoeffs_x24_syn_dadd_0(_S536, _S628);

#line 95
    dpsh_1->primal_1 = (*dpsh_1).primal_1;

#line 95
    dpsh_1->differential_0 = _S629;

#line 94
    return;
}


#line 112 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ void s_bwd_prop_geom_transform_points2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_4, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dptransf_matrix_3, float3  _s_dOut_8)
{
    float4  _S630 = make_float4 ((*dppoint_4).primal_1.x, (*dppoint_4).primal_1.y, (*dppoint_4).primal_1.z, 1.0f);

#line 114
    float4  _S631 = make_float4 (_s_dOut_8.x, _s_dOut_8.y, _s_dOut_8.z, 0.0f);

#line 114
    Matrix<float, 4, 4>  _S632 = makeMatrix<float, 4, 4> (0.0f);

#line 114
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S633;

#line 114
    (&_S633)->primal_1 = (*dptransf_matrix_3).primal_1;

#line 114
    (&_S633)->differential_0 = _S632;

#line 114
    float4  _S634 = make_float4 (0.0f);

#line 114
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S635;

#line 114
    (&_S635)->primal_1 = _S630;

#line 114
    (&_S635)->differential_0 = _S634;

#line 114
    s_bwd_prop_mul_1(&_S633, &_S635, _S631);

#line 114
    float3  _S636 = float3 {_S635.differential_0.x, _S635.differential_0.y, _S635.differential_0.z};

#line 114
    dptransf_matrix_3->primal_1 = (*dptransf_matrix_3).primal_1;

#line 114
    dptransf_matrix_3->differential_0 = _S633.differential_0;

#line 114
    dppoint_4->primal_1 = (*dppoint_4).primal_1;

#line 114
    dppoint_4->differential_0 = _S636;

#line 112
    return;
}


#line 112
__device__ void s_bwd_prop_mul_2(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S637, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S638, Matrix<float, 4, 4>  _S639)
{

#line 112
    mul_0(_S637, _S638, _S639);

#line 112
    return;
}


#line 119
__device__ void s_bwd_prop_project_point_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_5, DiffPair_Camera_0 * dpcam_6, float3  _s_dOut_9)
{

#line 119
    Matrix<float, 4, 4>  _S640 = s_primal_ctx_mul_0((*dpcam_6).primal_1.proj_mat_1, (*dpcam_6).primal_1.world_view_transform_1);

#line 119
    float3  _S641 = _s_dOut_9;

#line 119
    *&((&_S641)->z) = 0.0f;

    float3  _S642 = make_float3 (0.0f, 0.0f, _s_dOut_9.z);

#line 121
    float3  _S643 = make_float3 (0.0f);

#line 121
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S644;

#line 121
    (&_S644)->primal_1 = (*dppoint_5).primal_1;

#line 121
    (&_S644)->differential_0 = _S643;

#line 121
    Matrix<float, 4, 4>  _S645 = makeMatrix<float, 4, 4> (0.0f);

#line 121
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S646;

#line 121
    (&_S646)->primal_1 = (*dpcam_6).primal_1.world_view_transform_1;

#line 121
    (&_S646)->differential_0 = _S645;

#line 121
    s_bwd_prop_geom_transform_points2_0(&_S644, &_S646, _S642);

#line 120
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S647;

#line 120
    (&_S647)->primal_1 = (*dppoint_5).primal_1;

#line 120
    (&_S647)->differential_0 = _S643;

#line 120
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S648;

#line 120
    (&_S648)->primal_1 = _S640;

#line 120
    (&_S648)->differential_0 = _S645;

#line 120
    s_bwd_prop_geom_transform_points_0(&_S647, &_S648, _S641);

#line 120
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S649;

#line 120
    (&_S649)->primal_1 = (*dpcam_6).primal_1.proj_mat_1;

#line 120
    (&_S649)->differential_0 = _S645;

#line 120
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S650;

#line 120
    (&_S650)->primal_1 = (*dpcam_6).primal_1.world_view_transform_1;

#line 120
    (&_S650)->differential_0 = _S645;

#line 120
    s_bwd_prop_mul_2(&_S649, &_S650, _S648.differential_0);

#line 120
    Matrix<float, 4, 4>  _S651 = _S646.differential_0 + _S650.differential_0;

#line 120
    Camera_Differential_0 _S652 = Camera_x24_syn_dzero_0();

#line 120
    (&_S652)->world_view_transform_0 = _S651;

#line 120
    (&_S652)->proj_mat_0 = _S649.differential_0;

#line 120
    dpcam_6->primal_1 = (*dpcam_6).primal_1;

#line 120
    dpcam_6->differential_0 = _S652;

#line 120
    float3  _S653 = _S644.differential_0 + _S647.differential_0;

#line 120
    dppoint_5->primal_1 = (*dppoint_5).primal_1;

#line 120
    dppoint_5->differential_0 = _S653;

#line 119
    return;
}


#line 222
__device__ void s_bwd_prop_project_gaussian_to_camera_0(DiffPair_Gaussian_3D_0 * dpg_1, DiffPair_Camera_0 * dpcam_7, uint active_sh_10, Splat_2D_Vertex_0 _s_dOut_10)
{

#line 222
    DiffPair_Gaussian_3D_0 _S654 = *dpg_1;

#line 222
    DiffPair_Camera_0 _S655 = *dpcam_7;

#line 222
    float3  _S656 = make_float3 (0.0f);

#line 222
    float4  _S657 = make_float4 (0.0f);

#line 228
    Matrix<float, 3, 3>  _S658 = makeMatrix<float, 3, 3> (0.0f);

#line 228
    bool _S659 = !(s_primal_ctx_project_point_0((*dpg_1).primal_1.xyz_ws_0, (*dpcam_7).primal_1).z <= 0.20000000298023224f);

#line 228
    Matrix<float, 3, 3>  _S660;

#line 228
    float4  _S661;

#line 228
    float3  _S662;

#line 228
    float3  _S663;

#line 228
    SpherHarmCoeffs_0 _S664;

#line 228
    if(_S659)
    {

#line 228
        _S660 = s_primal_ctx_get_covariance_from_quat_scales_0(_S654.primal_1.rotations_0, _S654.primal_1.scales_0);

#line 228
        _S661 = _S654.primal_1.rotations_0;

#line 228
        _S662 = _S654.primal_1.scales_0;

#line 228
        _S664 = _S654.primal_1.sh_coeffs_0;

#line 228
        _S663 = _S655.primal_1.position_1;

#line 228
    }
    else
    {

#line 228
        _S660 = _S658;

#line 228
        _S661 = _S657;

#line 228
        _S662 = _S656;

#line 228
        (&_S664)->coeff0_0 = _S656;

#line 228
        (&_S664)->coeff1_0 = _S656;

#line 228
        (&_S664)->coeff2_0 = _S656;

#line 228
        (&_S664)->coeff3_0 = _S656;

#line 228
        (&_S664)->coeff4_0 = _S656;

#line 228
        (&_S664)->coeff5_0 = _S656;

#line 228
        (&_S664)->coeff6_0 = _S656;

#line 228
        (&_S664)->coeff7_0 = _S656;

#line 228
        (&_S664)->coeff8_0 = _S656;

#line 228
        (&_S664)->coeff9_0 = _S656;

#line 228
        (&_S664)->coeff10_0 = _S656;

#line 228
        (&_S664)->coeff11_0 = _S656;

#line 228
        (&_S664)->coeff12_0 = _S656;

#line 228
        (&_S664)->coeff13_0 = _S656;

#line 228
        (&_S664)->coeff14_0 = _S656;

#line 228
        (&_S664)->coeff15_0 = _S656;

#line 228
        _S663 = _S656;

#line 228
    }

#line 228
    Camera_Differential_0 _S665 = Camera_x24_syn_dzero_0();

#line 228
    Gaussian_3D_0 _S666 = Gaussian_3D_x24_syn_dzero_0();

#line 228
    Splat_2D_Vertex_0 _S667 = Splat_2D_Vertex_x24_syn_dadd_0(_s_dOut_10, Splat_2D_Vertex_x24_syn_dzero_0());

#line 228
    Camera_Differential_0 _S668;

#line 228
    Gaussian_3D_0 _S669;

#line 228
    if(_S659)
    {

#line 229
        DiffPair_Camera_0 _S670;

#line 229
        (&_S670)->primal_1 = _S655.primal_1;

#line 229
        (&_S670)->differential_0 = _S665;

#line 229
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S671;

#line 229
        (&_S671)->primal_1 = _S654.primal_1.xyz_ws_0;

#line 229
        (&_S671)->differential_0 = _S656;

#line 229
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S672;

#line 229
        (&_S672)->primal_1 = _S660;

#line 229
        (&_S672)->differential_0 = _S658;

#line 229
        s_bwd_prop_covariance_3d_to_2d_0(&_S670, &_S671, &_S672, _S667.cov_vs_0);

#line 228
        DiffPair_vectorx3Cfloatx2C4x3E_0 _S673;

#line 228
        (&_S673)->primal_1 = _S661;

#line 228
        (&_S673)->differential_0 = _S657;

#line 228
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S674;

#line 228
        (&_S674)->primal_1 = _S662;

#line 228
        (&_S674)->differential_0 = _S656;

#line 228
        s_bwd_prop_get_covariance_from_quat_scales_0(&_S673, &_S674, _S672.differential_0);

#line 227
        SpherHarmCoeffs_0 _S675 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 227
        DiffPair_SpherHarmCoeffs_0 _S676;

#line 227
        (&_S676)->primal_1 = _S664;

#line 227
        (&_S676)->differential_0 = _S675;

#line 227
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S677;

#line 227
        (&_S677)->primal_1 = _S654.primal_1.xyz_ws_0;

#line 227
        (&_S677)->differential_0 = _S656;

#line 227
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S678;

#line 227
        (&_S678)->primal_1 = _S663;

#line 227
        (&_S678)->differential_0 = _S656;

#line 227
        s_bwd_prop_compute_color_from_sh_coeffs_0(&_S676, &_S677, &_S678, active_sh_10, _S667.rgb_0);

#line 227
        Gaussian_3D_0 _S679 = _S666;

#line 227
        (&_S679)->scales_0 = _S674.differential_0;

#line 227
        (&_S679)->rotations_0 = _S673.differential_0;

#line 227
        (&_S679)->sh_coeffs_0 = _S676.differential_0;

#line 227
        Gaussian_3D_0 _S680 = Gaussian_3D_x24_syn_dadd_0(_S666, _S679);

#line 227
        float3  _S681 = _S671.differential_0 + _S677.differential_0;

#line 227
        Camera_Differential_0 _S682 = Camera_x24_syn_dadd_0(_S670.differential_0, _S665);

#line 227
        Camera_Differential_0 _S683 = _S665;

#line 227
        (&_S683)->position_0 = _S678.differential_0;

#line 227
        Camera_Differential_0 _S684 = Camera_x24_syn_dadd_0(_S682, _S683);

#line 227
        _S662 = _S667.xyz_vs_0;

#line 227
        _S663 = _S681;

#line 227
        _S668 = _S684;

#line 227
        _S669 = _S680;

#line 227
    }
    else
    {

#line 227
        _S662 = _S656;

#line 227
        _S663 = _S656;

#line 227
        _S668 = _S665;

#line 227
        _S669 = _S666;

#line 227
    }

#line 223
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S685;

#line 223
    (&_S685)->primal_1 = _S654.primal_1.xyz_ws_0;

#line 223
    (&_S685)->differential_0 = _S656;

#line 223
    DiffPair_Camera_0 _S686;

#line 223
    (&_S686)->primal_1 = _S655.primal_1;

#line 223
    (&_S686)->differential_0 = _S665;

#line 223
    s_bwd_prop_project_point_0(&_S685, &_S686, _S662);

#line 223
    float3  _S687 = _S685.differential_0 + _S663;

#line 223
    Camera_Differential_0 _S688 = Camera_x24_syn_dadd_0(_S686.differential_0, _S668);

#line 223
    dpcam_7->primal_1 = (*dpcam_7).primal_1;

#line 223
    dpcam_7->differential_0 = _S688;

#line 223
    Gaussian_3D_0 _S689 = _S666;

#line 223
    (&_S689)->xyz_ws_0 = _S687;

#line 223
    Gaussian_3D_0 _S690 = Gaussian_3D_x24_syn_dadd_0(_S669, _S689);

#line 223
    dpg_1->primal_1 = (*dpg_1).primal_1;

#line 223
    dpg_1->differential_0 = _S690;

#line 222
    return;
}


#line 26
__device__ void s_bwd_prop_read_t3_float3_0(uint idx_4, DiffTensorView_0 t3_2, float3  _s_dOut_11)
{
    uint2  _S691 = make_uint2 (idx_4, 0U);
    uint2  _S692 = make_uint2 (idx_4, 1U);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, make_uint2 (idx_4, 2U), _s_dOut_11.z);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, _S692, _s_dOut_11.y);

#line 28
    AtomicAdd_load_backward_0(t3_2.diff_1, _S691, _s_dOut_11.x);

#line 26
    return;
}


#line 34
__device__ void s_bwd_prop_read_t4_float4_0(uint idx_5, DiffTensorView_0 t4_2, float4  _s_dOut_12)
{
    uint2  _S693 = make_uint2 (idx_5, 0U);
    uint2  _S694 = make_uint2 (idx_5, 1U);
    uint2  _S695 = make_uint2 (idx_5, 2U);

#line 36
    AtomicAdd_load_backward_0(t4_2.diff_1, make_uint2 (idx_5, 3U), _s_dOut_12.w);

#line 36
    AtomicAdd_load_backward_0(t4_2.diff_1, _S695, _s_dOut_12.z);

#line 36
    AtomicAdd_load_backward_0(t4_2.diff_1, _S694, _s_dOut_12.y);

#line 36
    AtomicAdd_load_backward_0(t4_2.diff_1, _S693, _s_dOut_12.x);

#line 34
    return;
}


#line 62 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
__device__ void s_bwd_prop_read_spherical_harmonics_coeffs_0(uint g_idx_5, DiffTensorView_0 sh_coeffs_6, uint active_sh_11, SpherHarmCoeffs_0 _s_dOut_13)
{

#line 68
    uint3  _S696 = make_uint3 (0U);

#line 65
    uint3  _S697 = make_uint3 (g_idx_5, 0U, 0U);

#line 65
    uint3  _S698 = make_uint3 (g_idx_5, 0U, 1U);

#line 65
    uint3  _S699 = make_uint3 (g_idx_5, 0U, 2U);

    bool _S700 = active_sh_11 > 0U;

#line 67
    uint3  _S701;

#line 67
    uint3  _S702;

#line 67
    uint3  _S703;

#line 67
    uint3  _S704;

#line 67
    uint3  _S705;

#line 67
    uint3  _S706;

#line 67
    uint3  _S707;

#line 67
    uint3  _S708;

#line 67
    uint3  _S709;

#line 67
    uint3  _S710;

#line 67
    uint3  _S711;

#line 67
    uint3  _S712;

#line 67
    uint3  _S713;

#line 67
    uint3  _S714;

#line 67
    uint3  _S715;

#line 67
    uint3  _S716;

#line 67
    uint3  _S717;

#line 67
    uint3  _S718;

#line 67
    uint3  _S719;

#line 67
    uint3  _S720;

#line 67
    uint3  _S721;

#line 67
    uint3  _S722;

#line 67
    uint3  _S723;

#line 67
    uint3  _S724;

#line 67
    uint3  _S725;

#line 67
    uint3  _S726;

#line 67
    uint3  _S727;

#line 67
    uint3  _S728;

#line 67
    uint3  _S729;

#line 67
    uint3  _S730;

#line 67
    uint3  _S731;

#line 67
    uint3  _S732;

#line 67
    uint3  _S733;

#line 67
    uint3  _S734;

#line 67
    uint3  _S735;

#line 67
    uint3  _S736;

#line 67
    uint3  _S737;

#line 67
    uint3  _S738;

#line 67
    uint3  _S739;

#line 67
    uint3  _S740;

#line 67
    uint3  _S741;

#line 67
    uint3  _S742;

#line 67
    uint3  _S743;

#line 67
    uint3  _S744;

#line 67
    uint3  _S745;

#line 67
    bool _S746;

#line 67
    bool _S747;

#line 67
    if(_S700)
    {

#line 68
        uint3  _S748 = make_uint3 (g_idx_5, 1U, 0U);

#line 68
        uint3  _S749 = make_uint3 (g_idx_5, 1U, 1U);

#line 68
        uint3  _S750 = make_uint3 (g_idx_5, 1U, 2U);
        uint3  _S751 = make_uint3 (g_idx_5, 2U, 0U);

#line 69
        uint3  _S752 = make_uint3 (g_idx_5, 2U, 1U);

#line 69
        uint3  _S753 = make_uint3 (g_idx_5, 2U, 2U);
        uint3  _S754 = make_uint3 (g_idx_5, 3U, 0U);

#line 70
        uint3  _S755 = make_uint3 (g_idx_5, 3U, 1U);

#line 70
        uint3  _S756 = make_uint3 (g_idx_5, 3U, 2U);

        bool _S757 = active_sh_11 > 1U;

#line 72
        if(_S757)
        {

#line 73
            uint3  _S758 = make_uint3 (g_idx_5, 4U, 0U);

#line 73
            uint3  _S759 = make_uint3 (g_idx_5, 4U, 1U);

#line 73
            uint3  _S760 = make_uint3 (g_idx_5, 4U, 2U);
            uint3  _S761 = make_uint3 (g_idx_5, 5U, 0U);

#line 74
            uint3  _S762 = make_uint3 (g_idx_5, 5U, 1U);

#line 74
            uint3  _S763 = make_uint3 (g_idx_5, 5U, 2U);
            uint3  _S764 = make_uint3 (g_idx_5, 6U, 0U);

#line 75
            uint3  _S765 = make_uint3 (g_idx_5, 6U, 1U);

#line 75
            uint3  _S766 = make_uint3 (g_idx_5, 6U, 2U);
            uint3  _S767 = make_uint3 (g_idx_5, 7U, 0U);

#line 76
            uint3  _S768 = make_uint3 (g_idx_5, 7U, 1U);

#line 76
            uint3  _S769 = make_uint3 (g_idx_5, 7U, 2U);
            uint3  _S770 = make_uint3 (g_idx_5, 8U, 0U);

#line 77
            uint3  _S771 = make_uint3 (g_idx_5, 8U, 1U);

#line 77
            uint3  _S772 = make_uint3 (g_idx_5, 8U, 2U);

            bool _S773 = active_sh_11 > 2U;

#line 79
            if(_S773)
            {

#line 80
                uint3  _S774 = make_uint3 (g_idx_5, 9U, 0U);

#line 80
                uint3  _S775 = make_uint3 (g_idx_5, 9U, 1U);

#line 80
                uint3  _S776 = make_uint3 (g_idx_5, 9U, 2U);
                uint3  _S777 = make_uint3 (g_idx_5, 10U, 0U);

#line 81
                uint3  _S778 = make_uint3 (g_idx_5, 10U, 1U);

#line 81
                uint3  _S779 = make_uint3 (g_idx_5, 10U, 2U);
                uint3  _S780 = make_uint3 (g_idx_5, 11U, 0U);

#line 82
                uint3  _S781 = make_uint3 (g_idx_5, 11U, 1U);

#line 82
                uint3  _S782 = make_uint3 (g_idx_5, 11U, 2U);
                uint3  _S783 = make_uint3 (g_idx_5, 12U, 0U);

#line 83
                uint3  _S784 = make_uint3 (g_idx_5, 12U, 1U);

#line 83
                uint3  _S785 = make_uint3 (g_idx_5, 12U, 2U);
                uint3  _S786 = make_uint3 (g_idx_5, 13U, 0U);

#line 84
                uint3  _S787 = make_uint3 (g_idx_5, 13U, 1U);

#line 84
                uint3  _S788 = make_uint3 (g_idx_5, 13U, 2U);
                uint3  _S789 = make_uint3 (g_idx_5, 14U, 0U);

#line 85
                uint3  _S790 = make_uint3 (g_idx_5, 14U, 1U);

#line 85
                uint3  _S791 = make_uint3 (g_idx_5, 14U, 2U);
                uint3  _S792 = make_uint3 (g_idx_5, 15U, 0U);

#line 86
                uint3  _S793 = make_uint3 (g_idx_5, 15U, 1U);

#line 86
                _S701 = make_uint3 (g_idx_5, 15U, 2U);

#line 86
                _S702 = _S793;

#line 86
                _S703 = _S792;

#line 86
                _S704 = _S791;

#line 86
                _S705 = _S790;

#line 86
                _S706 = _S789;

#line 86
                _S707 = _S788;

#line 86
                _S708 = _S787;

#line 86
                _S709 = _S786;

#line 86
                _S710 = _S785;

#line 86
                _S711 = _S784;

#line 86
                _S712 = _S783;

#line 86
                _S713 = _S782;

#line 86
                _S714 = _S781;

#line 86
                _S715 = _S780;

#line 86
                _S716 = _S779;

#line 86
                _S717 = _S778;

#line 86
                _S718 = _S777;

#line 86
                _S719 = _S776;

#line 86
                _S720 = _S775;

#line 86
                _S721 = _S774;

#line 86
            }
            else
            {

#line 86
                _S701 = _S696;

#line 86
                _S702 = _S696;

#line 86
                _S703 = _S696;

#line 86
                _S704 = _S696;

#line 86
                _S705 = _S696;

#line 86
                _S706 = _S696;

#line 86
                _S707 = _S696;

#line 86
                _S708 = _S696;

#line 86
                _S709 = _S696;

#line 86
                _S710 = _S696;

#line 86
                _S711 = _S696;

#line 86
                _S712 = _S696;

#line 86
                _S713 = _S696;

#line 86
                _S714 = _S696;

#line 86
                _S715 = _S696;

#line 86
                _S716 = _S696;

#line 86
                _S717 = _S696;

#line 86
                _S718 = _S696;

#line 86
                _S719 = _S696;

#line 86
                _S720 = _S696;

#line 86
                _S721 = _S696;

#line 86
            }

#line 86
            _S746 = _S773;

#line 86
            _S722 = _S772;

#line 86
            _S723 = _S771;

#line 86
            _S724 = _S770;

#line 86
            _S725 = _S769;

#line 86
            _S726 = _S768;

#line 86
            _S727 = _S767;

#line 86
            _S728 = _S766;

#line 86
            _S729 = _S765;

#line 86
            _S730 = _S764;

#line 86
            _S731 = _S763;

#line 86
            _S732 = _S762;

#line 86
            _S733 = _S761;

#line 86
            _S734 = _S760;

#line 86
            _S735 = _S759;

#line 86
            _S736 = _S758;

#line 86
        }
        else
        {

#line 86
            _S746 = false;

#line 86
            _S701 = _S696;

#line 86
            _S702 = _S696;

#line 86
            _S703 = _S696;

#line 86
            _S704 = _S696;

#line 86
            _S705 = _S696;

#line 86
            _S706 = _S696;

#line 86
            _S707 = _S696;

#line 86
            _S708 = _S696;

#line 86
            _S709 = _S696;

#line 86
            _S710 = _S696;

#line 86
            _S711 = _S696;

#line 86
            _S712 = _S696;

#line 86
            _S713 = _S696;

#line 86
            _S714 = _S696;

#line 86
            _S715 = _S696;

#line 86
            _S716 = _S696;

#line 86
            _S717 = _S696;

#line 86
            _S718 = _S696;

#line 86
            _S719 = _S696;

#line 86
            _S720 = _S696;

#line 86
            _S721 = _S696;

#line 86
            _S722 = _S696;

#line 86
            _S723 = _S696;

#line 86
            _S724 = _S696;

#line 86
            _S725 = _S696;

#line 86
            _S726 = _S696;

#line 86
            _S727 = _S696;

#line 86
            _S728 = _S696;

#line 86
            _S729 = _S696;

#line 86
            _S730 = _S696;

#line 86
            _S731 = _S696;

#line 86
            _S732 = _S696;

#line 86
            _S733 = _S696;

#line 86
            _S734 = _S696;

#line 86
            _S735 = _S696;

#line 86
            _S736 = _S696;

#line 86
        }

#line 79
        bool _S794 = _S746;

#line 79
        _S746 = _S757;

#line 79
        _S747 = _S794;

#line 79
        _S737 = _S756;

#line 79
        _S738 = _S755;

#line 79
        _S739 = _S754;

#line 79
        _S740 = _S753;

#line 79
        _S741 = _S752;

#line 79
        _S742 = _S751;

#line 79
        _S743 = _S750;

#line 79
        _S744 = _S749;

#line 79
        _S745 = _S748;

#line 79
    }
    else
    {

#line 79
        _S746 = false;

#line 79
        _S747 = false;

#line 79
        _S701 = _S696;

#line 79
        _S702 = _S696;

#line 79
        _S703 = _S696;

#line 79
        _S704 = _S696;

#line 79
        _S705 = _S696;

#line 79
        _S706 = _S696;

#line 79
        _S707 = _S696;

#line 79
        _S708 = _S696;

#line 79
        _S709 = _S696;

#line 79
        _S710 = _S696;

#line 79
        _S711 = _S696;

#line 79
        _S712 = _S696;

#line 79
        _S713 = _S696;

#line 79
        _S714 = _S696;

#line 79
        _S715 = _S696;

#line 79
        _S716 = _S696;

#line 79
        _S717 = _S696;

#line 79
        _S718 = _S696;

#line 79
        _S719 = _S696;

#line 79
        _S720 = _S696;

#line 79
        _S721 = _S696;

#line 79
        _S722 = _S696;

#line 79
        _S723 = _S696;

#line 79
        _S724 = _S696;

#line 79
        _S725 = _S696;

#line 79
        _S726 = _S696;

#line 79
        _S727 = _S696;

#line 79
        _S728 = _S696;

#line 79
        _S729 = _S696;

#line 79
        _S730 = _S696;

#line 79
        _S731 = _S696;

#line 79
        _S732 = _S696;

#line 79
        _S733 = _S696;

#line 79
        _S734 = _S696;

#line 79
        _S735 = _S696;

#line 79
        _S736 = _S696;

#line 79
        _S737 = _S696;

#line 79
        _S738 = _S696;

#line 79
        _S739 = _S696;

#line 79
        _S740 = _S696;

#line 79
        _S741 = _S696;

#line 79
        _S742 = _S696;

#line 79
        _S743 = _S696;

#line 79
        _S744 = _S696;

#line 79
        _S745 = _S696;

#line 79
    }

#line 79
    SpherHarmCoeffs_0 _S795 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 77
    float3  _S796 = make_float3 (0.0f);

#line 77
    SpherHarmCoeffs_0 _S797;

#line 77
    float3  _S798;

#line 77
    if(_S700)
    {

#line 77
        float3  _S799;

#line 77
        float3  _S800;

#line 77
        float3  _S801;

#line 77
        if(_S746)
        {

#line 77
            float3  _S802;

#line 77
            float3  _S803;

#line 77
            float3  _S804;

#line 77
            float3  _S805;

#line 77
            float3  _S806;

#line 77
            if(_S747)
            {

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S701, _s_dOut_13.coeff15_0.z);

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S702, _s_dOut_13.coeff15_0.y);

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S703, _s_dOut_13.coeff15_0.x);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S704, _s_dOut_13.coeff14_0.z);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S705, _s_dOut_13.coeff14_0.y);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S706, _s_dOut_13.coeff14_0.x);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S707, _s_dOut_13.coeff13_0.z);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S708, _s_dOut_13.coeff13_0.y);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S709, _s_dOut_13.coeff13_0.x);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S710, _s_dOut_13.coeff12_0.z);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S711, _s_dOut_13.coeff12_0.y);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S712, _s_dOut_13.coeff12_0.x);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S713, _s_dOut_13.coeff11_0.z);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S714, _s_dOut_13.coeff11_0.y);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S715, _s_dOut_13.coeff11_0.x);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S716, _s_dOut_13.coeff10_0.z);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S717, _s_dOut_13.coeff10_0.y);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S718, _s_dOut_13.coeff10_0.x);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S719, _s_dOut_13.coeff9_0.z);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S720, _s_dOut_13.coeff9_0.y);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S721, _s_dOut_13.coeff9_0.x);

#line 80
                _S797 = _S795;

#line 80
                _S798 = _s_dOut_13.coeff8_0;

#line 80
                _S799 = _s_dOut_13.coeff7_0;

#line 80
                _S800 = _s_dOut_13.coeff6_0;

#line 80
                _S801 = _s_dOut_13.coeff5_0;

#line 80
                _S802 = _s_dOut_13.coeff4_0;

#line 80
                _S803 = _s_dOut_13.coeff0_0;

#line 80
                _S804 = _s_dOut_13.coeff1_0;

#line 80
                _S805 = _s_dOut_13.coeff2_0;

#line 80
                _S806 = _s_dOut_13.coeff3_0;

#line 80
            }
            else
            {

#line 80
                _S797 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S795);

#line 80
                _S798 = _S796;

#line 80
                _S799 = _S796;

#line 80
                _S800 = _S796;

#line 80
                _S801 = _S796;

#line 80
                _S802 = _S796;

#line 80
                _S803 = _S796;

#line 80
                _S804 = _S796;

#line 80
                _S805 = _S796;

#line 80
                _S806 = _S796;

#line 80
            }

#line 77
            float3  _S807 = _S797.coeff8_0 + _S798;

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S722, _S807.z);

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S723, _S807.y);

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S724, _S807.x);

#line 76
            float3  _S808 = _S797.coeff7_0 + _S799;

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S725, _S808.z);

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S726, _S808.y);

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S727, _S808.x);

#line 75
            float3  _S809 = _S797.coeff6_0 + _S800;

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S728, _S809.z);

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S729, _S809.y);

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S730, _S809.x);

#line 74
            float3  _S810 = _S797.coeff5_0 + _S801;

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S731, _S810.z);

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S732, _S810.y);

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S733, _S810.x);

#line 73
            float3  _S811 = _S797.coeff4_0 + _S802;

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S734, _S811.z);

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S735, _S811.y);

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S736, _S811.x);

#line 65
            float3  _S812 = _S797.coeff0_0 + _S803;


            float3  _S813 = _S797.coeff1_0 + _S804;
            float3  _S814 = _S797.coeff2_0 + _S805;
            float3  _S815 = _S797.coeff3_0 + _S806;

#line 70
            _S797 = _S795;

#line 70
            _S798 = _S815;

#line 70
            _S799 = _S814;

#line 70
            _S800 = _S813;

#line 70
            _S801 = _S812;

#line 70
        }
        else
        {

#line 70
            _S797 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S795);

#line 70
            _S798 = _S796;

#line 70
            _S799 = _S796;

#line 70
            _S800 = _S796;

#line 70
            _S801 = _S796;

#line 70
        }

#line 70
        float3  _S816 = _S797.coeff3_0 + _S798;

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S737, _S816.z);

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S738, _S816.y);

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S739, _S816.x);

#line 69
        float3  _S817 = _S797.coeff2_0 + _S799;

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S740, _S817.z);

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S741, _S817.y);

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S742, _S817.x);

#line 68
        float3  _S818 = _S797.coeff1_0 + _S800;

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S743, _S818.z);

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S744, _S818.y);

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S745, _S818.x);

#line 65
        float3  _S819 = _S797.coeff0_0 + _S801;

#line 65
        _S797 = _S795;

#line 65
        _S798 = _S819;

#line 65
    }
    else
    {

#line 65
        _S797 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S795);

#line 65
        _S798 = _S796;

#line 65
    }

#line 65
    float3  _S820 = _S797.coeff0_0 + _S798;

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S699, _S820.z);

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S698, _S820.y);

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S697, _S820.x);

#line 62
    return;
}


#line 170 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ void s_bwd_prop_load_gaussian_0(int g_idx_6, DiffTensorView_0 xyz_ws_6, DiffTensorView_0 sh_coeffs_7, DiffTensorView_0 rotations_4, DiffTensorView_0 scales_4, uint active_sh_12, Gaussian_3D_0 _s_dOut_14)
{

#line 177
    uint _S821 = uint(g_idx_6);

#line 177
    s_bwd_prop_read_t3_float3_0(_S821, scales_4, _s_dOut_14.scales_0);

#line 177
    s_bwd_prop_read_t4_float4_0(_S821, rotations_4, _s_dOut_14.rotations_0);

#line 177
    s_bwd_prop_read_spherical_harmonics_coeffs_0(_S821, sh_coeffs_7, active_sh_12, _s_dOut_14.sh_coeffs_0);

#line 177
    s_bwd_prop_read_t3_float3_0(_S821, xyz_ws_6, _s_dOut_14.xyz_ws_0);

#line 170
    return;
}


#line 37 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
__device__ void s_bwd_prop_vertex_shader_0(DiffTensorView_0 xyz_ws_7, DiffTensorView_0 sh_coeffs_8, DiffTensorView_0 rotations_5, DiffTensorView_0 scales_5, uint active_sh_13, TensorView world_view_transform_4, TensorView proj_mat_4, TensorView cam_pos_2, TensorView out_tiles_touched_1, TensorView out_rect_tile_space_1, TensorView out_radii_1, DiffTensorView_0 out_xyz_vs_1, DiffTensorView_0 out_inv_cov_vs_1, DiffTensorView_0 out_rgb_1, float fovy_4, float fovx_4, uint image_height_1, uint image_width_1, uint grid_height_2, uint grid_width_2, uint tile_height_2, uint tile_width_2, s_bwd_prop_vertex_shader_Intermediates_0 _s_diff_ctx_1)
{

#line 37
    Matrix<float, 2, 2>  _S822 = makeMatrix<float, 2, 2> (0.0f);

#line 92
    uint2  _S823 = make_uint2 (0U);

#line 100
    uint3  _S824 = make_uint3 (0U);

#line 60
    uint g_idx_7 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

#line 60
    bool _S825 = !(g_idx_7 >= DiffTensorView_size_0(xyz_ws_7, 0U));

#line 60
    bool _bflag_1;

#line 60
    bool _bflag_2;

#line 60
    bool _bflag_3;

#line 60
    uint2  _S826;

#line 60
    uint2  _S827;

#line 60
    uint2  _S828;

#line 60
    uint3  _S829;

#line 60
    uint3  _S830;

#line 60
    uint3  _S831;

#line 60
    uint3  _S832;

#line 60
    Matrix<float, 2, 2>  _S833;

#line 60
    Matrix<float, 2, 2>  _S834;

#line 60
    Matrix<float, 2, 2>  _S835;

#line 60
    Matrix<float, 2, 2>  _S836;

#line 60
    float _S837;

#line 60
    float _S838;

#line 60
    int _S839;

#line 60
    int _S840;

#line 60
    int _S841;

#line 60
    if(_S825)
    {

#line 66
        int _S842 = int(g_idx_7);

#line 66
        Splat_2D_Vertex_0 _S843 = s_primal_ctx_project_gaussian_to_camera_0(_s_diff_ctx_1._S112, _s_diff_ctx_1._S111, active_sh_13);

        if(_S843.xyz_vs_0.z <= 0.20000000298023224f)
        {

#line 68
            _bflag_1 = false;

#line 68
        }
        else
        {

#line 68
            _bflag_1 = _S825;

#line 68
        }

#line 68
        if(_bflag_1)
        {

#line 68
            float _S844 = s_primal_ctx_compute_det_0(_S843.cov_vs_0);

#line 88
            Matrix<float, 2, 2>  _S845 = makeMatrix<float, 2, 2> (_S844);

#line 74
            if(_S844 == 0.0f)
            {

#line 74
                _bflag_2 = false;

#line 74
            }
            else
            {

#line 74
                _bflag_2 = _bflag_1;

#line 74
            }

#line 74
            if(_bflag_2)
            {



                float _S846 = _S843.xyz_vs_0.x;

#line 79
                int _S847 = int(image_width_1);

#line 79
                float _S848 = _S843.xyz_vs_0.y;

#line 79
                int _S849 = int(image_height_1);
                rectangle_0 rect_tile_space_2 = get_rectangle_tile_space_0(make_float2 (s_primal_ctx_ndc2pix_0(_S846, _S847), s_primal_ctx_ndc2pix_0(_S848, _S849)), splat_radius_0(_S843.cov_vs_0, _S844), grid_height_2, grid_width_2, tile_height_2, tile_width_2);



                if((rect_tile_space_2.max_x_0 - rect_tile_space_2.min_x_0) * (rect_tile_space_2.max_y_0 - rect_tile_space_2.min_y_0) == int(0))
                {

#line 84
                    _bflag_3 = false;

#line 84
                }
                else
                {

#line 84
                    _bflag_3 = _bflag_2;

#line 84
                }

#line 84
                if(_bflag_3)
                {


                    Matrix<float, 2, 2>  _S850 = makeMatrix<float, 2, 2> (_S843.cov_vs_0.rows[int(1)].y, - _S843.cov_vs_0.rows[int(0)].y, - _S843.cov_vs_0.rows[int(1)].x, _S843.cov_vs_0.rows[int(0)].x);

#line 88
                    Matrix<float, 2, 2>  _S851 = makeMatrix<float, 2, 2> (_S844 * _S844);



                    uint2  _S852 = make_uint2 (g_idx_7, 0U);
                    uint2  _S853 = make_uint2 (g_idx_7, 1U);

#line 100
                    uint3  _S854 = make_uint3 (g_idx_7, 0U, 0U);
                    uint3  _S855 = make_uint3 (g_idx_7, 0U, 1U);
                    uint3  _S856 = make_uint3 (g_idx_7, 1U, 0U);
                    uint3  _S857 = make_uint3 (g_idx_7, 1U, 1U);

#line 103
                    _S826 = make_uint2 (g_idx_7, 2U);

#line 103
                    _S827 = _S853;

#line 103
                    _S828 = _S852;

#line 103
                    _S829 = _S857;

#line 103
                    _S830 = _S856;

#line 103
                    _S831 = _S855;

#line 103
                    _S832 = _S854;

#line 103
                    _S833 = _S851;

#line 103
                    _S834 = _S850;

#line 103
                }
                else
                {

#line 103
                    _S826 = _S823;

#line 103
                    _S827 = _S823;

#line 103
                    _S828 = _S823;

#line 103
                    _S829 = _S824;

#line 103
                    _S830 = _S824;

#line 103
                    _S831 = _S824;

#line 103
                    _S832 = _S824;

#line 103
                    _S833 = _S822;

#line 103
                    _S834 = _S822;

#line 103
                }

#line 103
                _S837 = _S848;

#line 103
                _S838 = _S846;

#line 103
                _S839 = _S849;

#line 103
                _S840 = _S847;

#line 103
            }
            else
            {

#line 103
                _bflag_3 = false;

#line 103
                _S826 = _S823;

#line 103
                _S827 = _S823;

#line 103
                _S828 = _S823;

#line 103
                _S829 = _S824;

#line 103
                _S830 = _S824;

#line 103
                _S831 = _S824;

#line 103
                _S832 = _S824;

#line 103
                _S837 = 0.0f;

#line 103
                _S838 = 0.0f;

#line 103
                _S833 = _S822;

#line 103
                _S834 = _S822;

#line 103
                _S839 = int(0);

#line 103
                _S840 = int(0);

#line 103
            }

#line 103
            _S835 = _S845;

#line 103
            _S836 = _S843.cov_vs_0;

#line 103
        }
        else
        {

#line 103
            _bflag_2 = false;

#line 103
            _bflag_3 = false;

#line 103
            _S826 = _S823;

#line 103
            _S827 = _S823;

#line 103
            _S828 = _S823;

#line 103
            _S829 = _S824;

#line 103
            _S830 = _S824;

#line 103
            _S831 = _S824;

#line 103
            _S832 = _S824;

#line 103
            _S837 = 0.0f;

#line 103
            _S838 = 0.0f;

#line 103
            _S833 = _S822;

#line 103
            _S834 = _S822;

#line 103
            _S835 = _S822;

#line 103
            _S839 = int(0);

#line 103
            _S840 = int(0);

#line 103
            _S836 = _S822;

#line 103
        }

#line 103
        _S841 = _S842;

#line 103
    }
    else
    {

#line 103
        _bflag_1 = false;

#line 103
        _bflag_2 = false;

#line 103
        _bflag_3 = false;

#line 103
        _S826 = _S823;

#line 103
        _S827 = _S823;

#line 103
        _S828 = _S823;

#line 103
        _S829 = _S824;

#line 103
        _S830 = _S824;

#line 103
        _S831 = _S824;

#line 103
        _S832 = _S824;

#line 103
        _S837 = 0.0f;

#line 103
        _S838 = 0.0f;

#line 103
        _S833 = _S822;

#line 103
        _S834 = _S822;

#line 103
        _S835 = _S822;

#line 103
        _S839 = int(0);

#line 103
        _S840 = int(0);

#line 103
        _S836 = _S822;

#line 103
        _S841 = int(0);

#line 103
    }

#line 1751 "core.meta.slang"
    float3  _S858 = make_float3 (0.0f);

#line 67 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
    Splat_2D_Vertex_0 _S859 = Splat_2D_Vertex_x24_syn_dzero_0();

#line 67
    if(_S825)
    {

#line 67
        Splat_2D_Vertex_0 _S860;

#line 67
        float3  _S861;

#line 67
        if(_bflag_1)
        {

#line 67
            if(_bflag_2)
            {

#line 67
                float _S862;

#line 67
                float _S863;

#line 67
                float _S864;

#line 67
                if(_bflag_3)
                {

#line 67
                    float3  _S865 = make_float3 (AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S828), AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S827), AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S826));

#line 103
                    float _S866 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S829);

#line 102
                    float _S867 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S830);

#line 1751 "core.meta.slang"
                    float2  _S868 = make_float2 (0.0f);

#line 1751
                    float2  _S869 = _S868;

#line 1751
                    *&((&_S869)->y) = _S866;

#line 1751
                    *&((&_S869)->x) = _S867;

#line 101 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
                    float _S870 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S831);

#line 100
                    float _S871 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S832);

#line 100
                    float2  _S872 = _S868;

#line 100
                    *&((&_S872)->y) = _S870;

#line 100
                    *&((&_S872)->x) = _S871;

#line 99
                    float _S873 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S826);

#line 98
                    float _S874 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S827);

#line 97
                    float _S875 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S828);

#line 88
                    Matrix<float, 2, 2>  _S876 = _S822;

#line 88
                    _S876[int(1)] = _S869;

#line 88
                    _S876[int(0)] = _S872;

#line 88
                    Matrix<float, 2, 2>  _S877 = _S876 / _S833;

#line 88
                    Matrix<float, 2, 2>  _S878 = _S834 * - _S877;

#line 88
                    Matrix<float, 2, 2>  _S879 = _S835 * _S877;

#line 88
                    float _S880 = - _S879.rows[int(1)].x;

#line 88
                    float _S881 = - _S879.rows[int(0)].y;

#line 88
                    float2  _S882 = _S868;

#line 88
                    *&((&_S882)->x) = _S879.rows[int(1)].y;

#line 88
                    *&((&_S882)->y) = _S881;

#line 88
                    float2  _S883 = _S868;

#line 88
                    *&((&_S883)->x) = _S880;

#line 88
                    *&((&_S883)->y) = _S879.rows[int(0)].x;

#line 67
                    Splat_2D_Vertex_0 _S884 = _S859;

#line 67
                    (&_S884)->rgb_0 = _S865;

#line 67
                    Splat_2D_Vertex_0 _S885 = Splat_2D_Vertex_x24_syn_dadd_0(_S859, _S884);

#line 67
                    Matrix<float, 2, 2>  _S886 = _S822;

#line 67
                    _S886[int(0)] = _S882;

#line 67
                    _S886[int(1)] = _S883;

#line 67
                    _S862 = _S874;

#line 67
                    _S863 = _S875;

#line 67
                    _S833 = _S878;

#line 67
                    _S834 = _S886;

#line 67
                    _S860 = _S885;

#line 67
                    _S864 = _S873;

#line 67
                }
                else
                {

#line 67
                    _S862 = 0.0f;

#line 67
                    _S863 = 0.0f;

#line 67
                    _S833 = _S822;

#line 67
                    _S834 = _S822;

#line 67
                    _S860 = _S859;

#line 67
                    _S864 = 0.0f;

#line 67
                }

#line 79
                DiffPair_float_0 _S887;

#line 79
                (&_S887)->primal_1 = _S837;

#line 79
                (&_S887)->differential_0 = 0.0f;

#line 79
                s_bwd_prop_ndc2pix_0(&_S887, _S839, 0.0f);

#line 79
                float _S888 = _S887.differential_0 + _S862;

#line 79
                DiffPair_float_0 _S889;

#line 79
                (&_S889)->primal_1 = _S838;

#line 79
                (&_S889)->differential_0 = 0.0f;

#line 79
                s_bwd_prop_ndc2pix_0(&_S889, _S840, 0.0f);

#line 79
                float3  _S890 = make_float3 (_S889.differential_0 + _S863, _S888, 0.0f);

#line 79
                _S837 = _S864;

#line 79
                _S861 = _S890;

#line 79
            }
            else
            {

#line 79
                _S833 = _S822;

#line 79
                _S834 = _S822;

#line 79
                _S860 = _S859;

#line 79
                _S837 = 0.0f;

#line 79
                _S861 = _S858;

#line 79
            }

#line 72
            float _S891 = _S833.rows[int(0)].x + _S833.rows[int(0)].y + _S833.rows[int(1)].x + _S833.rows[int(1)].y;

#line 72
            DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S892;

#line 72
            (&_S892)->primal_1 = _S836;

#line 72
            (&_S892)->differential_0 = _S822;

#line 72
            s_bwd_prop_compute_det_0(&_S892, _S891);

#line 72
            Matrix<float, 2, 2>  _S893 = _S892.differential_0 + _S834;

#line 67
            Splat_2D_Vertex_0 _S894 = _S859;

#line 67
            (&_S894)->cov_vs_0 = _S893;

#line 67
            _S860 = Splat_2D_Vertex_x24_syn_dadd_0(_S860, _S894);

#line 67
        }
        else
        {

#line 67
            _S837 = 0.0f;

#line 67
            _S861 = _S858;

#line 67
            _S860 = _S859;

#line 67
        }

#line 67
        float3  _S895 = _S861 + make_float3 (0.0f, 0.0f, _S837);

#line 67
        Splat_2D_Vertex_0 _S896 = _S859;

#line 67
        (&_S896)->xyz_vs_0 = _S895;

#line 67
        Splat_2D_Vertex_0 _S897 = Splat_2D_Vertex_x24_syn_dadd_0(_S860, _S896);

#line 67
        Gaussian_3D_0 _S898 = Gaussian_3D_x24_syn_dzero_0();

#line 67
        DiffPair_Gaussian_3D_0 _S899;

#line 67
        (&_S899)->primal_1 = _s_diff_ctx_1._S112;

#line 67
        (&_S899)->differential_0 = _S898;

#line 67
        Camera_Differential_0 _S900 = Camera_x24_syn_dzero_0();

#line 67
        DiffPair_Camera_0 _S901;

#line 67
        (&_S901)->primal_1 = _s_diff_ctx_1._S111;

#line 67
        (&_S901)->differential_0 = _S900;

#line 67
        s_bwd_prop_project_gaussian_to_camera_0(&_S899, &_S901, active_sh_13, _S897);

#line 67
        s_bwd_prop_load_gaussian_0(_S841, xyz_ws_7, sh_coeffs_8, rotations_5, scales_5, active_sh_13, _S899.differential_0);

#line 67
    }

#line 37
    return;
}


#line 37
__device__ void s_bwd_vertex_shader_0(DiffTensorView_0 _S902, DiffTensorView_0 _S903, DiffTensorView_0 _S904, DiffTensorView_0 _S905, uint _S906, TensorView _S907, TensorView _S908, TensorView _S909, TensorView _S910, TensorView _S911, TensorView _S912, DiffTensorView_0 _S913, DiffTensorView_0 _S914, DiffTensorView_0 _S915, float _S916, float _S917, uint _S918, uint _S919, uint _S920, uint _S921, uint _S922, uint _S923)
{

#line 58
    s_bwd_prop_vertex_shader_Intermediates_0 _S924;

#line 58
    s_primal_ctx_vertex_shader_0(_S902, _S903, _S904, _S905, _S906, _S907, _S908, _S909, _S910, _S911, _S912, _S913, _S914, _S915, _S916, _S917, _S918, _S919, _S920, _S921, _S922, _S923, &_S924);

#line 58
    s_bwd_prop_vertex_shader_0(_S902, _S903, _S904, _S905, _S906, _S907, _S908, _S909, _S910, _S911, _S912, _S913, _S914, _S915, _S916, _S917, _S918, _S919, _S920, _S921, _S922, _S923, _S924);

#line 58
    return;
}


#line 58
extern "C" {
__global__ void __kernel__vertex_shader_bwd_diff(DiffTensorView_0 xyz_ws_8, DiffTensorView_0 sh_coeffs_9, DiffTensorView_0 rotations_6, DiffTensorView_0 scales_6, uint active_sh_14, TensorView world_view_transform_5, TensorView proj_mat_5, TensorView cam_pos_3, TensorView out_tiles_touched_2, TensorView out_rect_tile_space_2, TensorView out_radii_2, DiffTensorView_0 out_xyz_vs_2, DiffTensorView_0 out_inv_cov_vs_2, DiffTensorView_0 out_rgb_2, float fovy_5, float fovx_5, uint image_height_2, uint image_width_2, uint grid_height_3, uint grid_width_3, uint tile_height_3, uint tile_width_3)
{

#line 58
    s_bwd_vertex_shader_0(xyz_ws_8, sh_coeffs_9, rotations_6, scales_6, active_sh_14, world_view_transform_5, proj_mat_5, cam_pos_3, out_tiles_touched_2, out_rect_tile_space_2, out_radii_2, out_xyz_vs_2, out_inv_cov_vs_2, out_rgb_2, fovy_5, fovx_5, image_height_2, image_width_2, grid_height_3, grid_width_3, tile_height_3, tile_width_3);

#line 58
    return;
}

}

#line 177 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_read_t3_float3_0(uint idx_6, DiffTensorView_0 t3_3)
{

#line 28
    uint2  _S925 = make_uint2 (idx_6, 0U);

#line 28
    float _S926 = ((t3_3.primal_0).load<float>((_S925)));

#line 28
    float _S927 = AtomicAdd_load_forward_0(t3_3.diff_1, _S925);
    uint2  _S928 = make_uint2 (idx_6, 1U);

#line 28
    float _S929 = ((t3_3.primal_0).load<float>((_S928)));

#line 28
    float _S930 = AtomicAdd_load_forward_0(t3_3.diff_1, _S928);

    uint2  _S931 = make_uint2 (idx_6, 2U);

#line 28
    float _S932 = ((t3_3.primal_0).load<float>((_S931)));

#line 28
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S933 = { make_float3 (_S926, _S929, _S932), make_float3 (_S927, _S930, AtomicAdd_load_forward_0(t3_3.diff_1, _S931)) };

#line 28
    return _S933;
}


#line 178
__device__ DiffPair_SpherHarmCoeffs_0 s_fwd_read_spherical_harmonics_coeffs_0(uint g_idx_8, DiffTensorView_0 sh_coeffs_10, uint active_sh_15)
{

#line 64 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
    float3  _S934 = make_float3 (0.0f);
    uint3  _S935 = make_uint3 (g_idx_8, 0U, 0U);

#line 65
    float _S936 = ((sh_coeffs_10.primal_0).load<float>((_S935)));

#line 65
    float _S937 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S935);

#line 65
    uint3  _S938 = make_uint3 (g_idx_8, 0U, 1U);

#line 65
    float _S939 = ((sh_coeffs_10.primal_0).load<float>((_S938)));

#line 65
    float _S940 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S938);

#line 65
    uint3  _S941 = make_uint3 (g_idx_8, 0U, 2U);

#line 65
    float _S942 = ((sh_coeffs_10.primal_0).load<float>((_S941)));

#line 65
    float3  _S943 = make_float3 (_S936, _S939, _S942);

#line 65
    float3  _S944 = make_float3 (_S937, _S940, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S941));

#line 65
    SpherHarmCoeffs_0 g_sh_coeffs_2;

#line 65
    SpherHarmCoeffs_0 s_diff_g_sh_coeffs_0;

    if(active_sh_15 > 0U)
    {

#line 68
        uint3  _S945 = make_uint3 (g_idx_8, 1U, 0U);

#line 68
        float _S946 = ((sh_coeffs_10.primal_0).load<float>((_S945)));

#line 68
        float _S947 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S945);

#line 68
        uint3  _S948 = make_uint3 (g_idx_8, 1U, 1U);

#line 68
        float _S949 = ((sh_coeffs_10.primal_0).load<float>((_S948)));

#line 68
        float _S950 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S948);

#line 68
        uint3  _S951 = make_uint3 (g_idx_8, 1U, 2U);

#line 68
        float _S952 = ((sh_coeffs_10.primal_0).load<float>((_S951)));

#line 68
        float3  _S953 = make_float3 (_S946, _S949, _S952);

#line 68
        float3  _S954 = make_float3 (_S947, _S950, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S951));
        uint3  _S955 = make_uint3 (g_idx_8, 2U, 0U);

#line 69
        float _S956 = ((sh_coeffs_10.primal_0).load<float>((_S955)));

#line 69
        float _S957 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S955);

#line 69
        uint3  _S958 = make_uint3 (g_idx_8, 2U, 1U);

#line 69
        float _S959 = ((sh_coeffs_10.primal_0).load<float>((_S958)));

#line 69
        float _S960 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S958);

#line 69
        uint3  _S961 = make_uint3 (g_idx_8, 2U, 2U);

#line 69
        float _S962 = ((sh_coeffs_10.primal_0).load<float>((_S961)));

#line 69
        float3  _S963 = make_float3 (_S956, _S959, _S962);

#line 69
        float3  _S964 = make_float3 (_S957, _S960, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S961));
        uint3  _S965 = make_uint3 (g_idx_8, 3U, 0U);

#line 70
        float _S966 = ((sh_coeffs_10.primal_0).load<float>((_S965)));

#line 70
        float _S967 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S965);

#line 70
        uint3  _S968 = make_uint3 (g_idx_8, 3U, 1U);

#line 70
        float _S969 = ((sh_coeffs_10.primal_0).load<float>((_S968)));

#line 70
        float _S970 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S968);

#line 70
        uint3  _S971 = make_uint3 (g_idx_8, 3U, 2U);

#line 70
        float _S972 = ((sh_coeffs_10.primal_0).load<float>((_S971)));

#line 70
        float3  _S973 = make_float3 (_S966, _S969, _S972);

#line 70
        float3  _S974 = make_float3 (_S967, _S970, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S971));

        if(active_sh_15 > 1U)
        {

#line 73
            uint3  _S975 = make_uint3 (g_idx_8, 4U, 0U);

#line 73
            float _S976 = ((sh_coeffs_10.primal_0).load<float>((_S975)));

#line 73
            float _S977 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S975);

#line 73
            uint3  _S978 = make_uint3 (g_idx_8, 4U, 1U);

#line 73
            float _S979 = ((sh_coeffs_10.primal_0).load<float>((_S978)));

#line 73
            float _S980 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S978);

#line 73
            uint3  _S981 = make_uint3 (g_idx_8, 4U, 2U);

#line 73
            float _S982 = ((sh_coeffs_10.primal_0).load<float>((_S981)));

#line 73
            float3  _S983 = make_float3 (_S976, _S979, _S982);

#line 73
            float3  _S984 = make_float3 (_S977, _S980, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S981));
            uint3  _S985 = make_uint3 (g_idx_8, 5U, 0U);

#line 74
            float _S986 = ((sh_coeffs_10.primal_0).load<float>((_S985)));

#line 74
            float _S987 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S985);

#line 74
            uint3  _S988 = make_uint3 (g_idx_8, 5U, 1U);

#line 74
            float _S989 = ((sh_coeffs_10.primal_0).load<float>((_S988)));

#line 74
            float _S990 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S988);

#line 74
            uint3  _S991 = make_uint3 (g_idx_8, 5U, 2U);

#line 74
            float _S992 = ((sh_coeffs_10.primal_0).load<float>((_S991)));

#line 74
            float3  _S993 = make_float3 (_S986, _S989, _S992);

#line 74
            float3  _S994 = make_float3 (_S987, _S990, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S991));
            uint3  _S995 = make_uint3 (g_idx_8, 6U, 0U);

#line 75
            float _S996 = ((sh_coeffs_10.primal_0).load<float>((_S995)));

#line 75
            float _S997 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S995);

#line 75
            uint3  _S998 = make_uint3 (g_idx_8, 6U, 1U);

#line 75
            float _S999 = ((sh_coeffs_10.primal_0).load<float>((_S998)));

#line 75
            float _S1000 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S998);

#line 75
            uint3  _S1001 = make_uint3 (g_idx_8, 6U, 2U);

#line 75
            float _S1002 = ((sh_coeffs_10.primal_0).load<float>((_S1001)));

#line 75
            float3  _S1003 = make_float3 (_S996, _S999, _S1002);

#line 75
            float3  _S1004 = make_float3 (_S997, _S1000, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1001));
            uint3  _S1005 = make_uint3 (g_idx_8, 7U, 0U);

#line 76
            float _S1006 = ((sh_coeffs_10.primal_0).load<float>((_S1005)));

#line 76
            float _S1007 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1005);

#line 76
            uint3  _S1008 = make_uint3 (g_idx_8, 7U, 1U);

#line 76
            float _S1009 = ((sh_coeffs_10.primal_0).load<float>((_S1008)));

#line 76
            float _S1010 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1008);

#line 76
            uint3  _S1011 = make_uint3 (g_idx_8, 7U, 2U);

#line 76
            float _S1012 = ((sh_coeffs_10.primal_0).load<float>((_S1011)));

#line 76
            float3  _S1013 = make_float3 (_S1006, _S1009, _S1012);

#line 76
            float3  _S1014 = make_float3 (_S1007, _S1010, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1011));
            uint3  _S1015 = make_uint3 (g_idx_8, 8U, 0U);

#line 77
            float _S1016 = ((sh_coeffs_10.primal_0).load<float>((_S1015)));

#line 77
            float _S1017 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1015);

#line 77
            uint3  _S1018 = make_uint3 (g_idx_8, 8U, 1U);

#line 77
            float _S1019 = ((sh_coeffs_10.primal_0).load<float>((_S1018)));

#line 77
            float _S1020 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1018);

#line 77
            uint3  _S1021 = make_uint3 (g_idx_8, 8U, 2U);

#line 77
            float _S1022 = ((sh_coeffs_10.primal_0).load<float>((_S1021)));

#line 77
            float3  _S1023 = make_float3 (_S1016, _S1019, _S1022);

#line 77
            float3  _S1024 = make_float3 (_S1017, _S1020, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1021));

            if(active_sh_15 > 2U)
            {

#line 80
                uint3  _S1025 = make_uint3 (g_idx_8, 9U, 0U);

#line 80
                float _S1026 = ((sh_coeffs_10.primal_0).load<float>((_S1025)));

#line 80
                float _S1027 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1025);

#line 80
                uint3  _S1028 = make_uint3 (g_idx_8, 9U, 1U);

#line 80
                float _S1029 = ((sh_coeffs_10.primal_0).load<float>((_S1028)));

#line 80
                float _S1030 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1028);

#line 80
                uint3  _S1031 = make_uint3 (g_idx_8, 9U, 2U);

#line 80
                float _S1032 = ((sh_coeffs_10.primal_0).load<float>((_S1031)));

#line 80
                float3  _S1033 = make_float3 (_S1026, _S1029, _S1032);

#line 80
                float3  _S1034 = make_float3 (_S1027, _S1030, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1031));
                uint3  _S1035 = make_uint3 (g_idx_8, 10U, 0U);

#line 81
                float _S1036 = ((sh_coeffs_10.primal_0).load<float>((_S1035)));

#line 81
                float _S1037 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1035);

#line 81
                uint3  _S1038 = make_uint3 (g_idx_8, 10U, 1U);

#line 81
                float _S1039 = ((sh_coeffs_10.primal_0).load<float>((_S1038)));

#line 81
                float _S1040 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1038);

#line 81
                uint3  _S1041 = make_uint3 (g_idx_8, 10U, 2U);

#line 81
                float _S1042 = ((sh_coeffs_10.primal_0).load<float>((_S1041)));

#line 81
                float3  _S1043 = make_float3 (_S1036, _S1039, _S1042);

#line 81
                float3  _S1044 = make_float3 (_S1037, _S1040, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1041));
                uint3  _S1045 = make_uint3 (g_idx_8, 11U, 0U);

#line 82
                float _S1046 = ((sh_coeffs_10.primal_0).load<float>((_S1045)));

#line 82
                float _S1047 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1045);

#line 82
                uint3  _S1048 = make_uint3 (g_idx_8, 11U, 1U);

#line 82
                float _S1049 = ((sh_coeffs_10.primal_0).load<float>((_S1048)));

#line 82
                float _S1050 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1048);

#line 82
                uint3  _S1051 = make_uint3 (g_idx_8, 11U, 2U);

#line 82
                float _S1052 = ((sh_coeffs_10.primal_0).load<float>((_S1051)));

#line 82
                float3  _S1053 = make_float3 (_S1046, _S1049, _S1052);

#line 82
                float3  _S1054 = make_float3 (_S1047, _S1050, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1051));
                uint3  _S1055 = make_uint3 (g_idx_8, 12U, 0U);

#line 83
                float _S1056 = ((sh_coeffs_10.primal_0).load<float>((_S1055)));

#line 83
                float _S1057 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1055);

#line 83
                uint3  _S1058 = make_uint3 (g_idx_8, 12U, 1U);

#line 83
                float _S1059 = ((sh_coeffs_10.primal_0).load<float>((_S1058)));

#line 83
                float _S1060 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1058);

#line 83
                uint3  _S1061 = make_uint3 (g_idx_8, 12U, 2U);

#line 83
                float _S1062 = ((sh_coeffs_10.primal_0).load<float>((_S1061)));

#line 83
                float3  _S1063 = make_float3 (_S1056, _S1059, _S1062);

#line 83
                float3  _S1064 = make_float3 (_S1057, _S1060, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1061));
                uint3  _S1065 = make_uint3 (g_idx_8, 13U, 0U);

#line 84
                float _S1066 = ((sh_coeffs_10.primal_0).load<float>((_S1065)));

#line 84
                float _S1067 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1065);

#line 84
                uint3  _S1068 = make_uint3 (g_idx_8, 13U, 1U);

#line 84
                float _S1069 = ((sh_coeffs_10.primal_0).load<float>((_S1068)));

#line 84
                float _S1070 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1068);

#line 84
                uint3  _S1071 = make_uint3 (g_idx_8, 13U, 2U);

#line 84
                float _S1072 = ((sh_coeffs_10.primal_0).load<float>((_S1071)));

#line 84
                float3  _S1073 = make_float3 (_S1066, _S1069, _S1072);

#line 84
                float3  _S1074 = make_float3 (_S1067, _S1070, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1071));
                uint3  _S1075 = make_uint3 (g_idx_8, 14U, 0U);

#line 85
                float _S1076 = ((sh_coeffs_10.primal_0).load<float>((_S1075)));

#line 85
                float _S1077 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1075);

#line 85
                uint3  _S1078 = make_uint3 (g_idx_8, 14U, 1U);

#line 85
                float _S1079 = ((sh_coeffs_10.primal_0).load<float>((_S1078)));

#line 85
                float _S1080 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1078);

#line 85
                uint3  _S1081 = make_uint3 (g_idx_8, 14U, 2U);

#line 85
                float _S1082 = ((sh_coeffs_10.primal_0).load<float>((_S1081)));

#line 85
                float3  _S1083 = make_float3 (_S1076, _S1079, _S1082);

#line 85
                float3  _S1084 = make_float3 (_S1077, _S1080, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1081));
                uint3  _S1085 = make_uint3 (g_idx_8, 15U, 0U);

#line 86
                float _S1086 = ((sh_coeffs_10.primal_0).load<float>((_S1085)));

#line 86
                float _S1087 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1085);

#line 86
                uint3  _S1088 = make_uint3 (g_idx_8, 15U, 1U);

#line 86
                float _S1089 = ((sh_coeffs_10.primal_0).load<float>((_S1088)));

#line 86
                float _S1090 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1088);

#line 86
                uint3  _S1091 = make_uint3 (g_idx_8, 15U, 2U);

#line 86
                float _S1092 = ((sh_coeffs_10.primal_0).load<float>((_S1091)));

#line 86
                float3  _S1093 = make_float3 (_S1086, _S1089, _S1092);

#line 86
                float3  _S1094 = make_float3 (_S1087, _S1090, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1091));

#line 86
                (&g_sh_coeffs_2)->coeff0_0 = _S943;

#line 86
                (&g_sh_coeffs_2)->coeff1_0 = _S953;

#line 86
                (&g_sh_coeffs_2)->coeff2_0 = _S963;

#line 86
                (&g_sh_coeffs_2)->coeff3_0 = _S973;

#line 86
                (&g_sh_coeffs_2)->coeff4_0 = _S983;

#line 86
                (&g_sh_coeffs_2)->coeff5_0 = _S993;

#line 86
                (&g_sh_coeffs_2)->coeff6_0 = _S1003;

#line 86
                (&g_sh_coeffs_2)->coeff7_0 = _S1013;

#line 86
                (&g_sh_coeffs_2)->coeff8_0 = _S1023;

#line 86
                (&g_sh_coeffs_2)->coeff9_0 = _S1033;

#line 86
                (&g_sh_coeffs_2)->coeff10_0 = _S1043;

#line 86
                (&g_sh_coeffs_2)->coeff11_0 = _S1053;

#line 86
                (&g_sh_coeffs_2)->coeff12_0 = _S1063;

#line 86
                (&g_sh_coeffs_2)->coeff13_0 = _S1073;

#line 86
                (&g_sh_coeffs_2)->coeff14_0 = _S1083;

#line 86
                (&g_sh_coeffs_2)->coeff15_0 = _S1093;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S944;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S954;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S964;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S974;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S984;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S994;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S1004;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S1014;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S1024;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S1034;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S1044;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S1054;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S1064;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S1074;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S1084;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S1094;

#line 79
            }
            else
            {

#line 79
                (&g_sh_coeffs_2)->coeff0_0 = _S943;

#line 79
                (&g_sh_coeffs_2)->coeff1_0 = _S953;

#line 79
                (&g_sh_coeffs_2)->coeff2_0 = _S963;

#line 79
                (&g_sh_coeffs_2)->coeff3_0 = _S973;

#line 79
                (&g_sh_coeffs_2)->coeff4_0 = _S983;

#line 79
                (&g_sh_coeffs_2)->coeff5_0 = _S993;

#line 79
                (&g_sh_coeffs_2)->coeff6_0 = _S1003;

#line 79
                (&g_sh_coeffs_2)->coeff7_0 = _S1013;

#line 79
                (&g_sh_coeffs_2)->coeff8_0 = _S1023;

#line 79
                (&g_sh_coeffs_2)->coeff9_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff10_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff11_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff12_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff13_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff14_0 = _S934;

#line 79
                (&g_sh_coeffs_2)->coeff15_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S944;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S954;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S974;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S984;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S994;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S1004;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S1014;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S1024;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S934;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S934;

#line 79
            }

#line 72
        }
        else
        {

#line 72
            (&g_sh_coeffs_2)->coeff0_0 = _S943;

#line 72
            (&g_sh_coeffs_2)->coeff1_0 = _S953;

#line 72
            (&g_sh_coeffs_2)->coeff2_0 = _S963;

#line 72
            (&g_sh_coeffs_2)->coeff3_0 = _S973;

#line 72
            (&g_sh_coeffs_2)->coeff4_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff5_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff6_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff7_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff8_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff9_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff10_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff11_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff12_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff13_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff14_0 = _S934;

#line 72
            (&g_sh_coeffs_2)->coeff15_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S944;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S954;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S974;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S934;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S934;

#line 72
        }

#line 67
    }
    else
    {

#line 67
        (&g_sh_coeffs_2)->coeff0_0 = _S943;

#line 67
        (&g_sh_coeffs_2)->coeff1_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff2_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff3_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff4_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff5_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff6_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff7_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff8_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff9_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff10_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff11_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff12_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff13_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff14_0 = _S934;

#line 67
        (&g_sh_coeffs_2)->coeff15_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S944;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S934;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S934;

#line 67
    }

#line 67
    DiffPair_SpherHarmCoeffs_0 _S1095 = { g_sh_coeffs_2, s_diff_g_sh_coeffs_0 };

#line 90
    return _S1095;
}


#line 179 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_read_t4_float4_0(uint idx_7, DiffTensorView_0 t4_3)
{

#line 36
    uint2  _S1096 = make_uint2 (idx_7, 0U);

#line 36
    float _S1097 = ((t4_3.primal_0).load<float>((_S1096)));

#line 36
    float _S1098 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1096);
    uint2  _S1099 = make_uint2 (idx_7, 1U);

#line 36
    float _S1100 = ((t4_3.primal_0).load<float>((_S1099)));

#line 36
    float _S1101 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1099);

    uint2  _S1102 = make_uint2 (idx_7, 2U);

#line 36
    float _S1103 = ((t4_3.primal_0).load<float>((_S1102)));

#line 36
    float _S1104 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1102);


    uint2  _S1105 = make_uint2 (idx_7, 3U);

#line 36
    float _S1106 = ((t4_3.primal_0).load<float>((_S1105)));

#line 36
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1107 = { make_float4 (_S1097, _S1100, _S1103, _S1106), make_float4 (_S1098, _S1101, _S1104, AtomicAdd_load_forward_0(t4_3.diff_1, _S1105)) };

#line 36
    return _S1107;
}


#line 36
__device__ DiffPair_Gaussian_3D_0 s_fwd_load_gaussian_0(int g_idx_9, DiffTensorView_0 xyz_ws_9, DiffTensorView_0 sh_coeffs_11, DiffTensorView_0 rotations_7, DiffTensorView_0 scales_7, uint active_sh_16)
{

#line 177
    uint _S1108 = uint(g_idx_9);

#line 177
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1109 = s_fwd_read_t3_float3_0(_S1108, xyz_ws_9);
    DiffPair_SpherHarmCoeffs_0 _S1110 = s_fwd_read_spherical_harmonics_coeffs_0(_S1108, sh_coeffs_11, active_sh_16);
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1111 = s_fwd_read_t4_float4_0(_S1108, rotations_7);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1112 = s_fwd_read_t3_float3_0(_S1108, scales_7);

    Gaussian_3D_0 _S1113 = { _S1109.primal_1, _S1110.primal_1, _S1111.primal_1, _S1112.primal_1 };

#line 182
    Gaussian_3D_0 _S1114 = { _S1109.differential_0, _S1110.differential_0, _S1111.differential_0, _S1112.differential_0 };

#line 182
    DiffPair_Gaussian_3D_0 _S1115 = { _S1113, _S1114 };

#line 182
    return _S1115;
}


#line 182
struct DiffPair_Splat_2D_Vertex_0
{
    Splat_2D_Vertex_0 primal_1;
    Splat_2D_Vertex_0 differential_0;
};


#line 120
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_geom_transform_points_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_6, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 dptransf_matrix_4)
{

#line 107
    float4  _S1116 = make_float4 (dppoint_6.primal_1.x, dppoint_6.primal_1.y, dppoint_6.primal_1.z, 1.0f);

#line 107
    float4  _S1117 = mul_4(dptransf_matrix_4.primal_1, _S1116);

#line 107
    float4  _S1118 = mul_4(dptransf_matrix_4.differential_0, _S1116) + mul_4(dptransf_matrix_4.primal_1, make_float4 (dppoint_6.differential_0.x, dppoint_6.differential_0.y, dppoint_6.differential_0.z, 0.0f));
    float3  _S1119 = float3 {_S1117.x, _S1117.y, _S1117.z};

#line 108
    float _S1120 = _S1117.w + 1.00000001168609742e-07f;

#line 108
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1121 = { _S1119 / make_float3 (_S1120), (float3 {_S1118.x, _S1118.y, _S1118.z} * make_float3 (_S1120) - _S1119 * make_float3 (_S1118.w)) / make_float3 (_S1120 * _S1120) };

#line 108
    return _S1121;
}


#line 108
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_geom_transform_points2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_7, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 dptransf_matrix_5)
{

#line 114
    float4  _S1122 = make_float4 (dppoint_7.primal_1.x, dppoint_7.primal_1.y, dppoint_7.primal_1.z, 1.0f);

#line 114
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1123 = { float3 {mul_4(dptransf_matrix_5.primal_1, _S1122).x, mul_4(dptransf_matrix_5.primal_1, _S1122).y, mul_4(dptransf_matrix_5.primal_1, _S1122).z}, float3 {(mul_4(dptransf_matrix_5.differential_0, _S1122) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).x, (mul_4(dptransf_matrix_5.differential_0, _S1122) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).y, (mul_4(dptransf_matrix_5.differential_0, _S1122) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).z} };
    return _S1123;
}


#line 115
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_project_point_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_8, DiffPair_Camera_0 dpcam_8)
{


    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1124 = { dppoint_8.primal_1, dppoint_8.differential_0 };

#line 119
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1125 = { mul_2(dpcam_8.primal_1.proj_mat_1, dpcam_8.primal_1.world_view_transform_1), mul_2(dpcam_8.differential_0.proj_mat_0, dpcam_8.primal_1.world_view_transform_1) + mul_2(dpcam_8.primal_1.proj_mat_1, dpcam_8.differential_0.world_view_transform_0) };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1126 = s_fwd_geom_transform_points_0(_S1124, _S1125);

#line 120
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1127 = { dpcam_8.primal_1.world_view_transform_1, dpcam_8.differential_0.world_view_transform_0 };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1128 = s_fwd_geom_transform_points2_0(_S1124, _S1127);
    float _S1129 = _S1128.primal_1.z;

#line 122
    float _S1130 = _S1128.differential_0.z;

#line 122
    float3  _S1131 = _S1126.primal_1;

#line 122
    *&((&_S1131)->z) = _S1129;

#line 122
    float3  _S1132 = _S1126.differential_0;

#line 122
    *&((&_S1132)->z) = _S1130;

#line 122
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1133 = { _S1131, _S1132 };
    return _S1133;
}


#line 2107 "diff.meta.slang"
__device__ DiffPair_float_0 s_fwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_12)
{

#line 2092
    float _S1134 = dpx_12.primal_1.x;

#line 2092
    float _S1135 = dpx_12.differential_0.x * dpx_12.primal_1.x;

#line 2092
    float _S1136 = dpx_12.primal_1.y;

#line 2092
    float _S1137 = dpx_12.differential_0.y * dpx_12.primal_1.y;

#line 2092
    float _S1138 = dpx_12.primal_1.z;

#line 2092
    float _S1139 = dpx_12.differential_0.z * dpx_12.primal_1.z;

#line 2092
    DiffPair_float_0 _S1140 = { _S1134 * _S1134 + _S1136 * _S1136 + _S1138 * _S1138, _S1135 + _S1135 + (_S1137 + _S1137) + (_S1139 + _S1139) };

#line 2099
    DiffPair_float_0 _S1141 = _d_sqrt_1(_S1140);

#line 2099
    DiffPair_float_0 _S1142 = { _S1141.primal_1, _S1141.differential_0 };

#line 2099
    return _S1142;
}


#line 2099
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_13)
{

#line 2154
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1143 = { dpx_13.primal_1, dpx_13.differential_0 };

    DiffPair_float_0 _S1144 = s_fwd_length_impl_0(_S1143);

#line 2156
    float _S1145 = 1.0f / _S1144.primal_1;

#line 2156
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1146 = { dpx_13.primal_1 * make_float3 (_S1145), dpx_13.differential_0 * make_float3 (_S1145) + make_float3 ((0.0f - _S1144.differential_0) / (_S1144.primal_1 * _S1144.primal_1)) * dpx_13.primal_1 };
    return _S1146;
}


#line 2157
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_compute_color_from_sh_coeffs_0(DiffPair_SpherHarmCoeffs_0 dpsh_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dpg_xyz_ws_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dpcam_pos_2, uint active_sh_17)
{

#line 94 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/spherical_harmonics.slang"
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1147 = { dpg_xyz_ws_2.primal_1 - dpcam_pos_2.primal_1, dpg_xyz_ws_2.differential_0 - dpcam_pos_2.differential_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1148 = s_fwd_normalize_impl_0(_S1147);

    float3  rgb_14 = make_float3 (0.282094806432724f) * dpsh_2.primal_1.coeff0_0;

#line 98
    float3  _S1149 = dpsh_2.differential_0.coeff0_0 * make_float3 (0.282094806432724f);

#line 98
    float3  rgb_15;

#line 98
    float3  s_diff_rgb_0;
    if(active_sh_17 > 0U)
    {

#line 100
        float _S1150 = _S1148.primal_1.y;

#line 100
        float _S1151 = _S1148.differential_0.y;

#line 100
        float _S1152 = 0.48860251903533936f * _S1150;

#line 100
        float _S1153 = _S1148.primal_1.z;

#line 100
        float _S1154 = _S1148.differential_0.z;

#line 100
        float _S1155 = 0.48860251903533936f * _S1153;

#line 100
        float _S1156 = _S1148.primal_1.x;

#line 100
        float _S1157 = _S1148.differential_0.x;

#line 100
        float _S1158 = 0.48860251903533936f * _S1156;

#line 100
        float3  rgb_16 = rgb_14 - make_float3 (_S1152) * dpsh_2.primal_1.coeff1_0 + make_float3 (_S1155) * dpsh_2.primal_1.coeff2_0 - make_float3 (_S1158) * dpsh_2.primal_1.coeff3_0;

#line 100
        float3  s_diff_rgb_1 = _S1149 - (make_float3 (_S1151 * 0.48860251903533936f) * dpsh_2.primal_1.coeff1_0 + dpsh_2.differential_0.coeff1_0 * make_float3 (_S1152)) + (make_float3 (_S1154 * 0.48860251903533936f) * dpsh_2.primal_1.coeff2_0 + dpsh_2.differential_0.coeff2_0 * make_float3 (_S1155)) - (make_float3 (_S1157 * 0.48860251903533936f) * dpsh_2.primal_1.coeff3_0 + dpsh_2.differential_0.coeff3_0 * make_float3 (_S1158));
        if(active_sh_17 > 1U)
        {
            float xx_3 = _S1156 * _S1156;

#line 103
            float _S1159 = _S1157 * _S1156;

#line 103
            float s_diff_xx_0 = _S1159 + _S1159;

#line 103
            float yy_3 = _S1150 * _S1150;

#line 103
            float _S1160 = _S1151 * _S1150;

#line 103
            float s_diff_yy_0 = _S1160 + _S1160;

#line 103
            float zz_3 = _S1153 * _S1153;

#line 103
            float _S1161 = _S1154 * _S1153;

#line 103
            float s_diff_zz_0 = _S1161 + _S1161;
            float xy_3 = _S1156 * _S1150;

#line 104
            float s_diff_xy_0 = _S1157 * _S1150 + _S1151 * _S1156;

            float _S1162 = 1.09254848957061768f * xy_3;
            float _S1163 = -1.09254848957061768f * (_S1150 * _S1153);
            float _S1164 = 2.0f * zz_3;

#line 108
            float _S1165 = s_diff_zz_0 * 2.0f;

#line 108
            float _S1166 = 0.31539157032966614f * (_S1164 - xx_3 - yy_3);
            float _S1167 = -1.09254848957061768f * (_S1156 * _S1153);
            float _S1168 = xx_3 - yy_3;

#line 110
            float _S1169 = s_diff_xx_0 - s_diff_yy_0;

#line 110
            float _S1170 = 0.54627424478530884f * _S1168;

#line 109
            float3  rgb_17 = rgb_16 + make_float3 (_S1162) * dpsh_2.primal_1.coeff4_0 + make_float3 (_S1163) * dpsh_2.primal_1.coeff5_0 + make_float3 (_S1166) * dpsh_2.primal_1.coeff6_0 + make_float3 (_S1167) * dpsh_2.primal_1.coeff7_0 + make_float3 (_S1170) * dpsh_2.primal_1.coeff8_0;

#line 109
            float3  s_diff_rgb_2 = s_diff_rgb_1 + (make_float3 (s_diff_xy_0 * 1.09254848957061768f) * dpsh_2.primal_1.coeff4_0 + dpsh_2.differential_0.coeff4_0 * make_float3 (_S1162)) + (make_float3 ((_S1151 * _S1153 + _S1154 * _S1150) * -1.09254848957061768f) * dpsh_2.primal_1.coeff5_0 + dpsh_2.differential_0.coeff5_0 * make_float3 (_S1163)) + (make_float3 ((_S1165 - s_diff_xx_0 - s_diff_yy_0) * 0.31539157032966614f) * dpsh_2.primal_1.coeff6_0 + dpsh_2.differential_0.coeff6_0 * make_float3 (_S1166)) + (make_float3 ((_S1157 * _S1153 + _S1154 * _S1156) * -1.09254848957061768f) * dpsh_2.primal_1.coeff7_0 + dpsh_2.differential_0.coeff7_0 * make_float3 (_S1167)) + (make_float3 (_S1169 * 0.54627424478530884f) * dpsh_2.primal_1.coeff8_0 + dpsh_2.differential_0.coeff8_0 * make_float3 (_S1170));


            if(active_sh_17 > 2U)
            {

                float _S1171 = -0.59004360437393188f * _S1150;

#line 115
                float _S1172 = 3.0f * xx_3;

#line 115
                float _S1173 = s_diff_xx_0 * 3.0f;

#line 115
                float _S1174 = _S1172 - yy_3;

#line 115
                float _S1175 = _S1171 * _S1174;
                float _S1176 = 2.89061141014099121f * xy_3;

#line 116
                float _S1177 = _S1176 * _S1153;
                float _S1178 = -0.4570457935333252f * _S1150;

#line 117
                float _S1179 = 4.0f * zz_3 - xx_3 - yy_3;

#line 117
                float _S1180 = s_diff_zz_0 * 4.0f - s_diff_xx_0 - s_diff_yy_0;

#line 117
                float _S1181 = _S1178 * _S1179;
                float _S1182 = 0.37317633628845215f * _S1153;

#line 118
                float _S1183 = 3.0f * yy_3;

#line 118
                float _S1184 = s_diff_yy_0 * 3.0f;

#line 118
                float _S1185 = _S1164 - _S1172 - _S1183;

#line 118
                float _S1186 = _S1182 * _S1185;
                float _S1187 = -0.4570457935333252f * _S1156;

#line 119
                float _S1188 = _S1187 * _S1179;
                float _S1189 = 1.44530570507049561f * _S1153;

#line 120
                float _S1190 = _S1189 * _S1168;
                float _S1191 = -0.59004360437393188f * _S1156;

#line 121
                float _S1192 = xx_3 - _S1183;

#line 121
                float _S1193 = _S1191 * _S1192;

#line 120
                float3  _S1194 = s_diff_rgb_2 + (make_float3 (_S1151 * -0.59004360437393188f * _S1174 + (_S1173 - s_diff_yy_0) * _S1171) * dpsh_2.primal_1.coeff9_0 + dpsh_2.differential_0.coeff9_0 * make_float3 (_S1175)) + (make_float3 (s_diff_xy_0 * 2.89061141014099121f * _S1153 + _S1154 * _S1176) * dpsh_2.primal_1.coeff10_0 + dpsh_2.differential_0.coeff10_0 * make_float3 (_S1177)) + (make_float3 (_S1151 * -0.4570457935333252f * _S1179 + _S1180 * _S1178) * dpsh_2.primal_1.coeff11_0 + dpsh_2.differential_0.coeff11_0 * make_float3 (_S1181)) + (make_float3 (_S1154 * 0.37317633628845215f * _S1185 + (_S1165 - _S1173 - _S1184) * _S1182) * dpsh_2.primal_1.coeff12_0 + dpsh_2.differential_0.coeff12_0 * make_float3 (_S1186)) + (make_float3 (_S1157 * -0.4570457935333252f * _S1179 + _S1180 * _S1187) * dpsh_2.primal_1.coeff13_0 + dpsh_2.differential_0.coeff13_0 * make_float3 (_S1188)) + (make_float3 (_S1154 * 1.44530570507049561f * _S1168 + _S1169 * _S1189) * dpsh_2.primal_1.coeff14_0 + dpsh_2.differential_0.coeff14_0 * make_float3 (_S1190)) + (make_float3 (_S1157 * -0.59004360437393188f * _S1192 + (s_diff_xx_0 - _S1184) * _S1191) * dpsh_2.primal_1.coeff15_0 + dpsh_2.differential_0.coeff15_0 * make_float3 (_S1193));

#line 120
                rgb_15 = rgb_17 + make_float3 (_S1175) * dpsh_2.primal_1.coeff9_0 + make_float3 (_S1177) * dpsh_2.primal_1.coeff10_0 + make_float3 (_S1181) * dpsh_2.primal_1.coeff11_0 + make_float3 (_S1186) * dpsh_2.primal_1.coeff12_0 + make_float3 (_S1188) * dpsh_2.primal_1.coeff13_0 + make_float3 (_S1190) * dpsh_2.primal_1.coeff14_0 + make_float3 (_S1193) * dpsh_2.primal_1.coeff15_0;

#line 120
                s_diff_rgb_0 = _S1194;

#line 112
            }
            else
            {

#line 112
                rgb_15 = rgb_17;

#line 112
                s_diff_rgb_0 = s_diff_rgb_2;

#line 112
            }

#line 101
        }
        else
        {

#line 101
            rgb_15 = rgb_16;

#line 101
            s_diff_rgb_0 = s_diff_rgb_1;

#line 101
        }

#line 99
    }
    else
    {

#line 99
        rgb_15 = rgb_14;

#line 99
        s_diff_rgb_0 = _S1149;

#line 99
    }

#line 99
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1195 = { rgb_15 + make_float3 (0.5f), s_diff_rgb_0 };

#line 99
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1196 = { make_float3 (0.0f), make_float3 (0.0f) };

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1197 = _d_max_vector_1(_S1195, _S1196);

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1198 = { _S1197.primal_1, _S1197.differential_0 };

#line 128
    return _S1198;
}


#line 228 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
__device__ DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 s_fwd_get_covariance_from_quat_scales_0(DiffPair_vectorx3Cfloatx2C4x3E_0 dpq_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dps_2)
{

#line 280
    float _S1199 = dpq_2.primal_1.z;



    float _S1200 = _S1199 * _S1199;

#line 284
    float _S1201 = dpq_2.differential_0.z * dpq_2.primal_1.z;

#line 284
    float _S1202 = _S1201 + _S1201;

#line 284
    float _S1203 = dpq_2.primal_1.w * dpq_2.primal_1.w;

#line 284
    float _S1204 = dpq_2.differential_0.w * dpq_2.primal_1.w;

#line 284
    float _S1205 = _S1204 + _S1204;

#line 284
    float _S1206 = dpq_2.primal_1.y * dpq_2.primal_1.z;

#line 284
    float _S1207 = dpq_2.differential_0.y * dpq_2.primal_1.z + dpq_2.differential_0.z * dpq_2.primal_1.y;

#line 284
    float _S1208 = dpq_2.primal_1.x * dpq_2.primal_1.w;

#line 284
    float _S1209 = dpq_2.differential_0.x * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.x;

#line 284
    float _S1210 = dpq_2.primal_1.y * dpq_2.primal_1.w;

#line 284
    float _S1211 = dpq_2.differential_0.y * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.y;

#line 284
    float _S1212 = dpq_2.primal_1.x * dpq_2.primal_1.z;

#line 284
    float _S1213 = dpq_2.differential_0.x * dpq_2.primal_1.z + dpq_2.differential_0.z * dpq_2.primal_1.x;
    float _S1214 = dpq_2.primal_1.y * dpq_2.primal_1.y;

#line 285
    float _S1215 = dpq_2.differential_0.y * dpq_2.primal_1.y;

#line 285
    float _S1216 = _S1215 + _S1215;

#line 285
    float _S1217 = dpq_2.primal_1.z * dpq_2.primal_1.w;

#line 285
    float _S1218 = dpq_2.differential_0.z * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.z;

#line 285
    float _S1219 = dpq_2.primal_1.x * dpq_2.primal_1.y;

#line 285
    float _S1220 = dpq_2.differential_0.x * dpq_2.primal_1.y + dpq_2.differential_0.y * dpq_2.primal_1.x;

#line 283
    Matrix<float, 3, 3>  rotation_matrix_1 = makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S1200 + _S1203), 2.0f * (_S1206 - _S1208), 2.0f * (_S1210 + _S1212), 2.0f * (_S1206 + _S1208), 1.0f - 2.0f * (_S1214 + _S1203), 2.0f * (_S1217 - _S1219), 2.0f * (_S1210 - _S1212), 2.0f * (_S1217 + _S1219), 1.0f - 2.0f * (_S1214 + _S1200));

#line 288
    Matrix<float, 3, 3>  scales_matrix_1 = makeMatrix<float, 3, 3> (dps_2.primal_1.x, 0.0f, 0.0f, 0.0f, dps_2.primal_1.y, 0.0f, 0.0f, 0.0f, dps_2.primal_1.z);



    Matrix<float, 3, 3>  _S1221 = mul_3(rotation_matrix_1, scales_matrix_1);

#line 292
    Matrix<float, 3, 3>  _S1222 = mul_3(makeMatrix<float, 3, 3> (0.0f - (_S1202 + _S1205) * 2.0f, (_S1207 - _S1209) * 2.0f, (_S1211 + _S1213) * 2.0f, (_S1207 + _S1209) * 2.0f, 0.0f - (_S1216 + _S1205) * 2.0f, (_S1218 - _S1220) * 2.0f, (_S1211 - _S1213) * 2.0f, (_S1218 + _S1220) * 2.0f, 0.0f - (_S1216 + _S1202) * 2.0f), scales_matrix_1) + mul_3(rotation_matrix_1, makeMatrix<float, 3, 3> (dps_2.differential_0.x, 0.0f, 0.0f, 0.0f, dps_2.differential_0.y, 0.0f, 0.0f, 0.0f, dps_2.differential_0.z));

    Matrix<float, 3, 3>  _S1223 = transpose_0(_S1221);

#line 294
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1224 = { mul_3(_S1221, _S1223), mul_3(_S1222, _S1223) + mul_3(_S1221, transpose_0(_S1222)) };

#line 294
    return _S1224;
}


#line 153
__device__ DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 s_fwd_compute_jacobian_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpxyz_ws_4, DiffPair_Camera_0 dpcam_9)
{

#line 127
    DiffPair_float_0 _S1225 = { dpcam_9.primal_1.fovx_1 / 2.0f, dpcam_9.differential_0.fovx_0 * 0.5f };
    DiffPair_float_0 _S1226 = _d_tan_1(_S1225);

#line 128
    DiffPair_float_0 _S1227 = { dpcam_9.primal_1.fovy_1 / 2.0f, dpcam_9.differential_0.fovy_0 * 0.5f };
    DiffPair_float_0 _S1228 = _d_tan_1(_S1227);
    float _S1229 = float(dpcam_9.primal_1.W_0);

#line 130
    float _S1230 = 2.0f * _S1226.primal_1;

#line 130
    float h_x_3 = _S1229 / _S1230;

#line 130
    float s_diff_h_x_0 = (0.0f - _S1229 * (_S1226.differential_0 * 2.0f)) / (_S1230 * _S1230);
    float _S1231 = float(dpcam_9.primal_1.H_0);

#line 131
    float _S1232 = 2.0f * _S1228.primal_1;

#line 131
    float h_y_3 = _S1231 / _S1232;

#line 131
    float s_diff_h_y_0 = (0.0f - _S1231 * (_S1228.differential_0 * 2.0f)) / (_S1232 * _S1232);

#line 131
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1233 = { dpxyz_ws_4.primal_1, dpxyz_ws_4.differential_0 };

#line 131
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1234 = { dpcam_9.primal_1.world_view_transform_1, dpcam_9.differential_0.world_view_transform_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1235 = s_fwd_geom_transform_points_0(_S1233, _S1234);


    float limx_3 = 1.29999995231628418f * _S1226.primal_1;

#line 136
    float _S1236 = _S1226.differential_0 * 1.29999995231628418f;
    float limy_3 = 1.29999995231628418f * _S1228.primal_1;

#line 137
    float _S1237 = _S1228.differential_0 * 1.29999995231628418f;
    float _S1238 = _S1235.primal_1.x;

#line 138
    float _S1239 = _S1235.primal_1.z;

#line 138
    float _S1240 = _S1235.differential_0.z;

#line 138
    float _S1241 = _S1239 * _S1239;
    float _S1242 = _S1235.primal_1.y;

#line 139
    float tytz_3 = _S1242 / _S1239;

#line 139
    float s_diff_tytz_0 = (_S1235.differential_0.y * _S1239 - _S1242 * _S1240) / _S1241;

#line 139
    DiffPair_float_0 _S1243 = { - limx_3, - _S1236 };

#line 139
    DiffPair_float_0 _S1244 = { _S1238 / _S1239, (_S1235.differential_0.x * _S1239 - _S1238 * _S1240) / _S1241 };
    DiffPair_float_0 _S1245 = _d_max_1(_S1243, _S1244);

#line 140
    DiffPair_float_0 _S1246 = { limx_3, _S1236 };

#line 140
    DiffPair_float_0 _S1247 = { _S1245.primal_1, _S1245.differential_0 };

#line 140
    DiffPair_float_0 _S1248 = _d_min_1(_S1246, _S1247);

#line 140
    float _S1249 = _S1248.primal_1 * _S1239;

#line 140
    float _S1250 = _S1248.differential_0 * _S1239 + _S1240 * _S1248.primal_1;

#line 140
    float3  _S1251 = _S1235.primal_1;

#line 140
    *&((&_S1251)->x) = _S1249;

#line 140
    float3  _S1252 = _S1235.differential_0;

#line 140
    *&((&_S1252)->x) = _S1250;

#line 140
    DiffPair_float_0 _S1253 = { - limy_3, - _S1237 };

#line 140
    DiffPair_float_0 _S1254 = { tytz_3, s_diff_tytz_0 };
    DiffPair_float_0 _S1255 = _d_max_1(_S1253, _S1254);

#line 141
    DiffPair_float_0 _S1256 = { limy_3, _S1237 };

#line 141
    DiffPair_float_0 _S1257 = { _S1255.primal_1, _S1255.differential_0 };

#line 141
    DiffPair_float_0 _S1258 = _d_min_1(_S1256, _S1257);

#line 141
    float _S1259 = _S1251.z;

#line 141
    float _S1260 = _S1258.differential_0 * _S1259 + _S1252.z * _S1258.primal_1;

#line 141
    *&((&_S1251)->y) = _S1258.primal_1 * _S1259;

#line 141
    *&((&_S1252)->y) = _S1260;

    float _S1261 = _S1251.z;

#line 143
    float _S1262 = _S1252.z;

#line 143
    float _S1263 = _S1261 * _S1261;

#line 143
    float _S1264 = _S1251.x;

#line 143
    float _S1265 = - (h_x_3 * _S1264);

#line 143
    float _S1266 = _S1262 * _S1261;

#line 143
    float _S1267 = _S1266 + _S1266;

#line 143
    float _S1268 = _S1263 * _S1263;
    float _S1269 = _S1251.y;

#line 144
    float _S1270 = - (h_y_3 * _S1269);

#line 144
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1271 = { makeMatrix<float, 3, 3> (h_x_3 / _S1261, 0.0f, _S1265 / _S1263, 0.0f, h_y_3 / _S1261, _S1270 / _S1263, 0.0f, 0.0f, 0.0f), makeMatrix<float, 3, 3> ((s_diff_h_x_0 * _S1261 - h_x_3 * _S1262) / _S1263, 0.0f, (- (s_diff_h_x_0 * _S1264 + _S1252.x * h_x_3) * _S1263 - _S1265 * _S1267) / _S1268, 0.0f, (s_diff_h_y_0 * _S1261 - h_y_3 * _S1262) / _S1263, (- (s_diff_h_y_0 * _S1269 + _S1252.y * h_y_3) * _S1263 - _S1270 * _S1267) / _S1268, 0.0f, 0.0f, 0.0f) };


    return _S1271;
}


#line 147
__device__ DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 s_fwd_covariance_3d_to_2d_0(DiffPair_Camera_0 dpcam_10, DiffPair_vectorx3Cfloatx2C3x3E_0 dpxyz_ws_5, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 dpcov_ws_2)
{


    Matrix<float, 3, 3>  _S1272 = makeMatrix<float, 3, 3> (float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(0)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(0)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(0)].z}, float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(1)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(1)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(1)].z}, float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(2)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(2)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(2)].z});

#line 151
    Matrix<float, 3, 3>  _S1273 = makeMatrix<float, 3, 3> (float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(0)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(0)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(0)].z}, float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(1)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(1)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(1)].z}, float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(2)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(2)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(2)].z});

#line 151
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1274 = { dpxyz_ws_5.primal_1, dpxyz_ws_5.differential_0 };

#line 151
    DiffPair_Camera_0 _S1275 = { dpcam_10.primal_1, dpcam_10.differential_0 };

    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1276 = s_fwd_compute_jacobian_0(_S1274, _S1275);
    Matrix<float, 3, 3>  _S1277 = transpose_0(_S1272);

#line 154
    Matrix<float, 3, 3>  _S1278 = transpose_0(_S1276.primal_1);

#line 154
    Matrix<float, 3, 3>  _S1279 = mul_3(_S1277, _S1278);

#line 154
    Matrix<float, 3, 3>  _S1280 = mul_3(dpcov_ws_2.primal_1, _S1279);

#line 154
    Matrix<float, 3, 3>  _S1281 = mul_3(_S1272, _S1280);

#line 154
    Matrix<float, 3, 3>  _S1282 = mul_3(_S1276.primal_1, _S1281);

#line 154
    Matrix<float, 3, 3>  _S1283 = mul_3(_S1276.differential_0, _S1281) + mul_3(_S1276.primal_1, mul_3(_S1273, _S1280) + mul_3(_S1272, mul_3(dpcov_ws_2.differential_0, _S1279) + mul_3(dpcov_ws_2.primal_1, mul_3(transpose_0(_S1273), _S1278) + mul_3(_S1277, transpose_0(_S1276.differential_0)))));
    float _S1284 = _S1282.rows[int(0)].x + 0.30000001192092896f;

#line 155
    Matrix<float, 3, 3>  _S1285 = _S1282;

#line 155
    *&(((&_S1285)->rows + (int(0)))->x) = _S1284;

#line 155
    Matrix<float, 3, 3>  _S1286 = _S1283;

#line 155
    *&(((&_S1286)->rows + (int(0)))->x) = _S1283.rows[int(0)].x;

#line 155
    *&(((&_S1285)->rows + (int(1)))->y) = _S1282.rows[int(1)].y + 0.30000001192092896f;

#line 155
    *&(((&_S1286)->rows + (int(1)))->y) = _S1283.rows[int(1)].y;

#line 155
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1287 = { makeMatrix<float, 2, 2> (float2 {_S1285.rows[int(0)].x, _S1285.rows[int(0)].y}, float2 {_S1285.rows[int(1)].x, _S1285.rows[int(1)].y}), makeMatrix<float, 2, 2> (float2 {_S1286.rows[int(0)].x, _S1286.rows[int(0)].y}, float2 {_S1286.rows[int(1)].x, _S1286.rows[int(1)].y}) };


    return _S1287;
}


#line 158
__device__ DiffPair_Splat_2D_Vertex_0 s_fwd_project_gaussian_to_camera_0(DiffPair_Gaussian_3D_0 dpg_2, DiffPair_Camera_0 dpcam_11, uint active_sh_18)
{

#line 222
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1288 = { dpg_2.primal_1.xyz_ws_0, dpg_2.differential_0.xyz_ws_0 };

#line 222
    DiffPair_Camera_0 _S1289 = { dpcam_11.primal_1, dpcam_11.differential_0 };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1290 = s_fwd_project_point_0(_S1288, _S1289);
    if(_S1290.primal_1.z <= 0.20000000298023224f)
    {

#line 225
        float3  _S1291 = make_float3 (0.0f);

#line 225
        float3  _S1292 = make_float3 (0.0f);

#line 225
        Splat_2D_Vertex_0 _S1293 = { _S1291, _S1291, makeMatrix<float, 2, 2> (0.0f) };

#line 225
        Splat_2D_Vertex_0 _S1294 = { _S1292, _S1292, makeMatrix<float, 2, 2> (0.0f) };

#line 225
        DiffPair_Splat_2D_Vertex_0 _S1295 = { _S1293, _S1294 };

#line 225
        return _S1295;
    }

#line 225
    DiffPair_SpherHarmCoeffs_0 _S1296 = { dpg_2.primal_1.sh_coeffs_0, dpg_2.differential_0.sh_coeffs_0 };

#line 225
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1297 = { dpcam_11.primal_1.position_1, dpcam_11.differential_0.position_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1298 = s_fwd_compute_color_from_sh_coeffs_0(_S1296, _S1288, _S1297, active_sh_18);

#line 227
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1299 = { dpg_2.primal_1.rotations_0, dpg_2.differential_0.rotations_0 };

#line 227
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1300 = { dpg_2.primal_1.scales_0, dpg_2.differential_0.scales_0 };
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1301 = s_fwd_get_covariance_from_quat_scales_0(_S1299, _S1300);

#line 228
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1302 = { _S1301.primal_1, _S1301.differential_0 };
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1303 = s_fwd_covariance_3d_to_2d_0(_S1289, _S1288, _S1302);

    Splat_2D_Vertex_0 _S1304 = { _S1290.primal_1, _S1298.primal_1, _S1303.primal_1 };

#line 231
    Splat_2D_Vertex_0 _S1305 = { _S1290.differential_0, _S1298.differential_0, _S1303.differential_0 };

#line 231
    DiffPair_Splat_2D_Vertex_0 _S1306 = { _S1304, _S1305 };

#line 231
    return _S1306;
}


#line 72 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
__device__ DiffPair_float_0 s_fwd_compute_det_0(DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 dpM_2)
{

#line 203 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    DiffPair_float_0 _S1307 = { dpM_2.primal_1.rows[int(0)].x * dpM_2.primal_1.rows[int(1)].y - dpM_2.primal_1.rows[int(0)].y * dpM_2.primal_1.rows[int(1)].x, dpM_2.differential_0.rows[int(0)].x * dpM_2.primal_1.rows[int(1)].y + dpM_2.differential_0.rows[int(1)].y * dpM_2.primal_1.rows[int(0)].x - (dpM_2.differential_0.rows[int(0)].y * dpM_2.primal_1.rows[int(1)].x + dpM_2.differential_0.rows[int(1)].x * dpM_2.primal_1.rows[int(0)].y) };
    return _S1307;
}


#line 79 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
__device__ DiffPair_float_0 s_fwd_ndc2pix_0(DiffPair_float_0 dpv_2, int S_3)
{

#line 63 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/utils.slang"
    float _S1308 = float(S_3);

#line 63
    DiffPair_float_0 _S1309 = { ((dpv_2.primal_1 + 1.0f) * _S1308 - 1.0f) * 0.5f, dpv_2.differential_0 * _S1308 * 0.5f };

#line 63
    return _S1309;
}


#line 63
__device__ void s_fwd_vertex_shader_0(DiffTensorView_0 xyz_ws_10, DiffTensorView_0 sh_coeffs_12, DiffTensorView_0 rotations_8, DiffTensorView_0 scales_8, uint active_sh_19, TensorView world_view_transform_6, TensorView proj_mat_6, TensorView cam_pos_4, TensorView out_tiles_touched_3, TensorView out_rect_tile_space_3, TensorView out_radii_3, DiffTensorView_0 out_xyz_vs_3, DiffTensorView_0 out_inv_cov_vs_3, DiffTensorView_0 out_rgb_3, float fovy_6, float fovx_6, uint image_height_3, uint image_width_3, uint grid_height_4, uint grid_width_4, uint tile_height_4, uint tile_width_4)
{

#line 60 "d:/A_study/nerf3dgs/tiny-nerf/slangtorch3dgs/shader/vertex_shader.slang"
    uint g_idx_10 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

    if(g_idx_10 >= DiffTensorView_size_0(xyz_ws_10, 0U))
    {

#line 63
        return;
    }
    Camera_0 cam_5 = load_camera_0(world_view_transform_6, proj_mat_6, cam_pos_4, fovy_6, fovx_6, image_height_3, image_width_3);
    DiffPair_Gaussian_3D_0 _S1310 = s_fwd_load_gaussian_0(int(g_idx_10), xyz_ws_10, sh_coeffs_12, rotations_8, scales_8, active_sh_19);

#line 66
    DiffPair_Gaussian_3D_0 _S1311 = { _S1310.primal_1, _S1310.differential_0 };

#line 66
    DiffPair_Camera_0 _S1312 = { cam_5, Camera_x24_syn_dzero_0() };
    DiffPair_Splat_2D_Vertex_0 _S1313 = s_fwd_project_gaussian_to_camera_0(_S1311, _S1312, active_sh_19);
    float _S1314 = _S1313.primal_1.xyz_vs_0.z;

#line 68
    float _S1315 = _S1313.differential_0.xyz_vs_0.z;

#line 68
    if(_S1314 <= 0.20000000298023224f)
    {

#line 69
        return;
    }

#line 69
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1316 = { _S1313.primal_1.cov_vs_0, _S1313.differential_0.cov_vs_0 };


    DiffPair_float_0 _S1317 = s_fwd_compute_det_0(_S1316);

    if(_S1317.primal_1 == 0.0f)
    {

#line 75
        return;
    }

#line 76
    float radius_2 = splat_radius_0(_S1313.primal_1.cov_vs_0, _S1317.primal_1);

#line 76
    DiffPair_float_0 _S1318 = { _S1313.primal_1.xyz_vs_0.x, _S1313.differential_0.xyz_vs_0.x };

#line 76
    DiffPair_float_0 _S1319 = { _S1313.primal_1.xyz_vs_0.y, _S1313.differential_0.xyz_vs_0.y };



    rectangle_0 rect_tile_space_3 = get_rectangle_tile_space_0(make_float2 (s_fwd_ndc2pix_0(_S1318, int(image_width_3)).primal_1, s_fwd_ndc2pix_0(_S1319, int(image_height_3)).primal_1), radius_2, grid_height_4, grid_width_4, tile_height_4, tile_width_4);

    int n_tiles_1 = (rect_tile_space_3.max_x_0 - rect_tile_space_3.min_x_0) * (rect_tile_space_3.max_y_0 - rect_tile_space_3.min_y_0);

    if(n_tiles_1 == int(0))
    {

#line 85
        return;
    }

    Matrix<float, 2, 2>  _S1320 = makeMatrix<float, 2, 2> (_S1313.primal_1.cov_vs_0.rows[int(1)].y, - _S1313.primal_1.cov_vs_0.rows[int(0)].y, - _S1313.primal_1.cov_vs_0.rows[int(1)].x, _S1313.primal_1.cov_vs_0.rows[int(0)].x);

#line 88
    Matrix<float, 2, 2>  g_inv_cov_vs_1 = _S1320 / makeMatrix<float, 2, 2> (_S1317.primal_1);

#line 88
    Matrix<float, 2, 2>  s_diff_g_inv_cov_vs_0 = (makeMatrix<float, 2, 2> (_S1313.differential_0.cov_vs_0.rows[int(1)].y, - _S1313.differential_0.cov_vs_0.rows[int(0)].y, - _S1313.differential_0.cov_vs_0.rows[int(1)].x, _S1313.differential_0.cov_vs_0.rows[int(0)].x) * makeMatrix<float, 2, 2> (_S1317.primal_1) - _S1320 * makeMatrix<float, 2, 2> (_S1317.differential_0)) / makeMatrix<float, 2, 2> (_S1317.primal_1 * _S1317.primal_1);

    (out_radii_3).store<int>((g_idx_10), (int(uint(radius_2))));
    (out_tiles_touched_3).store<int>((g_idx_10), (n_tiles_1));
    uint2  _S1321 = make_uint2 (g_idx_10, 0U);

#line 92
    (out_rect_tile_space_3).store<int>((g_idx_10), (0U), (rect_tile_space_3.min_x_0));
    uint2  _S1322 = make_uint2 (g_idx_10, 1U);

#line 93
    (out_rect_tile_space_3).store<int>((g_idx_10), (1U), (rect_tile_space_3.min_y_0));
    uint2  _S1323 = make_uint2 (g_idx_10, 2U);

#line 94
    (out_rect_tile_space_3).store<int>((g_idx_10), (2U), (rect_tile_space_3.max_x_0));
    (out_rect_tile_space_3).store<int>((g_idx_10), (3U), (rect_tile_space_3.max_y_0));

    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1321, _S1318);
    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1322, _S1319);

#line 98
    DiffPair_float_0 _S1324 = { _S1314, _S1315 };
    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1323, _S1324);

#line 99
    DiffPair_float_0 _S1325 = { g_inv_cov_vs_1.rows[int(0)].x, s_diff_g_inv_cov_vs_0.rows[int(0)].x };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 0U, 0U), _S1325);

#line 100
    DiffPair_float_0 _S1326 = { g_inv_cov_vs_1.rows[int(0)].y, s_diff_g_inv_cov_vs_0.rows[int(0)].y };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 0U, 1U), _S1326);

#line 101
    DiffPair_float_0 _S1327 = { g_inv_cov_vs_1.rows[int(1)].x, s_diff_g_inv_cov_vs_0.rows[int(1)].x };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 1U, 0U), _S1327);

#line 102
    DiffPair_float_0 _S1328 = { g_inv_cov_vs_1.rows[int(1)].y, s_diff_g_inv_cov_vs_0.rows[int(1)].y };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 1U, 1U), _S1328);

#line 103
    DiffPair_float_0 _S1329 = { _S1313.primal_1.rgb_0.x, _S1313.differential_0.rgb_0.x };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1321, _S1329);

#line 104
    DiffPair_float_0 _S1330 = { _S1313.primal_1.rgb_0.y, _S1313.differential_0.rgb_0.y };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1322, _S1330);

#line 105
    DiffPair_float_0 _S1331 = { _S1313.primal_1.rgb_0.z, _S1313.differential_0.rgb_0.z };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1323, _S1331);
    return;
}


#line 107
extern "C" {
__global__ void __kernel__vertex_shader_fwd_diff(DiffTensorView_0 xyz_ws_11, DiffTensorView_0 sh_coeffs_13, DiffTensorView_0 rotations_9, DiffTensorView_0 scales_9, uint active_sh_20, TensorView world_view_transform_7, TensorView proj_mat_7, TensorView cam_pos_5, TensorView out_tiles_touched_4, TensorView out_rect_tile_space_4, TensorView out_radii_4, DiffTensorView_0 out_xyz_vs_4, DiffTensorView_0 out_inv_cov_vs_4, DiffTensorView_0 out_rgb_4, float fovy_7, float fovx_7, uint image_height_4, uint image_width_4, uint grid_height_5, uint grid_width_5, uint tile_height_5, uint tile_width_5)
{

#line 107
    s_fwd_vertex_shader_0(xyz_ws_11, sh_coeffs_13, rotations_9, scales_9, active_sh_20, world_view_transform_7, proj_mat_7, cam_pos_5, out_tiles_touched_4, out_rect_tile_space_4, out_radii_4, out_xyz_vs_4, out_inv_cov_vs_4, out_rgb_4, fovy_7, fovx_7, image_height_4, image_width_4, grid_height_5, grid_width_5, tile_height_5, tile_width_5);

#line 107
    return;
}

}

#line 37
__global__ void __kernel__vertex_shader(DiffTensorView_0 xyz_ws_12, DiffTensorView_0 sh_coeffs_14, DiffTensorView_0 rotations_10, DiffTensorView_0 scales_10, uint active_sh_21, TensorView world_view_transform_8, TensorView proj_mat_8, TensorView cam_pos_6, TensorView out_tiles_touched_5, TensorView out_rect_tile_space_5, TensorView out_radii_5, DiffTensorView_0 out_xyz_vs_5, DiffTensorView_0 out_inv_cov_vs_5, DiffTensorView_0 out_rgb_5, float fovy_8, float fovx_8, uint image_height_5, uint image_width_5, uint grid_height_6, uint grid_width_6, uint tile_height_6, uint tile_width_6)
{

#line 60
    uint g_idx_11 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

    if(g_idx_11 >= DiffTensorView_size_0(xyz_ws_12, 0U))
    {

#line 63
        return;
    }
    Camera_0 cam_6 = load_camera_0(world_view_transform_8, proj_mat_8, cam_pos_6, fovy_8, fovx_8, image_height_5, image_width_5);

    Splat_2D_Vertex_0 splat_0 = project_gaussian_to_camera_0(load_gaussian_0(int(g_idx_11), xyz_ws_12, sh_coeffs_14, rotations_10, scales_10, active_sh_21), cam_6, active_sh_21);
    float _S1332 = splat_0.xyz_vs_0.z;

#line 68
    if(_S1332 <= 0.20000000298023224f)
    {

#line 69
        return;
    }

    float det_1 = compute_det_0(splat_0.cov_vs_0);

    if(det_1 == 0.0f)
    {

#line 75
        return;
    }

#line 76
    float radius_3 = splat_radius_0(splat_0.cov_vs_0, det_1);


    float _S1333 = splat_0.xyz_vs_0.x;

#line 79
    float _S1334 = splat_0.xyz_vs_0.y;
    rectangle_0 rect_tile_space_4 = get_rectangle_tile_space_0(make_float2 (ndc2pix_0(_S1333, int(image_width_5)), ndc2pix_0(_S1334, int(image_height_5))), radius_3, grid_height_6, grid_width_6, tile_height_6, tile_width_6);

    int n_tiles_2 = (rect_tile_space_4.max_x_0 - rect_tile_space_4.min_x_0) * (rect_tile_space_4.max_y_0 - rect_tile_space_4.min_y_0);

    if(n_tiles_2 == int(0))
    {

#line 85
        return;
    }

    Matrix<float, 2, 2>  g_inv_cov_vs_2 = makeMatrix<float, 2, 2> (splat_0.cov_vs_0.rows[int(1)].y, - splat_0.cov_vs_0.rows[int(0)].y, - splat_0.cov_vs_0.rows[int(1)].x, splat_0.cov_vs_0.rows[int(0)].x) / makeMatrix<float, 2, 2> (det_1);

    (out_radii_5).store<int>((g_idx_11), (int(uint(radius_3))));
    (out_tiles_touched_5).store<int>((g_idx_11), (n_tiles_2));
    (out_rect_tile_space_5).store<int>((g_idx_11), (0U), (rect_tile_space_4.min_x_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (1U), (rect_tile_space_4.min_y_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (2U), (rect_tile_space_4.max_x_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (3U), (rect_tile_space_4.max_y_0));

    uint2  _S1335 = make_uint2 (g_idx_11, 0U);

#line 97
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1335, _S1333);
    uint2  _S1336 = make_uint2 (g_idx_11, 1U);

#line 98
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1336, _S1334);
    uint2  _S1337 = make_uint2 (g_idx_11, 2U);

#line 99
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1337, _S1332);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 0U, 0U), g_inv_cov_vs_2.rows[int(0)].x);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 0U, 1U), g_inv_cov_vs_2.rows[int(0)].y);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 1U, 0U), g_inv_cov_vs_2.rows[int(1)].x);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 1U, 1U), g_inv_cov_vs_2.rows[int(1)].y);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1335, splat_0.rgb_0.x);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1336, splat_0.rgb_0.y);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1337, splat_0.rgb_0.z);
    return;
}


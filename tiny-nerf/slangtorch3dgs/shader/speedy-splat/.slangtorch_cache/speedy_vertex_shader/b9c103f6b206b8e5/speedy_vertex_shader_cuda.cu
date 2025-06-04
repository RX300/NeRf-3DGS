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


#line 85 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
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


#line 34 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
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


#line 168 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
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


#line 193 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
struct Splat_2D_Vertex_0
{
    float3  xyz_vs_0;
    float3  rgb_0;
    Matrix<float, 2, 2>  cov_vs_0;
};


#line 193
__device__ Splat_2D_Vertex_0 Splat_2D_Vertex_x24_syn_dzero_0()
{

#line 193
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


#line 85 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
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

#line 97
    float _S10 = ((world_view_transform_t_0).load<float>((0U), (0U)));

#line 97
    float _S11 = ((world_view_transform_t_0).load<float>((0U), (1U)));

#line 97
    float _S12 = ((world_view_transform_t_0).load<float>((0U), (2U)));

#line 97
    float _S13 = ((world_view_transform_t_0).load<float>((0U), (3U)));

#line 97
    float _S14 = ((world_view_transform_t_0).load<float>((1U), (0U)));

#line 97
    float _S15 = ((world_view_transform_t_0).load<float>((1U), (1U)));

#line 97
    float _S16 = ((world_view_transform_t_0).load<float>((1U), (2U)));

#line 97
    float _S17 = ((world_view_transform_t_0).load<float>((1U), (3U)));

#line 97
    float _S18 = ((world_view_transform_t_0).load<float>((2U), (0U)));

#line 97
    float _S19 = ((world_view_transform_t_0).load<float>((2U), (1U)));

#line 97
    float _S20 = ((world_view_transform_t_0).load<float>((2U), (2U)));

#line 97
    float _S21 = ((world_view_transform_t_0).load<float>((2U), (3U)));

#line 97
    float _S22 = ((world_view_transform_t_0).load<float>((3U), (0U)));

#line 97
    float _S23 = ((world_view_transform_t_0).load<float>((3U), (1U)));

#line 97
    float _S24 = ((world_view_transform_t_0).load<float>((3U), (2U)));

#line 97
    float _S25 = ((world_view_transform_t_0).load<float>((3U), (3U)));

#line 97
    Matrix<float, 4, 4>  world_view_transform_2 = makeMatrix<float, 4, 4> (_S10, _S11, _S12, _S13, _S14, _S15, _S16, _S17, _S18, _S19, _S20, _S21, _S22, _S23, _S24, _S25);

#line 102
    float _S26 = ((proj_mat_t_0).load<float>((0U), (0U)));

#line 102
    float _S27 = ((proj_mat_t_0).load<float>((0U), (1U)));

#line 102
    float _S28 = ((proj_mat_t_0).load<float>((0U), (2U)));

#line 102
    float _S29 = ((proj_mat_t_0).load<float>((0U), (3U)));

#line 102
    float _S30 = ((proj_mat_t_0).load<float>((1U), (0U)));

#line 102
    float _S31 = ((proj_mat_t_0).load<float>((1U), (1U)));

#line 102
    float _S32 = ((proj_mat_t_0).load<float>((1U), (2U)));

#line 102
    float _S33 = ((proj_mat_t_0).load<float>((1U), (3U)));

#line 102
    float _S34 = ((proj_mat_t_0).load<float>((2U), (0U)));

#line 102
    float _S35 = ((proj_mat_t_0).load<float>((2U), (1U)));

#line 102
    float _S36 = ((proj_mat_t_0).load<float>((2U), (2U)));

#line 102
    float _S37 = ((proj_mat_t_0).load<float>((2U), (3U)));

#line 102
    float _S38 = ((proj_mat_t_0).load<float>((3U), (0U)));

#line 102
    float _S39 = ((proj_mat_t_0).load<float>((3U), (1U)));

#line 102
    float _S40 = ((proj_mat_t_0).load<float>((3U), (2U)));

#line 102
    float _S41 = ((proj_mat_t_0).load<float>((3U), (3U)));

#line 102
    Matrix<float, 4, 4>  proj_mat_2 = makeMatrix<float, 4, 4> (_S26, _S27, _S28, _S29, _S30, _S31, _S32, _S33, _S34, _S35, _S36, _S37, _S38, _S39, _S40, _S41);



    float _S42 = ((position_t_0).load<float>((0U)));

#line 106
    float _S43 = ((position_t_0).load<float>((1U)));

#line 106
    float _S44 = ((position_t_0).load<float>((2U)));

    Camera_0 _S45 = { world_view_transform_2, proj_mat_2, make_float3 (_S42, _S43, _S44), fovy_2, fovx_2, int(H_1), int(W_1) };

#line 108
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


#line 33 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ float3  read_t3_float3_0(uint idx_0, DiffTensorView_0 t3_0)
{
    return make_float3 (DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 0U)), DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 1U)), DiffTensorView_load_0(t3_0, make_uint2 (idx_0, 2U)));
}


#line 62 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
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


#line 41 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ float4  read_t4_float4_0(uint idx_1, DiffTensorView_0 t4_0)
{
    return make_float4 (DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 0U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 1U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 2U)), DiffTensorView_load_0(t4_0, make_uint2 (idx_1, 3U)));
}


#line 177
__device__ Gaussian_3D_0 load_gaussian_0(int g_idx_1, DiffTensorView_0 xyz_ws_1, DiffTensorView_0 sh_coeffs_2, DiffTensorView_0 rotations_1, DiffTensorView_0 scales_1, uint active_sh_1)
{

#line 184
    uint _S48 = uint(g_idx_1);

#line 189
    Gaussian_3D_0 _S49 = { read_t3_float3_0(_S48, xyz_ws_1), read_spherical_harmonics_coeffs_0(_S48, sh_coeffs_2, active_sh_1), read_t4_float4_0(_S48, rotations_1), read_t3_float3_0(_S48, scales_1) };

#line 189
    return _S49;
}


#line 189
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


#line 112 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
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

#line 127
    float3  proj_point_0 = geom_transform_points_0(point_2, mul_2(cam_0.proj_mat_1, cam_0.world_view_transform_1));

    *&((&proj_point_0)->z) = geom_transform_points2_0(point_2, cam_0.world_view_transform_1).z;
    return proj_point_0;
}


#line 130
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


#line 94 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
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


#line 287 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ Matrix<float, 3, 3>  get_covariance_from_quat_scales_0(float4  q_0, float3  s_0)
{

#line 287
    float y_2 = q_0.z;



    float _S80 = y_2 * y_2;

#line 291
    float _S81 = q_0.w * q_0.w;

#line 291
    float _S82 = q_0.y * q_0.z;

#line 291
    float _S83 = q_0.x * q_0.w;

#line 291
    float _S84 = q_0.y * q_0.w;

#line 291
    float _S85 = q_0.x * q_0.z;
    float _S86 = q_0.y * q_0.y;

#line 292
    float _S87 = q_0.z * q_0.w;

#line 292
    float _S88 = q_0.x * q_0.y;

#line 299
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


#line 134 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ Matrix<float, 3, 3>  compute_jacobian_0(float3  xyz_ws_2, Camera_0 cam_1)
{

#line 135
    float tan_half_fovx_0 = (F32_tan((cam_1.fovx_1 / 2.0f)));
    float tan_half_fovy_0 = (F32_tan((cam_1.fovy_1 / 2.0f)));
    float h_x_0 = float(cam_1.W_0) / (2.0f * tan_half_fovx_0);
    float h_y_0 = float(cam_1.H_0) / (2.0f * tan_half_fovy_0);

    float3  _S98 = geom_transform_points_0(xyz_ws_2, cam_1.world_view_transform_1);

#line 140
    float3  t_0 = _S98;


    float limx_0 = 1.29999995231628418f * tan_half_fovx_0;
    float limy_0 = 1.29999995231628418f * tan_half_fovy_0;
    float _S99 = _S98.z;
    float tytz_0 = _S98.y / _S99;
    *&((&t_0)->x) = (F32_min((limx_0), ((F32_max((- limx_0), (_S98.x / _S99)))))) * _S99;
    *&((&t_0)->y) = (F32_min((limy_0), ((F32_max((- limy_0), (tytz_0)))))) * t_0.z;

#line 154
    return makeMatrix<float, 3, 3> (h_x_0 / t_0.z, 0.0f, - (h_x_0 * t_0.x) / (t_0.z * t_0.z), 0.0f, h_y_0 / t_0.z, - (h_y_0 * t_0.y) / (t_0.z * t_0.z), 0.0f, 0.0f, 0.0f);
}


__device__ Matrix<float, 2, 2>  covariance_3d_to_2d_0(Camera_0 cam_2, float3  xyz_ws_3, Matrix<float, 3, 3>  cov_ws_0)
{

#line 158
    Matrix<float, 3, 3>  _S100 = makeMatrix<float, 3, 3> (float3 {cam_2.world_view_transform_1.rows[int(0)].x, cam_2.world_view_transform_1.rows[int(0)].y, cam_2.world_view_transform_1.rows[int(0)].z}, float3 {cam_2.world_view_transform_1.rows[int(1)].x, cam_2.world_view_transform_1.rows[int(1)].y, cam_2.world_view_transform_1.rows[int(1)].z}, float3 {cam_2.world_view_transform_1.rows[int(2)].x, cam_2.world_view_transform_1.rows[int(2)].y, cam_2.world_view_transform_1.rows[int(2)].z});

    Matrix<float, 3, 3>  J_0 = compute_jacobian_0(xyz_ws_3, cam_2);
    Matrix<float, 3, 3>  cov_vs_1 = mul_3(J_0, mul_3(_S100, mul_3(cov_ws_0, mul_3(transpose_0(_S100), transpose_0(J_0)))));
    *&(((&cov_vs_1)->rows + (int(0)))->x) = *&(((&cov_vs_1)->rows + (int(0)))->x) + 0.30000001192092896f;
    *&(((&cov_vs_1)->rows + (int(1)))->y) = *&(((&cov_vs_1)->rows + (int(1)))->y) + 0.30000001192092896f;

    return makeMatrix<float, 2, 2> (float2 {cov_vs_1.rows[int(0)].x, cov_vs_1.rows[int(0)].y}, float2 {cov_vs_1.rows[int(1)].x, cov_vs_1.rows[int(1)].y});
}


#line 229
__device__ Splat_2D_Vertex_0 project_gaussian_to_camera_0(Gaussian_3D_0 g_0, Camera_0 cam_3, uint active_sh_3)
{

#line 230
    float3  xyz_vs_1 = project_point_0(g_0.xyz_ws_0, cam_3);
    if(xyz_vs_1.z <= 0.20000000298023224f)
    {

#line 232
        float3  _S101 = make_float3 (0.0f);

#line 232
        Splat_2D_Vertex_0 _S102 = { _S101, _S101, makeMatrix<float, 2, 2> (0.0f) };

#line 232
        return _S102;
    }

#line 238
    Splat_2D_Vertex_0 _S103 = { xyz_vs_1, compute_color_from_sh_coeffs_0(g_0.sh_coeffs_0, g_0.xyz_ws_0, cam_3.position_1, active_sh_3), covariance_3d_to_2d_0(cam_3, g_0.xyz_ws_0, get_covariance_from_quat_scales_0(g_0.rotations_0, g_0.scales_0)) };

#line 238
    return _S103;
}


#line 210
__device__ float compute_det_0(Matrix<float, 2, 2>  M_0)
{

#line 211
    return M_0.rows[int(0)].x * M_0.rows[int(1)].y - M_0.rows[int(0)].y * M_0.rows[int(1)].x;
}


#line 200
__device__ float splat_radius_0(Matrix<float, 2, 2>  cov_vs_2, float det_0)
{

#line 201
    float mid_0 = 0.5f * (cov_vs_2.rows[int(0)].x + cov_vs_2.rows[int(1)].y);
    float _S104 = (F32_sqrt(((F32_max((0.10000000149011612f), (mid_0 * mid_0 - det_0))))));



    return (F32_ceil((3.0f * (F32_sqrt(((F32_max((mid_0 + _S104), (mid_0 - _S104)))))))));
}


#line 68
__device__ float ndc2pix_0(float v_0, int S_0)
{
    return ((v_0 + 1.0f) * float(S_0) - 1.0f) * 0.5f;
}


#line 304
__device__ float2  computeEllipseIntersection_0(float3  invCov2D_0, float disc_0, float t_1, float2  pixel2D_0, bool coordIsY_0, float coord_0)
{
    float p_u_0;


    if(coordIsY_0)
    {

#line 309
        p_u_0 = pixel2D_0.y;

#line 309
    }
    else
    {

#line 309
        p_u_0 = pixel2D_0.x;

#line 309
    }

#line 309
    float p_v_0;
    if(coordIsY_0)
    {

#line 310
        p_v_0 = pixel2D_0.x;

#line 310
    }
    else
    {

#line 310
        p_v_0 = pixel2D_0.y;

#line 310
    }

#line 310
    float coeff_0;
    if(coordIsY_0)
    {

#line 311
        coeff_0 = invCov2D_0.x;

#line 311
    }
    else
    {

#line 311
        coeff_0 = invCov2D_0.z;

#line 311
    }

    float h_0 = coord_0 - p_u_0;
    float sqrt_term_0 = (F32_sqrt((disc_0 * h_0 * h_0 + t_1 * coeff_0)));


    float _S105 = - invCov2D_0.y * h_0;

#line 316
    return make_float2 ((_S105 - sqrt_term_0) / coeff_0 + p_v_0, (_S105 + sqrt_term_0) / coeff_0 + p_v_0);
}


#line 19
struct rectangle_0
{
    int min_x_0;
    int min_y_0;
    int max_x_0;
    int max_y_0;
};


#line 322
__device__ rectangle_0 computeSnugBox_0(float3  invCov2D_1, float2  pixel2D_1, float opacity_0, int2  grid_0, int2  tile_0, uint * n_tiles_0)
{

    float _S106 = invCov2D_1.y;

#line 325
    float _S107 = _S106 * _S106;

#line 325
    float _S108 = invCov2D_1.x;

#line 325
    float _S109 = invCov2D_1.z;

#line 325
    float disc_1 = _S107 - _S108 * _S109;

#line 330
    float t_2 = 2.0f * (F32_log((opacity_0 * 255.0f)));

    float _S110 = - (_S107 * t_2);

#line 332
    float x_term_0 = (F32_sqrt((_S110 / (disc_1 * _S108))));
    bool _S111 = _S106 < 0.0f;

#line 333
    float x_term_1;

#line 333
    if(_S111)
    {

#line 333
        x_term_1 = x_term_0;

#line 333
    }
    else
    {

#line 333
        x_term_1 = - x_term_0;

#line 333
    }
    float y_term_0 = (F32_sqrt((_S110 / (disc_1 * _S109))));

#line 334
    float y_term_1;
    if(_S111)
    {

#line 335
        y_term_1 = y_term_0;

#line 335
    }
    else
    {

#line 335
        y_term_1 = - y_term_0;

#line 335
    }

    float _S112 = pixel2D_1.y;

#line 337
    float _S113 = pixel2D_1.x;

#line 350
    int _S114 = grid_0.x;

#line 350
    float _S115 = float(tile_0.x);

#line 350
    int _S116 = (I32_max((int(0)), ((I32_min((_S114), (int(computeEllipseIntersection_0(invCov2D_1, disc_1, t_2, pixel2D_1, true, _S112 - y_term_1).x / _S115)))))));
    int _S117 = grid_0.y;

#line 351
    float _S118 = float(tile_0.y);

#line 351
    int _S119 = (I32_max((int(0)), ((I32_min((_S117), (int(computeEllipseIntersection_0(invCov2D_1, disc_1, t_2, pixel2D_1, false, _S113 - x_term_1).x / _S118)))))));


    int _S120 = (I32_max((int(0)), ((I32_min((_S114), (int(computeEllipseIntersection_0(invCov2D_1, disc_1, t_2, pixel2D_1, true, _S112 + y_term_1).y / _S115 + 1.0f)))))));
    int _S121 = (I32_max((int(0)), ((I32_min((_S117), (int(computeEllipseIntersection_0(invCov2D_1, disc_1, t_2, pixel2D_1, false, _S113 + x_term_1).y / _S118 + 1.0f)))))));

#line 361
    *n_tiles_0 = uint((_S121 - _S119) * (_S120 - _S116));
    rectangle_0 _S122 = { _S116, _S119, _S120, _S121 };

#line 362
    return _S122;
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
__device__ void DiffTensorView_storeOnce_0(DiffTensorView_0 this_13, uint2  x_7, float val_0)
{

#line 1026
    (this_13.primal_0).store<float>((x_7), (val_0));

#line 1026
    return;
}


#line 1026
__device__ void DiffTensorView_storeOnce_1(DiffTensorView_0 this_14, uint3  x_8, float val_1)
{

#line 1026
    (this_14.primal_0).store<float>((x_8), (val_1));

#line 1026
    return;
}


#line 53 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
struct s_bwd_prop_vertex_shader_Intermediates_0
{
    Camera_0 _S123;
    Gaussian_3D_0 _S124;
    float _S125;
};


#line 82
__device__ float3  s_primal_ctx_read_t3_float3_0(uint idx_2, DiffTensorView_0 t3_1)
{

#line 33 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    return make_float3 (DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 0U)), DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 1U)), DiffTensorView_load_0(t3_1, make_uint2 (idx_2, 2U)));
}


#line 33
__device__ SpherHarmCoeffs_0 s_primal_ctx_read_spherical_harmonics_coeffs_0(uint g_idx_2, DiffTensorView_0 sh_coeffs_3, uint active_sh_4)
{

#line 64 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
    float3  _S126 = make_float3 (0.0f);
    float3  _S127 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 0U, 2U)));

#line 65
    SpherHarmCoeffs_0 g_sh_coeffs_1;

    if(active_sh_4 > 0U)
    {

#line 68
        float3  _S128 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 1U, 2U)));
        float3  _S129 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 2U, 2U)));
        float3  _S130 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 3U, 2U)));

        if(active_sh_4 > 1U)
        {

#line 73
            float3  _S131 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 4U, 2U)));
            float3  _S132 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 5U, 2U)));
            float3  _S133 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 6U, 2U)));
            float3  _S134 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 7U, 2U)));
            float3  _S135 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 8U, 2U)));

            if(active_sh_4 > 2U)
            {

#line 80
                float3  _S136 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 9U, 2U)));
                float3  _S137 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 10U, 2U)));
                float3  _S138 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 11U, 2U)));
                float3  _S139 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 12U, 2U)));
                float3  _S140 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 13U, 2U)));
                float3  _S141 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 14U, 2U)));
                float3  _S142 = make_float3 (DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 0U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 1U)), DiffTensorView_load_1(sh_coeffs_3, make_uint3 (g_idx_2, 15U, 2U)));

#line 86
                (&g_sh_coeffs_1)->coeff0_0 = _S127;

#line 86
                (&g_sh_coeffs_1)->coeff1_0 = _S128;

#line 86
                (&g_sh_coeffs_1)->coeff2_0 = _S129;

#line 86
                (&g_sh_coeffs_1)->coeff3_0 = _S130;

#line 86
                (&g_sh_coeffs_1)->coeff4_0 = _S131;

#line 86
                (&g_sh_coeffs_1)->coeff5_0 = _S132;

#line 86
                (&g_sh_coeffs_1)->coeff6_0 = _S133;

#line 86
                (&g_sh_coeffs_1)->coeff7_0 = _S134;

#line 86
                (&g_sh_coeffs_1)->coeff8_0 = _S135;

#line 86
                (&g_sh_coeffs_1)->coeff9_0 = _S136;

#line 86
                (&g_sh_coeffs_1)->coeff10_0 = _S137;

#line 86
                (&g_sh_coeffs_1)->coeff11_0 = _S138;

#line 86
                (&g_sh_coeffs_1)->coeff12_0 = _S139;

#line 86
                (&g_sh_coeffs_1)->coeff13_0 = _S140;

#line 86
                (&g_sh_coeffs_1)->coeff14_0 = _S141;

#line 86
                (&g_sh_coeffs_1)->coeff15_0 = _S142;

#line 79
            }
            else
            {

#line 79
                (&g_sh_coeffs_1)->coeff0_0 = _S127;

#line 79
                (&g_sh_coeffs_1)->coeff1_0 = _S128;

#line 79
                (&g_sh_coeffs_1)->coeff2_0 = _S129;

#line 79
                (&g_sh_coeffs_1)->coeff3_0 = _S130;

#line 79
                (&g_sh_coeffs_1)->coeff4_0 = _S131;

#line 79
                (&g_sh_coeffs_1)->coeff5_0 = _S132;

#line 79
                (&g_sh_coeffs_1)->coeff6_0 = _S133;

#line 79
                (&g_sh_coeffs_1)->coeff7_0 = _S134;

#line 79
                (&g_sh_coeffs_1)->coeff8_0 = _S135;

#line 79
                (&g_sh_coeffs_1)->coeff9_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff10_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff11_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff12_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff13_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff14_0 = _S126;

#line 79
                (&g_sh_coeffs_1)->coeff15_0 = _S126;

#line 79
            }

#line 72
        }
        else
        {

#line 72
            (&g_sh_coeffs_1)->coeff0_0 = _S127;

#line 72
            (&g_sh_coeffs_1)->coeff1_0 = _S128;

#line 72
            (&g_sh_coeffs_1)->coeff2_0 = _S129;

#line 72
            (&g_sh_coeffs_1)->coeff3_0 = _S130;

#line 72
            (&g_sh_coeffs_1)->coeff4_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff5_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff6_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff7_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff8_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff9_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff10_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff11_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff12_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff13_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff14_0 = _S126;

#line 72
            (&g_sh_coeffs_1)->coeff15_0 = _S126;

#line 72
        }

#line 67
    }
    else
    {

#line 67
        (&g_sh_coeffs_1)->coeff0_0 = _S127;

#line 67
        (&g_sh_coeffs_1)->coeff1_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff2_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff3_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff4_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff5_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff6_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff7_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff8_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff9_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff10_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff11_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff12_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff13_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff14_0 = _S126;

#line 67
        (&g_sh_coeffs_1)->coeff15_0 = _S126;

#line 67
    }

#line 67
    return g_sh_coeffs_1;
}


#line 67
__device__ float4  s_primal_ctx_read_t4_float4_0(uint idx_3, DiffTensorView_0 t4_1)
{

#line 41 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    return make_float4 (DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 0U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 1U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 2U)), DiffTensorView_load_0(t4_1, make_uint2 (idx_3, 3U)));
}


#line 41
__device__ Gaussian_3D_0 s_primal_ctx_load_gaussian_0(int g_idx_3, DiffTensorView_0 xyz_ws_4, DiffTensorView_0 sh_coeffs_4, DiffTensorView_0 rotations_2, DiffTensorView_0 scales_2, uint active_sh_5)
{

#line 184
    uint _S143 = uint(g_idx_3);

#line 189
    Gaussian_3D_0 _S144 = { s_primal_ctx_read_t3_float3_0(_S143, xyz_ws_4), s_primal_ctx_read_spherical_harmonics_coeffs_0(_S143, sh_coeffs_4, active_sh_5), s_primal_ctx_read_t4_float4_0(_S143, rotations_2), s_primal_ctx_read_t3_float3_0(_S143, scales_2) };

#line 189
    return _S144;
}


#line 189
__device__ Matrix<float, 4, 4>  s_primal_ctx_mul_0(Matrix<float, 4, 4>  _S145, Matrix<float, 4, 4>  _S146)
{

#line 189
    return mul_2(_S145, _S146);
}


#line 189
__device__ float4  s_primal_ctx_mul_1(Matrix<float, 4, 4>  _S147, float4  _S148)
{

#line 189
    return mul_4(_S147, _S148);
}


#line 189
__device__ float3  s_primal_ctx_geom_transform_points_0(float3  dppoint_0, Matrix<float, 4, 4>  dptransf_matrix_0)
{

#line 112
    float4  _S149 = s_primal_ctx_mul_1(dptransf_matrix_0, make_float4 (dppoint_0.x, dppoint_0.y, dppoint_0.z, 1.0f));

#line 112
    return float3 {_S149.x, _S149.y, _S149.z} / make_float3 (_S149.w + 1.00000001168609742e-07f);
}


#line 112
__device__ float3  s_primal_ctx_geom_transform_points2_0(float3  dppoint_1, Matrix<float, 4, 4>  dptransf_matrix_1)
{

#line 119
    return float3 {s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).x, s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).y, s_primal_ctx_mul_1(dptransf_matrix_1, make_float4 (dppoint_1.x, dppoint_1.y, dppoint_1.z, 1.0f)).z};
}


#line 119
__device__ float3  s_primal_ctx_project_point_0(float3  dppoint_2, Camera_0 dpcam_0)
{

#line 129
    float _S150 = s_primal_ctx_geom_transform_points2_0(dppoint_2, dpcam_0.world_view_transform_1).z;

#line 129
    float3  _S151 = s_primal_ctx_geom_transform_points_0(dppoint_2, s_primal_ctx_mul_0(dpcam_0.proj_mat_1, dpcam_0.world_view_transform_1));

#line 129
    *&((&_S151)->z) = _S150;

#line 129
    return _S151;
}


#line 129
__device__ float3  s_primal_ctx_max_0(float3  _S152, float3  _S153)
{

#line 129
    return max_0(_S152, _S153);
}


#line 129
__device__ float3  s_primal_ctx_compute_color_from_sh_coeffs_0(SpherHarmCoeffs_0 dpsh_0, float3  dpg_xyz_ws_0, float3  dpcam_pos_0, uint active_sh_6)
{

#line 96 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
    float3  _S154 = normalize_0(dpg_xyz_ws_0 - dpcam_pos_0);

    float3  rgb_5 = make_float3 (0.282094806432724f) * dpsh_0.coeff0_0;

#line 98
    float3  rgb_6;
    if(active_sh_6 > 0U)
    {

#line 100
        float _S155 = _S154.y;

#line 100
        float _S156 = _S154.z;

#line 100
        float _S157 = _S154.x;

#line 100
        float3  rgb_7 = rgb_5 - make_float3 (0.48860251903533936f * _S155) * dpsh_0.coeff1_0 + make_float3 (0.48860251903533936f * _S156) * dpsh_0.coeff2_0 - make_float3 (0.48860251903533936f * _S157) * dpsh_0.coeff3_0;
        if(active_sh_6 > 1U)
        {
            float xx_1 = _S157 * _S157;

#line 103
            float yy_1 = _S155 * _S155;

#line 103
            float zz_1 = _S156 * _S156;
            float xy_1 = _S157 * _S155;



            float _S158 = 2.0f * zz_1;

            float _S159 = xx_1 - yy_1;

#line 109
            float3  rgb_8 = rgb_7 + make_float3 (1.09254848957061768f * xy_1) * dpsh_0.coeff4_0 + make_float3 (-1.09254848957061768f * (_S155 * _S156)) * dpsh_0.coeff5_0 + make_float3 (0.31539157032966614f * (_S158 - xx_1 - yy_1)) * dpsh_0.coeff6_0 + make_float3 (-1.09254848957061768f * (_S157 * _S156)) * dpsh_0.coeff7_0 + make_float3 (0.54627424478530884f * _S159) * dpsh_0.coeff8_0;


            if(active_sh_6 > 2U)
            {

                float _S160 = 3.0f * xx_1;

                float _S161 = 4.0f * zz_1 - xx_1 - yy_1;
                float _S162 = 3.0f * yy_1;

#line 118
                rgb_6 = rgb_8 + make_float3 (-0.59004360437393188f * _S155 * (_S160 - yy_1)) * dpsh_0.coeff9_0 + make_float3 (2.89061141014099121f * xy_1 * _S156) * dpsh_0.coeff10_0 + make_float3 (-0.4570457935333252f * _S155 * _S161) * dpsh_0.coeff11_0 + make_float3 (0.37317633628845215f * _S156 * (_S158 - _S160 - _S162)) * dpsh_0.coeff12_0 + make_float3 (-0.4570457935333252f * _S157 * _S161) * dpsh_0.coeff13_0 + make_float3 (1.44530570507049561f * _S156 * _S159) * dpsh_0.coeff14_0 + make_float3 (-0.59004360437393188f * _S157 * (xx_1 - _S162)) * dpsh_0.coeff15_0;

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
__device__ Matrix<float, 3, 3>  s_primal_ctx_mul_2(Matrix<float, 3, 3>  _S163, Matrix<float, 3, 3>  _S164)
{

#line 99
    return mul_3(_S163, _S164);
}


#line 99
__device__ Matrix<float, 3, 3>  s_primal_ctx_get_covariance_from_quat_scales_0(float4  dpq_0, float3  dps_0)
{

#line 287 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    float _S165 = dpq_0.z;



    float _S166 = _S165 * _S165;

#line 291
    float _S167 = dpq_0.w * dpq_0.w;

#line 291
    float _S168 = dpq_0.y * dpq_0.z;

#line 291
    float _S169 = dpq_0.x * dpq_0.w;

#line 291
    float _S170 = dpq_0.y * dpq_0.w;

#line 291
    float _S171 = dpq_0.x * dpq_0.z;
    float _S172 = dpq_0.y * dpq_0.y;

#line 292
    float _S173 = dpq_0.z * dpq_0.w;

#line 292
    float _S174 = dpq_0.x * dpq_0.y;

#line 292
    Matrix<float, 3, 3>  _S175 = s_primal_ctx_mul_2(makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S166 + _S167), 2.0f * (_S168 - _S169), 2.0f * (_S170 + _S171), 2.0f * (_S168 + _S169), 1.0f - 2.0f * (_S172 + _S167), 2.0f * (_S173 - _S174), 2.0f * (_S170 - _S171), 2.0f * (_S173 + _S174), 1.0f - 2.0f * (_S172 + _S166)), makeMatrix<float, 3, 3> (dps_0.x, 0.0f, 0.0f, 0.0f, dps_0.y, 0.0f, 0.0f, 0.0f, dps_0.z));

#line 292
    return s_primal_ctx_mul_2(_S175, transpose_0(_S175));
}


#line 292
__device__ float s_primal_ctx_tan_0(float _S176)
{

#line 292
    return (F32_tan((_S176)));
}


#line 292
__device__ float s_primal_ctx_max_1(float _S177, float _S178)
{

#line 292
    return (F32_max((_S177), (_S178)));
}


#line 292
__device__ float s_primal_ctx_min_0(float _S179, float _S180)
{

#line 292
    return (F32_min((_S179), (_S180)));
}


#line 292
__device__ Matrix<float, 3, 3>  s_primal_ctx_compute_jacobian_0(float3  dpxyz_ws_0, Camera_0 dpcam_1)
{

#line 134
    float _S181 = s_primal_ctx_tan_0(dpcam_1.fovx_1 / 2.0f);

#line 134
    float _S182 = s_primal_ctx_tan_0(dpcam_1.fovy_1 / 2.0f);


    float h_x_1 = float(dpcam_1.W_0) / (2.0f * _S181);
    float h_y_1 = float(dpcam_1.H_0) / (2.0f * _S182);

#line 138
    float3  _S183 = s_primal_ctx_geom_transform_points_0(dpxyz_ws_0, dpcam_1.world_view_transform_1);

#line 143
    float limx_1 = 1.29999995231628418f * _S181;
    float limy_1 = 1.29999995231628418f * _S182;
    float _S184 = _S183.z;
    float tytz_1 = _S183.y / _S184;
    float _S185 = s_primal_ctx_min_0(limx_1, s_primal_ctx_max_1(- limx_1, _S183.x / _S184)) * _S184;

#line 147
    float3  _S186 = _S183;

#line 147
    *&((&_S186)->x) = _S185;

#line 147
    *&((&_S186)->y) = s_primal_ctx_min_0(limy_1, s_primal_ctx_max_1(- limy_1, tytz_1)) * _S186.z;


    float _S187 = _S186.z;

#line 150
    float _S188 = _S187 * _S187;

#line 150
    return makeMatrix<float, 3, 3> (h_x_1 / _S187, 0.0f, - (h_x_1 * _S186.x) / _S188, 0.0f, h_y_1 / _S187, - (h_y_1 * _S186.y) / _S188, 0.0f, 0.0f, 0.0f);
}


#line 150
__device__ Matrix<float, 2, 2>  s_primal_ctx_covariance_3d_to_2d_0(Camera_0 dpcam_2, float3  dpxyz_ws_1, Matrix<float, 3, 3>  dpcov_ws_0)
{

#line 158
    Matrix<float, 3, 3>  _S189 = makeMatrix<float, 3, 3> (float3 {dpcam_2.world_view_transform_1.rows[int(0)].x, dpcam_2.world_view_transform_1.rows[int(0)].y, dpcam_2.world_view_transform_1.rows[int(0)].z}, float3 {dpcam_2.world_view_transform_1.rows[int(1)].x, dpcam_2.world_view_transform_1.rows[int(1)].y, dpcam_2.world_view_transform_1.rows[int(1)].z}, float3 {dpcam_2.world_view_transform_1.rows[int(2)].x, dpcam_2.world_view_transform_1.rows[int(2)].y, dpcam_2.world_view_transform_1.rows[int(2)].z});

#line 158
    Matrix<float, 3, 3>  _S190 = s_primal_ctx_compute_jacobian_0(dpxyz_ws_1, dpcam_2);

#line 158
    Matrix<float, 3, 3>  _S191 = s_primal_ctx_mul_2(_S190, s_primal_ctx_mul_2(_S189, s_primal_ctx_mul_2(dpcov_ws_0, s_primal_ctx_mul_2(transpose_0(_S189), transpose_0(_S190)))));



    float _S192 = _S191.rows[int(0)].x + 0.30000001192092896f;

#line 162
    Matrix<float, 3, 3>  _S193 = _S191;

#line 162
    *&(((&_S193)->rows + (int(0)))->x) = _S192;

#line 162
    *&(((&_S193)->rows + (int(1)))->y) = _S191.rows[int(1)].y + 0.30000001192092896f;

#line 162
    return makeMatrix<float, 2, 2> (float2 {_S193.rows[int(0)].x, _S193.rows[int(0)].y}, float2 {_S193.rows[int(1)].x, _S193.rows[int(1)].y});
}


#line 162
__device__ Splat_2D_Vertex_0 s_primal_ctx_project_gaussian_to_camera_0(Gaussian_3D_0 dpg_0, Camera_0 dpcam_3, uint active_sh_7)
{

#line 229
    float3  _S194 = s_primal_ctx_project_point_0(dpg_0.xyz_ws_0, dpcam_3);

    bool _S195 = _S194.z <= 0.20000000298023224f;

#line 231
    Splat_2D_Vertex_0 _S196;

#line 231
    if(_S195)
    {

#line 232
        float3  _S197 = make_float3 (0.0f);

#line 232
        Matrix<float, 2, 2>  _S198 = makeMatrix<float, 2, 2> (0.0f);

#line 232
        (&_S196)->xyz_vs_0 = _S197;

#line 232
        (&_S196)->rgb_0 = _S197;

#line 232
        (&_S196)->cov_vs_0 = _S198;

#line 232
    }

#line 232
    bool _S199 = !_S195;

#line 232
    if(_S199)
    {

#line 232
        float3  _S200 = s_primal_ctx_compute_color_from_sh_coeffs_0(dpg_0.sh_coeffs_0, dpg_0.xyz_ws_0, dpcam_3.position_1, active_sh_7);

#line 232
        Matrix<float, 2, 2>  _S201 = s_primal_ctx_covariance_3d_to_2d_0(dpcam_3, dpg_0.xyz_ws_0, s_primal_ctx_get_covariance_from_quat_scales_0(dpg_0.rotations_0, dpg_0.scales_0));

#line 232
        (&_S196)->xyz_vs_0 = _S194;

#line 232
        (&_S196)->rgb_0 = _S200;

#line 232
        (&_S196)->cov_vs_0 = _S201;

#line 232
    }

#line 232
    return _S196;
}


#line 232
__device__ float s_primal_ctx_compute_det_0(Matrix<float, 2, 2>  dpM_0)
{

#line 210
    return dpM_0.rows[int(0)].x * dpM_0.rows[int(1)].y - dpM_0.rows[int(0)].y * dpM_0.rows[int(1)].x;
}


#line 210
__device__ float s_primal_ctx_ndc2pix_0(float dpv_0, int S_1)
{

#line 68
    return ((dpv_0 + 1.0f) * float(S_1) - 1.0f) * 0.5f;
}


#line 5 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
__device__ void s_primal_ctx_vertex_shader_0(DiffTensorView_0 xyz_ws_5, DiffTensorView_0 sh_coeffs_5, DiffTensorView_0 rotations_3, DiffTensorView_0 scales_3, TensorView opcities_0, uint active_sh_8, TensorView world_view_transform_3, TensorView proj_mat_3, TensorView cam_pos_1, TensorView out_tiles_touched_0, TensorView out_rect_tile_space_0, TensorView out_radii_0, DiffTensorView_0 out_xyz_vs_0, DiffTensorView_0 out_inv_cov_vs_0, DiffTensorView_0 out_rgb_0, float fovy_3, float fovx_3, uint image_height_0, uint image_width_0, uint grid_height_0, uint grid_width_0, uint tile_height_0, uint tile_width_0, s_bwd_prop_vertex_shader_Intermediates_0 * _s_diff_ctx_0)
{

#line 75
    Matrix<float, 4, 4>  _S202 = makeMatrix<float, 4, 4> (0.0f);

#line 75
    float3  _S203 = make_float3 (0.0f);

#line 75
    Camera_0 _S204 = { _S202, _S202, _S203, 0.0f, 0.0f, int(0), int(0) };

#line 75
    SpherHarmCoeffs_0 _S205 = { _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203, _S203 };

#line 75
    float4  _S206 = make_float4 (0.0f);

#line 75
    Gaussian_3D_0 _S207 = { _S203, _S205, _S206, _S203 };

#line 75
    _s_diff_ctx_0->_S123 = _S204;

#line 75
    _s_diff_ctx_0->_S124 = _S207;

#line 75
    _s_diff_ctx_0->_S125 = 0.0f;

#line 75
    (&_s_diff_ctx_0->_S123)->world_view_transform_1 = _S202;

#line 75
    (&_s_diff_ctx_0->_S123)->proj_mat_1 = _S202;

#line 75
    (&_s_diff_ctx_0->_S123)->position_1 = _S203;

#line 75
    (&_s_diff_ctx_0->_S123)->fovy_1 = 0.0f;

#line 75
    (&_s_diff_ctx_0->_S123)->fovx_1 = 0.0f;

#line 75
    (&_s_diff_ctx_0->_S123)->H_0 = int(0);

#line 75
    (&_s_diff_ctx_0->_S123)->W_0 = int(0);

#line 75
    (&_s_diff_ctx_0->_S124)->xyz_ws_0 = _S203;

#line 75
    (&_s_diff_ctx_0->_S124)->sh_coeffs_0 = _S205;

#line 75
    (&_s_diff_ctx_0->_S124)->rotations_0 = _S206;

#line 75
    (&_s_diff_ctx_0->_S124)->scales_0 = _S203;

#line 101
    _s_diff_ctx_0->_S125 = 0.0f;

#line 77
    uint g_idx_4 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

#line 77
    bool _S208 = !(g_idx_4 >= DiffTensorView_size_0(xyz_ws_5, 0U));

#line 77
    if(_S208)
    {



        Camera_0 cam_4 = load_camera_0(world_view_transform_3, proj_mat_3, cam_pos_1, fovy_3, fovx_3, image_height_0, image_width_0);

#line 82
        _s_diff_ctx_0->_S123 = cam_4;

#line 82
        Gaussian_3D_0 _S209 = s_primal_ctx_load_gaussian_0(int(g_idx_4), xyz_ws_5, sh_coeffs_5, rotations_3, scales_3, active_sh_8);
        _s_diff_ctx_0->_S124 = _S209;

#line 83
        Splat_2D_Vertex_0 _S210 = s_primal_ctx_project_gaussian_to_camera_0(_S209, cam_4, active_sh_8);

        float _S211 = _S210.xyz_vs_0.z;

#line 85
        bool _bflag_0;

#line 85
        if(_S211 <= 0.20000000298023224f)
        {

#line 85
            _bflag_0 = false;

#line 85
        }
        else
        {

#line 85
            _bflag_0 = _S208;

#line 85
        }

#line 85
        if(_bflag_0)
        {

#line 85
            float _S212 = s_primal_ctx_compute_det_0(_S210.cov_vs_0);

#line 91
            if(_S212 == 0.0f)
            {

#line 91
                _bflag_0 = false;

#line 91
            }

#line 91
            if(_bflag_0)
            {
                float radius_0 = splat_radius_0(_S210.cov_vs_0, _S212);

                Matrix<float, 2, 2>  g_inv_cov_vs_0 = makeMatrix<float, 2, 2> (_S210.cov_vs_0.rows[int(1)].y, - _S210.cov_vs_0.rows[int(0)].y, - _S210.cov_vs_0.rows[int(1)].x, _S210.cov_vs_0.rows[int(0)].x) / makeMatrix<float, 2, 2> (_S212);

                float _S213 = _S210.xyz_vs_0.x;

#line 97
                float _S214 = _S210.xyz_vs_0.y;

#line 97
                float2  pixelspace_xy_0 = make_float2 (s_primal_ctx_ndc2pix_0(_S213, int(image_width_0)), s_primal_ctx_ndc2pix_0(_S214, int(image_height_0)));



                float3  _S215 = make_float3 (g_inv_cov_vs_0.rows[int(0)].x, g_inv_cov_vs_0.rows[int(0)].y, g_inv_cov_vs_0.rows[int(1)].y);

#line 101
                float _S216 = ((opcities_0).load<float>((g_idx_4)));

#line 101
                _s_diff_ctx_0->_S125 = _S216;

#line 101
                int2  _S217 = make_int2 (int(grid_width_0), int(grid_height_0));

#line 101
                int2  _S218 = make_int2 (int(tile_width_0), int(tile_height_0));

#line 101
                uint _S219 = 0U;

#line 101
                rectangle_0 rect_tile_space_0 = computeSnugBox_0(_S215, pixelspace_xy_0, _S216, _S217, _S218, &_S219);

#line 101
                uint n_tiles_1 = _S219;

                if(_S219 == 0U)
                {

#line 103
                    _bflag_0 = false;

#line 103
                }

#line 103
                if(_bflag_0)
                {


                    (out_radii_0).store<int>((g_idx_4), (int(uint(radius_0))));
                    (out_tiles_touched_0).store<int>((g_idx_4), (int(n_tiles_1)));
                    uint2  _S220 = make_uint2 (g_idx_4, 0U);

#line 109
                    (out_rect_tile_space_0).store<int>((g_idx_4), (0U), (rect_tile_space_0.min_x_0));
                    uint2  _S221 = make_uint2 (g_idx_4, 1U);

#line 110
                    (out_rect_tile_space_0).store<int>((g_idx_4), (1U), (rect_tile_space_0.min_y_0));
                    uint2  _S222 = make_uint2 (g_idx_4, 2U);

#line 111
                    (out_rect_tile_space_0).store<int>((g_idx_4), (2U), (rect_tile_space_0.max_x_0));
                    (out_rect_tile_space_0).store<int>((g_idx_4), (3U), (rect_tile_space_0.max_y_0));

                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S220, _S213);
                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S221, _S214);
                    DiffTensorView_storeOnce_0(out_xyz_vs_0, _S222, _S211);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 0U, 0U), g_inv_cov_vs_0.rows[int(0)].x);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 0U, 1U), g_inv_cov_vs_0.rows[int(0)].y);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 1U, 0U), g_inv_cov_vs_0.rows[int(1)].x);
                    DiffTensorView_storeOnce_1(out_inv_cov_vs_0, make_uint3 (g_idx_4, 1U, 1U), g_inv_cov_vs_0.rows[int(1)].y);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S220, _S210.rgb_0.x);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S221, _S210.rgb_0.y);
                    DiffTensorView_storeOnce_0(out_rgb_0, _S222, _S210.rgb_0.z);

#line 123
                }

#line 123
            }

#line 123
        }

#line 123
    }

#line 123
    return;
}


#line 68 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ void s_bwd_prop_ndc2pix_0(DiffPair_float_0 * dpv_1, int S_2, float _s_dOut_0)
{
    float _S223 = float(S_2) * (0.5f * _s_dOut_0);

#line 70
    dpv_1->primal_1 = (*dpv_1).primal_1;

#line 70
    dpv_1->differential_0 = _S223;

#line 68
    return;
}


#line 68
struct DiffPair_matrixx3Cfloatx2C2x2C2x3E_0
{
    Matrix<float, 2, 2>  primal_1;
    Matrix<float, 2, 2>  differential_0;
};


#line 210
__device__ void s_bwd_prop_compute_det_0(DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 * dpM_1, float _s_dOut_1)
{

#line 211
    float _S224 = - _s_dOut_1;

#line 211
    float _S225 = (*dpM_1).primal_1.rows[int(0)].y * _S224;

#line 211
    float _S226 = (*dpM_1).primal_1.rows[int(1)].x * _S224;

#line 211
    float _S227 = (*dpM_1).primal_1.rows[int(0)].x * _s_dOut_1;

#line 211
    float _S228 = (*dpM_1).primal_1.rows[int(1)].y * _s_dOut_1;

#line 1751 "core.meta.slang"
    float2  _S229 = make_float2 (0.0f);

#line 1751
    float2  _S230 = _S229;

#line 1751
    *&((&_S230)->x) = _S225;

#line 1751
    *&((&_S230)->y) = _S227;

#line 1751
    float2  _S231 = _S229;

#line 1751
    *&((&_S231)->y) = _S226;

#line 1751
    *&((&_S231)->x) = _S228;

#line 1751
    Matrix<float, 2, 2>  _S232 = makeMatrix<float, 2, 2> (0.0f);

#line 1751
    _S232[int(1)] = _S230;

#line 1751
    _S232[int(0)] = _S231;

#line 1751
    dpM_1->primal_1 = (*dpM_1).primal_1;

#line 1751
    dpM_1->differential_0 = _S232;

#line 210 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    return;
}


#line 210
struct DiffPair_Gaussian_3D_0
{
    Gaussian_3D_0 primal_1;
    Gaussian_3D_0 differential_0;
};


#line 84 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
struct DiffPair_Camera_0
{
    Camera_0 primal_1;
    Camera_Differential_0 differential_0;
};


#line 236 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ void s_bwd_prop_mul_0(DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S233, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * _S234, Matrix<float, 3, 3>  _S235)
{

#line 236
    mul_1(_S233, _S234, _S235);

#line 236
    return;
}


#line 236
__device__ void s_bwd_prop_min_0(DiffPair_float_0 * _S236, DiffPair_float_0 * _S237, float _S238)
{

#line 236
    _d_min_0(_S236, _S237, _S238);

#line 236
    return;
}


#line 236
__device__ void s_bwd_prop_max_0(DiffPair_float_0 * _S239, DiffPair_float_0 * _S240, float _S241)
{

#line 236
    _d_max_0(_S239, _S240, _S241);

#line 236
    return;
}


#line 235
__device__ void s_bwd_prop_mul_1(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S242, DiffPair_vectorx3Cfloatx2C4x3E_0 * _S243, float4  _S244)
{

#line 235
    _d_mul_0(_S242, _S243, _S244);

#line 235
    return;
}


#line 112
__device__ void s_bwd_prop_geom_transform_points_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_3, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dptransf_matrix_2, float3  _s_dOut_2)
{
    float4  _S245 = make_float4 ((*dppoint_3).primal_1.x, (*dppoint_3).primal_1.y, (*dppoint_3).primal_1.z, 1.0f);

#line 114
    float4  _S246 = s_primal_ctx_mul_1((*dptransf_matrix_2).primal_1, _S245);
    float _S247 = _S246.w + 1.00000001168609742e-07f;

#line 115
    float3  _S248 = _s_dOut_2 / make_float3 (_S247 * _S247);

#line 115
    float3  _S249 = float3 {_S246.x, _S246.y, _S246.z} * - _S248;

#line 115
    float3  _S250 = make_float3 (_S247) * _S248;

#line 114
    float4  _S251 = make_float4 (_S250.x, _S250.y, _S250.z, _S249.x + _S249.y + _S249.z);

#line 114
    Matrix<float, 4, 4>  _S252 = makeMatrix<float, 4, 4> (0.0f);

#line 114
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S253;

#line 114
    (&_S253)->primal_1 = (*dptransf_matrix_2).primal_1;

#line 114
    (&_S253)->differential_0 = _S252;

#line 114
    float4  _S254 = make_float4 (0.0f);

#line 114
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S255;

#line 114
    (&_S255)->primal_1 = _S245;

#line 114
    (&_S255)->differential_0 = _S254;

#line 114
    s_bwd_prop_mul_1(&_S253, &_S255, _S251);

#line 114
    float3  _S256 = float3 {_S255.differential_0.x, _S255.differential_0.y, _S255.differential_0.z};

#line 114
    dptransf_matrix_2->primal_1 = (*dptransf_matrix_2).primal_1;

#line 114
    dptransf_matrix_2->differential_0 = _S253.differential_0;

#line 114
    dppoint_3->primal_1 = (*dppoint_3).primal_1;

#line 114
    dppoint_3->differential_0 = _S256;

#line 112
    return;
}


#line 112
__device__ void s_bwd_prop_tan_0(DiffPair_float_0 * _S257, float _S258)
{

#line 112
    _d_tan_0(_S257, _S258);

#line 112
    return;
}


#line 134
__device__ void s_bwd_prop_compute_jacobian_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpxyz_ws_2, DiffPair_Camera_0 * dpcam_4, Matrix<float, 3, 3>  s_diff_J_T_0)
{

#line 135
    float _S259 = (*dpcam_4).primal_1.fovx_1 / 2.0f;

#line 135
    float _S260 = s_primal_ctx_tan_0(_S259);
    float _S261 = (*dpcam_4).primal_1.fovy_1 / 2.0f;

#line 136
    float _S262 = s_primal_ctx_tan_0(_S261);
    float _S263 = float((*dpcam_4).primal_1.W_0);

#line 137
    float _S264 = 2.0f * _S260;

#line 137
    float h_x_2 = _S263 / _S264;

#line 137
    float _S265 = _S264 * _S264;
    float _S266 = float((*dpcam_4).primal_1.H_0);

#line 138
    float _S267 = 2.0f * _S262;

#line 138
    float h_y_2 = _S266 / _S267;

#line 138
    float _S268 = _S267 * _S267;

#line 138
    float3  _S269 = s_primal_ctx_geom_transform_points_0((*dpxyz_ws_2).primal_1, (*dpcam_4).primal_1.world_view_transform_1);

#line 143
    float limx_2 = 1.29999995231628418f * _S260;
    float limy_2 = 1.29999995231628418f * _S262;
    float _S270 = _S269.x;

#line 145
    float _S271 = _S269.z;

#line 145
    float txtz_0 = _S270 / _S271;

#line 145
    float _S272 = _S271 * _S271;
    float _S273 = _S269.y;

#line 146
    float tytz_2 = _S273 / _S271;
    float _S274 = - limx_2;

#line 147
    float _S275 = s_primal_ctx_max_1(_S274, txtz_0);

#line 147
    float _S276 = s_primal_ctx_min_0(limx_2, _S275);

#line 147
    float _S277 = _S276 * _S271;

#line 147
    float3  _S278 = _S269;

#line 147
    *&((&_S278)->x) = _S277;
    float _S279 = - limy_2;

#line 148
    float _S280 = s_primal_ctx_max_1(_S279, tytz_2);

#line 148
    float _S281 = s_primal_ctx_min_0(limy_2, _S280);

#line 148
    float _S282 = _S278.z;

#line 148
    *&((&_S278)->y) = _S281 * _S282;

    float _S283 = _S278.z;

#line 150
    float _S284 = _S283 * _S283;

#line 150
    float _S285 = _S278.x;

#line 150
    float _S286 = _S284 * _S284;
    float _S287 = _S278.y;

#line 151
    float _S288 = s_diff_J_T_0.rows[int(1)].z / _S286;

#line 151
    float _S289 = - (_S284 * _S288);

#line 151
    float _S290 = h_y_2 * _S289;

#line 151
    float _S291 = _S287 * _S289;

#line 151
    float _S292 = s_diff_J_T_0.rows[int(1)].y / _S284;

#line 151
    float _S293 = _S283 * _S292;

#line 150
    float _S294 = s_diff_J_T_0.rows[int(0)].z / _S286;

#line 150
    float _S295 = _S283 * (- (h_y_2 * _S287) * - _S288 + - (h_x_2 * _S285) * - _S294);

#line 150
    float _S296 = - (_S284 * _S294);

#line 150
    float _S297 = _S285 * _S296;

#line 150
    float _S298 = s_diff_J_T_0.rows[int(0)].x / _S284;

#line 150
    float _S299 = _S283 * _S298;

#line 150
    _S278 = make_float3 (h_x_2 * _S296, _S290, h_y_2 * - _S292 + _S295 + _S295 + h_x_2 * - _S298);

#line 150
    *&((&_S278)->y) = 0.0f;

#line 148
    float _S300 = _S281 * _S290;

#line 148
    float _S301 = _S282 * _S290;

#line 148
    DiffPair_float_0 _S302;

#line 148
    (&_S302)->primal_1 = limy_2;

#line 148
    (&_S302)->differential_0 = 0.0f;

#line 148
    DiffPair_float_0 _S303;

#line 148
    (&_S303)->primal_1 = _S280;

#line 148
    (&_S303)->differential_0 = 0.0f;

#line 148
    s_bwd_prop_min_0(&_S302, &_S303, _S301);

#line 148
    DiffPair_float_0 _S304;

#line 148
    (&_S304)->primal_1 = _S279;

#line 148
    (&_S304)->differential_0 = 0.0f;

#line 148
    DiffPair_float_0 _S305;

#line 148
    (&_S305)->primal_1 = tytz_2;

#line 148
    (&_S305)->differential_0 = 0.0f;

#line 148
    s_bwd_prop_max_0(&_S304, &_S305, _S303.differential_0);

#line 148
    float _S306 = - _S304.differential_0;

#line 147
    float3  _S307 = _S278 + make_float3 (0.0f, 0.0f, _S300);

#line 147
    _S278 = _S307;

#line 147
    *&((&_S278)->x) = 0.0f;

#line 147
    float _S308 = _S276 * _S307.x;

#line 147
    float _S309 = _S271 * _S307.x;

#line 147
    DiffPair_float_0 _S310;

#line 147
    (&_S310)->primal_1 = limx_2;

#line 147
    (&_S310)->differential_0 = 0.0f;

#line 147
    DiffPair_float_0 _S311;

#line 147
    (&_S311)->primal_1 = _S275;

#line 147
    (&_S311)->differential_0 = 0.0f;

#line 147
    s_bwd_prop_min_0(&_S310, &_S311, _S309);

#line 147
    DiffPair_float_0 _S312;

#line 147
    (&_S312)->primal_1 = _S274;

#line 147
    (&_S312)->differential_0 = 0.0f;

#line 147
    DiffPair_float_0 _S313;

#line 147
    (&_S313)->primal_1 = txtz_0;

#line 147
    (&_S313)->differential_0 = 0.0f;

#line 147
    s_bwd_prop_max_0(&_S312, &_S313, _S311.differential_0);

#line 146
    float _S314 = _S305.differential_0 / _S272;

#line 145
    float _S315 = _S313.differential_0 / _S272;

#line 144
    float _S316 = 1.29999995231628418f * (_S302.differential_0 + _S306);

#line 143
    float _S317 = 1.29999995231628418f * (_S310.differential_0 + - _S312.differential_0);

#line 140
    float3  _S318 = _S278 + make_float3 (_S271 * _S315, _S271 * _S314, _S308 + _S273 * - _S314 + _S270 * - _S315);

#line 140
    float3  _S319 = make_float3 (0.0f);

#line 140
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S320;

#line 140
    (&_S320)->primal_1 = (*dpxyz_ws_2).primal_1;

#line 140
    (&_S320)->differential_0 = _S319;

#line 140
    Matrix<float, 4, 4>  _S321 = makeMatrix<float, 4, 4> (0.0f);

#line 140
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S322;

#line 140
    (&_S322)->primal_1 = (*dpcam_4).primal_1.world_view_transform_1;

#line 140
    (&_S322)->differential_0 = _S321;

#line 140
    s_bwd_prop_geom_transform_points_0(&_S320, &_S322, _S318);

#line 137
    float _S323 = 2.0f * (_S263 * - ((_S297 + _S299) / _S265));

#line 136
    float _S324 = _S316 + 2.0f * (_S266 * - ((_S291 + _S293) / _S268));

#line 136
    DiffPair_float_0 _S325;

#line 136
    (&_S325)->primal_1 = _S261;

#line 136
    (&_S325)->differential_0 = 0.0f;

#line 136
    s_bwd_prop_tan_0(&_S325, _S324);

#line 136
    float _S326 = 0.5f * _S325.differential_0;

#line 135
    float _S327 = _S317 + _S323;

#line 135
    DiffPair_float_0 _S328;

#line 135
    (&_S328)->primal_1 = _S259;

#line 135
    (&_S328)->differential_0 = 0.0f;

#line 135
    s_bwd_prop_tan_0(&_S328, _S327);

#line 135
    float _S329 = 0.5f * _S328.differential_0;

#line 135
    Camera_Differential_0 _S330 = Camera_x24_syn_dzero_0();

#line 135
    (&_S330)->world_view_transform_0 = _S322.differential_0;

#line 135
    (&_S330)->fovy_0 = _S326;

#line 135
    (&_S330)->fovx_0 = _S329;

#line 135
    dpcam_4->primal_1 = (*dpcam_4).primal_1;

#line 135
    dpcam_4->differential_0 = _S330;

#line 135
    dpxyz_ws_2->primal_1 = (*dpxyz_ws_2).primal_1;

#line 135
    dpxyz_ws_2->differential_0 = _S320.differential_0;

#line 134
    return;
}


#line 158
__device__ void s_bwd_prop_covariance_3d_to_2d_0(DiffPair_Camera_0 * dpcam_5, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpxyz_ws_3, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 * dpcov_ws_1, Matrix<float, 2, 2>  _s_dOut_3)
{

#line 158
    Matrix<float, 3, 3>  _S331 = makeMatrix<float, 3, 3> (float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(0)].z}, float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(1)].z}, float3 {(*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].x, (*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].y, (*dpcam_5).primal_1.world_view_transform_1.rows[int(2)].z});

#line 158
    Matrix<float, 3, 3>  _S332 = s_primal_ctx_compute_jacobian_0((*dpxyz_ws_3).primal_1, (*dpcam_5).primal_1);


    Matrix<float, 3, 3>  _S333 = transpose_0(_S331);

#line 161
    Matrix<float, 3, 3>  _S334 = transpose_0(_S332);

#line 161
    Matrix<float, 3, 3>  _S335 = s_primal_ctx_mul_2(_S333, _S334);

#line 161
    Matrix<float, 3, 3>  _S336 = s_primal_ctx_mul_2((*dpcov_ws_1).primal_1, _S335);

#line 161
    Matrix<float, 3, 3>  _S337 = s_primal_ctx_mul_2(_S331, _S336);

#line 161
    float3  _S338 = make_float3 (_s_dOut_3.rows[int(1)].x, _s_dOut_3.rows[int(1)].y, 0.0f);

#line 161
    float3  _S339 = make_float3 (_s_dOut_3.rows[int(0)].x, _s_dOut_3.rows[int(0)].y, 0.0f);

    Matrix<float, 3, 3>  _S340 = makeMatrix<float, 3, 3> (0.0f);

#line 163
    Matrix<float, 3, 3>  _S341 = _S340;

#line 163
    _S341[int(1)] = _S338;

#line 163
    _S341[int(0)] = _S339;

#line 163
    Matrix<float, 3, 3>  _S342 = _S341;

#line 163
    *&(((&_S342)->rows + (int(1)))->y) = 0.0f;

#line 1751 "core.meta.slang"
    float3  _S343 = make_float3 (0.0f);

#line 1751
    float3  _S344 = _S343;

#line 1751
    *&((&_S344)->y) = _S341.rows[int(1)].y;

#line 1751
    *&(((&_S342)->rows + (int(0)))->x) = 0.0f;

#line 162 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    float3  _S345 = _S343;

#line 162
    *&((&_S345)->x) = _S341.rows[int(0)].x;

#line 161
    Matrix<float, 3, 3>  _S346 = _S340;

#line 161
    _S346[int(1)] = _S344;

#line 161
    _S346[int(0)] = _S345;

#line 161
    Matrix<float, 3, 3>  _S347 = _S342 + _S346;

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S348;

#line 161
    (&_S348)->primal_1 = _S332;

#line 161
    (&_S348)->differential_0 = _S340;

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S349;

#line 161
    (&_S349)->primal_1 = _S337;

#line 161
    (&_S349)->differential_0 = _S340;

#line 161
    s_bwd_prop_mul_0(&_S348, &_S349, _S347);

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S350;

#line 161
    (&_S350)->primal_1 = _S331;

#line 161
    (&_S350)->differential_0 = _S340;

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S351;

#line 161
    (&_S351)->primal_1 = _S336;

#line 161
    (&_S351)->differential_0 = _S340;

#line 161
    s_bwd_prop_mul_0(&_S350, &_S351, _S349.differential_0);

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S352;

#line 161
    (&_S352)->primal_1 = (*dpcov_ws_1).primal_1;

#line 161
    (&_S352)->differential_0 = _S340;

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S353;

#line 161
    (&_S353)->primal_1 = _S335;

#line 161
    (&_S353)->differential_0 = _S340;

#line 161
    s_bwd_prop_mul_0(&_S352, &_S353, _S351.differential_0);

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S354;

#line 161
    (&_S354)->primal_1 = _S333;

#line 161
    (&_S354)->differential_0 = _S340;

#line 161
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S355;

#line 161
    (&_S355)->primal_1 = _S334;

#line 161
    (&_S355)->differential_0 = _S340;

#line 161
    s_bwd_prop_mul_0(&_S354, &_S355, _S353.differential_0);

#line 161
    Matrix<float, 3, 3>  _S356 = transpose_0(_S354.differential_0);

#line 160
    Matrix<float, 3, 3>  _S357 = _S348.differential_0 + transpose_0(_S355.differential_0);

#line 160
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S358;

#line 160
    (&_S358)->primal_1 = (*dpxyz_ws_3).primal_1;

#line 160
    (&_S358)->differential_0 = _S343;

#line 160
    Camera_Differential_0 _S359 = Camera_x24_syn_dzero_0();

#line 160
    DiffPair_Camera_0 _S360;

#line 160
    (&_S360)->primal_1 = (*dpcam_5).primal_1;

#line 160
    (&_S360)->differential_0 = _S359;

#line 160
    s_bwd_prop_compute_jacobian_0(&_S358, &_S360, _S357);

#line 160
    Matrix<float, 3, 3>  _S361 = _S350.differential_0 + _S356;

#line 160
    float4  _S362 = make_float4 (_S361.rows[int(2)].x, _S361.rows[int(2)].y, _S361.rows[int(2)].z, 0.0f);

#line 160
    float4  _S363 = make_float4 (_S361.rows[int(1)].x, _S361.rows[int(1)].y, _S361.rows[int(1)].z, 0.0f);

#line 160
    float4  _S364 = make_float4 (_S361.rows[int(0)].x, _S361.rows[int(0)].y, _S361.rows[int(0)].z, 0.0f);

#line 160
    Matrix<float, 4, 4>  _S365 = makeMatrix<float, 4, 4> (0.0f);

#line 160
    _S365[int(2)] = _S362;

#line 160
    _S365[int(1)] = _S363;

#line 160
    _S365[int(0)] = _S364;

#line 160
    dpcov_ws_1->primal_1 = (*dpcov_ws_1).primal_1;

#line 160
    dpcov_ws_1->differential_0 = _S352.differential_0;

#line 160
    dpxyz_ws_3->primal_1 = (*dpxyz_ws_3).primal_1;

#line 160
    dpxyz_ws_3->differential_0 = _S358.differential_0;

#line 160
    Camera_Differential_0 _S366 = _S359;

#line 160
    (&_S366)->world_view_transform_0 = _S365;

#line 160
    Camera_Differential_0 _S367 = Camera_x24_syn_dadd_0(_S360.differential_0, _S366);

#line 160
    dpcam_5->primal_1 = (*dpcam_5).primal_1;

#line 160
    dpcam_5->differential_0 = _S367;

#line 158
    return;
}


#line 287
__device__ void s_bwd_prop_get_covariance_from_quat_scales_0(DiffPair_vectorx3Cfloatx2C4x3E_0 * dpq_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dps_1, Matrix<float, 3, 3>  _s_dOut_4)
{

#line 287
    float _S368 = (*dpq_1).primal_1.z;



    float _S369 = _S368 * _S368;

#line 291
    float _S370 = (*dpq_1).primal_1.w * (*dpq_1).primal_1.w;

#line 291
    float _S371 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.z;

#line 291
    float _S372 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.w;

#line 291
    float _S373 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.w;

#line 291
    float _S374 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.z;
    float _S375 = (*dpq_1).primal_1.y * (*dpq_1).primal_1.y;

#line 292
    float _S376 = (*dpq_1).primal_1.z * (*dpq_1).primal_1.w;

#line 292
    float _S377 = (*dpq_1).primal_1.x * (*dpq_1).primal_1.y;

#line 290
    Matrix<float, 3, 3>  rotation_matrix_0 = makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S369 + _S370), 2.0f * (_S371 - _S372), 2.0f * (_S373 + _S374), 2.0f * (_S371 + _S372), 1.0f - 2.0f * (_S375 + _S370), 2.0f * (_S376 - _S377), 2.0f * (_S373 - _S374), 2.0f * (_S376 + _S377), 1.0f - 2.0f * (_S375 + _S369));

#line 295
    Matrix<float, 3, 3>  scales_matrix_0 = makeMatrix<float, 3, 3> ((*dps_1).primal_1.x, 0.0f, 0.0f, 0.0f, (*dps_1).primal_1.y, 0.0f, 0.0f, 0.0f, (*dps_1).primal_1.z);

#line 295
    Matrix<float, 3, 3>  _S378 = s_primal_ctx_mul_2(rotation_matrix_0, scales_matrix_0);

#line 301
    Matrix<float, 3, 3>  _S379 = transpose_0(_S378);

#line 301
    Matrix<float, 3, 3>  _S380 = makeMatrix<float, 3, 3> (0.0f);

#line 301
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S381;

#line 301
    (&_S381)->primal_1 = _S378;

#line 301
    (&_S381)->differential_0 = _S380;

#line 301
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S382;

#line 301
    (&_S382)->primal_1 = _S379;

#line 301
    (&_S382)->differential_0 = _S380;

#line 301
    s_bwd_prop_mul_0(&_S381, &_S382, _s_dOut_4);

#line 299
    Matrix<float, 3, 3>  _S383 = _S381.differential_0 + transpose_0(_S382.differential_0);

#line 299
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S384;

#line 299
    (&_S384)->primal_1 = rotation_matrix_0;

#line 299
    (&_S384)->differential_0 = _S380;

#line 299
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S385;

#line 299
    (&_S385)->primal_1 = scales_matrix_0;

#line 299
    (&_S385)->differential_0 = _S380;

#line 299
    s_bwd_prop_mul_0(&_S384, &_S385, _S383);

#line 293
    float _S386 = 2.0f * - _S384.differential_0.rows[int(2)].z;

#line 293
    float _S387 = 2.0f * _S384.differential_0.rows[int(2)].y;

#line 293
    float _S388 = 2.0f * _S384.differential_0.rows[int(2)].x;

#line 292
    float _S389 = 2.0f * _S384.differential_0.rows[int(1)].z;

#line 292
    float _S390 = _S387 + - _S389;

#line 292
    float _S391 = _S387 + _S389;

#line 292
    float _S392 = 2.0f * - _S384.differential_0.rows[int(1)].y;

#line 292
    float _S393 = (*dpq_1).primal_1.y * (_S386 + _S392);

#line 292
    float _S394 = 2.0f * _S384.differential_0.rows[int(1)].x;

#line 291
    float _S395 = 2.0f * _S384.differential_0.rows[int(0)].z;

#line 291
    float _S396 = - _S388 + _S395;

#line 291
    float _S397 = _S388 + _S395;

#line 291
    float _S398 = 2.0f * _S384.differential_0.rows[int(0)].y;

#line 291
    float _S399 = _S394 + - _S398;

#line 291
    float _S400 = _S394 + _S398;

#line 291
    float _S401 = 2.0f * - _S384.differential_0.rows[int(0)].x;

#line 291
    float _S402 = (*dpq_1).primal_1.w * (_S392 + _S401);

#line 291
    float _S403 = (*dpq_1).primal_1.z * (_S386 + _S401);

#line 958 "core.meta.slang"
    float _S404 = (*dpq_1).primal_1.z * _S391 + (*dpq_1).primal_1.y * _S397 + (*dpq_1).primal_1.x * _S399 + _S402 + _S402;

#line 958
    float _S405 = (*dpq_1).primal_1.w * _S391 + (*dpq_1).primal_1.x * _S396 + (*dpq_1).primal_1.y * _S400 + _S403 + _S403;

#line 958
    float _S406 = (*dpq_1).primal_1.x * _S390 + _S393 + _S393 + (*dpq_1).primal_1.w * _S397 + (*dpq_1).primal_1.z * _S400;

#line 958
    float _S407 = (*dpq_1).primal_1.y * _S390 + (*dpq_1).primal_1.z * _S396 + (*dpq_1).primal_1.w * _S399;

#line 958
    float3  _S408 = make_float3 (0.0f);

#line 958
    *&((&_S408)->z) = _S385.differential_0.rows[int(2)].z;

#line 958
    *&((&_S408)->y) = _S385.differential_0.rows[int(1)].y;

#line 958
    *&((&_S408)->x) = _S385.differential_0.rows[int(0)].x;

#line 958
    dps_1->primal_1 = (*dps_1).primal_1;

#line 958
    dps_1->differential_0 = _S408;

#line 958
    float4  _S409 = make_float4 (0.0f);

#line 958
    *&((&_S409)->w) = _S404;

#line 958
    *&((&_S409)->z) = _S405;

#line 958
    *&((&_S409)->y) = _S406;

#line 958
    *&((&_S409)->x) = _S407;

#line 958
    dpq_1->primal_1 = (*dpq_1).primal_1;

#line 958
    dpq_1->differential_0 = _S409;

#line 287 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    return;
}


#line 287
struct DiffPair_SpherHarmCoeffs_0
{
    SpherHarmCoeffs_0 primal_1;
    SpherHarmCoeffs_0 differential_0;
};


#line 234
__device__ void s_bwd_prop_max_1(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S410, DiffPair_vectorx3Cfloatx2C3x3E_0 * _S411, float3  _S412)
{

#line 234
    _d_max_vector_0(_S410, _S411, _S412);

#line 234
    return;
}


#line 2117 "diff.meta.slang"
__device__ void s_bwd_prop_sqrt_0(DiffPair_float_0 * _S413, float _S414)
{

#line 2117
    _d_sqrt_0(_S413, _S414);

#line 2117
    return;
}


#line 2092
__device__ void s_bwd_prop_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_10, float _s_dOut_5)
{

#line 2092
    float _S415 = (*dpx_10).primal_1.x;

#line 2092
    float _S416 = (*dpx_10).primal_1.y;

#line 2092
    float _S417 = (*dpx_10).primal_1.z;

#line 2099
    DiffPair_float_0 _S418;

#line 2099
    (&_S418)->primal_1 = _S415 * _S415 + _S416 * _S416 + _S417 * _S417;

#line 2099
    (&_S418)->differential_0 = 0.0f;

#line 2099
    s_bwd_prop_sqrt_0(&_S418, _s_dOut_5);

#line 2099
    float _S419 = (*dpx_10).primal_1.z * _S418.differential_0;

#line 958 "core.meta.slang"
    float _S420 = _S419 + _S419;

#line 958
    float _S421 = (*dpx_10).primal_1.y * _S418.differential_0;

#line 958
    float _S422 = _S421 + _S421;

#line 958
    float _S423 = (*dpx_10).primal_1.x * _S418.differential_0;

#line 958
    float _S424 = _S423 + _S423;

#line 958
    float3  _S425 = make_float3 (0.0f);

#line 958
    *&((&_S425)->z) = _S420;

#line 958
    *&((&_S425)->y) = _S422;

#line 958
    *&((&_S425)->x) = _S424;

#line 958
    dpx_10->primal_1 = (*dpx_10).primal_1;

#line 958
    dpx_10->differential_0 = _S425;

#line 2092 "diff.meta.slang"
    return;
}


#line 2092
__device__ void s_bwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S426, float _S427)
{

#line 2092
    s_bwd_prop_length_impl_0(_S426, _S427);

#line 2092
    return;
}


#line 2154
__device__ void s_bwd_prop_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dpx_11, float3  _s_dOut_6)
{
    float _S428 = length_0((*dpx_11).primal_1);
    float3  _S429 = (*dpx_11).primal_1 * _s_dOut_6;

#line 2157
    float3  _S430 = make_float3 (1.0f / _S428) * _s_dOut_6;

#line 2157
    float _S431 = - ((_S429.x + _S429.y + _S429.z) / (_S428 * _S428));

#line 2156
    float3  _S432 = make_float3 (0.0f);

#line 2156
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S433;

#line 2156
    (&_S433)->primal_1 = (*dpx_11).primal_1;

#line 2156
    (&_S433)->differential_0 = _S432;

#line 2156
    s_bwd_length_impl_0(&_S433, _S431);

#line 2156
    float3  _S434 = _S430 + _S433.differential_0;

#line 2156
    dpx_11->primal_1 = (*dpx_11).primal_1;

#line 2156
    dpx_11->differential_0 = _S434;

#line 2154
    return;
}


#line 2154
__device__ void s_bwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * _S435, float3  _S436)
{

#line 2154
    s_bwd_prop_normalize_impl_0(_S435, _S436);

#line 2154
    return;
}


#line 94 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
__device__ void s_bwd_prop_compute_color_from_sh_coeffs_0(DiffPair_SpherHarmCoeffs_0 * dpsh_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpg_xyz_ws_1, DiffPair_vectorx3Cfloatx2C3x3E_0 * dpcam_pos_1, uint active_sh_9, float3  _s_dOut_7)
{

#line 94
    DiffPair_SpherHarmCoeffs_0 _S437 = *dpsh_1;

#line 100
    float3  _S438 = make_float3 (0.0f);

#line 95
    float3  dir_1 = (*dpg_xyz_ws_1).primal_1 - (*dpcam_pos_1).primal_1;
    float3  _S439 = normalize_0(dir_1);

    float3  rgb_9 = make_float3 (0.282094806432724f) * (*dpsh_1).primal_1.coeff0_0;
    bool _S440 = active_sh_9 > 0U;

#line 99
    float3  rgb_10;

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
    float3  _S453;

#line 99
    float3  _S454;

#line 99
    float3  _S455;

#line 99
    float3  _S456;

#line 99
    float3  _S457;

#line 99
    float3  _S458;

#line 99
    float3  _S459;

#line 99
    float3  _S460;

#line 99
    float3  _S461;

#line 99
    float3  _S462;

#line 99
    float3  _S463;

#line 99
    float3  _S464;

#line 99
    float3  _S465;

#line 99
    float3  _S466;

#line 99
    float3  _S467;

#line 99
    float3  _S468;

#line 99
    float3  _S469;

#line 99
    float3  _S470;

#line 99
    float _S471;

#line 99
    float _S472;

#line 99
    float _S473;

#line 99
    float _S474;

#line 99
    float _S475;

#line 99
    float _S476;

#line 99
    float _S477;

#line 99
    float _S478;

#line 99
    float _S479;

#line 99
    float _S480;

#line 99
    float _S481;

#line 99
    float _S482;

#line 99
    float _S483;

#line 99
    float _S484;

#line 99
    float _S485;

#line 99
    bool _S486;

#line 99
    bool _S487;

#line 99
    if(_S440)
    {

#line 100
        float _S488 = _S439.y;

#line 100
        float _S489 = 0.48860251903533936f * _S488;

#line 100
        float3  _S490 = make_float3 (_S489);

#line 100
        float _S491 = _S439.z;

#line 100
        float _S492 = 0.48860251903533936f * _S491;

#line 100
        float3  _S493 = make_float3 (_S492);

#line 100
        float _S494 = _S439.x;

#line 100
        float _S495 = 0.48860251903533936f * _S494;

#line 100
        float3  _S496 = make_float3 (_S495);

#line 100
        float3  rgb_11 = rgb_9 - make_float3 (_S489) * _S437.primal_1.coeff1_0 + make_float3 (_S492) * _S437.primal_1.coeff2_0 - make_float3 (_S495) * _S437.primal_1.coeff3_0;
        bool _S497 = active_sh_9 > 1U;

#line 101
        if(_S497)
        {
            float xx_2 = _S494 * _S494;

#line 103
            float yy_2 = _S488 * _S488;

#line 103
            float zz_2 = _S491 * _S491;
            float xy_2 = _S494 * _S488;

            float _S498 = 1.09254848957061768f * xy_2;

#line 106
            float3  _S499 = make_float3 (_S498);
            float _S500 = -1.09254848957061768f * (_S488 * _S491);

#line 107
            float3  _S501 = make_float3 (_S500);
            float _S502 = 2.0f * zz_2;

#line 108
            float _S503 = 0.31539157032966614f * (_S502 - xx_2 - yy_2);

#line 108
            float3  _S504 = make_float3 (_S503);
            float _S505 = -1.09254848957061768f * (_S494 * _S491);

#line 109
            float3  _S506 = make_float3 (_S505);
            float _S507 = xx_2 - yy_2;

#line 110
            float _S508 = 0.54627424478530884f * _S507;

#line 110
            float3  _S509 = make_float3 (_S508);

#line 109
            float3  rgb_12 = rgb_11 + make_float3 (_S498) * _S437.primal_1.coeff4_0 + make_float3 (_S500) * _S437.primal_1.coeff5_0 + make_float3 (_S503) * _S437.primal_1.coeff6_0 + make_float3 (_S505) * _S437.primal_1.coeff7_0 + make_float3 (_S508) * _S437.primal_1.coeff8_0;


            bool _S510 = active_sh_9 > 2U;

#line 112
            if(_S510)
            {

                float _S511 = -0.59004360437393188f * _S488;

#line 115
                float _S512 = 3.0f * xx_2;

#line 115
                float _S513 = _S512 - yy_2;

#line 115
                float _S514 = _S511 * _S513;

#line 115
                float3  _S515 = make_float3 (_S514);
                float _S516 = 2.89061141014099121f * xy_2;

#line 116
                float _S517 = _S516 * _S491;

#line 116
                float3  _S518 = make_float3 (_S517);
                float _S519 = -0.4570457935333252f * _S488;

#line 117
                float _S520 = 4.0f * zz_2 - xx_2 - yy_2;

#line 117
                float _S521 = _S519 * _S520;

#line 117
                float3  _S522 = make_float3 (_S521);
                float _S523 = 0.37317633628845215f * _S491;

#line 118
                float _S524 = 3.0f * yy_2;

#line 118
                float _S525 = _S502 - _S512 - _S524;

#line 118
                float _S526 = _S523 * _S525;

#line 118
                float3  _S527 = make_float3 (_S526);
                float _S528 = -0.4570457935333252f * _S494;

#line 119
                float _S529 = _S528 * _S520;

#line 119
                float3  _S530 = make_float3 (_S529);
                float _S531 = 1.44530570507049561f * _S491;

#line 120
                float _S532 = _S531 * _S507;

#line 120
                float3  _S533 = make_float3 (_S532);
                float _S534 = -0.59004360437393188f * _S494;

#line 121
                float _S535 = xx_2 - _S524;

#line 121
                float _S536 = _S534 * _S535;

#line 121
                float3  _S537 = make_float3 (_S536);

#line 121
                rgb_10 = rgb_12 + make_float3 (_S514) * _S437.primal_1.coeff9_0 + make_float3 (_S517) * _S437.primal_1.coeff10_0 + make_float3 (_S521) * _S437.primal_1.coeff11_0 + make_float3 (_S526) * _S437.primal_1.coeff12_0 + make_float3 (_S529) * _S437.primal_1.coeff13_0 + make_float3 (_S532) * _S437.primal_1.coeff14_0 + make_float3 (_S536) * _S437.primal_1.coeff15_0;

#line 121
                _S441 = _S537;

#line 121
                _S442 = _S437.primal_1.coeff15_0;

#line 121
                _S471 = _S534;

#line 121
                _S472 = _S535;

#line 121
                _S443 = _S533;

#line 121
                _S444 = _S437.primal_1.coeff14_0;

#line 121
                _S473 = _S531;

#line 121
                _S445 = _S530;

#line 121
                _S446 = _S437.primal_1.coeff13_0;

#line 121
                _S474 = _S528;

#line 121
                _S475 = _S520;

#line 121
                _S447 = _S527;

#line 121
                _S448 = _S437.primal_1.coeff12_0;

#line 121
                _S476 = _S523;

#line 121
                _S477 = _S525;

#line 121
                _S449 = _S522;

#line 121
                _S450 = _S437.primal_1.coeff11_0;

#line 121
                _S478 = _S519;

#line 121
                _S451 = _S518;

#line 121
                _S452 = _S437.primal_1.coeff10_0;

#line 121
                _S479 = _S516;

#line 121
                _S453 = _S515;

#line 121
                _S454 = _S437.primal_1.coeff9_0;

#line 121
                _S480 = _S511;

#line 121
                _S481 = _S513;

#line 121
            }
            else
            {

#line 121
                rgb_10 = rgb_12;

#line 121
                _S441 = _S438;

#line 121
                _S442 = _S438;

#line 121
                _S471 = 0.0f;

#line 121
                _S472 = 0.0f;

#line 121
                _S443 = _S438;

#line 121
                _S444 = _S438;

#line 121
                _S473 = 0.0f;

#line 121
                _S445 = _S438;

#line 121
                _S446 = _S438;

#line 121
                _S474 = 0.0f;

#line 121
                _S475 = 0.0f;

#line 121
                _S447 = _S438;

#line 121
                _S448 = _S438;

#line 121
                _S476 = 0.0f;

#line 121
                _S477 = 0.0f;

#line 121
                _S449 = _S438;

#line 121
                _S450 = _S438;

#line 121
                _S478 = 0.0f;

#line 121
                _S451 = _S438;

#line 121
                _S452 = _S438;

#line 121
                _S479 = 0.0f;

#line 121
                _S453 = _S438;

#line 121
                _S454 = _S438;

#line 121
                _S480 = 0.0f;

#line 121
                _S481 = 0.0f;

#line 121
            }

#line 119
            float _S538 = _S474;

#line 117
            float _S539 = _S475;
            float _S540 = _S476;

#line 118
            float _S541 = _S477;

#line 117
            float _S542 = _S478;

#line 116
            float _S543 = _S479;

#line 115
            float _S544 = _S480;

#line 115
            float _S545 = _S481;

#line 115
            _S486 = _S510;

#line 115
            _S474 = _S507;

#line 115
            _S475 = _S538;

#line 115
            _S476 = _S539;

#line 115
            _S477 = _S540;

#line 115
            _S478 = _S541;

#line 115
            _S479 = _S542;

#line 115
            _S480 = _S543;

#line 115
            _S481 = _S544;

#line 115
            _S482 = _S545;

#line 115
            _S455 = _S509;

#line 115
            _S456 = _S437.primal_1.coeff8_0;

#line 115
            _S457 = _S506;

#line 115
            _S458 = _S437.primal_1.coeff7_0;

#line 115
            _S459 = _S504;

#line 115
            _S460 = _S437.primal_1.coeff6_0;

#line 115
            _S461 = _S501;

#line 115
            _S462 = _S437.primal_1.coeff5_0;

#line 115
            _S463 = _S499;

#line 115
            _S464 = _S437.primal_1.coeff4_0;

#line 115
        }
        else
        {

#line 115
            rgb_10 = rgb_11;

#line 115
            _S486 = false;

#line 115
            _S441 = _S438;

#line 115
            _S442 = _S438;

#line 115
            _S471 = 0.0f;

#line 115
            _S472 = 0.0f;

#line 115
            _S443 = _S438;

#line 115
            _S444 = _S438;

#line 115
            _S473 = 0.0f;

#line 115
            _S474 = 0.0f;

#line 115
            _S445 = _S438;

#line 115
            _S446 = _S438;

#line 115
            _S475 = 0.0f;

#line 115
            _S476 = 0.0f;

#line 115
            _S447 = _S438;

#line 115
            _S448 = _S438;

#line 115
            _S477 = 0.0f;

#line 115
            _S478 = 0.0f;

#line 115
            _S449 = _S438;

#line 115
            _S450 = _S438;

#line 115
            _S479 = 0.0f;

#line 115
            _S451 = _S438;

#line 115
            _S452 = _S438;

#line 115
            _S480 = 0.0f;

#line 115
            _S453 = _S438;

#line 115
            _S454 = _S438;

#line 115
            _S481 = 0.0f;

#line 115
            _S482 = 0.0f;

#line 115
            _S455 = _S438;

#line 115
            _S456 = _S438;

#line 115
            _S457 = _S438;

#line 115
            _S458 = _S438;

#line 115
            _S459 = _S438;

#line 115
            _S460 = _S438;

#line 115
            _S461 = _S438;

#line 115
            _S462 = _S438;

#line 115
            _S463 = _S438;

#line 115
            _S464 = _S438;

#line 115
        }

#line 112
        bool _S546 = _S486;


        float _S547 = _S481;

#line 115
        float _S548 = _S482;

#line 115
        _S486 = _S497;

#line 115
        _S487 = _S546;

#line 115
        _S481 = _S491;

#line 115
        _S482 = _S547;

#line 115
        _S483 = _S548;

#line 115
        _S484 = _S494;

#line 115
        _S485 = _S488;

#line 115
        _S465 = _S496;

#line 115
        _S466 = _S437.primal_1.coeff3_0;

#line 115
        _S467 = _S493;

#line 115
        _S468 = _S437.primal_1.coeff2_0;

#line 115
        _S469 = _S490;

#line 115
        _S470 = _S437.primal_1.coeff1_0;

#line 115
    }
    else
    {

#line 115
        rgb_10 = rgb_9;

#line 115
        _S486 = false;

#line 115
        _S487 = false;

#line 115
        _S441 = _S438;

#line 115
        _S442 = _S438;

#line 115
        _S471 = 0.0f;

#line 115
        _S472 = 0.0f;

#line 115
        _S443 = _S438;

#line 115
        _S444 = _S438;

#line 115
        _S473 = 0.0f;

#line 115
        _S474 = 0.0f;

#line 115
        _S445 = _S438;

#line 115
        _S446 = _S438;

#line 115
        _S475 = 0.0f;

#line 115
        _S476 = 0.0f;

#line 115
        _S447 = _S438;

#line 115
        _S448 = _S438;

#line 115
        _S477 = 0.0f;

#line 115
        _S478 = 0.0f;

#line 115
        _S449 = _S438;

#line 115
        _S450 = _S438;

#line 115
        _S479 = 0.0f;

#line 115
        _S451 = _S438;

#line 115
        _S452 = _S438;

#line 115
        _S480 = 0.0f;

#line 115
        _S481 = 0.0f;

#line 115
        _S453 = _S438;

#line 115
        _S454 = _S438;

#line 115
        _S482 = 0.0f;

#line 115
        _S483 = 0.0f;

#line 115
        _S455 = _S438;

#line 115
        _S456 = _S438;

#line 115
        _S457 = _S438;

#line 115
        _S458 = _S438;

#line 115
        _S459 = _S438;

#line 115
        _S460 = _S438;

#line 115
        _S461 = _S438;

#line 115
        _S462 = _S438;

#line 115
        _S463 = _S438;

#line 115
        _S464 = _S438;

#line 115
        _S484 = 0.0f;

#line 115
        _S485 = 0.0f;

#line 115
        _S465 = _S438;

#line 115
        _S466 = _S438;

#line 115
        _S467 = _S438;

#line 115
        _S468 = _S438;

#line 115
        _S469 = _S438;

#line 115
        _S470 = _S438;

#line 115
    }

#line 126
    float3  rgb_13 = rgb_10 + make_float3 (0.5f);

    float3  _S549 = make_float3 (0.0f);

#line 128
    SpherHarmCoeffs_0 _S550 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S551;

#line 128
    (&_S551)->primal_1 = rgb_13;

#line 128
    (&_S551)->differential_0 = _S438;

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S552;

#line 128
    (&_S552)->primal_1 = _S549;

#line 128
    (&_S552)->differential_0 = _S438;

#line 128
    s_bwd_prop_max_1(&_S551, &_S552, _s_dOut_7);

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S553 = _S551;

#line 128
    SpherHarmCoeffs_0 _S554;

#line 128
    if(_S440)
    {

#line 128
        if(_S486)
        {

#line 128
            if(_S487)
            {

#line 121
                float3  _S555 = _S441 * _S553.differential_0;

#line 121
                float3  _S556 = _S442 * _S553.differential_0;

#line 121
                float _S557 = _S556.x + _S556.y + _S556.z;

#line 121
                float _S558 = _S471 * _S557;

#line 120
                float3  _S559 = _S443 * _S553.differential_0;

#line 120
                float3  _S560 = _S444 * _S553.differential_0;

#line 120
                float _S561 = _S560.x + _S560.y + _S560.z;

#line 120
                float _S562 = _S473 * _S561;

#line 120
                float _S563 = 1.44530570507049561f * (_S474 * _S561);

#line 119
                float3  _S564 = _S445 * _S553.differential_0;

#line 119
                float3  _S565 = _S446 * _S553.differential_0;

#line 119
                float _S566 = _S565.x + _S565.y + _S565.z;

#line 118
                float3  _S567 = _S447 * _S553.differential_0;

#line 118
                float3  _S568 = _S448 * _S553.differential_0;

#line 118
                float _S569 = _S568.x + _S568.y + _S568.z;

#line 118
                float _S570 = _S477 * _S569;

#line 118
                float _S571 = - _S570;

#line 118
                float _S572 = 3.0f * (- _S558 + _S571);

#line 118
                float _S573 = 0.37317633628845215f * (_S478 * _S569);

#line 117
                float3  _S574 = _S449 * _S553.differential_0;

#line 117
                float3  _S575 = _S450 * _S553.differential_0;

#line 117
                float _S576 = _S575.x + _S575.y + _S575.z;

#line 117
                float _S577 = _S475 * _S566 + _S479 * _S576;

#line 117
                float _S578 = - _S577;

#line 117
                float _S579 = 4.0f * _S577;

#line 117
                float _S580 = -0.4570457935333252f * (_S476 * _S576);

#line 116
                float3  _S581 = _S451 * _S553.differential_0;

#line 116
                float3  _S582 = _S452 * _S553.differential_0;

#line 116
                float _S583 = _S582.x + _S582.y + _S582.z;

#line 116
                float _S584 = _S480 * _S583;

#line 116
                float _S585 = 2.89061141014099121f * (_S481 * _S583);

#line 115
                float3  _S586 = _S453 * _S553.differential_0;

#line 115
                float3  _S587 = _S454 * _S553.differential_0;

#line 115
                float _S588 = _S587.x + _S587.y + _S587.z;

#line 115
                float _S589 = _S482 * _S588;

#line 115
                float _S590 = - _S589;

#line 115
                float _S591 = 3.0f * (_S571 + _S589);

#line 115
                float _S592 = -0.59004360437393188f * (_S483 * _S588);

#line 100
                float _S593 = -0.59004360437393188f * (_S472 * _S557) + -0.4570457935333252f * (_S476 * _S566);

#line 100
                SpherHarmCoeffs_0 _S594 = _S550;

#line 100
                (&_S594)->coeff15_0 = _S555;

#line 100
                (&_S594)->coeff14_0 = _S559;

#line 100
                (&_S594)->coeff13_0 = _S564;

#line 100
                (&_S594)->coeff12_0 = _S567;

#line 100
                (&_S594)->coeff11_0 = _S574;

#line 100
                (&_S594)->coeff10_0 = _S581;

#line 100
                (&_S594)->coeff9_0 = _S586;

#line 100
                SpherHarmCoeffs_0 _S595 = SpherHarmCoeffs_x24_syn_dadd_0(_S550, _S594);


                float _S596 = _S572 + _S578 + _S590;

#line 103
                float _S597 = _S558 + _S578 + _S591;

#line 100
                float _S598 = _S563 + _S573 + _S584;

#line 100
                float _S599 = _S580 + _S592;

#line 100
                _S471 = _S562;

#line 100
                _S472 = _S570;

#line 100
                _S473 = _S585;

#line 100
                _S474 = _S579;

#line 100
                _S475 = _S596;

#line 100
                _S476 = _S597;

#line 100
                _S554 = _S595;

#line 100
                _S477 = _S593;

#line 100
                _S478 = _S599;

#line 100
                _S479 = _S598;

#line 100
            }
            else
            {

#line 100
                _S471 = 0.0f;

#line 100
                _S472 = 0.0f;

#line 100
                _S473 = 0.0f;

#line 100
                _S474 = 0.0f;

#line 100
                _S475 = 0.0f;

#line 100
                _S476 = 0.0f;

#line 100
                _S554 = _S550;

#line 100
                _S477 = 0.0f;

#line 100
                _S478 = 0.0f;

#line 100
                _S479 = 0.0f;

#line 100
            }

#line 110
            float3  _S600 = _S455 * _S553.differential_0;

#line 110
            float3  _S601 = _S456 * _S553.differential_0;

#line 110
            float _S602 = 0.54627424478530884f * (_S601.x + _S601.y + _S601.z) + _S471;

#line 109
            float3  _S603 = _S457 * _S553.differential_0;

#line 109
            float3  _S604 = _S458 * _S553.differential_0;

#line 109
            float s_diff_xz_T_0 = -1.09254848957061768f * (_S604.x + _S604.y + _S604.z);

#line 108
            float3  _S605 = _S459 * _S553.differential_0;

#line 108
            float3  _S606 = _S460 * _S553.differential_0;

#line 108
            float _S607 = 0.31539157032966614f * (_S606.x + _S606.y + _S606.z);

#line 108
            float _S608 = - _S607;

#line 107
            float3  _S609 = _S461 * _S553.differential_0;

#line 107
            float3  _S610 = _S462 * _S553.differential_0;

#line 107
            float s_diff_yz_T_0 = -1.09254848957061768f * (_S610.x + _S610.y + _S610.z);

#line 106
            float3  _S611 = _S463 * _S553.differential_0;

#line 106
            float3  _S612 = _S464 * _S553.differential_0;

#line 104
            float _S613 = _S484 * s_diff_xz_T_0;

#line 104
            float _S614 = _S481 * s_diff_xz_T_0;

#line 104
            float _S615 = _S485 * s_diff_yz_T_0;

#line 104
            float _S616 = _S481 * s_diff_yz_T_0;

#line 104
            float _S617 = 1.09254848957061768f * (_S612.x + _S612.y + _S612.z) + _S473;

#line 104
            float _S618 = _S484 * _S617;

#line 104
            float _S619 = _S485 * _S617;

#line 103
            float _S620 = 2.0f * (_S607 + _S472) + _S474;

#line 103
            float _S621 = _S481 * _S620;

#line 103
            float _S622 = _S481 * _S620;

#line 103
            float _S623 = - _S602 + _S608 + _S475;

#line 103
            float _S624 = _S485 * _S623;

#line 103
            float _S625 = _S485 * _S623;

#line 103
            float _S626 = _S602 + _S608 + _S476;

#line 103
            float _S627 = _S484 * _S626;

#line 103
            float _S628 = _S484 * _S626;

#line 103
            SpherHarmCoeffs_0 _S629 = _S550;

#line 103
            (&_S629)->coeff8_0 = _S600;

#line 103
            (&_S629)->coeff7_0 = _S603;

#line 103
            (&_S629)->coeff6_0 = _S605;

#line 103
            (&_S629)->coeff5_0 = _S609;

#line 103
            (&_S629)->coeff4_0 = _S611;

#line 103
            SpherHarmCoeffs_0 _S630 = SpherHarmCoeffs_x24_syn_dadd_0(_S554, _S629);

#line 100
            float _S631 = _S616 + _S618 + _S624 + _S625 + _S478;

#line 100
            float _S632 = _S613 + _S615 + _S621 + _S622 + _S479;

#line 100
            _S471 = _S614 + _S619 + _S627 + _S628 + _S477;

#line 100
            _S472 = _S632;

#line 100
            _S473 = _S631;

#line 100
            _S554 = _S630;

#line 100
        }
        else
        {

#line 100
            _S471 = 0.0f;

#line 100
            _S472 = 0.0f;

#line 100
            _S473 = 0.0f;

#line 100
            _S554 = _S550;

#line 100
        }

#line 100
        float3  _S633 = - _S553.differential_0;

#line 100
        float3  _S634 = _S465 * _S633;

#line 100
        float3  _S635 = _S466 * _S633;

#line 100
        float3  _S636 = _S467 * _S553.differential_0;

#line 100
        float3  _S637 = _S468 * _S553.differential_0;

#line 100
        float3  _S638 = _S469 * _S633;

#line 100
        float3  _S639 = _S470 * _S633;

#line 96
        float3  _S640 = make_float3 (0.48860251903533936f * (_S635.x + _S635.y + _S635.z) + _S471, 0.48860251903533936f * (_S639.x + _S639.y + _S639.z) + _S473, 0.48860251903533936f * (_S637.x + _S637.y + _S637.z) + _S472);

#line 96
        SpherHarmCoeffs_0 _S641 = _S550;

#line 96
        (&_S641)->coeff3_0 = _S634;

#line 96
        (&_S641)->coeff2_0 = _S636;

#line 96
        (&_S641)->coeff1_0 = _S638;

#line 96
        SpherHarmCoeffs_0 _S642 = SpherHarmCoeffs_x24_syn_dadd_0(_S554, _S641);

#line 96
        rgb_10 = _S640;

#line 96
        _S554 = _S642;

#line 96
    }
    else
    {

#line 96
        rgb_10 = _S438;

#line 96
        _S554 = _S550;

#line 96
    }

    float3  _S643 = make_float3 (0.282094806432724f) * _S553.differential_0;

#line 96
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S644;

#line 96
    (&_S644)->primal_1 = dir_1;

#line 96
    (&_S644)->differential_0 = _S438;

#line 96
    s_bwd_normalize_impl_0(&_S644, rgb_10);

#line 95
    float3  _S645 = - _S644.differential_0;

#line 95
    dpcam_pos_1->primal_1 = (*dpcam_pos_1).primal_1;

#line 95
    dpcam_pos_1->differential_0 = _S645;

#line 95
    dpg_xyz_ws_1->primal_1 = (*dpg_xyz_ws_1).primal_1;

#line 95
    dpg_xyz_ws_1->differential_0 = _S644.differential_0;

#line 95
    SpherHarmCoeffs_0 _S646 = _S550;

#line 95
    (&_S646)->coeff0_0 = _S643;

#line 95
    SpherHarmCoeffs_0 _S647 = SpherHarmCoeffs_x24_syn_dadd_0(_S554, _S646);

#line 95
    dpsh_1->primal_1 = (*dpsh_1).primal_1;

#line 95
    dpsh_1->differential_0 = _S647;

#line 94
    return;
}


#line 119 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ void s_bwd_prop_geom_transform_points2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_4, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * dptransf_matrix_3, float3  _s_dOut_8)
{
    float4  _S648 = make_float4 ((*dppoint_4).primal_1.x, (*dppoint_4).primal_1.y, (*dppoint_4).primal_1.z, 1.0f);

#line 121
    float4  _S649 = make_float4 (_s_dOut_8.x, _s_dOut_8.y, _s_dOut_8.z, 0.0f);

#line 121
    Matrix<float, 4, 4>  _S650 = makeMatrix<float, 4, 4> (0.0f);

#line 121
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S651;

#line 121
    (&_S651)->primal_1 = (*dptransf_matrix_3).primal_1;

#line 121
    (&_S651)->differential_0 = _S650;

#line 121
    float4  _S652 = make_float4 (0.0f);

#line 121
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S653;

#line 121
    (&_S653)->primal_1 = _S648;

#line 121
    (&_S653)->differential_0 = _S652;

#line 121
    s_bwd_prop_mul_1(&_S651, &_S653, _S649);

#line 121
    float3  _S654 = float3 {_S653.differential_0.x, _S653.differential_0.y, _S653.differential_0.z};

#line 121
    dptransf_matrix_3->primal_1 = (*dptransf_matrix_3).primal_1;

#line 121
    dptransf_matrix_3->differential_0 = _S651.differential_0;

#line 121
    dppoint_4->primal_1 = (*dppoint_4).primal_1;

#line 121
    dppoint_4->differential_0 = _S654;

#line 119
    return;
}


#line 119
__device__ void s_bwd_prop_mul_2(DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S655, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 * _S656, Matrix<float, 4, 4>  _S657)
{

#line 119
    mul_0(_S655, _S656, _S657);

#line 119
    return;
}


#line 126
__device__ void s_bwd_prop_project_point_0(DiffPair_vectorx3Cfloatx2C3x3E_0 * dppoint_5, DiffPair_Camera_0 * dpcam_6, float3  _s_dOut_9)
{

#line 126
    Matrix<float, 4, 4>  _S658 = s_primal_ctx_mul_0((*dpcam_6).primal_1.proj_mat_1, (*dpcam_6).primal_1.world_view_transform_1);

#line 126
    float3  _S659 = _s_dOut_9;

#line 126
    *&((&_S659)->z) = 0.0f;

    float3  _S660 = make_float3 (0.0f, 0.0f, _s_dOut_9.z);

#line 128
    float3  _S661 = make_float3 (0.0f);

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S662;

#line 128
    (&_S662)->primal_1 = (*dppoint_5).primal_1;

#line 128
    (&_S662)->differential_0 = _S661;

#line 128
    Matrix<float, 4, 4>  _S663 = makeMatrix<float, 4, 4> (0.0f);

#line 128
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S664;

#line 128
    (&_S664)->primal_1 = (*dpcam_6).primal_1.world_view_transform_1;

#line 128
    (&_S664)->differential_0 = _S663;

#line 128
    s_bwd_prop_geom_transform_points2_0(&_S662, &_S664, _S660);

#line 127
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S665;

#line 127
    (&_S665)->primal_1 = (*dppoint_5).primal_1;

#line 127
    (&_S665)->differential_0 = _S661;

#line 127
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S666;

#line 127
    (&_S666)->primal_1 = _S658;

#line 127
    (&_S666)->differential_0 = _S663;

#line 127
    s_bwd_prop_geom_transform_points_0(&_S665, &_S666, _S659);

#line 127
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S667;

#line 127
    (&_S667)->primal_1 = (*dpcam_6).primal_1.proj_mat_1;

#line 127
    (&_S667)->differential_0 = _S663;

#line 127
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S668;

#line 127
    (&_S668)->primal_1 = (*dpcam_6).primal_1.world_view_transform_1;

#line 127
    (&_S668)->differential_0 = _S663;

#line 127
    s_bwd_prop_mul_2(&_S667, &_S668, _S666.differential_0);

#line 127
    Matrix<float, 4, 4>  _S669 = _S664.differential_0 + _S668.differential_0;

#line 127
    Camera_Differential_0 _S670 = Camera_x24_syn_dzero_0();

#line 127
    (&_S670)->world_view_transform_0 = _S669;

#line 127
    (&_S670)->proj_mat_0 = _S667.differential_0;

#line 127
    dpcam_6->primal_1 = (*dpcam_6).primal_1;

#line 127
    dpcam_6->differential_0 = _S670;

#line 127
    float3  _S671 = _S662.differential_0 + _S665.differential_0;

#line 127
    dppoint_5->primal_1 = (*dppoint_5).primal_1;

#line 127
    dppoint_5->differential_0 = _S671;

#line 126
    return;
}


#line 229
__device__ void s_bwd_prop_project_gaussian_to_camera_0(DiffPair_Gaussian_3D_0 * dpg_1, DiffPair_Camera_0 * dpcam_7, uint active_sh_10, Splat_2D_Vertex_0 _s_dOut_10)
{

#line 229
    DiffPair_Gaussian_3D_0 _S672 = *dpg_1;

#line 229
    DiffPair_Camera_0 _S673 = *dpcam_7;

#line 229
    float3  _S674 = make_float3 (0.0f);

#line 229
    float4  _S675 = make_float4 (0.0f);

#line 235
    Matrix<float, 3, 3>  _S676 = makeMatrix<float, 3, 3> (0.0f);

#line 235
    bool _S677 = !(s_primal_ctx_project_point_0((*dpg_1).primal_1.xyz_ws_0, (*dpcam_7).primal_1).z <= 0.20000000298023224f);

#line 235
    Matrix<float, 3, 3>  _S678;

#line 235
    float4  _S679;

#line 235
    float3  _S680;

#line 235
    float3  _S681;

#line 235
    SpherHarmCoeffs_0 _S682;

#line 235
    if(_S677)
    {

#line 235
        _S678 = s_primal_ctx_get_covariance_from_quat_scales_0(_S672.primal_1.rotations_0, _S672.primal_1.scales_0);

#line 235
        _S679 = _S672.primal_1.rotations_0;

#line 235
        _S680 = _S672.primal_1.scales_0;

#line 235
        _S682 = _S672.primal_1.sh_coeffs_0;

#line 235
        _S681 = _S673.primal_1.position_1;

#line 235
    }
    else
    {

#line 235
        _S678 = _S676;

#line 235
        _S679 = _S675;

#line 235
        _S680 = _S674;

#line 235
        (&_S682)->coeff0_0 = _S674;

#line 235
        (&_S682)->coeff1_0 = _S674;

#line 235
        (&_S682)->coeff2_0 = _S674;

#line 235
        (&_S682)->coeff3_0 = _S674;

#line 235
        (&_S682)->coeff4_0 = _S674;

#line 235
        (&_S682)->coeff5_0 = _S674;

#line 235
        (&_S682)->coeff6_0 = _S674;

#line 235
        (&_S682)->coeff7_0 = _S674;

#line 235
        (&_S682)->coeff8_0 = _S674;

#line 235
        (&_S682)->coeff9_0 = _S674;

#line 235
        (&_S682)->coeff10_0 = _S674;

#line 235
        (&_S682)->coeff11_0 = _S674;

#line 235
        (&_S682)->coeff12_0 = _S674;

#line 235
        (&_S682)->coeff13_0 = _S674;

#line 235
        (&_S682)->coeff14_0 = _S674;

#line 235
        (&_S682)->coeff15_0 = _S674;

#line 235
        _S681 = _S674;

#line 235
    }

#line 235
    Camera_Differential_0 _S683 = Camera_x24_syn_dzero_0();

#line 235
    Gaussian_3D_0 _S684 = Gaussian_3D_x24_syn_dzero_0();

#line 235
    Splat_2D_Vertex_0 _S685 = Splat_2D_Vertex_x24_syn_dadd_0(_s_dOut_10, Splat_2D_Vertex_x24_syn_dzero_0());

#line 235
    Camera_Differential_0 _S686;

#line 235
    Gaussian_3D_0 _S687;

#line 235
    if(_S677)
    {

#line 236
        DiffPair_Camera_0 _S688;

#line 236
        (&_S688)->primal_1 = _S673.primal_1;

#line 236
        (&_S688)->differential_0 = _S683;

#line 236
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S689;

#line 236
        (&_S689)->primal_1 = _S672.primal_1.xyz_ws_0;

#line 236
        (&_S689)->differential_0 = _S674;

#line 236
        DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S690;

#line 236
        (&_S690)->primal_1 = _S678;

#line 236
        (&_S690)->differential_0 = _S676;

#line 236
        s_bwd_prop_covariance_3d_to_2d_0(&_S688, &_S689, &_S690, _S685.cov_vs_0);

#line 235
        DiffPair_vectorx3Cfloatx2C4x3E_0 _S691;

#line 235
        (&_S691)->primal_1 = _S679;

#line 235
        (&_S691)->differential_0 = _S675;

#line 235
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S692;

#line 235
        (&_S692)->primal_1 = _S680;

#line 235
        (&_S692)->differential_0 = _S674;

#line 235
        s_bwd_prop_get_covariance_from_quat_scales_0(&_S691, &_S692, _S690.differential_0);

#line 234
        SpherHarmCoeffs_0 _S693 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 234
        DiffPair_SpherHarmCoeffs_0 _S694;

#line 234
        (&_S694)->primal_1 = _S682;

#line 234
        (&_S694)->differential_0 = _S693;

#line 234
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S695;

#line 234
        (&_S695)->primal_1 = _S672.primal_1.xyz_ws_0;

#line 234
        (&_S695)->differential_0 = _S674;

#line 234
        DiffPair_vectorx3Cfloatx2C3x3E_0 _S696;

#line 234
        (&_S696)->primal_1 = _S681;

#line 234
        (&_S696)->differential_0 = _S674;

#line 234
        s_bwd_prop_compute_color_from_sh_coeffs_0(&_S694, &_S695, &_S696, active_sh_10, _S685.rgb_0);

#line 234
        Gaussian_3D_0 _S697 = _S684;

#line 234
        (&_S697)->scales_0 = _S692.differential_0;

#line 234
        (&_S697)->rotations_0 = _S691.differential_0;

#line 234
        (&_S697)->sh_coeffs_0 = _S694.differential_0;

#line 234
        Gaussian_3D_0 _S698 = Gaussian_3D_x24_syn_dadd_0(_S684, _S697);

#line 234
        float3  _S699 = _S689.differential_0 + _S695.differential_0;

#line 234
        Camera_Differential_0 _S700 = Camera_x24_syn_dadd_0(_S688.differential_0, _S683);

#line 234
        Camera_Differential_0 _S701 = _S683;

#line 234
        (&_S701)->position_0 = _S696.differential_0;

#line 234
        Camera_Differential_0 _S702 = Camera_x24_syn_dadd_0(_S700, _S701);

#line 234
        _S680 = _S685.xyz_vs_0;

#line 234
        _S681 = _S699;

#line 234
        _S686 = _S702;

#line 234
        _S687 = _S698;

#line 234
    }
    else
    {

#line 234
        _S680 = _S674;

#line 234
        _S681 = _S674;

#line 234
        _S686 = _S683;

#line 234
        _S687 = _S684;

#line 234
    }

#line 230
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S703;

#line 230
    (&_S703)->primal_1 = _S672.primal_1.xyz_ws_0;

#line 230
    (&_S703)->differential_0 = _S674;

#line 230
    DiffPair_Camera_0 _S704;

#line 230
    (&_S704)->primal_1 = _S673.primal_1;

#line 230
    (&_S704)->differential_0 = _S683;

#line 230
    s_bwd_prop_project_point_0(&_S703, &_S704, _S680);

#line 230
    float3  _S705 = _S703.differential_0 + _S681;

#line 230
    Camera_Differential_0 _S706 = Camera_x24_syn_dadd_0(_S704.differential_0, _S686);

#line 230
    dpcam_7->primal_1 = (*dpcam_7).primal_1;

#line 230
    dpcam_7->differential_0 = _S706;

#line 230
    Gaussian_3D_0 _S707 = _S684;

#line 230
    (&_S707)->xyz_ws_0 = _S705;

#line 230
    Gaussian_3D_0 _S708 = Gaussian_3D_x24_syn_dadd_0(_S687, _S707);

#line 230
    dpg_1->primal_1 = (*dpg_1).primal_1;

#line 230
    dpg_1->differential_0 = _S708;

#line 229
    return;
}


#line 33
__device__ void s_bwd_prop_read_t3_float3_0(uint idx_4, DiffTensorView_0 t3_2, float3  _s_dOut_11)
{
    uint2  _S709 = make_uint2 (idx_4, 0U);
    uint2  _S710 = make_uint2 (idx_4, 1U);

#line 35
    AtomicAdd_load_backward_0(t3_2.diff_1, make_uint2 (idx_4, 2U), _s_dOut_11.z);

#line 35
    AtomicAdd_load_backward_0(t3_2.diff_1, _S710, _s_dOut_11.y);

#line 35
    AtomicAdd_load_backward_0(t3_2.diff_1, _S709, _s_dOut_11.x);

#line 33
    return;
}


#line 41
__device__ void s_bwd_prop_read_t4_float4_0(uint idx_5, DiffTensorView_0 t4_2, float4  _s_dOut_12)
{
    uint2  _S711 = make_uint2 (idx_5, 0U);
    uint2  _S712 = make_uint2 (idx_5, 1U);
    uint2  _S713 = make_uint2 (idx_5, 2U);

#line 43
    AtomicAdd_load_backward_0(t4_2.diff_1, make_uint2 (idx_5, 3U), _s_dOut_12.w);

#line 43
    AtomicAdd_load_backward_0(t4_2.diff_1, _S713, _s_dOut_12.z);

#line 43
    AtomicAdd_load_backward_0(t4_2.diff_1, _S712, _s_dOut_12.y);

#line 43
    AtomicAdd_load_backward_0(t4_2.diff_1, _S711, _s_dOut_12.x);

#line 41
    return;
}


#line 62 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
__device__ void s_bwd_prop_read_spherical_harmonics_coeffs_0(uint g_idx_5, DiffTensorView_0 sh_coeffs_6, uint active_sh_11, SpherHarmCoeffs_0 _s_dOut_13)
{

#line 68
    uint3  _S714 = make_uint3 (0U);

#line 65
    uint3  _S715 = make_uint3 (g_idx_5, 0U, 0U);

#line 65
    uint3  _S716 = make_uint3 (g_idx_5, 0U, 1U);

#line 65
    uint3  _S717 = make_uint3 (g_idx_5, 0U, 2U);

    bool _S718 = active_sh_11 > 0U;

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
    uint3  _S746;

#line 67
    uint3  _S747;

#line 67
    uint3  _S748;

#line 67
    uint3  _S749;

#line 67
    uint3  _S750;

#line 67
    uint3  _S751;

#line 67
    uint3  _S752;

#line 67
    uint3  _S753;

#line 67
    uint3  _S754;

#line 67
    uint3  _S755;

#line 67
    uint3  _S756;

#line 67
    uint3  _S757;

#line 67
    uint3  _S758;

#line 67
    uint3  _S759;

#line 67
    uint3  _S760;

#line 67
    uint3  _S761;

#line 67
    uint3  _S762;

#line 67
    uint3  _S763;

#line 67
    bool _S764;

#line 67
    bool _S765;

#line 67
    if(_S718)
    {

#line 68
        uint3  _S766 = make_uint3 (g_idx_5, 1U, 0U);

#line 68
        uint3  _S767 = make_uint3 (g_idx_5, 1U, 1U);

#line 68
        uint3  _S768 = make_uint3 (g_idx_5, 1U, 2U);
        uint3  _S769 = make_uint3 (g_idx_5, 2U, 0U);

#line 69
        uint3  _S770 = make_uint3 (g_idx_5, 2U, 1U);

#line 69
        uint3  _S771 = make_uint3 (g_idx_5, 2U, 2U);
        uint3  _S772 = make_uint3 (g_idx_5, 3U, 0U);

#line 70
        uint3  _S773 = make_uint3 (g_idx_5, 3U, 1U);

#line 70
        uint3  _S774 = make_uint3 (g_idx_5, 3U, 2U);

        bool _S775 = active_sh_11 > 1U;

#line 72
        if(_S775)
        {

#line 73
            uint3  _S776 = make_uint3 (g_idx_5, 4U, 0U);

#line 73
            uint3  _S777 = make_uint3 (g_idx_5, 4U, 1U);

#line 73
            uint3  _S778 = make_uint3 (g_idx_5, 4U, 2U);
            uint3  _S779 = make_uint3 (g_idx_5, 5U, 0U);

#line 74
            uint3  _S780 = make_uint3 (g_idx_5, 5U, 1U);

#line 74
            uint3  _S781 = make_uint3 (g_idx_5, 5U, 2U);
            uint3  _S782 = make_uint3 (g_idx_5, 6U, 0U);

#line 75
            uint3  _S783 = make_uint3 (g_idx_5, 6U, 1U);

#line 75
            uint3  _S784 = make_uint3 (g_idx_5, 6U, 2U);
            uint3  _S785 = make_uint3 (g_idx_5, 7U, 0U);

#line 76
            uint3  _S786 = make_uint3 (g_idx_5, 7U, 1U);

#line 76
            uint3  _S787 = make_uint3 (g_idx_5, 7U, 2U);
            uint3  _S788 = make_uint3 (g_idx_5, 8U, 0U);

#line 77
            uint3  _S789 = make_uint3 (g_idx_5, 8U, 1U);

#line 77
            uint3  _S790 = make_uint3 (g_idx_5, 8U, 2U);

            bool _S791 = active_sh_11 > 2U;

#line 79
            if(_S791)
            {

#line 80
                uint3  _S792 = make_uint3 (g_idx_5, 9U, 0U);

#line 80
                uint3  _S793 = make_uint3 (g_idx_5, 9U, 1U);

#line 80
                uint3  _S794 = make_uint3 (g_idx_5, 9U, 2U);
                uint3  _S795 = make_uint3 (g_idx_5, 10U, 0U);

#line 81
                uint3  _S796 = make_uint3 (g_idx_5, 10U, 1U);

#line 81
                uint3  _S797 = make_uint3 (g_idx_5, 10U, 2U);
                uint3  _S798 = make_uint3 (g_idx_5, 11U, 0U);

#line 82
                uint3  _S799 = make_uint3 (g_idx_5, 11U, 1U);

#line 82
                uint3  _S800 = make_uint3 (g_idx_5, 11U, 2U);
                uint3  _S801 = make_uint3 (g_idx_5, 12U, 0U);

#line 83
                uint3  _S802 = make_uint3 (g_idx_5, 12U, 1U);

#line 83
                uint3  _S803 = make_uint3 (g_idx_5, 12U, 2U);
                uint3  _S804 = make_uint3 (g_idx_5, 13U, 0U);

#line 84
                uint3  _S805 = make_uint3 (g_idx_5, 13U, 1U);

#line 84
                uint3  _S806 = make_uint3 (g_idx_5, 13U, 2U);
                uint3  _S807 = make_uint3 (g_idx_5, 14U, 0U);

#line 85
                uint3  _S808 = make_uint3 (g_idx_5, 14U, 1U);

#line 85
                uint3  _S809 = make_uint3 (g_idx_5, 14U, 2U);
                uint3  _S810 = make_uint3 (g_idx_5, 15U, 0U);

#line 86
                uint3  _S811 = make_uint3 (g_idx_5, 15U, 1U);

#line 86
                _S719 = make_uint3 (g_idx_5, 15U, 2U);

#line 86
                _S720 = _S811;

#line 86
                _S721 = _S810;

#line 86
                _S722 = _S809;

#line 86
                _S723 = _S808;

#line 86
                _S724 = _S807;

#line 86
                _S725 = _S806;

#line 86
                _S726 = _S805;

#line 86
                _S727 = _S804;

#line 86
                _S728 = _S803;

#line 86
                _S729 = _S802;

#line 86
                _S730 = _S801;

#line 86
                _S731 = _S800;

#line 86
                _S732 = _S799;

#line 86
                _S733 = _S798;

#line 86
                _S734 = _S797;

#line 86
                _S735 = _S796;

#line 86
                _S736 = _S795;

#line 86
                _S737 = _S794;

#line 86
                _S738 = _S793;

#line 86
                _S739 = _S792;

#line 86
            }
            else
            {

#line 86
                _S719 = _S714;

#line 86
                _S720 = _S714;

#line 86
                _S721 = _S714;

#line 86
                _S722 = _S714;

#line 86
                _S723 = _S714;

#line 86
                _S724 = _S714;

#line 86
                _S725 = _S714;

#line 86
                _S726 = _S714;

#line 86
                _S727 = _S714;

#line 86
                _S728 = _S714;

#line 86
                _S729 = _S714;

#line 86
                _S730 = _S714;

#line 86
                _S731 = _S714;

#line 86
                _S732 = _S714;

#line 86
                _S733 = _S714;

#line 86
                _S734 = _S714;

#line 86
                _S735 = _S714;

#line 86
                _S736 = _S714;

#line 86
                _S737 = _S714;

#line 86
                _S738 = _S714;

#line 86
                _S739 = _S714;

#line 86
            }

#line 86
            _S764 = _S791;

#line 86
            _S740 = _S790;

#line 86
            _S741 = _S789;

#line 86
            _S742 = _S788;

#line 86
            _S743 = _S787;

#line 86
            _S744 = _S786;

#line 86
            _S745 = _S785;

#line 86
            _S746 = _S784;

#line 86
            _S747 = _S783;

#line 86
            _S748 = _S782;

#line 86
            _S749 = _S781;

#line 86
            _S750 = _S780;

#line 86
            _S751 = _S779;

#line 86
            _S752 = _S778;

#line 86
            _S753 = _S777;

#line 86
            _S754 = _S776;

#line 86
        }
        else
        {

#line 86
            _S764 = false;

#line 86
            _S719 = _S714;

#line 86
            _S720 = _S714;

#line 86
            _S721 = _S714;

#line 86
            _S722 = _S714;

#line 86
            _S723 = _S714;

#line 86
            _S724 = _S714;

#line 86
            _S725 = _S714;

#line 86
            _S726 = _S714;

#line 86
            _S727 = _S714;

#line 86
            _S728 = _S714;

#line 86
            _S729 = _S714;

#line 86
            _S730 = _S714;

#line 86
            _S731 = _S714;

#line 86
            _S732 = _S714;

#line 86
            _S733 = _S714;

#line 86
            _S734 = _S714;

#line 86
            _S735 = _S714;

#line 86
            _S736 = _S714;

#line 86
            _S737 = _S714;

#line 86
            _S738 = _S714;

#line 86
            _S739 = _S714;

#line 86
            _S740 = _S714;

#line 86
            _S741 = _S714;

#line 86
            _S742 = _S714;

#line 86
            _S743 = _S714;

#line 86
            _S744 = _S714;

#line 86
            _S745 = _S714;

#line 86
            _S746 = _S714;

#line 86
            _S747 = _S714;

#line 86
            _S748 = _S714;

#line 86
            _S749 = _S714;

#line 86
            _S750 = _S714;

#line 86
            _S751 = _S714;

#line 86
            _S752 = _S714;

#line 86
            _S753 = _S714;

#line 86
            _S754 = _S714;

#line 86
        }

#line 79
        bool _S812 = _S764;

#line 79
        _S764 = _S775;

#line 79
        _S765 = _S812;

#line 79
        _S755 = _S774;

#line 79
        _S756 = _S773;

#line 79
        _S757 = _S772;

#line 79
        _S758 = _S771;

#line 79
        _S759 = _S770;

#line 79
        _S760 = _S769;

#line 79
        _S761 = _S768;

#line 79
        _S762 = _S767;

#line 79
        _S763 = _S766;

#line 79
    }
    else
    {

#line 79
        _S764 = false;

#line 79
        _S765 = false;

#line 79
        _S719 = _S714;

#line 79
        _S720 = _S714;

#line 79
        _S721 = _S714;

#line 79
        _S722 = _S714;

#line 79
        _S723 = _S714;

#line 79
        _S724 = _S714;

#line 79
        _S725 = _S714;

#line 79
        _S726 = _S714;

#line 79
        _S727 = _S714;

#line 79
        _S728 = _S714;

#line 79
        _S729 = _S714;

#line 79
        _S730 = _S714;

#line 79
        _S731 = _S714;

#line 79
        _S732 = _S714;

#line 79
        _S733 = _S714;

#line 79
        _S734 = _S714;

#line 79
        _S735 = _S714;

#line 79
        _S736 = _S714;

#line 79
        _S737 = _S714;

#line 79
        _S738 = _S714;

#line 79
        _S739 = _S714;

#line 79
        _S740 = _S714;

#line 79
        _S741 = _S714;

#line 79
        _S742 = _S714;

#line 79
        _S743 = _S714;

#line 79
        _S744 = _S714;

#line 79
        _S745 = _S714;

#line 79
        _S746 = _S714;

#line 79
        _S747 = _S714;

#line 79
        _S748 = _S714;

#line 79
        _S749 = _S714;

#line 79
        _S750 = _S714;

#line 79
        _S751 = _S714;

#line 79
        _S752 = _S714;

#line 79
        _S753 = _S714;

#line 79
        _S754 = _S714;

#line 79
        _S755 = _S714;

#line 79
        _S756 = _S714;

#line 79
        _S757 = _S714;

#line 79
        _S758 = _S714;

#line 79
        _S759 = _S714;

#line 79
        _S760 = _S714;

#line 79
        _S761 = _S714;

#line 79
        _S762 = _S714;

#line 79
        _S763 = _S714;

#line 79
    }

#line 79
    SpherHarmCoeffs_0 _S813 = SpherHarmCoeffs_x24_syn_dzero_0();

#line 77
    float3  _S814 = make_float3 (0.0f);

#line 77
    SpherHarmCoeffs_0 _S815;

#line 77
    float3  _S816;

#line 77
    if(_S718)
    {

#line 77
        float3  _S817;

#line 77
        float3  _S818;

#line 77
        float3  _S819;

#line 77
        if(_S764)
        {

#line 77
            float3  _S820;

#line 77
            float3  _S821;

#line 77
            float3  _S822;

#line 77
            float3  _S823;

#line 77
            float3  _S824;

#line 77
            if(_S765)
            {

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S719, _s_dOut_13.coeff15_0.z);

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S720, _s_dOut_13.coeff15_0.y);

#line 86
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S721, _s_dOut_13.coeff15_0.x);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S722, _s_dOut_13.coeff14_0.z);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S723, _s_dOut_13.coeff14_0.y);

#line 85
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S724, _s_dOut_13.coeff14_0.x);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S725, _s_dOut_13.coeff13_0.z);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S726, _s_dOut_13.coeff13_0.y);

#line 84
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S727, _s_dOut_13.coeff13_0.x);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S728, _s_dOut_13.coeff12_0.z);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S729, _s_dOut_13.coeff12_0.y);

#line 83
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S730, _s_dOut_13.coeff12_0.x);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S731, _s_dOut_13.coeff11_0.z);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S732, _s_dOut_13.coeff11_0.y);

#line 82
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S733, _s_dOut_13.coeff11_0.x);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S734, _s_dOut_13.coeff10_0.z);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S735, _s_dOut_13.coeff10_0.y);

#line 81
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S736, _s_dOut_13.coeff10_0.x);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S737, _s_dOut_13.coeff9_0.z);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S738, _s_dOut_13.coeff9_0.y);

#line 80
                AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S739, _s_dOut_13.coeff9_0.x);

#line 80
                _S815 = _S813;

#line 80
                _S816 = _s_dOut_13.coeff8_0;

#line 80
                _S817 = _s_dOut_13.coeff7_0;

#line 80
                _S818 = _s_dOut_13.coeff6_0;

#line 80
                _S819 = _s_dOut_13.coeff5_0;

#line 80
                _S820 = _s_dOut_13.coeff4_0;

#line 80
                _S821 = _s_dOut_13.coeff0_0;

#line 80
                _S822 = _s_dOut_13.coeff1_0;

#line 80
                _S823 = _s_dOut_13.coeff2_0;

#line 80
                _S824 = _s_dOut_13.coeff3_0;

#line 80
            }
            else
            {

#line 80
                _S815 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S813);

#line 80
                _S816 = _S814;

#line 80
                _S817 = _S814;

#line 80
                _S818 = _S814;

#line 80
                _S819 = _S814;

#line 80
                _S820 = _S814;

#line 80
                _S821 = _S814;

#line 80
                _S822 = _S814;

#line 80
                _S823 = _S814;

#line 80
                _S824 = _S814;

#line 80
            }

#line 77
            float3  _S825 = _S815.coeff8_0 + _S816;

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S740, _S825.z);

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S741, _S825.y);

#line 77
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S742, _S825.x);

#line 76
            float3  _S826 = _S815.coeff7_0 + _S817;

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S743, _S826.z);

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S744, _S826.y);

#line 76
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S745, _S826.x);

#line 75
            float3  _S827 = _S815.coeff6_0 + _S818;

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S746, _S827.z);

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S747, _S827.y);

#line 75
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S748, _S827.x);

#line 74
            float3  _S828 = _S815.coeff5_0 + _S819;

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S749, _S828.z);

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S750, _S828.y);

#line 74
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S751, _S828.x);

#line 73
            float3  _S829 = _S815.coeff4_0 + _S820;

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S752, _S829.z);

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S753, _S829.y);

#line 73
            AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S754, _S829.x);

#line 65
            float3  _S830 = _S815.coeff0_0 + _S821;


            float3  _S831 = _S815.coeff1_0 + _S822;
            float3  _S832 = _S815.coeff2_0 + _S823;
            float3  _S833 = _S815.coeff3_0 + _S824;

#line 70
            _S815 = _S813;

#line 70
            _S816 = _S833;

#line 70
            _S817 = _S832;

#line 70
            _S818 = _S831;

#line 70
            _S819 = _S830;

#line 70
        }
        else
        {

#line 70
            _S815 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S813);

#line 70
            _S816 = _S814;

#line 70
            _S817 = _S814;

#line 70
            _S818 = _S814;

#line 70
            _S819 = _S814;

#line 70
        }

#line 70
        float3  _S834 = _S815.coeff3_0 + _S816;

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S755, _S834.z);

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S756, _S834.y);

#line 70
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S757, _S834.x);

#line 69
        float3  _S835 = _S815.coeff2_0 + _S817;

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S758, _S835.z);

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S759, _S835.y);

#line 69
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S760, _S835.x);

#line 68
        float3  _S836 = _S815.coeff1_0 + _S818;

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S761, _S836.z);

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S762, _S836.y);

#line 68
        AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S763, _S836.x);

#line 65
        float3  _S837 = _S815.coeff0_0 + _S819;

#line 65
        _S815 = _S813;

#line 65
        _S816 = _S837;

#line 65
    }
    else
    {

#line 65
        _S815 = SpherHarmCoeffs_x24_syn_dadd_0(_s_dOut_13, _S813);

#line 65
        _S816 = _S814;

#line 65
    }

#line 65
    float3  _S838 = _S815.coeff0_0 + _S816;

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S717, _S838.z);

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S716, _S838.y);

#line 65
    AtomicAdd_load_backward_1(sh_coeffs_6.diff_1, _S715, _S838.x);

#line 62
    return;
}


#line 177 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ void s_bwd_prop_load_gaussian_0(int g_idx_6, DiffTensorView_0 xyz_ws_6, DiffTensorView_0 sh_coeffs_7, DiffTensorView_0 rotations_4, DiffTensorView_0 scales_4, uint active_sh_12, Gaussian_3D_0 _s_dOut_14)
{

#line 184
    uint _S839 = uint(g_idx_6);

#line 184
    s_bwd_prop_read_t3_float3_0(_S839, scales_4, _s_dOut_14.scales_0);

#line 184
    s_bwd_prop_read_t4_float4_0(_S839, rotations_4, _s_dOut_14.rotations_0);

#line 184
    s_bwd_prop_read_spherical_harmonics_coeffs_0(_S839, sh_coeffs_7, active_sh_12, _s_dOut_14.sh_coeffs_0);

#line 184
    s_bwd_prop_read_t3_float3_0(_S839, xyz_ws_6, _s_dOut_14.xyz_ws_0);

#line 177
    return;
}


#line 53 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
__device__ void s_bwd_prop_vertex_shader_0(DiffTensorView_0 xyz_ws_7, DiffTensorView_0 sh_coeffs_8, DiffTensorView_0 rotations_5, DiffTensorView_0 scales_5, TensorView opcities_1, uint active_sh_13, TensorView world_view_transform_4, TensorView proj_mat_4, TensorView cam_pos_2, TensorView out_tiles_touched_1, TensorView out_rect_tile_space_1, TensorView out_radii_1, DiffTensorView_0 out_xyz_vs_1, DiffTensorView_0 out_inv_cov_vs_1, DiffTensorView_0 out_rgb_1, float fovy_4, float fovx_4, uint image_height_1, uint image_width_1, uint grid_height_1, uint grid_width_1, uint tile_height_1, uint tile_width_1, s_bwd_prop_vertex_shader_Intermediates_0 _s_diff_ctx_1)
{

#line 53
    Matrix<float, 2, 2>  _S840 = makeMatrix<float, 2, 2> (0.0f);

#line 109
    uint2  _S841 = make_uint2 (0U);

#line 117
    uint3  _S842 = make_uint3 (0U);

#line 77
    uint g_idx_7 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

#line 77
    bool _S843 = !(g_idx_7 >= DiffTensorView_size_0(xyz_ws_7, 0U));

#line 77
    bool _bflag_1;

#line 77
    bool _bflag_2;

#line 77
    bool _bflag_3;

#line 77
    uint2  _S844;

#line 77
    uint2  _S845;

#line 77
    uint2  _S846;

#line 77
    uint3  _S847;

#line 77
    uint3  _S848;

#line 77
    uint3  _S849;

#line 77
    uint3  _S850;

#line 77
    float _S851;

#line 77
    float _S852;

#line 77
    int _S853;

#line 77
    int _S854;

#line 77
    int _S855;

#line 77
    Matrix<float, 2, 2>  _S856;

#line 77
    Matrix<float, 2, 2>  _S857;

#line 77
    Matrix<float, 2, 2>  _S858;

#line 77
    Matrix<float, 2, 2>  _S859;

#line 77
    if(_S843)
    {

#line 83
        int _S860 = int(g_idx_7);

#line 83
        Splat_2D_Vertex_0 _S861 = s_primal_ctx_project_gaussian_to_camera_0(_s_diff_ctx_1._S124, _s_diff_ctx_1._S123, active_sh_13);

        if(_S861.xyz_vs_0.z <= 0.20000000298023224f)
        {

#line 85
            _bflag_1 = false;

#line 85
        }
        else
        {

#line 85
            _bflag_1 = _S843;

#line 85
        }

#line 85
        if(_bflag_1)
        {

#line 85
            float _S862 = s_primal_ctx_compute_det_0(_S861.cov_vs_0);

#line 95
            Matrix<float, 2, 2>  _S863 = makeMatrix<float, 2, 2> (_S862);

#line 91
            if(_S862 == 0.0f)
            {

#line 91
                _bflag_2 = false;

#line 91
            }
            else
            {

#line 91
                _bflag_2 = _bflag_1;

#line 91
            }

#line 91
            if(_bflag_2)
            {


                Matrix<float, 2, 2>  _S864 = makeMatrix<float, 2, 2> (_S861.cov_vs_0.rows[int(1)].y, - _S861.cov_vs_0.rows[int(0)].y, - _S861.cov_vs_0.rows[int(1)].x, _S861.cov_vs_0.rows[int(0)].x);

#line 95
                Matrix<float, 2, 2>  g_inv_cov_vs_1 = _S864 / makeMatrix<float, 2, 2> (_S862);

#line 95
                Matrix<float, 2, 2>  _S865 = makeMatrix<float, 2, 2> (_S862 * _S862);

                float _S866 = _S861.xyz_vs_0.x;

#line 97
                int _S867 = int(image_width_1);

#line 97
                float _S868 = _S861.xyz_vs_0.y;

#line 97
                int _S869 = int(image_height_1);

#line 97
                float2  pixelspace_xy_1 = make_float2 (s_primal_ctx_ndc2pix_0(_S866, _S867), s_primal_ctx_ndc2pix_0(_S868, _S869));



                float3  _S870 = make_float3 (g_inv_cov_vs_1.rows[int(0)].x, g_inv_cov_vs_1.rows[int(0)].y, g_inv_cov_vs_1.rows[int(1)].y);

#line 101
                int2  _S871 = make_int2 (int(grid_width_1), int(grid_height_1));

#line 101
                int2  _S872 = make_int2 (int(tile_width_1), int(tile_height_1));

#line 101
                uint _S873 = 0U;

#line 101
                rectangle_0 rect_tile_space_1 = computeSnugBox_0(_S870, pixelspace_xy_1, _s_diff_ctx_1._S125, _S871, _S872, &_S873);

                if(_S873 == 0U)
                {

#line 103
                    _bflag_3 = false;

#line 103
                }
                else
                {

#line 103
                    _bflag_3 = _bflag_2;

#line 103
                }

#line 103
                if(_bflag_3)
                {

#line 109
                    uint2  _S874 = make_uint2 (g_idx_7, 0U);
                    uint2  _S875 = make_uint2 (g_idx_7, 1U);

#line 117
                    uint3  _S876 = make_uint3 (g_idx_7, 0U, 0U);
                    uint3  _S877 = make_uint3 (g_idx_7, 0U, 1U);
                    uint3  _S878 = make_uint3 (g_idx_7, 1U, 0U);
                    uint3  _S879 = make_uint3 (g_idx_7, 1U, 1U);

#line 120
                    _S844 = make_uint2 (g_idx_7, 2U);

#line 120
                    _S845 = _S875;

#line 120
                    _S846 = _S874;

#line 120
                    _S847 = _S879;

#line 120
                    _S848 = _S878;

#line 120
                    _S849 = _S877;

#line 120
                    _S850 = _S876;

#line 120
                }
                else
                {

#line 120
                    _S844 = _S841;

#line 120
                    _S845 = _S841;

#line 120
                    _S846 = _S841;

#line 120
                    _S847 = _S842;

#line 120
                    _S848 = _S842;

#line 120
                    _S849 = _S842;

#line 120
                    _S850 = _S842;

#line 120
                }

#line 120
                _S851 = _S868;

#line 120
                _S852 = _S866;

#line 120
                _S853 = _S869;

#line 120
                _S854 = _S867;

#line 120
                _S856 = _S865;

#line 120
                _S857 = _S864;

#line 120
            }
            else
            {

#line 120
                _bflag_3 = false;

#line 120
                _S844 = _S841;

#line 120
                _S845 = _S841;

#line 120
                _S846 = _S841;

#line 120
                _S847 = _S842;

#line 120
                _S848 = _S842;

#line 120
                _S849 = _S842;

#line 120
                _S850 = _S842;

#line 120
                _S851 = 0.0f;

#line 120
                _S852 = 0.0f;

#line 120
                _S853 = int(0);

#line 120
                _S854 = int(0);

#line 120
                _S856 = _S840;

#line 120
                _S857 = _S840;

#line 120
            }

#line 120
            _S858 = _S863;

#line 120
            _S859 = _S861.cov_vs_0;

#line 120
        }
        else
        {

#line 120
            _bflag_2 = false;

#line 120
            _bflag_3 = false;

#line 120
            _S844 = _S841;

#line 120
            _S845 = _S841;

#line 120
            _S846 = _S841;

#line 120
            _S847 = _S842;

#line 120
            _S848 = _S842;

#line 120
            _S849 = _S842;

#line 120
            _S850 = _S842;

#line 120
            _S851 = 0.0f;

#line 120
            _S852 = 0.0f;

#line 120
            _S853 = int(0);

#line 120
            _S854 = int(0);

#line 120
            _S856 = _S840;

#line 120
            _S857 = _S840;

#line 120
            _S858 = _S840;

#line 120
            _S859 = _S840;

#line 120
        }

#line 120
        _S855 = _S860;

#line 120
    }
    else
    {

#line 120
        _bflag_1 = false;

#line 120
        _bflag_2 = false;

#line 120
        _bflag_3 = false;

#line 120
        _S844 = _S841;

#line 120
        _S845 = _S841;

#line 120
        _S846 = _S841;

#line 120
        _S847 = _S842;

#line 120
        _S848 = _S842;

#line 120
        _S849 = _S842;

#line 120
        _S850 = _S842;

#line 120
        _S851 = 0.0f;

#line 120
        _S852 = 0.0f;

#line 120
        _S853 = int(0);

#line 120
        _S854 = int(0);

#line 120
        _S856 = _S840;

#line 120
        _S857 = _S840;

#line 120
        _S858 = _S840;

#line 120
        _S859 = _S840;

#line 120
        _S855 = int(0);

#line 120
    }

#line 1751 "core.meta.slang"
    float3  _S880 = make_float3 (0.0f);

#line 1751
    float2  _S881 = make_float2 (0.0f);

#line 84 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
    Splat_2D_Vertex_0 _S882 = Splat_2D_Vertex_x24_syn_dzero_0();

#line 84
    if(_S843)
    {

#line 84
        Splat_2D_Vertex_0 _S883;

#line 84
        float3  _S884;

#line 84
        if(_bflag_1)
        {

#line 84
            if(_bflag_2)
            {

#line 84
                float _S885;

#line 84
                float _S886;

#line 84
                float _S887;

#line 84
                float _S888;

#line 84
                float _S889;

#line 84
                float _S890;

#line 84
                float2  _S891;

#line 84
                if(_bflag_3)
                {

#line 84
                    float3  _S892 = make_float3 (AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S846), AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S845), AtomicAdd_storeOnce_backward_0(out_rgb_1.diff_1, _S844));

#line 120
                    float _S893 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S847);

#line 119
                    float _S894 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S848);

#line 118
                    float _S895 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S849);

#line 117
                    float _S896 = AtomicAdd_storeOnce_backward_1(out_inv_cov_vs_1.diff_1, _S850);

#line 116
                    float _S897 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S844);

#line 115
                    float _S898 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S845);

#line 114
                    float _S899 = AtomicAdd_storeOnce_backward_0(out_xyz_vs_1.diff_1, _S846);

#line 84
                    Splat_2D_Vertex_0 _S900 = _S882;

#line 84
                    (&_S900)->rgb_0 = _S892;

#line 84
                    Splat_2D_Vertex_0 _S901 = Splat_2D_Vertex_x24_syn_dadd_0(_S882, _S900);

#line 84
                    float2  _S902 = _S881;

#line 84
                    *&((&_S902)->x) = _S894;

#line 84
                    _S885 = _S893;

#line 84
                    _S891 = _S902;

#line 84
                    _S886 = _S895;

#line 84
                    _S887 = _S896;

#line 84
                    _S888 = _S898;

#line 84
                    _S889 = _S899;

#line 84
                    _S883 = _S901;

#line 84
                    _S890 = _S897;

#line 84
                }
                else
                {

#line 84
                    _S885 = 0.0f;

#line 84
                    _S891 = _S881;

#line 84
                    _S886 = 0.0f;

#line 84
                    _S887 = 0.0f;

#line 84
                    _S888 = 0.0f;

#line 84
                    _S889 = 0.0f;

#line 84
                    _S883 = _S882;

#line 84
                    _S890 = 0.0f;

#line 84
                }

#line 84
                float2  _S903 = _S881;

#line 84
                *&((&_S903)->y) = _S885;

#line 84
                float2  _S904 = _S891 + _S903;

#line 84
                float2  _S905 = _S881;

#line 84
                *&((&_S905)->y) = _S886;

#line 84
                *&((&_S905)->x) = _S887;

#line 97
                DiffPair_float_0 _S906;

#line 97
                (&_S906)->primal_1 = _S851;

#line 97
                (&_S906)->differential_0 = 0.0f;

#line 97
                s_bwd_prop_ndc2pix_0(&_S906, _S853, 0.0f);

#line 97
                float _S907 = _S906.differential_0 + _S888;

#line 97
                DiffPair_float_0 _S908;

#line 97
                (&_S908)->primal_1 = _S852;

#line 97
                (&_S908)->differential_0 = 0.0f;

#line 97
                s_bwd_prop_ndc2pix_0(&_S908, _S854, 0.0f);

#line 97
                float _S909 = _S908.differential_0 + _S889;

#line 95
                Matrix<float, 2, 2>  _S910 = _S840;

#line 95
                _S910[int(1)] = _S904;

#line 95
                _S910[int(0)] = _S905;

#line 95
                Matrix<float, 2, 2>  _S911 = _S910 / _S856;

#line 95
                Matrix<float, 2, 2>  _S912 = _S857 * - _S911;

#line 95
                Matrix<float, 2, 2>  _S913 = _S858 * _S911;

#line 95
                float _S914 = - _S913.rows[int(1)].x;

#line 95
                float _S915 = - _S913.rows[int(0)].y;

#line 95
                float2  _S916 = _S881;

#line 95
                *&((&_S916)->x) = _S913.rows[int(1)].y;

#line 95
                *&((&_S916)->y) = _S915;

#line 95
                float2  _S917 = _S881;

#line 95
                *&((&_S917)->x) = _S914;

#line 95
                *&((&_S917)->y) = _S913.rows[int(0)].x;

#line 95
                float3  _S918 = make_float3 (_S909, _S907, 0.0f);

#line 95
                Matrix<float, 2, 2>  _S919 = _S840;

#line 95
                _S919[int(0)] = _S916;

#line 95
                _S919[int(1)] = _S917;

#line 95
                _S856 = _S912;

#line 95
                _S857 = _S919;

#line 95
                _S851 = _S890;

#line 95
                _S884 = _S918;

#line 95
            }
            else
            {

#line 95
                _S856 = _S840;

#line 95
                _S857 = _S840;

#line 95
                _S883 = _S882;

#line 95
                _S851 = 0.0f;

#line 95
                _S884 = _S880;

#line 95
            }

#line 89
            float _S920 = _S856.rows[int(0)].x + _S856.rows[int(0)].y + _S856.rows[int(1)].x + _S856.rows[int(1)].y;

#line 89
            DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S921;

#line 89
            (&_S921)->primal_1 = _S859;

#line 89
            (&_S921)->differential_0 = _S840;

#line 89
            s_bwd_prop_compute_det_0(&_S921, _S920);

#line 89
            Matrix<float, 2, 2>  _S922 = _S921.differential_0 + _S857;

#line 84
            Splat_2D_Vertex_0 _S923 = _S882;

#line 84
            (&_S923)->cov_vs_0 = _S922;

#line 84
            _S883 = Splat_2D_Vertex_x24_syn_dadd_0(_S883, _S923);

#line 84
        }
        else
        {

#line 84
            _S851 = 0.0f;

#line 84
            _S884 = _S880;

#line 84
            _S883 = _S882;

#line 84
        }

#line 84
        float3  _S924 = _S884 + make_float3 (0.0f, 0.0f, _S851);

#line 84
        Splat_2D_Vertex_0 _S925 = _S882;

#line 84
        (&_S925)->xyz_vs_0 = _S924;

#line 84
        Splat_2D_Vertex_0 _S926 = Splat_2D_Vertex_x24_syn_dadd_0(_S883, _S925);

#line 84
        Gaussian_3D_0 _S927 = Gaussian_3D_x24_syn_dzero_0();

#line 84
        DiffPair_Gaussian_3D_0 _S928;

#line 84
        (&_S928)->primal_1 = _s_diff_ctx_1._S124;

#line 84
        (&_S928)->differential_0 = _S927;

#line 84
        Camera_Differential_0 _S929 = Camera_x24_syn_dzero_0();

#line 84
        DiffPair_Camera_0 _S930;

#line 84
        (&_S930)->primal_1 = _s_diff_ctx_1._S123;

#line 84
        (&_S930)->differential_0 = _S929;

#line 84
        s_bwd_prop_project_gaussian_to_camera_0(&_S928, &_S930, active_sh_13, _S926);

#line 84
        s_bwd_prop_load_gaussian_0(_S855, xyz_ws_7, sh_coeffs_8, rotations_5, scales_5, active_sh_13, _S928.differential_0);

#line 84
    }

#line 53
    return;
}


#line 53
__device__ void s_bwd_vertex_shader_0(DiffTensorView_0 _S931, DiffTensorView_0 _S932, DiffTensorView_0 _S933, DiffTensorView_0 _S934, TensorView _S935, uint _S936, TensorView _S937, TensorView _S938, TensorView _S939, TensorView _S940, TensorView _S941, TensorView _S942, DiffTensorView_0 _S943, DiffTensorView_0 _S944, DiffTensorView_0 _S945, float _S946, float _S947, uint _S948, uint _S949, uint _S950, uint _S951, uint _S952, uint _S953)
{

#line 75
    s_bwd_prop_vertex_shader_Intermediates_0 _S954;

#line 75
    s_primal_ctx_vertex_shader_0(_S931, _S932, _S933, _S934, _S935, _S936, _S937, _S938, _S939, _S940, _S941, _S942, _S943, _S944, _S945, _S946, _S947, _S948, _S949, _S950, _S951, _S952, _S953, &_S954);

#line 75
    s_bwd_prop_vertex_shader_0(_S931, _S932, _S933, _S934, _S935, _S936, _S937, _S938, _S939, _S940, _S941, _S942, _S943, _S944, _S945, _S946, _S947, _S948, _S949, _S950, _S951, _S952, _S953, _S954);

#line 75
    return;
}


#line 75
extern "C" {
__global__ void __kernel__vertex_shader_bwd_diff(DiffTensorView_0 xyz_ws_8, DiffTensorView_0 sh_coeffs_9, DiffTensorView_0 rotations_6, DiffTensorView_0 scales_6, TensorView opcities_2, uint active_sh_14, TensorView world_view_transform_5, TensorView proj_mat_5, TensorView cam_pos_3, TensorView out_tiles_touched_2, TensorView out_rect_tile_space_2, TensorView out_radii_2, DiffTensorView_0 out_xyz_vs_2, DiffTensorView_0 out_inv_cov_vs_2, DiffTensorView_0 out_rgb_2, float fovy_5, float fovx_5, uint image_height_2, uint image_width_2, uint grid_height_2, uint grid_width_2, uint tile_height_2, uint tile_width_2)
{

#line 75
    s_bwd_vertex_shader_0(xyz_ws_8, sh_coeffs_9, rotations_6, scales_6, opcities_2, active_sh_14, world_view_transform_5, proj_mat_5, cam_pos_3, out_tiles_touched_2, out_rect_tile_space_2, out_radii_2, out_xyz_vs_2, out_inv_cov_vs_2, out_rgb_2, fovy_5, fovx_5, image_height_2, image_width_2, grid_height_2, grid_width_2, tile_height_2, tile_width_2);

#line 75
    return;
}

}

#line 184 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_read_t3_float3_0(uint idx_6, DiffTensorView_0 t3_3)
{

#line 35
    uint2  _S955 = make_uint2 (idx_6, 0U);

#line 35
    float _S956 = ((t3_3.primal_0).load<float>((_S955)));

#line 35
    float _S957 = AtomicAdd_load_forward_0(t3_3.diff_1, _S955);
    uint2  _S958 = make_uint2 (idx_6, 1U);

#line 35
    float _S959 = ((t3_3.primal_0).load<float>((_S958)));

#line 35
    float _S960 = AtomicAdd_load_forward_0(t3_3.diff_1, _S958);

    uint2  _S961 = make_uint2 (idx_6, 2U);

#line 35
    float _S962 = ((t3_3.primal_0).load<float>((_S961)));

#line 35
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S963 = { make_float3 (_S956, _S959, _S962), make_float3 (_S957, _S960, AtomicAdd_load_forward_0(t3_3.diff_1, _S961)) };

#line 35
    return _S963;
}


#line 185
__device__ DiffPair_SpherHarmCoeffs_0 s_fwd_read_spherical_harmonics_coeffs_0(uint g_idx_8, DiffTensorView_0 sh_coeffs_10, uint active_sh_15)
{

#line 64 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
    float3  _S964 = make_float3 (0.0f);
    uint3  _S965 = make_uint3 (g_idx_8, 0U, 0U);

#line 65
    float _S966 = ((sh_coeffs_10.primal_0).load<float>((_S965)));

#line 65
    float _S967 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S965);

#line 65
    uint3  _S968 = make_uint3 (g_idx_8, 0U, 1U);

#line 65
    float _S969 = ((sh_coeffs_10.primal_0).load<float>((_S968)));

#line 65
    float _S970 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S968);

#line 65
    uint3  _S971 = make_uint3 (g_idx_8, 0U, 2U);

#line 65
    float _S972 = ((sh_coeffs_10.primal_0).load<float>((_S971)));

#line 65
    float3  _S973 = make_float3 (_S966, _S969, _S972);

#line 65
    float3  _S974 = make_float3 (_S967, _S970, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S971));

#line 65
    SpherHarmCoeffs_0 g_sh_coeffs_2;

#line 65
    SpherHarmCoeffs_0 s_diff_g_sh_coeffs_0;

    if(active_sh_15 > 0U)
    {

#line 68
        uint3  _S975 = make_uint3 (g_idx_8, 1U, 0U);

#line 68
        float _S976 = ((sh_coeffs_10.primal_0).load<float>((_S975)));

#line 68
        float _S977 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S975);

#line 68
        uint3  _S978 = make_uint3 (g_idx_8, 1U, 1U);

#line 68
        float _S979 = ((sh_coeffs_10.primal_0).load<float>((_S978)));

#line 68
        float _S980 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S978);

#line 68
        uint3  _S981 = make_uint3 (g_idx_8, 1U, 2U);

#line 68
        float _S982 = ((sh_coeffs_10.primal_0).load<float>((_S981)));

#line 68
        float3  _S983 = make_float3 (_S976, _S979, _S982);

#line 68
        float3  _S984 = make_float3 (_S977, _S980, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S981));
        uint3  _S985 = make_uint3 (g_idx_8, 2U, 0U);

#line 69
        float _S986 = ((sh_coeffs_10.primal_0).load<float>((_S985)));

#line 69
        float _S987 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S985);

#line 69
        uint3  _S988 = make_uint3 (g_idx_8, 2U, 1U);

#line 69
        float _S989 = ((sh_coeffs_10.primal_0).load<float>((_S988)));

#line 69
        float _S990 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S988);

#line 69
        uint3  _S991 = make_uint3 (g_idx_8, 2U, 2U);

#line 69
        float _S992 = ((sh_coeffs_10.primal_0).load<float>((_S991)));

#line 69
        float3  _S993 = make_float3 (_S986, _S989, _S992);

#line 69
        float3  _S994 = make_float3 (_S987, _S990, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S991));
        uint3  _S995 = make_uint3 (g_idx_8, 3U, 0U);

#line 70
        float _S996 = ((sh_coeffs_10.primal_0).load<float>((_S995)));

#line 70
        float _S997 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S995);

#line 70
        uint3  _S998 = make_uint3 (g_idx_8, 3U, 1U);

#line 70
        float _S999 = ((sh_coeffs_10.primal_0).load<float>((_S998)));

#line 70
        float _S1000 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S998);

#line 70
        uint3  _S1001 = make_uint3 (g_idx_8, 3U, 2U);

#line 70
        float _S1002 = ((sh_coeffs_10.primal_0).load<float>((_S1001)));

#line 70
        float3  _S1003 = make_float3 (_S996, _S999, _S1002);

#line 70
        float3  _S1004 = make_float3 (_S997, _S1000, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1001));

        if(active_sh_15 > 1U)
        {

#line 73
            uint3  _S1005 = make_uint3 (g_idx_8, 4U, 0U);

#line 73
            float _S1006 = ((sh_coeffs_10.primal_0).load<float>((_S1005)));

#line 73
            float _S1007 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1005);

#line 73
            uint3  _S1008 = make_uint3 (g_idx_8, 4U, 1U);

#line 73
            float _S1009 = ((sh_coeffs_10.primal_0).load<float>((_S1008)));

#line 73
            float _S1010 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1008);

#line 73
            uint3  _S1011 = make_uint3 (g_idx_8, 4U, 2U);

#line 73
            float _S1012 = ((sh_coeffs_10.primal_0).load<float>((_S1011)));

#line 73
            float3  _S1013 = make_float3 (_S1006, _S1009, _S1012);

#line 73
            float3  _S1014 = make_float3 (_S1007, _S1010, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1011));
            uint3  _S1015 = make_uint3 (g_idx_8, 5U, 0U);

#line 74
            float _S1016 = ((sh_coeffs_10.primal_0).load<float>((_S1015)));

#line 74
            float _S1017 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1015);

#line 74
            uint3  _S1018 = make_uint3 (g_idx_8, 5U, 1U);

#line 74
            float _S1019 = ((sh_coeffs_10.primal_0).load<float>((_S1018)));

#line 74
            float _S1020 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1018);

#line 74
            uint3  _S1021 = make_uint3 (g_idx_8, 5U, 2U);

#line 74
            float _S1022 = ((sh_coeffs_10.primal_0).load<float>((_S1021)));

#line 74
            float3  _S1023 = make_float3 (_S1016, _S1019, _S1022);

#line 74
            float3  _S1024 = make_float3 (_S1017, _S1020, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1021));
            uint3  _S1025 = make_uint3 (g_idx_8, 6U, 0U);

#line 75
            float _S1026 = ((sh_coeffs_10.primal_0).load<float>((_S1025)));

#line 75
            float _S1027 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1025);

#line 75
            uint3  _S1028 = make_uint3 (g_idx_8, 6U, 1U);

#line 75
            float _S1029 = ((sh_coeffs_10.primal_0).load<float>((_S1028)));

#line 75
            float _S1030 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1028);

#line 75
            uint3  _S1031 = make_uint3 (g_idx_8, 6U, 2U);

#line 75
            float _S1032 = ((sh_coeffs_10.primal_0).load<float>((_S1031)));

#line 75
            float3  _S1033 = make_float3 (_S1026, _S1029, _S1032);

#line 75
            float3  _S1034 = make_float3 (_S1027, _S1030, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1031));
            uint3  _S1035 = make_uint3 (g_idx_8, 7U, 0U);

#line 76
            float _S1036 = ((sh_coeffs_10.primal_0).load<float>((_S1035)));

#line 76
            float _S1037 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1035);

#line 76
            uint3  _S1038 = make_uint3 (g_idx_8, 7U, 1U);

#line 76
            float _S1039 = ((sh_coeffs_10.primal_0).load<float>((_S1038)));

#line 76
            float _S1040 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1038);

#line 76
            uint3  _S1041 = make_uint3 (g_idx_8, 7U, 2U);

#line 76
            float _S1042 = ((sh_coeffs_10.primal_0).load<float>((_S1041)));

#line 76
            float3  _S1043 = make_float3 (_S1036, _S1039, _S1042);

#line 76
            float3  _S1044 = make_float3 (_S1037, _S1040, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1041));
            uint3  _S1045 = make_uint3 (g_idx_8, 8U, 0U);

#line 77
            float _S1046 = ((sh_coeffs_10.primal_0).load<float>((_S1045)));

#line 77
            float _S1047 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1045);

#line 77
            uint3  _S1048 = make_uint3 (g_idx_8, 8U, 1U);

#line 77
            float _S1049 = ((sh_coeffs_10.primal_0).load<float>((_S1048)));

#line 77
            float _S1050 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1048);

#line 77
            uint3  _S1051 = make_uint3 (g_idx_8, 8U, 2U);

#line 77
            float _S1052 = ((sh_coeffs_10.primal_0).load<float>((_S1051)));

#line 77
            float3  _S1053 = make_float3 (_S1046, _S1049, _S1052);

#line 77
            float3  _S1054 = make_float3 (_S1047, _S1050, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1051));

            if(active_sh_15 > 2U)
            {

#line 80
                uint3  _S1055 = make_uint3 (g_idx_8, 9U, 0U);

#line 80
                float _S1056 = ((sh_coeffs_10.primal_0).load<float>((_S1055)));

#line 80
                float _S1057 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1055);

#line 80
                uint3  _S1058 = make_uint3 (g_idx_8, 9U, 1U);

#line 80
                float _S1059 = ((sh_coeffs_10.primal_0).load<float>((_S1058)));

#line 80
                float _S1060 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1058);

#line 80
                uint3  _S1061 = make_uint3 (g_idx_8, 9U, 2U);

#line 80
                float _S1062 = ((sh_coeffs_10.primal_0).load<float>((_S1061)));

#line 80
                float3  _S1063 = make_float3 (_S1056, _S1059, _S1062);

#line 80
                float3  _S1064 = make_float3 (_S1057, _S1060, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1061));
                uint3  _S1065 = make_uint3 (g_idx_8, 10U, 0U);

#line 81
                float _S1066 = ((sh_coeffs_10.primal_0).load<float>((_S1065)));

#line 81
                float _S1067 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1065);

#line 81
                uint3  _S1068 = make_uint3 (g_idx_8, 10U, 1U);

#line 81
                float _S1069 = ((sh_coeffs_10.primal_0).load<float>((_S1068)));

#line 81
                float _S1070 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1068);

#line 81
                uint3  _S1071 = make_uint3 (g_idx_8, 10U, 2U);

#line 81
                float _S1072 = ((sh_coeffs_10.primal_0).load<float>((_S1071)));

#line 81
                float3  _S1073 = make_float3 (_S1066, _S1069, _S1072);

#line 81
                float3  _S1074 = make_float3 (_S1067, _S1070, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1071));
                uint3  _S1075 = make_uint3 (g_idx_8, 11U, 0U);

#line 82
                float _S1076 = ((sh_coeffs_10.primal_0).load<float>((_S1075)));

#line 82
                float _S1077 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1075);

#line 82
                uint3  _S1078 = make_uint3 (g_idx_8, 11U, 1U);

#line 82
                float _S1079 = ((sh_coeffs_10.primal_0).load<float>((_S1078)));

#line 82
                float _S1080 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1078);

#line 82
                uint3  _S1081 = make_uint3 (g_idx_8, 11U, 2U);

#line 82
                float _S1082 = ((sh_coeffs_10.primal_0).load<float>((_S1081)));

#line 82
                float3  _S1083 = make_float3 (_S1076, _S1079, _S1082);

#line 82
                float3  _S1084 = make_float3 (_S1077, _S1080, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1081));
                uint3  _S1085 = make_uint3 (g_idx_8, 12U, 0U);

#line 83
                float _S1086 = ((sh_coeffs_10.primal_0).load<float>((_S1085)));

#line 83
                float _S1087 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1085);

#line 83
                uint3  _S1088 = make_uint3 (g_idx_8, 12U, 1U);

#line 83
                float _S1089 = ((sh_coeffs_10.primal_0).load<float>((_S1088)));

#line 83
                float _S1090 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1088);

#line 83
                uint3  _S1091 = make_uint3 (g_idx_8, 12U, 2U);

#line 83
                float _S1092 = ((sh_coeffs_10.primal_0).load<float>((_S1091)));

#line 83
                float3  _S1093 = make_float3 (_S1086, _S1089, _S1092);

#line 83
                float3  _S1094 = make_float3 (_S1087, _S1090, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1091));
                uint3  _S1095 = make_uint3 (g_idx_8, 13U, 0U);

#line 84
                float _S1096 = ((sh_coeffs_10.primal_0).load<float>((_S1095)));

#line 84
                float _S1097 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1095);

#line 84
                uint3  _S1098 = make_uint3 (g_idx_8, 13U, 1U);

#line 84
                float _S1099 = ((sh_coeffs_10.primal_0).load<float>((_S1098)));

#line 84
                float _S1100 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1098);

#line 84
                uint3  _S1101 = make_uint3 (g_idx_8, 13U, 2U);

#line 84
                float _S1102 = ((sh_coeffs_10.primal_0).load<float>((_S1101)));

#line 84
                float3  _S1103 = make_float3 (_S1096, _S1099, _S1102);

#line 84
                float3  _S1104 = make_float3 (_S1097, _S1100, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1101));
                uint3  _S1105 = make_uint3 (g_idx_8, 14U, 0U);

#line 85
                float _S1106 = ((sh_coeffs_10.primal_0).load<float>((_S1105)));

#line 85
                float _S1107 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1105);

#line 85
                uint3  _S1108 = make_uint3 (g_idx_8, 14U, 1U);

#line 85
                float _S1109 = ((sh_coeffs_10.primal_0).load<float>((_S1108)));

#line 85
                float _S1110 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1108);

#line 85
                uint3  _S1111 = make_uint3 (g_idx_8, 14U, 2U);

#line 85
                float _S1112 = ((sh_coeffs_10.primal_0).load<float>((_S1111)));

#line 85
                float3  _S1113 = make_float3 (_S1106, _S1109, _S1112);

#line 85
                float3  _S1114 = make_float3 (_S1107, _S1110, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1111));
                uint3  _S1115 = make_uint3 (g_idx_8, 15U, 0U);

#line 86
                float _S1116 = ((sh_coeffs_10.primal_0).load<float>((_S1115)));

#line 86
                float _S1117 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1115);

#line 86
                uint3  _S1118 = make_uint3 (g_idx_8, 15U, 1U);

#line 86
                float _S1119 = ((sh_coeffs_10.primal_0).load<float>((_S1118)));

#line 86
                float _S1120 = AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1118);

#line 86
                uint3  _S1121 = make_uint3 (g_idx_8, 15U, 2U);

#line 86
                float _S1122 = ((sh_coeffs_10.primal_0).load<float>((_S1121)));

#line 86
                float3  _S1123 = make_float3 (_S1116, _S1119, _S1122);

#line 86
                float3  _S1124 = make_float3 (_S1117, _S1120, AtomicAdd_load_forward_1(sh_coeffs_10.diff_1, _S1121));

#line 86
                (&g_sh_coeffs_2)->coeff0_0 = _S973;

#line 86
                (&g_sh_coeffs_2)->coeff1_0 = _S983;

#line 86
                (&g_sh_coeffs_2)->coeff2_0 = _S993;

#line 86
                (&g_sh_coeffs_2)->coeff3_0 = _S1003;

#line 86
                (&g_sh_coeffs_2)->coeff4_0 = _S1013;

#line 86
                (&g_sh_coeffs_2)->coeff5_0 = _S1023;

#line 86
                (&g_sh_coeffs_2)->coeff6_0 = _S1033;

#line 86
                (&g_sh_coeffs_2)->coeff7_0 = _S1043;

#line 86
                (&g_sh_coeffs_2)->coeff8_0 = _S1053;

#line 86
                (&g_sh_coeffs_2)->coeff9_0 = _S1063;

#line 86
                (&g_sh_coeffs_2)->coeff10_0 = _S1073;

#line 86
                (&g_sh_coeffs_2)->coeff11_0 = _S1083;

#line 86
                (&g_sh_coeffs_2)->coeff12_0 = _S1093;

#line 86
                (&g_sh_coeffs_2)->coeff13_0 = _S1103;

#line 86
                (&g_sh_coeffs_2)->coeff14_0 = _S1113;

#line 86
                (&g_sh_coeffs_2)->coeff15_0 = _S1123;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S974;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S984;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S994;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S1004;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S1014;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S1024;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S1034;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S1044;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S1054;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S1064;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S1074;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S1084;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S1094;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S1104;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S1114;

#line 86
                (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S1124;

#line 79
            }
            else
            {

#line 79
                (&g_sh_coeffs_2)->coeff0_0 = _S973;

#line 79
                (&g_sh_coeffs_2)->coeff1_0 = _S983;

#line 79
                (&g_sh_coeffs_2)->coeff2_0 = _S993;

#line 79
                (&g_sh_coeffs_2)->coeff3_0 = _S1003;

#line 79
                (&g_sh_coeffs_2)->coeff4_0 = _S1013;

#line 79
                (&g_sh_coeffs_2)->coeff5_0 = _S1023;

#line 79
                (&g_sh_coeffs_2)->coeff6_0 = _S1033;

#line 79
                (&g_sh_coeffs_2)->coeff7_0 = _S1043;

#line 79
                (&g_sh_coeffs_2)->coeff8_0 = _S1053;

#line 79
                (&g_sh_coeffs_2)->coeff9_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff10_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff11_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff12_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff13_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff14_0 = _S964;

#line 79
                (&g_sh_coeffs_2)->coeff15_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S974;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S984;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S994;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S1004;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S1014;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S1024;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S1034;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S1044;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S1054;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S964;

#line 79
                (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S964;

#line 79
            }

#line 72
        }
        else
        {

#line 72
            (&g_sh_coeffs_2)->coeff0_0 = _S973;

#line 72
            (&g_sh_coeffs_2)->coeff1_0 = _S983;

#line 72
            (&g_sh_coeffs_2)->coeff2_0 = _S993;

#line 72
            (&g_sh_coeffs_2)->coeff3_0 = _S1003;

#line 72
            (&g_sh_coeffs_2)->coeff4_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff5_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff6_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff7_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff8_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff9_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff10_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff11_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff12_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff13_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff14_0 = _S964;

#line 72
            (&g_sh_coeffs_2)->coeff15_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S974;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S984;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S994;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S1004;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S964;

#line 72
            (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S964;

#line 72
        }

#line 67
    }
    else
    {

#line 67
        (&g_sh_coeffs_2)->coeff0_0 = _S973;

#line 67
        (&g_sh_coeffs_2)->coeff1_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff2_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff3_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff4_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff5_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff6_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff7_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff8_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff9_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff10_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff11_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff12_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff13_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff14_0 = _S964;

#line 67
        (&g_sh_coeffs_2)->coeff15_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff0_0 = _S974;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff1_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff2_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff3_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff4_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff5_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff6_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff7_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff8_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff9_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff10_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff11_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff12_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff13_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff14_0 = _S964;

#line 67
        (&s_diff_g_sh_coeffs_0)->coeff15_0 = _S964;

#line 67
    }

#line 67
    DiffPair_SpherHarmCoeffs_0 _S1125 = { g_sh_coeffs_2, s_diff_g_sh_coeffs_0 };

#line 90
    return _S1125;
}


#line 186 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ DiffPair_vectorx3Cfloatx2C4x3E_0 s_fwd_read_t4_float4_0(uint idx_7, DiffTensorView_0 t4_3)
{

#line 43
    uint2  _S1126 = make_uint2 (idx_7, 0U);

#line 43
    float _S1127 = ((t4_3.primal_0).load<float>((_S1126)));

#line 43
    float _S1128 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1126);
    uint2  _S1129 = make_uint2 (idx_7, 1U);

#line 43
    float _S1130 = ((t4_3.primal_0).load<float>((_S1129)));

#line 43
    float _S1131 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1129);

    uint2  _S1132 = make_uint2 (idx_7, 2U);

#line 43
    float _S1133 = ((t4_3.primal_0).load<float>((_S1132)));

#line 43
    float _S1134 = AtomicAdd_load_forward_0(t4_3.diff_1, _S1132);


    uint2  _S1135 = make_uint2 (idx_7, 3U);

#line 43
    float _S1136 = ((t4_3.primal_0).load<float>((_S1135)));

#line 43
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1137 = { make_float4 (_S1127, _S1130, _S1133, _S1136), make_float4 (_S1128, _S1131, _S1134, AtomicAdd_load_forward_0(t4_3.diff_1, _S1135)) };

#line 43
    return _S1137;
}


#line 43
__device__ DiffPair_Gaussian_3D_0 s_fwd_load_gaussian_0(int g_idx_9, DiffTensorView_0 xyz_ws_9, DiffTensorView_0 sh_coeffs_11, DiffTensorView_0 rotations_7, DiffTensorView_0 scales_7, uint active_sh_16)
{

#line 184
    uint _S1138 = uint(g_idx_9);

#line 184
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1139 = s_fwd_read_t3_float3_0(_S1138, xyz_ws_9);
    DiffPair_SpherHarmCoeffs_0 _S1140 = s_fwd_read_spherical_harmonics_coeffs_0(_S1138, sh_coeffs_11, active_sh_16);
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1141 = s_fwd_read_t4_float4_0(_S1138, rotations_7);
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1142 = s_fwd_read_t3_float3_0(_S1138, scales_7);

    Gaussian_3D_0 _S1143 = { _S1139.primal_1, _S1140.primal_1, _S1141.primal_1, _S1142.primal_1 };

#line 189
    Gaussian_3D_0 _S1144 = { _S1139.differential_0, _S1140.differential_0, _S1141.differential_0, _S1142.differential_0 };

#line 189
    DiffPair_Gaussian_3D_0 _S1145 = { _S1143, _S1144 };

#line 189
    return _S1145;
}


#line 189
struct DiffPair_Splat_2D_Vertex_0
{
    Splat_2D_Vertex_0 primal_1;
    Splat_2D_Vertex_0 differential_0;
};


#line 127
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_geom_transform_points_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_6, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 dptransf_matrix_4)
{

#line 114
    float4  _S1146 = make_float4 (dppoint_6.primal_1.x, dppoint_6.primal_1.y, dppoint_6.primal_1.z, 1.0f);

#line 114
    float4  _S1147 = mul_4(dptransf_matrix_4.primal_1, _S1146);

#line 114
    float4  _S1148 = mul_4(dptransf_matrix_4.differential_0, _S1146) + mul_4(dptransf_matrix_4.primal_1, make_float4 (dppoint_6.differential_0.x, dppoint_6.differential_0.y, dppoint_6.differential_0.z, 0.0f));
    float3  _S1149 = float3 {_S1147.x, _S1147.y, _S1147.z};

#line 115
    float _S1150 = _S1147.w + 1.00000001168609742e-07f;

#line 115
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1151 = { _S1149 / make_float3 (_S1150), (float3 {_S1148.x, _S1148.y, _S1148.z} * make_float3 (_S1150) - _S1149 * make_float3 (_S1148.w)) / make_float3 (_S1150 * _S1150) };

#line 115
    return _S1151;
}


#line 115
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_geom_transform_points2_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_7, DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 dptransf_matrix_5)
{

#line 121
    float4  _S1152 = make_float4 (dppoint_7.primal_1.x, dppoint_7.primal_1.y, dppoint_7.primal_1.z, 1.0f);

#line 121
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1153 = { float3 {mul_4(dptransf_matrix_5.primal_1, _S1152).x, mul_4(dptransf_matrix_5.primal_1, _S1152).y, mul_4(dptransf_matrix_5.primal_1, _S1152).z}, float3 {(mul_4(dptransf_matrix_5.differential_0, _S1152) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).x, (mul_4(dptransf_matrix_5.differential_0, _S1152) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).y, (mul_4(dptransf_matrix_5.differential_0, _S1152) + mul_4(dptransf_matrix_5.primal_1, make_float4 (dppoint_7.differential_0.x, dppoint_7.differential_0.y, dppoint_7.differential_0.z, 0.0f))).z} };
    return _S1153;
}


#line 122
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_project_point_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dppoint_8, DiffPair_Camera_0 dpcam_8)
{


    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1154 = { dppoint_8.primal_1, dppoint_8.differential_0 };

#line 126
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1155 = { mul_2(dpcam_8.primal_1.proj_mat_1, dpcam_8.primal_1.world_view_transform_1), mul_2(dpcam_8.differential_0.proj_mat_0, dpcam_8.primal_1.world_view_transform_1) + mul_2(dpcam_8.primal_1.proj_mat_1, dpcam_8.differential_0.world_view_transform_0) };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1156 = s_fwd_geom_transform_points_0(_S1154, _S1155);

#line 127
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1157 = { dpcam_8.primal_1.world_view_transform_1, dpcam_8.differential_0.world_view_transform_0 };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1158 = s_fwd_geom_transform_points2_0(_S1154, _S1157);
    float _S1159 = _S1158.primal_1.z;

#line 129
    float _S1160 = _S1158.differential_0.z;

#line 129
    float3  _S1161 = _S1156.primal_1;

#line 129
    *&((&_S1161)->z) = _S1159;

#line 129
    float3  _S1162 = _S1156.differential_0;

#line 129
    *&((&_S1162)->z) = _S1160;

#line 129
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1163 = { _S1161, _S1162 };
    return _S1163;
}


#line 2107 "diff.meta.slang"
__device__ DiffPair_float_0 s_fwd_length_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_12)
{

#line 2092
    float _S1164 = dpx_12.primal_1.x;

#line 2092
    float _S1165 = dpx_12.differential_0.x * dpx_12.primal_1.x;

#line 2092
    float _S1166 = dpx_12.primal_1.y;

#line 2092
    float _S1167 = dpx_12.differential_0.y * dpx_12.primal_1.y;

#line 2092
    float _S1168 = dpx_12.primal_1.z;

#line 2092
    float _S1169 = dpx_12.differential_0.z * dpx_12.primal_1.z;

#line 2092
    DiffPair_float_0 _S1170 = { _S1164 * _S1164 + _S1166 * _S1166 + _S1168 * _S1168, _S1165 + _S1165 + (_S1167 + _S1167) + (_S1169 + _S1169) };

#line 2099
    DiffPair_float_0 _S1171 = _d_sqrt_1(_S1170);

#line 2099
    DiffPair_float_0 _S1172 = { _S1171.primal_1, _S1171.differential_0 };

#line 2099
    return _S1172;
}


#line 2099
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_normalize_impl_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpx_13)
{

#line 2154
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1173 = { dpx_13.primal_1, dpx_13.differential_0 };

    DiffPair_float_0 _S1174 = s_fwd_length_impl_0(_S1173);

#line 2156
    float _S1175 = 1.0f / _S1174.primal_1;

#line 2156
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1176 = { dpx_13.primal_1 * make_float3 (_S1175), dpx_13.differential_0 * make_float3 (_S1175) + make_float3 ((0.0f - _S1174.differential_0) / (_S1174.primal_1 * _S1174.primal_1)) * dpx_13.primal_1 };
    return _S1176;
}


#line 2157
__device__ DiffPair_vectorx3Cfloatx2C3x3E_0 s_fwd_compute_color_from_sh_coeffs_0(DiffPair_SpherHarmCoeffs_0 dpsh_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dpg_xyz_ws_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dpcam_pos_2, uint active_sh_17)
{

#line 94 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../spherical_harmonics.slang"
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1177 = { dpg_xyz_ws_2.primal_1 - dpcam_pos_2.primal_1, dpg_xyz_ws_2.differential_0 - dpcam_pos_2.differential_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1178 = s_fwd_normalize_impl_0(_S1177);

    float3  rgb_14 = make_float3 (0.282094806432724f) * dpsh_2.primal_1.coeff0_0;

#line 98
    float3  _S1179 = dpsh_2.differential_0.coeff0_0 * make_float3 (0.282094806432724f);

#line 98
    float3  rgb_15;

#line 98
    float3  s_diff_rgb_0;
    if(active_sh_17 > 0U)
    {

#line 100
        float _S1180 = _S1178.primal_1.y;

#line 100
        float _S1181 = _S1178.differential_0.y;

#line 100
        float _S1182 = 0.48860251903533936f * _S1180;

#line 100
        float _S1183 = _S1178.primal_1.z;

#line 100
        float _S1184 = _S1178.differential_0.z;

#line 100
        float _S1185 = 0.48860251903533936f * _S1183;

#line 100
        float _S1186 = _S1178.primal_1.x;

#line 100
        float _S1187 = _S1178.differential_0.x;

#line 100
        float _S1188 = 0.48860251903533936f * _S1186;

#line 100
        float3  rgb_16 = rgb_14 - make_float3 (_S1182) * dpsh_2.primal_1.coeff1_0 + make_float3 (_S1185) * dpsh_2.primal_1.coeff2_0 - make_float3 (_S1188) * dpsh_2.primal_1.coeff3_0;

#line 100
        float3  s_diff_rgb_1 = _S1179 - (make_float3 (_S1181 * 0.48860251903533936f) * dpsh_2.primal_1.coeff1_0 + dpsh_2.differential_0.coeff1_0 * make_float3 (_S1182)) + (make_float3 (_S1184 * 0.48860251903533936f) * dpsh_2.primal_1.coeff2_0 + dpsh_2.differential_0.coeff2_0 * make_float3 (_S1185)) - (make_float3 (_S1187 * 0.48860251903533936f) * dpsh_2.primal_1.coeff3_0 + dpsh_2.differential_0.coeff3_0 * make_float3 (_S1188));
        if(active_sh_17 > 1U)
        {
            float xx_3 = _S1186 * _S1186;

#line 103
            float _S1189 = _S1187 * _S1186;

#line 103
            float s_diff_xx_0 = _S1189 + _S1189;

#line 103
            float yy_3 = _S1180 * _S1180;

#line 103
            float _S1190 = _S1181 * _S1180;

#line 103
            float s_diff_yy_0 = _S1190 + _S1190;

#line 103
            float zz_3 = _S1183 * _S1183;

#line 103
            float _S1191 = _S1184 * _S1183;

#line 103
            float s_diff_zz_0 = _S1191 + _S1191;
            float xy_3 = _S1186 * _S1180;

#line 104
            float s_diff_xy_0 = _S1187 * _S1180 + _S1181 * _S1186;

            float _S1192 = 1.09254848957061768f * xy_3;
            float _S1193 = -1.09254848957061768f * (_S1180 * _S1183);
            float _S1194 = 2.0f * zz_3;

#line 108
            float _S1195 = s_diff_zz_0 * 2.0f;

#line 108
            float _S1196 = 0.31539157032966614f * (_S1194 - xx_3 - yy_3);
            float _S1197 = -1.09254848957061768f * (_S1186 * _S1183);
            float _S1198 = xx_3 - yy_3;

#line 110
            float _S1199 = s_diff_xx_0 - s_diff_yy_0;

#line 110
            float _S1200 = 0.54627424478530884f * _S1198;

#line 109
            float3  rgb_17 = rgb_16 + make_float3 (_S1192) * dpsh_2.primal_1.coeff4_0 + make_float3 (_S1193) * dpsh_2.primal_1.coeff5_0 + make_float3 (_S1196) * dpsh_2.primal_1.coeff6_0 + make_float3 (_S1197) * dpsh_2.primal_1.coeff7_0 + make_float3 (_S1200) * dpsh_2.primal_1.coeff8_0;

#line 109
            float3  s_diff_rgb_2 = s_diff_rgb_1 + (make_float3 (s_diff_xy_0 * 1.09254848957061768f) * dpsh_2.primal_1.coeff4_0 + dpsh_2.differential_0.coeff4_0 * make_float3 (_S1192)) + (make_float3 ((_S1181 * _S1183 + _S1184 * _S1180) * -1.09254848957061768f) * dpsh_2.primal_1.coeff5_0 + dpsh_2.differential_0.coeff5_0 * make_float3 (_S1193)) + (make_float3 ((_S1195 - s_diff_xx_0 - s_diff_yy_0) * 0.31539157032966614f) * dpsh_2.primal_1.coeff6_0 + dpsh_2.differential_0.coeff6_0 * make_float3 (_S1196)) + (make_float3 ((_S1187 * _S1183 + _S1184 * _S1186) * -1.09254848957061768f) * dpsh_2.primal_1.coeff7_0 + dpsh_2.differential_0.coeff7_0 * make_float3 (_S1197)) + (make_float3 (_S1199 * 0.54627424478530884f) * dpsh_2.primal_1.coeff8_0 + dpsh_2.differential_0.coeff8_0 * make_float3 (_S1200));


            if(active_sh_17 > 2U)
            {

                float _S1201 = -0.59004360437393188f * _S1180;

#line 115
                float _S1202 = 3.0f * xx_3;

#line 115
                float _S1203 = s_diff_xx_0 * 3.0f;

#line 115
                float _S1204 = _S1202 - yy_3;

#line 115
                float _S1205 = _S1201 * _S1204;
                float _S1206 = 2.89061141014099121f * xy_3;

#line 116
                float _S1207 = _S1206 * _S1183;
                float _S1208 = -0.4570457935333252f * _S1180;

#line 117
                float _S1209 = 4.0f * zz_3 - xx_3 - yy_3;

#line 117
                float _S1210 = s_diff_zz_0 * 4.0f - s_diff_xx_0 - s_diff_yy_0;

#line 117
                float _S1211 = _S1208 * _S1209;
                float _S1212 = 0.37317633628845215f * _S1183;

#line 118
                float _S1213 = 3.0f * yy_3;

#line 118
                float _S1214 = s_diff_yy_0 * 3.0f;

#line 118
                float _S1215 = _S1194 - _S1202 - _S1213;

#line 118
                float _S1216 = _S1212 * _S1215;
                float _S1217 = -0.4570457935333252f * _S1186;

#line 119
                float _S1218 = _S1217 * _S1209;
                float _S1219 = 1.44530570507049561f * _S1183;

#line 120
                float _S1220 = _S1219 * _S1198;
                float _S1221 = -0.59004360437393188f * _S1186;

#line 121
                float _S1222 = xx_3 - _S1213;

#line 121
                float _S1223 = _S1221 * _S1222;

#line 120
                float3  _S1224 = s_diff_rgb_2 + (make_float3 (_S1181 * -0.59004360437393188f * _S1204 + (_S1203 - s_diff_yy_0) * _S1201) * dpsh_2.primal_1.coeff9_0 + dpsh_2.differential_0.coeff9_0 * make_float3 (_S1205)) + (make_float3 (s_diff_xy_0 * 2.89061141014099121f * _S1183 + _S1184 * _S1206) * dpsh_2.primal_1.coeff10_0 + dpsh_2.differential_0.coeff10_0 * make_float3 (_S1207)) + (make_float3 (_S1181 * -0.4570457935333252f * _S1209 + _S1210 * _S1208) * dpsh_2.primal_1.coeff11_0 + dpsh_2.differential_0.coeff11_0 * make_float3 (_S1211)) + (make_float3 (_S1184 * 0.37317633628845215f * _S1215 + (_S1195 - _S1203 - _S1214) * _S1212) * dpsh_2.primal_1.coeff12_0 + dpsh_2.differential_0.coeff12_0 * make_float3 (_S1216)) + (make_float3 (_S1187 * -0.4570457935333252f * _S1209 + _S1210 * _S1217) * dpsh_2.primal_1.coeff13_0 + dpsh_2.differential_0.coeff13_0 * make_float3 (_S1218)) + (make_float3 (_S1184 * 1.44530570507049561f * _S1198 + _S1199 * _S1219) * dpsh_2.primal_1.coeff14_0 + dpsh_2.differential_0.coeff14_0 * make_float3 (_S1220)) + (make_float3 (_S1187 * -0.59004360437393188f * _S1222 + (s_diff_xx_0 - _S1214) * _S1221) * dpsh_2.primal_1.coeff15_0 + dpsh_2.differential_0.coeff15_0 * make_float3 (_S1223));

#line 120
                rgb_15 = rgb_17 + make_float3 (_S1205) * dpsh_2.primal_1.coeff9_0 + make_float3 (_S1207) * dpsh_2.primal_1.coeff10_0 + make_float3 (_S1211) * dpsh_2.primal_1.coeff11_0 + make_float3 (_S1216) * dpsh_2.primal_1.coeff12_0 + make_float3 (_S1218) * dpsh_2.primal_1.coeff13_0 + make_float3 (_S1220) * dpsh_2.primal_1.coeff14_0 + make_float3 (_S1223) * dpsh_2.primal_1.coeff15_0;

#line 120
                s_diff_rgb_0 = _S1224;

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
        s_diff_rgb_0 = _S1179;

#line 99
    }

#line 99
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1225 = { rgb_15 + make_float3 (0.5f), s_diff_rgb_0 };

#line 99
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1226 = { make_float3 (0.0f), make_float3 (0.0f) };

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1227 = _d_max_vector_1(_S1225, _S1226);

#line 128
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1228 = { _S1227.primal_1, _S1227.differential_0 };

#line 128
    return _S1228;
}


#line 235 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
__device__ DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 s_fwd_get_covariance_from_quat_scales_0(DiffPair_vectorx3Cfloatx2C4x3E_0 dpq_2, DiffPair_vectorx3Cfloatx2C3x3E_0 dps_2)
{

#line 287
    float _S1229 = dpq_2.primal_1.z;



    float _S1230 = _S1229 * _S1229;

#line 291
    float _S1231 = dpq_2.differential_0.z * dpq_2.primal_1.z;

#line 291
    float _S1232 = _S1231 + _S1231;

#line 291
    float _S1233 = dpq_2.primal_1.w * dpq_2.primal_1.w;

#line 291
    float _S1234 = dpq_2.differential_0.w * dpq_2.primal_1.w;

#line 291
    float _S1235 = _S1234 + _S1234;

#line 291
    float _S1236 = dpq_2.primal_1.y * dpq_2.primal_1.z;

#line 291
    float _S1237 = dpq_2.differential_0.y * dpq_2.primal_1.z + dpq_2.differential_0.z * dpq_2.primal_1.y;

#line 291
    float _S1238 = dpq_2.primal_1.x * dpq_2.primal_1.w;

#line 291
    float _S1239 = dpq_2.differential_0.x * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.x;

#line 291
    float _S1240 = dpq_2.primal_1.y * dpq_2.primal_1.w;

#line 291
    float _S1241 = dpq_2.differential_0.y * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.y;

#line 291
    float _S1242 = dpq_2.primal_1.x * dpq_2.primal_1.z;

#line 291
    float _S1243 = dpq_2.differential_0.x * dpq_2.primal_1.z + dpq_2.differential_0.z * dpq_2.primal_1.x;
    float _S1244 = dpq_2.primal_1.y * dpq_2.primal_1.y;

#line 292
    float _S1245 = dpq_2.differential_0.y * dpq_2.primal_1.y;

#line 292
    float _S1246 = _S1245 + _S1245;

#line 292
    float _S1247 = dpq_2.primal_1.z * dpq_2.primal_1.w;

#line 292
    float _S1248 = dpq_2.differential_0.z * dpq_2.primal_1.w + dpq_2.differential_0.w * dpq_2.primal_1.z;

#line 292
    float _S1249 = dpq_2.primal_1.x * dpq_2.primal_1.y;

#line 292
    float _S1250 = dpq_2.differential_0.x * dpq_2.primal_1.y + dpq_2.differential_0.y * dpq_2.primal_1.x;

#line 290
    Matrix<float, 3, 3>  rotation_matrix_1 = makeMatrix<float, 3, 3> (1.0f - 2.0f * (_S1230 + _S1233), 2.0f * (_S1236 - _S1238), 2.0f * (_S1240 + _S1242), 2.0f * (_S1236 + _S1238), 1.0f - 2.0f * (_S1244 + _S1233), 2.0f * (_S1247 - _S1249), 2.0f * (_S1240 - _S1242), 2.0f * (_S1247 + _S1249), 1.0f - 2.0f * (_S1244 + _S1230));

#line 295
    Matrix<float, 3, 3>  scales_matrix_1 = makeMatrix<float, 3, 3> (dps_2.primal_1.x, 0.0f, 0.0f, 0.0f, dps_2.primal_1.y, 0.0f, 0.0f, 0.0f, dps_2.primal_1.z);



    Matrix<float, 3, 3>  _S1251 = mul_3(rotation_matrix_1, scales_matrix_1);

#line 299
    Matrix<float, 3, 3>  _S1252 = mul_3(makeMatrix<float, 3, 3> (0.0f - (_S1232 + _S1235) * 2.0f, (_S1237 - _S1239) * 2.0f, (_S1241 + _S1243) * 2.0f, (_S1237 + _S1239) * 2.0f, 0.0f - (_S1246 + _S1235) * 2.0f, (_S1248 - _S1250) * 2.0f, (_S1241 - _S1243) * 2.0f, (_S1248 + _S1250) * 2.0f, 0.0f - (_S1246 + _S1232) * 2.0f), scales_matrix_1) + mul_3(rotation_matrix_1, makeMatrix<float, 3, 3> (dps_2.differential_0.x, 0.0f, 0.0f, 0.0f, dps_2.differential_0.y, 0.0f, 0.0f, 0.0f, dps_2.differential_0.z));

    Matrix<float, 3, 3>  _S1253 = transpose_0(_S1251);

#line 301
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1254 = { mul_3(_S1251, _S1253), mul_3(_S1252, _S1253) + mul_3(_S1251, transpose_0(_S1252)) };

#line 301
    return _S1254;
}


#line 160
__device__ DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 s_fwd_compute_jacobian_0(DiffPair_vectorx3Cfloatx2C3x3E_0 dpxyz_ws_4, DiffPair_Camera_0 dpcam_9)
{

#line 134
    DiffPair_float_0 _S1255 = { dpcam_9.primal_1.fovx_1 / 2.0f, dpcam_9.differential_0.fovx_0 * 0.5f };
    DiffPair_float_0 _S1256 = _d_tan_1(_S1255);

#line 135
    DiffPair_float_0 _S1257 = { dpcam_9.primal_1.fovy_1 / 2.0f, dpcam_9.differential_0.fovy_0 * 0.5f };
    DiffPair_float_0 _S1258 = _d_tan_1(_S1257);
    float _S1259 = float(dpcam_9.primal_1.W_0);

#line 137
    float _S1260 = 2.0f * _S1256.primal_1;

#line 137
    float h_x_3 = _S1259 / _S1260;

#line 137
    float s_diff_h_x_0 = (0.0f - _S1259 * (_S1256.differential_0 * 2.0f)) / (_S1260 * _S1260);
    float _S1261 = float(dpcam_9.primal_1.H_0);

#line 138
    float _S1262 = 2.0f * _S1258.primal_1;

#line 138
    float h_y_3 = _S1261 / _S1262;

#line 138
    float s_diff_h_y_0 = (0.0f - _S1261 * (_S1258.differential_0 * 2.0f)) / (_S1262 * _S1262);

#line 138
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1263 = { dpxyz_ws_4.primal_1, dpxyz_ws_4.differential_0 };

#line 138
    DiffPair_matrixx3Cfloatx2C4x2C4x3E_0 _S1264 = { dpcam_9.primal_1.world_view_transform_1, dpcam_9.differential_0.world_view_transform_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1265 = s_fwd_geom_transform_points_0(_S1263, _S1264);


    float limx_3 = 1.29999995231628418f * _S1256.primal_1;

#line 143
    float _S1266 = _S1256.differential_0 * 1.29999995231628418f;
    float limy_3 = 1.29999995231628418f * _S1258.primal_1;

#line 144
    float _S1267 = _S1258.differential_0 * 1.29999995231628418f;
    float _S1268 = _S1265.primal_1.x;

#line 145
    float _S1269 = _S1265.primal_1.z;

#line 145
    float _S1270 = _S1265.differential_0.z;

#line 145
    float _S1271 = _S1269 * _S1269;
    float _S1272 = _S1265.primal_1.y;

#line 146
    float tytz_3 = _S1272 / _S1269;

#line 146
    float s_diff_tytz_0 = (_S1265.differential_0.y * _S1269 - _S1272 * _S1270) / _S1271;

#line 146
    DiffPair_float_0 _S1273 = { - limx_3, - _S1266 };

#line 146
    DiffPair_float_0 _S1274 = { _S1268 / _S1269, (_S1265.differential_0.x * _S1269 - _S1268 * _S1270) / _S1271 };
    DiffPair_float_0 _S1275 = _d_max_1(_S1273, _S1274);

#line 147
    DiffPair_float_0 _S1276 = { limx_3, _S1266 };

#line 147
    DiffPair_float_0 _S1277 = { _S1275.primal_1, _S1275.differential_0 };

#line 147
    DiffPair_float_0 _S1278 = _d_min_1(_S1276, _S1277);

#line 147
    float _S1279 = _S1278.primal_1 * _S1269;

#line 147
    float _S1280 = _S1278.differential_0 * _S1269 + _S1270 * _S1278.primal_1;

#line 147
    float3  _S1281 = _S1265.primal_1;

#line 147
    *&((&_S1281)->x) = _S1279;

#line 147
    float3  _S1282 = _S1265.differential_0;

#line 147
    *&((&_S1282)->x) = _S1280;

#line 147
    DiffPair_float_0 _S1283 = { - limy_3, - _S1267 };

#line 147
    DiffPair_float_0 _S1284 = { tytz_3, s_diff_tytz_0 };
    DiffPair_float_0 _S1285 = _d_max_1(_S1283, _S1284);

#line 148
    DiffPair_float_0 _S1286 = { limy_3, _S1267 };

#line 148
    DiffPair_float_0 _S1287 = { _S1285.primal_1, _S1285.differential_0 };

#line 148
    DiffPair_float_0 _S1288 = _d_min_1(_S1286, _S1287);

#line 148
    float _S1289 = _S1281.z;

#line 148
    float _S1290 = _S1288.differential_0 * _S1289 + _S1282.z * _S1288.primal_1;

#line 148
    *&((&_S1281)->y) = _S1288.primal_1 * _S1289;

#line 148
    *&((&_S1282)->y) = _S1290;

    float _S1291 = _S1281.z;

#line 150
    float _S1292 = _S1282.z;

#line 150
    float _S1293 = _S1291 * _S1291;

#line 150
    float _S1294 = _S1281.x;

#line 150
    float _S1295 = - (h_x_3 * _S1294);

#line 150
    float _S1296 = _S1292 * _S1291;

#line 150
    float _S1297 = _S1296 + _S1296;

#line 150
    float _S1298 = _S1293 * _S1293;
    float _S1299 = _S1281.y;

#line 151
    float _S1300 = - (h_y_3 * _S1299);

#line 151
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1301 = { makeMatrix<float, 3, 3> (h_x_3 / _S1291, 0.0f, _S1295 / _S1293, 0.0f, h_y_3 / _S1291, _S1300 / _S1293, 0.0f, 0.0f, 0.0f), makeMatrix<float, 3, 3> ((s_diff_h_x_0 * _S1291 - h_x_3 * _S1292) / _S1293, 0.0f, (- (s_diff_h_x_0 * _S1294 + _S1282.x * h_x_3) * _S1293 - _S1295 * _S1297) / _S1298, 0.0f, (s_diff_h_y_0 * _S1291 - h_y_3 * _S1292) / _S1293, (- (s_diff_h_y_0 * _S1299 + _S1282.y * h_y_3) * _S1293 - _S1300 * _S1297) / _S1298, 0.0f, 0.0f, 0.0f) };


    return _S1301;
}


#line 154
__device__ DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 s_fwd_covariance_3d_to_2d_0(DiffPair_Camera_0 dpcam_10, DiffPair_vectorx3Cfloatx2C3x3E_0 dpxyz_ws_5, DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 dpcov_ws_2)
{


    Matrix<float, 3, 3>  _S1302 = makeMatrix<float, 3, 3> (float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(0)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(0)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(0)].z}, float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(1)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(1)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(1)].z}, float3 {dpcam_10.primal_1.world_view_transform_1.rows[int(2)].x, dpcam_10.primal_1.world_view_transform_1.rows[int(2)].y, dpcam_10.primal_1.world_view_transform_1.rows[int(2)].z});

#line 158
    Matrix<float, 3, 3>  _S1303 = makeMatrix<float, 3, 3> (float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(0)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(0)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(0)].z}, float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(1)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(1)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(1)].z}, float3 {dpcam_10.differential_0.world_view_transform_0.rows[int(2)].x, dpcam_10.differential_0.world_view_transform_0.rows[int(2)].y, dpcam_10.differential_0.world_view_transform_0.rows[int(2)].z});

#line 158
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1304 = { dpxyz_ws_5.primal_1, dpxyz_ws_5.differential_0 };

#line 158
    DiffPair_Camera_0 _S1305 = { dpcam_10.primal_1, dpcam_10.differential_0 };

    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1306 = s_fwd_compute_jacobian_0(_S1304, _S1305);
    Matrix<float, 3, 3>  _S1307 = transpose_0(_S1302);

#line 161
    Matrix<float, 3, 3>  _S1308 = transpose_0(_S1306.primal_1);

#line 161
    Matrix<float, 3, 3>  _S1309 = mul_3(_S1307, _S1308);

#line 161
    Matrix<float, 3, 3>  _S1310 = mul_3(dpcov_ws_2.primal_1, _S1309);

#line 161
    Matrix<float, 3, 3>  _S1311 = mul_3(_S1302, _S1310);

#line 161
    Matrix<float, 3, 3>  _S1312 = mul_3(_S1306.primal_1, _S1311);

#line 161
    Matrix<float, 3, 3>  _S1313 = mul_3(_S1306.differential_0, _S1311) + mul_3(_S1306.primal_1, mul_3(_S1303, _S1310) + mul_3(_S1302, mul_3(dpcov_ws_2.differential_0, _S1309) + mul_3(dpcov_ws_2.primal_1, mul_3(transpose_0(_S1303), _S1308) + mul_3(_S1307, transpose_0(_S1306.differential_0)))));
    float _S1314 = _S1312.rows[int(0)].x + 0.30000001192092896f;

#line 162
    Matrix<float, 3, 3>  _S1315 = _S1312;

#line 162
    *&(((&_S1315)->rows + (int(0)))->x) = _S1314;

#line 162
    Matrix<float, 3, 3>  _S1316 = _S1313;

#line 162
    *&(((&_S1316)->rows + (int(0)))->x) = _S1313.rows[int(0)].x;

#line 162
    *&(((&_S1315)->rows + (int(1)))->y) = _S1312.rows[int(1)].y + 0.30000001192092896f;

#line 162
    *&(((&_S1316)->rows + (int(1)))->y) = _S1313.rows[int(1)].y;

#line 162
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1317 = { makeMatrix<float, 2, 2> (float2 {_S1315.rows[int(0)].x, _S1315.rows[int(0)].y}, float2 {_S1315.rows[int(1)].x, _S1315.rows[int(1)].y}), makeMatrix<float, 2, 2> (float2 {_S1316.rows[int(0)].x, _S1316.rows[int(0)].y}, float2 {_S1316.rows[int(1)].x, _S1316.rows[int(1)].y}) };


    return _S1317;
}


#line 165
__device__ DiffPair_Splat_2D_Vertex_0 s_fwd_project_gaussian_to_camera_0(DiffPair_Gaussian_3D_0 dpg_2, DiffPair_Camera_0 dpcam_11, uint active_sh_18)
{

#line 229
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1318 = { dpg_2.primal_1.xyz_ws_0, dpg_2.differential_0.xyz_ws_0 };

#line 229
    DiffPair_Camera_0 _S1319 = { dpcam_11.primal_1, dpcam_11.differential_0 };
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1320 = s_fwd_project_point_0(_S1318, _S1319);
    if(_S1320.primal_1.z <= 0.20000000298023224f)
    {

#line 232
        float3  _S1321 = make_float3 (0.0f);

#line 232
        float3  _S1322 = make_float3 (0.0f);

#line 232
        Splat_2D_Vertex_0 _S1323 = { _S1321, _S1321, makeMatrix<float, 2, 2> (0.0f) };

#line 232
        Splat_2D_Vertex_0 _S1324 = { _S1322, _S1322, makeMatrix<float, 2, 2> (0.0f) };

#line 232
        DiffPair_Splat_2D_Vertex_0 _S1325 = { _S1323, _S1324 };

#line 232
        return _S1325;
    }

#line 232
    DiffPair_SpherHarmCoeffs_0 _S1326 = { dpg_2.primal_1.sh_coeffs_0, dpg_2.differential_0.sh_coeffs_0 };

#line 232
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1327 = { dpcam_11.primal_1.position_1, dpcam_11.differential_0.position_0 };

    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1328 = s_fwd_compute_color_from_sh_coeffs_0(_S1326, _S1318, _S1327, active_sh_18);

#line 234
    DiffPair_vectorx3Cfloatx2C4x3E_0 _S1329 = { dpg_2.primal_1.rotations_0, dpg_2.differential_0.rotations_0 };

#line 234
    DiffPair_vectorx3Cfloatx2C3x3E_0 _S1330 = { dpg_2.primal_1.scales_0, dpg_2.differential_0.scales_0 };
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1331 = s_fwd_get_covariance_from_quat_scales_0(_S1329, _S1330);

#line 235
    DiffPair_matrixx3Cfloatx2C3x2C3x3E_0 _S1332 = { _S1331.primal_1, _S1331.differential_0 };
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1333 = s_fwd_covariance_3d_to_2d_0(_S1319, _S1318, _S1332);

    Splat_2D_Vertex_0 _S1334 = { _S1320.primal_1, _S1328.primal_1, _S1333.primal_1 };

#line 238
    Splat_2D_Vertex_0 _S1335 = { _S1320.differential_0, _S1328.differential_0, _S1333.differential_0 };

#line 238
    DiffPair_Splat_2D_Vertex_0 _S1336 = { _S1334, _S1335 };

#line 238
    return _S1336;
}


#line 89 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
__device__ DiffPair_float_0 s_fwd_compute_det_0(DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 dpM_2)
{

#line 210 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    DiffPair_float_0 _S1337 = { dpM_2.primal_1.rows[int(0)].x * dpM_2.primal_1.rows[int(1)].y - dpM_2.primal_1.rows[int(0)].y * dpM_2.primal_1.rows[int(1)].x, dpM_2.differential_0.rows[int(0)].x * dpM_2.primal_1.rows[int(1)].y + dpM_2.differential_0.rows[int(1)].y * dpM_2.primal_1.rows[int(0)].x - (dpM_2.differential_0.rows[int(0)].y * dpM_2.primal_1.rows[int(1)].x + dpM_2.differential_0.rows[int(1)].x * dpM_2.primal_1.rows[int(0)].y) };
    return _S1337;
}


#line 97 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
__device__ DiffPair_float_0 s_fwd_ndc2pix_0(DiffPair_float_0 dpv_2, int S_3)
{

#line 70 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/../utils.slang"
    float _S1338 = float(S_3);

#line 70
    DiffPair_float_0 _S1339 = { ((dpv_2.primal_1 + 1.0f) * _S1338 - 1.0f) * 0.5f, dpv_2.differential_0 * _S1338 * 0.5f };

#line 70
    return _S1339;
}


#line 70
__device__ void s_fwd_vertex_shader_0(DiffTensorView_0 xyz_ws_10, DiffTensorView_0 sh_coeffs_12, DiffTensorView_0 rotations_8, DiffTensorView_0 scales_8, TensorView opcities_3, uint active_sh_19, TensorView world_view_transform_6, TensorView proj_mat_6, TensorView cam_pos_4, TensorView out_tiles_touched_3, TensorView out_rect_tile_space_3, TensorView out_radii_3, DiffTensorView_0 out_xyz_vs_3, DiffTensorView_0 out_inv_cov_vs_3, DiffTensorView_0 out_rgb_3, float fovy_6, float fovx_6, uint image_height_3, uint image_width_3, uint grid_height_3, uint grid_width_3, uint tile_height_3, uint tile_width_3)
{

#line 77 "D:/A_study/nerf3dgs/gaussian-splatting(test)/slangRenderers/shader/speedy-splat/speedy_vertex_shader.slang"
    uint g_idx_10 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

    if(g_idx_10 >= DiffTensorView_size_0(xyz_ws_10, 0U))
    {

#line 80
        return;
    }
    Camera_0 cam_5 = load_camera_0(world_view_transform_6, proj_mat_6, cam_pos_4, fovy_6, fovx_6, image_height_3, image_width_3);
    DiffPair_Gaussian_3D_0 _S1340 = s_fwd_load_gaussian_0(int(g_idx_10), xyz_ws_10, sh_coeffs_12, rotations_8, scales_8, active_sh_19);

#line 83
    DiffPair_Gaussian_3D_0 _S1341 = { _S1340.primal_1, _S1340.differential_0 };

#line 83
    DiffPair_Camera_0 _S1342 = { cam_5, Camera_x24_syn_dzero_0() };
    DiffPair_Splat_2D_Vertex_0 _S1343 = s_fwd_project_gaussian_to_camera_0(_S1341, _S1342, active_sh_19);
    float _S1344 = _S1343.primal_1.xyz_vs_0.z;

#line 85
    float _S1345 = _S1343.differential_0.xyz_vs_0.z;

#line 85
    if(_S1344 <= 0.20000000298023224f)
    {

#line 86
        return;
    }

#line 86
    DiffPair_matrixx3Cfloatx2C2x2C2x3E_0 _S1346 = { _S1343.primal_1.cov_vs_0, _S1343.differential_0.cov_vs_0 };


    DiffPair_float_0 _S1347 = s_fwd_compute_det_0(_S1346);

    if(_S1347.primal_1 == 0.0f)
    {

#line 92
        return;
    }

#line 93
    float radius_1 = splat_radius_0(_S1343.primal_1.cov_vs_0, _S1347.primal_1);

    Matrix<float, 2, 2>  _S1348 = makeMatrix<float, 2, 2> (_S1343.primal_1.cov_vs_0.rows[int(1)].y, - _S1343.primal_1.cov_vs_0.rows[int(0)].y, - _S1343.primal_1.cov_vs_0.rows[int(1)].x, _S1343.primal_1.cov_vs_0.rows[int(0)].x);

#line 95
    Matrix<float, 2, 2>  g_inv_cov_vs_2 = _S1348 / makeMatrix<float, 2, 2> (_S1347.primal_1);

#line 95
    Matrix<float, 2, 2>  s_diff_g_inv_cov_vs_0 = (makeMatrix<float, 2, 2> (_S1343.differential_0.cov_vs_0.rows[int(1)].y, - _S1343.differential_0.cov_vs_0.rows[int(0)].y, - _S1343.differential_0.cov_vs_0.rows[int(1)].x, _S1343.differential_0.cov_vs_0.rows[int(0)].x) * makeMatrix<float, 2, 2> (_S1347.primal_1) - _S1348 * makeMatrix<float, 2, 2> (_S1347.differential_0)) / makeMatrix<float, 2, 2> (_S1347.primal_1 * _S1347.primal_1);

#line 95
    DiffPair_float_0 _S1349 = { _S1343.primal_1.xyz_vs_0.x, _S1343.differential_0.xyz_vs_0.x };

#line 95
    DiffPair_float_0 _S1350 = { _S1343.primal_1.xyz_vs_0.y, _S1343.differential_0.xyz_vs_0.y };

    float2  pixelspace_xy_2 = make_float2 (s_fwd_ndc2pix_0(_S1349, int(image_width_3)).primal_1, s_fwd_ndc2pix_0(_S1350, int(image_height_3)).primal_1);



    float3  _S1351 = make_float3 (g_inv_cov_vs_2.rows[int(0)].x, g_inv_cov_vs_2.rows[int(0)].y, g_inv_cov_vs_2.rows[int(1)].y);

#line 101
    float _S1352 = ((opcities_3).load<float>((g_idx_10)));

#line 101
    int2  _S1353 = make_int2 (int(grid_width_3), int(grid_height_3));

#line 101
    int2  _S1354 = make_int2 (int(tile_width_3), int(tile_height_3));

#line 101
    uint _S1355 = 0U;

#line 101
    rectangle_0 rect_tile_space_2 = computeSnugBox_0(_S1351, pixelspace_xy_2, _S1352, _S1353, _S1354, &_S1355);

#line 101
    uint n_tiles_2 = _S1355;

    if(_S1355 == 0U)
    {

#line 104
        return;
    }

    (out_radii_3).store<int>((g_idx_10), (int(uint(radius_1))));
    (out_tiles_touched_3).store<int>((g_idx_10), (int(n_tiles_2)));
    uint2  _S1356 = make_uint2 (g_idx_10, 0U);

#line 109
    (out_rect_tile_space_3).store<int>((g_idx_10), (0U), (rect_tile_space_2.min_x_0));
    uint2  _S1357 = make_uint2 (g_idx_10, 1U);

#line 110
    (out_rect_tile_space_3).store<int>((g_idx_10), (1U), (rect_tile_space_2.min_y_0));
    uint2  _S1358 = make_uint2 (g_idx_10, 2U);

#line 111
    (out_rect_tile_space_3).store<int>((g_idx_10), (2U), (rect_tile_space_2.max_x_0));
    (out_rect_tile_space_3).store<int>((g_idx_10), (3U), (rect_tile_space_2.max_y_0));

    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1356, _S1349);
    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1357, _S1350);

#line 115
    DiffPair_float_0 _S1359 = { _S1344, _S1345 };
    DiffTensorView_storeOnce_forward_0(out_xyz_vs_3, _S1358, _S1359);

#line 116
    DiffPair_float_0 _S1360 = { g_inv_cov_vs_2.rows[int(0)].x, s_diff_g_inv_cov_vs_0.rows[int(0)].x };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 0U, 0U), _S1360);

#line 117
    DiffPair_float_0 _S1361 = { g_inv_cov_vs_2.rows[int(0)].y, s_diff_g_inv_cov_vs_0.rows[int(0)].y };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 0U, 1U), _S1361);

#line 118
    DiffPair_float_0 _S1362 = { g_inv_cov_vs_2.rows[int(1)].x, s_diff_g_inv_cov_vs_0.rows[int(1)].x };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 1U, 0U), _S1362);

#line 119
    DiffPair_float_0 _S1363 = { g_inv_cov_vs_2.rows[int(1)].y, s_diff_g_inv_cov_vs_0.rows[int(1)].y };
    DiffTensorView_storeOnce_forward_1(out_inv_cov_vs_3, make_uint3 (g_idx_10, 1U, 1U), _S1363);

#line 120
    DiffPair_float_0 _S1364 = { _S1343.primal_1.rgb_0.x, _S1343.differential_0.rgb_0.x };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1356, _S1364);

#line 121
    DiffPair_float_0 _S1365 = { _S1343.primal_1.rgb_0.y, _S1343.differential_0.rgb_0.y };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1357, _S1365);

#line 122
    DiffPair_float_0 _S1366 = { _S1343.primal_1.rgb_0.z, _S1343.differential_0.rgb_0.z };
    DiffTensorView_storeOnce_forward_0(out_rgb_3, _S1358, _S1366);
    return;
}


#line 124
extern "C" {
__global__ void __kernel__vertex_shader_fwd_diff(DiffTensorView_0 xyz_ws_11, DiffTensorView_0 sh_coeffs_13, DiffTensorView_0 rotations_9, DiffTensorView_0 scales_9, TensorView opcities_4, uint active_sh_20, TensorView world_view_transform_7, TensorView proj_mat_7, TensorView cam_pos_5, TensorView out_tiles_touched_4, TensorView out_rect_tile_space_4, TensorView out_radii_4, DiffTensorView_0 out_xyz_vs_4, DiffTensorView_0 out_inv_cov_vs_4, DiffTensorView_0 out_rgb_4, float fovy_7, float fovx_7, uint image_height_4, uint image_width_4, uint grid_height_4, uint grid_width_4, uint tile_height_4, uint tile_width_4)
{

#line 124
    s_fwd_vertex_shader_0(xyz_ws_11, sh_coeffs_13, rotations_9, scales_9, opcities_4, active_sh_20, world_view_transform_7, proj_mat_7, cam_pos_5, out_tiles_touched_4, out_rect_tile_space_4, out_radii_4, out_xyz_vs_4, out_inv_cov_vs_4, out_rgb_4, fovy_7, fovx_7, image_height_4, image_width_4, grid_height_4, grid_width_4, tile_height_4, tile_width_4);

#line 124
    return;
}

}

#line 53
__global__ void __kernel__vertex_shader(DiffTensorView_0 xyz_ws_12, DiffTensorView_0 sh_coeffs_14, DiffTensorView_0 rotations_10, DiffTensorView_0 scales_10, TensorView opcities_5, uint active_sh_21, TensorView world_view_transform_8, TensorView proj_mat_8, TensorView cam_pos_6, TensorView out_tiles_touched_5, TensorView out_rect_tile_space_5, TensorView out_radii_5, DiffTensorView_0 out_xyz_vs_5, DiffTensorView_0 out_inv_cov_vs_5, DiffTensorView_0 out_rgb_5, float fovy_8, float fovx_8, uint image_height_5, uint image_width_5, uint grid_height_5, uint grid_width_5, uint tile_height_5, uint tile_width_5)
{

#line 77
    uint g_idx_11 = ((blockIdx)).x * ((blockDim)).x + ((threadIdx)).x;

    if(g_idx_11 >= DiffTensorView_size_0(xyz_ws_12, 0U))
    {

#line 80
        return;
    }
    Camera_0 cam_6 = load_camera_0(world_view_transform_8, proj_mat_8, cam_pos_6, fovy_8, fovx_8, image_height_5, image_width_5);

    Splat_2D_Vertex_0 splat_0 = project_gaussian_to_camera_0(load_gaussian_0(int(g_idx_11), xyz_ws_12, sh_coeffs_14, rotations_10, scales_10, active_sh_21), cam_6, active_sh_21);
    float _S1367 = splat_0.xyz_vs_0.z;

#line 85
    if(_S1367 <= 0.20000000298023224f)
    {

#line 86
        return;
    }

    float det_1 = compute_det_0(splat_0.cov_vs_0);

    if(det_1 == 0.0f)
    {

#line 92
        return;
    }

#line 93
    float radius_2 = splat_radius_0(splat_0.cov_vs_0, det_1);

    Matrix<float, 2, 2>  g_inv_cov_vs_3 = makeMatrix<float, 2, 2> (splat_0.cov_vs_0.rows[int(1)].y, - splat_0.cov_vs_0.rows[int(0)].y, - splat_0.cov_vs_0.rows[int(1)].x, splat_0.cov_vs_0.rows[int(0)].x) / makeMatrix<float, 2, 2> (det_1);

    float _S1368 = splat_0.xyz_vs_0.x;

#line 97
    float _S1369 = splat_0.xyz_vs_0.y;

#line 97
    float2  pixelspace_xy_3 = make_float2 (ndc2pix_0(_S1368, int(image_width_5)), ndc2pix_0(_S1369, int(image_height_5)));



    float3  _S1370 = make_float3 (g_inv_cov_vs_3.rows[int(0)].x, g_inv_cov_vs_3.rows[int(0)].y, g_inv_cov_vs_3.rows[int(1)].y);

#line 101
    float _S1371 = ((opcities_5).load<float>((g_idx_11)));

#line 100
    uint n_tiles_3;
    rectangle_0 rect_tile_space_3 = computeSnugBox_0(_S1370, pixelspace_xy_3, _S1371, make_int2 (int(grid_width_5), int(grid_height_5)), make_int2 (int(tile_width_5), int(tile_height_5)), &n_tiles_3);

    if(n_tiles_3 == 0U)
    {

#line 104
        return;
    }

    (out_radii_5).store<int>((g_idx_11), (int(uint(radius_2))));
    (out_tiles_touched_5).store<int>((g_idx_11), (int(n_tiles_3)));
    (out_rect_tile_space_5).store<int>((g_idx_11), (0U), (rect_tile_space_3.min_x_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (1U), (rect_tile_space_3.min_y_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (2U), (rect_tile_space_3.max_x_0));
    (out_rect_tile_space_5).store<int>((g_idx_11), (3U), (rect_tile_space_3.max_y_0));

    uint2  _S1372 = make_uint2 (g_idx_11, 0U);

#line 114
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1372, _S1368);
    uint2  _S1373 = make_uint2 (g_idx_11, 1U);

#line 115
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1373, _S1369);
    uint2  _S1374 = make_uint2 (g_idx_11, 2U);

#line 116
    DiffTensorView_storeOnce_0(out_xyz_vs_5, _S1374, _S1367);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 0U, 0U), g_inv_cov_vs_3.rows[int(0)].x);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 0U, 1U), g_inv_cov_vs_3.rows[int(0)].y);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 1U, 0U), g_inv_cov_vs_3.rows[int(1)].x);
    DiffTensorView_storeOnce_1(out_inv_cov_vs_5, make_uint3 (g_idx_11, 1U, 1U), g_inv_cov_vs_3.rows[int(1)].y);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1372, splat_0.rgb_0.x);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1373, splat_0.rgb_0.y);
    DiffTensorView_storeOnce_0(out_rgb_5, _S1374, splat_0.rgb_0.z);
    return;
}


//
// Created by paramath on 1/19/24.
//

#ifndef FLASH_LLM_SPTC_TYPE_UTILS_H
#define FLASH_LLM_SPTC_TYPE_UTILS_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>

typedef __half half;
typedef __half2 half2;

struct __align__(8) half4 {
half2 x, y;
};

struct __align__(16) half8 {
half2 x, y, z, w;
};

//struct __align__(8) short4 {
//short2 x, y;
//};

struct __align__(16) short8 {
short2 x, y, z, w;
};


template <typename T>
static __device__ __forceinline__ T* MaybeOffset(T* ptr, int off) {
    return ptr + off;
}

static __device__ __forceinline__ void FMA(float x1, float2 x2, float2 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
}

static __device__ __forceinline__ void FMA(float x1, half2 x2, float2 *out) {
    float2 x2_f2 = __half22float2(x2);
    FMA(x1, x2_f2, out);
}


/**
 * @brief Helper to convert between datatypes.
 */
template <typename To, typename From>
__device__ __forceinline__ void FSP_Convert(const From *in, To *out) {
    // In the default case, don't perform any conversion. Reinterpret.
    *out = *reinterpret_cast<const To *>(in);
}

__device__ __forceinline__ void FSP_Convert(const float *in, half2 *out) {
    // Convert two 32-bit floats into 16-bit floats and pack into
    // a single half2.
    *out = __float22half2_rn(*reinterpret_cast<const float2 *>(in));
}

__device__ __forceinline__ void FSP_Convert(const float *in, half4 *out) {
    // Convert four 32-bit floats into 16-bit floats and pack into
    // a single half4.
    const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
    out->x = __float22half2_rn(in_f2[0]);
    out->y = __float22half2_rn(in_f2[1]);
}

__device__ __forceinline__ void FSP_Convert(const float *in, half8 *out) {
    // Convert 8 32-bit floats into 16-bits floats and pack into
    // a single half8
    const float2 *in_f2 = reinterpret_cast<const float2 *>(in);
    out->x = __float22half2_rn(in_f2[0]);
    out->y = __float22half2_rn(in_f2[1]);
    out->z = __float22half2_rn(in_f2[2]);
    out->w = __float22half2_rn(in_f2[3]);
}

__device__ __forceinline__ void FSP_Convert(const short2 *x, int *out) {
    // Extract two 16-bit integers into 2 32-bit integers. Useful for
    // all variants of the kernels with low precision inputs. To
    // support a wide enough range of input matrix sizes, we need to
    // use 32-bits for all offsets derived from 16-bit indices.
    out[0] = static_cast<int>(x->x);
    out[1] = static_cast<int>(x->y);
}

__device__ __forceinline__ void FSP_Convert(const short4 *x, int *out) {
    FSP_Convert(&x->x, out);
    FSP_Convert(&x->y, out + 2);
}

__device__ __forceinline__ void FSP_Convert(const short2 x, int *out) {
    FSP_Convert(&x, out);
}

__device__ __forceinline__ void FSP_Convert(short4 x, int *out) {
    FSP_Convert(&x.x, out);
    FSP_Convert(&x.y, out + 2);
}

__device__ __forceinline__ void FSP_Convert(const half2 *x, float *out) {
    // Extract two 16-bit IEEE floating-point values into two 32-bit
    // IEEE floating-point values. Useful for pseudo-fp16 kernels.
    float2 tmp = __half22float2(*x);
    out[0] = tmp.x;
    out[1] = tmp.y;
}


template <typename OutType, typename InType>
__device__ __forceinline__ OutType* OffsetCast(InType* ptr, int offset) {
    return reinterpret_cast<OutType*>(
            const_cast<char*>(reinterpret_cast<const char*>(ptr)) + offset);
}

template <class To, class From>
__device__ __forceinline__ To BitCast(const From& src) noexcept {
To dst;
std::memcpy(&dst, &src, sizeof(To));
return dst;
}

template <typename T>
__device__ __forceinline__ T Load(const T* address) {
    return __ldg(address);
}

template <typename T>
__device__ __forceinline__ void Store(const T& value, T* ptr) {
    *ptr = value;
}

__device__ __forceinline__ half4 Load(const half4* address) {
    float2 x = __ldg(reinterpret_cast<const float2*>(address));
    return BitCast<half4>(x);
}

__device__ __forceinline__ short4 Load(const short4* address) {
    int2 x = __ldg(reinterpret_cast<const int2*>(address));
    return BitCast<short4>(x);
}

__device__ __forceinline__ half8 Load(const half8* address) {
    float4 x = __ldg(reinterpret_cast<const float4*>(address));
    return BitCast<half8>(x);
}

__device__ __forceinline__ short8 Load(const short8* address) {
    int4 x = __ldg(reinterpret_cast<const int4*>(address));
    return BitCast<short8>(x);
}


#endif //FLASH_LLM_SPTC_TYPE_UTILS_H

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "helper.h"


template <typename T>
__forceinline__ __device__ float convert2float(T a) {
    return static_cast<float>(a);
}

template <>
__forceinline__ __device__ float convert2float<__device_bfloat16>(__device_bfloat16 a) {
    return __bfloat162float(a);
}

template <typename T>
__forceinline__ __device__ __device_bfloat16 convert2bf16(T a) {
    return static_cast<__device_bfloat16>(a);
}

template <>
__forceinline__ __device__ __device_bfloat16 convert2bf16<float>(float a) {
    return __float2bfloat16(a);
}
template <>
__forceinline__ __device__ __device_bfloat16 convert2bf16<__device_bfloat16>(__device_bfloat16 a) {
    return a;
}


// Convert X to Y
template <typename Y, typename X>
__forceinline__ __host__ __device__ constexpr Y type_convert(X x)
{
    static_assert(!std::is_reference_v<Y> && !std::is_reference_v<X>);

    return static_cast<Y>(x);
}

template <>
__forceinline__ __host__ __device__ constexpr float type_convert<float, __device_bfloat16>(__device_bfloat16 x)
{
    return __bfloat162float(x);
}


template <>
__forceinline__ __host__ __device__ constexpr __device_bfloat16 type_convert<__device_bfloat16, float>(float x)
{
    return __float2bfloat16(x);
}

template <typename T>
__forceinline__ __device__ T add_mul(T a, T b, T c) {
    return (a + b) * c;
}

template<>
__forceinline__ __device__ half add_mul<half>(half a, half b, half c) {
    return __hmul(__hadd(a, b), c);
}


template<>
__forceinline__ __device__ __device_bfloat16 add_mul<__device_bfloat16>(__device_bfloat16 a, __device_bfloat16 b, __device_bfloat16 c) {
#ifdef PADDLE_WITH_HIP
    return convert2bf16(convert2float(a) + convert2float(b) * convert2float(c));
#else
  #if __CUDA_ARCH__ >= 800
    return __hmul(__hadd(a, b), c);
  #else
    return (static_cast<float>(a) + static_cast<float>(b)) * static_cast<float>(c);
  #endif
#endif
}


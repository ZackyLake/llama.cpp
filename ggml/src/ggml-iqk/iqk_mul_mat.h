//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "iqk_config.h"
#ifdef __cplusplus
extern "C" {
#endif

IQK_API bool iqk_mul_mat(long Nx, long Ny, long ne00,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

IQK_API bool iqk_mul_mat_4d(long Nx, long Ny, long ne00,
        long ne02, long ne03, long ne12, long ne13,
        long nb02, long nb03, long nb12, long nb13, long nb2, long nb3,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long stride_C, int ith, int nth);

IQK_API bool iqk_mul_mat_moe(long Nx, long Ny, long ne00, int ne11,
        int typeA, const void * A, long strideA,
        int typeB, const void * B, long strideB,
        float * C, long nb1, long nb2, const void * vrow_mapping, int ith, int nth);

IQK_API bool iqk_moe_fused_up_gate(long Nx, long Ny, long ne00, int ne11, int unary_op,
        int typeA, const void * Aup, const void * Agate, long strideA,
        int typeB, const void * B, long strideB,
        const char * up_b, const char * gate_b,
        float * C, long nb1, long nb2, const void * vrow_mapping, float limit, int ith, int nth);

IQK_API bool iqk_compute_glu(long nc, long nr, int glu_op, float alpha, float limit,
        const float * gate, long gate_stride, const float * up, long up_stride,
        float * dst, long dst_stride, int ith, int nth);

#ifdef __cplusplus
}
#endif

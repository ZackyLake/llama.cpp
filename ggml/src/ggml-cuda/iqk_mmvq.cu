//
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "iqk_mmvq.cuh"
#include "iqk_mmvq_templates.cuh"


static void mul_mat_vec_iq2_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_K, VDR_IQ2_K_Q8_1_MMVQ, vec_dot_iq2_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_K, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_K, VDR_IQ4_K_Q8_1_MMVQ, vec_dot_iq4_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq5_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_K, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq6_k_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ6_K, VDR_IQ6_K_Q8_1_MMVQ, vec_dot_iq6_k_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_kss_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KSS, VDR_IQ4_KSS_Q8_1_MMVQ, vec_dot_iq4_kss_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KS, VDR_IQ2_KS_Q8_1_MMVQ, vec_dot_iq2_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KS, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq3_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KS, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq5_ks_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ5_KS, VDR_IQ5_K_Q8_1_MMVQ, vec_dot_iq5_ks_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_kl_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KL, VDR_IQ3_K_Q8_1_MMVQ, vec_dot_iq2_kl_q8_1>(args, stream);
}

static void mul_mat_vec_iq1_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ1_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq1_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq2_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ2_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq2_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq3_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ3_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq3_kt_q8_1>(args, stream);
}

static void mul_mat_vec_iq4_kt_q8_1_cuda(const mmvq_args & args, cudaStream_t stream) {
    iqk_mul_mat_vec_q_cuda<GGML_TYPE_IQ4_KT, VDR_IQ4_KS_Q8_1_MMVQ, vec_dot_iq4_kt_q8_1>(args, stream);
}

//extern void mul_mat_vec_iq2_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq3_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq4_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq5_k_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq1_s_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq1_m_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq4_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq5_ks_r4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq1_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
//extern void mul_mat_vec_iq2_bn_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);


#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.

void iqk_mul_mat_vec_q(ggml_type type, const mmvq_args & args, cudaStream_t stream) {
    switch (type) {
        //case GGML_TYPE_IQ1_BN:
        //    mul_mat_vec_iq1_bn_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ2_BN:
        //    mul_mat_vec_iq2_bn_q8_1_cuda(args, stream);
        //    break;
        case GGML_TYPE_IQ2_K:
            mul_mat_vec_iq2_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_K:
            mul_mat_vec_iq3_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            mul_mat_vec_iq2_kl_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KS:
            mul_mat_vec_iq3_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_K:
            mul_mat_vec_iq4_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            mul_mat_vec_iq4_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            mul_mat_vec_iq4_kss_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ1_KT:
            mul_mat_vec_iq1_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            mul_mat_vec_iq2_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            mul_mat_vec_iq3_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            mul_mat_vec_iq4_kt_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            mul_mat_vec_iq2_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_K:
            mul_mat_vec_iq5_k_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            mul_mat_vec_iq5_ks_q8_1_cuda(args, stream);
            break;
        case GGML_TYPE_IQ6_K:
            mul_mat_vec_iq6_k_q8_1_cuda(args, stream);
            break;
        //case GGML_TYPE_IQ2_K_R4:
        //    mul_mat_vec_iq2_k_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ3_K_R4:
        //    mul_mat_vec_iq3_k_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ4_K_R4:
        //    mul_mat_vec_iq4_k_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ4_KS_R4:
        //    mul_mat_vec_iq4_ks_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ5_K_R4:
        //    mul_mat_vec_iq5_k_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ5_KS_R4:
        //    mul_mat_vec_iq5_ks_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ1_S_R4:
        //    mul_mat_vec_iq1_s_r4_q8_1_cuda(args, stream);
        //    break;
        //case GGML_TYPE_IQ1_M_R4:
        //    mul_mat_vec_iq1_m_r4_q8_1_cuda(args, stream);
        //    break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0, const int64_t bias_nb1,
        const char * src0_dd_u, const char * src0_dd_g, const char * src1_ddq_i, float * dst_dd_i, const char * ids_data,
        const void * bias_u, const void * bias_g,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, ggml_unary_op unary_op, float limit, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    mmvq_args args{/* vx_u     */ src0_dd_u,
                   /* vx_g     */ src0_dd_g,
                   /* bias_u   */ bias_u,
                   /* bias_g   */ bias_g,
                   /* vy       */ src1_ddq_i,
                   /* dst      */ dst_dd_i,
                   /* ids_data */ ids_data,
                   /* ncols_x  */ int(ne00),
                   /* nrows_x  */ int(row_diff),
                   /* nrows_y  */ int(src1_padded_row_size),
                   /* ncols_y  */ int(src1_ncols),
                   /* nrows_dst*/ int(nrows_dst),
                   /* ne2      */ int(ne2),
                   /* nb02     */ uint64_t(nb02),
                   /* nb12     */ uint64_t(nb12),
                   /* nb2      */ uint64_t(nb2),
                   /* ids_nb0  */ uint64_t(ids_nb0),
                   /* bias_nb1 */ uint64_t(bias_nb1),
                   /* unary_op */ unary_op,
                   /* limit    */ limit > 1e-6f ? limit : INFINITY
    };

    iqk_mul_mat_vec_q(type, args, stream);
}

void ggml_cuda_iqk_op_mul_mat_vec_q_3D(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src0->ne[2] == src1->ne[2] && src0->ne[2] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    const int64_t src1_row_size = ggml_row_size(GGML_TYPE_Q8_1, src1_padded_row_size);

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1_row_size, dst->nb[2], 0, 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, nullptr, nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_iqk_op_mul_mat_vec_q_biased(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    if (bias) {
        if (bias->ne[0] != ne0) {
            printf("Oops: bias %s is %ld x %ld x %ld x %ld, dst %s is %ld x %ld x %ld x %ld\n",
                    bias->name, bias->ne[0], bias->ne[1], bias->ne[2], bias->ne[3],
                    dst->name, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        }
        GGML_ASSERT(bias->ne[0] == ne0);
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        if (ggml_nrows(bias) != 1) {
            printf("Oops: bias %s is %ld x %ld x %ld x %ld\n", bias->name, bias->ne[0], bias->ne[1], bias->ne[2], bias->ne[3]);
        }
        GGML_ASSERT(ggml_nrows(bias) == 1);
    }

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, 1, 0, 0, 0, 0, 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, nullptr, bias ? bias->data : nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}
void ggml_cuda_iqk_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {
    ggml_cuda_iqk_op_mul_mat_vec_q_biased(ctx, src0, src1, dst, nullptr, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i, row_low, row_high, src1_ncols,
            src1_padded_row_size, stream);
}

void ggml_cuda_iqk_op_mul_mat_vec_q_id(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1);
    GGML_ASSERT(ids->ne[0] == dst->ne[2]);

    const int64_t ne0 = dst->ne[0];

    if (bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(bias->ne[0] == ne0);
        if (ids) {
            //GGML_ASSERT(bias->ne[1] == src0->ne[2]);
            GGML_ASSERT(bias->ne[2] == 1 && bias->ne[3] == 1);
        } else {
            GGML_ASSERT(ggml_nrows(bias) == 1);
        }
    }

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids->nb[0], bias ? bias->nb[1] : 0,
        src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, (const char *)ids->data, bias ? bias->data : nullptr, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, GGML_UNARY_OP_COUNT, 0.0f, stream);

    GGML_UNUSED(src1_ddf_i);
}

void ggml_cuda_iqk_op_fused_mul_mat_vec_q_id(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
    const ggml_tensor * bias_u, const ggml_tensor * bias_g,
    const char * src0_dd_u, const char * src0_dd_g, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, ggml_unary_op unary_op, float limit, cudaStream_t stream) {

    if (!bias_u && !bias_g) {
        GGML_ASSERT(unary_op == GGML_UNARY_OP_SILU ||
                    unary_op == GGML_UNARY_OP_RELU ||
                    unary_op == GGML_UNARY_OP_GELU);
    } else {
        GGML_ASSERT(unary_op == GGML_UNARY_OP_SWIGLU_OAI_IQK);
        GGML_ASSERT(bias_u && bias_g);
        GGML_ASSERT(bias_u->data && bias_g->data);
        GGML_ASSERT(bias_u->nb[1] == bias_g->nb[1]);
        GGML_ASSERT(bias_u->ne[0] == dst->ne[0]);
        GGML_ASSERT(bias_g->ne[0] == dst->ne[0]);
    }
    GGML_ASSERT(src0_dd_u && src0_dd_g);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);
    GGML_ASSERT(src0->ne[3] == 1 && src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->ne[1] == 1 && src1->ne[2] == 1);
    //if (ids && ids->ne[0] != dst->ne[2]) {
    //    printf("%s(%s->%s): unexpected situation\n", __func__, src0->name, dst->name);
    //    printf("  src0 = %ld x %ld x %ld x %ld\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //    printf("  src1 = %ld x %ld x %ld x %ld\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //    printf("   ids = %ld x %ld x %ld x %ld\n", ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3]);
    //    printf("   dst = %ld x %ld x %ld x %ld\n", dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    //    GGML_ABORT("Fatal error");
    //}

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, dst->ne[2],
        src0->nb[2], src1->nb[2], dst->nb[2], ids ? ids->nb[0] : 0, bias_u ? bias_u->nb[1] : 0,
        src0_dd_u, src0_dd_g, src1_ddq_i, dst_dd_i, ids ? (const char *)ids->data : nullptr,
        bias_u ? bias_u->data : nullptr, bias_g ? bias_g->data : nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, unary_op, limit, stream);

    GGML_UNUSED(src1_ddf_i);
}


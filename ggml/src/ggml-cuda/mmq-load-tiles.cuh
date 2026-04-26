#pragma once

#include "vecdotq.cuh"

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q1_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int blocks_per_iter = MMQ_ITER_K / QK1_0;
    constexpr int threads_per_row = blocks_per_iter * QI1_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK1_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI1_0;
    const int kqsx = txi % QI1_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *)(x + i*stride) + kbx0 + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 2;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int q  = qxi[j];

            // unpack crumbs into nibble indices
            const int n0 = __byte_perm(0x11100100, 0x11100100, q >> 0); // [0, 1, 4, 5] [ 8,  9, 12, 13]
            const int n1 = __byte_perm(0x11100100, 0x11100100, q >> 2); // [2, 3, 6, 7] [10, 11, 14, 15]
            // unpack nibbles into byte values
            const int s0 = __byte_perm(0x01FF, 0x01FF, n0 >>  0);
            const int s1 = __byte_perm(0x01FF, 0x01FF, n1 >>  0);
            const int s2 = __byte_perm(0x01FF, 0x01FF, n0 >> 16);
            const int s3 = __byte_perm(0x01FF, 0x01FF, n1 >> 16);
            // unshuffle values
            const int v0 = __byte_perm(s0, s1, 0x5410);
            const int v1 = __byte_perm(s0, s1, 0x7632);
            const int v2 = __byte_perm(s2, s3, 0x5410);
            const int v3 = __byte_perm(s2, s3, 0x7632);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + dst_offset + j*4+0] = v0;
            x_qs[i*sram_stride           + dst_offset + j*4+1] = v1;
            x_qs[i*sram_stride           + dst_offset + j*4+2] = v2;
            x_qs[i*sram_stride           + dst_offset + j*4+3] = v3;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+0] = v0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+1] = v1;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+2] = v2;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*4+3] = v3;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q1_0 * bxi = (const block_q1_0 *)(x + i*stride) + kbx0 + scale_block;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int blocks_per_iter = MMQ_ITER_K / QK2_0;
    constexpr int threads_per_row = blocks_per_iter * QI2_0;
    constexpr int nrows = warp_size / threads_per_row;
    constexpr int scale_entries_per_block = QK2_0 / QK8_1;
    constexpr int scale_entries_per_row = blocks_per_iter * scale_entries_per_block;

    const int txi  = threadIdx.x % threads_per_row;
    const int kbx  = txi / QI2_0;
    const int kqsx = txi % QI2_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *)(x + i*stride) + kbx0 + kbx;
        const int16_t    * qxi = (const int16_t *) bxi->qs + kqsx * 4;

        const int dst_offset = kbx*(scale_entries_per_block*QI8_0) + kqsx*QI8_0;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q  = qxi[j];

            // unpack even and odd crumbs into byte values
            const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
            // unshuffle values
            const int qx = __byte_perm(qe, qo, 0x5140);
            const int qy = __byte_perm(qe, qo, 0x7362);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + dst_offset + j*2+0] = qx;
            x_qs[i*sram_stride           + dst_offset + j*2+1] = qy;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+0] = qx;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + dst_offset + j*2+1] = qy;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    const int ksx = threadIdx.x % scale_entries_per_row;
    const int scale_block = ksx / scale_entries_per_block;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_0 * bxi = (const block_q2_0 *)(x + i*stride) + kbx0 + scale_block;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + ksx] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + ksx] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_0;
    const int kqsx = txi % QI4_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *)(x + i*stride) + kbx0 + kbx;
        const int qs0 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + 0]     = __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
        x_qs[i*sram_stride + kbx*(2*QI4_0) + kqsx + QI4_0] = __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = (const block_q4_0 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                     + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_1(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_1;
    const int kqsx = txi % QI4_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *)(x + i*stride) + kbx0 + kbx;
        const int qs0 = get_int_b4(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI4_1) + kqsx + 0]     = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + kbx*(2*QI4_1) + kqsx + QI4_1] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = (const block_q4_1 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride                     + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_0;
    const int kqsx = txi % QI5_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *)(x + i*stride) + kbx0 + kbx;

        const int ql = get_int_b2(bxi->qs, kqsx);
        const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = (const block_q5_0 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                     + kbxd] = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI5_0) + i/QI5_0 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_1(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_1);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI5_1;
    const int kqsx = txi % QI5_1;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *)(x + i*stride) + kbx0 + kbx;

        const int ql = get_int_b4(bxi->qs, kqsx);
        const int qh = get_int_b4(bxi->qh, 0) >> (4 * kqsx);

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*sram_stride + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + 0]     = qs0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kbx*(2*QI5_1) + kqsx + QI5_1] = qs1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_1;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = (const block_q5_1 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride                     + kbxd] = bxi->dm;
#else
        x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + kbxd] = bxi->dm;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q8_0(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_tile + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    // MMQ_ITER_K / (4 * QR8_0) == 64 required. but NV has only 32 threads per warp
    constexpr int threads_per_row = 32;
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI8_0;
    const int kqsx = txi % QI8_0;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *)(x + i*stride) + kbx0 + kbx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*sram_stride + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 0             + txi] = get_int_b2(bxi[0].qs,                   kqsx);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + MMQ_TILE_NE_K + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = (const block_q8_0 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + kbxd] = bxi->d;
#else
        x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q2_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = (const block_q2_K *)(x + i*stride) + kbx0;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_qs_k = (x_ql_0 >> (2*l)) & 0x03030303;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(sc_m & 0x0F, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x*(sc_m & 0x0F), bxi_dmf.y*(sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i*sram_stride         + kqsx] = x_dm_ik;
#else
        x_dm[i*(MMQ_TILE_NE_K + 1) + kqsx] = x_dm_ik;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q3_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        const int x_ql_0 = get_int_b2(bxi->qs,    kqsx);
        const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));

#pragma unroll
        for (int l = 0; l < QR3_K; ++l) {
            const int k = (kqsx/8)*32 + l*8 + kqsx % 8;

            const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
            const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;

            const int x_qs_k = __vsubss4(x_ql_k | x_qh_k, 0x04040404);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + k] = x_qs_k;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        const int ksc = threadIdx.x % 4;

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_b2(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_b2(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        const int8_t * sc8 = (const int8_t *) &sc;
        const float d = bxi->d;

#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            x_df[i*sram_stride + sizeof(int)*ksc + l] = d*sc8[l];
        }
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = sc;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE))
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = (const block_q3_K *)(x + i*stride) + kbx0;

        x_df[i] = bxi->d;
    }
#endif // !(defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)) || defined(AMD_WMMA_AVAILABLE)
}

static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q4_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + 2*MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;
        const int qs0 = get_int_b4(bxi->qs, txi);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
        x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
#else
        x_qs[i*(MMQ_TILE_NE_K + 1) + txi] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < I) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

    #pragma unroll
            for (int l = 0; l < sizeof(int); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0;

        x_dm[i] = bxi->dm;
    }
    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = (const block_q4_K *)(x + i*stride) + kbx0 + (threadIdx.x % (MMQ_TILE_NE_K/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q5_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_dm = (half2 *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_dm + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;
        const int ky = QR5_K*txi;

        const int ql = get_int_b4(bxi->qs, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kq0] = ql0 | qh0;
        x_qs[i*sram_stride + kq1] = ql1 | qh1;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = ql0 | qh0;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = ql1 | qh1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int rows_per_warp = warp_size / 2;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
#if defined(AMD_MFMA_AVAILABLE)
        // Need if on AMD instead of % because warp_size == 64
        // This causes double work and throughput loss (MI300X)
        // H100 loses about 100 t/s with 'if' condition over '%'
        int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/2;
        if (i < I) {
#else
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
        {
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            if (fallback) {
                i = min(i, i_max);
            }

            const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

            const int * scales = (const int *) bxi->scales;
            const int ksc = threadIdx.x % 2;

            const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
            const int  m32 = unpack_scales_q45_K(scales, ksc + 2);

            const uint8_t * sc8 = (const uint8_t *) &sc32;
            const uint8_t *  m8 = (const uint8_t *)  &m32;

            const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);

#pragma unroll
            for (int l = 0; l < int(sizeof(int)); ++l) {
                x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
            }
        }
    }
#else
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

        x_dm[i] = bxi->dm;
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = (const block_q5_K *)(x + i*stride) + kbx0;

        const int * scales = (const int *) bxi->scales;

        const int ksc = threadIdx.x % (MMQ_TILE_NE_K/8);
        const int scales8 = unpack_scales_q45_K(scales, ksc);

        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + ksc] = scales8;
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_q6_K(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
    int   * x_sc = (int   *) (x_df + MMQ_TILE_NE_K/QI6_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
    int   * x_sc = (int   *) (x_df + txs.dm);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR6_K);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0;

        const int ql = get_int_b2(bxi->ql, txi);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_b2(bxi->qh, (QI6_K/4) * (txi / (QI6_K/2)) + txi % (QI6_K/4));
        const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
        const int qh1 =  (qh >> ((txi & 0x08) >> 2))       & 0x30303030;

        const int kq0 = 2*txi - txi % (QI6_K/2) + 0;
        const int kq1 = 2*txi - txi % (QI6_K/2) + QI6_K/2;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*sram_stride + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*warp_size) {
        int i = (i0 + threadIdx.y*warp_size + threadIdx.x) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride]                     = bxi->d;
#else
        x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 4;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps*rows_per_warp) {
        int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/(MMQ_TILE_NE_K/8)) % I;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = (const block_q6_K *)(x + i*stride) + kbx0 + (threadIdx.x % (MMQ_TILE_NE_K/8)) / 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_sc[i*sram_stride + threadIdx.x%4] = get_int_b2(bxi->scales, threadIdx.x % (MMQ_TILE_NE_K/8));
#else
        x_sc[i*(MMQ_TILE_NE_K/8) + i/8 + threadIdx.x%(MMQ_TILE_NE_K/8)] = get_int_b2(bxi->scales, threadIdx.x%(QI6_K/8));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, I);
    int   * x_qs = (int   *)  x_tile;
    half2 * x_ds = (half2 *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR1_S);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_s * bxi = (const block_iq1_s *)(x + i*stride) + kbx0;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

    #pragma unroll
        for (int l = 0; l < QR1_S/2; ++l) {
            const int grid = iq1s_grid_gpu[qs[l] | (((qh >> (3*l)) & 0x07) << 8)];

            const int grid0 = (grid >> 0) & 0x0F0F0F0F;
            const int grid1 = (grid >> 4) & 0x0F0F0F0F;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const float  d1q   = __half2float(bxi->d) * (((qh >> 11) & 0x0E) + 1);
        const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_ds[i*sram_stride             + kqsx] = make_half2(d1q, d1q*delta);
#else
        x_ds[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = make_half2(d1q, d1q*delta);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XXS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xxs * bxi = (const block_iq2_xxs *)(x + i*stride) + kbx0;

        const int q2 = get_int_b2(bxi->qs, 2*kqsx+0);
        const uint8_t * aux8 = (const uint8_t *) &q2;
        const uint32_t aux32 = get_int_b2(bxi->qs, 2*kqsx+1);

#pragma unroll
        for (int l = 0; l < QR2_XXS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * ls / 8; // (d * scale + d / 2) / 4
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_XS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_xs * bxi = (const block_iq2_xs *)(x + i*stride) + kbx0;

        const int2 q2_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint16_t * q2 = (const uint16_t *) &q2_packed;

    #pragma unroll
        for (int l = 0; l < QR2_XS; ++l) {
            const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t signs = unpack_ksigns(q2[l] >> 9);

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ2_S, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR2_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_s * bxi = (const block_iq2_s *)(x + i*stride) + kbx0;

        const int       qs_packed = get_int_b2(bxi->qs, kqsx);
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->qs, QK_K/32 + kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR2_S; ++l) {
            const int * grid_pos = (const int *)(iq2s_grid + (qs[l] | ((qh << (8-2*l)) & 0x300)));

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = bxi->scales[kqsx];
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*sram_stride                             + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = ((ls &  0x0F)*d + d/2)/4;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = ((ls >>    4)*d + d/2)/4;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_xxs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_XXS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_XXS)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_xxs * bxi = (const block_iq3_xxs *)(x + i*stride) + kbx0;

        const int2 q3_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * q3 = (const uint8_t *) &q3_packed;
        const uint32_t aux32 = get_int_b2(bxi->qs, QK_K/16 + kqsx);

#pragma unroll
        for (int l = 0; l < QR3_XXS; ++l) {
            const int2 grid_pos = make_int2(iq3xxs_grid[q3[2*l+0]], iq3xxs_grid[q3[2*l+1]]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7*l));

            const int signs0 = __vcmpne4(signs & 0x08040201, 0);
            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

            const int signs1 = __vcmpne4(signs & 0x80402010, 0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l + 1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l + 1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = aux32 >> 28;
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = (ls*d + d/2)/2;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = (ls*d + d/2)/2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_s(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ3_S, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = (MMQ_ITER_K / (4 * QR3_S)) / 2;
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * nrows) {
        int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_s * bxi = (const block_iq3_s *)(x + i*stride) + kbx0;

        const int2      qs_packed = make_int2(get_int_b2(bxi->qs, 2*kqsx+0), get_int_b2(bxi->qs, 2*kqsx+1));
        const uint8_t * qs        = (const uint8_t *) &qs_packed;

        const int qh = bxi->qh[kqsx];

        const int       signs_packed_32 = get_int_b2(bxi->signs, kqsx);
        const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

#pragma unroll
        for (int l = 0; l < QR3_S; ++l) {
            const int2 grid_pos = make_int2(
                iq3s_grid[qs[2*l+0] | ((qh << (8 - 2*l)) & 0x100)],
                iq3s_grid[qs[2*l+1] | ((qh << (7 - 2*l)) & 0x100)]);

            const int signs0 = __vcmpne4(((signs_packed_8[l] & 0x03) << 7) | ((signs_packed_8[l] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((signs_packed_8[l] & 0x30) << 3) | ((signs_packed_8[l] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
            const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*sram_stride + 8*kqsx + (2*l+1)] = grid_h;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+0)] = grid_l;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + (2*l+1)] = grid_h;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int ls = 1 + 2*((bxi->scales[kqsx/2] >> (((2*kqsx) << 1) & 0x04)) & 0x0F);
        const float d = bxi->d;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = ls*d;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = ls*d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_xs(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_XS, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_XS);
    constexpr int nrows = warp_size / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *)(x + i*stride) + kbx0;

        const int aux_q4 = get_int_b4(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0] = v.x;
        x_qs[i*sram_stride + k0 + 4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int rows_per_warp = warp_size / 8;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (MMQ_TILE_NE_K/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_xs * bxi = (const block_iq4_xs *)(x + i*stride) + kbx0;

        const float d = __half2float(bxi->d);

        const int ls = ((bxi->scales_l[(threadIdx.x % 8)/2] >> (4*(threadIdx.x % 2))) & 0x0F)
            | (((bxi->scales_h >> (2*(threadIdx.x % 8))) & 0x03) << 4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = d * (ls - 32);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * (ls - 32);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_nl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_IQ4_NL, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_NL);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI4_NL;
    const int kqsx = txi % QI4_NL;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *)(x + i*stride) + kbx0 + kbx;

        const int aux_q4 = get_int_b2(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int k0 = kbx * (2 * QI4_NL) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0]      = v.x;
        x_qs[i*sram_stride + k0 + QI4_NL] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]      = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI4_NL] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI4_NL;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_nl * bxi = (const block_iq4_nl *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                       + kbxd] = __half2float(bxi->d);
#else
        x_df[i*(MMQ_TILE_NE_K/QI4_NL) + i/QI4_NL + kbxd] = __half2float(bxi->d);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_MXFP4, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR_MXFP4);
    constexpr int nrows = warp_size / threads_per_row;
    const int txi = warp_size > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
    const int kbx  = txi / QI_MXFP4;
    const int kqsx = txi % QI_MXFP4;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);

        if (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *)(x + i*stride) + kbx0 + kbx;

        const int aux_q4 = get_int_b1(bxi->qs, kqsx);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4);
        const int k0 = kbx * (2 * QI_MXFP4) + kqsx;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride + k0 + 0]        = v.x;
        x_qs[i*sram_stride + k0 + QI_MXFP4] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + 0]        = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0 + QI_MXFP4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE)  || defined(AMD_WMMA_AVAILABLE)
    }

    constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI_MXFP4;
    constexpr int rows_per_warp = warp_size / blocks_per_tile_x_row;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *)(x + i*stride) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                           + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#else
        x_df[i*(MMQ_TILE_NE_K/QI_MXFP4) + i/QI_MXFP4 + kbxd] = ggml_cuda_e8m0_to_fp32(bxi->e)*0.5f;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_mxfp4_fp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    int *      x_qs = (int *) x_tile;
    uint32_t * x_sc = (uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);

    const int txi = threadIdx.x;

    constexpr int iter_k = ggml_cuda_mmq_get_K_vram(type, J, fallback);

    constexpr int threads_per_row = iter_k / QK_MXFP4;  // each thread processes 1 block
    constexpr int rows_per_warp   = warp_size / threads_per_row;
    const int     kbx             = txi % threads_per_row;
    const int     row_in_warp     = txi / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_mxfp4 * bxi = (const block_mxfp4 *)(x + i*stride) + kbx0 + kbx;

        // quantize_mxfp4_mmq permutes nibbles to match the quantized format
        const int k0 = kbx * 4;
        memcpy(x_qs + i*sram_stride + k0, bxi->qs, 16);

        // Load E8M0 scales: pack 2 consecutive scales into one uint32
        if (kbx % 2 == 0) {
            uint32_t e = bxi->e;
            e |= ((bxi + 1)->e << 8);
            x_sc[i*sram_stride + kbx / 2] = e;
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kb0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_NVFP4, I);
    int   * x_qs = (int   *) x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / QK_NVFP4;
    constexpr int rows_per_warp = warp_size / threads_per_row;
    const int kbx = threadIdx.x % threads_per_row;
    const int row_in_warp = threadIdx.x / threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = (const block_nvfp4 *)(x + i*stride) + kb0 + kbx;
        const uint32_t * __restrict__ src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);
        const int kqs = 16 * kbx;
        const int ksc = 4 * kbx;

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            const int2 q0 = get_int_from_table_16(src_qs[2 * sub + 0], kvalues_mxfp4);
            const int2 q1 = get_int_from_table_16(src_qs[2 * sub + 1], kvalues_mxfp4);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride + kqs + 4 * sub + 0] = q0.x;
            x_qs[i*sram_stride + kqs + 4 * sub + 1] = q1.x;
            x_qs[i*sram_stride + kqs + 4 * sub + 2] = q0.y;
            x_qs[i*sram_stride + kqs + 4 * sub + 3] = q1.y;
            x_df[i*sram_stride + ksc + sub] = ggml_cuda_ue4m3_to_fp32(bxi->d[sub]);
#else
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 0] = q0.x;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 1] = q1.x;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 2] = q0.y;
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + kqs + 4 * sub + 3] = q1.y;
            x_df[i * (2 * MMQ_TILE_NE_K * 2 / QI_NVFP4) + i / (QK_NVFP4_SUB / QI_NVFP4) + ksc + sub] = ggml_cuda_ue4m3_to_fp32(bxi->d[sub]);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_nvfp4_nvfp4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size       = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I               = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int iter_k          = ggml_cuda_mmq_get_K_vram(type, J, fallback);
    constexpr int threads_per_row = iter_k / QK_NVFP4; // each thread processes 1 block
    constexpr int rows_per_warp   = warp_size / threads_per_row;
    constexpr int sram_stride     = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    uint32_t * x_u32 = (uint32_t *) x_tile;

    const int txi = threadIdx.x;
    const int kbx = txi % threads_per_row;
    const int row_in_warp = txi / threads_per_row;

    uint32_t * x_u32_scale = x_u32 + 64 + kbx;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += rows_per_warp * nwarps) {
        int i = i0 + threadIdx.y * rows_per_warp + row_in_warp;

        if constexpr (fallback) {
            i = min(i, i_max);
        }

        const block_nvfp4 * bxi = (const block_nvfp4 *)(x + i*stride) + kbx0 + kbx;

        const uint32_t * src_qs = reinterpret_cast<const uint32_t *>(bxi->qs);

#pragma unroll
        for (int sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
            x_u32[i*sram_stride + 8*kbx + 2 * sub + 0] = src_qs[2 * sub + 0];
            x_u32[i*sram_stride + 8*kbx + 2 * sub + 1] = src_qs[2 * sub + 1];
        }

        x_u32_scale[i*sram_stride] = get_int_b4(bxi->d, 0);
    }
}


// ---------------------------------------------------------------------------------------------

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_k(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_k * bxi = (const block_iq2_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

#ifdef __CUDA_ARCH__

        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {
            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 2) & 0x44444444);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 0) & 0x44444444);
            int2 v1 = get_int_from_table_8(val1, iq2nl_values);
            int2 v2 = get_int_from_table_8(val2, iq2nl_values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride           + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride           + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride           + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#else

        auto all_values = (const int *)iq2k_table;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*sram_stride           + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*sram_stride           + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*sram_stride           + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  0] = int_from_table_4((ql >> 0) & 0x03030303, all_values + ((extra & 0x01) << 8));
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  8] = int_from_table_4((ql >> 2) & 0x03030303, all_values + ((extra & 0x04) << 6));
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 16] = int_from_table_4((ql >> 4) & 0x03030303, all_values + ((extra & 0x10) << 4));
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 24] = int_from_table_4((ql >> 6) & 0x03030303, all_values + ((extra & 0x40) << 2));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

            extra >>= 8;
        }
#endif // __CUDA_ARCH__

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*sram_stride                             + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * (((bxi->scales[kqsx] >> 0) & 0xf) - 8);
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * (((bxi->scales[kqsx] >> 4) & 0xf) - 8);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_k(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_k * bxi = (const block_iq3_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;

        uint16_t extra = bxi->extra >> (kqsx/4);
        uint32_t extra32[2] = { uint32_t(extra & 0xff) * 0x01010101, uint32_t(extra >> 8) * 0x01010101 };
        int qh = get_int_b2(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            //extra << 3, extra << 1, extra >> 1, extra >> 3
            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((extra32[l] << 3) & 0x88888888)
                          | ((qh << 2) & 0x04040404) | ((qh << 4) & 0x40404040);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((extra32[l] << 1) & 0x88888888)
                          | ((qh << 1) & 0x04040404) | ((qh << 3) & 0x40404040);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            qh >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride           + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride           + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride           + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        uint16_t sh = bxi->scales_h >> 2*kqsx;
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*sram_stride                             + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ((2*(bxi->scales_l[kqsx] & 0xf) + 1) * (sh & 1 ? -1 : 1));
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ((2*(bxi->scales_l[kqsx] >>  4) + 1) * (sh & 2 ? -1 : 1));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_k(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_k * bxi = (const block_iq4_k *)(x + i*stride) + kbx0;
        const uint16_t extra = bxi->extra >> 2*kqsx;

        auto values_l = iq4k_table + ((extra & 1) << 8);
        auto values_h = iq4k_table + ((extra & 2) << 7);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int q4 = get_int_b4(bxi->qs, (qstep/2)*kqsx + l);

            aux32[0] = (q4 >> 0) & 0x0f0f0f0f;
            aux32[1] = (q4 >> 4) & 0x0f0f0f0f;

            int val0 = int_from_table_x(aux8+0, values_l);
            int val1 = int_from_table_x(aux8+4, values_h);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + 8*kqsx + l + 0] = val0;
            x_qs[i*sram_stride           + 8*kqsx + l + 4] = val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + l + 0] = val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + l + 4] = val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * ls1;
        x_df[i*sram_stride                             + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq5_k(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq5_k * bxi = (const block_iq5_k *)(x + i*stride) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);
        uint16_t extra = bxi->extra >> (kqsx/4);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) * 0x20202020); // this is very slightly faster
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) * 0x08080808); // then the version below
            //aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((extra & 1) ? 0x20202020 : 0);
            //aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((extra & 4) ? 0x20202020 : 0);
            qh    >>= 2;
            extra >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride           + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const uint8_t sh = bxi->scales_h[kqsx/2] >> 4*(kqsx%2);
        const int ls1 = ((bxi->scales_l[kqsx] & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = ((bxi->scales_l[kqsx] >>  4) | ((sh << 2) & 0x30)) - 32;

        const float d = bxi->d;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * ls1;
        x_df[i*sram_stride                             + 2*kqsx+1] = d * ls2;
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * ls1;
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * ls2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq6_k(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq6nl_values;
    int qh[2];

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq6_k * bxi = (const block_iq6_k *)(x + i*stride) + kbx0;

        const float d = bxi->d;
        uint16_t extra = bxi->extra >> (kqsx/4);

        qh[0] = get_int_b4(bxi->qh, kqsx+0);
        qh[1] = get_int_b4(bxi->qh, kqsx+8);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh[l/2] & 0x03030303) << 4) | ((extra & 1) * 0x40404040);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh[l/2] & 0x0c0c0c0c) << 2) | ((extra & 4) * 0x10101010);
            qh[l/2] >>= 4;
            extra   >>= 4;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride           + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 8] = *(const int *)&val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }


#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*sram_stride                             + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+0] = d * bxi->scales[2*kqsx+0];
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + 2*kqsx+1] = d * bxi->scales[2*kqsx+1];
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_kss(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kss * bxi = (const block_iq4_kss *)(dptr + 1) + kbx0;
        const uint32_t * q4 = bxi->qs + 4*kqsx;
        uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
        uint8_t ls = (s32 | (s32 >> 15)) & 0xff;

        auto values = iq4k_values + ((ls & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t val = q4[j] & 0xfffefffe;
            val = val ^ (val >> 1);
            auto v = get_int_from_table_16(val, values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + 8*kqsx + j + 0] = v.x;
            x_qs[i*sram_stride           + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + j + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = dptr[0] * ((ls & 254) - 127);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = dptr[0] * ((ls & 254) - 127);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_ks(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x%16;

#ifdef __CUDA_ARCH__
    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

        uint32_t extra32 = uint32_t(extra & 0xf) * 0x01010101;
        uint32_t val1 = ((q2 >> 0) & 0x33333333) | ((extra32 << 2) & 0x04040404) | ((extra32 << 4) & 0x40404040);
        uint32_t val2 = ((q2 >> 2) & 0x33333333) | ((extra32 << 1) & 0x04040404) | ((extra32 << 3) & 0x40404040);
        int2 v1 = get_int_from_table_8(val1, iq2nl_values);
        int2 v2 = get_int_from_table_8(val2, iq2nl_values);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) +  0] = v1.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) +  8] = v2.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) + 16] = v1.y;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#else // __CUDA_ARCH__

    const int * all_values = (const int *)iq2k_table;
    #pragma unroll
    for (int i0 = 0; i0 < I; i0 += 2*nwarps) {
        int i = i0 + 2*threadIdx.y + threadIdx.x/16;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_ks * bxi = (const block_iq2_ks *)(x + i*stride + sizeof(half)) + kbx0;

        uint16_t extra = bxi->extra >> 4*(kqsx/8);
        int q2 = get_int_b2(bxi->qs, kqsx);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*sram_stride           + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) +  0] = int_from_table_4((q2 >> 0) & 0x03030303, all_values + ((extra & 1) << 8));
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) +  8] = int_from_table_4((q2 >> 2) & 0x03030303, all_values + ((extra & 2) << 7));
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) + 16] = int_from_table_4((q2 >> 4) & 0x03030303, all_values + ((extra & 4) << 6));
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx%8 + 32*(kqsx/8) + 24] = int_from_table_4((q2 >> 6) & 0x03030303, all_values + ((extra & 8) << 5));
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
#endif // __CUDA_ARCH__

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 8) {
        int i = i0 + threadIdx.y * 8 + threadIdx.x / 4;

        if (fallback) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = dptr[0];
        const block_iq2_ks * bxi = (const block_iq2_ks *)(dptr + 1) + kbx0;
        const int ls1 = ((bxi->scales[threadIdx.x % 4] >> 0) & 0xf) | ((bxi->extra >> (4 + 2*(threadIdx.x % 4))) & 0x10);
        const int ls2 = ((bxi->scales[threadIdx.x % 4] >> 4) & 0xf) | ((bxi->extra >> (5 + 2*(threadIdx.x % 4))) & 0x10);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride           + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*sram_stride           + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + 2*(threadIdx.x % 4) + 0] = d * (ls1 - 16);
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + 2*(threadIdx.x % 4) + 1] = d * (ls2 - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_ks(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = __half2float(dptr[0]);
        const block_iq3_ks * bxi = (const block_iq3_ks *)(dptr + 1) + kbx0;

        //uint16_t extra = bxi->extra >> 8;
        int qh = get_int_b2(bxi->qh, kqsx);

        uint32_t extra32 = uint32_t(bxi->extra >> 8) * 0x01010101;

        #pragma unroll
        for (int l = 0; l < qstep/4; ++l) {

            const int ql = get_int_b2(bxi->qs, kqsx + qstep*l);
            uint32_t val1 = ((ql >> 0) & 0x33333333) | ((qh << 2) & 0x04040404) | ((extra32 << 3) & 0x08080808)
                                                     | ((qh << 4) & 0x40404040) | ((extra32 << 5) & 0x80808080);
            uint32_t val2 = ((ql >> 2) & 0x33333333) | ((qh << 1) & 0x04040404) | ((extra32 << 2) & 0x08080808)
                                                     | ((qh << 3) & 0x40404040) | ((extra32 << 4) & 0x80808080);
            int2 v1 = get_int_from_table_16(val1, iq3nl_values);
            int2 v2 = get_int_from_table_16(val2, iq3nl_values);

            extra32 >>= 4;
            qh      >>= 4;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 32*l +  0] = v1.x;
            x_qs[i*sram_stride           + kqsx + 32*l +  8] = v2.x;
            x_qs[i*sram_stride           + kqsx + 32*l + 16] = v1.y;
            x_qs[i*sram_stride           + kqsx + 32*l + 24] = v2.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  0] = v1.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l +  8] = v2.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 16] = v1.y;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 32*l + 24] = v2.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * (int(((bxi->scales[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->extra >> kqsx) & 1) << 4)) - 16);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_ks(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x / 4;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_ks * bxi = (const block_iq4_ks *)(dptr + 1) + kbx0;
        const int ls = (bxi->scales[kqsx] & 254) - 127;

        auto values = iq4k_values + ((bxi->scales[kqsx] & 1) << 4);

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q4 = get_int_b4(bxi->qs, 4*kqsx+j);
            const int2 v = get_int_from_table_16(q4, values);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + 8*kqsx + j + 0] = v.x;
            x_qs[i*sram_stride           + 8*kqsx + j + 4] = v.y;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + j + 0] = v.x;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*kqsx + j + 4] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = dptr[0] * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = dptr[0] * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq5_ks(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int qstep = 8;
    const int kqsx = threadIdx.x % qstep;

    auto values = iq5nl_values;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;
#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * WARP_SIZE/qstep) {
        int i = i0 + threadIdx.y*(WARP_SIZE/qstep) + threadIdx.x/qstep;

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0];
        const block_iq5_ks * bxi = (const block_iq5_ks *)(dptr + 1) + kbx0;

        int qh = get_int_b4(bxi->qh, kqsx);

    #pragma unroll
        for (int l = 0; l < qstep/2; ++l) {

            const int ql = get_int_b4(bxi->qs, kqsx + qstep*l);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh & 0x01010101) << 4) | ((bxi->scales[2*l+0] & 1) * 0x20202020);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh & 0x02020202) << 3) | ((bxi->scales[2*l+1] & 1) * 0x20202020);
            qh    >>= 2;

            const char4 val0  = make_char4(values[aux8[0]], values[aux8[1]], values[aux8[2]], values[aux8[3]]);
            const char4 val1  = make_char4(values[aux8[4]], values[aux8[5]], values[aux8[6]], values[aux8[7]]);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i*sram_stride           + kqsx + 16*l + 0] = *(const int *)&val0;
            x_qs[i*sram_stride           + kqsx + 16*l + 8] = *(const int *)&val1;
#else
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 0] = *(const int *) &val0;
            x_qs[i*(2*MMQ_TILE_NE_K + 1) + kqsx + 16*l + 8] = *(const int *) &val1;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride                             + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#else
        x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + kqsx] = d * ((bxi->scales[kqsx] & 254) - 127);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_kl(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x/4;

    uint32_t aux32[2];
    const uint8_t * a8 = (const uint8_t *)aux32;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += 4*nwarps) {
        int i = i0 + 4*threadIdx.y + threadIdx.x%4;

        if (fallback) {
            i = min(i, i_max);
        }

        const half * dptr = (const half *)(x + i*stride);
        const float d = *dptr;
        const block_iq2_kl * bxi = (const block_iq2_kl *)(dptr + 1) + kbx0;

        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            auto ql = get_int_b2(bxi->qs, 4*(kqsx/2) + 2*(kqsx%2) + j);
            auto qh = get_int_b2(bxi->qh, 2*(kqsx%2) + j) >> 2*(kqsx/2);
            aux32[0] = ((ql >> 0) & 0x0f0f0f0f) | ((qh << 4) & 0x10101010);
            aux32[1] = ((ql >> 4) & 0x0f0f0f0f) | ((qh << 3) & 0x10101010);
            #pragma unroll
            for (int l = 0; l < 2; ++l) {
                int val1 = iq2kl_values[a8[2*l+0]] | (iq2kl_values[a8[2*l+1]] << 16);
                int val2 = iq2kl_values[a8[2*l+4]] | (iq2kl_values[a8[2*l+5]] << 16);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
                x_qs[i*sram_stride           + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*sram_stride           + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#else
                x_qs[i*(2*MMQ_TILE_NE_K + 1) + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 0] = val1;
                x_qs[i*(2*MMQ_TILE_NE_K + 1) + 16*(kqsx/2) + 4*(kqsx%2) + 2*j + l + 8] = val2;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            }
        }

        int ls = int(((bxi->scales_l[kqsx%4] >> 4*(kqsx/4)) & 0xf) | (((bxi->scales_h >> 2*kqsx) & 3) << 4)) - 32;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kqsx] = d * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kqsx] = d * ls;
#endif
    }

}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_kt(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq1_kt * bxi = (const block_iq1_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        uint32_t val = bxi->ql[kqsx] + ((bxi->qh[kqsx%16] << (8 - 4*(kqsx/16))) & 0xf00) + ((bxi->sh[kqsx/4] << (8 - (kqsx%4))) & 0x1000) + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0];
        const block_iq1_kt * bxi = (const block_iq1_kt *)(dptr + 1) + kbx0;
        const int ls = iq4k_values[bxi->sh[threadIdx.x % 8] & 0xf];

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq2_kt(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq2_kt * bxi = (const block_iq2_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= (ggml_cuda_dp4a(val & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.05f;
        const block_iq2_kt * bxi = (const block_iq2_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = iq4k_values[(bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf];

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq3_kt(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq3_kt * bxi = (const block_iq3_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto ql = (const uint16_t *)bxi->ql;
        const auto qh = (const uint32_t *)bxi->qh;
        uint32_t mask = 0x01010101 << ib32;
        uint32_t val = ql[4*ib32+j] + 4096;
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.x |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        auto signs = __vcmpne4(qh[2*j+0] & mask, 0);
        v.x = __vsub4(v.x ^ signs, signs);
        for (int k = 0; k < 4; ++k) {
            val *= ka;
            v.y |= std::abs(ggml_cuda_dp4a(val & km, 0x01010101, -126)) << 8*k;
        }
        signs = __vcmpne4(qh[2*j+1] & mask, 0);
        v.y = __vsub4(v.y ^ signs, signs);
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const float d = dptr[0] * 1.01f;
        const block_iq3_kt * bxi = (const block_iq3_kt *)(dptr + 1) + kbx0;
        int ib32 = threadIdx.x % 8;
        const int ls = (bxi->scales[ib32%4] >> 4*(ib32/4)) & 0xf;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = d * ls;
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = d * ls;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq4_kt(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kqsx = threadIdx.x;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_iq4_kt * bxi = (const block_iq4_kt *)(x + i*stride + sizeof(float)) + kbx0;

        int ib32 = kqsx/4;
        int j    = kqsx%4;
        const auto shb = bxi->qs;
        const auto ql  = (const uint8_t *)(shb + 8);
        const auto qh  = ql + 64;
        const uint32_t sh = shb[ib32] >> (8 + 6*j);
        uint32_t offset = 4096 + ((shb[ib32] & 1) << 15);
        uint32_t val1 = offset + ql[8*ib32+2*j+0] + ((qh[8*(ib32%4)+2*j+0] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 7) << 12);
        uint32_t val2 = offset + ql[8*ib32+2*j+1] + ((qh[8*(ib32%4)+2*j+1] << (8 - 4*(ib32/4))) & 0xf00) + ((sh & 56) << 9);
        int2 v = {0, 0};
        for (int k = 0; k < 4; ++k) {
            val1 *= ka;
            val2 *= ka;
            v.x |= (ggml_cuda_dp4a(val1 & km, 0x01010101, -126) & 0xff) << 8*k;
            v.y |= (ggml_cuda_dp4a(val2 & km, 0x01010101, -126) & 0xff) << 8*k;
        }
#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*sram_stride           + 8*ib32 + 2*j + 1] = v.y;
#else
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 0] = v.x;
        x_qs[i*(2*MMQ_TILE_NE_K + 1) + 8*ib32 + 2*j + 1] = v.y;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / (WARP_SIZE/4);

        if (fallback) {
            i = min(i, i_max);
        }

        const float * dptr = (const float *)(x + i*stride);
        const block_iq4_kt * bxi = (const block_iq4_kt *)(dptr + 1) + kbx0;
        const int ls = (bxi->qs[threadIdx.x % 8] & 0xff) >> 1;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + threadIdx.x % 8] = dptr[0] * (ls - 64);
#else
        x_df[i*(MMQ_TILE_NE_K/4) + i/4 + threadIdx.x % 8] = dptr[0] * (ls - 64);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_load_tiles_iq1_s_r4(
        const char * __restrict__ x, int * __restrict__ x_tile, const int kbx0, const int i_max, const int stride) {
    constexpr int warp_size   = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps      = ggml_cuda_mmq_get_nthreads(type, J, fallback) / WARP_SIZE;
    constexpr int I           = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + MMQ_TILE_NE_K*2);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    int   * x_qs = (int   *)  x_tile;
    float * x_df = (float *) (x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    const int kbx  = threadIdx.x / 4;
    const int kqsx = threadIdx.x % 4;

    int32_t grid32[2];

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps) {
        int i = i0 + threadIdx.y;

        if (fallback) {
            i = min(i, i_max);
        }
        const int i4 = i/4;
        const int ir = i%4;

        const block_iq1_s_r4 * bxi = (const block_iq1_s_r4 *)(x + 4*i4*stride + 4*sizeof(half)) + kbx0 + kbx;

        grid32[0] = iq1s_grid_gpu[bxi->qs[4*kqsx+ir] | (((bxi->qh[ir] >> 3*kqsx) & 7) << 8)];
        grid32[1] = ((grid32[0] >> 4) & 0x0f0f0f0f) << 3;
        grid32[0] = (grid32[0] & 0x0f0f0f0f) << 3;
        const int shift = bxi->qh[ir] & 0x8000 ? 0x09090909 : 0x07070707;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_qs[i*sram_stride           + 8*kbx + 2*kqsx + 0] = __vsubss4(grid32[0], shift);
        x_qs[i*sram_stride           + 8*kbx + 2*kqsx + 1] = __vsubss4(grid32[1], shift);
#else
        // TODO
        //x_qs[i*(2*MMQ_TILE_NE_K + 1) + threadIdx.x] = qs0;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

    const int blocks_per_tile_x_row = WARP_SIZE / 4;
    const int kbxd = threadIdx.x % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nwarps * 4) {
        int i = i0 + threadIdx.y * 4 + threadIdx.x / blocks_per_tile_x_row;

        if (fallback) {
            i = min(i, i_max);
        }
        const int i4 = i/4;
        const int ir = i%4;

        const half * dptr = (const half *)(x + 4*i4*stride);
        const block_iq1_s_r4 * bxi = (const block_iq1_s_r4 *)(dptr + 4) + kbx0 + kbxd;

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_df[i*sram_stride             + kbxd] = 0.125f * __half2float(dptr[ir]) * (((bxi->qh[ir] >> 11) & 14) + 1);
#else
        // TODO
        //x_df[i*(MMQ_TILE_NE_K/4) + i/4 + kbxd] = bxi->d;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }
}

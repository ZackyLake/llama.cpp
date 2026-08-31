#include "common.cuh"
#include "convert.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q2_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q2_0 * x = (const block_q2_0 *) vx;

    const float d = x[ib].d;

    // Q2_0: 2 bits per element, 4 elements per byte.
    // Stored code c in {0,1,2,3} maps to symbol s = c - 1 in {-1, 0, +1, +2}.
    const int byte_index_0 = iqs / 4;
    const int bit_offset_0 = (iqs % 4) * 2;

    const int byte_index_1 = (iqs + 1) / 4;
    const int bit_offset_1 = ((iqs + 1) % 4) * 2;

    const int c0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 0x3;
    const int c1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 0x3;

    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    dst_t * y = yy + 128*n;

    float dall = __low2half(x[ib].dm);
    float dmin = __high2half(x[ib].dm);
    y[l+ 0] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = ggml_cuda_cast<dst_t>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[ib].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = ggml_cuda_cast<dst_t>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + 64*il + n*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = ggml_cuda_cast<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = ggml_cuda_cast<dst_t>(d2 * (q[l] >>  4) - m2);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + 64*il + 2*ir;

    const float dall = __low2half(x[ib].dm);
    const float dmin = __high2half(x[ib].dm);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = ggml_cuda_cast<dst_t>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = ggml_cuda_cast<dst_t>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, dst_t * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + 128*ip + il;

    const float d = x[ib].d;

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = ggml_cuda_cast<dst_t>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = ggml_cuda_cast<dst_t>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = ggml_cuda_cast<dst_t>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = ggml_cuda_cast<dst_t>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[ibs].d * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = ggml_cuda_cast<dst_t>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = ggml_cuda_cast<dst_t>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[ibs].d * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[ibs].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = ggml_cuda_cast<dst_t>(d * (q[j] + delta));
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_s_r4(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t nblock = n_per_row/32;
    int64_t row  = (8*ii)/nblock;
    int64_t row4 = row/4;
    int64_t ir   = row%4;
    int64_t ibl  = (8*ii)%nblock;

    const int  il = tid/8; // 0...3
    const int  ib = tid%8; // 0...7

    const half * dptr = (const half *)((const char *)vx + 4*row4*row_size);
    const float d = __half2float(dptr[ir]);
    const block_iq1_s_r4 * x = (const block_iq1_s_r4 *)(dptr + 4) + ibl;
    dst_t * y = yy + 32*ib + 8*il;

    const float dl = d*(2*((x[ib].qh[ir] >> 12) & 7) + 1);
    const float delta = dl * (x[ib].qh[ir] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA);

    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ib].qs[4*il+ir] | (((x[ib].qh[ir] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;

    for (int j = 0; j < 8; ++j) y[j] = ggml_cuda_cast<dst_t>(dl*q[j] + delta);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_m_r4(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t nblock = n_per_row/32;
    int64_t row  = (8*ii)/nblock;
    int64_t row4 = row/4;
    int64_t ir   = row%4;
    int64_t ibl  = (8*ii)%nblock;

    const int  il = tid/8; // 0...3
    const int  ib = tid%8; // 0...7

    const half * dptr = (const half *)((const char *)vx + 4*row4*row_size);
    const float d = __half2float(dptr[ir]);
    const block_iq1_m_r4 * x = (const block_iq1_m_r4 *)(dptr + 4) + ibl;
    dst_t * y = yy + 32*ib + 8*il;

    const uint8_t qh = x[ib].qh[4*(il/2)+ir] >> 4*(il%2);
    const float dl = d*((x[ib].scales[ir] >> 4*(il/2)) & 0xf);
    const float delta = dl * (qh & 0x8 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA);

    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ib].qs[4*il+ir] | ((qh & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;

    for (int j = 0; j < 8; ++j) y[j] = ggml_cuda_cast<dst_t>(dl*q[j] + delta);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = (float)x[ibs].d * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_mxfp4(const void * vx, const int64_t ibs, dst_t * yy, const int tid) {

    const block_mxfp4 * x = (const block_mxfp4 *) vx + ibs*(QK_K/QK_MXFP4);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] & 0xf]*0.5f);
        y[j+16] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q4[j] >>  4]*0.5f);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_k(const void * vx, const int64_t i, dst_t * yy, const int tid) {

    const block_iq2_k * x = (const block_iq2_k *) vx;

    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales[4*ib128+0] >> 4*(il/8)) & 0xf) - 8);
    const float dl2 = d * (((x[i].scales[4*ib128+1] >> 4*(il/8)) & 0xf) - 8);
    const float dl3 = d * (((x[i].scales[4*ib128+2] >> 4*(il/8)) & 0xf) - 8);
    const float dl4 = d * (((x[i].scales[4*ib128+3] >> 4*(il/8)) & 0xf) - 8);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 0) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 2) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 4) & 4)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_k(const void * vx, const int64_t i, dst_t * yy, const int tid) {

    const block_iq3_k * x = (const block_iq3_k *) vx;

    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + 128*ib128 + 2*il;
    const float d = (float)x[i].d;
    const uint16_t sh = x[i].scales_h >> (8*ib128 + (il/8));
    const float dl1 = d * ((2*((x[i].scales_l[4*ib128+0] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x01) ? -1 : 1));
    const float dl2 = d * ((2*((x[i].scales_l[4*ib128+1] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x04) ? -1 : 1));
    const float dl3 = d * ((2*((x[i].scales_l[4*ib128+2] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x10) ? -1 : 1));
    const float dl4 = d * ((2*((x[i].scales_l[4*ib128+3] >> 4*(il/8)) & 0xf) + 1) * ((sh & 0x40) ? -1 : 1));
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const int16_t extra = x[i].extra >> (8*ib128 + (il/8));
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = __float2bfloat16(dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)]);
            y[j+32] = __float2bfloat16(dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)]);
            y[j+64] = __float2bfloat16(dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)]);
            y[j+96] = __float2bfloat16(dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h = qh[j] >> (4*(ib128%2));
            y[j+ 0] = dl1 * iq3nl_values[(((qs[j] >> 0) & 0x03) | ((h & 0x01) << 2)) + ((extra << 3) & 8)];
            y[j+32] = dl2 * iq3nl_values[(((qs[j] >> 2) & 0x03) | ((h & 0x02) << 1)) + ((extra << 1) & 8)];
            y[j+64] = dl3 * iq3nl_values[(((qs[j] >> 4) & 0x03) | ((h & 0x04) >> 0)) + ((extra >> 1) & 8)];
            y[j+96] = dl4 * iq3nl_values[(((qs[j] >> 6) & 0x03) | ((h & 0x08) >> 1)) + ((extra >> 3) & 8)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_k(const void * vx, const int64_t i, dst_t * yy, const int tid) {
    const block_iq4_k * x = (const block_iq4_k *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d;
    const uint8_t sh = x[i].scales_h[ib/2] >> 4*(ib%2);
    const float d1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
    const int8_t * values1 = iq4k_values + 16*((x[i].extra >> (2*ib+0)) & 1);
    const int8_t * values2 = iq4k_values + 16*((x[i].extra >> (2*ib+1)) & 1);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d1 * values1[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d2 * values2[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d1 * values1[q4[j] & 0xf];
            y[j+16] = d2 * values2[q4[j] >>  4];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq5_k(const void * vx, const int64_t i, dst_t * yy, const int tid) {

    const block_iq5_k * x = (const block_iq5_k *) vx;

    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * (((x[i].scales_l[2*ib64+0] & 0xf) | ((x[i].scales_h[ib64] << 4) & 0x30)) - 32);
    const float dl2 = d * (((x[i].scales_l[2*ib64+0] >>  4) | ((x[i].scales_h[ib64] << 2) & 0x30)) - 32);
    const float dl3 = d * (((x[i].scales_l[2*ib64+1] & 0xf) | ((x[i].scales_h[ib64] >> 0) & 0x30)) - 32);
    const float dl4 = d * (((x[i].scales_l[2*ib64+1] >>  4) | ((x[i].scales_h[ib64] >> 2) & 0x30)) - 32);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)]);
            y[j+16] = __float2bfloat16(dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)]);
            y[j+32] = __float2bfloat16(dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)]);
            y[j+48] = __float2bfloat16(dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * iq5nl_values[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4) | ((extra << 5) & 0x20)];
            y[j+16] = dl2 * iq5nl_values[(qs[j+16] & 0xf) | ((h2 & 1) << 4) | ((extra << 4) & 0x20)];
            y[j+32] = dl3 * iq5nl_values[(qs[j+ 0] >>  4) | ((h1 & 2) << 3) | ((extra << 3) & 0x20)];
            y[j+48] = dl4 * iq5nl_values[(qs[j+16] >>  4) | ((h2 & 2) << 3) | ((extra << 2) & 0x20)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq6_k(const void * vx, const int64_t i, dst_t * yy, const int tid) {

    const block_iq6_k * x = (const block_iq6_k *) vx;

    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + 64*ib64 + 2*il;
    const float d = (float)x[i].d;
    const float dl1 = d * x[i].scales[4*ib64+0];
    const float dl2 = d * x[i].scales[4*ib64+1];
    const float dl3 = d * x[i].scales[4*ib64+2];
    const float dl4 = d * x[i].scales[4*ib64+3];
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 32*(ib64/2) + 2*il;
    const uint8_t extra = x[i].extra >> 4*(ib64%4);
    for (int j = 0; j < 2; ++j) {
        const uint8_t h1 = qh[j] >> 4*(ib64%2), h2 = qh[j+16] >> 4*(ib64%2);
        uint8_t q1 = (qs[j+ 0] & 0xf) | ((h1 & 0x03) << 4);
        uint8_t q2 = (qs[j+16] & 0xf) | ((h2 & 0x03) << 4);
        uint8_t q3 = (qs[j+ 0] >>  4) | ((h1 & 0x0c) << 2);
        uint8_t q4 = (qs[j+16] >>  4) | ((h2 & 0x0c) << 2);
        if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
            y[j+ 0] = __float2bfloat16(dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0)));
            y[j+16] = __float2bfloat16(dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0)));
            y[j+32] = __float2bfloat16(dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0)));
            y[j+48] = __float2bfloat16(dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0)));
        } else {
            y[j+ 0] = dl1 * (iq6nl_values[q1] + (extra & 1 ? 1 : 0));
            y[j+16] = dl2 * (iq6nl_values[q2] + (extra & 2 ? 1 : 0));
            y[j+32] = dl3 * (iq6nl_values[q3] + (extra & 4 ? 1 : 0));
            y[j+48] = dl4 * (iq6nl_values[q4] + (extra & 8 ? 1 : 0));
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_kss(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq4_kss * x = (const block_iq4_kss *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint32_t * q4 = x[i].qs + 4*ib;
    uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2) | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    uint8_t ls = (s32 | (s32 >> 15)) & 0xff;
    const float d = scale * ((ls & 254) - 127);
    const int8_t * values = iq4k_values + ((ls & 1) << 4);
    uint32_t aux32[2];
    aux32[0] = q4[il] & 0xfffefffe;
    aux32[0] ^= (aux32[0] >> 1);
    aux32[1] = ((aux32[0] >> 4) & 0x0f0f0f0f);
    aux32[0] &= 0x0f0f0f0f;
    const uint8_t * aux8 = (const uint8_t *)aux32;
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d * values[aux8[j+0]]);
            y[j+16] = __float2bfloat16(d * values[aux8[j+4]]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d * values[aux8[j+0]];
            y[j+16] = d * values[aux8[j+4]];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_ks(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float d = (float)*(const half *)cx;
    const block_iq2_ks * x = (const block_iq2_ks *)(cx + sizeof(half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    int ib128 = tid/16; // 0 or 1
    int il    = tid%16; // 0...15
    dst_t * y = yy + 128*ib128 + 2*il;
    const int16_t extra = x[i].extra >> 4*ib128;
    const float dl1 = d * (((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
    const float dl2 = d * (((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
    const float dl3 = d * (((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
    const float dl4 = d * (((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = __float2bfloat16(dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
            y[j+32] = __float2bfloat16(dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)]);
            y[j+64] = __float2bfloat16(dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)]);
            y[j+96] = __float2bfloat16(dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            y[j+ 0] = dl1 * iq2nl_values[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)];
            y[j+32] = dl2 * iq2nl_values[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)];
            y[j+64] = dl3 * iq2nl_values[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)];
            y[j+96] = dl4 * iq2nl_values[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_ks(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const ggml_half *)cx;
    const block_iq3_ks * x = (const block_iq3_ks *)(cx + sizeof(ggml_half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t is = tid/16;
    const int64_t il = tid%16;
    dst_t * y = yy + 128*is + 2*il;
    const uint8_t  * qs = x[i].qs + 32*is + 2*il;
    const uint8_t  * qh = x[i].qh + 2*il;
    uint16_t extra = x[i].extra >> 4*is;
    const float d0 = scale * (int(((x[i].scales[0] >> 4*is) & 0xf) | ((extra << 4) & 0x10)) - 16);
    const float d1 = scale * (int(((x[i].scales[1] >> 4*is) & 0xf) | ((extra << 3) & 0x10)) - 16);
    const float d2 = scale * (int(((x[i].scales[2] >> 4*is) & 0xf) | ((extra << 2) & 0x10)) - 16);
    const float d3 = scale * (int(((x[i].scales[3] >> 4*is) & 0xf) | ((extra << 1) & 0x10)) - 16);
    extra >>= 8;
    const int8_t * values0 = iq3nl_values + ((extra & 1) << 3);
    const int8_t * values1 = iq3nl_values + ((extra & 2) << 2);
    const int8_t * values2 = iq3nl_values + ((extra & 4) << 1);
    const int8_t * values3 = iq3nl_values + ((extra & 8) << 0);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = __float2bfloat16(d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)]);
            y[j+32] = __float2bfloat16(d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)]);
            y[j+64] = __float2bfloat16(d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)]);
            y[j+96] = __float2bfloat16(d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 4*is;
            y[j+ 0] = d0 * values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)];
            y[j+32] = d1 * values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)];
            y[j+64] = d2 * values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)];
            y[j+96] = d3 * values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_ks(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq4_ks * x = (const block_iq4_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = scale * ((x[i].scales[ib] & 254) - 127);
    const int8_t * values = iq4k_values + ((x[i].scales[ib] & 1) << 4);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = __float2bfloat16(d * values[q4[j] & 0xf]);
            y[j+16] = __float2bfloat16(d * values[q4[j] >>  4]);
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            y[j+ 0] = d * values[q4[j] & 0xf];
            y[j+16] = d * values[q4[j] >>  4];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq5_ks(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float d = *(const float *)cx;
    const block_iq5_ks * x = (const block_iq5_ks *)(cx + sizeof(float));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    int ib64 = tid/8; // 0...3
    int il   = tid%8; // 0...7
    dst_t * y = yy + 64*ib64 + 2*il;
    const float dl1 = d * ((int)(x[i].scales[2*ib64+0] & 254) - 127);
    const float dl2 = d * ((int)(x[i].scales[2*ib64+1] & 254) - 127);
    const uint8_t * qs = x[i].qs + 32*ib64 + 2*il;
    const uint8_t * qh = x[i].qh + 2*il;
    auto values1 = iq5nl_values + ((x[i].scales[2*ib64+0] & 1) << 5);
    auto values2 = iq5nl_values + ((x[i].scales[2*ib64+1] & 1) << 5);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = __float2bfloat16(dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)]);
            y[j+16] = __float2bfloat16(dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)]);
            y[j+32] = __float2bfloat16(dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)]);
            y[j+48] = __float2bfloat16(dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            const uint8_t h1 = qh[j] >> 2*(ib64%4), h2 = qh[j+16] >> 2*(ib64%4);
            y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)];
            y[j+16] = dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)];
            y[j+32] = dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)];
            y[j+48] = dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)];
        }
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_kl(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = (float)*(const ggml_half *)cx;
    const block_iq2_kl * x = (const block_iq2_kl *)(cx + sizeof(ggml_half));
    const int64_t i   = ii - (row*n_per_row)/QK_K;

    const int64_t ib64 = tid/8;
    const int64_t il   = tid%8;
    dst_t * y = yy + 64*ib64 + 4*il;
    const uint8_t  * qs = x[i].qs + 16*ib64 + 2*il;
    const uint8_t  * qh = x[i].qh + 2*il;
    auto sh = x[i].scales_h >> 4*ib64;
    const float d1 = scale * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 4) & 0x30)) - 32);
    const float d2 = scale * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | ((sh << 2) & 0x30)) - 32);
    if constexpr (std::is_same_v<dst_t, nv_bfloat16>) {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 2*ib64;
            auto val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | ((h & 1) << 4)));
            auto val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | ((h & 2) << 3)));
            y[2*j+ 0] = __float2bfloat16(d1 * val1[0]);
            y[2*j+ 1] = __float2bfloat16(d1 * val1[1]);
            y[2*j+32] = __float2bfloat16(d2 * val2[0]);
            y[2*j+33] = __float2bfloat16(d2 * val2[1]);
        }
    } else {
        for (int j = 0; j < 2; ++j) {
            uint8_t h = qh[j] >> 2*ib64;
            auto val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | ((h & 1) << 4)));
            auto val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | ((h & 2) << 3)));
            y[2*j+ 0] = d1 * val1[0];
            y[2*j+ 1] = d1 * val1[1];
            y[2*j+32] = d2 * val2[0];
            y[2*j+33] = d2 * val2[1];
        }
    }
}

float __device__ __forceinline__ trellis_next(uint32_t& val) {
    constexpr uint32_t ka = 89226354;
    constexpr uint32_t kb = 64248484;
    constexpr uint32_t kmask = 0x8fff8fff;
    constexpr uint32_t km32 = 0x3b603b60;
    uint32_t s;
    const half * h = (const half *)&s;
    val = ka*val + kb;
    s = (val & kmask) ^ km32;
    return (float)(h[0]+h[1]);
}

int __device__ __forceinline__ trellis_next_int(uint32_t& val) {
    constexpr uint32_t ka = 0xCBAC1FED;
    val = ka*val;
    return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq1_kt(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq1_kt * x = (const block_iq1_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;

    const int64_t ib = tid; // 0...31
    dst_t * y = yy + 8*ib;
    uint32_t idx = (x[i].ql[ib] | ((x[i].qh[ib%16] << (8 - 4*(ib/16))) & 0xf00) | ((x[i].sh[ib/4] << (8 - (ib%4))) & 0x1000)) + 4096;
    const float dl = scale * iq4k_values[x[i].sh[ib/4] & 0xf];
    for (int j = 0; j < 8; ++j) {
        y[j] = dl * trellis_next_int(idx);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq2_kt(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq2_kt * x = (const block_iq2_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;

    const int64_t ib = tid; // 0...31
    dst_t * y = yy + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * iq4k_values[((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf)]; // fudge factor 1.05f removed
    for (int j = 0; j < 8; ++j) {
        y[j] = dl * trellis_next_int(idx);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq3_kt(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq3_kt * x = (const block_iq3_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;

    const int64_t ib = tid; // 0...31
    dst_t * y = yy + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * ((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf); // fudge factor 1.01f removed
    uint8_t mask = 1 << (ib/4);
    for (int j = 0; j < 8; ++j) {
        y[j] = dl * std::abs(trellis_next_int(idx)) * (x[i].qh[(8*ib+j)%32] & mask ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ __forceinline__ void dequantize_iq4_kt(const void * vx, const int64_t ii, dst_t * yy, const int tid, int64_t n_per_row, int64_t row_size) {

    int64_t row = (QK_K * ii) / n_per_row;
    const float * dptr = (const float *)((const char *)vx + row * row_size);
    float scale = dptr[0] * 1.00f;
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const int64_t i = ii - (row*n_per_row)/QK_K;

    constexpr int kNumGroups = 64;

    const int64_t ib = tid; // 0...31
    dst_t * y = yy + 8*ib;
    const uint32_t * shb = x[i].qs;
    const uint8_t * ql = (const uint8_t *)(shb + 8); //Q::kNblock;
    const uint8_t * qh = ql + kNumGroups;
    const int ib32 = ib/4;
    const int ig = ib%4;
    const int jj = ib32*8 + 2*ig;
    uint32_t offset = shb[ib32] & 1 ? 4096 + 32768 : 4096;
    uint32_t idx1 = ql[jj+0] + ((qh[(jj+0)%(kNumGroups/2)] << (8 - 4*((jj+0)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+0)) & 7) << 12) + offset;
    uint32_t idx2 = ql[jj+1] + ((qh[(jj+1)%(kNumGroups/2)] << (8 - 4*((jj+1)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+3)) & 7) << 12) + offset;
    int ls = ((shb[ib32] & 0xff) >> 1) - 64;
    const float dl = scale * ls;
    for (int j = 0; j < 4; ++j) {
        y[j+0] = dl * trellis_next_int(idx1);
        y[j+4] = dl * trellis_next_int(idx2);
    }
}

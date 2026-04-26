#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu/ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <random>
#include <cstdio>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE inline
#if !defined __x86_64__ && defined _M_X64
#define __x86_64__
#endif
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#include <ammintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
inline int popcount(uint8_t x) { return __popcnt(x); }
inline int popcount(uint16_t x) { return __popcnt(x); }
inline int popcount(uint32_t x) { return __popcnt(x); }
inline int popcount(uint64_t x) { return _mm_popcnt_u64(x); }
#else
constexpr int popcount(uint8_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint16_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint32_t x) { return __builtin_popcount(x); }
constexpr int popcount(uint64_t x) { return __builtin_popcountll(x); }
#endif


#if defined(_MSC_VER) || defined(__AVX2__)

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

static inline float hsum_float_4(__m128 x) {
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
    return _mm_cvtss_f32(x);
}
static inline float hsum_float_8(__m256 x) {
    return hsum_float_4(_mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1)));
}
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
static inline float hmax_f32_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
    return  _mm_cvtss_f32(max4);
}
static inline float hmax_float_8(__m256 x) {
    __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    max4 = _mm_max_ps( max4, _mm_movehl_ps(max4, max4));
    max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4));
    return  _mm_cvtss_f32(max4);
}

static inline __m128 hsum_float_4x4(__m128 * accm) {
    accm[0] = _mm_add_ps(_mm_unpacklo_ps(accm[0], accm[2]), _mm_unpackhi_ps(accm[0], accm[2]));
    accm[1] = _mm_add_ps(_mm_unpacklo_ps(accm[1], accm[3]), _mm_unpackhi_ps(accm[1], accm[3]));
    return _mm_add_ps(_mm_unpacklo_ps(accm[0], accm[1]), _mm_unpackhi_ps(accm[0], accm[1]));
}
static inline __m256 hsum_float_8x8(__m256 * accm) {
    for (int i = 0; i < 4; ++i) {
        accm[i] = _mm256_add_ps(_mm256_permute2f128_ps(accm[i], accm[i + 4], 0x20), _mm256_permute2f128_ps(accm[i], accm[i + 4], 0x31));
        //accm[i] = _mm256_set_m128(_mm_add_ps(_mm256_castps256_ps128(accm[i+4]), _mm256_extractf128_ps(accm[i+4], 1)),
        //                          _mm_add_ps(_mm256_castps256_ps128(accm[i+0]), _mm256_extractf128_ps(accm[i+0], 1)));
    }
    for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i + 2]), _mm256_unpackhi_ps(accm[i], accm[i + 2]));
    return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}
static inline __m256 hsum_float_4x8(__m256 * accm) {
    for (int i = 0; i < 2; ++i) accm[i] = _mm256_add_ps(_mm256_unpacklo_ps(accm[i], accm[i + 2]), _mm256_unpackhi_ps(accm[i], accm[i + 2]));
    return _mm256_add_ps(_mm256_unpacklo_ps(accm[0], accm[1]), _mm256_unpackhi_ps(accm[0], accm[1]));
}

static inline __m128i load_iq4nl_values_128() {
    static const uint8_t kvalues_iq4nl[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    return _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
}

static inline __m256i load_iq4nl_values_256() {
    auto val128 = load_iq4nl_values_128();
    return MM256_SET_M128I(val128, val128);
}

#ifdef HAVE_FANCY_SIMD
static inline __m512i load_iq4nl_values_512() {
    auto val256 = load_iq4nl_values_256();
    return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
}
#endif

static inline __m128i load_iq4k_values_128() {
    return _mm_loadu_si128((const __m128i *)iq4k_values);
}

static inline __m256i load_iq4k_values_256() {
    auto val128 = load_iq4k_values_128();
    return MM256_SET_M128I(val128, val128);
}

#endif


namespace {

inline int best_index_iq2nl(const int8_t * values, float x) {
    int idx = x < values[1] ? 0 : x > values[2] ? 2 : 1;
    return x - values[idx] < values[idx+1] - x ? idx : idx + 1;
}

const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}

const int8_t iq5nl_index[248] = {
     0,  0,  0,  0,  0,  0, 32,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 33, 33,  2,  2,  2,  2,  2,  2,  2,  2,  2, 34, 34,  3,  3,
     3,  3,  3,  3,  3,  3, 35, 35,  4,  4,  4,  4,  4,  4,  4, 36, 36,  5,  5,  5,  5,  5,  5,  5, 37, 37,  6,  6,  6,  6,  6,  6,
     6, 38,  7,  7,  7,  7,  7,  7, 39, 39,  8,  8,  8,  8,  8, 40, 40,  9,  9,  9,  9,  9, 41, 41, 10, 10, 10, 10, 10, 42, 11, 11,
    11, 11, 11, 43, 12, 12, 12, 12, 12, 44, 13, 13, 13, 13, 13, 45, 14, 14, 14, 14, 14, 46, 15, 15, 15, 15, 47, 47, 16, 16, 16, 16,
    48, 17, 17, 17, 17, 17, 49, 18, 18, 18, 18, 18, 50, 19, 19, 19, 19, 19, 51, 20, 20, 20, 20, 20, 52, 21, 21, 21, 21, 21, 53, 53,
    22, 22, 22, 22, 22, 54, 54, 23, 23, 23, 23, 23, 23, 55, 24, 24, 24, 24, 24, 24, 24, 56, 25, 25, 25, 25, 25, 25, 25, 57, 57, 26,
    26, 26, 26, 26, 26, 26, 58, 58, 27, 27, 27, 27, 27, 27, 27, 27, 59, 28, 28, 28, 28, 28, 28, 28, 28, 28, 60, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 61, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 62, 31, 31, 31, 31, 31, 31
};
inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}

}



namespace {

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

typedef void (*quantize_func_t)(const float * src, void * qdata, int n_per_row, const float * imatrix);

struct QHelper {
    QHelper(const float * imatrix, int n_per_row, int block_size) : m_imatrix(imatrix),
        m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};

template <int block_size, typename Block, typename Block_repacked, int n_repack, typename Func, typename RepackFunc>
size_t quantize_repack(ggml_type type, const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix,
        const Func& q_func, const RepackFunc& repack) {
    GGML_ASSERT(nrows%n_repack == 0);
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(type, n_per_row);
    std::vector<char> qtmp(n_repack*row_size);
    QHelper helper(imatrix, n_per_row, block_size);
    char * qrow = (char *)dst;
    for (int row = 0; row < nrows; row += n_repack) {
        helper.quantize(n_repack, src, qtmp.data(), row_size, q_func);
        repack(n_repack, n_per_row, (const Block *)qtmp.data(), (Block_repacked *)qrow, false);
        src += n_repack*n_per_row;
        qrow += n_repack*row_size;
    }
    return nrows*row_size;
}


float make_qx_quants(int n, int nmax, const float * x, int8_t * L, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (!amax) { // all zero
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = std::max(-nmax, std::min(nmax-1, l));
        L[i] = l + nmax;
        sumlx += qw[i]*x[i]*l;
        suml2 += qw[i]*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = std::max(-nmax, std::min(nmax-1, l));
            sumlx += qw[i]*x[i]*l;
            suml2 += qw[i]*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + std::max(-nmax, std::min(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

template <int block_size, int group_size, int num_bits, bool is_abs = false, bool is_int = false>
class QuantizerIQKT {
    static_assert(group_size == 8 || group_size == 4);
    static_assert(block_size >= 8 && block_size%8 == 0);
public:
    constexpr static int kSuperBlockSize = QK_K;
    constexpr static int kBlockSize = block_size;
    constexpr static int kGroupSize = group_size;
    constexpr static int kNg = kBlockSize/kGroupSize;
    constexpr static int kNblock = kSuperBlockSize/kBlockSize;
    constexpr static int kNumVal = 1 << num_bits; // i.e, 16 bits per group of 8
    constexpr static float kScale = is_int ? 1.f : 31.75f;
    constexpr static bool kVerbose = false;

    QuantizerIQKT(int num_clusters, int num_neighbours, int offset = 4096);
    const float * values() const { return m_values.data(); }

    inline void find_best_match(float d, const float * xb, const float * weight, int * best_idx) const;
    inline std::pair<float, float> find_best_scale(const float * xb, const float * weight, const int * best_idx) const;
    inline float find_best_inverse_scale(const float * xb, const float * weight, const int * best_idx) const;

    static inline void set_values(uint32_t i, float * result, float scale, int offset = 4096) {
        uint32_t x = i + offset;
        if constexpr (is_int) {
            constexpr uint32_t ka = 0xCBAC1FED;
            uint32_t s;
            auto i8 = (const int8_t *)&s;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x;
                s = x & 0x3f3f3f3f;
                if constexpr (is_abs) {
                    result[k] = scale*std::abs(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                } else {
                    result[k] = scale*(i8[0] + i8[1] + i8[2] + i8[3] - 126.f);
                }
            }
        } else {
            constexpr uint32_t ka = 89226354;
            constexpr uint32_t kb = 64248484;
            constexpr uint32_t kmask = 0x8fff8fff;
            constexpr uint32_t km32 = 0x3b603b60;
            for (int k = 0; k < kGroupSize; ++k) {
                x = ka*x + kb;
                uint32_t s = (x & kmask) ^ km32;
                float val = GGML_FP16_TO_FP32(s & 65535) + GGML_FP16_TO_FP32(s >> 16);
                if constexpr (is_abs) result[k] = scale*std::abs(val);
                else result[k] = scale*val;
            }
        }
    }

    static inline int bin4(float x) {
        if constexpr (is_abs) {
            return x < 16.f ? 0 : x < 32.f ? 1 : x < 64.f ? 2 : 3;
        } else {
            return x < -24.f ? 0 : x < 0.0f ? 1 : x < 24.f ? 2 : 3;
        }
    }
    static inline int bin5(float x) {
        if constexpr (is_abs) {
            return x < 11.2f ? 0 : x < 24.f ? 1 : x < 39.f ? 2 : x < 58.f ? 3 : 4;
        } else {
            return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
        }
    }
    inline int bin3(int idim, float x) const { return x < m_mid[2*idim+0] ? 0 : x < m_mid[2*idim+1] ? 1 : 2; }

    static inline void set_weights(float sigma2_scale, int nblock, const float * x, const float * imatrix, float * row_weights) {
        constexpr float kEps2   = 1e-14f;
        constexpr float kWeight = 1e-4f;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*kSuperBlockSize;
            float * wbl = row_weights + ibl*kSuperBlockSize;

            float sumx2 = 0;
            for (int j = 0; j < kSuperBlockSize; ++j) sumx2 += xbl[j]*xbl[j];
            if (sumx2 < kEps2*kSuperBlockSize) {
                // all x in th super block are (almost) zero
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = sigma2_scale*sumx2/kSuperBlockSize;

            if (imatrix) {
                for (int ib = 0; ib < kSuperBlockSize/kBlockSize; ++ib) {
                    const float * qw = imatrix + ibl*kSuperBlockSize + ib*kBlockSize;
                    const float * xb = xbl + ib*kBlockSize;
                    float * wb = wbl + ib*kBlockSize;
                    float sumwx = 0, sumw2 = 0, sumx2 = 0;
                    for (int j = 0; j < kBlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                        sumwx += wb[j]*std::abs(xb[j]);
                        sumw2 += wb[j]*wb[j];
                        sumx2 += xb[j]*xb[j];
                    }
                    if (sumx2 < kEps2 || sumw2 < kEps2 || sumwx < kEps2) {
                        for (int j = 0; j < kBlockSize; ++j) wb[j] = kWeight;
                    }
                }
            } else {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = 0.25f*sigma2 + xbl[j]*xbl[j];
            }
        }
    }
private:
    static std::vector<float> cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid);
    static std::vector<std::vector<int>> finalize_clusters(int num_neighbours, const std::vector<float>& points, const std::vector<float>& clusters,
            std::vector<std::vector<float>>& c_values);
    std::vector<float> m_values;
    std::vector<float> m_clusters;
    std::vector<std::vector<int>> m_in_cluster;
    std::vector<std::vector<float>> m_c_values;
    float m_mid[4*kGroupSize];
};

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::QuantizerIQKT(int num_clusters, int num_neighbours, int offset) {
    m_values.resize(kNumVal*kGroupSize);
    float * data = m_values.data();
    for (int i = 0; i < kNumVal; ++i) {
        set_values(i, data, kScale, offset);
        data += kGroupSize;
    }
    if (num_clusters == 0) return;
    // Make 128 clusters.
    // Note: we get a slightly better result by using 64 clusters
    //       at the expense of almost doubling the quantization time.
    m_clusters = cluster_points(m_values, num_clusters, 200, m_mid);
    GGML_ASSERT(!m_clusters.empty());
    m_in_cluster = finalize_clusters(num_neighbours, m_values, m_clusters, m_c_values);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::pair<float, float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumq2 = 0;
#if defined(_MSC_VER) || defined(__AVX2__)
    auto vqx = _mm256_setzero_ps();
    auto vq2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vqw = _mm256_mul_ps(vq, vw);
        vqx = _mm256_fmadd_ps(vqw, vx, vqx);
        vq2 = _mm256_fmadd_ps(vqw, vq, vq2);
    }
    sumqx = hsum_float_8(vqx);
    sumq2 = hsum_float_8(vq2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumq2 += wl[k]*ql[k]*ql[k];
        }
    }
#endif
    return sumq2 > 0 ? std::make_pair(sumqx/sumq2, sumqx*sumqx/sumq2) : std::make_pair(0.f, 0.f);
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
float QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_inverse_scale(
        const float * xb, const float * weight, const int * best_idx) const {
    float sumqx = 0, sumx2 = 0;
#if defined(_MSC_VER) || defined(__AVX2__)
    auto vqx = _mm256_setzero_ps();
    auto vx2 = _mm256_setzero_ps();
    for (int l = 0; l < kBlockSize; l += 8) {
        auto vx = _mm256_loadu_ps(xb+l);
        auto vw = _mm256_loadu_ps(weight+l);
        auto vq = kGroupSize == 8 ? _mm256_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize]) :
            _mm256_set_m128(_mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+1]),
                            _mm_loadu_ps(m_values.data() + kGroupSize*best_idx[l/kGroupSize+0]));
        auto vxw = _mm256_mul_ps(vx, vw);
        vx2 = _mm256_fmadd_ps(vxw, vx, vx2);
        vqx = _mm256_fmadd_ps(vxw, vq, vqx);
    }
    sumqx = hsum_float_8(vqx);
    sumx2 = hsum_float_8(vx2);
#else
    for (int l = 0; l < kNg; ++l) {
        auto xl = xb + kGroupSize*l;
        auto wl = weight + kGroupSize*l;
        auto ql = m_values.data() + kGroupSize*best_idx[l];
        for (int k = 0; k < kGroupSize; ++k) {
            sumqx += wl[k]*ql[k]*xl[k];
            sumx2 += wl[k]*xl[k]*xl[k];
        }
    }
#endif
    return sumx2 > 0 ? sumqx/sumx2 : 0.f;
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
void QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::find_best_match(float d,
        [[maybe_unused]] const float * xb, [[maybe_unused]] const float * weight, int * best_idx) const {
    if (!d) {
        std::memset(best_idx, 0, kNg*sizeof(int));
        return;
    }
    [[maybe_unused]] int ncluster = m_clusters.size()/kGroupSize;
    [[maybe_unused]] float id = 1/d;
#if defined(_MSC_VER) || defined(__AVX2__)
    if constexpr (kGroupSize == 8) {
        __m256 sqx[8];
        const __m256i add_idx = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        float sx[8];
        int   index[8];
        auto vid = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 8*l;
            auto wl = weight + 8*l;
            auto vx = _mm256_mul_ps(vid, _mm256_loadu_ps(xl));
            auto vw = _mm256_loadu_ps(wl);
            int jbest = -1;
            if (kGroupSize == 8 && (ncluster == 256 || ncluster == 6561)) {
                _mm256_store_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int j = 0; j < 8; ++j) if (sx[j] > m_mid[j]) u |= (1 << j);
                } else {
                    int s = 1;
                    for (int j = 0; j < 8; ++j) { u += s*bin3(j, sx[j]); s *= 3; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 8; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                    }
                    auto score = hsum_float_8x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = points.empty() ? m_values : m_c_values[jbest];
            int npoint = values.size()/kGroupSize;
            GGML_ASSERT(npoint > 0 && npoint%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            auto best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < npoint; j += 8) {
                for (int i = 0; i < 8; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_8x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                        _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points.empty() ? jbest : points[jbest];
        }
    } else {
        __m256 sqx[4];
        const __m256i add_idx = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        const __m256 sign_bit = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        float sx[8];
        int   index[8];
        auto vid_p = _mm256_set1_ps(id);
        auto add8 = _mm256_set1_epi32(8);
        for (int l = 0; l < kNg; ++l) {
            auto xl = xb + 4*l;
            auto wl = weight + 4*l;
            auto vx4 = _mm_loadu_ps(xl);
            auto vx = _mm256_mul_ps(vid_p, _mm256_set_m128(vx4, vx4));
            auto vw4 = _mm_loadu_ps(wl);
            auto vw = _mm256_set_m128(vw4, vw4);
            int jbest = -1;
            if (ncluster == 256 || ncluster == 625) {
                _mm256_storeu_ps(sx, vx);
                uint16_t u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < 4; ++k) u |= (bin4(sx[k]) << 2*k);
                } else {
                    int l = 1;
                    for (int k = 0; k < 4; ++k) { u += bin5(sx[k])*l; l *= 5; }
                }
                jbest = u;
            } else {
                auto vbest = _mm256_set1_ps(INFINITY);
                auto best_index = _mm256_set1_epi32(-1);
                float best = INFINITY;
                auto idx = add_idx;
                for (int j = 0; j < ncluster; j += 8) {
                    for (int i = 0; i < 4; ++i) {
                        auto vq = _mm256_loadu_ps(m_clusters.data() + kGroupSize*(j+2*i));
                        auto vdiff = _mm256_sub_ps(vq, vx);
                        vdiff = _mm256_and_ps(sign_bit, vdiff);
                        sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, _mm256_mul_ps(vdiff, vdiff)));
                    }
                    auto score = hsum_float_4x8(sqx);
                    auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                    best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                            _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                    vbest = _mm256_min_ps(vbest, score);
                    idx = _mm256_add_epi32(idx, add8);
                }
                _mm256_store_ps(sx, vbest);
                _mm256_store_si256((__m256i *)index, best_index);
                for (int i = 0; i < 8; ++i) {
                    if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
                }
            }
            auto& points = m_in_cluster[jbest];
            auto& values = m_c_values[jbest];
            GGML_ASSERT(!points.empty() && points.size()%8 == 0);
            int jbest_cluster = jbest;
            auto vbest = _mm256_set1_ps(INFINITY);
            auto best_index = _mm256_set1_epi32(-1);
            float best = INFINITY; jbest = -1;
            auto idx = add_idx;
            for (int j = 0; j < int(points.size()); j += 8) {
                for (int i = 0; i < 4; ++i) {
                    auto vq = _mm256_loadu_ps(values.data() + kGroupSize*(j+2*i));
                    auto vdiff = _mm256_sub_ps(vq, vx);
                    sqx[i] = _mm256_mul_ps(vw, _mm256_mul_ps(vdiff, vdiff));
                }
                auto score = hsum_float_4x8(sqx);
                auto mask  = _mm256_cmp_ps(score, vbest, _CMP_LT_OQ);
                best_index = _mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(mask), idx),
                                       _mm256_andnot_si256(_mm256_castps_si256(mask), best_index));
                vbest = _mm256_min_ps(vbest, score);
                idx = _mm256_add_epi32(idx, add8);
            }
            _mm256_store_ps(sx, vbest);
            _mm256_store_si256((__m256i *)index, best_index);
            for (int i = 0; i < 8; ++i) {
                if (sx[i] < best) { best = sx[i]; jbest = index[i]; }
            }
            if (jbest < 0) {
                fprintf(stderr, "Oops: jbest = %d for cluster %d with %d points\n", jbest, jbest_cluster, int(points.size()));
                GGML_ASSERT(false);
            }
            best_idx[l] = points[jbest];
        }
    }
#else
    // TODO
    std::memset(best_idx, 0, kNg*sizeof(int));
#endif
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<std::vector<int>> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::finalize_clusters(int num_neighbours,
        const std::vector<float>& values, const std::vector<float>& clusters, std::vector<std::vector<float>>& c_values) {
    int ncluster = clusters.size()/kGroupSize;
    std::vector<std::vector<int>> p_in_cluster(ncluster);
    std::vector<int> which_cluster(num_neighbours*kNumVal);
    std::vector<int> ibest(num_neighbours);
    std::vector<float> best(num_neighbours);
    for (int ip = 0; ip < kNumVal; ++ip) {
        auto vp = values.data() + ip*kGroupSize;
        for (int j = 0; j < num_neighbours; ++j) {
            best[j] = INFINITY; ibest[j] = -1;
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            auto vc = clusters.data() + ic*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            for (int j = 0; j < num_neighbours; ++j) {
                if (dist2 < best[j]) {
                    for (int k = num_neighbours-1; k > j; --k) {
                        best[k] = best[k-1]; ibest[k] = ibest[k-1];
                    }
                    best[j] = dist2; ibest[j] = ic;
                    break;
                }
            }
        }
        for (int j = 0; j < num_neighbours; ++j) {
            if (ibest[j] < 0) {
                printf("Oops: ibest[%d] = %d\n", j, ibest[j]);
            }
            GGML_ASSERT(ibest[j] >= 0);
            p_in_cluster[ibest[j]].push_back(ip);
        }
        std::memcpy(which_cluster.data() + num_neighbours*ip, ibest.data(), num_neighbours*sizeof(int));
    }
    std::vector<std::pair<float, int>> extra;
    extra.reserve(kNumVal);
    for (int ic = 0; ic < ncluster; ++ic) {
        auto& points = p_in_cluster[ic];
        if (!points.empty() && points.size()%8 == 0) continue;
        extra.clear();
        auto vc = clusters.data() + ic*kGroupSize;
        for (int ip = 0; ip < kNumVal; ++ip) {
            bool can_add = true;
            for (int j = 0; j < num_neighbours; ++j) {
                if (which_cluster[num_neighbours*ip+j] == ic) { can_add = false; break; }
            }
            if (!can_add) continue;
            auto vp = values.data() + ip*kGroupSize;
            float dist2 = 0;
            for (int k = 0; k < kGroupSize; ++k) {
                float d = vp[k] - vc[k]; dist2 += d*d;
            }
            extra.push_back(std::make_pair(dist2, ip));
        }
        std::sort(extra.begin(), extra.end());
        int nadd = 8*((points.size()+7)/8) - points.size();
        for (int i = 0; i < nadd; ++i) points.push_back(extra[i].second);
        GGML_ASSERT(points.size()%8 == 0);
    }
    auto min = p_in_cluster.front().size(), max = p_in_cluster.front().size();
    for (auto& points : p_in_cluster) {
        min = std::min(min, points.size());
        max = std::max(max, points.size());
    }
    c_values.resize(p_in_cluster.size());
    for (int i = 0; i < int(p_in_cluster.size()); ++i) {
        auto& points = p_in_cluster[i];
        c_values[i].resize(points.size()*kGroupSize);
        auto ptr = c_values[i].data();
        for (auto j : points) {
            std::memcpy(ptr, values.data() + j*kGroupSize, kGroupSize*sizeof(float));
            ptr += kGroupSize;
        }
    }

    if (kVerbose) {
        printf("%s: prepared %d clusters\n", __func__, ncluster);
        printf("    min number of points in a cluster: %d\n", int(min));
        printf("    max number of points in a cluster: %d\n", int(max));
    }
    return p_in_cluster;
}

template <int block_size, int group_size, int num_bits, bool is_abs, bool is_int>
std::vector<float> QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int>::cluster_points(const std::vector<float>& points, int ncluster, int niter, float * mid) {
    constexpr int ndim = kGroupSize;
    GGML_ASSERT(points.size() % ndim == 0);
    int npoint = points.size() / ndim;
    GGML_ASSERT(npoint >= 2*ncluster);
    std::vector<std::pair<float, float>> range(ndim, std::make_pair(INFINITY, -INFINITY));
    double Fo = 0;
    for (int i = 0; i < npoint; ++i) {
        auto v = points.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) {
            Fo += v[k]*v[k];
            range[k].first  = std::min(range[k].first, v[k]);
            range[k].second = std::max(range[k].second, v[k]);
        }
    }
    if (kVerbose) printf("%s (ndim = %d, npoint = %d): Fo = %g\n", __func__, ndim, npoint, Fo/points.size());
    if constexpr (is_abs) {
        std::vector<int> P(npoint);
        for (int idim = 0; idim < ndim; ++idim) {
            for (int ip = 0; ip < npoint; ++ip) P[ip] = points[ip*ndim+idim];
            std::sort(P.begin(), P.end());
            if (ndim == 8 && ncluster == 6561) {
                mid[2*idim + 0] = P[npoint/3];
                mid[2*idim + 1] = P[2*npoint/3];
            } else {
                mid[idim] = npoint%2 == 0 ? 0.5f*(P[npoint/2] + P[npoint/2-1]) : P[npoint/2];
                if (kVerbose) printf("%s: mid[%d] = %g\n", __func__, idim, mid[idim]);
            }
        }
    } else {
        for (int k = 0; k < ndim; ++k) mid[k] = 0.5f*(range[k].first + range[k].second);
    }
    std::vector<float> sump(ncluster*ndim);
    std::vector<int> counts(ncluster);
    std::vector<float> result(ncluster*ndim);
    if (ndim == 8 && (ncluster == 256 || ncluster == 6561)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) if (vp[k] > mid[k]) u |= (1 << k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) {
                    int bin = vp[k] < mid[2*k+0] ? 0 : vp[k] < mid[2*k+1] ? 1 : 2;
                    u += s*bin; s *= 3;
                }
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                printf("%s: Oops. Cluster %d has no points\n", __func__, ic);
                GGML_ABORT("fatal error");
            }
            for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
        }
        return result;
    }
    else if (ndim == 4 && (ncluster == 256 || ncluster == 625)) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            uint16_t u = 0;
            if (ncluster == 256) {
                for (int k = 0; k < ndim; ++k) u |= (bin4(vp[k]) << 2*k);
            } else {
                int s = 1;
                for (int k = 0; k < ndim; ++k) { u += s*bin5(vp[k]); s *= 5; }
            }
            if (u >= int(counts.size())) {
                printf("Oops: u = %u, vp = %g, %g, %g, %g\n", u, vp[0], vp[1], vp[2], vp[3]);
                u = 0;
                if (ncluster == 256) {
                    for (int k = 0; k < ndim; ++k) {
                        auto bin = bin4(vp[k]); u |= (bin << 2*k);
                        printf(" bin[%d] = %d, u = %u", k, bin, u);
                    }
                } else {
                    for (int k = 0; k < ndim; ++k) printf(" bin[%d] = %d", k, bin5(vp[k]));
                }
                printf("\n");
                GGML_ABORT("fatal error");
            }
            ++counts[u];
            for (int k = 0; k < ndim; ++k) sump[ndim*u + k] += vp[k];
        }
        int nzero = 0;
        for (int ic = 0; ic < ncluster; ++ic) {
            if (!counts[ic]) {
                ++nzero;
                printf("%s: Oops. Cluster %d has no points: ", __func__, ic);
                for (int k = 0; k < ndim; ++k) {
                    int l = (ic >> 2*k) & 3;
                    printf(" %d", l);
                }
                printf("\n");
            } else {
                for (int k = 0; k < ndim; ++k) result[ic*ndim + k] = sump[ic*ndim + k]/counts[ic];
            }
        }
        if (nzero > 0) printf("%s: %d out of %d clusters dir not have any points\n", __func__, nzero, ncluster);
        return result;
    }
    std::mt19937 rndm(1234);
    float scale = 1.f/4294967296.f;
    for (int i = 0; i < ncluster; ++i) {
        auto v = result.data() + i*ndim;
        for (int k = 0; k < ndim; ++k) v[k] = range[k].first + (range[k].second - range[k].first)*scale*rndm();
    }
    std::vector<int> which_cluster(npoint, -1);
    double Flast = Fo;
    for (int iter = 0; iter < niter; ++iter) {
        std::memset(sump.data(), 0, sump.size()*sizeof(float));
        std::memset(counts.data(), 0, counts.size()*sizeof(int));
        int nchanged = 0;
        double F = 0;
        for (int ip = 0; ip < npoint; ++ip) {
            auto vp = points.data() + ndim*ip;
            float best = INFINITY; int ibest = -1;
            for (int ic = 0; ic < ncluster; ++ic) {
                auto vc = result.data() + ndim*ic;
                float dist2 = 0;
                for (int k = 0; k < ndim; ++k) {
                    float d = vp[k] - vc[k]; dist2 += d*d;
                }
                if (dist2 < best) {
                    best = dist2; ibest = ic;
                }
            }
            if (ibest < 0) {
                printf("Oops(iteration %d) - failed to find cluster for point", iter);
                for (int k = 0; k < ndim; ++k) printf(" %g", vp[k]);
                printf("\nHave %d clusters\n", ncluster);
            }
            GGML_ASSERT(ibest >= 0);
            F += best;
            if (which_cluster[ip] != ibest) ++nchanged;
            which_cluster[ip] = ibest;
            ++counts[ibest];
            auto vc = sump.data() + ndim*ibest;
            for (int k = 0; k < ndim; ++k) vc[k] += vp[k];
        }
        if (nchanged == 0) break;
        for (int ic = 0; ic < ncluster; ++ic) {
            float norm = counts[ic] > 0 ? 1.f/counts[ic] : 0.f;
            auto vc = sump.data() + ndim*ic;
            auto r  = result.data() + ndim*ic;
            for (int k = 0; k < ndim; ++k) r[k] = vc[k]*norm;
        }
        if (kVerbose) printf("%s(iteration %d): F = %g, nchanged = %d\n", __func__, iter+1, F/points.size(), nchanged);
        if (iter > 1 && Flast/F - 1 < 1e-6) break;
        Flast = F;
    }
    int nzero = 0;
    for (int ic = 0; ic < ncluster; ++ic) {
        if (!counts[ic]) ++nzero;
    }
    if (nzero > 0) printf("%s: there are %d empty clusters\n", __func__, nzero);
    return result;
}

}

using QuantizerIQ1KT = QuantizerIQKT<32, 8, 13, false, true>;
static const QuantizerIQ1KT& iq1kt_quantizer() {
    static QuantizerIQ1KT quantizer(256, 32);
    return quantizer;
}

using QuantizerIQ2KT = QuantizerIQKT<32, 8, 16, false, true>;
static const QuantizerIQ2KT& iq2kt_quantizer() {
    static QuantizerIQ2KT quantizer(256, 8);
    return quantizer;
}

using QuantizerIQ3KT = QuantizerIQKT<32, 8, 16, true, true>;
static const QuantizerIQ3KT& iq3kt_quantizer() {
    static QuantizerIQ3KT quantizer(256, 8);
    return quantizer;
}

using QuantizerIQ4KT = QuantizerIQKT<32, 4, 15, false, true>;
static const QuantizerIQ4KT& iq4kt_quantizer(bool with_offset = false) {
    static QuantizerIQ4KT quantizer1(625, 6, 4096);
    static QuantizerIQ4KT quantizer2(625, 6, 4096+32768);
    return with_offset ? quantizer2 : quantizer1;
}
static const QuantizerIQ4KT& iq4kt_dequantizer() {
    static QuantizerIQ4KT quantizer(0, 0, 4096);
    return quantizer;
}


// ========================================== iq*_k* ====================================================

namespace {
// ========================================== iq4_kss ====================================================
const uint16_t * scramble_table() {
    static auto table = []() {
        std::vector<uint16_t> table;
        table.resize(1 << 15);
        for (int i = 0; i < int(table.size()); ++i) {
            uint16_t val = i;
            int non = popcount(val);
            if (non%2) val |= (1 << 15);
            bool found = false;
            for (int j = 0; j < int(table.size()); ++j) {
                if ((j ^ (j << 1)) == val) {
                    table[i] = j; found = true; break;
                }
            }
            if (!found) {
                printf("Oops: did not find for %d %u\n", i, val);
                exit(1);
            }
        }
        return table;
    }();
    return table.data();
}
uint16_t prune_iq4ks(uint16_t v, const int8_t * values, const float * x, const float * w, float dl) {
    if (popcount(v)%2 == 0) return v;
    float best_score = std::numeric_limits<float>::max();
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        auto pc = popcount(q);
        float diff0 = dl*iq4k_values[q] - x[j];
        int qmin = std::max(int(q)-2,  0);
        int qmax = std::min(int(q)+2, 15);
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = iq;
            if (qq == q) continue;
            int pci = popcount(qq);
            if (std::abs(pci - pc)%2) {
                float diff1 = dl*values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) {
                    best_score = score; jbest = j; bestq = qq;
                }
            }
        }
    }
    GGML_ASSERT(jbest >= 0);
    q4[jbest] = bestq;
    return (q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}
static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const uint16_t * table,
        const int ntry) {

    constexpr int super_block_size = 256;
    constexpr int block_size = 32;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float d = -max/iq4k_values[0];
            std::memset(vs, 0, block_size);
            for (int itry = -ntry; itry <= ntry; ++itry) {
                float id = (itry + values[0])/max;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                float this_d = 1/id;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(values,  al) << 4*j);
                        vm |= (best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }

                id = (itry + shifted_values[0])/max;
                this_d = 1/id;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = prune_iq4ks(vp, shifted_values, xv, wv,  this_d);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j];
                        sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j];
                        sumq2_m += w*q*q;
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; copy_p = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; copy_m = true;
                }
                if (copy_m) {
                    std::memcpy(vs, vms, block_size);
                } else if (copy_p) {
                    std::memcpy(vs, vps, block_size);
                }
            }
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = d;
    if (!d) return;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = (std::max(0, std::min(127, l)) << 1) - 127;
            if (l) {
                float dl = d*l;
                float idl = 1/dl;
                float mse_p = 0, mse_m = 0;
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (best_index_iq4nl(        values, al) << 4*j);
                        vm |= (best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = prune_iq4ks(vp,         values, xv, wv,  dl);
                    vm = prune_iq4ks(vm, shifted_values, xv, wv,  dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[k] = vp;
                    vms[k] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) {
                    v = vms;
                    block_values = values + 16;
                }
                for (int k = 0; k < block_size/4; ++k) {
                    xv[0] =     xb[2*k+0]; xv[1] =     xb[2*k+0+block_size/2]; xv[2] =     xb[2*k+1]; xv[3] =     xb[2*k+1+block_size/2];
                    wv[0] = weight[2*k+0]; wv[1] = weight[2*k+0+block_size/2]; wv[2] = weight[2*k+1]; wv[3] = weight[2*k+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[k] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                l += 127;
                if (mse_m < mse_p) l |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    auto val = table[v[k] & 0x7fff];
                    q16[k] = (val << 1) | ((l >> k) & 1);
                }
            } else {
                l += 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int k = 0; k < block_size/4; ++k) {
                    q16[k] = ((l >> k) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2 * 1.01f;
}


// ========================================== iq2_ks ====================================================
#if defined(__AVX2__)
inline void to_values_i32(__m256i idx, __m256i ivalues, __m256i * iv) {
    auto ival = _mm256_shuffle_epi8(ivalues, idx);
    auto ival_1 = _mm256_srli_si256(ival, 8);
    iv[0] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival));
    iv[1] = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(ival_1));
    iv[2] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival, 1));
    iv[3] = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(ival_1, 1));
}
inline __m256i to_int8(const __m256i * ibest) {
    auto i0 = _mm256_packs_epi32(ibest[0], ibest[1]); // 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
    auto i1 = _mm256_packs_epi32(ibest[2], ibest[3]); // 16, 17, 18, 19, 24, 25, 26, 27, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx = _mm256_packs_epi16(i0, i1); // 0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
    auto idx_l = _mm256_castsi256_si128(idx);
    auto idx_h = _mm256_extracti128_si256(idx, 1);
    auto idx1  = _mm_unpacklo_epi32(idx_l, idx_h); // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    auto idx2  = _mm_unpackhi_epi32(idx_l, idx_h);
    return MM256_SET_M128I(idx2, idx1);
}
bool compute_1block_iq2ks(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx, float & best_d, float & score) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[8];
    __m256i ibest[8];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
        diff = _mm256_add_ps(vx[k], val);
        vbest[k+4] = _mm256_mul_ps(diff, diff);
        ibest[k+4] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
            diff = _mm256_add_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            mask = _mm256_cmp_ps(diff, vbest[k+4], _CMP_LT_OQ);
            vbest[k+4] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k+4]));
            imask = _mm256_castps_si256(mask);
            ibest[k+4] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k+4]));
        }
    }
    bool result = false;
    auto idx1 = to_int8(ibest+0);
    auto idx2 = to_int8(ibest+4);
    to_values_i32(idx1, ivalues, ibest+0);
    to_values_i32(idx2, ivalues, ibest+4);
    auto vsqx_1 = _mm256_setzero_ps();
    auto vsq2_1 = _mm256_setzero_ps();
    auto vsqx_2 = _mm256_setzero_ps();
    auto vsq2_2 = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq1  = _mm256_cvtepi32_ps(ibest[k+0]);
        auto vwq1 = _mm256_mul_ps(vw[k], vq1);
        auto vq2  = _mm256_cvtepi32_ps(ibest[k+4]);
        auto vwq2 = _mm256_mul_ps(vw[k], vq2);
        vsqx_1 = _mm256_fmadd_ps(vwq1, vx[k], vsqx_1);
        vsq2_1 = _mm256_fmadd_ps(vwq1, vq1,   vsq2_1);
        vsqx_2 = _mm256_fmadd_ps(vwq2, vx[k], vsqx_2);
        vsq2_2 = _mm256_fmadd_ps(vwq2, vq2,   vsq2_2);
    }
    auto sumqx_1 = hsum_float_8(vsqx_1);
    auto sumq2_1 = hsum_float_8(vsq2_1);
    auto sumqx_2 = hsum_float_8(vsqx_2);
    auto sumq2_2 = hsum_float_8(vsq2_2);
    if (sumq2_1 > 0) {
        best_d = sumqx_1/sumq2_1;
        score = sumqx_1 * best_d;
        this_idx = idx1;
        result = true;
    }
    if (sumq2_2 > 0 && (!result || sumqx_2*sumqx_2 > score*sumq2_2)) {
        best_d = sumqx_2/sumq2_2;
        score = sumqx_2 * best_d;
        this_idx = idx2;
        result = true;
    }
    return result;
}
float compute_1block_iq2ks_rmse(float d, const __m256 * vx, const __m256 * vw, const int8_t * values, __m256i & this_idx) {
    constexpr int kBlockSize = 32;
    uint32_t aux32;
    std::memcpy(&aux32, values, sizeof(aux32));
    auto ivalues = _mm256_set1_epi32(aux32);
    __m256 vbest[4];
    __m256i ibest[4];
    auto val = _mm256_set1_ps(d*values[0]);
    auto ival = _mm256_set1_epi32(0);
    for (int k = 0; k < kBlockSize/8; ++k) {
        auto diff = _mm256_sub_ps(vx[k], val);
        vbest[k] = _mm256_mul_ps(diff, diff);
        ibest[k] = ival;
    }
    for (int j = 1; j < 4; ++j) {
        val = _mm256_set1_ps(d*values[j]);
        ival = _mm256_set1_epi32(j);
        for (int k = 0; k < kBlockSize/8; ++k) {
            auto diff = _mm256_sub_ps(vx[k], val);
            diff = _mm256_mul_ps(diff, diff);
            auto mask = _mm256_cmp_ps(diff, vbest[k], _CMP_LT_OQ);
            vbest[k] = _mm256_or_ps(_mm256_and_ps(mask, diff), _mm256_andnot_ps(mask, vbest[k]));
            auto imask = _mm256_castps_si256(mask);
            ibest[k] = _mm256_or_si256(_mm256_and_si256(imask, ival), _mm256_andnot_si256(imask, ibest[k]));
        }
    }
    auto idx = to_int8(ibest);
    to_values_i32(idx, ivalues, ibest);
    auto vd = _mm256_set1_ps(-d);
    auto vrmse = _mm256_setzero_ps();
    for (int k = 0; k < 4; ++k) {
        auto vq   = _mm256_cvtepi32_ps(ibest[k]);
        auto diff = _mm256_fmadd_ps(vd, vq, vx[k]);
        auto wdiff = _mm256_mul_ps(vw[k], diff);
        vrmse = _mm256_fmadd_ps(wdiff, diff, vrmse);
    }
    this_idx = idx;
    return hsum_float_8(vrmse);
}
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    __m256 vx[4], vw[4];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0, sumw = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
                sumw += weight[j];
            }
            sw[ib] = sumw;
            if (amax < 1e-14f) {
                scales[ib] = 0;
                continue;
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            float d = max/iq2nl_values[7];
            float best = 0;
            __m256i this_idx;
            float this_d, this_score;
            if (compute_1block_iq2ks(d, vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                best = this_score; d = this_d;
            }
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[0] + 0.5f*itry), vx, vw, iq2nl_values, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d;
                    }
                }
            }
            bool is_shifted = false;
            for (int itry = -13; itry <= 13; ++itry) {
                if (compute_1block_iq2ks(max/(iq2nl_values[4] + 0.5f*itry), vx, vw, iq2nl_values + 4, this_idx, this_d, this_score)) {
                    if (this_score > best) {
                        best = this_score; d = this_d; is_shifted = true;
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    auto vsumqx = _mm256_setzero_ps();
    auto vsumq2 = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        __m256i idx[4];
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
            uint32_t aux32;
            std::memcpy(&aux32, block_values, sizeof(aux32));
            auto ivalues = _mm256_set1_epi32(aux32);
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            for (int k = 0; k < 4; ++k) {
                vx[k] = _mm256_loadu_ps(xb + 8*k);
                vw[k] = _mm256_loadu_ps(weight + 8*k);
            }
            int ls = Ls[ib] - 16;
            float dl = d*ls;
            __m256i idx1, idx2;
            auto rmse1 = compute_1block_iq2ks_rmse(dl, vx, vw, block_values, idx1);
            if (Ls[ib] > 0 && dl > scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 17), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    --Ls[ib]; idx1 = idx2;
                }
            }
            else if (Ls[ib] < 15 && dl < scales[ib]) {
                auto rmse2 = compute_1block_iq2ks_rmse(d*(Ls[ib] - 15), vx, vw, block_values, idx2);
                if (rmse2 < rmse1) {
                    ++Ls[ib]; idx1 = idx2;
                }
            }
            __m256i iv[4];
            to_values_i32(idx1, ivalues, iv);
            auto vd = _mm256_set1_ps(Ls[ib] - 16);
            for (int k = 0; k < 4; ++k) {
                auto vq = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(iv[k]));
                auto wvq = _mm256_mul_ps(vw[k], vq);
                vsumqx = _mm256_fmadd_ps(wvq, vx[k], vsumqx);
                vsumq2 = _mm256_fmadd_ps(wvq, vq,    vsumq2);
            }
            ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            idx[ib % 4] = idx1;
            if ((ib % 4) == 3) {
                auto vqs1 = _mm256_or_si256(idx[0], _mm256_slli_epi16(idx[1], 2));
                auto vqs2 = _mm256_or_si256(_mm256_slli_epi16(idx[2], 4), _mm256_slli_epi16(idx[3], 6));
                auto vqs = _mm256_or_si256(vqs1, vqs2);
                _mm256_storeu_si256((__m256i *)y[ibl].qs + ib/4, vqs);
            }
        }
    }
    float sumqx = hsum_float_8(vsumqx);
    float sumq2 = hsum_float_8(vsumq2);
    *dptr = GGML_FP32_TO_FP16(1.000f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
#else
void quantize_row_iq2_ks_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_sw, int8_t * all_Ls) {

    constexpr int kBlockSize = 32;
    constexpr int kMax_i1 = 3*kBlockSize/4;
    constexpr int kMin_i3 = kBlockSize/4;
    //constexpr int kNtry = 5;
    //constexpr float kStep = 1.f;

    ggml_half * dptr = (ggml_half *)vy;
    *dptr = GGML_FP32_TO_FP16(0.f);

    block_iq2_ks * y = (block_iq2_ks *)(dptr + 1);

    float weight[kBlockSize];
    float sumx[kBlockSize+1], sumw[kBlockSize+1];

    std::array<std::pair<float,int>, kBlockSize> pairs;

    float val [4] = {float(iq2nl_values[0]), float(iq2nl_values[1]), float(iq2nl_values[2]), float(iq2nl_values[3])};
    float sval[4] = {float(iq2nl_values[4]), float(iq2nl_values[5]), float(iq2nl_values[6]), float(iq2nl_values[7])};

    const int8_t * shifted_values = iq2nl_values + 4;

    const int nblock = n_per_row/QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_ks));

        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto sw = all_sw + ibl*(QK_K/kBlockSize);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float * xb = xbl + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            sw[ib] = 0;
            float amax = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                sw[ib] += weight[j];
                pairs[j] = {xb[j], j};
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            std::sort(pairs.begin(), pairs.end());
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                int jj = pairs[j].second;
                sumw[j+1] = sumw[j] + weight[jj];
                sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
            }
            float best = 0, d = 0;
            bool is_shifted = false;
            float sumqx, sumq2;
            for (int i1 = 0; i1 < kMax_i1; ++i1) {
                for (int i2 = i1; i2 < kBlockSize; ++i2) {
                    for (int i3 = std::max(i2, kMin_i3); i3 < kBlockSize; ++i3) {
                        sumqx = (sumx[i1] - sumx[ 0])*val[0] + (sumx[i2] - sumx[i1])*val[1]
                              + (sumx[i3] - sumx[i2])*val[2] + (sumx[kBlockSize] - sumx[i3])*val[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[0]*val[0] + (sumw[i2] - sumw[i1])*val[1]*val[1]
                              + (sumw[i3] - sumw[i2])*val[2]*val[2] + (sumw[kBlockSize] - sumw[i3])*val[3]*val[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[0] + (sumx[i2] - sumx[i1])*sval[1]
                              + (sumx[i3] - sumx[i2])*sval[2] + (sumx[kBlockSize] - sumx[i3])*sval[3];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[0]*sval[0] + (sumw[i2] - sumw[i1])*sval[1]*sval[1]
                              + (sumw[i3] - sumw[i2])*sval[2]*sval[2] + (sumw[kBlockSize] - sumw[i3])*sval[3]*sval[3];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*val[3] + (sumx[i2        ] - sumx[i1])*val[2]
                              + (sumx[i3] - sumx[i2])*val[1] + (sumx[kBlockSize] - sumx[i3])*val[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*val[3]*val[3] + (sumw[i2        ] - sumw[i1])*val[2]*val[2]
                              + (sumw[i3] - sumw[i2])*val[1]*val[1] + (sumw[kBlockSize] - sumw[i3])*val[0]*val[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = false;
                        }
                        sumqx = (sumx[i1] - sumx[ 0])*sval[3] + (sumx[i2        ] - sumx[i1])*sval[2]
                              + (sumx[i3] - sumx[i2])*sval[1] + (sumx[kBlockSize] - sumx[i3])*sval[0];
                        sumq2 = (sumw[i1] - sumw[ 0])*sval[3]*sval[3] + (sumw[i2        ] - sumw[i1])*sval[2]*sval[2]
                              + (sumw[i3] - sumw[i2])*sval[1]*sval[1] + (sumw[kBlockSize] - sumw[i3])*sval[0]*sval[0];
                        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                            d = sumqx/sumq2; best = d*sumqx; is_shifted = true;
                        }
                    }
                }
            }
            scales[ib] = d;
            if (is_shifted) extra |= (1 << ib);
        }
        y[ibl].extra = extra;
    }

    float d = make_qx_quants(nblock*(QK_K/kBlockSize), 16, all_scales, all_Ls, all_sw);

    if (!d) return;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;
        auto Ls = all_Ls + ibl*(QK_K/kBlockSize);
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            int ls = Ls[ib];
            y[ibl].scales[ib/2] |= ((ls & 0xf) << 4*(ib%2));
            y[ibl].extra |= ((ls >> 4) << (8 + ib));
            ls -= 16;
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq2nl_values;
                const float * xb = xbl + kBlockSize*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                uint8_t * qs = y[ibl].qs + 32*(ib/4);
                for (int j = 0; j < 32; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq2nl(block_values, al);
                    qs[j] |= (ibest << 2*(ib%4));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
    }
    *dptr = GGML_FP32_TO_FP16(1.030f*(sumq2 > 0 ? sumqx/sumq2 : d));
}
#endif


// ============================================== iq3_ks 
static void quantize_row_iq3_ks_impl(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(d);
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(sumqx/sumq2);
}


// ========================================== iq4_ks ====================================================
static void quantize_row_iq4_k_impl_bs128(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    //GGML_ASSERT(super_block_size == 256 && block_size == 128);

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    //float mse = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            //printf("d = %g, id = %g, scales = %g, l = %d, dl = %g\n", d, id, scales[ib], l, d*(l - 127));
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
                //float diff = xb[j] - d*q1; mse += diff*diff;
                //diff = xb[j+block_size/2] - d*q2; mse += diff*diff;
            }
        }
    }
    //printf("rmse = %g\n", sqrt(mse/n_per_row));
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}


// ============================================== iq5_ks 
static void quantize_row_iq5_ks_impl(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float * dptr = (float *)cy;
    dptr[0] = 0;
    block_iq5_ks * y = (block_iq5_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 32;

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq5_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq5nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq5nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            for (int j = 0; j < block_size; ++j) {
                uint8_t idx = best_index_iq5nl(block_values, idl*xb[j]);
                y[ibl].qs[block_size*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float w = weight[j];
                float q = block_values[idx]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}


// ========================================== iq2_kl ====================================================
void quantize_row_iq2_kl_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales) {
    constexpr int kBlockSize = 32;
    constexpr float kSigmaFactor = 2.25f;
    constexpr int ntry = 5;
    static const int k_index[64] = {-1, -2, 0, -3, -4, 1, -5, -6, 2, -7, -8, 3, -9, 4, -10, 5, -11, 6, 7, -12, 8, 9, 10, -13, 11, -14, -15, -16, 12, 13, -17,
        14, -18, -19, 15, 16, 17, 18, 19, -20, -21, 20, 21, 22, 23, 24, -22, -23, 25, -24, 26, -25, 27, -26, 28, 29, -27, -28, 30, -29, -30, 31, -31, -32};
    static const std::vector<std::vector<int>> k_neighbours = {
        { 2, 0, 6, 11, 7, 3, 8, 15,  },
        { 0, 2, 3, 6, 7, 1, 8, 4,  },
        { 0, 1, 3, 4, 8, 7, 9, 6,  },
        { 1, 0, 3, 4, 8, 9, 7, 10,  },
        { 1, 4, 5, 10, 9, 3, 8, 0,  },
        { 5, 1, 4, 10, 9, 14, 8, 3,  },
        { 6, 2, 7, 0, 3, 11, 8, 15,  },
        { 3, 7, 0, 6, 8, 4, 12, 9,  },
        { 3, 4, 8, 9, 1, 7, 12, 10,  },
        { 4, 10, 5, 9, 1, 8, 13, 14,  },
        { 11, 2, 6, 7, 20, 15, 25, 21,  },
        { 8, 7, 3, 12, 9, 16, 17, 13,  },
        { 14, 5, 10, 19, 9, 13, 4, 18,  },
        { 6, 15, 7, 11, 20, 21, 16, 2,  },
        { 15, 7, 16, 6, 21, 12, 17, 22,  },
        { 12, 16, 17, 8, 15, 7, 13, 22,  },
        { 19, 10, 13, 18, 14, 9, 12, 24,  },
        { 11, 20, 25, 6, 15, 2, 21, 7,  },
        { 20, 15, 21, 6, 11, 7, 16, 26,  },
        { 14, 19, 29, 10, 28, 18, 13, 24,  },
        { 25, 11, 20, 21, 15, 6, 26, 30,  },
        { 19, 24, 28, 18, 29, 23, 13, 17,  },
        { 29, 19, 14, 28, 24, 18, 10, 13,  },
        { 20, 26, 21, 25, 30, 15, 22, 16,  },
        { 27, 26, 22, 23, 21, 30, 16, 24,  },
        { 27, 24, 28, 31, 23, 18, 22, 17,  },
        { 25, 30, 20, 26, 21, 11, 15, 22,  },
        { 30, 26, 25, 20, 21, 27, 22, 15,  },
        { 30, 27, 31, 26, 22, 23, 21, 24,  },
        { 31, 27, 30, 26, 28, 23, 22, 24,  },
        { 31, 28, 29, 27, 24, 23, 19, 18,  },
        { 29, 28, 31, 24, 19, 27, 14, 18,  },
    };
    auto values = iq3nl_values;
    std::pair<int8_t, int8_t> grid[32];
    for (int j = 0; j < 64; ++j) {
        if (int i = k_index[j]; i >= 0) {
            int i1 = j/8, i2 = j%8;
            grid[i] = {values[i1], values[i2]};
        }
    }

    ggml_half * dptr = (ggml_half *)vy;
    auto y = (block_iq2_kl *)(dptr + 1);

    float weight[kBlockSize];

    auto index = [&grid, values] (float id, float x1, float x2, float w1, float w2) {
        float sx1 = id*x1;
        float sx2 = id*x2;
        int l1 = best_index_iq3nl(values, sx1);
        int l2 = best_index_iq3nl(values, sx2);
        int i = k_index[8*l1 + l2];
        if (i >= 0) return i;
        auto& neigh = k_neighbours[-i-1];
        float best = std::numeric_limits<float>::max();
        int ibest = -1;
        for (auto& n : neigh) {
            float diff1 = grid[n].first  - sx1;
            float diff2 = grid[n].second - sx2;
            float score = w1*diff1*diff1 + w2*diff2*diff2;
            if (score < best) {
                best = score; ibest = n;
            }
        }
        GGML_ASSERT(ibest >= 0);
        return ibest;
    };

    float max_scale = 0, max_abs_scale = 0;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq2_kl));
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < kBlockSize; j += 2) {
                float w1 = weight[j+0];
                float w2 = weight[j+1];
                int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                float q1 = grid[idx].first ;
                float q2 = grid[idx].second;
                sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_p += w1*q1*q1 + w2*q2*q2;
                idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                q1 = grid[idx].first ;
                q2 = grid[idx].second;
                sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                sumq2_m += w1*q1*q1 + w2*q2*q2;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < kBlockSize; j += 2) {
                    float w1 = weight[j+0];
                    float w2 = weight[j+1];
                    int idx = index(id, xb[j+0], xb[j+1], w1, w2);
                    float q1 = grid[idx].first ;
                    float q2 = grid[idx].second;
                    sumqx_p += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_p += w1*q1*q1 + w2*q2*q2;
                    idx = index(-id, xb[j+0], xb[j+1], w1, w2);
                    q1 = grid[idx].first ;
                    q2 = grid[idx].second;
                    sumqx_m += w1*q1*xb[j] + w2*q2*xb[j+1];
                    sumq2_m += w1*q1*q1 + w2*q2*q2;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m;
                }
            }
            scales[ib] = d;
            float ad = std::abs(d);
            if (ad > max_abs_scale) {
                max_abs_scale = ad; max_scale = d;
            }
        }
    }

    if (!max_abs_scale) {
        dptr[0] = GGML_FP32_TO_FP16(0.f);
        return;
    }

    float d = -max_scale/32;
    float id = 1/d;

    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        auto scales = all_scales + ibl*(QK_K/kBlockSize);
        auto xbl = x + ibl*QK_K;
        float sigma2 = 0;
        for (int j = 0; j < QK_K; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= kSigmaFactor/QK_K;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            auto xb = xbl + ib*kBlockSize;
            if (quant_weights) {
                auto qw = quant_weights + ibl*QK_K + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j]*sqrt(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = std::abs(xb[j]); //xb[j]*xb[j];
            }
            int ls = nearest_int(id*scales[ib]);
            ls = std::max(-32, std::min(31, ls));
            int lsmin = std::max(-32, ls-1);
            int lsmax = std::min( 31, ls+1);
            float best_score = std::numeric_limits<float>::max();
            int best_ls = ls;
            for (int ils = lsmin; ils <= lsmax; ++ils) {
                float dl = d*ils;
                float idl = dl ? 1/dl : 0.f;
                float score = 0;
                for (int j = 0; j < kBlockSize/2; ++j) {
                    float w1 = weight[2*j+0];
                    float w2 = weight[2*j+1];
                    int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                    float diff1 = dl*grid[idx].first  - xb[2*j+0];
                    float diff2 = dl*grid[idx].second - xb[2*j+1];
                    score += w1*diff1*diff1 + w2*diff2*diff2;
                }
                if (score < best_score) {
                    best_score = score;
                    best_ls = ils;
                }
            }
            ls = best_ls;
            int uls = ls + 32;
            y[ibl].scales_l[ib%4] |= ((uls & 0xf) << 4*(ib/4));
            y[ibl].scales_h |= ((uls >> 4) << 2*ib);
            if (ls == 0) continue;
            float dl = d*ls;
            float idl = 1/dl;
            for (int j = 0; j < kBlockSize/2; ++j) {
                float w1 = weight[2*j+0];
                float w2 = weight[2*j+1];
                int idx = index(idl, xb[2*j+0], xb[2*j+1], w1, w2);
                y[ibl].qs[16*(ib/2) + j] |= ((idx & 0xf) << 4*(ib%2));
                y[ibl].qh[j] |= ((idx >> 4) << ib);
                float q1 = ls*grid[idx].first ;
                float q2 = ls*grid[idx].second;
                sumqx += w1*q1*xb[2*j] + w2*q2*xb[2*j+1];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) d = sumqx/sumq2;

    dptr[0] = GGML_FP32_TO_FP16(1.025f * d);

}
}


extern "C" {
size_t quantize_iq4_kss(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size    = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    std::vector<float> all_scales(n_per_row/kBlockSize);
    float weight[kBlockSize];
    auto table = scramble_table();
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, table] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq4_kss_impl(n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, table, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq4_kss_ref(const float * x, block_iq4_kss * y, int64_t k) {
    quantize_iq4_kss(x, y, 1, k, nullptr);
}

size_t quantize_iq2_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize)), all_sw(nblock*(QK_K/kBlockSize));
    std::vector<int8_t> all_Ls(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales, &all_sw, &all_Ls] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq2_ks_impl(x, vy, n_per_row, imatrix, all_scales.data(), all_sw.data(), all_Ls.data());
    };
    QHelper helper(imatrix, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq2_ks_ref(const float * x, block_iq2_ks * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq3_ks_impl(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq3nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq3_ks_ref(const float * x, block_iq3_ks * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq4_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq4_k_impl_bs128(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq4k_values, imatrix, 7);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq4_ks_ref(const float * x, block_iq4_ks * y, int64_t k) {
    quantize_iq4_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq5_ks_ref(const float * x, block_iq5_ks * y, int64_t k) {
    quantize_iq5_ks(x, (void *)y, 1, k, nullptr);
}
size_t quantize_iq5_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq5_ks_impl(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq5nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

size_t quantize_iq2_kl(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KL, n_per_row);
    int nblock = n_per_row/QK_K;
    std::vector<float> all_scales(nblock*(QK_K/kBlockSize));
    auto q_func = [&all_scales] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq2_kl_impl(x, vy, n_per_row, imatrix, all_scales.data());
    };
    QHelper helper(imatrix, n_per_row, kBlockSize);
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}
void quantize_row_iq2_kl_ref(const float * x, block_iq2_kl * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kl(x, (void *)y, 1, k, nullptr);
}
}


// ========================================== iq_kt ====================================================

namespace {
void quantize_row_iq1_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ1KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq1kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq1_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg; score_p = score_m;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            scale_0 -= 8;
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp1, score_p1] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm1, score_m1] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            if (score_p1 > score_p || score_m1 > score_p) {
                idx = best_idx;
                if (score_p1 > score_m1) scales[ib] = dp1;
                else {
                    scales[ib] = dm1; idx += Q::kNg;
                }
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
            y[ibl].sh[ib] = ls;
        }
    }

    *dptr = d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[y[ibl].sh[ib] & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    y[ibl].ql[ib*Q::kNg+j] = best_idx[j] & 0xff;
                    y[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg+j] |= (((best_idx[j] >> 8) & 0xf) << 4*(ib/(Q::kNblock/2)));
                    y[ibl].sh[ib] |= ((best_idx[j] >> 12) << (4+j));
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d * 1.07f;
            if (!d) return;
        } else {
            break;
        }

    }

}


void quantize_row_iq2_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights,
        int * all_idx) {

    constexpr float kSigmaScale = 2.0f;
    using Q = QuantizerIQ2KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq2kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq2_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = 0;
                continue;
            }
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            quantizer.find_best_match( amax/scale_0, xb, weight, best_idx);
            auto [dp, score_p] = quantizer.find_best_scale(xb, weight, best_idx);
            quantizer.find_best_match(-amax/scale_0, xb, weight, best_idx + Q::kNg);
            auto [dm, score_m] = quantizer.find_best_scale(xb, weight, best_idx + Q::kNg);

            auto idx = best_idx;
            if (score_p > score_m) scales[ib] = dp;
            else {
                scales[ib] = dm; idx += Q::kNg;
            }
            for (int ig = 0; ig < Q::kNg; ++ig) all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig] = idx[ig];

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    if (!max_scale) {
        *dptr = 0;
        return;
    }

    float d = max_scale/iq4k_values[0];
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry + iq4k_values[0])/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = best_index_iq4nl(iq4k_values, id*scales[ib]);
                float dl = iq4k_values[ls];
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto qb = quantizer.values() + Q::kGroupSize*all_idx[(ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize + ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float q = dl*qb[j];
                        sumqx += wb[jj]*xb[jj]*q;
                        sumq2 += wb[jj]*q*q;
                    }
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
            int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d;
    if (!d) return;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            auto qs = (uint16_t *)y[ibl].ql;
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                int ls = iq4k_values[(y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf];
                float dl = d*ls;
                quantizer.find_best_match(dl, xb, weight, best_idx);

                auto prev_idx = all_idx + (ibl*Q::kSuperBlockSize + ib*Q::kBlockSize)/Q::kGroupSize;

                float mse1 = 0, mse2 = 0;
                for (int ig = 0; ig < Q::kNg; ++ig) {
                    auto q1 = quantizer.values() + Q::kGroupSize*prev_idx[ig];
                    auto q2 = quantizer.values() + Q::kGroupSize*best_idx[ig];
                    for (int j = 0; j < Q::kGroupSize; ++j) {
                        int jj = ig*Q::kGroupSize + j;
                        float diff1 = xb[jj] - dl*q1[j];
                        float diff2 = xb[jj] - dl*q2[j];
                        mse1 += weight[jj]*diff1*diff1;
                        mse2 += weight[jj]*diff2*diff2;
                    }
                }
                if (mse1 < mse2) {
                    for (int ig = 0; ig < Q::kNg; ++ig) best_idx[ig] = prev_idx[ig];
                } else {
                    for (int ig = 0; ig < Q::kNg; ++ig) prev_idx[ig] = best_idx[ig];
                }

                for (int j = 0; j < Q::kNg; ++j) {
                    qs[j] = best_idx[j];
                    auto xl = xb + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
                qs += Q::kNg;
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d;
            if (!d) return;
        } else {
            break;
        }

        if (false) {
            for (int ibl = 0; ibl < nblock; ++ibl) {
                const float * xbl = x + ibl*Q::kSuperBlockSize;
                auto scales = all_scales + ibl*Q::kNblock;
                auto qs = (uint16_t *)y[ibl].ql;
                for (int ib = 0; ib < Q::kNblock; ++ib) {
                    const float * xb = xbl + Q::kBlockSize*ib;
                    const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                    for (int j = 0; j < Q::kNg; ++j) best_idx[j] = qs[ib*Q::kNg+j];
                    auto pair = quantizer.find_best_scale(xb, weight, best_idx);
                    scales[ib] = pair.first;
                }
            }
            float id = d ? 1/d : 0.f;
            for (int ibl = 0; ibl < nblock; ++ibl) {
                auto scales = all_scales + ibl*Q::kNblock;
                for (int ib = 0; ib < Q::kNblock/2; ++ib) {
                    int ls1 = best_index_iq4nl(iq4k_values, id*scales[ib]);
                    int ls2 = best_index_iq4nl(iq4k_values, id*scales[ib + Q::kNblock/2]);
                    y[ibl].scales[ib] = ls1 | (ls2 << 4);
                }
            }
        }

    }

}


void quantize_row_iq3_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales,
        float * all_weights, float * qtmp) {

    constexpr float kSigmaScale = 2.0f;
    constexpr float kStep = 8.0f;

    using Q = QuantizerIQ3KT;

    static_assert(Q::kNumVal%8 == 0);

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    float * dptr = (float *)vy;

    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);

    int   best_idx[2*Q::kNg];

    auto& quantizer = iq3kt_quantizer();

    int nblock = n_per_row / Q::kSuperBlockSize;

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (!amax_row) {
        *dptr = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq3_kt));
        return;
    }

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_scale = 0, max_scale = 0;

    float xaux[Q::kBlockSize];

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_kt));

        auto scales = all_scales + ibl*Q::kNblock;
        auto xbl = x + ibl*Q::kSuperBlockSize;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * xb = xbl + Q::kBlockSize*ib;
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                float ax = std::abs(xb[j]);
                xaux[j] = ax;
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0.0f;
                continue;
            }

            //quantizer.find_best_match(amax/96.f, xaux, weight, best_idx+Q::kNg);
            //scales[ib] = quantizer.find_best_scale(xaux, weight, best_idx+Q::kNg).first;

            float scale_0 = std::max(84.f, 123.f*amax/amax_row);
            //float scale_0 = std::max(64.f, 123.f*amax/amax_row);
            float best = 0;
            bool found_solution = false;
            for (int itry = -3; itry <= 3; ++itry) {
                quantizer.find_best_match(amax/(scale_0 + kStep*itry), xaux, weight, best_idx);
                auto [d, score] = quantizer.find_best_scale(xaux, weight, best_idx);
                if (score > best) {
                    best = score;
                    found_solution = true;
                    scales[ib] = d;
                    std::memcpy(best_idx+Q::kNg, best_idx, Q::kNg*sizeof(int));
                }
            }
            if (!found_solution) {
                fprintf(stderr, "======================= %s: failed to find solution for a block\n", __func__);
                fprintf(stderr, "Model weights and importances:\n");
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    fprintf(stderr, "%2d  %g  %g\n", j, xaux[j], weight[j]);
                }
                GGML_ASSERT(false);
            }

            auto xt = qtmp + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                auto q = quantizer.values() + Q::kGroupSize*best_idx[Q::kNg+ig];
                for (int j = 0; j < Q::kGroupSize; ++j) *xt++ = q[j];
            }

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    GGML_ASSERT(max_scale >= 0);
    float d = max_scale/15;
    float best = 0;
    for (int itry = -9; itry <= 9; ++itry) {
        float id = (itry*0.2f + 15)/max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl*Q::kSuperBlockSize;
            const float * qb = qtmp + ibl*Q::kSuperBlockSize;
            const float * wb = all_weights + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;
            for (int ib = 0; ib < Q::kNblock; ++ib) {
                int ls = nearest_int(id*scales[ib]);
                ls = std::max(0, std::min(15, ls));
                float dl = ls;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    float q = dl*qb[j];
                    sumqx += wb[j]*std::abs(xb[j])*q;
                    sumq2 += wb[j]*q*q;
                }
                xb += Q::kBlockSize;
                wb += Q::kBlockSize;
                qb += Q::kBlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
            d = sumqx/sumq2; best = d*sumqx;
        }
    }

    float id = d ? 1/d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto scales = all_scales + ibl*Q::kNblock;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            int ls1 = nearest_int(id*scales[ib]);
            int ls2 = nearest_int(id*scales[ib + Q::kNblock/2]);
            ls1 = std::max(0, std::min(15, ls1));
            ls2 = std::max(0, std::min(15, ls2));
            y[ibl].scales[ib] = ls1 | (ls2 << 4);
        }
    }

    *dptr = d;

    for (int iloop = 0; iloop < 1; ++iloop) {

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            uint16_t * ql = (uint16_t *)y[ibl].ql;

            std::memset(y[ibl].qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                const float * xb = xbl + Q::kBlockSize*ib;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) {
                    xaux[j] = std::abs(xb[j]);
                    if (xb[j] < 0) y[ibl].qh[j] |= (1 << ib);
                }
                int ls = (y[ibl].scales[ib%(Q::kNblock/2)] >> 4*(ib/(Q::kNblock/2))) & 0xf;
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    ql[ib*Q::kNg+j] = best_idx[j];
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + best_idx[j]*Q::kGroupSize;
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            *dptr = d;
            if (!d) break;
        } else {
            break;
        }
    }
}


void quantize_row_iq4_kt_impl(const float * x, void * vy, int n_per_row, const float * quant_weights, float * all_scales, float * all_weights) {

    constexpr float kSigmaScale = 2.0f;
    constexpr int kNtry = 2;
    using Q = QuantizerIQ4KT;

    static_assert(Q::kNumVal%8 == 0);

    float * dptr = (float *)vy;

    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);

    auto& quantizer1 = iq4kt_quantizer();
    auto& quantizer2 = iq4kt_quantizer(true);

    int nblock = n_per_row / Q::kSuperBlockSize;

    Q::set_weights(kSigmaScale, nblock, x, quant_weights, all_weights);

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
    if (!amax_row) {
        dptr[0] = 0.f;
        std::memset(y, 0, nblock*sizeof(block_iq4_kt));
        return;
    }

    int   best_idx[2*Q::kNg];
    float xaux[Q::kBlockSize];

    float amax_scale = 0, max_scale = 0;

    for (int ibl = 0; ibl < nblock; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq4_kt));

        const float * xbl = x + ibl*Q::kSuperBlockSize;
        auto scales = all_scales + ibl*Q::kNblock;

        for (int ib = 0; ib < Q::kNblock; ++ib) {
            const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
            float amax = 0;
            for (int j = 0; j < Q::kBlockSize; ++j) {
                xaux[j] = xbl[ib*Q::kBlockSize+j];
                float ax = std::abs(xaux[j]);
                amax = std::max(amax, ax);
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale_0 = std::max(90.f, 124.f*amax/amax_row);
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer1.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp;
                }
                quantizer1.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer1.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm;
                }
            }

            quantizer2.find_best_match(scales[ib], xaux, weight, best_idx);
            auto [d, score] = quantizer2.find_best_scale(xaux, weight, best_idx);
            if (score > best) {
                scales[ib] = d;
                y[ibl].qs[ib] = 1;
            }
            bool with_offset = false;
            for (int itry = -kNtry; itry <= kNtry; ++itry) {
                quantizer2.find_best_match( amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dp, score_p] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_p > best) {
                    best = score_p; scales[ib] = dp; with_offset = true;
                }
                quantizer2.find_best_match(-amax/(8.f*itry + scale_0), xaux, weight, best_idx);
                auto [dm, score_m] = quantizer2.find_best_scale(xaux, weight, best_idx);
                if (score_m > best) {
                    best = score_m; scales[ib] = dm; with_offset = true;
                }
            }
            if (with_offset) y[ibl].qs[ib] = 1;

            float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) {
                amax_scale = abs_scale;
                max_scale = scales[ib];
            }
        }

    }

    float d = -max_scale/64;

    dptr[0] = d;
    if (!d) return;

    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;

    for (int iloop = 0; iloop < 1; ++iloop) {

        const float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {

            // high 3 bits + scales
            // each block of 32 needs 8 x 3 (high bits) + 1 x 8 (scale) = 32 bits = 1 x uint32_t
            // we have 8 blocks
            auto shb = y[ibl].qs;  // high 3 bits + scales
            auto ql = (uint8_t *)(shb + Q::kNblock);
            auto qh = ql + kNumGroups;
            std::memset(qh, 0, kNumGroups/2);
            const float * xbl = x + ibl*Q::kSuperBlockSize;
            auto scales = all_scales + ibl*Q::kNblock;

            for (int ib = 0; ib < Q::kNblock; ++ib) {
                auto& quantizer = y[ibl].qs[ib] & 1 ? quantizer2 : quantizer1;
                const float * weight = all_weights + ibl*Q::kSuperBlockSize + ib*Q::kBlockSize;
                for (int j = 0; j < Q::kBlockSize; ++j) xaux[j] = xbl[ib*Q::kBlockSize+j];
                int ls = nearest_int(id*scales[ib]);
                ls = std::min(ls, 63);
                *(uint8_t *)(shb + ib) = ((ls + 64) << 1) | (shb[ib] & 1);
                float dl = d*ls;
                quantizer.find_best_match(dl, xaux, weight, best_idx);

                for (int j = 0; j < Q::kNg; ++j) {
                    shb[ib] |= ((best_idx[j] >> 12) << (8 + 3*j));
                    ql[Q::kNg*ib + j] = best_idx[j] & 255;
                    qh[(Q::kNg*ib + j)%(kNumGroups/2)] |= ((best_idx[j] >> 8) & 0xf) << 4*((Q::kNg*ib + j)/(kNumGroups/2));
                    auto xl = xaux + Q::kGroupSize*j;
                    auto wl = weight + Q::kGroupSize*j;
                    auto ql = quantizer.values() + Q::kGroupSize*best_idx[j];
                    for (int k = 0; k < Q::kGroupSize; ++k) {
                        float q = ql[k]*ls;
                        sumqx += wl[k]*xl[k]*q;
                        sumq2 += wl[k]*q*q;
                    }
                }
            }
        }
        if (sumq2 > 0) {
            d = sumqx/sumq2;
            dptr[0] = d;
            if (!d) break;
        } else {
            break;
        }
    }
}
}


extern "C" {
size_t quantize_iq1_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ1KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    std::vector<int> idx(n_per_row/QuantizerIQ1KT::kGroupSize);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq1_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}
void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq1_kt(x, (void *)y, 1, k, nullptr);
}
void dequantize_row_iq1_kt(const block_iq1_kt * x, float * y, int64_t k) {
    assert(k % QuantizerIQ1KT::kSuperBlockSize == 0);
    using Q = QuantizerIQ1KT;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq1_kt *)(dptr + 1);
    auto& deq = iq1kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            float sl = d * iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < Q::kNg; ++ig) {
                uint16_t idx = x[ibl].ql[ib*Q::kNg + ig] | ((x[ibl].qh[(ib%(Q::kNblock/2))*Q::kNg + ig] << (8 - 4*(ib/(Q::kNblock/2)))) & 0xf00);
                idx |= (x[ibl].sh[ib] << (8 - ig) & 0x1000);
                deq.set_values(idx, y, sl);
                y += Q::kGroupSize;
            }
        }
    }
}


size_t quantize_iq2_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ2KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    std::vector<int> idx(n_per_row/QuantizerIQ2KT::kGroupSize);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq2_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), idx.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}
void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kt(x, (void *)y, 1, k, nullptr);
}
void dequantize_row_iq2_kt(const block_iq2_kt * x, float * y, int64_t k) {
    assert(k % QuantizerIQ2KT::kSuperBlockSize == 0);
    const int nb = k / QuantizerIQ2KT::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * QuantizerIQ2KT::kScale;
    x = (const block_iq2_kt *)(dptr + 1);
    auto& deq = iq2kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*QuantizerIQ2KT::kSuperBlockSize;
        auto yh = yl + QuantizerIQ2KT::kSuperBlockSize/2;
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        const uint16_t * qh = ql + QuantizerIQ2KT::kNg*QuantizerIQ2KT::kNblock/2;
        for (int ib = 0; ib < QuantizerIQ2KT::kNblock/2; ++ib) {
            float sl = d * iq4k_values[x[ibl].scales[ib] & 0xf];
            float sh = d * iq4k_values[x[ibl].scales[ib] >>  4];
            for (int ig = 0; ig < QuantizerIQ2KT::kNg; ++ig) {
                deq.set_values(ql[ig], yl, sl);
                deq.set_values(qh[ig], yh, sh);
                yl += QuantizerIQ2KT::kGroupSize;
                yh += QuantizerIQ2KT::kGroupSize;
            }
            ql += QuantizerIQ2KT::kNg;
            qh += QuantizerIQ2KT::kNg;
        }
    }
}


size_t quantize_iq3_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ3KT::kBlockSize);
    std::vector<float> weights(n_per_row), xtmp(n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq3_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data(), xtmp.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}
void quantize_row_iq3_kt_ref(const float * x, block_iq3_kt * y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_kt(x, (void *)y, 1, k, nullptr);
}
void dequantize_row_iq3_kt(const block_iq3_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ3KT;
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    assert(k % Q::kSuperBlockSize == 0);
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = *dptr * Q::kScale;
    x = (const block_iq3_kt *)(dptr + 1);
    auto& deq = iq3kt_quantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto yl = y + ibl*Q::kSuperBlockSize;
        auto yh = yl + Q::kSuperBlockSize/2;
        auto qll = (const uint16_t *)x[ibl].ql;
        auto qlh = qll + kNumGroups/2;
        int jj = 0;
        for (int ib = 0; ib < Q::kNblock/2; ++ib) {
            float sl = d * (x[ibl].scales[ib] & 0xf);
            float sh = d * (x[ibl].scales[ib] >>  4);
            uint8_t l_mask = 1 << ib;
            uint8_t h_mask = l_mask << (Q::kNblock/2);
            for (int ig = 0; ig < Q::kNg; ++ig) {
                deq.set_values(qll[jj], yl, sl);
                deq.set_values(qlh[jj], yh, sh);
                for (int j = 0; j < Q::kGroupSize; ++j) {
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & l_mask) yl[j] = -yl[j];
                    if (x[ibl].qh[ig*Q::kGroupSize+j] & h_mask) yh[j] = -yh[j];
                }
                yl += Q::kGroupSize;
                yh += Q::kGroupSize;
                ++jj;
            }
        }
    }
}


size_t quantize_iq4_kt(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_ASSERT(n_per_row%QK_K == 0);
    auto row_size = ggml_row_size(GGML_TYPE_IQ4_KT, n_per_row);
    std::vector<float> scales(n_per_row/QuantizerIQ4KT::kBlockSize);
    std::vector<float> weights(n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq4_kt_impl(src, (void *)qrow, n_per_row, imatrix, scales.data(), weights.data());
        src += n_per_row;
        qrow += row_size;
    }
    return nrows * row_size;
}
void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_kt(x, (void *)y, 1, k, nullptr);
}
void dequantize_row_iq4_kt(const block_iq4_kt * x, float * y, int64_t k) {
    using Q = QuantizerIQ4KT;
    assert(k % Q::kSuperBlockSize == 0);
    constexpr int kNumGroups = Q::kSuperBlockSize/Q::kGroupSize;
    const int nb = k / Q::kSuperBlockSize;
    const float * dptr = (const float *)x;
    const float d = dptr[0] * Q::kScale;
    x = (const block_iq4_kt *)(dptr + 1);
    auto& deq = iq4kt_dequantizer();
    for (int ibl = 0; ibl < nb; ++ibl) {
        auto shb = x[ibl].qs;
        auto ql = (const uint8_t *)(shb + Q::kNblock);
        auto qh = ql + kNumGroups;
        for (int ib = 0; ib < Q::kNblock; ++ib) {
            int offset = shb[ib] & 1 ? 32768 + 4096 : 4096;
            int ls = int((shb[ib] & 0xff) >> 1) - 64;
            float sl = d * ls;
            for (int ig = 0; ig < Q::kNg; ++ig) {
                int jj = ib*Q::kNg+ig;
                uint16_t idx = ql[jj] | ((qh[jj%(kNumGroups/2)] << (8 - 4*(jj/(kNumGroups/2)))) & 0xf00) | (((shb[ib] >> (8 + 3*ig)) & 7) << 12);
                deq.set_values(idx, y, sl, offset);
                y += Q::kGroupSize;
            }
        }
    }
}
}



extern "C" {
void dequantize_row_iq4_kss(const block_iq4_kss * x, float * y, int64_t k) {
    const float * dptr = (const float *)x;
    const float d = *dptr;
    x = (const block_iq4_kss *)(dptr + 1);
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    for (int ibl = 0; ibl < k/QK_K; ++ibl) {
        auto qs = (const uint16_t *)x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int k = 0; k < 8; ++k) {
                aux16[k] = qs[k] & 0xfffe;
                aux16[k] ^= (aux16[k] >> 1);
                ls |= (qs[k] & 1) << k;
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            float dl = d * ((ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * values[aux8[j] & 0xf];
                y[j+16] = dl * values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}

void dequantize_row_iq2_ks(const block_iq2_ks  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_ks *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (((x[i].scales[ib64] & 0xf) | ((extra >> 4) & 0x10)) - 16);
            float dl2 = d * (((x[i].scales[ib64] >>  4) | ((extra >> 5) & 0x10)) - 16);
            const int8_t * values1 = extra & 1 ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = extra & 2 ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 32; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j] >> (shift+0)) & 3];
                y[j+32] = dl2 * values2[(qs[j] >> (shift+2)) & 3];
            }
            y += 64;
            shift += 4;
            if (shift == 8) { qs += 32; shift = 0; }
        }
    }
}

void dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    static_assert(QK_K/kBlockSize == 8);
    GGML_ASSERT(k%QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

void dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

void dequantize_row_iq5_ks(const block_iq5_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq5_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int ib64 = 0; ib64 < QK_K/(2*kBlockSize); ++ib64) {
            float dl1 = d * ((int)(x[ibl].scales[2*ib64+0] & 254) - 127);
            float dl2 = d * ((int)(x[ibl].scales[2*ib64+1] & 254) - 127);
            const int8_t * values1 = iq5nl_values + ((x[ibl].scales[2*ib64+0] & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((x[ibl].scales[2*ib64+1] & 1) << 5);
            for (int j = 0; j < kBlockSize; ++j) {
                y[j           ] = dl1 * values1[(qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)];
                y[j+kBlockSize] = dl2 * values2[(qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)];
            }
            y  += 2*kBlockSize;
            qs += kBlockSize;
        }
    }
}

void dequantize_row_iq2_kl(const block_iq2_kl  * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const ggml_half * dptr = (const ggml_half *)x;
    const float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq2_kl *)(dptr + 1);

    for (int i = 0; i < nb; i++) {

        auto qs = x[i].qs;
        auto qh = x[i].qh;
        auto scales_h = x[i].scales_h;

        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (int(((x[i].scales_l[(2*ib64+0)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+0)) & 3) << 4)) - 32);
            float dl2 = d * (int(((x[i].scales_l[(2*ib64+1)%4] >> 4*(ib64/2)) & 0xf) | (((scales_h >> (4*ib64+2)) & 3) << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                const int8_t * val1 = (const int8_t *)(iq2kl_values + ((qs[j] & 0xf) | (((qh[j] >> (2*ib64+0)) & 1) << 4)));
                const int8_t * val2 = (const int8_t *)(iq2kl_values + ((qs[j] >>  4) | (((qh[j] >> (2*ib64+1)) & 1) << 4)));
                y[2*j+ 0] = dl1 * val1[0];
                y[2*j+ 1] = dl1 * val1[1];
                y[2*j+32] = dl2 * val2[0];
                y[2*j+33] = dl2 * val2[1];
            }
            y  += 64;
            qs += 16;
        }

    }
}
}

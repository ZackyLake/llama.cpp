
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require

uint32_t test_bit_to_mask(uint32_t dat, int idx) {
    return uint32_t(bitfieldExtract(int32_t(dat), idx, 1));
}

uint32_t test_bit_to_get_bit(uint32_t dat, int idx_test, int idx_get) {
    const uint32_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    return dat_shift & (1u << idx_get);
}
uint16_t test_bit_to_get_bit(uint16_t dat, int idx_test, int idx_get) {
    const uint16_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    return dat_shift & (uint16_t(1) << idx_get);
}
uint8_t test_bit_to_get_bit(uint8_t dat, int idx_test, int idx_get) {
    const uint8_t dat_shift = idx_test == idx_get ? dat : 
        (idx_test < idx_get ? dat << (idx_get - idx_test) : dat >> (idx_test - idx_get));
    // intermediates are free to be promoted to i16
    return dat_shift & (uint8_t(1) << idx_get);
}

uint8_t test_bit_to_get_high4(uint32_t dat, int idx, uint8_t val) {
    const uint32_t shift = test_bit_to_get_bit(dat, idx, 2);
    return uint8_t(bitfieldExtract(uint32_t(val), int32_t(shift), 4));
    // const uint32_t dat_shift = idx == 2 ? dat : (idx < 2 ? dat << (2 - idx) : dat >> (idx - 2));
    // return uint8_t(bitfieldExtract(uint32_t(val), int32_t(dat_shift & 4u), 4));
    // return (val >> (dat_shift & 4u)) & uint8_t(0x0F);
}

uint16_t shift_to_get_high4(uint16_t shift, uint16_t val) {
    return (val >> shift) & uint16_t(0x0F);
    // return uint16_t(bitfieldExtract(uint32_t(val), shift, 4));
}
uint8_t shift_to_get_high4(uint16_t shift, uint8_t val) {
    return (val >> shift) & uint8_t(0x0F);
    // return uint8_t(bitfieldExtract(uint32_t(val), shift, 4));
}

uint16_t test_bit_to_get_high4(uint16_t dat, int idx, uint16_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}
uint8_t test_bit_to_get_high4(uint16_t dat, int idx, uint8_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);

    // arithmetic/logic execution dtype is at least 16bit and src1 does not accept byte
    // LLVM's shift uses same dtype for all srcs, so promote val to u16
}

uint16_t test_bit_to_get_high4(uint8_t dat, int idx, uint16_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}
uint8_t test_bit_to_get_high4(uint8_t dat, int idx, uint8_t val) {
    const uint16_t shift = test_bit_to_get_bit(dat, idx, 2);
    return shift_to_get_high4(shift, val);
}

#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ2_KS)

#if 1

shared uint32_t iq2k_table[512];
const uint kpacked_iq2_k_lo = 0x1101F3E1u; // -31, -13, 1, 17
const uint kpacked_iq2_k_hi = 0x1606F8E6u; // -26,  -8, 6, 22

void init_iq2k_table() {
    const uint shift1 = (gl_SubgroupInvocationID << 3u) & 0x18u;
    const uint shift2 = (gl_SubgroupInvocationID << 1u) & 0x18u;
    const uint8_t val1_lo = uint8_t(kpacked_iq2_k_lo >> shift1);
    const uint8_t val2_lo = uint8_t(kpacked_iq2_k_lo >> shift2);
    const uint8_t val1_hi = uint8_t(kpacked_iq2_k_hi >> shift1);
    const uint8_t val2_hi = uint8_t(kpacked_iq2_k_hi >> shift2);
    const uint16_t base_lo_u16 = pack16(i8vec2(val1_lo, val2_lo));
    const uint16_t base_hi_u16 = pack16(i8vec2(val1_hi, val2_hi));
    const uint32_t base_lo = uint32_t(base_lo_u16) << 8u;
    const uint32_t base_hi = uint32_t(base_hi_u16) << 8u;
    // assume only gl_WorkGroupSize.x and 256 being multiplies of 4*it.
    for (uint index_ = (gl_SubgroupInvocationID + gl_SubgroupSize * gl_SubgroupID) * 4u;
         index_ < 256u; index_ += gl_WorkGroupSize.x * 4u) {

        const uint shift3 = (index_ >> 3u) & 0x18u;

        const uint32_t pack_lo = base_lo | ((kpacked_iq2_k_lo << (24u - shift3)) & 0xff000000u);
        const uint32_t pack_hi = base_hi | ((kpacked_iq2_k_hi << (24u - shift3)) & 0xff000000u);

        iq2k_table[index_] = pack_lo + uint8_t(kvalues_iq2_k_const[0]);
        iq2k_table[256u + index_] = pack_hi + uint8_t(kvalues_iq2_k_const[4]);

        iq2k_table[index_ + 1u] = pack_lo + uint8_t(kvalues_iq2_k_const[1]);
        iq2k_table[256u + index_ + 1u] = pack_hi + uint8_t(kvalues_iq2_k_const[5]);

        iq2k_table[index_ + 2u] = pack_lo + uint8_t(kvalues_iq2_k_const[2]);
        iq2k_table[256u + index_ + 2u] = pack_hi + uint8_t(kvalues_iq2_k_const[6]);

        iq2k_table[index_ + 3u] = pack_lo + uint8_t(kvalues_iq2_k_const[3]);
        iq2k_table[256u + index_ + 3u] = pack_hi + uint8_t(kvalues_iq2_k_const[7]);
    }
    barrier();
}

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    const uint table_offset = is_hi_table ? 256u : 0u;
    const uint index = table_offset + dotPacked4x8EXT(values, 0x40100401u);
    return int32_t(iq2k_table[index]);
}

#elif 0

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_subgroup_extended_types_int8 : require

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    if (is_hi_table) values += 0x04040404u;
    const u8vec4 indexes = unpack8(values);
    const int8_t lut = gl_SubgroupInvocationID < 8
        ? kvalues_iq2_k_const[gl_SubgroupInvocationID]
        : int8_t(0);
    return pack32(i8vec4(subgroupShuffle(lut, indexes.x),
                         subgroupShuffle(lut, indexes.y),
                         subgroupShuffle(lut, indexes.z),
                         subgroupShuffle(lut, indexes.w)));
}

#elif 0

int32_t unpack_iq2_k(uint32_t values, bool is_hi_table) {
    if (is_hi_table) values += 0x04040404u;
    const u8vec4 indexes = unpack8(values);
    return pack32(i8vec4(kvalues_iq2_k[indexes.x],
                         kvalues_iq2_k[indexes.y],
                         kvalues_iq2_k[indexes.z],
                         kvalues_iq2_k[indexes.w]));
}

#endif

#endif

#if defined(DATA_A_IQ3_K) || defined(DATA_A_IQ3_KS)

shared uint16_t iq3k_table[128];
const uint32_t kpacked32_iq3_k_lo = 0xF6E9D8C1u;
const uint32_t kpacked32_iq3_k_hi = 0x2F1C0D01u;

void init_iq3k_table() {
#define WG0_INIT 0
#if WG0_INIT
    if (gl_SubgroupID == 0) {
#endif
    const uint laneid = gl_SubgroupInvocationID;
    const uint index = laneid * 4u;
    const i8vec4 low = i8vec4(unpack8(bool(laneid & 1u) ? kpacked32_iq3_k_hi : kpacked32_iq3_k_lo));
    const uint high_target4 = bool(laneid & 8u) ? kpacked32_iq3_k_hi : kpacked32_iq3_k_lo;
    const uint high_target_shift = ((laneid / 2u) % 4u) * 8u;
    const int8_t high = int8_t(high_target4 >> high_target_shift);
    const i8vec4 low_hi = low + i8vec4(4);
    const int8_t high_hi = high + int8_t(4);
    if (laneid < 16u) {
        iq3k_table[index     ] = uint16_t(pack16(i8vec2(low.x, high)));
        iq3k_table[index + 1u] = uint16_t(pack16(i8vec2(low.y, high)));
        iq3k_table[index + 2u] = uint16_t(pack16(i8vec2(low.z, high)));
        iq3k_table[index + 3u] = uint16_t(pack16(i8vec2(low.w, high)));
        iq3k_table[64u + index     ] = uint16_t(pack16(i8vec2(low_hi.x, high_hi)));
        iq3k_table[64u + index + 1u] = uint16_t(pack16(i8vec2(low_hi.y, high_hi)));
        iq3k_table[64u + index + 2u] = uint16_t(pack16(i8vec2(low_hi.z, high_hi)));
        iq3k_table[64u + index + 3u] = uint16_t(pack16(i8vec2(low_hi.w, high_hi)));
    }
#if WG0_INIT
    }
    subgroupBarrier();
#endif
}

int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, bool is_hi_table) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                         (((qh >> shift_h) & 0x01010101u) << 2u);
    const uint table_offset = is_hi_table ? 64u : 0u;
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}

int32_t unpack_iq3_k(uint32_t ql, uint32_t qh, uint16_t shift_h, uint16_t extra_bit01, int hi_table_bit_idx) {
    const uint16_t shift_l = (shift_h & uint16_t(3)) << 1;
    const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                         (((qh >> shift_h) & 0x01010101u) << 2u);
    const uint table_offset = test_bit_to_get_bit(extra_bit01, hi_table_bit_idx, 6);
    const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u); // WA for intel bug
    const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
    return int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
}

#endif

#if defined(DATA_A_IQ2_KL)

shared uint16_t iq2kl_table[64];

void init_iq2kl_table() {
    if (gl_LocalInvocationIndex < 16u) {
        const i8vec4 values = kvalues_iq2_kl_const[gl_LocalInvocationIndex];
        const uint index = 4u * gl_LocalInvocationIndex;
        iq2kl_table[index     ] = pack16(values.xy);
        iq2kl_table[index + 1u] = pack16(values.yx);
        iq2kl_table[index + 2u] = pack16(values.zw);
        iq2kl_table[index + 3u] = pack16(values.wz);
    }
    barrier();
}

#endif

#if defined(DATA_A_IQ2_K)
void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint8_t qs_idx = bool(ib32 & uint8_t(4)) ? uint8_t(8) : uint8_t(0);
    const uint8_t ib32_shift = ib32 << 1;
    const uint8_t shift = ib32_shift & uint8_t(6);
    const bool is_hi_table0 = bool((data_a[ib_k].extra >> ib32_shift) & uint16_t(1));
    const bool is_hi_table1 = bool((data_a[ib_k].extra >> (ib32_shift + uint8_t(1))) & uint16_t(1));

    qs_a[0] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx    ] >> shift) & 0x03030303, is_hi_table0);
    qs_a[1] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 1] >> shift) & 0x03030303, is_hi_table0);
    qs_a[2] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 2] >> shift) & 0x03030303, is_hi_table0);
    qs_a[3] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 3] >> shift) & 0x03030303, is_hi_table0);
    qs_a[4] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 4] >> shift) & 0x03030303, is_hi_table1);
    qs_a[5] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 5] >> shift) & 0x03030303, is_hi_table1);
    qs_a[6] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 6] >> shift) & 0x03030303, is_hi_table1);
    qs_a[7] = unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 7] >> shift) & 0x03030303, is_hi_table1);
    const uint8_t scale = data_a[ib_k].scales[ib32];
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(int32_t(scale & uint8_t(0x0F)) - 8),
                    d * float(int32_t(scale >> uint8_t(4)) - 8));
}
#endif

#if defined(DATA_A_IQ3_K)
void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint q_idx = bool(ib32 & uint8_t(4)) ? 16u : 0u;
    const uint8_t ib32_shift = ib32 << 1;
    const uint16_t extra_bit01 = data_a[ib_k].extra >> ib32_shift;

    const uint ql0 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  0], data_a_packed16[ib_k].qs[q_idx +  1]));
    const uint ql1 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  2], data_a_packed16[ib_k].qs[q_idx +  3]));
    const uint ql2 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  4], data_a_packed16[ib_k].qs[q_idx +  5]));
    const uint ql3 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  6], data_a_packed16[ib_k].qs[q_idx +  7]));
    const uint ql4 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx +  8], data_a_packed16[ib_k].qs[q_idx +  9]));
    const uint ql5 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 10], data_a_packed16[ib_k].qs[q_idx + 11]));
    const uint ql6 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 12], data_a_packed16[ib_k].qs[q_idx + 13]));
    const uint ql7 = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 14], data_a_packed16[ib_k].qs[q_idx + 15]));
    const uint qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 0], data_a_packed16[ib_k].qh[ 1]));
    const uint qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 2], data_a_packed16[ib_k].qh[ 3]));
    const uint qh2 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 4], data_a_packed16[ib_k].qh[ 5]));
    const uint qh3 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 6], data_a_packed16[ib_k].qh[ 7]));
    const uint qh4 = pack32(u16vec2(data_a_packed16[ib_k].qh[ 8], data_a_packed16[ib_k].qh[ 9]));
    const uint qh5 = pack32(u16vec2(data_a_packed16[ib_k].qh[10], data_a_packed16[ib_k].qh[11]));
    const uint qh6 = pack32(u16vec2(data_a_packed16[ib_k].qh[12], data_a_packed16[ib_k].qh[13]));
    const uint qh7 = pack32(u16vec2(data_a_packed16[ib_k].qh[14], data_a_packed16[ib_k].qh[15]));

    qs_a[0] = unpack_iq3_k(ql0, qh0, uint16_t(ib32), extra_bit01, 0);
    qs_a[1] = unpack_iq3_k(ql1, qh1, uint16_t(ib32), extra_bit01, 0);
    qs_a[2] = unpack_iq3_k(ql2, qh2, uint16_t(ib32), extra_bit01, 0);
    qs_a[3] = unpack_iq3_k(ql3, qh3, uint16_t(ib32), extra_bit01, 0);
    qs_a[4] = unpack_iq3_k(ql4, qh4, uint16_t(ib32), extra_bit01, 1);
    qs_a[5] = unpack_iq3_k(ql5, qh5, uint16_t(ib32), extra_bit01, 1);
    qs_a[6] = unpack_iq3_k(ql6, qh6, uint16_t(ib32), extra_bit01, 1);
    qs_a[7] = unpack_iq3_k(ql7, qh7, uint16_t(ib32), extra_bit01, 1);

    const uint16_t scale_l = uint16_t(data_a[ib_k].scales_l[ib32]);
    const uint16_t sign01 = data_a[ib_k].scales_h >> ib32_shift;
    const bool sign0 = bool(sign01 & uint16_t(1));
    const bool sign1 = bool(sign01 & uint16_t(2));
    const int16_t scale0 = int16_t((scale_l & uint16_t(0x0F)) * uint16_t(2) + uint16_t(1));
    const int16_t scale1 = int16_t((scale_l >> uint16_t(3)) | uint16_t(1));
    const float d = data_a[ib_k].d;
    a_scales = vec2(d * float(sign0 ? -scale0 : scale0),
                    d * float(sign1 ? -scale1 : scale1));
}
#endif

#if defined(DATA_A_IQ4_K)
int32_t unpack_iq4_k(uint32_t values, uint shift, uint table_offset) {
    const u8vec4 indexes = unpack8((values >> shift) & 0x0F0F0F0F);
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
}

void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 4 * ib32_u;
    const uint table_offset0 = 16 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u)) & 1);
    const uint table_offset1 = 16 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u + 1)) & 1);

    qs_a[0] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], 0, table_offset0);
    qs_a[1] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], 0, table_offset0);
    qs_a[2] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], 0, table_offset0);
    qs_a[3] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], 0, table_offset0);
    qs_a[4] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], 4, table_offset1);
    qs_a[5] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], 4, table_offset1);
    qs_a[6] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], 4, table_offset1);
    qs_a[7] = unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], 4, table_offset1);
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32_u / 2]) >> (4 * (ib32_u % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
    const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(scale0), d * float(scale1));
}
#endif

#if defined(DATA_A_IQ5_K)
int32_t unpack_iq5_k(uint32_t values, uint32_t qh, uint shift, uint table_offset) {
    const uint32_t low = (values >> (4 * (shift & 1))) & 0x0F0F0F0Fu;
    const uint32_t high = ((qh >> shift) & 0x01010101u) << 4u;
    const u8vec4 indexes = unpack8(low | high);
    return pack32(i8vec4(kvalues_iq5_k[indexes.x + table_offset],
                         kvalues_iq5_k[indexes.y + table_offset],
                         kvalues_iq5_k[indexes.z + table_offset],
                         kvalues_iq5_k[indexes.w + table_offset]));
}

void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 8 * (ib32_u / 2);
    const uint qh_idx = 0;
    const uint table_offset0 = 32 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u)) & 1);
    const uint table_offset1 = 32 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u + 1)) & 1);

    qs_a[0] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], ib32_u, table_offset0);
    qs_a[1] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], ib32_u, table_offset0);
    qs_a[2] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], ib32_u, table_offset0);
    qs_a[3] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], ib32_u, table_offset0);
    qs_a[4] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 4], data_a_packed32[ib_k].qh[qh_idx + 4], ib32_u, table_offset1);
    qs_a[5] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 5], data_a_packed32[ib_k].qh[qh_idx + 5], ib32_u, table_offset1);
    qs_a[6] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 6], data_a_packed32[ib_k].qh[qh_idx + 6], ib32_u, table_offset1);
    qs_a[7] = unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 7], data_a_packed32[ib_k].qh[qh_idx + 7], ib32_u, table_offset1);
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32_u / 2]) >> (4 * (ib32_u % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale0 = int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32;
    const int32_t scale1 = int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(scale0), d * float(scale1));
}
#endif

#if defined(DATA_A_IQ6_K)
int32_t unpack_iq6_k(uint32_t values, uint32_t qh, uint shift, uint table_offset) {
    const uint32_t low = (values >> (4 * ((shift / 2) & 1))) & 0x0F0F0F0Fu;
    const uint32_t high = ((qh >> shift) & 0x03030303u) << 4u;
    const u8vec4 indexes = unpack8(low | high);
    return pack32(i8vec4(kvalues_iq6_k[indexes.x + table_offset],
                         kvalues_iq6_k[indexes.y + table_offset],
                         kvalues_iq6_k[indexes.z + table_offset],
                         kvalues_iq6_k[indexes.w + table_offset]));
}

void prepare_a_k32_iter(uint ib_k, uint8_t ib32, out int32_t qs_a[8], out vec2 a_scales) {
    const uint ib32_u = uint(ib32);
    const uint qs_idx = 8 * (ib32_u / 2);
    const uint qh_idx = 8 * (ib32_u / 4);
    const uint shift = 2 * (ib32_u % 4);
    const uint table_offset0 = 64 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u)) & 1);
    const uint table_offset1 = 64 * ((uint(data_a[ib_k].extra) >> (2 * ib32_u + 1)) & 1);

    qs_a[0] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], shift, table_offset0);
    qs_a[1] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], shift, table_offset0);
    qs_a[2] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], shift, table_offset0);
    qs_a[3] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], shift, table_offset0);
    qs_a[4] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 4], data_a_packed32[ib_k].qh[qh_idx + 4], shift, table_offset1);
    qs_a[5] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 5], data_a_packed32[ib_k].qh[qh_idx + 5], shift, table_offset1);
    qs_a[6] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 6], data_a_packed32[ib_k].qh[qh_idx + 6], shift, table_offset1);
    qs_a[7] = unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 7], data_a_packed32[ib_k].qh[qh_idx + 7], shift, table_offset1);
    const float d = float(data_a[ib_k].d);
    a_scales = vec2(d * float(data_a[ib_k].scales[2 * ib32_u]),
                    d * float(data_a[ib_k].scales[2 * ib32_u + 1]));
}
#endif

#if defined(DATA_A_IQ2_K)
i32vec4 repack4(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t qs_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(8) : uint8_t(0)) + (iqs << 2);
    const uint8_t ib32_shift = ib32 << 1;
    const uint8_t shift = ib32_shift & uint8_t(6);
    const bool is_hi_table = bool((data_a[ib_k].extra >> (ib32_shift + iqs)) & uint16_t(1));

    return i32vec4(unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx    ] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 1] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 2] >> shift) & 0x03030303, is_hi_table),
                   unpack_iq2_k((data_a_packed32[ib_k].qs[qs_idx + 3] >> shift) & 0x03030303, is_hi_table));
}

float get_d_scale(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t scale = data_a[ib_k].scales[ib32];
    return float(data_a[ib_k].d) * float(int32_t((scale >> (uint8_t(4) * iqs)) & uint8_t(0x0F)) - 8);
}

#endif

#if defined(DATA_A_IQ3_K)
i32vec4 repack4(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t qh_idx = iqs << 3;
    const uint8_t q_idx = (bool(ib32 & uint8_t(4)) ? uint8_t(16) : uint8_t(0)) + qh_idx;
    const bool is_hi_table = bool((data_a[ib_k].extra >> ((ib32 << 1) + iqs)) & uint16_t(1));

    i32vec4 result;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        const uint ql = pack32(u16vec2(data_a_packed16[ib_k].qs[q_idx + 2 * j], data_a_packed16[ib_k].qs[q_idx + 2 * j + 1]));
        const uint qh = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_idx + 2 * j], data_a_packed16[ib_k].qh[qh_idx + 2 * j + 1]));
        result[j] = unpack_iq3_k(ql, qh, uint16_t(ib32), is_hi_table);
    }
    return result;
}

float get_d_scale(uint ib, uint8_t iqs) {
    const uint ib_k = ib / 8;
    const uint8_t ib32 = uint8_t(ib) & uint8_t(7);
    const uint8_t scale_l = (data_a[ib_k].scales_l[ib32] >> (uint8_t(4) * iqs)) & uint8_t(0x0F);
    const bool sign = bool((data_a[ib_k].scales_h >> ((ib32 << 1) + iqs)) & uint16_t(1));
    return float(data_a[ib_k].d) * float(2 * scale_l + 1) * (sign ? -1.0 : 1.0);
}

#endif

#if defined(DATA_A_IQ4_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 4 * ib32;
    const uint table_offset = 16 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + iqs)) & 1);
    const uint shift = 4 * iqs;

    return i32vec4(unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx    ], shift, table_offset),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 1], shift, table_offset),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 2], shift, table_offset),
                   unpack_iq4_k(data_a_packed32[ib_k].qs[qs_idx + 3], shift, table_offset));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale = iqs == 0 ? int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32
                                   : int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    return float(data_a[ib_k].d) * float(scale);
}

#endif

#if defined(DATA_A_IQ5_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 8 * (ib32 / 2) + 4 * iqs;
    const uint qh_idx = 4 * iqs;
    const uint table_offset = 32 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + iqs)) & 1);

    return i32vec4(unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], ib32, table_offset),
                   unpack_iq5_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], ib32, table_offset));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint scales_h = uint(data_a[ib_k].scales_h[ib32 / 2]) >> (4 * (ib32 % 2));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const int32_t scale = iqs == 0 ? int32_t((scales_l & 0x0F) | ((scales_h << 4) & 0x30)) - 32
                                   : int32_t((scales_l >> 4) | ((scales_h << 2) & 0x30)) - 32;
    return float(data_a[ib_k].d) * float(scale);
}

#endif

#if defined(DATA_A_IQ6_K)
i32vec4 repack4(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    const uint qs_idx = 8 * (ib32 / 2) + 4 * iqs;
    const uint qh_idx = 8 * (ib32 / 4) + 4 * iqs;
    const uint shift = 2 * (ib32 % 4);
    const uint table_offset = 64 * ((uint(data_a[ib_k].extra) >> (2 * ib32 + iqs)) & 1);

    return i32vec4(unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx    ], data_a_packed32[ib_k].qh[qh_idx    ], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 1], data_a_packed32[ib_k].qh[qh_idx + 1], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 2], data_a_packed32[ib_k].qh[qh_idx + 2], shift, table_offset),
                   unpack_iq6_k(data_a_packed32[ib_k].qs[qs_idx + 3], data_a_packed32[ib_k].qh[qh_idx + 3], shift, table_offset));
}

float get_d_scale(uint ib, uint iqs) {
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    return float(data_a[ib_k].d) * float(data_a[ib_k].scales[2 * ib32 + iqs]);
}

#endif


#if defined(SG_IQK)

#if defined(DATA_A_IQ2_K)
#define SG_IQK_LANES_PER_32       (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF        (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra_bit_idx = uint16_t(lane / (SUBGROUP_SIZE / 16u));
    const uint qs_u32 = data_a_packed32[ib_k].qs[lane % uint16_t(16)];

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const bool is_hi_table = bool((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1));

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint q2 = subgroupShuffle(qs_u32, qs_base + q32);
        qs_a[q] = unpack_iq2_k((q2 >> shift_l) & 0x03030303u, is_hi_table);
    }

    const uint8_t scale_l = data_a[ib_k].scales[ib32];
    const bool scale_high_half = bool(lane & (SG_IQK_LANES_PER_32 / 2u));
    a_scale = float(data_a[ib_k].d) * float(int32_t((scale_l >> (scale_high_half ? 4 : 0)) & uint8_t(0x0F)) - 8);
}
#endif

#if defined(DATA_A_IQ3_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_HALF       (QUANT_K_Q8_1 / 4)

// A 32-element IQ3K sub-block is split across SG_IQK_LANES_PER_32 lanes.
// Each lane owns SG_K elements and SG_QS_PER_ITER packed int32 values.
void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const uint16_t extra_bit_idx = uint16_t(lane / (SUBGROUP_SIZE / 16u));

    const uint qs_offset = 2u * (lane % uint16_t(16));
    const uint qs_u32 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_offset],
                                       data_a_packed16[ib_k].qs[qs_offset + 1u]));

    const uint qh_offset = 2u * (lane % uint16_t(8));
    const uint qh_u32 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_offset],
                                       data_a_packed16[ib_k].qh[qh_offset + 1u]));

    const uint16_t q_offset = uint16_t((lane * SG_QS_PER_ITER) % SG_IQK_QS_PER_HALF);
    const uint16_t qs_base = lane >= (SUBGROUP_SIZE / 2u) ? uint16_t(SG_IQK_QS_PER_HALF) : uint16_t(0);
    const uint16_t shift_h = uint16_t(ib32);
    const uint16_t shift_l = uint16_t(2u * (ib32 % 4u));
    const uint table_offset = ((data_a[ib_k].extra >> extra_bit_idx) & uint16_t(1)) << 6;

    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q32 = q_offset + q;
        const uint ql = subgroupShuffle(qs_u32, qs_base + q32);
        const uint qh = subgroupShuffle(qh_u32, q32);
        const uint indexes = ((ql >> shift_l) & 0x03030303u) |
                             (((qh >> shift_h) & 0x01010101u) << 2u);
        const uint index0 = table_offset + dotPacked4x8EXT(indexes & 0xffffu, 0x00000801u);
        const uint index1 = table_offset + dotPacked4x8EXT(indexes, 0x08010000u);
        qs_a[q] = int32_t(pack32(u16vec2(iq3k_table[index0], iq3k_table[index1])));
    }

    const uint16_t scale_l = uint16_t(data_a[ib_k].scales_l[ib32]);
    const bool scale_high_half = bool(lane & (SG_IQK_LANES_PER_32 / 2u));
    const uint16_t scale_l_shifted = scale_l >> (scale_high_half ? 4 : 0);
    const int16_t scale = int16_t((scale_l_shifted & uint16_t(0x0F)) * uint16_t(2) + uint16_t(1));
    const bool sign = bool((data_a[ib_k].scales_h >> extra_bit_idx) & uint16_t(1));
    a_scale = float(data_a[ib_k].d) * float(sign ? -scale : scale);
}
#endif

#if defined(DATA_A_IQ4_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 8)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const uint table_offset = ((uint(data_a[ib_k].extra) >> extra_bit_idx) & 1u) << 4u;
    const uint shift = scale_high_half ? 4u : 0u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        qs_a[q] = unpack_iq4_k(qs_pair.x, shift, table_offset);
        qs_a[q + 1u] = unpack_iq4_k(qs_pair.y, shift, table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 4u * uint(ib32) + uint(q_offset) + q;
        const uint q4 = subgroupShuffle(qs_u32, q_idx);
        qs_a[q] = unpack_iq4_k(q4, shift, table_offset);
    }
#endif

    const uint scales_h = uint(data_a[ib_k].scales_h[uint(ib32) / 2u]) >> (4u * (uint(ib32) % 2u));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const uint scale = scale_high_half
        ? ((scales_l >> 4u) | ((scales_h << 2u) & 0x30u))
        : ((scales_l & 0x0Fu) | ((scales_h << 4u) & 0x30u));
    a_scale = float(data_a[ib_k].d) * float(int32_t(scale) - 32);
}
#endif

#if defined(DATA_A_IQ5_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif
    const uint qh_u32 = data_a_packed32[ib_k].qh[uint(lane) & 7u];

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const uint table_offset = ((uint(data_a[ib_k].extra) >> extra_bit_idx) & 1u) << 5u;

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        const uint qh0 = subgroupShuffle(qh_u32, uint(q_offset) + q);
        const uint qh1 = subgroupShuffle(qh_u32, uint(q_offset) + q + 1u);
        qs_a[q] = unpack_iq5_k(qs_pair.x, qh0, uint(ib32), table_offset);
        qs_a[q + 1u] = unpack_iq5_k(qs_pair.y, qh1, uint(ib32), table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uint qs = subgroupShuffle(qs_u32, q_idx);
        const uint qh = subgroupShuffle(qh_u32, uint(q_offset) + q);
        qs_a[q] = unpack_iq5_k(qs, qh, uint(ib32), table_offset);
    }
#endif

    const uint scales_h = uint(data_a[ib_k].scales_h[uint(ib32) / 2u]) >> (4u * (uint(ib32) % 2u));
    const uint scales_l = uint(data_a[ib_k].scales_l[ib32]);
    const uint scale = scale_high_half
        ? ((scales_l >> 4u) | ((scales_h << 2u) & 0x30u))
        : ((scales_l & 0x0Fu) | ((scales_h << 4u) & 0x30u));
    a_scale = float(data_a[ib_k].d) * float(int32_t(scale) - 32);
}
#endif

#if defined(DATA_A_IQ6_K)
#define SG_IQK_LANES_PER_32      (QUANT_K_Q8_1 / SG_K)
#define SG_IQK_QS_PER_32         (QUANT_K_Q8_1 / 4)

void prepare_a_sg(uint ib_k, out int32_t qs_a[SG_QS_PER_ITER], out float a_scale) {
    const uint16_t lane = uint16_t(gl_SubgroupInvocationID);
    const uint16_t ib32 = uint16_t(lane / SG_IQK_LANES_PER_32);
    const bool scale_high_half = bool(lane & uint16_t(SG_IQK_LANES_PER_32 / 2u));
    const uint16_t q_offset = uint16_t((uint(lane) * SG_QS_PER_ITER) % SG_IQK_QS_PER_32);

#if SUBGROUP_SIZE == 16
    const uvec2 qs_u32 = uvec2(data_a_packed32[ib_k].qs[uint(lane) * 2u],
                               data_a_packed32[ib_k].qs[uint(lane) * 2u + 1u]);
#else
    const uint qs_u32 = data_a_packed32[ib_k].qs[uint(lane) & 31u];
#endif
    const uint qh_u32 = data_a_packed32[ib_k].qh[uint(lane) & 15u];

    const uint extra_bit_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    const uint table_offset = ((uint(data_a[ib_k].extra) >> extra_bit_idx) & 1u) << 6u;
    const uint shift = 2u * (uint(ib32) % 4u);

#if SUBGROUP_SIZE == 16
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; q += 2u) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uvec2 qs_pair = subgroupShuffle(qs_u32, q_idx >> 1u);
        const uint qh0 = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q);
        const uint qh1 = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q + 1u);
        qs_a[q] = unpack_iq6_k(qs_pair.x, qh0, shift, table_offset);
        qs_a[q + 1u] = unpack_iq6_k(qs_pair.y, qh1, shift, table_offset);
    }
#else
    [[unroll]] for (uint q = 0; q < SG_QS_PER_ITER; ++q) {
        const uint q_idx = 8u * (uint(ib32) / 2u) + uint(q_offset) + q;
        const uint qs = subgroupShuffle(qs_u32, q_idx);
        const uint qh = subgroupShuffle(qh_u32, 8u * (uint(ib32) / 4u) + uint(q_offset) + q);
        qs_a[q] = unpack_iq6_k(qs, qh, shift, table_offset);
    }
#endif

    const uint scale_idx = 2u * uint(ib32) + (scale_high_half ? 1u : 0u);
    a_scale = float(data_a[ib_k].d) * float(data_a[ib_k].scales[scale_idx]);
}
#endif

#endif


#if defined(DATA_A_IQK_ROW)
#extension GL_EXT_expect_assume : require
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#endif

uint8_t iqks_load_u8(uint offset) {
    return data_a[offset];
}

uint16_t iqks_load_u16(uint offset) {
#if defined(A_TYPE_PACKED16)
    assumeEXT((offset & 1u) == 0u);
    assumeEXT((offset & uint32_t(-2)) == offset);
    return data_a_packed16[offset / 2];
#else
    return pack8(u8vec2(iqks_load_u8(offset), iqks_load_u8(offset + 1)));
#endif
}

uint32_t iqks_load_u32(uint offset) {
#if defined(A_TYPE_PACKED32)
    assumeEXT((offset & 3u) == 0u);
    assumeEXT((offset & uint32_t(-4)) == offset);
    return data_a_packed32[offset / 4];
#else
    return pack32(u16vec2(iqks_load_u16(offset), iqks_load_u16(offset + 2)));
#endif
}

float iqks_row_scale(uint row_offset) {
#if defined(DATA_A_IQ2_KS) || defined(DATA_A_IQ2_KL) || defined(DATA_A_IQ3_KS)
    return float(uint16BitsToHalf(iqks_load_u16(row_offset)));
#else
    return uintBitsToFloat(iqks_load_u32(row_offset));
#endif
}

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
int8_t iqkt_next(inout uint state) {
    state *= 0xCBAC1FEDu;
    const uint value = state & 0x3F3F3F3Fu;
    return int8_t(dotPacked4x8EXT(int32_t(value), int32_t(0x01010101u)) + (-126));
}

//#define IQKT_MULTI_TURN 1
#if defined(IQKT_MULTI_TURN)
#extension GL_KHR_shader_subgroup_basic : require
const uint32_t iqkt_turn_muler[8] = {
    0x1u, 0xCBAC1FEDu, 0xC8734169u, 0xBD2B4535u, 0xE50C7D11u, 0x1220D7BDu, 0x94839CF9u, 0x58267985u
};
uint32_t iqkt_multi_turn(uint32_t state, uint8_t turn) {
    const uint32_t muler = gl_SubgroupInvocationID < 8
        ? iqkt_turn_muler[gl_SubgroupInvocationID]
        : 0u;
    // return state * subgroupShuffle(muler, turn);
    return state * iqkt_turn_muler[turn];
}
#endif

#endif

int32_t iqks_value4(uint block_offset, uint8_t element) {
    const uint8_t pos = uint8_t(element % 32u);

#if defined(DATA_A_IQ1_KT) || defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT) || defined(DATA_A_IQ4_KT)
    const uint8_t group8 = uint8_t(element / 8u);
    const uint8_t pos8 = uint8_t(element % 8u);
    const uint8_t ib32 = uint8_t(element / 32u);
    uint state;

#if defined(DATA_A_IQ1_KT)
    const uint8_t sh = iqks_load_u8(block_offset + ib32);
    const uint8_t x = iqks_load_u8(block_offset + 8 + group8);
    const uint8_t y_ = iqks_load_u8(block_offset + 40 + group8 % 16);
    const uint8_t y = bool(element & 128u) ? y_ >> 4u : y_ & uint8_t(0x0F);
    const bool z = bool(sh & (uint8_t(1) << (4 + group8 % 4)));
    state = pack16(u8vec2(x, y)) | uint16_t(z ? 0x2000u : 0x1000u);
#elif defined(DATA_A_IQ2_KT) || defined(DATA_A_IQ3_KT)
    state = uint(iqks_load_u16(block_offset + 4 + 2 * group8)) + 4096;
#else
    const uint header = iqks_load_u32(block_offset + 4 * ib32);
    //const uint8_t seed_index = uint16_t(2 * group8 + pos8 / 4u);
    const uint16_t seed_index = uint16_t(bitfieldExtract(uint32_t(element), 2, 1)) + ((element >> 2) & uint8_t(0x3e));
    const uint8_t x = iqks_load_u8(block_offset + 32 + seed_index);
    const uint8_t y_ = iqks_load_u8(block_offset + 96 + (seed_index % 32u));
    const uint8_t y = test_bit_to_get_high4(element, 7, y_);
    // const uint8_t z = uint8_t((header >> (8 + 3 * (seed_index % 8))) & 7);
    const uint32_t z = bitfieldExtract(header, 8 + 3 * (seed_index % uint16_t(8)), 3);
    // const bool w = bool(header & 1u);
    const uint32_t high = z + test_bit_to_get_bit(header, 0, 3) + 1u;
    state = pack16(u8vec2(x, y)) | (uint32_t(high) << 12u);
#endif

    const uint8_t count =
#if defined(DATA_A_IQ4_KT)
        uint8_t(pos8 % 4);
#else
        pos8;
#endif

#if defined(IQKT_MULTI_TURN)
    state = iqkt_multi_turn(state, count);
#else
    [[unroll]] for (uint8_t j = uint8_t(0); j < count; ++j) {
        iqkt_next(state);
    }
#endif

    i8vec4 values = i8vec4(iqkt_next(state), iqkt_next(state),
                           iqkt_next(state), iqkt_next(state));

#if defined(DATA_A_IQ3_KT)
    const u8vec4 signs = unpack8((iqks_load_u32(block_offset + 68 + element % 32) >> ib32) & 0x01010101u);
    const i8vec4 desired_sign = -i8vec4(signs);
    const i8vec4 sign_mask = (values >> int8_t(7)) ^ desired_sign;
    values = (values ^ sign_mask) - sign_mask;
#endif
    return pack32(values);
#elif defined(DATA_A_IQ4_KSS)
    const uint ib32 = element / 32;
    const uint word_offset = block_offset + 4 * (4 * ib32 + (pos % 16) / 4);
    uint values = iqks_load_u32(word_offset) & 0xFFFEFFFE;
    values ^= values >> 1;
    uint scale_bits = 0;
    [[unroll]] for (uint j = 0; j < 4; ++j) {
        scale_bits |= (iqks_load_u32(block_offset + 4 * (4 * ib32 + j)) & 0x00010001) << (2 * j);
    }
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    const u8vec4 indexes = unpack8((values >> (4 * (pos / 16))) & 0x0F0F0F0Fu);
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
#elif defined(DATA_A_IQ2_KS)
    const bool high_half = bool(element & 128u);
    const uint group = (element % 128) / 32;
    const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint values = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const bool is_hi_table = bool(extra & (uint16_t(1) << group));
    return unpack_iq2_k((values >> (2 * group)) & 0x03030303u, is_hi_table);
#elif defined(DATA_A_IQ2_KL)
    const uint8_t ib64 = uint8_t(element / 64u);
    const uint8_t pos32 = uint8_t(element % 32u);
    const bool high_half = bool(element & uint8_t(32));
    const bool odd = bool(element & uint8_t(1));
    const uint16_t packed = iqks_load_u16(block_offset + 6 + 16 * ib64 + pos32 / 2);
    const uint16_t high = iqks_load_u16(block_offset + 70 + pos32 / 2);
    const uint high_shift = 2 * ib64 + (high_half ? 1u : 0u);
    const uint16_t index_even = ((high_half ? packed >> 3u : packed << 1u) & uint16_t(0x1E1E)) |
                                (((high >> high_shift) & uint16_t(0x0101)) << 5u);
    const uint16_t index = odd ? index_even + uint16_t(0x0101) : index_even;
    const u8vec2 indexes = unpack8(index);
    return int32_t(pack32(u16vec2(iq2kl_table[indexes.x], iq2kl_table[indexes.y])));
#elif defined(DATA_A_IQ3_KS)
    const uint8_t shift_h = element >> 5;
    const uint8_t group = shift_h & uint8_t(3);
    const bool high_half = bool(shift_h & uint8_t(4));
    //const uint16_t extra = iqks_load_u16(block_offset) >> (high_half ? 4u : 0u);
    const uint low = iqks_load_u32(block_offset + 6 + (high_half ? 32u : 0u) + pos);
    const uint high = iqks_load_u32(block_offset + 70 + pos);
    const uint16_t hi_table_bit_idx = uint16_t(shift_h) + uint16_t(8);
    const bool is_hi_table = bool(iqks_load_u16(block_offset) & (uint16_t(1) << hi_table_bit_idx));
    //const bool is_hi_table = bool(extra & (uint16_t(1) << (8 + group)));
    return unpack_iq3_k(low, high, uint16_t(shift_h), is_hi_table);
#elif defined(DATA_A_IQ4_KS)
    const uint ib32 = element / 32;
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    const uint values = iqks_load_u32(block_offset + 8 + 16 * ib32 + pos % 16);
    const u8vec4 indexes = unpack8((values >> (4 * (pos / 16))) & 0x0F0F0F0Fu);
    const uint table_offset = bool(scale & uint8_t(1)) ? 16u : 0u;
    return pack32(i8vec4(kvalues_iq4_k[indexes.x + table_offset],
                         kvalues_iq4_k[indexes.y + table_offset],
                         kvalues_iq4_k[indexes.z + table_offset],
                         kvalues_iq4_k[indexes.w + table_offset]));
#else
    const uint ib64 = element / 64;
    const uint pos64 = element % 64;
    const uint pos32 = pos64 % 32;
    const bool high_half = bool(pos64 & 32u);
    const uint8_t scale = iqks_load_u8(block_offset + 2 * ib64 + (high_half ? 1u : 0u));
    const uint values = iqks_load_u32(block_offset + 8 + 32 * ib64 + pos32);
    const uint high = iqks_load_u32(block_offset + 136 + pos32);
    const uint value_shift = high_half ? 4u : 0u;
    const uint high_shift = 2 * ib64 + (high_half ? 1u : 0u);
    const uint table_offset = bool(scale & uint8_t(1)) ? 32u : 0u;
    const uint packed = ((values >> value_shift) & 0x0F0F0F0Fu) |
                        (((high >> high_shift) & 0x01010101u) << 4);
    const u8vec4 indexes = unpack8(packed);
    return pack32(i8vec4(kvalues_iq5_k[indexes.x + table_offset],
                         kvalues_iq5_k[indexes.y + table_offset],
                         kvalues_iq5_k[indexes.z + table_offset],
                         kvalues_iq5_k[indexes.w + table_offset]));
#endif
}

// The caller splits each 256-element IQKS block into eight 32-element sub-blocks.
// ib32 identifies the selected sub-block and is always in the range 0..7.
float iqks_d_scale(float row_scale, uint block_offset, uint8_t ib32) {
#if defined(DATA_A_IQ1_KT)
    const uint8_t scale = iqks_load_u8(block_offset + ib32) & uint8_t(0x0F);
    return row_scale * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ2_KT)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + group);
    const uint8_t scale = test_bit_to_get_high4(ib32, uint8_t(2), scale_);
    return row_scale * float(kvalues_iqkt_scale[scale]);
#elif defined(DATA_A_IQ3_KT)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + group);
    const uint8_t scale = test_bit_to_get_high4(ib32, uint8_t(2), scale_);
    return row_scale * float(scale);
#elif defined(DATA_A_IQ4_KT)
    return row_scale * float(int16_t(iqks_load_u8(block_offset + (ib32 << 2)) >> 1) - int16_t(64));
#elif defined(DATA_A_IQ4_KSS)
    const uint8_t ib32_offset = ib32 << 4;
    const uint scale_offset = block_offset + ib32_offset;
    const uint scale_bits =
        ( iqks_load_u32(scale_offset     ) & 0x00010001) |
        ((iqks_load_u32(scale_offset +  4) & 0x00010001) << 2) |
        ((iqks_load_u32(scale_offset +  8) & 0x00010001) << 4) |
        ((iqks_load_u32(scale_offset + 12) & 0x00010001) << 6);
    const uint8_t scale = uint8_t(scale_bits | (scale_bits >> 15));
    return row_scale * int8_t((scale & uint8_t(254)) - uint8_t(127));
#elif defined(DATA_A_IQ2_KS)
    const uint8_t packed_scale = iqks_load_u8(block_offset + 2 + (ib32 >> 1));
    const uint8_t scale_lo = test_bit_to_get_high4(ib32, 0, packed_scale);
    const uint16_t scale_hi_bit = uint16_t(1) << (uint16_t(ib32) + uint16_t(8));
    const uint16_t scale_hi = bool(iqks_load_u16(block_offset) & scale_hi_bit) ? uint16_t(0) : uint16_t(-16);
    const int16_t scale = int16_t(uint16_t(scale_lo) | scale_hi);
    return row_scale * float(scale);
#elif defined(DATA_A_IQ2_KL)
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_ = iqks_load_u8(block_offset + 2 + group);
    //const uint scale_lo = test_bit_to_get_high4(ib32, 2, scale_);
    const uint scale_lo = bitfieldExtract(uint32_t(scale_), ib32 & 4, 4);
    const uint scale_hi = iqks_load_u16(block_offset) >> uint16_t(ib32 << 1);
    const uint scale = bitfieldInsert(scale_lo, scale_hi, 4, 2);
    return row_scale * float(int(scale) - 32);
#elif defined(DATA_A_IQ3_KS)
    const bool high_half = bool(ib32 & uint8_t(4));
    const uint8_t group = ib32 & uint8_t(3);
    const uint8_t scale_lo = test_bit_to_get_high4(ib32, 2, iqks_load_u8(block_offset + 2 + group));
    const uint16_t scale_hi = bool(iqks_load_u16(block_offset) & (uint16_t(1) << ib32)) ? uint16_t(0) : uint16_t(-16);
    const int16_t scale = int16_t(uint16_t(scale_lo) | scale_hi);
    return row_scale * float(scale);
#else
    const uint8_t scale = iqks_load_u8(block_offset + ib32);
    return row_scale * float(int(scale & uint8_t(254)) - 127);
#endif
}

#endif
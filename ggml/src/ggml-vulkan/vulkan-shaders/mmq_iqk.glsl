
#if defined(DATA_A_IQ2_K) || defined(DATA_A_IQ2_KS)

#if 1

shared uint32_t iq2k_table[512];
const uint kpacked_iq2_k_lo = 0x1101F3E1u; // -31, -13, 1, 17
const uint kpacked_iq2_k_hi = 0x1606F8E6u; // -26,  -8, 6, 22

void init_iq2k_table() {
    const uint workgroup_size = gl_WorkGroupSize.x; // * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
    // assume only gl_WorkGroupSize.x and 256 being multiplies of 4*it.
    for (uint index_ = gl_LocalInvocationIndex * 4u; index_ < 256u; index_ += workgroup_size * 4u) {

        const uint shift1 = (index_ << 1u) & 0x18u;
        const uint shift2 = (index_ >> 1u) & 0x18u;
        const uint shift3 = (index_ >> 3u) & 0x18u;

        const uint32_t pack_lo = pack32(i8vec4(0,
            uint8_t(kpacked_iq2_k_lo >> shift1),
            uint8_t(kpacked_iq2_k_lo >> shift2),
            uint8_t(kpacked_iq2_k_lo >> shift3)));
        const uint32_t pack_hi = pack32(i8vec4(0,
            uint8_t(kpacked_iq2_k_hi >> shift1),
            uint8_t(kpacked_iq2_k_hi >> shift2),
            uint8_t(kpacked_iq2_k_hi >> shift3)));

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

void init_iq3k_table() {
    for (uint index = gl_LocalInvocationIndex; index < 64u; index += gl_WorkGroupSize.x) {
        const i8vec2 values = i8vec2(kvalues_iq3_k[index & 7u], kvalues_iq3_k[index >> 3u]);
        iq3k_table[index] = uint16_t(pack16(values));
        iq3k_table[64u + index] = uint16_t(pack16(values + i8vec2(4)));
    }
    barrier();
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
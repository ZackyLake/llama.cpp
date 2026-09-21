#ifndef PTQ1_0_GLSL
#define PTQ1_0_GLSL

float ptq1_0_trit(uint ib, uint a_offset, uint e) {
    uint b;
    uint n;
    if (e < 80u) {
        b = uint(data_a[a_offset + ib].qs[e & 15u]);
        n = e >> 4u;
    } else if (e < 120u) {
        const uint t = e - 80u;
        b = uint(data_a[a_offset + ib].qs[16u + (t & 7u)]);
        n = t >> 3u;
    } else {
        const uint t = e - 120u;
        b = uint(data_a[a_offset + ib].qh[t & 1u]);
        n = t >> 1u;
    }

    uint v = b;
    for (uint i = 0u; i < n; ++i) {
        v = (v * 3u) & 0xFFu;
    }
    return float(int((v * 3u) >> 8u) - 1);
}

#endif

//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#ifdef GGML_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        if defined(GGML_BUILD) || defined(GGML_BACKEND_BUILD)
#            define IQK_API __declspec(dllexport)
#        else
#            define IQK_API __declspec(dllimport)
#        endif
#    else
#        define IQK_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define IQK_API
#endif

#ifdef _MSC_VER
#define IQK_NOINLINE __declspec(noinline)
#define IQK_ALWAYS_INLINE __forceinline
#if !defined __x86_64__ && defined _M_X64
#define __x86_64__
#endif
#else
#define IQK_NOINLINE __attribute__((__noinline__))
#define IQK_ALWAYS_INLINE __attribute__((__always_inline__))
#endif

#if defined __x86_64__
#if defined HAVE_AVX512_BW
    #undef HAVE_AVX512_BW
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__)
    #define HAVE_AVX512_BW
#endif
#if defined HAVE_FANCY_SIMD
    #undef HAVE_FANCY_SIMD
#endif
#if defined(HAVE_AVX512_BW) && defined(__AVX512VNNI__) && defined(__AVX512DQ__)
    #define HAVE_FANCY_SIMD
#endif
#if defined HAVE_VNNI256
    #undef HAVE_VNNI256
#endif
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    #define HAVE_VNNI256
#endif
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_epi32
#elif defined(__AVXVNNI__)
    #define ggml_mm256_dpbusd_epi32 _mm256_dpbusd_avx_epi32
    #define ggml_mm256_dpwssd_epi32 _mm256_dpwssd_avx_epi32
    #define ggml_mm_dpbusd_epi32    _mm_dpbusd_avx_epi32
#endif
#endif


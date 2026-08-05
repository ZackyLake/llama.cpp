if (NOT GGML_SYSTEM_ARCH STREQUAL "x86")
    message(FATAL_ERROR "IQK backend requires x86-64")
endif()

set(GGML_IQK_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(GGML_IQK_KERNEL_SOURCES
    ${GGML_IQK_SOURCE_DIR}/iqk_mul_mat.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_1bit.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_iquants.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_iqk_quants.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_kquants.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_ktquants.cpp
    ${GGML_IQK_SOURCE_DIR}/iqk_gemm_legacy_quants.cpp)

set(GGML_IQK_SOURCE_DIR ${GGML_IQK_SOURCE_DIR} PARENT_SCOPE)
set(GGML_IQK_KERNEL_SOURCES ${GGML_IQK_KERNEL_SOURCES} PARENT_SCOPE)

function(ggml_add_iqk_backend_variant tag_name)
    if (tag_name)
        set(GGML_IQK_TARGET "ggml-iqk-${tag_name}")
        set(IQK_TARGET_NAME "${tag_name}")
        set(IQK_CPU_FEATS_TARGET "ggml-cpu-${tag_name}-feats")
    else()
        set(GGML_IQK_TARGET ggml-iqk)
        set(IQK_TARGET_NAME unified)
        set(IQK_CPU_FEATS_TARGET ggml-cpu-feats)
    endif()

    set(IQK_ARCH_FLAGS)
    set(IQK_ARCH_DEFINITIONS)

    if (tag_name STREQUAL "haswell")
    elseif (tag_name STREQUAL "cascadelake")
        set(IQK_AVX512 ON)
        set(IQK_AVX512_VNNI ON)
    elseif (tag_name STREQUAL "icelake" OR tag_name STREQUAL "zen4")
        set(IQK_AVX512 ON)
        set(IQK_AVX512_VBMI ON)
        set(IQK_AVX512_VNNI ON)
    elseif (tag_name STREQUAL "alderlake")
        set(IQK_AVX_VNNI ON)
    elseif (NOT tag_name)
        if (GGML_AVX512)
            set(IQK_AVX512 ON)
            set(IQK_AVX512_VBMI ${GGML_AVX512_VBMI})
            set(IQK_AVX512_VNNI ${GGML_AVX512_VNNI})
        else()
            set(IQK_AVX_VNNI ${GGML_AVX_VNNI})
        endif()
    else()
        message(FATAL_ERROR "Unsupported IQK backend variant: ${tag_name}")
    endif()

    if (MSVC)
        if (IQK_AVX512)
            list(APPEND IQK_ARCH_FLAGS /arch:AVX512)
            list(APPEND IQK_ARCH_DEFINITIONS
                __AVX512F__ __AVX512VL__ __AVX512DQ__ __AVX512BW__)
            if (IQK_AVX512_VNNI)
                list(APPEND IQK_ARCH_DEFINITIONS __AVX512VNNI__)
            endif()
            if (IQK_AVX512_VBMI)
                list(APPEND IQK_ARCH_DEFINITIONS
                    __AVX512VBMI__ __AVX512VPOPCNTDQ__ __AVX512VBMI2__)
            endif()
        else()
            list(APPEND IQK_ARCH_FLAGS /arch:AVX2)
        endif()
        if (IQK_AVX_VNNI)
            list(APPEND IQK_ARCH_DEFINITIONS __AVXVNNI__)
        endif()
    else()
        list(APPEND IQK_ARCH_FLAGS -mavx2 -mfma -mf16c)
        if (IQK_AVX512)
            list(APPEND IQK_ARCH_FLAGS
                -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)
            if (IQK_AVX512_VNNI)
                list(APPEND IQK_ARCH_FLAGS -mavx512vnni)
            endif()
            if (IQK_AVX512_VBMI)
                list(APPEND IQK_ARCH_FLAGS
                    -mavx512vbmi -mavx512vpopcntdq -mavx512vbmi2)
            endif()
        elseif (IQK_AVX_VNNI)
            list(APPEND IQK_ARCH_FLAGS -mavxvnni)
        endif()
    endif()

    ggml_get_variant_extra_flags("${tag_name}" IQK_EXTRA_FLAGS)
    list(APPEND IQK_ARCH_FLAGS ${IQK_EXTRA_FLAGS})

    ggml_add_backend_library(${GGML_IQK_TARGET}
        ${GGML_IQK_SOURCE_DIR}/ggml-iqk.cpp
        ${GGML_IQK_KERNEL_SOURCES})

    target_compile_features(${GGML_IQK_TARGET} PRIVATE cxx_std_17)
    target_compile_options(${GGML_IQK_TARGET} PRIVATE ${IQK_ARCH_FLAGS})
    target_compile_definitions(${GGML_IQK_TARGET} PRIVATE
        "IQK_TARGET_NAME=\"${IQK_TARGET_NAME}\""
        ${IQK_ARCH_DEFINITIONS})
    target_include_directories(${GGML_IQK_TARGET} PRIVATE
        ${GGML_IQK_SOURCE_DIR}
        ${GGML_IQK_SOURCE_DIR}/..
        ${GGML_IQK_SOURCE_DIR}/../../include)
    target_link_libraries(${GGML_IQK_TARGET} PRIVATE Threads::Threads)
    if (GGML_BACKEND_DL)
        target_link_libraries(${GGML_IQK_TARGET} PRIVATE ${IQK_CPU_FEATS_TARGET})
    endif()
    set_target_properties(${GGML_IQK_TARGET} PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

    if (GGML_OPENMP_ENABLED)
        target_compile_definitions(${GGML_IQK_TARGET} PRIVATE GGML_USE_OPENMP)
        target_link_libraries(${GGML_IQK_TARGET} PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
    endif()
endfunction()

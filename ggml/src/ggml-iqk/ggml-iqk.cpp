#include "ggml-iqk.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

struct iqk_compute_state_shared {
    int n_threads;
    std::atomic<int> n_barrier{0};
    std::atomic<int> n_barrier_passed{0};
    std::atomic<int> ec{0};
};

using iqk_compute_callback = bool (*)(void *, int, int, int, iqk_compute_state_shared *);

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

using iqk_thread_t = HANDLE;
using iqk_thread_ret_t = DWORD;

static int iqk_thread_create(iqk_thread_t * out, iqk_thread_ret_t (*func)(void *), void * arg) {
    HANDLE handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) func, arg, 0, NULL);
    if (handle == NULL) {
        return EAGAIN;
    }

    *out = handle;
    return 0;
}

static int iqk_thread_join(iqk_thread_t thread) {
    const int ret = (int) WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return ret;
}
#else
#include <pthread.h>
#include <sched.h>

using iqk_thread_t = pthread_t;
using iqk_thread_ret_t = void *;

static int iqk_thread_create(iqk_thread_t * out, iqk_thread_ret_t (*func)(void *), void * arg) {
    return pthread_create(out, NULL, func, arg);
}

static int iqk_thread_join(iqk_thread_t thread) {
    return pthread_join(thread, NULL);
}
#endif

static void iqk_thread_cpu_relax(void) {
#if defined(_WIN32)
    Sleep(0);
#elif defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#else
    sched_yield();
#endif
}

static void iqk_barrier(iqk_compute_state_shared * shared) {
    if (shared->n_threads == 1) {
        return;
    }

#ifdef GGML_USE_OPENMP
#pragma omp barrier
#else
    const int n_passed = shared->n_barrier_passed.load(std::memory_order_relaxed);
    const int n_barrier = shared->n_barrier.fetch_add(1, std::memory_order_seq_cst);

    if (n_barrier == shared->n_threads - 1) {
        shared->n_barrier.store(0, std::memory_order_relaxed);
        shared->n_barrier_passed.fetch_add(1, std::memory_order_seq_cst);
        return;
    }

    while (shared->n_barrier_passed.load(std::memory_order_relaxed) == n_passed) {
        iqk_thread_cpu_relax();
    }
#endif
}

struct iqk_compute_state {
    iqk_thread_t thrd;
    int ith;
    int n_nodes;
    iqk_compute_state_shared * shared;
    iqk_compute_callback callback;
    void * user_data;
};

static iqk_thread_ret_t iqk_graph_worker(void * arg) {
    iqk_compute_state * state = (iqk_compute_state *) arg;

    for (int i = 0; i < state->n_nodes; ++i) {
        if (!state->callback(state->user_data, i, state->ith, state->shared->n_threads, state->shared)) {
            state->shared->ec.store(1, std::memory_order_relaxed);
        }
        iqk_barrier(state->shared);
    }

    return 0;
}

static bool iqk_graph_compute(int n_nodes, int n_threads, iqk_compute_callback callback, void * user_data) {
    iqk_compute_state_shared shared;
    shared.n_threads = n_threads > 0 ? n_threads : 1;

#ifdef GGML_USE_OPENMP
#pragma omp parallel num_threads(shared.n_threads)
    {
        #pragma omp single
        {
            shared.n_threads = omp_get_num_threads();
        }

        iqk_compute_state state = {
            {}, omp_get_thread_num(), n_nodes, &shared, callback, user_data,
        };
        iqk_graph_worker(&state);
    }
#else
    std::vector<iqk_compute_state> workers(shared.n_threads);
    for (int i = 0; i < shared.n_threads; ++i) {
        workers[i] = {
            {}, i, n_nodes, &shared, callback, user_data,
        };
    }

    for (int i = 1; i < shared.n_threads; ++i) {
        if (iqk_thread_create(&workers[i].thrd, iqk_graph_worker, &workers[i]) != 0) {
            std::abort();
        }
    }

    iqk_graph_worker(&workers[0]);

    for (int i = 1; i < shared.n_threads; ++i) {
        iqk_thread_join(workers[i].thrd);
    }
#endif

    return shared.ec.load(std::memory_order_relaxed) == 0;
}

struct ggml_backend_iqk_context {
    int n_threads = 1;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
};

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

struct ggml_iqk_node_state {
    std::vector<std::vector<mmid_row_mapping>> matrix_rows;
};

static constexpr int GGML_IQK_UNARY_OP_SWIGLU_OAI = GGML_UNARY_OP_COUNT;

struct ggml_iqk_fusion_plan {
    int start_index = -1;
    int gate_index  = -1;
    int up_index    = -1;
    int gate_view_index = -1;
    int up_view_index   = -1;
    int glu_index   = -1;

    const ggml_tensor * gate_mat = nullptr;
    const ggml_tensor * up_mat   = nullptr;
    const ggml_tensor * input    = nullptr;
    const ggml_tensor * ids      = nullptr;
    const ggml_tensor * gate_weight = nullptr;
    const ggml_tensor * up_weight   = nullptr;
    ggml_tensor *       glu      = nullptr;

    int   unary_op = 0;
    float limit    = 0.0f;
    bool  merged   = false;
    bool  use_id   = false;
};

struct ggml_iqk_compute_context {
    ggml_backend_iqk_context * backend;
    ggml_cgraph * cgraph;
    std::vector<ggml_iqk_node_state> * node_states;
    std::vector<ggml_iqk_fusion_plan> fusion_plans;
    std::vector<int> fusion_start_plan;
    std::vector<uint8_t> fusion_skip;
};

static bool ggml_iqk_glu_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("IQK_GLU");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

static bool ggml_iqk_glu_debug_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("IQK_GLU_DEBUG");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

static std::set<uintptr_t> ggml_iqk_fused_node_ids;
static std::mutex ggml_iqk_fused_nodes_mutex;

static bool ggml_iqk_is_supported_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return true;
        default:
            return false;
    }
}

static enum ggml_type ggml_iqk_vec_dot_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
            return GGML_TYPE_Q8_2_X4;
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
            return GGML_TYPE_Q8_K;
        default:
            GGML_ABORT("%s: unsupported type %s\n", __func__, ggml_type_name(type));
    }
}

static bool ggml_iqk_mat_op_supported(const ggml_tensor * op) {
    if (op == nullptr || (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID)) {
        return false;
    }

    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * inputs  = op->src[1];
    if (weights == nullptr || inputs == nullptr || !ggml_iqk_is_supported_type(weights->type)) {
        return false;
    }
    const int64_t k_step = weights->type == GGML_TYPE_IQ4_NL ? QK8_0 : QK_K;
    if (inputs->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || inputs->ne[0] % k_step != 0) {
        return false;
    }
    if (weights->ne[0] != inputs->ne[0] || weights->ne[0] == 0 || weights->ne[1] == 0 ||
        weights->ne[2] == 0 || weights->ne[3] == 0 || inputs->ne[1] == 0 || inputs->ne[2] == 0 ||
        inputs->ne[3] == 0) {
        return false;
    }
    if (op->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * ids = op->src[2];
        if (ids == nullptr || ids->type != GGML_TYPE_I32 || !ggml_is_contiguous_to_1(ids)) {
            return false;
        }
        if (weights->ne[3] != 1 || inputs->ne[3] != 1 || op->ne[3] != 1) {
            return false;
        }
        if (ids->ne[2] != 1 || ids->ne[3] != 1 || ids->ne[1] != inputs->ne[2] ||
            ids->ne[0] % inputs->ne[1] != 0) {
            return false;
        }
        if (op->ne[0] != weights->ne[1] || op->ne[1] != ids->ne[0] || op->ne[2] != ids->ne[1]) {
            return false;
        }
    } else {
        if (inputs->ne[2] % weights->ne[2] != 0 || inputs->ne[3] % weights->ne[3] != 0) {
            return false;
        }
        if (op->ne[0] != weights->ne[1] || op->ne[1] != inputs->ne[1] ||
            op->ne[2] != inputs->ne[2] || op->ne[3] != inputs->ne[3]) {
            return false;
        }
    }

    return true;
}

static void ggml_iqk_debug_glu_rejection(const ggml_tensor * glu, const char * reason) {
    if (!ggml_iqk_glu_debug_enabled()) {
        return;
    }

    const ggml_tensor * src0 = glu ? glu->src[0] : nullptr;
    const ggml_tensor * src1 = glu ? glu->src[1] : nullptr;
    GGML_LOG_DEBUG("IQK_GLU: reject glu=%p name=%s reason=%s src0=%s src1=%s\n",
            (const void *) glu,
            glu ? ggml_get_name(glu) : "<null>",
            reason,
            src0 ? ggml_op_name(src0->op) : "<null>",
            src1 ? ggml_op_name(src1->op) : "<null>");
}

static void ggml_iqk_debug_support(const ggml_tensor * op, bool supported, const char * reason) {
    if (!ggml_iqk_glu_debug_enabled()) {
        return;
    }

    GGML_LOG_DEBUG("IQK_GLU: supports=%s op=%s name=%s type=%s ne=[%lld,%lld,%lld,%lld] reason=%s\n",
            supported ? "true" : "false",
            ggml_op_name(op->op),
            ggml_get_name(op),
            ggml_type_name(op->type),
            (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2], (long long) op->ne[3],
            reason);
}

static bool ggml_iqk_glu_activation(const ggml_tensor * glu, int * unary_op, float * limit) {
    int ignored_unary_op = 0;
    float ignored_limit = 0.0f;
    if (unary_op == nullptr) {
        unary_op = &ignored_unary_op;
    }
    if (limit == nullptr) {
        limit = &ignored_limit;
    }
    if (glu == nullptr || glu->op != GGML_OP_GLU || glu->src[0] == nullptr || glu->src[1] == nullptr ||
        ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }

    *limit = 0.0f;
    switch (ggml_get_glu_op(glu)) {
        case GGML_GLU_OP_REGLU:
            *unary_op = GGML_UNARY_OP_RELU;
            return true;
        case GGML_GLU_OP_GEGLU:
            *unary_op = GGML_UNARY_OP_GELU;
            return true;
        case GGML_GLU_OP_SWIGLU:
            *unary_op = GGML_UNARY_OP_SILU;
            return true;
        case GGML_GLU_OP_SWIGLU_OAI:
            if (std::abs(ggml_get_op_params_f32(glu, 2) - 1.702f) > 1e-6f ||
                std::abs(ggml_get_op_params_f32(glu, 3) - 7.0f) > 1e-6f) {
                return false;
            }
            *unary_op = GGML_IQK_UNARY_OP_SWIGLU_OAI;
            *limit = 7.0f;
            return true;
        default:
            return false;
    }
}

static bool ggml_iqk_can_fuse_glu(const ggml_tensor * glu) {
    if (!ggml_iqk_glu_enabled()) {
        ggml_iqk_debug_glu_rejection(glu, "IQK_GLU=0");
        return false;
    }

    if (!ggml_iqk_glu_activation(glu, nullptr, nullptr)) {
        ggml_iqk_debug_glu_rejection(glu, "activation");
        return false;
    }

    const ggml_tensor * first  = glu->src[0];
    const ggml_tensor * second = glu->src[1];
    if (glu->type != GGML_TYPE_F32 || first->type != GGML_TYPE_F32 || second->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(first, second) || !ggml_are_same_shape(glu, first) || !ggml_is_contiguous(glu)) {
        ggml_iqk_debug_glu_rejection(glu, "shape-or-type");
        return false;
    }

    if (first->op == GGML_OP_VIEW && second->op == GGML_OP_VIEW &&
        first->src[0] == second->src[0] && first->src[0] != nullptr) {
        const ggml_tensor * mat = first->src[0];
        const int64_t n = first->ne[0];
        if (mat->ne[0] != 2*n || first->view_offs != 0 ||
            second->view_offs != (size_t) n*mat->nb[0]) {
            ggml_iqk_debug_glu_rejection(glu, "merged-view-offset");
            return false;
        }
        if (first->ne[0] != n || second->ne[0] != n || first->nb[0] != mat->nb[0] ||
            second->nb[0] != mat->nb[0]) {
            ggml_iqk_debug_glu_rejection(glu, "merged-view-layout");
            return false;
        }
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            if (first->ne[i] != mat->ne[i] || second->ne[i] != mat->ne[i] ||
                first->nb[i] != mat->nb[i] || second->nb[i] != mat->nb[i]) {
                ggml_iqk_debug_glu_rejection(glu, "merged-view-layout");
                return false;
            }
        }
        if (!ggml_iqk_mat_op_supported(mat)) {
            ggml_iqk_debug_glu_rejection(glu, "merged-mat-op");
            return false;
        }
        return true;
    }

    if ((first->op != GGML_OP_MUL_MAT && first->op != GGML_OP_MUL_MAT_ID) ||
        second->op != first->op || first->src[1] != second->src[1]) {
        ggml_iqk_debug_glu_rejection(glu, "direct-mat-shape");
        return false;
    }
    if (first->op == GGML_OP_MUL_MAT_ID && first->src[2] != second->src[2]) {
        ggml_iqk_debug_glu_rejection(glu, "different-ids");
        return false;
    }

    if (!ggml_iqk_mat_op_supported(first) || !ggml_iqk_mat_op_supported(second)) {
        ggml_iqk_debug_glu_rejection(glu, "direct-mat-op");
        return false;
    }

    return true;
}

static int ggml_iqk_find_node_index(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i] == tensor) {
            return i;
        }
    }
    return -1;
}

static bool ggml_iqk_make_fusion_plan(const ggml_cgraph * cgraph, int glu_index, ggml_iqk_fusion_plan * plan) {
    ggml_tensor * glu = cgraph->nodes[glu_index];
    if (!ggml_iqk_can_fuse_glu(glu) || !ggml_iqk_glu_activation(glu, &plan->unary_op, &plan->limit)) {
        return false;
    }

    const ggml_tensor * first  = glu->src[0];
    const ggml_tensor * second = glu->src[1];
    plan->glu = glu;
    plan->glu_index = glu_index;
    plan->merged = first->op == GGML_OP_VIEW;

    if (plan->merged) {
        const ggml_tensor * mat = first->src[0];
        plan->start_index = ggml_iqk_find_node_index(cgraph, mat);
        plan->gate_view_index = ggml_iqk_find_node_index(cgraph, first);
        plan->up_view_index   = ggml_iqk_find_node_index(cgraph, second);
        plan->gate_mat = mat;
        plan->up_mat   = mat;
        plan->gate_weight = mat->src[0];
        plan->up_weight   = mat->src[0];
    } else {
        plan->gate_index = ggml_iqk_find_node_index(cgraph, first);
        plan->up_index   = ggml_iqk_find_node_index(cgraph, second);
        plan->start_index = std::min(plan->gate_index, plan->up_index);
        plan->gate_mat = first;
        plan->up_mat   = second;
        plan->gate_weight = first->src[0];
        plan->up_weight   = second->src[0];
    }

    if (plan->start_index < 0 || plan->gate_weight == nullptr || plan->up_weight == nullptr ||
        plan->gate_index < -1 || plan->up_index < -1 ||
        (plan->merged && (plan->gate_view_index < 0 || plan->up_view_index < 0)) ||
        (!plan->merged && (plan->gate_index < 0 || plan->up_index < 0))) {
        return false;
    }

    if (plan->gate_weight->type != plan->up_weight->type ||
        plan->gate_weight->ne[0] != plan->up_weight->ne[0] ||
        plan->gate_weight->ne[1] != plan->up_weight->ne[1] ||
        plan->gate_weight->ne[2] != plan->up_weight->ne[2] ||
        plan->gate_weight->ne[3] != plan->up_weight->ne[3]) {
        return false;
    }

    plan->input = plan->gate_mat->src[1];
    plan->use_id = plan->gate_mat->op == GGML_OP_MUL_MAT_ID;
    plan->ids = plan->use_id ? plan->gate_mat->src[2] : nullptr;
    if (plan->input == nullptr || (plan->use_id && plan->ids == nullptr)) {
        return false;
    }

    if (plan->merged) {
        return ggml_node_get_use_count(cgraph, plan->start_index) == 2 &&
               ggml_node_get_use_count(cgraph, plan->gate_view_index) == 1 &&
               ggml_node_get_use_count(cgraph, plan->up_view_index) == 1 &&
               (plan->gate_mat->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
    }

    return ggml_node_get_use_count(cgraph, plan->gate_index) == 1 &&
           ggml_node_get_use_count(cgraph, plan->up_index) == 1 &&
           (plan->gate_mat->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
           (plan->up_mat->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
}

static void ggml_iqk_log_fused_node(const ggml_tensor * node) {
    const uintptr_t node_id = reinterpret_cast<uintptr_t>(node);
    std::lock_guard<std::mutex> lock(ggml_iqk_fused_nodes_mutex);
    if (ggml_iqk_fused_node_ids.insert(node_id).second) {
        GGML_LOG_INFO("IQK: fused GLU node %p (%s)\n", (const void *) node_id, ggml_get_name(node));
    }
}

static bool ggml_iqk_quantize_src1(ggml_backend_iqk_context * ctx, const ggml_tensor * src1,
        enum ggml_type type, int ith, int nth, iqk_compute_state_shared * shared,
        const void ** data, size_t * row_size);

static bool ggml_iqk_compute_forward_fused(
        ggml_backend_iqk_context * ctx,
        const ggml_iqk_fusion_plan & plan,
        ggml_iqk_node_state * state,
        int ith,
        int nth,
        iqk_compute_state_shared * shared) {
    const ggml_tensor * gate_weight = plan.gate_weight;
    const ggml_tensor * up_weight   = plan.up_weight;
    const ggml_tensor * input       = plan.input;
    const ggml_tensor * ids         = plan.ids;
    const ggml_tensor * gate_mat    = plan.gate_mat;
    ggml_tensor * glu               = plan.glu;

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(gate_weight->type);
    const void * input_data = nullptr;
    size_t input_row_size = 0;
    if (!ggml_iqk_quantize_src1(ctx, input, typeB, ith, nth, shared, &input_data, &input_row_size)) {
        return false;
    }

    const int64_t nx = plan.merged ? gate_mat->ne[0]/2 : gate_mat->ne[0];
    const size_t up_stride = up_weight->nb[1];
    const size_t input_plane_size = input_row_size*input->ne[1];
    const size_t input_volume_size = input_plane_size*input->ne[2];

    if (!plan.use_id) {
        const int64_t r2 = input->ne[2]/gate_weight->ne[2];
        const int64_t r3 = input->ne[3]/gate_weight->ne[3];
        const size_t gate_plane_size = gate_weight->nb[2];
        const size_t up_plane_size = up_weight->nb[2];

        for (int64_t i13 = 0; i13 < input->ne[3]; ++i13) {
            for (int64_t i12 = 0; i12 < input->ne[2]; ++i12) {
                const int64_t gate_i12 = i12/r2;
                const int64_t gate_i13 = i13/r3;
                const char * gate_data = (const char *) gate_weight->data + gate_i12*gate_plane_size + gate_i13*gate_weight->nb[3];
                const char * up_data = (const char *) up_weight->data + (i12/(input->ne[2]/up_weight->ne[2]))*up_plane_size +
                        (i13/(input->ne[3]/up_weight->ne[3]))*up_weight->nb[3];
                if (plan.merged) {
                    up_data = gate_data + gate_weight->nb[2]/2;
                }

                float * output = (float *) ((char *) glu->data + i12*glu->nb[2] + i13*glu->nb[3]);
                const char * input_plane = (const char *) input_data + i12*input_plane_size + i13*input_volume_size;
                if (!iqk_moe_fused_up_gate(nx, input->ne[1], input->ne[0], input->ne[1], plan.unary_op,
                            gate_weight->type, up_data, gate_data, up_stride,
                            typeB, input_plane, input_row_size,
                            nullptr, nullptr, output, glu->nb[1], glu->nb[2], nullptr,
                            plan.limit, ith, nth)) {
                    return false;
                }
            }
        }
        return true;
    }

    const int n_ids = ids->ne[0];
    const int n_as = gate_weight->ne[2];
    if (input->ne[3] != 1 || gate_weight->ne[3] != 1 || glu->ne[3] != 1) {
        return false;
    }

    for (int64_t iid1 = ith; iid1 < ids->ne[1]; iid1 += nth) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                std::memset((char *) glu->data + id*glu->nb[1] + iid1*glu->nb[2], 0, glu->ne[0]*sizeof(float));
            }
        }
    }

    if (ith == 0) {
        state->matrix_rows.clear();
        state->matrix_rows.resize(n_as);
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 >= 0 && i02 < n_as) {
                    state->matrix_rows[i02].push_back({id, (int32_t) iid1});
                }
            }
        }
    }
    iqk_barrier(shared);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t n_rows = state->matrix_rows[cur_a].size();
        if (n_rows == 0) {
            continue;
        }

        const char * gate_data = (const char *) gate_weight->data + cur_a*gate_weight->nb[2];
        const char * up_data = (const char *) up_weight->data + cur_a*up_weight->nb[2];
        if (plan.merged) {
            up_data = gate_data + gate_weight->nb[2]/2;
        }
        if (!iqk_moe_fused_up_gate(nx, n_rows, input->ne[0], input->ne[1], plan.unary_op,
                    gate_weight->type, up_data, gate_data, up_stride,
                    typeB, input_data, input_row_size,
                    nullptr, nullptr, (float *) glu->data, glu->nb[1], glu->nb[2],
                    state->matrix_rows[cur_a].data(), plan.limit, ith, nth)) {
            return false;
        }
    }

    return true;
}

static bool ggml_iqk_compute_forward_glu(const ggml_tensor * dst, int ith, int nth) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0 == nullptr || src1 == nullptr || src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ggml_get_op_params_i32(dst, 1) != 0) {
        return false;
    }

    const int64_t nc = src0->ne[0];
    const int64_t nr = ggml_nrows(src0);
    const int64_t dr = (nr + nth - 1)/nth;
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = std::min<int64_t>(ir0 + dr, nr);
    const ggml_glu_op op = ggml_get_glu_op(dst);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    return iqk_compute_glu(nc, nr, op, alpha, limit,
            (const float *) src0->data, src0->nb[1],
            (const float *) src1->data, src1->nb[1],
            (float *) dst->data, dst->nb[1], ith, nth);
}

static void ggml_iqk_build_fusion_plans(ggml_iqk_compute_context * compute_ctx) {
    const int n_nodes = compute_ctx->cgraph->n_nodes;
    compute_ctx->fusion_start_plan.assign(n_nodes, -1);
    compute_ctx->fusion_skip.assign(n_nodes, 0);

    for (int i = 0; i < n_nodes; ++i) {
        if (compute_ctx->cgraph->nodes[i]->op != GGML_OP_GLU) {
            continue;
        }

        ggml_iqk_fusion_plan plan;
        if (!ggml_iqk_make_fusion_plan(compute_ctx->cgraph, i, &plan)) {
            continue;
        }

        const int plan_nodes[] = {
            plan.start_index, plan.gate_index, plan.up_index, plan.gate_view_index, plan.up_view_index, plan.glu_index,
        };
        bool conflict = false;
        for (int node_index : plan_nodes) {
            if (node_index >= 0 && (compute_ctx->fusion_skip[node_index] ||
                    compute_ctx->fusion_start_plan[node_index] >= 0)) {
                conflict = true;
                break;
            }
        }
        if (conflict || compute_ctx->fusion_start_plan[plan.start_index] >= 0) {
            continue;
        }

        const int plan_index = (int) compute_ctx->fusion_plans.size();
        compute_ctx->fusion_plans.push_back(plan);
        compute_ctx->fusion_start_plan[plan.start_index] = plan_index;
        for (int node_index : plan_nodes) {
            if (node_index >= 0 && node_index != plan.start_index) {
                compute_ctx->fusion_skip[node_index] = 1;
            }
        }
    }
}

static void ggml_iqk_quantize_row(enum ggml_type type, const float * src, void * dst, int64_t n) {
    if (type == GGML_TYPE_Q8_2_X4) {
        quantize_row_q8_2_x4(src, dst, n);
    } else {
        quantize_row_q8_K_ref(src, (block_q8_K *) dst, n);
    }
}

static char * ggml_iqk_reserve_work_data(ggml_backend_iqk_context * ctx, size_t size) {
    if (ctx->work_size < size) {
        ctx->work_data.reset(new char[size]);
        ctx->work_size = size;
    }
    return ctx->work_data.get();
}

static bool ggml_iqk_quantize_src1(ggml_backend_iqk_context * ctx, const ggml_tensor * src1,
        enum ggml_type type, int ith, int nth, iqk_compute_state_shared * shared,
        const void ** data, size_t * row_size) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    *row_size = ggml_row_size(type, src1->ne[0]);
    const size_t plane_size = *row_size * src1->ne[1];
    const size_t volume_size = plane_size * src1->ne[2];
    if (ith == 0) {
        ggml_iqk_reserve_work_data(ctx, volume_size * src1->ne[3]);
    }

    iqk_barrier(shared);

    char * work_data = ctx->work_data.get();

    for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            for (int64_t i11 = ith; i11 < src1->ne[1]; i11 += nth) {
                const float * src = (const float *)((const char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1]);
                void * dst = work_data + i13*volume_size + i12*plane_size + i11*(*row_size);
                ggml_iqk_quantize_row(type, src, dst, src1->ne[0]);
            }
        }
    }

    iqk_barrier(shared);

    *data = work_data;
    return true;
}

static bool ggml_iqk_compute_forward_mul_mat(
        ggml_backend_iqk_context * ctx,
    ggml_tensor * dst,
    int ith,
    int nth,
    iqk_compute_state_shared * shared) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(src0->type);
    const void * src1_data;
    size_t src1_row_size;
    if (!ggml_iqk_quantize_src1(ctx, src1, typeB, ith, nth, shared, &src1_data, &src1_row_size)) {
        return false;
    }

    return iqk_mul_mat_4d(ne01, ne11, ne00,
                ne02, ne03, ne12, ne13, nb02, nb03,
                src1_row_size*ne11, src1_row_size*ne11*ne12,
                nb2/sizeof(float), nb3/sizeof(float),
                src0->type, src0->data, nb01,
                typeB, src1_data, src1_row_size,
                (float *)dst->data, nb1/sizeof(float), ith, nth);
}

static bool ggml_iqk_compute_forward_mul_mat_id(
        ggml_backend_iqk_context * ctx,
            ggml_tensor * dst,
            ggml_iqk_node_state * state,
            int ith,
            int nth,
            iqk_compute_state_shared * shared) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    if (ne2 == 0 || ne11 == 0) {
        return true;
    }

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(src0->type);
    const void * src1_data;
    size_t src1_row_size;
    if (!ggml_iqk_quantize_src1(ctx, src1, typeB, ith, nth, shared, &src1_data, &src1_row_size)) {
        return false;
    }

    const int n_ids = ids->ne[0];
    const int n_as  = ne02;

    for (int64_t iid1 = ith; iid1 < ids->ne[1]; iid1 += nth) {
        for (int id = 0; id < n_ids; ++id) {
            const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            if (i02 < 0 || i02 >= n_as) {
                std::memset((char *)dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, dst->ne[0]*sizeof(float));
                continue;
            }
        }
    }

    if (ith == 0) {
        state->matrix_rows.clear();
        state->matrix_rows.resize(n_as);
        for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 < 0 || i02 >= n_as) {
                    continue;
                }
                state->matrix_rows[i02].push_back({id, (int32_t) iid1});
            }
        }
    }

    iqk_barrier(shared);

    for (int cur_a = 0; cur_a < n_as; ++cur_a) {
        const int64_t n_rows = state->matrix_rows[cur_a].size();
        if (n_rows == 0) {
            continue;
        }

        const char * src0_cur = (const char *) src0->data + cur_a*nb02;
        if (!iqk_mul_mat_moe(ne01, n_rows, ne00, ne11,
                    src0->type, src0_cur, nb01,
                    typeB, src1_data, src1_row_size,
                    (float *)dst->data, nb1, nb2,
                    state->matrix_rows[cur_a].data(), ith, nth)) {
            return false;
        }
    }

    return true;
}

static const char * ggml_backend_iqk_get_name(ggml_backend_t backend) {
    return "IQK";

    GGML_UNUSED(backend);
}

static void ggml_backend_iqk_free(ggml_backend_t backend) {
    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend->context;
    delete ctx;
    delete backend;
}

static bool ggml_iqk_compute_node(void * user_data, int node_index, int ith, int nth,
        iqk_compute_state_shared * shared) {
    ggml_iqk_compute_context * compute_ctx = (ggml_iqk_compute_context *) user_data;
    ggml_backend_iqk_context * ctx = compute_ctx->backend;
    struct ggml_tensor * node = compute_ctx->cgraph->nodes[node_index];

    if (compute_ctx->fusion_skip[node_index]) {
        return true;
    }

    const int plan_index = compute_ctx->fusion_start_plan[node_index];
    if (plan_index >= 0) {
        const ggml_iqk_fusion_plan & plan = compute_ctx->fusion_plans[plan_index];
        const bool fused = ggml_iqk_compute_forward_fused(ctx, plan,
                &(*compute_ctx->node_states)[node_index], ith, nth, shared);
        if (fused) {
            ggml_iqk_log_fused_node(plan.glu);
        }
        return fused;
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return true;
    }

    ggml_iqk_node_state * state = &(*compute_ctx->node_states)[node_index];
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return ggml_iqk_compute_forward_mul_mat(ctx, node, ith, nth, shared);
        case GGML_OP_MUL_MAT_ID:
            return ggml_iqk_compute_forward_mul_mat_id(ctx, node, state, ith, nth, shared);
        case GGML_OP_GLU:
            return ggml_iqk_compute_forward_glu(node, ith, nth);
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
    }
}

static ggml_status ggml_backend_iqk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend->context;
    std::vector<ggml_iqk_node_state> node_states(cgraph->n_nodes);
    ggml_iqk_compute_context compute_ctx = { ctx, cgraph, &node_states };
    ggml_iqk_build_fusion_plans(&compute_ctx);

    if (!iqk_graph_compute(cgraph->n_nodes, ctx->n_threads, ggml_iqk_compute_node, &compute_ctx)) {
        GGML_LOG_ERROR("%s: IQK kernel rejected graph\n", __func__);
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_iqk_i = {
    /* .get_name                = */ ggml_backend_iqk_get_name,
    /* .free                    = */ ggml_backend_iqk_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_iqk_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_iqk_guid(void) {
    static const char * guid_str = "GGML-IQK-ACCEL";
    return reinterpret_cast<ggml_guid_t>(const_cast<char*>(guid_str));
}

ggml_backend_t ggml_backend_iqk_init(void) {
    ggml_backend_iqk_context * ctx = new ggml_backend_iqk_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_iqk_guid(),
        /* .iface   = */ ggml_backend_iqk_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_iqk_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_iqk(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_iqk_guid());
}

void ggml_backend_iqk_set_n_threads(ggml_backend_t backend_iqk, int n_threads) {
    GGML_ASSERT(ggml_backend_is_iqk(backend_iqk));

    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend_iqk->context;
    ctx->n_threads = n_threads;
}

static const char * ggml_backend_iqk_device_get_name(ggml_backend_dev_t dev) {
    return "IQK";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_iqk_device_get_description(ggml_backend_dev_t dev) {
    return "IQK x86 quantized matrix multiplication backend";

    GGML_UNUSED(dev);
}

static void ggml_backend_iqk_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_iqk_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_iqk_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_iqk_device_get_name(dev);
    props->description = ggml_backend_iqk_device_get_description(dev);
    props->type        = ggml_backend_iqk_device_get_type(dev);
    ggml_backend_iqk_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false
    };
}

static ggml_backend_t ggml_backend_iqk_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_t backend = ggml_backend_iqk_init();
    if (backend == NULL) {
        GGML_LOG_ERROR("%s: error: failed to initialize IQK backend\n", __func__);
        return NULL;
    }

    return backend;

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_iqk_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_iqk_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_iqk_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            ggml_iqk_debug_support(op, true, "view-or-none");
            return true;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                const bool supported = ggml_iqk_mat_op_supported(op);
                ggml_iqk_debug_support(op, supported, supported ? "mat-op" : "mat-op-shape-or-type");
                return supported;
            }
        case GGML_OP_GLU:
            {
                const bool supported = ggml_iqk_can_fuse_glu(op);
                ggml_iqk_debug_support(op, supported, supported ? "glu-pattern" : "glu-pattern-rejected");
                return supported;
            }
        default:
            ggml_iqk_debug_support(op, false, "unsupported-op");
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_iqk_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_iqk_device_i = {
    /* .get_name               = */ ggml_backend_iqk_device_get_name,
    /* .get_description        = */ ggml_backend_iqk_device_get_description,
    /* .get_memory             = */ ggml_backend_iqk_device_get_memory,
    /* .get_type               = */ ggml_backend_iqk_device_get_type,
    /* .get_props              = */ ggml_backend_iqk_device_get_props,
    /* .init_backend           = */ ggml_backend_iqk_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_iqk_device_get_buffer_type,
    /* .get_host_buffer_type   = */ NULL,
    /* .buffer_from_host_ptr   = */ ggml_backend_iqk_device_buffer_from_host_ptr,
    /* .supports_op            = */ ggml_backend_iqk_device_supports_op,
    /* .supports_buft          = */ ggml_backend_iqk_device_supports_buft,
    /* .offload_op             = */ NULL,
    /* .event_new              = */ NULL,
    /* .event_free             = */ NULL,
    /* .event_synchronize      = */ NULL,
};

static const char * ggml_backend_iqk_reg_get_name(ggml_backend_reg_t reg) {
    return "IQK";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_iqk_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_iqk_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_iqk_device = {
        /* .iface   = */ ggml_backend_iqk_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_iqk_device;
}

static void * ggml_backend_iqk_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_iqk_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_iqk_reg_i = {
    /* .get_name         = */ ggml_backend_iqk_reg_get_name,
    /* .get_device_count = */ ggml_backend_iqk_reg_get_device_count,
    /* .get_device       = */ ggml_backend_iqk_reg_get_device,
    /* .get_proc_address = */ ggml_backend_iqk_get_proc_address,
};

ggml_backend_reg_t ggml_backend_iqk_reg(void) {
    static struct ggml_backend_reg ggml_backend_iqk_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_iqk_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_iqk_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_iqk_reg)

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include "ggml-iqk.h"
#include "ggml-cpu.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "iqk_mul_mat.h"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#else
#include <sched.h>
#endif

struct ggml_backend_iqk_context;

static void iqk_thread_cpu_relax(void) {
#if defined(_WIN32)
    _mm_pause();
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static void iqk_thread_yield(void) {
#if defined(_WIN32)
    SwitchToThread();
#else
    sched_yield();
#endif
}

struct iqk_phase_latch {
    static_assert(std::atomic<uint64_t>::is_always_lock_free, "IQK phase latch requires lock-free 64-bit atomics");
    explicit iqk_phase_latch(uint32_t task_count) noexcept : task_count_(task_count) {
    }

    iqk_phase_latch(const iqk_phase_latch &) = delete;
    iqk_phase_latch & operator=(const iqk_phase_latch &) = delete;

    // Latches only move before their phase is published to workers.
    iqk_phase_latch(iqk_phase_latch && other) noexcept :
        task_count_(other.task_count_),
        state_(other.state_.load(std::memory_order_relaxed)) {
    }

    iqk_phase_latch & operator=(iqk_phase_latch && other) noexcept {
        if (this != &other) {
            task_count_ = other.task_count_;
            state_.store(other.state_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }
        return *this;
    }

    [[nodiscard]]
    std::optional<uint32_t> next(std::optional<uint32_t> completed) noexcept {
        const uint64_t delta = (uint64_t{1} << 32) + (completed ? 1 : 0);
        const uint64_t old = state_.fetch_add(delta, std::memory_order_acq_rel);
        uint32_t completed_after = (uint32_t) old;
        const uint32_t candidate = (uint32_t) (old >> 32);

        if (completed) {
            ++completed_after;
        }
        if (candidate < task_count_) {
            return candidate;
        }

        const int n_spin_before_yield = 100000;
        while (completed_after < task_count_) {
            for (int i = 0; i < n_spin_before_yield; ++i) {
                const uint64_t current = state_.load(std::memory_order_acquire);
                completed_after = (uint32_t) current;
                if (completed_after >= task_count_) {
                    return std::nullopt;
                }
                iqk_thread_cpu_relax();
            }
            iqk_thread_yield();
        }

        return std::nullopt;
    }

    uint32_t task_count() const noexcept {
        return task_count_;
    }

private:
    uint32_t task_count_;
    std::atomic<uint64_t> state_{0};
};


struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

struct ggml_iqk_node_state {
    explicit ggml_iqk_node_state(const std::array<uint32_t, 4> & latch_sizes) :
        latches{iqk_phase_latch(latch_sizes[0]), iqk_phase_latch(latch_sizes[1]),
                iqk_phase_latch(latch_sizes[2]), iqk_phase_latch(latch_sizes[3])} {
    }

    std::vector<std::vector<mmid_row_mapping>> matrix_rows;
    std::vector<int> active_experts;
    std::array<iqk_phase_latch, 4> latches;
    int chunks_per_expert = 1;
};

static constexpr int GGML_IQK_UNARY_OP_SWIGLU_OAI = GGML_UNARY_OP_COUNT;
static constexpr int GGML_IQK_EXPERT_CHUNK_MIN_NX = 32;
static constexpr int GGML_IQK_DENSE_CHUNK_NX = 64;
static constexpr int GGML_IQK_MAX_TASKS_PER_THREAD = 5;

static int ggml_iqk_expert_chunk_count(int64_t nx, int n_threads) {
    return (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, nx/GGML_IQK_EXPERT_CHUNK_MIN_NX));
}

static uint32_t ggml_iqk_task_count(enum ggml_op op, int n_threads, int64_t nx) {
    GGML_ASSERT(n_threads > 0);
    if (op != GGML_OP_MUL_MAT) {
        return (uint32_t) n_threads;
    }

    int64_t chunk_nx = GGML_IQK_DENSE_CHUNK_NX;
    int64_t n_tasks = (nx + chunk_nx - 1)/chunk_nx;
    if (n_tasks <= n_threads) {
        return (uint32_t) n_threads;
    }

    while (n_tasks >= GGML_IQK_MAX_TASKS_PER_THREAD*n_threads) {
        chunk_nx += GGML_IQK_DENSE_CHUNK_NX;
        n_tasks = (nx + chunk_nx - 1)/chunk_nx;
    }
    GGML_ASSERT(n_tasks > n_threads);
    return (uint32_t) n_tasks;
}

struct ggml_iqk_fusion_plan {
    int start_index = -1;
    int gate_index  = -1;
    int up_index    = -1;
    int gate_view_index = -1;
    int up_view_index   = -1;
    int glu_index   = -1;
    int gate_clamp_index = -1;
    int up_clamp_index   = -1;

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
    bool  gate_clamped = false;
    bool  up_clamped   = false;
};

struct ggml_iqk_compute_task {
    ggml_iqk_compute_task(bool fused, size_t task_index, const std::array<uint32_t, 4> & latch_sizes) :
        is_fused(fused), index(task_index), node_state(latch_sizes) {
    }

    bool is_fused = false;
    size_t index = 0;
    ggml_iqk_node_state node_state;
};

struct iqk_compute_state_shared;

static void ggml_iqk_compute_node(iqk_compute_state_shared * shared);

struct iqk_compute_state_shared {
    ggml_backend_iqk_context * backend = nullptr;
    ggml_cgraph * cgraph = nullptr;
    std::vector<ggml_iqk_compute_task> compute_tasks;
    std::vector<ggml_iqk_fusion_plan> fusion_plans;
    char * work_data = nullptr;

    int n_threads = 0;
    std::atomic<int> n_barrier{0};
    std::atomic<int> n_barrier_passed{0};
    std::atomic<int> ec{0};

    void barrier();
};

void iqk_compute_state_shared::barrier() {
    if (n_threads == 1) {
        return;
    }

    const int n_passed = n_barrier_passed.load(std::memory_order_relaxed);
    const int n_barrier_count = n_barrier.fetch_add(1, std::memory_order_seq_cst);

    if (n_barrier_count == n_threads - 1) {
        n_barrier.store(0, std::memory_order_relaxed);
        n_barrier_passed.fetch_add(1, std::memory_order_seq_cst);
        return;
    }

    // spin a bounded number of times before yielding, so the last thread is
    // not starved by the scheduler while the others spin
    const int n_spin_before_sleep = 100000;
    while (n_barrier_passed.load(std::memory_order_relaxed) == n_passed) {
        for (int i = 0; i < n_spin_before_sleep; ++i) {
            if (n_barrier_passed.load(std::memory_order_relaxed) != n_passed) {
                goto done;
            }
            iqk_thread_cpu_relax();
        }
        iqk_thread_yield();
    }
done:

    // exit barrier (full seq-cst fence)
    // TSAN doesn't support standalone fence yet, we use a dummy read-modify-write instead
#ifdef GGML_TSAN_ENABLED
    n_barrier_passed.fetch_add(0, std::memory_order_seq_cst);
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

// bitmask covering bits [begin, end); handles end == 64 where 1ULL << 64 is UB
static uint64_t iqk_mask_bits(uint32_t begin, uint32_t end) {
    if (end == 64) {
        return ~0ULL << begin;
    }
    return (1ULL << end) - (1ULL << begin);
}

enum class iqk_threadpool_state {
    idle,
    config,
    compute,
    stop,
};

// Persistent threadpool. Worker threads are spawned on demand and reused
// across graphs. The main thread only dispatches and waits.
struct iqk_threadpool {
    std::mutex mutex;

    // current graph (set before kickoff, read by workers)
    iqk_compute_state_shared * shared = nullptr;

    std::atomic<uint64_t> act_mask{0}; // bitmask of workers still to act on the current state
    std::atomic<uint64_t> dispatch_seq{0};
    std::atomic<iqk_threadpool_state> state{iqk_threadpool_state::idle};

    std::vector<std::thread> threads;

    // CPU placement params, forwarded from the CPU backend via private API
    ggml_threadpool_params params = {};
    std::vector<int> cpu_ids; // CPU indices allowed by the mask

    // store the CPU placement params; only update when they actually changed
    void set_params(const ggml_threadpool_params & tpp) {
        std::lock_guard<std::mutex> lock(mutex);
        if (ggml_threadpool_params_match(&params, &tpp)) {
            return;
        }
        params = tpp;

        // collect the CPU indices allowed by the mask and build the log string
        cpu_ids.clear();
        std::string ids;
        for (int i = 0; i < 64; ++i) {
            if (params.cpumask[i]) {
                cpu_ids.push_back(i);
                if (!ids.empty()) {
                    ids += ",";
                }
                ids += std::to_string(i);
            }
        }
        GGML_LOG_INFO("IQK: threadpool cpu mask: %s\n", ids.c_str());

        // trigger all workers to re-apply the config and wait for them to finish
        dispatch_locked(threads.size(), iqk_threadpool_state::config);
    }

    // apply priority and affinity for worker ith; thread ith binds to cpu_ids[ith]
    void apply_thread_config(uint32_t ith) {
        ggml_thread_apply_priority(params.prio);
        if (ith < cpu_ids.size()) {
            bool mask[GGML_MAX_N_THREADS] = {false};
            mask[cpu_ids[ith]] = true;
            ggml_thread_apply_affinity(mask);
        }
    }

    void complete(uint64_t bit) {
        if (act_mask.fetch_and(~bit, std::memory_order_acq_rel) == bit) {
            act_mask.notify_one();
        }
    }

    void wait_for_workers() {
        uint64_t mask = act_mask.load(std::memory_order_acquire);
        while (mask != 0) {
            act_mask.wait(mask, std::memory_order_acquire);
            mask = act_mask.load(std::memory_order_acquire);
        }
    }

    void worker(uint32_t ith) {
        // worker ith uses bit ith
        const uint64_t bit = 1ULL << ith;

        apply_thread_config(ith);
        uint64_t dispatch = dispatch_seq.load(std::memory_order_acquire);
        complete(bit);

        while (true) {
            dispatch_seq.wait(dispatch, std::memory_order_acquire);
            dispatch = dispatch_seq.load(std::memory_order_acquire);

            if ((act_mask.load(std::memory_order_acquire) & bit) == 0) {
                continue;
            }

            switch (state.load(std::memory_order_acquire)) {
                case iqk_threadpool_state::stop:
                    complete(bit);
                    return;
                case iqk_threadpool_state::config:
                    apply_thread_config(ith);
                    break;
                case iqk_threadpool_state::compute:
                    ggml_iqk_compute_node(shared);
                    break;
                default:
                    break;
            }
            complete(bit);
        }
    }

    // dispatch a state change to all workers and wait for them to finish.
    // `count` is the number of worker threads; workers cover bits 0..count-1.
    void dispatch_locked(size_t count, iqk_threadpool_state st) {
        if (count == 0) {
            return;
        }
        state.store(st, std::memory_order_relaxed);
        act_mask.store(iqk_mask_bits(0, (uint32_t) count), std::memory_order_relaxed);
        dispatch_seq.fetch_add(1, std::memory_order_release);
        dispatch_seq.notify_all();
        wait_for_workers();

        if (st == iqk_threadpool_state::stop) {
            for (auto & t : threads) {
                t.join();
            }
            threads.clear();
        } else {
            state.store(iqk_threadpool_state::idle, std::memory_order_relaxed);
        }
    }

    // grow the pool; must be called while no graph is running
    void resize(uint32_t n_threads) {
        std::lock_guard<std::mutex> lock(mutex);
        // act_mask is a uint64_t, so at most 64 workers
        n_threads = std::min(n_threads, 64u);
        const uint32_t old = (uint32_t) threads.size();
        if (n_threads <= old) {
            return;
        }
        GGML_LOG_INFO("IQK: threadpool resize %u -> %u\n", old, n_threads);
        threads.reserve(n_threads);
        // set the bits for the new workers before spawning them
        act_mask.store(iqk_mask_bits(old, n_threads), std::memory_order_relaxed);
        for (uint32_t i = old; i < n_threads; ++i) {
            threads.emplace_back(&iqk_threadpool::worker, this, i);
        }
        // wait for all new workers to clear their bits before returning
        wait_for_workers();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex);
        dispatch_locked(threads.size(), iqk_threadpool_state::stop);
    }

    bool compute(iqk_compute_state_shared & shared) {
        std::lock_guard<std::mutex> lock(mutex);
        this->shared = &shared;
        dispatch_locked((size_t) shared.n_threads, iqk_threadpool_state::compute);

        return shared.ec.load(std::memory_order_relaxed) == 0;
    }
};

using ggml_backend_cpu_get_type_traits_t = const struct ggml_type_traits_cpu * (*)(enum ggml_type type);

struct ggml_backend_iqk_context {
    int n_threads = 1;
    std::unique_ptr<char[]> work_data;
    size_t work_size = 0;
    iqk_threadpool threadpool;
    ggml_backend_reg_t cpu_reg = nullptr;
    ggml_backend_cpu_get_type_traits_t cpu_get_type_traits = nullptr;
    ggml_from_float_t cpu_quantize_row_q8_K = (ggml_from_float_t) quantize_row_q8_K_ref;
    ggml_from_float_t cpu_quantize_row_q8_K128 = (ggml_from_float_t) quantize_row_q8_K128_ref;
    ggml_from_float_t cpu_quantize_row_q8_2_x4 = quantize_row_q8_2_x4_ref;
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
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
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
        case GGML_TYPE_IQ1_S_R4:
        case GGML_TYPE_IQ1_M_R4:
            return GGML_TYPE_Q8_K128;
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
    const bool is_r4 = weights->type == GGML_TYPE_IQ1_S_R4 || weights->type == GGML_TYPE_IQ1_M_R4;
    const int64_t k_step = weights->type == GGML_TYPE_IQ4_NL ? QK8_0 : is_r4 ? 128 : QK_K;
    if (inputs->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 || inputs->ne[0] % k_step != 0) {
        return false;
    }
    if (is_r4 && weights->ne[1] % 4 != 0) {
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

// unwrap an optional CLAMP node wrapping a matmul result
// returns the underlying mat tensor, or nullptr on invalid CLAMP
static const ggml_tensor * ggml_iqk_unwrap_clamp(const ggml_tensor * node, float * min_val, float * max_val) {
    if (node->op == GGML_OP_CLAMP) {
        const ggml_tensor * src = node->src[0];
        if (src == nullptr) {
            return nullptr;
        }
        *min_val = ggml_get_op_params_f32(node, 0);
        *max_val = ggml_get_op_params_f32(node, 1);
        return src;
    }
    return node;
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

    // unwrap optional CLAMP nodes wrapping the gate/up matmul results
    float gate_min = 0.0f, gate_max = 0.0f;
    float up_min   = 0.0f, up_max   = 0.0f;
    const ggml_tensor * first  = ggml_iqk_unwrap_clamp(glu->src[0], &gate_min, &gate_max);
    const ggml_tensor * second = ggml_iqk_unwrap_clamp(glu->src[1], &up_min, &up_max);
    if (first == nullptr || second == nullptr) {
        ggml_iqk_debug_glu_rejection(glu, "unwrap");
        return false;
    }

    if (glu->type != GGML_TYPE_F32 || first->type != GGML_TYPE_F32 || second->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(first, second) || !ggml_are_same_shape(glu, first) || !ggml_is_contiguous(glu)) {
        ggml_iqk_debug_glu_rejection(glu, "shape-or-type");
        return false;
    }

    // validate CLAMP params against the fused kernel semantics:
    //   gate = min(silu(x), limit), up = clamp(x, -limit, limit)
    // both sides must be clamped together, with the same limit
    const bool gate_clamped = glu->src[0]->op == GGML_OP_CLAMP;
    const bool up_clamped   = glu->src[1]->op == GGML_OP_CLAMP;
    if (gate_clamped != up_clamped) {
        ggml_iqk_debug_glu_rejection(glu, "clamp-mismatch");
        return false;
    }
    if (gate_clamped) {
        // gate clamp must be [-INF, limit]; silu output is >= 0 so min <= 0 is a no-op
        if (gate_min > 0.0f) {
            ggml_iqk_debug_glu_rejection(glu, "gate-clamp-min");
            return false;
        }
        // up clamp must be symmetric [-limit, limit]
        if (up_min != -up_max) {
            ggml_iqk_debug_glu_rejection(glu, "up-clamp-symmetric");
            return false;
        }
        if (gate_max != up_max || gate_max <= 1e-6f) {
            ggml_iqk_debug_glu_rejection(glu, "clamp-limit");
            return false;
        }
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
        // R4 rows are interleaved 4 at a time; the fused kernel halves the
        // merged row count, so the total must be a multiple of 8
        if (mat->src[0] != nullptr &&
            (mat->src[0]->type == GGML_TYPE_IQ1_S_R4 || mat->src[0]->type == GGML_TYPE_IQ1_M_R4) &&
            mat->src[0]->ne[1] % 8 != 0) {
            ggml_iqk_debug_glu_rejection(glu, "merged-r4-nrows");
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
    if ((glu->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        !ggml_iqk_can_fuse_glu(glu) || !ggml_iqk_glu_activation(glu, &plan->unary_op, &plan->limit)) {
        return false;
    }

    const ggml_tensor * first  = glu->src[0];
    const ggml_tensor * second = glu->src[1];
    plan->glu = glu;
    plan->glu_index = glu_index;

    // unwrap optional CLAMP nodes wrapping the gate/up matmul results
    float gate_min = 0.0f, gate_max = 0.0f;
    float up_min   = 0.0f, up_max   = 0.0f;
    const ggml_tensor * first_mat  = ggml_iqk_unwrap_clamp(first, &gate_min, &gate_max);
    const ggml_tensor * second_mat = ggml_iqk_unwrap_clamp(second, &up_min, &up_max);
    plan->gate_clamped = first->op == GGML_OP_CLAMP;
    plan->up_clamped   = second->op == GGML_OP_CLAMP;
    if (plan->gate_clamped) {
        plan->gate_clamp_index = ggml_iqk_find_node_index(cgraph, first);
        plan->limit = gate_max;
    }
    if (plan->up_clamped) {
        plan->up_clamp_index = ggml_iqk_find_node_index(cgraph, second);
        plan->limit = up_max;
    }

    plan->merged = first_mat->op == GGML_OP_VIEW;

    if (plan->merged) {
        const ggml_tensor * mat = first_mat->src[0];
        plan->start_index = ggml_iqk_find_node_index(cgraph, mat);
        plan->gate_view_index = ggml_iqk_find_node_index(cgraph, first_mat);
        plan->up_view_index   = ggml_iqk_find_node_index(cgraph, second_mat);
        plan->gate_mat = mat;
        plan->up_mat   = mat;
        plan->gate_weight = mat->src[0];
        plan->up_weight   = mat->src[0];
    } else {
        plan->gate_index = ggml_iqk_find_node_index(cgraph, first_mat);
        plan->up_index   = ggml_iqk_find_node_index(cgraph, second_mat);
        plan->start_index = std::min(plan->gate_index, plan->up_index);
        plan->gate_mat = first_mat;
        plan->up_mat   = second_mat;
        plan->gate_weight = first_mat->src[0];
        plan->up_weight   = second_mat->src[0];
    }

    if (plan->start_index < 0 || plan->gate_weight == nullptr || plan->up_weight == nullptr ||
        plan->gate_index < -1 || plan->up_index < -1 ||
        (plan->merged && (plan->gate_view_index < 0 || plan->up_view_index < 0)) ||
        (!plan->merged && (plan->gate_index < 0 || plan->up_index < 0)) ||
        (plan->gate_clamped && plan->gate_clamp_index < 0) ||
        (plan->up_clamped && plan->up_clamp_index < 0)) {
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

    if ((plan->gate_mat->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (plan->up_mat->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
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

static size_t ggml_iqk_src1_work_size(const ggml_tensor * src1, enum ggml_type type);
static void ggml_iqk_quantize_src1(const ggml_tensor * src1,
    enum ggml_type type, iqk_compute_state_shared * shared,
    int ith, int nth);

static bool ggml_iqk_compute_forward_fused(
        const ggml_iqk_fusion_plan & plan,
    ggml_iqk_node_state * state,
        iqk_compute_state_shared * shared) {
    const ggml_tensor * gate_weight = plan.gate_weight;
    const ggml_tensor * up_weight   = plan.up_weight;
    const ggml_tensor * input       = plan.input;
    const ggml_tensor * gate_mat    = plan.gate_mat;
    ggml_tensor * glu               = plan.glu;

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(gate_weight->type);
    const void * input_data = shared->work_data;
    const size_t input_row_size = ggml_row_size(typeB, input->ne[0]);

    std::optional<uint32_t> completed;
    const int quantize_nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        ggml_iqk_quantize_src1(input, typeB, shared, (int) *task, quantize_nth);
        completed = task;
    }
    completed.reset();

    const int64_t nx = plan.merged ? gate_mat->ne[0]/2 : gate_mat->ne[0];
    const size_t up_stride = up_weight->nb[1];
    const size_t input_plane_size = input_row_size*input->ne[1];
    const size_t input_volume_size = input_plane_size*input->ne[2];

    const int64_t r2 = input->ne[2]/gate_weight->ne[2];
    const int64_t r3 = input->ne[3]/gate_weight->ne[3];
    const size_t gate_plane_size = gate_weight->nb[2];
    const size_t up_plane_size = up_weight->nb[2];

    bool success = true;
    const int mul_mat_nth = (int) state->latches[1].task_count();
    while (const auto task = state->latches[1].next(completed)) {
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
                            plan.limit, (int) *task, mul_mat_nth)) {
                    success = false;
                }
            }
        }
        completed = task;
    }

    return success;
}

static bool ggml_iqk_compute_forward_fused_id(
        const ggml_iqk_fusion_plan & plan,
        ggml_iqk_node_state * state,
        iqk_compute_state_shared * shared) {
    const ggml_tensor * gate_weight = plan.gate_weight;
    const ggml_tensor * up_weight   = plan.up_weight;
    const ggml_tensor * input       = plan.input;
    const ggml_tensor * ids         = plan.ids;
    const ggml_tensor * gate_mat    = plan.gate_mat;
    ggml_tensor * glu               = plan.glu;

    const int n_ids = ids->ne[0];
    const int n_as = gate_weight->ne[2];
    if (input->ne[3] != 1 || gate_weight->ne[3] != 1 || glu->ne[3] != 1) {
        return false;
    }

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(gate_weight->type);
    const void * input_data = shared->work_data;
    const size_t input_row_size = ggml_row_size(typeB, input->ne[0]);

    std::optional<uint32_t> completed;
    const int prepare_nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        const int task_id = (int) *task;

        ggml_iqk_quantize_src1(input, typeB, shared, task_id, prepare_nth);

        for (int64_t iid1 = task_id; iid1 < ids->ne[1]; iid1 += prepare_nth) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 < 0 || i02 >= n_as) {
                    std::memset((char *) glu->data + id*glu->nb[1] + iid1*glu->nb[2], 0, glu->ne[0]*sizeof(float));
                }
            }
        }

        if (task_id == 0) {
            state->matrix_rows.clear();
            state->matrix_rows.resize(n_as);
            state->active_experts.clear();
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int id = 0; id < n_ids; ++id) {
                    const int32_t i02 = *(const int32_t *) ((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                    if (i02 >= 0 && i02 < n_as) {
                        state->matrix_rows[i02].push_back({id, (int32_t) iid1});
                    }
                }
            }
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (!state->matrix_rows[cur_a].empty()) {
                    state->active_experts.push_back(cur_a);
                }
            }
            GGML_ASSERT(state->active_experts.size() <= UINT32_MAX/(uint32_t) state->chunks_per_expert);
            state->latches[1] = iqk_phase_latch((uint32_t) state->active_experts.size()*state->chunks_per_expert);
        }

        completed = task;
    }
    completed.reset();

    const int64_t nx = plan.merged ? gate_mat->ne[0]/2 : gate_mat->ne[0];
    const size_t up_stride = up_weight->nb[1];

    bool success = true;
    while (const auto task = state->latches[1].next(completed)) {
        const int cur_a = state->active_experts[*task/state->chunks_per_expert];
        const int chunk = *task%state->chunks_per_expert;
        const int64_t n_rows = state->matrix_rows[cur_a].size();

        const char * gate_data = (const char *) gate_weight->data + cur_a*gate_weight->nb[2];
        const char * up_data = (const char *) up_weight->data + cur_a*up_weight->nb[2];
        if (plan.merged) {
            up_data = gate_data + gate_weight->nb[2]/2;
        }
        if (!iqk_moe_fused_up_gate(nx, n_rows, input->ne[0], input->ne[1], plan.unary_op,
                    gate_weight->type, up_data, gate_data, up_stride,
                    typeB, input_data, input_row_size,
                    nullptr, nullptr, (float *) glu->data, glu->nb[1], glu->nb[2],
                    state->matrix_rows[cur_a].data(), plan.limit, chunk, state->chunks_per_expert)) {
            success = false;
        }
        completed = task;
    }

    return success;
}

static bool ggml_iqk_compute_forward_glu(const ggml_tensor * dst, ggml_iqk_node_state * state) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0 == nullptr || src1 == nullptr || src0->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        ggml_get_op_params_i32(dst, 1) != 0) {
        return false;
    }

    const ggml_glu_op op = ggml_get_glu_op(dst);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    bool success = true;
    std::optional<uint32_t> completed;
    const int nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        if (!iqk_compute_glu(src0->ne[0], ggml_nrows(src0), op, alpha, limit,
                (const float *) src0->data, src0->nb[1],
                (const float *) src1->data, src1->nb[1],
                (float *) dst->data, dst->nb[1], (int) *task, nth)) {
            success = false;
        }
        completed = task;
    }
    return success;
}

static bool ggml_iqk_compute_forward_clamp(const ggml_tensor * dst, ggml_iqk_node_state * state) {
    const float min = ggml_get_op_params_f32(dst, 0);
    const float max = ggml_get_op_params_f32(dst, 1);
    float * data = (float *) dst->data;
    const int64_t n = ggml_nelements(dst);
    std::optional<uint32_t> completed;
    const int nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        const int64_t dr = (n + nth - 1)/nth;
        const int64_t i0 = dr*(*task);
        const int64_t i1 = std::min<int64_t>(i0 + dr, n);
        for (int64_t i = i0; i < i1; ++i) {
            data[i] = std::max(min, std::min(max, data[i]));
        }
        completed = task;
    }
    return true;
}

static size_t ggml_iqk_build_compute_tasks(iqk_compute_state_shared * shared) {
    const int n_nodes = shared->cgraph->n_nodes;
    std::vector<int> fusion_start_plan(n_nodes, -1);
    std::vector<uint8_t> fusion_skip(n_nodes, 0);
    size_t work_size = 0;
    shared->fusion_plans.clear();

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = shared->cgraph->nodes[i];
        if (node->op != GGML_OP_GLU) {
            continue;
        }

        ggml_iqk_fusion_plan plan;
        if (!ggml_iqk_make_fusion_plan(shared->cgraph, i, &plan)) {
            continue;
        }

        const int plan_nodes[] = {
            plan.start_index, plan.gate_index, plan.up_index, plan.gate_view_index, plan.up_view_index, plan.glu_index,
            plan.gate_clamp_index, plan.up_clamp_index,
        };
        bool conflict = false;
        for (int node_index : plan_nodes) {
            if (node_index >= 0 && (fusion_skip[node_index] ||
                    fusion_start_plan[node_index] >= 0)) {
                conflict = true;
                break;
            }
        }
        if (conflict || fusion_start_plan[plan.start_index] >= 0) {
            continue;
        }

        const int plan_index = (int) shared->fusion_plans.size();
        shared->fusion_plans.push_back(plan);
        fusion_start_plan[plan.start_index] = plan_index;
        for (int node_index : plan_nodes) {
            if (node_index >= 0 && node_index != plan.start_index) {
                fusion_skip[node_index] = 1;
            }
        }
    }

    shared->compute_tasks.clear();
    shared->compute_tasks.reserve(n_nodes);
    const uint32_t n_threads = (uint32_t) shared->n_threads;
    for (int node_index = 0; node_index < n_nodes; ++node_index) {
        if (fusion_skip[node_index]) {
            continue;
        }

        const ggml_tensor * node = shared->cgraph->nodes[node_index];
        const int plan_index = fusion_start_plan[node_index];
        if (plan_index >= 0) {
            const ggml_iqk_fusion_plan & plan = shared->fusion_plans[plan_index];
            const int64_t nx = plan.merged ? plan.gate_mat->ne[0]/2 : plan.gate_mat->ne[0];
            const enum ggml_op op = plan.use_id ? GGML_OP_MUL_MAT_ID : GGML_OP_MUL_MAT;
            const std::array<uint32_t, 4> latch_sizes = {
                n_threads, ggml_iqk_task_count(op, shared->n_threads, nx), n_threads, n_threads,
            };
            shared->compute_tasks.emplace_back(true, plan_index, latch_sizes);

            if (plan.use_id) {
                shared->compute_tasks.back().node_state.chunks_per_expert =
                    ggml_iqk_expert_chunk_count(nx, shared->n_threads);
            }
            if (plan.input != nullptr && plan.input->type == GGML_TYPE_F32) {
                work_size = std::max(work_size,
                        ggml_iqk_src1_work_size(plan.input, ggml_iqk_vec_dot_type(plan.gate_weight->type)));
            }
        } else if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
            (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID ||
             node->op == GGML_OP_GLU || node->op == GGML_OP_CLAMP)) {
            const int64_t nx = node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID ? node->src[0]->ne[1] : 0;
            const std::array<uint32_t, 4> latch_sizes = {
                n_threads, ggml_iqk_task_count(node->op, shared->n_threads, nx), n_threads, n_threads,
            };
            shared->compute_tasks.emplace_back(false, node_index, latch_sizes);

            if (node->op == GGML_OP_MUL_MAT_ID) {
                shared->compute_tasks.back().node_state.chunks_per_expert =
                    ggml_iqk_expert_chunk_count(node->src[0]->ne[1], shared->n_threads);
            }

            if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
                const ggml_tensor * src1 = node->src[1];
                if (src1 != nullptr && src1->type == GGML_TYPE_F32) {
                    work_size = std::max(work_size,
                            ggml_iqk_src1_work_size(src1, ggml_iqk_vec_dot_type(node->src[0]->type)));
                }
            }
        }
    }

    return work_size;
}

static size_t ggml_iqk_src1_work_size(const ggml_tensor * src1, enum ggml_type type) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const size_t row_size = ggml_row_size(type, src1->ne[0]);
    return row_size * src1->ne[1] * src1->ne[2] * src1->ne[3];
}

static void ggml_iqk_quantize_src1(const ggml_tensor * src1,
    enum ggml_type type, iqk_compute_state_shared * shared,
    int ith, int nth) {
    ggml_from_float_t quantize_row = nullptr;
    switch (type) {
        case GGML_TYPE_Q8_2_X4:
            quantize_row = shared->backend->cpu_quantize_row_q8_2_x4;
            break;
        case GGML_TYPE_Q8_K128:
            quantize_row = shared->backend->cpu_quantize_row_q8_K128;
            break;
        case GGML_TYPE_Q8_K:
            quantize_row = shared->backend->cpu_quantize_row_q8_K;
            break;
        default:
            GGML_ABORT("%s: unsupported type %s\n", __func__, ggml_type_name(type));
    }
    GGML_ASSERT(quantize_row != nullptr);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const size_t row_size = ggml_row_size(type, src1->ne[0]);
    const size_t plane_size = row_size * src1->ne[1];
    const size_t volume_size = plane_size * src1->ne[2];

    for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            for (int64_t i11 = ith; i11 < src1->ne[1]; i11 += nth) {
                const float * src = (const float *)((const char *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1]);
                void * dst = shared->work_data + i13*volume_size + i12*plane_size + i11*row_size;
                quantize_row(src, dst, src1->ne[0]);
            }
        }
    }
}

static bool ggml_iqk_compute_forward_mul_mat(
    ggml_tensor * dst,
    ggml_iqk_node_state * state,
    iqk_compute_state_shared * shared) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(src0->type);
    const void * src1_data = shared->work_data;
    const size_t src1_row_size = ggml_row_size(typeB, src1->ne[0]);

    std::optional<uint32_t> completed;
    const int quantize_nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        ggml_iqk_quantize_src1(src1, typeB, shared, (int) *task, quantize_nth);
        completed = task;
    }
    completed.reset();

    bool success = true;
    const int mul_mat_nth = (int) state->latches[1].task_count();
    while (const auto task = state->latches[1].next(completed)) {
        if (!iqk_mul_mat_4d(ne01, ne11, ne00,
                ne02, ne03, ne12, ne13, nb02, nb03,
                src1_row_size*ne11, src1_row_size*ne11*ne12,
                nb2/sizeof(float), nb3/sizeof(float),
                src0->type, src0->data, nb01,
                typeB, src1_data, src1_row_size,
                (float *)dst->data, nb1/sizeof(float), (int) *task, mul_mat_nth)) {
            success = false;
        }
        completed = task;
    }

    return success;
}

static bool ggml_iqk_compute_forward_mul_mat_id(
            ggml_tensor * dst,
            ggml_iqk_node_state * state,
            iqk_compute_state_shared * shared) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_TENSOR_BINARY_OP_LOCALS

    if (ne2 == 0 || ne11 == 0) {
        return true;
    }

    const enum ggml_type typeB = ggml_iqk_vec_dot_type(src0->type);
    const int n_ids = ids->ne[0];
    const int n_as  = ne02;
    const void * src1_data = shared->work_data;
    const size_t src1_row_size = ggml_row_size(typeB, src1->ne[0]);

    std::optional<uint32_t> completed;
    const int prepare_nth = (int) state->latches[0].task_count();
    while (const auto task = state->latches[0].next(completed)) {
        const int task_id = (int) *task;

        ggml_iqk_quantize_src1(src1, typeB, shared, task_id, prepare_nth);

        for (int64_t iid1 = task_id; iid1 < ids->ne[1]; iid1 += prepare_nth) {
            for (int id = 0; id < n_ids; ++id) {
                const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                if (i02 < 0 || i02 >= n_as) {
                    std::memset((char *)dst->data + id*dst->nb[1] + iid1*dst->nb[2], 0, dst->ne[0]*sizeof(float));
                }
            }
        }

        if (task_id == 0) {
            state->matrix_rows.clear();
            state->matrix_rows.resize(n_as);
            state->active_experts.clear();
            for (int64_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int id = 0; id < n_ids; ++id) {
                    const int32_t i02 = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
                    if (i02 < 0 || i02 >= n_as) {
                        continue;
                    }
                    state->matrix_rows[i02].push_back({id, (int32_t) iid1});
                }
            }
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                if (!state->matrix_rows[cur_a].empty()) {
                    state->active_experts.push_back(cur_a);
                }
            }
            GGML_ASSERT(state->active_experts.size() <= UINT32_MAX/(uint32_t) state->chunks_per_expert);
            state->latches[1] = iqk_phase_latch((uint32_t) state->active_experts.size()*state->chunks_per_expert);
        }

        completed = task;
    }
    completed.reset();

    bool success = true;
    while (const auto task = state->latches[1].next(completed)) {
        const int cur_a = state->active_experts[*task/state->chunks_per_expert];
        const int chunk = *task%state->chunks_per_expert;
        const int64_t n_rows = state->matrix_rows[cur_a].size();

        const char * src0_cur = (const char *) src0->data + cur_a*nb02;
        if (!iqk_mul_mat_moe(ne01, n_rows, ne00, ne11,
                    src0->type, src0_cur, nb01,
                    typeB, src1_data, src1_row_size,
                    (float *)dst->data, nb1, nb2,
                    state->matrix_rows[cur_a].data(), chunk, state->chunks_per_expert)) {
            success = false;
        }
        completed = task;
    }

    return success;
}

static const char * ggml_backend_iqk_get_name(ggml_backend_t backend) {
    return "IQK";

    GGML_UNUSED(backend);
}

static void ggml_backend_iqk_free(ggml_backend_t backend) {
    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend->context;
    ctx->threadpool.stop();
    delete ctx;
    delete backend;
}

static void ggml_iqk_compute_node(iqk_compute_state_shared * shared) {
    for (size_t index = 0; index < shared->compute_tasks.size(); ++index) {
        ggml_iqk_compute_task & task = shared->compute_tasks[index];
        bool success = true;

        if (task.is_fused) {
            const ggml_iqk_fusion_plan & plan = shared->fusion_plans[task.index];
            success = plan.use_id ?
                ggml_iqk_compute_forward_fused_id(plan, &task.node_state, shared) :
                ggml_iqk_compute_forward_fused(plan, &task.node_state, shared);
        } else {
            ggml_tensor * node = shared->cgraph->nodes[task.index];
            ggml_iqk_node_state * state = &task.node_state;
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    success = ggml_iqk_compute_forward_mul_mat(node, state, shared);
                    break;
                case GGML_OP_MUL_MAT_ID:
                    success = ggml_iqk_compute_forward_mul_mat_id(node, state, shared);
                    break;
                case GGML_OP_GLU:
                    success = ggml_iqk_compute_forward_glu(node, state);
                    break;
                case GGML_OP_CLAMP:
                    success = ggml_iqk_compute_forward_clamp(node, state);
                    break;
                default:
                    GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            }
        }

        if (!success) {
            shared->ec.store(1, std::memory_order_relaxed);
        }
    }
}

static ggml_status ggml_backend_iqk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend->context;

    iqk_compute_state_shared shared = {};
    shared.backend = ctx;
    shared.cgraph = cgraph;

    GGML_ASSERT(ctx->n_threads > 0 && !ctx->threadpool.threads.empty());
    shared.n_threads = std::min(ctx->n_threads, (int) ctx->threadpool.threads.size());
    const size_t work_size = ggml_iqk_build_compute_tasks(&shared);

    // allocate the work buffer on demand
    if (ctx->work_size < work_size) {
        GGML_LOG_INFO("IQK: compute enlarge word data [%zu] -> [%zu]\n", ctx->work_size, work_size);
        ctx->work_data.reset(new char[work_size]);
        ctx->work_size = work_size;
    }

    shared.work_data = ctx->work_data.get();

    if (!ctx->threadpool.compute(shared)) {
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

    // grow the persistent threadpool on demand
    ctx->threadpool.resize(n_threads);
}

static void ggml_backend_iqk_set_threadpool_params(ggml_backend_t backend_iqk, const ggml_threadpool_params * params) {
    GGML_ASSERT(ggml_backend_is_iqk(backend_iqk));

    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend_iqk->context;
    ctx->threadpool.set_params(*params);
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
        case GGML_OP_CLAMP:
            // CLAMP is a view sharing the src data; it is either fused into a
            // GLU plan (and skipped) or computed in-place here
            ggml_iqk_debug_support(op, true, "clamp");
            return true;
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

static void ggml_backend_iqk_assign_cpu_reg(ggml_backend_t backend_iqk, ggml_backend_reg_t cpu_reg) {
    GGML_ASSERT(ggml_backend_is_iqk(backend_iqk));

    ggml_backend_iqk_context * ctx = (ggml_backend_iqk_context *)backend_iqk->context;
    ctx->cpu_reg = cpu_reg;
    ctx->cpu_get_type_traits = nullptr;
    ctx->cpu_quantize_row_q8_K = (ggml_from_float_t) quantize_row_q8_K_ref;
    ctx->cpu_quantize_row_q8_K128 = (ggml_from_float_t) quantize_row_q8_K128_ref;
    ctx->cpu_quantize_row_q8_2_x4 = quantize_row_q8_2_x4_ref;
    if (cpu_reg) {
        ctx->cpu_get_type_traits = (ggml_backend_cpu_get_type_traits_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_get_type_traits_cpu");
        if (!ctx->cpu_get_type_traits) {
            ctx->cpu_get_type_traits = (ggml_backend_cpu_get_type_traits_t)
                ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_cpu_get_type_traits");
        }
        if (ctx->cpu_get_type_traits) {
            std::string names;
            const struct ggml_type_traits_cpu * traits = ctx->cpu_get_type_traits(GGML_TYPE_Q8_K);
            if (traits && traits->from_float) {
                ctx->cpu_quantize_row_q8_K = traits->from_float;
                names.append(" quantize_row_q8_K");
            }
            traits = ctx->cpu_get_type_traits(GGML_TYPE_Q8_K128);
            if (traits && traits->from_float) {
                ctx->cpu_quantize_row_q8_K128 = traits->from_float;
                names.append(" quantize_row_q8_K128");
            }
            traits = ctx->cpu_get_type_traits(GGML_TYPE_Q8_2_X4);
            if (traits && traits->from_float) {
                ctx->cpu_quantize_row_q8_2_x4 = traits->from_float;
                names.append(" quantize_row_q8_2_x4");
            }
            if (!names.empty()) {
                GGML_LOG_INFO("IQK: registed cpu_path:%s\n", names.c_str());
            }
        }
    }
}

static void * ggml_backend_iqk_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_iqk_set_n_threads;
    }
    if (std::strcmp(name, "ggml_backend_assign_cpu_reg") == 0) {
        return (void *) ggml_backend_iqk_assign_cpu_reg;
    }
    if (std::strcmp(name, "ggml_backend_iqk_set_threadpool_params") == 0) {
        return (void *) ggml_backend_iqk_set_threadpool_params;
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

#pragma once

#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

using ggml_get_type_traits_cpu_fn = const struct ggml_type_traits_cpu * (*)(enum ggml_type);

static std::function<const struct ggml_type_traits_cpu *(enum ggml_type)> get_type_traits_cpu;

#ifdef GGML_BACKEND_DL
#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

class cpu_backend {
public:
#ifdef _WIN32
    using handle_t = HMODULE;
#else
    using handle_t = void *;
#endif
    using score_fn_t = int (*)();
    using init_fn_t = ggml_backend_reg_t (*)();

    explicit cpu_backend(handle_t handle) : handle_(handle) {
        const auto score_fn = reinterpret_cast<score_fn_t>(get_proc("ggml_backend_score"));
        score_ = score_fn ? score_fn() : 0;
    }

    cpu_backend(const cpu_backend &) = delete;
    cpu_backend & operator=(const cpu_backend &) = delete;

    ~cpu_backend() {
        if (handle_ == nullptr) {
            return;
        }
#ifdef _WIN32
        FreeLibrary(handle_);
#else
        dlclose(handle_);
#endif
    }

    int score() const {
        return score_;
    }

    ggml_get_type_traits_cpu_fn traits = nullptr;

    bool init() {
        const auto backend_init = reinterpret_cast<init_fn_t>(get_proc("ggml_backend_init"));
        const ggml_backend_reg_t reg = backend_init ? backend_init() : nullptr;
        if (reg == nullptr) {
            return false;
        }

        traits = reinterpret_cast<ggml_get_type_traits_cpu_fn>(
            ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu"));
        return traits != nullptr;
    }

private:
    void * get_proc(const char * name) const {
#ifdef _WIN32
        return reinterpret_cast<void *>(GetProcAddress(handle_, name));
#else
        return dlsym(handle_, name);
#endif
    }

    handle_t handle_ = nullptr;
    int score_ = 0;
};
#endif

static void init_cpu_backend([[maybe_unused]] const std::string & variant) {
    if (get_type_traits_cpu) {
        return;
    }

#ifdef GGML_BACKEND_DL
#if defined(GGML_TEST_BACKEND_DIR)
    const std::filesystem::path dir = std::filesystem::u8path(GGML_TEST_BACKEND_DIR);
#elif defined(GGML_BACKEND_DIR)
    const std::filesystem::path dir = std::filesystem::u8path(GGML_BACKEND_DIR);
#else
    const std::filesystem::path dir = std::filesystem::current_path();
#endif

#ifdef _WIN32
    const std::string prefix = "ggml-cpu";
    const std::string extension = ".dll";
#else
    const std::string prefix = "libggml-cpu";
    const std::string extension = ".so";
#endif

    const std::string base_name = prefix + extension;
    const std::string variant_prefix = prefix + "-";
    const std::string requested_name = variant_prefix + variant + extension;
    std::shared_ptr<cpu_backend> best;
    const auto load = [](const std::filesystem::path & path) -> cpu_backend::handle_t {
#ifdef _WIN32
        return LoadLibraryW(path.wstring().c_str());
#else
        return dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    };

    std::error_code ec;
    std::filesystem::directory_iterator it(dir, std::filesystem::directory_options::skip_permission_denied, ec);
    const std::filesystem::directory_iterator end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::filesystem::path path = it->path();
        std::error_code file_ec;
        if (!it->is_regular_file(file_ec) || file_ec) {
            continue;
        }

        const std::string filename = path.filename().string();
        const bool is_cpu_backend = filename == base_name ||
            (filename.rfind(variant_prefix, 0) == 0 && path.extension().string() == extension);
        if (!is_cpu_backend || (!variant.empty() && filename != requested_name)) {
            continue;
        }

        const auto handle = load(path);
        if (handle == nullptr) {
            continue;
        }

        auto candidate = std::make_shared<cpu_backend>(handle);
        std::printf("loaded CPU backend %s, score %d\n", filename.c_str(), candidate->score());
        if (candidate->score() > 0 && (!best || candidate->score() > best->score())) {
            best = std::move(candidate);
        }
    }

    if (best && !best->init()) {
        best.reset();
    }

    if (!best) {
        std::fprintf(stderr, "error: no supported CPU backend found");
        if (!variant.empty()) {
            std::fprintf(stderr, " for variant '%s'", variant.c_str());
        }
        std::fprintf(stderr, "\n");
        std::exit(EXIT_FAILURE);
    }

    get_type_traits_cpu = [backend = std::move(best)](enum ggml_type type) {
        return backend->traits(type);
    };
#else
    ggml_cpu_init();
    get_type_traits_cpu = ggml_get_type_traits_cpu;
#endif
}

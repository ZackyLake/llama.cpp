#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t ggml_backend_iqk_init(void);

GGML_BACKEND_API bool ggml_backend_is_iqk(ggml_backend_t backend);

GGML_BACKEND_API void ggml_backend_iqk_set_n_threads(ggml_backend_t backend_iqk, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_iqk_reg(void);

#ifdef __cplusplus
}
#endif

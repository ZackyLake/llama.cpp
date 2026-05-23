# Vulkan Quantized MUL_MAT Notes

This note summarizes the current Vulkan-side `mul_mat` and `mul_mat_id` structure for quantized inference after the grouped `iqk`, `iqks`, and `iqkt` bring-up. CUDA remains the semantic reference. In practice, the most useful anchors are:

- `ggml/src/ggml-cuda/vecdotq.cuh`
- `ggml/src/ggml-cuda/mmq.cuh`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
- `ggml/src/ggml-vulkan/vulkan-shaders/`

## Execution Paths

For quantized `mul_mat` and `mul_mat_id`, Vulkan still splits execution into three buckets:

- `MMQ`: matrix-matrix style kernels for larger `n`
- `MMVQ` or `DMMV`: matrix-vector style kernels for small `n`, especially `n == 1`
- Generic pre-dequant fallback: convert `src0` to `fp16` or `fp32`, then run a float path

For `mul_mat_id`, there is an extra front-end choice between the vector-style `vec_id` path and the matrix-style `matmul_id` path. The current runtime sends quantized `src0` through `vec_id` when `ids->ne[1] <= 8`, and then may still quantize the F32 RHS to `Q8_1` if integer dot is available and the MMVQ heuristic allows it.

The CUDA implementation is still the semantic source of truth, but Vulkan does not share CUDA's warp model or memory layout assumptions. Packed integer dot maps well to `GL_EXT_integer_dot_product`; row-aware addressing, tile layout, and reduction structure still need Vulkan-specific GLSL.

## Current Shader Families

The Vulkan quantized implementation is now organized around three grouped shader families:

- `iqk`: regular `IQ2_K`, `IQ3_K`, `IQ4_K`, `IQ5_K`, `IQ6_K`
- `iqks`: `IQ4_KS`, `IQ4_KSS`, `IQ5_KS`, `IQ2_KS`, `IQ3_KS`, `IQ2_KL`
- `iqkt`: `IQ1_KT`, `IQ2_KT`, `IQ3_KT`, `IQ4_KT`

This grouping is intentional. The key question is not whether a type has its own filename, but whether its packed layout and row semantics are close enough to share one shader family without hiding byte-addressing rules.

## Row-Metadata Formats Change the Rules

The `iqks` and `iqkt` families are row-metadata formats. Their physical row stride is not a simple function of block count, so packed-byte addressing must be based on:

```c
ggml_row_size(type, ne0)
```

not on formulas derived only from `ggml_type_size(type)` and `ggml_blck_size(type)`.

For Vulkan, that rule affects all of the following:

- `src0` descriptor range sizing
- temporary packed buffer sizes
- row and batch stride push constants
- contiguity checks
- pre-dequant shaders
- `MMVQ`, `MMQ`, and `mul_mat_id` indexing

Supporting only one of those surfaces is not enough. A row-metadata type can look correct in one path and still fail when runtime selection switches to another path.

One bug found during the KT bring-up was unrelated to decode math itself: `prealloc_y` reuse was keyed only by tensor pointer, which is not sufficient for views. That made stale temporaries look like quant decode corruption. The fix was to treat view identity as part of the cache key instead of assuming pointer identity is enough.

## Family Notes

### Regular IQK (`iqk`)

The regular `IQ*_K` family now shares:

- `dequant_iqk.comp`
- `mul_mat_vec_iqk.comp`
- `mul_mm_iqk.comp`

This grouped family covers `IQ2_K` through `IQ6_K`, including `Q8_1` fast paths for `mul_mat`, `mul_mat_id`, and native matrix kernels.

Two recent details matter here:

- The regular `IQ2_K` and `IQ3_K` `Q8_1` correctness bug was not a host-routing problem. The real issue was shader-side B indexing: the second 128-element half of a superblock must add `4 * ib128` when addressing the B-side `Q8_1` blocks in both `mul_mat_vec_iqk.comp` and `mul_mm_iqk.comp`.
- `IQ6_K` is fully integrated into the same grouped family. Its Vulkan `Q8_1` decode and dot logic follows CUDA's `vec_dot_iq6_k_q8_1` and `load_tiles_iq6_k` structure rather than a CPU-style dequant-first interpretation.

### IQKS / KL / KSS (`iqks`)

The row-metadata `iqks` family currently shares:

- `dequant_iqks.comp`
- `mul_mat_vec_iqks.comp`
- `mul_mm_iqks.comp`

This family covers `IQ4_KS`, `IQ4_KSS`, `IQ5_KS`, `IQ2_KS`, `IQ3_KS`, and `IQ2_KL`.

One important cleanup landed here: `IQ4_KS` no longer keeps a standalone dequant shader file. Its row-aware dequant branch now lives inside `dequant_iqks.comp`, which keeps the dequant side aligned with the already-shared `mul_mat_vec_iqks.comp` implementation.

This family now also has a dedicated grouped native matrix shader family, plus matching runtime generation / allowlists for `mul_mat`, `mul_mat_id`, and `Q8_1` fast paths.

The important constraint did not change: `iqks` native matrix kernels still have to use byte-based row addressing end to end. A recent `IQ2_KS` bring-up bug here was caused by reading the high scale bit from the wrong part of the 16-bit `extra` field. For each 128-element half, the low nibble selects the lookup-table half, while the high nibble carries the extra scale bits. `mul_mm_iqks.comp` now matches the existing `dequant_iqks.comp` layout and the CUDA reference on that point.

### KT (`iqkt`)

The `iqkt` family shares:

- `dequant_iqkt.comp`
- `mul_mat_vec_iqkt.comp`
- `mul_mm_iqkt.comp`

This grouped family covers `IQ1_KT`, `IQ2_KT`, `IQ3_KT`, and `IQ4_KT`.

All four variants now have grouped dequant, grouped vec-style kernels, grouped native matrix kernels, `Q8_1` MMQ support, and `mul_mat_id` shader generation/runtime allowlists. The right mental model is still "dedicated row-aware family" rather than "generic flat-block helper with a few special cases".

## CUDA-Specific Features vs Vulkan

CUDA `IQ_K*` kernels use several CUDA-specific implementation details:

- packed integer dot in the style of `dp4a`
- warp-local reductions and warp-synchronous execution
- lane-oriented loads tuned for CUDA scheduling

Vulkan can reproduce the packed integer dot part, but not CUDA's warp semantics. The practical consequence is:

- block-regular families can often share generic packed-dot structure once indexing is correct
- row-metadata families still need explicit row-aware byte addressing first
- for `Q8_1` fast paths, the CUDA reference should be the vector dot and tile-loading code, not the CPU dequant loop

That last point matters. The regular `IQ2_K`, `IQ3_K`, and `IQ6_K` Vulkan work was stabilized by matching CUDA's `vec_dot_*_q8_1` and `load_tiles_*` behavior rather than by reasoning from CPU dequant output alone.

## Runtime Heuristics Still Matter

Even after a correct shader exists, runtime heuristics can still hide it.

The current `ggml_vk_should_use_mmvq()` heuristic explicitly allowlists the following families for `Q8_1` MMVQ when `k >= 256`:

- regular `IQ2_K` through `IQ6_K`
- `IQ4_KS`, `IQ4_KSS`, `IQ5_KS`, `IQ2_KS`, `IQ3_KS`, `IQ2_KL`
- `IQ1_KT` through `IQ4_KT`

This is important because shader existence alone is not enough. If runtime disables `quantize_y`, bypasses `vec_id`, or falls back to float dequant too early, both performance and bug diagnosis become misleading.

The regular `IQ2_K` / `IQ3_K` bring-up is the clearest example: a temporary host-side workaround could avoid failures, but it also hid the real shader bug and caused a large performance regression.

There is still one open performance point worth documenting: on Intel Arc B580, `IQ6_K` in `MUL_MAT_ID_FUSION` with `n == 1` is materially slower than `IQ4_K` and `IQ5_K` even though Vulkan is already taking the intended `vec_id + Q8_1 MMVQ` route and the grouped `IQ6_K` shader matches CUDA indexing. That should be treated as kernel-tuning work, not as a missing-MMVQ or host-routing bug.

## Validation Strategy

`test-backend-ops` is expensive, especially in `perf` mode. Prefer the narrowest possible `-p` filter that exercises only the quant type or family you are touching.

Good patterns:

- rebuild only the affected target:

```powershell
cmake --build build --config Release --target ggml-vulkan
```

- single-type correctness checks:

```powershell
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p "type_a=iq4_ks,type_b=f32"
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p "type_a=iq6_k,type_b=f32"
```

- small-family correctness checks when the change is shared across a grouped family:

```powershell
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p "type_a=iq[23]_k,type_b=f32"
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT_ID -p "type_a=iq[23]_k,type_b=f32"
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p "iq4_kss|iq2_ks|iq3_ks|iq4_ks|iq5_ks|iq2_kl"
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT_ID -p "iq4_kss|iq2_ks|iq3_ks|iq4_ks|iq5_ks|iq2_kl"
build/bin/Release/test-backend-ops.exe test -b Vulkan1 -o MUL_MAT -p "type_a=iq[1-4]_kt,type_b=f32"
```

- targeted perf checks for one suspected regression:

```powershell
build/bin/Release/test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT_ID_FUSION -p "type_a=iq6_k,.*,n=1"
```

- only compare a small peer set when cross-type context is actually needed:

```powershell
build/bin/Release/test-backend-ops.exe perf -b Vulkan1 -o MUL_MAT_ID_FUSION -p "type_a=iq[3456]_k,.*,n=1"
```

Avoid broad filters like `type_a=iq.*` in perf mode during normal iteration. Those are useful only when the goal is an explicit cross-family comparison.

## Guidance for Future Quantized Vulkan Work

When adding another `K` / `KS` / `KT` style quant type to Vulkan, treat the task as an end-to-end audit instead of a single shader port.

Checklist:

1. Start from the packed physical layout and decide whether the type belongs in an existing grouped family or needs a new row-aware family.
2. For row-metadata types, base packed-byte math on `ggml_row_size(type, ne0)` and audit row stride, batch stride, and contiguity checks together.
3. Keep pre-dequant, `MMVQ`, `MMQ`, and `mul_mat_id` in sync. Passing one path does not validate the others.
4. Use CUDA `vec_dot_*_q8_1` and `load_tiles_*` as the reference for `Q8_1` fast paths. Do not derive those paths from CPU dequant logic.
5. Validate runtime reachability as well as shader correctness. A correct shader that never gets selected is still a broken feature.
6. Keep `test-backend-ops` filters focused on the touched quant type or family. Large perf sweeps are too expensive to use as the default iteration loop.
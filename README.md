# DINO on the car: export-side latency work

How the DINOv3-B foundation model (`dino-foundation-e2e_train-903ak8:step_600`) went from
**42.4 ms to 26.9 ms** of TensorRT engine time on the gen1 RTX 4080 without retraining, and
what was tried and rejected along the way.

> **Status (2026-10-03): on the car.** `aoligei_map` (FP16 encoder + dense map seg) is on
> `1stage/09_25_2026_test` ([#133922](https://github.com/AppliedNeuron/core-stack/pull/133922))
> and was test driven: the gen1 stack runs it at **10 Hz**, and driving behavior is decent for a
> foundation model trained on a ~3 month old recipe. See [On the car](#on-the-car).

- Training run: [research/hybrid_rl/dino-foundation-e2e_train-903ak8](https://appliedintuition.wandb.io/research/hybrid_rl/runs/dino-foundation-e2e_train-903ak8)
- Export branch: [`export-dino-903ak8`](https://github.com/AppliedNeuron/core-stack/tree/export-dino-903ak8) (core-stack)
- Pipeline: [mosaic/ml-opt-suite](https://buildkite.com/mosaic/ml-opt-suite), `export_validate_profile`, STRONG (strongly typed) TRT 10.8, CC89
- Reference: chip_shot (ResNet-50 1-stage, [build 4990](https://buildkite.com/mosaic/ml-opt-suite/builds/4990))

## Results

| engine | 4080 engine time (suite per-layer sum) | local p50 (4070 Ti SUPER) | dADE vs BF16 (m, 95% CI) | det recall vs BF16 | status |
|---|---|---|---|---|---|
| BF16 baseline ([5014](https://buildkite.com/mosaic/ml-opt-suite/builds/5014)) | 42.4 ms | 44.4 ms | ref | 1.000 | |
| **aoligei** ([5018](https://buildkite.com/mosaic/ml-opt-suite/builds/5018)) | **26.9 ms** | 27.8 ms | +0.027 ± 0.031 | 0.973 | on `1stage/09_25_2026_test` ([#133889](https://github.com/AppliedNeuron/core-stack/pull/133889)) |
| **aoligei_map** (aoligei + dense map seg) | ~30 ms (est.: aoligei + ~3 ms seg branch) | ~31 ms (est.) | planner path identical to aoligei | identical to aoligei | **on the car at 10 Hz** ([#133922](https://github.com/AppliedNeuron/core-stack/pull/133922), build [5028](https://buildkite.com/mosaic/ml-opt-suite/builds/5028)) |
| aoligei + INT8 qkv/fc1 | — | 24.4 ms | +0.028 ± 0.027 | 0.974 | flag ready; next lever on top of aoligei_map |
| aoligei + INT8 qkv/fc1/fc2 ([5023](https://buildkite.com/mosaic/ml-opt-suite/builds/5023)) | 22.8 ms | 22.3 ms | −0.017 ± 0.025 | 0.956 | not shipped (perception drift) |
| chip_shot (ResNet-50) | — | 17.1 ms | — | — | reference |

dADE / det recall: 100 validation scenarios, ground-truth harness (see [Measurement](#measurement)).
The 4070 Ti SUPER runs the same engine about 1.24x slower than the 4080. CUDA graphs are
already on onboard (`use_cuda_graph: true` in `strongly_typed_inference`), so the car runs
slightly under the non-graph numbers here.

Where the time went in the BF16 baseline (4080, per-layer sum): DINOv3 backbone 27.0, neck 3.0,
decoder 4.1 (tokenizer 2.2, planner 1.8), det head 2.9, map head 2.3, preprocessor 0.9 ms.
By op type: linear 17.4, attention 7.6, conv 5.9 ms. The heads cost about the same as in
chip_shot; the gap is the ViT.

## What shipped (aoligei)

Three changes, all in the export path, none touching weights. Commit
[`7aa458b`](https://github.com/AppliedNeuron/core-stack/commit/7aa458b33842fef483c7b9b8be9761d0a3caef20).

### 1. FP16 image encoder (most of the gain: −12 ms local, ~−9.5 ms on the 4080)

The STRONG export is BF16 end to end, and in a strongly typed build the ONNX dtypes *are* the
engine dtypes. On Ada GeForce (RTX 4080), FP16 tensor-core GEMMs run at 2x the BF16 rate, so the
encoder is retyped to FP16 at the ONNX level before the TRT build:

- `onnx_node_utils.retype_bf16_scope_to_fp16(model, scope)` rewrites every BF16 tensor, cast and
  constant inside `pipeline_modules/img_enc/` to FP16 and puts named boundary casts at the scope
  edge (`{scope}{src}__{suffix}_cast`), so heads keep their exported dtypes.
- `onnx_node_utils.unbracket_fp32_in_scope(model, scope)`: the export wraps ConvTranspose/Resize in
  FP32 casts because BF16 has no deconv kernel. Inside the FP16 scope those brackets are
  retargeted to FP16 (the FPN `deconv_3`, 768→768, was a visible FP32 island).
- Build flag: `--tensorrt_build_config.fp16_onnx_scope pipeline_modules/img_enc/`
  (`OneStageTensorRTBuildConfig.fp16_onnx_scope`). Transform order:
  bracket → retype → unbracket.

Effect: backbone GEMM/attention/MLP time down ~40%, plus ~5 ms of BF16 cast/transpose glue
between blocks removed. FP16 has more mantissa than BF16, so the drift against the BF16 PyTorch
reference (FDE@4s 2.7 → 10.3 cm, 2/50 mode flips) is a divergence, not an error: against ground
truth the change is within noise (dADE +0.001 ± 0.019 on 30 scenarios, +0.027 ± 0.031 on 100).

Tried and rejected in this family:

| variant | local p50 | why not |
|---|---|---|
| builder optimization level 5 | 44.7 (vs 44.5) | no gain |
| non-strong FP16 everywhere | 28.8 | unvalidated; map head is FP32 by design, loses STRONG precision control |
| non-strong FP16, heads pinned FP32 | 40.7 | planner forced BF16 → FP32 makes the heads slower than baseline |
| ViT blocks FP16, LayerNorm/Softmax kept BF16 | 80.8 | NaN outputs; casts break attention fusion |
| whole model FP16 | −0.24 ms vs encoder-only | not worth the head-precision risk |

Where the time goes after FP16 (fused-layer attribution, local, FP16-encoder engine with seg):
backbone 20.2, det head 3.9, map head 3.8 (of which ~3 is the dense seg branch), neck 3.1,
tokenizer 2.9, planner 2.2, other 2.5 ms. The ViT is still two thirds of the engine.

Parity against the BF16 PyTorch model (suite-style validation, 5 scenarios x 10 frames). These
measure divergence from BF16, not error: FP16 carries more mantissa than BF16.

| | BF16 STRONG | FP16 image encoder | FP16 ViT blocks only |
|---|---|---|---|
| mean FDE@1s | 0.4 cm | 1.4 cm | 1.4 cm |
| mean FDE@4s | 2.7 cm | 10.4 cm | 10.2 cm |
| worst scenario FDE@4s | 4.4 cm | 27.8 cm | 27.5 cm |
| mode-selection flips | 0/50 | 2/50 | 2/50 |
| map seg label mismatch | 0.12% | 0.71% | 0.71% |

The ViT blocks alone account for all of the drift; the FPN/neck in FP16 adds nothing measurable.

### 2. Batched PETR position embeddings (−0.5 to −1.7 ms, lossless)

`PVFeatureTokenizer` built the PETR 3D position embedding in a per-camera loop: 7 calls, each a
few hundred tiny FP32 ops in the engine. `_camera_runs_by_level` now groups runs of adjacent
cameras that share an FPN level (scale index `[2,3,3,3,4,4,4]` → 3 calls). Identical math
(equivalence unit test added); the engine loses ~600 small kernels.

### 3. Drop the dense map seg branch (−3 ms) — reverted for closed loop

The map head is a sparse instance decoder (feeds the planner through the top-256 map instance
tokens) plus a dense BEV seg branch (`feature_transforms.map_grid` rasterization + Hardswish
upsampler → `map_seg_*`). The planner does not read the seg output, so removing it leaves map
instance features bit-identical and saved ~3 ms.

However, **closed-loop testing requires the map seg outputs** (safety). `aoligei_map` restores
the branch ([`7c6714a`](https://github.com/AppliedNeuron/core-stack/commit/7c6714a33f6bb9519cd4dfc8ffc40ea5719e3367)),
keeping changes 1 and 2. The head is 8 channels (drivable area + 7 static map types, no
lane-color channel), so the onboard entry uses the tethys preset set:
`e2e_1stage_with_map_7cam_trajectory_dense` / `..._dense_drivable_no_intention` /
`runtime_config_steering_angle_map_dense_drivable` / `waypoint_q2_extended_2x_fov`.

## On the car

| model | contents | branch | result |
|---|---|---|---|
| `aoligei` | FP16 encoder, batched PETR, no map seg | [#133889](https://github.com/AppliedNeuron/core-stack/pull/133889) | not closed-loop testable: no `map_seg_*` |
| `aoligei_map` | FP16 encoder, batched PETR, dense map seg (8ch drivable) | [#133922](https://github.com/AppliedNeuron/core-stack/pull/133922) | **10 Hz on the gen1 stack PC; driving behavior decent** |

- Closed-loop testing needs the map seg outputs (safety), so the seg branch is the price of the
  road: ~3 ms of engine time.
- The 10 Hz is the whole stack, not just the engine: the TRT engine time is ~30 ms of the 100 ms
  frame, and the stack runs it under CUDA graphs (`use_cuda_graph: true`), which comes in slightly
  under the profiler number.
- The recipe behind 903ak8 is ~3 months old. Behavior that is already decent points at the DINO
  backbone carrying a lot; a re-run on the current recipe is the natural next comparison.
- Cloud CI note: a `gpu-g6` export (L4) profiles the same engine at 65.7 ms. The L4 is much slower
  than the car's 4080, so use the on-prem 4080 queue (`ml_optimization_gpu_4080`) for latency
  numbers and g6 only as a CC89 build host.

### Next levers

1. **INT8 qkv/fc1** on top of aoligei_map: about −3.4 ms local, same open-loop accuracy and
   detection recall; build flag ready, needs a fresh calibration JSON for any new checkpoint.
2. **Lower ViT resolution** (biggest lever, ~−9 ms): needs a short fine-tune; if the next DINO run
   is a retrain on the current recipe, training at 352x640 from the start removes the extra step.
3. **End-to-end breakdown**: split the 100 ms frame into preprocessing, engine and postprocessing
   from a drive log to see where the remaining headroom is.

## INT8 SmoothQuant for the ViT (built, not shipped)

Commits [`6ecf7ac`](https://github.com/AppliedNeuron/core-stack/commit/6ecf7ac12ea40eef27e05107f463ef3938b6522f),
[`658a75e`](https://github.com/AppliedNeuron/core-stack/commit/658a75ea1c13e7d2a1454295c221631cac289b61).

DINO LayerNorm outputs carry massive outlier channels (max |x| 70–124 vs p99.99 ≈ 5 in later
blocks; one GELU output reaches 1075), so plain per-tensor INT8 is unusable. SmoothQuant
(α = 0.5) folds a per-channel factor into the LayerNorm scale/bias and the next MatMul's weight
rows — exact in float — after which activation max/p99.99 ratios are ≤ ~4.

- `onnx_node_utils.apply_int8_smoothquant(model, calibration)` inserts explicit Q/DQ:
  activations per-tensor, weights per output channel (axis 1 of `[in, out]`), FP16 scales.
  Post-smoothing activation amax is computed exactly as `max(a / s)`.
- Calibration: 80 frames dumped from the eval harness; JSONs in
  `onroad/ml_optimization/experimental/one_stage/calibration/`:
  - `dino_903ak8_img_enc_int8_qkv_fc1.json` — conservative: 24 LayerNorm-fed MatMuls (qkv, fc1)
  - `dino_903ak8_img_enc_int8.json` — full: + 11 fc2 (GELU-fed) MatMuls, block 2 excluded
- Build flag: `--tensorrt_build_config.int8_calibration_path <json>` (requires `fp16_onnx_scope`).
  The ranges belong to one checkpoint; re-calibrate for any other.

| variant | local p50 | dADE | det recall | verdict |
|---|---|---|---|---|
| aoligei | 27.8 | +0.027 ± 0.031 | 0.973 | — |
| + INT8 qkv, fc1 | **24.4** | +0.028 ± 0.027 | 0.974 | keep: same accuracy, −3.4 ms |
| + INT8 attention proj (V-folded smoothing) | 25.5 | — | — | slower: Q/DQ breaks MHA+proj fusion |
| + INT8 fc2, per-tensor p99.99 clip | 22.2 | +0.024 ± 0.037 | 0.921 | reject: perception drifts |
| + INT8 fc2, full amax (suite 5023) | 22.3 | −0.017 ± 0.025 | 0.956 | marginal |
| INT8 neck | slower | — | — | reject |
| FP8 attention (MHA) | −0.6 ms | — | — | reject: TRT forbids INT8 + FP8 in one engine |

INT8 is a pure stack on top of aoligei_map: with aoligei_map now running at 10 Hz on the car,
the qkv/fc1 level is the next candidate if more headroom is needed.

## Resolution (needs a fine-tune; not launched)

The DINOv3 backbone now takes `input_size: [H, W]` (resize before patchify) and
`low_res_cameras` + `low_res_input_size` (listed cameras encoded smaller, patch features resized
to the common grid) — commit [`c07d276`](https://github.com/AppliedNeuron/core-stack/commit/c07d276f1f9097fa3bc4e3871fa7479df6ad3b01).

| ViT input | per-layer engine time (local, FP16 enc) | traj dADE vs native (no retrain) | det recall vs native |
|---|---|---|---|
| native (aoligei) | 31.4 ms | ref | ref |
| 384x704 | 25.7 ms | — | 0.74 |
| 352x640 | 22.0 ms | −0.031 ± 0.042 | 0.58 |
| 320x576 | 18.8 ms | — | 0.43 |
| 288x512 | 16.4 ms | +0.070 ± 0.064 | — |
| mixed: front full, others low | — | — | front 0.81 |
| rear cameras only low | — | — | rear 0.40 |

Trajectories survive, detections do not. A short resolution-adaptation fine-tune (warm start
903ak8, `input_size: [352, 640]`, img_enc + det/map heads trainable) is a config change; 352x640
with INT8 would land near chip_shot (~19 ms local).

## Measurement

- **Suite metric**: per-layer GPU time summed over runs (IProfiler) — what ml-opt-suite reports
  as "Total latency per run". Used for the 4080 numbers.
- **Wall clock**: CUDA events around `execute_async_v3`, inputs on device, 300 runs.
- **GPU clocks drift ~10% between runs** (same engine file: 30.5 then 33.6 ms), so serial
  timings cannot resolve 1–3 ms. All comparisons use interleaved A/B: several engines loaded,
  timed in alternating rounds; rebuilds of one ONNX agree within 0.2 ms.
- **Ground-truth harness**: the suite's own validation compares an engine to its own PyTorch
  model, which says nothing about lossy changes. The harness replays validation scenarios
  through the suite's TensorRT adapter (temporal caches carried frame to frame), scores
  `predicted_trajectory` against the logged ego future (`target_trajectory`), and reports paired
  per-frame dADE with a 95% CI plus detection recall vs the BF16 engine, split front / other /
  rear sectors.
- Per-layer profiling inflates wall clock by ~14%, and the 4070 Ti SUPER runs the same engine
  ~1.24x slower than the 4080 (per-layer sum), so local wall clock / 1.24 ≈ 4080 wall clock.
- Local loop: RTX 4070 Ti SUPER, TRT 10.8.0.43 + the deformable aggregation plugin from the
  same source tree, no Bazel build needed per iteration. The loop: rewrite the suite's ONNX
  (retype / unbracket / Q/DQ insertion), build with the suite's builder settings, time with
  interleaved A/B, then score with the ground-truth harness (run through `bazel run --run_under`
  so it gets the suite's venv, runfiles and S3 env). Only winners go through Buildkite.
- ONNX surgery on the exported graph tests an idea in minutes; anything that survives is then
  made suite-native (build flag or model code) so CI exports reproduce it.

## Export config for a DINO foundation checkpoint

Arch parity items the 903ak8 export needed on top of master's `latest_sdv2_export.yaml` (all
checked by the load summary: 0 shape mismatches, 0 missing):

- encoder `dinov3_b_dinofpn`; every consumer of the neck sized for 768 channels (depth head
  `dim_in`, neck out/intermediate width, det/map `feature_projection` Conv2d 768 → 256, PV
  tokenizer `input_embed_dims: 768`), with `embed_dims` pinned to 256
- det decoder `apply_quality_score: true` (the foundation ranks boxes by
  sigmoid(cls) x sigmoid(centerness))
- instance tokenizer top-k: det 450, map 256
- `pv_camera_indices: null` (all 7 cameras feed PV tokens)
- seg head 8 channels: `drivable_area_seg_enabled: true`, `lane_color_seg_enabled: false`
- `navigation_crop_radius_meters: 200` (trained at the doubled nav FOV)
- checkpoint pinned by W&B artifact (`checkpoint_wandb_artifact`), `checkpoint_s3_url: null`

## Code map (export branch)

| file | what |
|---|---|
| `onroad/ml_optimization/tools/onnx_node_utils.py` | `retype_bf16_scope_to_fp16`, `unbracket_fp32_in_scope`, `apply_int8_smoothquant` (+ tests) |
| `onroad/ml_optimization/experimental/one_stage/one_stage_tensorrt.py` | `fp16_onnx_scope`, `int8_calibration_path` build flags |
| `onroad/ml_optimization/experimental/one_stage/calibration/` | INT8 calibration JSONs for 903ak8 |
| `onroad/ml/rl/rllib/rl_module/planner/models/decoder/tokenizer/pv_feature_tokenizer.py` | batched PETR position embeddings (+ tests) |
| `onroad/ml/rl/rllib/rl_module/planner/models/backbone/dinov3_backbone_v2.py` | `input_size`, `low_res_cameras` |
| `onroad/ml/rl/configs/il/latest_sdv2_export.yaml` | 903ak8 export config (768-dim neck consumers, quality-score ranking, 8ch drivable seg) |

## Export gotchas hit along the way

- `embed_dims` was picked up from the neck width (768); pinned to 256.
- Master defaults the PV tokenizer to front camera only; the foundation needs `pv_camera_indices: null`.
- A stale gin S3 checkpoint pin in `latest_sdv2.yaml` would have mislabeled the model family tree; nulled.
- Retyping an already-retyped ONNX leaves BF16 casts outside the scope; always start from the BF16 ONNX.
- Day-branch runtimes derive the engine name without the `_linux_x64` token: land the unsuffixed
  `.trt` and always ship the `.onnx`.
- Land the TRT-preprocessed ONNX (`tensorrt/one_stage_modified.onnx`), not `onnx/model.onnx`: the
  FP16 retype happens in the TRT transform, and the car rebuilds its engine from the landed ONNX.
- `branch_prep.py`'s `camelot_style` bundle writes the plain dense presets
  (`runtime_config_steering_angle_map_dense`, `waypoint_q2_extended`). A drivable-seg model trained
  at 200 m needs the tethys set; diff the new entry against `tethys` before landing.
- The export branch also needs the forkserver DataLoader / metrics-namespace fixes for the suite
  step (cherry-picked); without them validation dies with `ConnectionRefusedError`.
- The on-prem 4080 queue has a single host (butter) and can sit idle-queued for 30+ minutes;
  `gpu-g6` builds a CC89 engine that lands fine, but its latency numbers are not the car's.
- The suite's S3 engine upload is keyed by checkpoint, so every export of 903ak8 (aoligei,
  aoligei_map, INT8 variants) writes the same S3 path; take engines from the build artifacts.
